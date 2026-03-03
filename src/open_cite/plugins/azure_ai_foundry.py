"""
Azure AI Foundry discovery plugin for Open-CITE.

This plugin discovers AI resources in Azure AI Foundry (Cognitive Services):
- Foundry resources (kind=AIServices)
- OpenAI resources (kind=OpenAI)
- Model deployments (GPT, embedding, etc.)
- Projects (preview API)
- Agents (OpenAI Assistants API)
- Tools (extracted from agents)
- Traces (via Application Insights / Log Analytics)

Authentication is via Microsoft Entra ID (Azure AD) using either:
- Service principal (client_id + client_secret + tenant_id)
- A pre-acquired access token

Three APIs are used:
- ARM (management.azure.com) — accounts, deployments, projects
- Foundry Service (<account>.services.ai.azure.com) — agents, tools
- Log Analytics (api.loganalytics.azure.com) — traces from Application Insights
"""

import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)

ARM_BASE = "https://management.azure.com"
ENTRA_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
ARM_SCOPE = "https://management.azure.com/.default"
COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
LOG_ANALYTICS_SCOPE = "https://api.loganalytics.io/.default"
LOG_ANALYTICS_API = "https://api.loganalytics.azure.com/v1"

# ARM API versions
ACCOUNTS_API_VERSION = "2024-10-01"
DEPLOYMENTS_API_VERSION = "2024-10-01"
PROJECTS_API_VERSION = "2025-04-01-preview"
DIAGNOSTIC_SETTINGS_API_VERSION = "2021-05-01-preview"
WORKSPACE_API_VERSION = "2023-09-01"

# Foundry service API version for assistants
ASSISTANTS_API_VERSION = "2025-01-01-preview"


# ------------------------------------------------------------------
# OTLP conversion helpers (source-specific, kept inside plugin)
# ------------------------------------------------------------------

def _make_attr(key: str, value: str) -> Dict[str, Any]:
    """Build an OTLP string attribute dict."""
    return {"key": key, "value": {"stringValue": str(value)}}


def _make_attr_int(key: str, value: int) -> Dict[str, Any]:
    """Build an OTLP integer attribute dict."""
    return {"key": key, "value": {"intValue": str(int(value))}}


def _generate_span_id(seed: str, length: int = 16) -> str:
    """Generate a deterministic hex ID from a seed string via MD5."""
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:length]


def _appinsights_row_to_otlp(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an Application Insights row to an OTLP resourceSpans payload.

    Each row becomes a single-span trace with resource attributes identifying
    the discovery source as azure_ai_foundry.
    """
    operation_id = row.get("OperationId", "")
    trace_id = _generate_span_id(operation_id, 32) if operation_id else _generate_span_id(str(time.time()), 32)
    span_id = _generate_span_id(f"{operation_id}-span", 16)

    name = row.get("Name", "AppRequest")
    duration_ms = row.get("DurationMs", 0)
    success = row.get("Success", True)
    result_code = str(row.get("ResultCode", ""))
    time_generated = row.get("TimeGenerated", "")

    # Convert ISO timestamp to nanoseconds
    start_ns = 0
    if time_generated:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(time_generated.replace("Z", "+00:00"))
            start_ns = int(dt.timestamp() * 1_000_000_000)
        except (ValueError, TypeError):
            start_ns = int(time.time() * 1_000_000_000)
    else:
        start_ns = int(time.time() * 1_000_000_000)

    end_ns = start_ns + int(float(duration_ms) * 1_000_000)

    status_code = 1 if success else 2  # 1=OK, 2=ERROR

    # Build span attributes from Properties
    span_attrs = [
        _make_attr("http.status_code", result_code),
    ]
    properties = row.get("Properties", {})
    if isinstance(properties, dict):
        for k, v in properties.items():
            span_attrs.append(_make_attr(f"appinsights.{k}", str(v)))

    resource_attrs = [
        _make_attr("service.name", "azure-ai-foundry"),
        _make_attr("opencite.discovery_source", "azure_ai_foundry"),
    ]

    return {
        "resourceSpans": [{
            "resource": {"attributes": resource_attrs},
            "scopeSpans": [{
                "scope": {"name": "open_cite.azure_ai_foundry"},
                "spans": [{
                    "traceId": trace_id,
                    "spanId": span_id,
                    "name": name,
                    "kind": 2,  # SERVER
                    "startTimeUnixNano": str(start_ns),
                    "endTimeUnixNano": str(end_ns),
                    "status": {"code": status_code},
                    "attributes": span_attrs,
                }],
            }],
        }],
    }


class AzureAIFoundryPlugin(BaseDiscoveryPlugin):
    """
    Azure AI Foundry discovery plugin.

    Discovers AI resources, model deployments, and projects across an Azure
    subscription using the Azure Resource Manager (ARM) API.
    """

    plugin_type = "azure_ai_foundry"

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        account_filter: Optional[List[str]] = None,
        log_analytics_workspace_id: Optional[str] = None,
        lookback_hours: int = 24,
        poll_interval: int = 60,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(instance_id=instance_id, display_name=display_name)

        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token = access_token
        self._token_expires_at: float = 0
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.account_filter = account_filter or []
        self.log_analytics_workspace_id = log_analytics_workspace_id
        self.lookback_hours = max(1, int(lookback_hours))
        self.poll_interval = max(10, int(poll_interval))

        # Service / monitor token caches (separate from ARM token)
        self._service_token: Optional[str] = None
        self._service_token_expires_at: float = 0
        self._monitor_token: Optional[str] = None
        self._monitor_token_expires_at: float = 0

        # Caches
        self._accounts_cache: List[Dict[str, Any]] = []
        self._deployments_cache: List[Dict[str, Any]] = []
        self._projects_cache: List[Dict[str, Any]] = []
        self._agents_cache: List[Dict[str, Any]] = []
        self._tools_cache: List[Dict[str, Any]] = []
        self._traces_cache: List[Dict[str, Any]] = []
        self._last_trace_time: Optional[str] = None  # ISO high-water mark
        self._seen_trace_ids: set = set()  # dedup within lookback window
        self._lock = threading.Lock()

        # Polling thread state
        self._poll_stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Plugin metadata & factory
    # ------------------------------------------------------------------

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        return {
            "name": "Azure AI Foundry",
            "description": (
                "Discovers AI resources, model deployments, and projects "
                "in Azure AI Foundry (Cognitive Services)"
            ),
            "required_fields": {
                "tenant_id": {
                    "label": "Tenant ID (Azure AD)",
                    "default": "",
                    "required": True,
                },
                "client_id": {
                    "label": "Client (Application) ID",
                    "default": "",
                    "required": True,
                },
                "client_secret": {
                    "label": "Client Secret",
                    "default": "",
                    "required": True,
                    "type": "password",
                },
                "subscription_id": {
                    "label": "Azure Subscription ID",
                    "default": "",
                    "required": True,
                },
                "resource_group": {
                    "label": "Resource Group (leave blank for all)",
                    "default": "",
                    "required": False,
                },
                "account_filter": {
                    "label": "Account Names (comma-separated, leave blank for all)",
                    "default": "",
                    "required": False,
                },
                "lookback_hours": {
                    "label": "Initial Lookback (hours)",
                    "default": "24",
                    "required": False,
                    "type": "number",
                    "min": 1,
                    "max": 720,
                },
                "poll_interval": {
                    "label": "Poll Interval (seconds)",
                    "default": "60",
                    "required": False,
                    "type": "number",
                    "min": 10,
                    "max": 3600,
                },
            },
            "env_vars": [
                "AZURE_TENANT_ID",
                "AZURE_CLIENT_ID",
                "AZURE_CLIENT_SECRET",
                "AZURE_SUBSCRIPTION_ID",
            ],
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        account_filter_raw = config.get("account_filter", "")
        if isinstance(account_filter_raw, str):
            account_filter = [
                a.strip() for a in account_filter_raw.split(",") if a.strip()
            ]
        else:
            account_filter = list(account_filter_raw or [])

        lookback_hours = int(config.get("lookback_hours") or 24)
        poll_interval = int(config.get("poll_interval") or 60)

        return cls(
            tenant_id=config.get("tenant_id"),
            client_id=config.get("client_id"),
            client_secret=config.get("client_secret"),
            access_token=config.get("access_token"),
            subscription_id=config.get("subscription_id"),
            resource_group=config.get("resource_group"),
            account_filter=account_filter,
            log_analytics_workspace_id=config.get("log_analytics_workspace_id") or None,
            lookback_hours=lookback_hours,
            poll_interval=poll_interval,
            instance_id=instance_id,
            display_name=display_name,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @property
    def supported_asset_types(self) -> Set[str]:
        return {
            "foundry_resource",
            "openai_resource",
            "deployment",
            "model",
            "project",
            "agent",
            "tool",
            "trace",
        }

    def get_identification_attributes(self) -> List[str]:
        return [
            "foundry.subscription_id",
            "foundry.resource_group",
            "foundry.account_name",
            "foundry.account_kind",
            "foundry.deployment_name",
            "foundry.agent_id",
        ]

    def verify_connection(self) -> Dict[str, Any]:
        """Verify connection by listing Cognitive Services accounts."""
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)
            url = (
                f"{ARM_BASE}/subscriptions/{self.subscription_id}"
                f"/providers/Microsoft.CognitiveServices/accounts"
                f"?api-version={ACCOUNTS_API_VERSION}"
            )
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "subscription_id": self.subscription_id,
                    "message": "Successfully connected to Azure AI Foundry",
                    "account_count_hint": len(data.get("value", [])),
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                "message": "Failed to connect to Azure AI Foundry",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to connect to Azure AI Foundry",
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        if asset_type not in self.supported_asset_types:
            raise ValueError(
                f"Unsupported asset type: {asset_type}. "
                f"Supported: {', '.join(sorted(self.supported_asset_types))}"
            )

        with self._lock:
            if asset_type == "foundry_resource":
                return self._list_accounts(kind_filter="AIServices")
            elif asset_type == "openai_resource":
                return self._list_accounts(kind_filter="OpenAI")
            elif asset_type == "deployment":
                return self._list_deployments()
            elif asset_type == "model":
                return self._list_models()
            elif asset_type == "project":
                return self._list_projects()
            elif asset_type == "agent":
                return self._list_agents()
            elif asset_type == "tool":
                return self._list_tools()
            elif asset_type == "trace":
                return self._list_traces()
            return []

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "client_secret": "****" if self.client_secret else None,
            "access_token": "****" if self._access_token else None,
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "account_filter": self.account_filter,
            "log_analytics_workspace_id": self.log_analytics_workspace_id,
            "lookback_hours": self.lookback_hours,
            "poll_interval": self.poll_interval,
        }

    def export_assets(self) -> Dict[str, Any]:
        return {
            "foundry_resources": self.list_assets("foundry_resource"),
            "openai_resources": self.list_assets("openai_resource"),
            "foundry_deployments": self.list_assets("deployment"),
            "foundry_projects": self.list_assets("project"),
            "foundry_agents": self.list_assets("agent"),
            "foundry_tools": self.list_assets("tool"),
            "foundry_traces": self.list_assets("trace"),
        }

    # ------------------------------------------------------------------
    # Lifecycle — background polling
    # ------------------------------------------------------------------

    def start(self):
        """Start background polling for trace discovery."""
        super().start()
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True,
            name=f"azure-foundry-poll-{self.instance_id[:8]}",
        )
        self._poll_thread.start()
        logger.info(
            "Azure AI Foundry polling started (interval=%ds, lookback=%dh)",
            self.poll_interval, self.lookback_hours,
        )

    def stop(self):
        """Stop background polling."""
        if self._poll_thread is not None:
            self._poll_stop.set()
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None
        super().stop()

    def _poll_loop(self):
        """Background loop that refreshes all caches periodically."""
        while not self._poll_stop.is_set():
            try:
                self.list_assets("trace")
            except Exception as e:
                logger.warning("Azure AI Foundry poll error: %s", e)
            self._poll_stop.wait(timeout=self.poll_interval)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Return a valid access token, refreshing via OAuth2 if needed."""
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        if self.tenant_id and self.client_id and self.client_secret:
            return self._acquire_token_client_credentials()

        if self._access_token:
            return self._access_token

        raise ValueError(
            "No valid authentication configured. Provide either "
            "(tenant_id + client_id + client_secret) or an access_token."
        )

    def _acquire_token_client_credentials(self) -> str:
        """Acquire a token using OAuth2 client-credentials flow."""
        url = ENTRA_TOKEN_URL.format(tenant_id=self.tenant_id)
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": ARM_SCOPE,
        }
        resp = requests.post(url, data=data, timeout=15)
        resp.raise_for_status()
        token_data = resp.json()
        self._access_token = token_data["access_token"]
        self._token_expires_at = time.time() + token_data.get("expires_in", 3600) - 120
        logger.debug("Acquired new ARM access token (expires in %ss)", token_data.get("expires_in"))
        return self._access_token

    @staticmethod
    def _auth_headers(token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def _get_service_token(self) -> str:
        """Return a valid service token for Foundry service API (Cognitive Services scope)."""
        if self._service_token and time.time() < self._service_token_expires_at:
            return self._service_token

        if not (self.tenant_id and self.client_id and self.client_secret):
            raise ValueError(
                "Service token requires tenant_id + client_id + client_secret."
            )

        url = ENTRA_TOKEN_URL.format(tenant_id=self.tenant_id)
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": COGNITIVE_SCOPE,
        }
        resp = requests.post(url, data=data, timeout=15)
        resp.raise_for_status()
        token_data = resp.json()
        self._service_token = token_data["access_token"]
        self._service_token_expires_at = time.time() + token_data.get("expires_in", 3600) - 120
        logger.debug("Acquired new Cognitive Services token (expires in %ss)", token_data.get("expires_in"))
        return self._service_token

    def _get_monitor_token(self) -> str:
        """Return a valid token for Log Analytics API."""
        if self._monitor_token and time.time() < self._monitor_token_expires_at:
            return self._monitor_token

        if not (self.tenant_id and self.client_id and self.client_secret):
            raise ValueError(
                "Monitor token requires tenant_id + client_id + client_secret."
            )

        url = ENTRA_TOKEN_URL.format(tenant_id=self.tenant_id)
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": LOG_ANALYTICS_SCOPE,
        }
        resp = requests.post(url, data=data, timeout=15)
        resp.raise_for_status()
        token_data = resp.json()
        self._monitor_token = token_data["access_token"]
        self._monitor_token_expires_at = time.time() + token_data.get("expires_in", 3600) - 120
        logger.debug("Acquired new Log Analytics token (expires in %ss)", token_data.get("expires_in"))
        return self._monitor_token

    # ------------------------------------------------------------------
    # ARM paginated GET helper
    # ------------------------------------------------------------------

    def _arm_paginated_get(
        self,
        url: str,
        headers: Dict[str, str],
        max_items: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all pages from an ARM REST API list endpoint.

        ARM uses ``nextLink`` (a full URL) in the response body to signal
        more pages, unlike Fabric's ``continuationToken`` parameter approach.
        """
        items: List[Dict[str, Any]] = []

        while url:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("GET %s returned %d: %s", url, resp.status_code, resp.text[:200])
                break

            data = resp.json()
            page_items = data.get("value", [])
            items.extend(page_items)

            if len(items) >= max_items:
                items = items[:max_items]
                break

            url = data.get("nextLink")

        return items

    # ------------------------------------------------------------------
    # Account discovery
    # ------------------------------------------------------------------

    def _list_accounts(self, kind_filter: str) -> List[Dict[str, Any]]:
        """List Cognitive Services accounts filtered by kind."""
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)

            if self.resource_group:
                url = (
                    f"{ARM_BASE}/subscriptions/{self.subscription_id}"
                    f"/resourceGroups/{self.resource_group}"
                    f"/providers/Microsoft.CognitiveServices/accounts"
                    f"?api-version={ACCOUNTS_API_VERSION}"
                )
            else:
                url = (
                    f"{ARM_BASE}/subscriptions/{self.subscription_id}"
                    f"/providers/Microsoft.CognitiveServices/accounts"
                    f"?api-version={ACCOUNTS_API_VERSION}"
                )

            raw = self._arm_paginated_get(url, headers)

            accounts = []
            for acct in raw:
                acct_kind = acct.get("kind", "")
                if acct_kind != kind_filter:
                    continue

                acct_name = acct.get("name", "")
                if self.account_filter and acct_name not in self.account_filter:
                    continue

                # Extract resource group from the resource ID
                resource_id = acct.get("id", "")
                rg = self._extract_resource_group(resource_id)

                asset_type = "foundry_resource" if kind_filter == "AIServices" else "openai_resource"
                accounts.append({
                    "id": resource_id,
                    "name": acct_name,
                    "type": asset_type,
                    "location": acct.get("location", ""),
                    "kind": acct_kind,
                    "sku": acct.get("sku", {}),
                    "metadata": {
                        "foundry.subscription_id": self.subscription_id,
                        "foundry.resource_group": rg,
                        "foundry.account_name": acct_name,
                        "foundry.account_kind": acct_kind,
                    },
                })

            self._accounts_cache = accounts
            logger.info("Discovered %d Azure %s accounts", len(accounts), kind_filter)
            return accounts
        except Exception as e:
            logger.error("Failed to list Azure %s accounts: %s", kind_filter, e)
            return []

    # ------------------------------------------------------------------
    # Deployment discovery
    # ------------------------------------------------------------------

    def _list_deployments(self) -> List[Dict[str, Any]]:
        """List model deployments across all discovered accounts."""
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)

            # Discover all accounts (both AIServices and OpenAI)
            all_accounts = self._get_all_accounts(headers)

            deployments = []
            for acct in all_accounts:
                resource_id = acct.get("id", "")
                acct_name = acct.get("name", "")
                rg = self._extract_resource_group(resource_id)

                url = (
                    f"{ARM_BASE}{resource_id}/deployments"
                    f"?api-version={DEPLOYMENTS_API_VERSION}"
                )

                try:
                    raw = self._arm_paginated_get(url, headers)
                    for dep in raw:
                        dep_name = dep.get("name", "")
                        properties = dep.get("properties", {})
                        model = properties.get("model", {})
                        sku = dep.get("sku", {})

                        deployments.append({
                            "id": f"{resource_id}/deployments/{dep_name}",
                            "name": dep_name,
                            "type": "deployment",
                            "account_name": acct_name,
                            "account_kind": acct.get("kind", ""),
                            "metadata": {
                                "foundry.subscription_id": self.subscription_id,
                                "foundry.resource_group": rg,
                                "foundry.account_name": acct_name,
                                "foundry.account_kind": acct.get("kind", ""),
                                "foundry.deployment_name": dep_name,
                                "foundry.model_name": model.get("name", ""),
                                "foundry.model_format": model.get("format", ""),
                                "foundry.model_version": model.get("version", ""),
                                "foundry.model_publisher": model.get("publisher", ""),
                                "foundry.sku_name": sku.get("name", ""),
                                "foundry.sku_capacity": sku.get("capacity", 0),
                            },
                        })
                except Exception as e:
                    logger.warning(
                        "Failed to list deployments for account %s: %s",
                        acct_name, e,
                    )

            self._deployments_cache = deployments
            logger.info("Discovered %d Azure AI deployments", len(deployments))
            return deployments
        except Exception as e:
            logger.error("Failed to list Azure AI deployments: %s", e)
            return []

    # ------------------------------------------------------------------
    # Model view (deployments as models for UI)
    # ------------------------------------------------------------------

    def _list_models(self) -> List[Dict[str, Any]]:
        """Return deployments in the standard model format for the UI."""
        deployments = self._list_deployments()

        # Count calls per model from traces
        usage_counts = self._count_model_usage()

        models = []
        for dep in deployments:
            meta = dep.get("metadata", {})
            model_name = meta.get("foundry.model_name", dep.get("name", ""))
            dep_name = meta.get("foundry.deployment_name", dep.get("name", ""))
            count = usage_counts.get(dep_name, 0) or usage_counts.get(model_name, 0)
            models.append({
                "name": model_name,
                "provider": "Azure AI Foundry",
                "type": "model",
                "tools": [],
                "usage_count": count,
                "metadata": meta,
            })
        return models

    def _count_model_usage(self) -> Dict[str, int]:
        """Query Log Analytics for request counts per deployment/model."""
        if self.log_analytics_workspace_id:
            workspace_ids = [self.log_analytics_workspace_id]
        else:
            workspace_ids = self._discover_workspace_ids()

        if not workspace_ids:
            return {}

        try:
            monitor_token = self._get_monitor_token()
            headers = self._auth_headers(monitor_token)

            kql = (
                "AzureDiagnostics"
                f" | where TimeGenerated > ago({self.lookback_hours}h)"
                " | where Category == 'RequestResponse'"
                " | extend props = parse_json(properties_s)"
                " | extend model = tostring(props.modelDeploymentName)"
                " | where isnotempty(model)"
                " | summarize calls = count() by model"
            )

            counts: Dict[str, int] = {}
            for workspace_id in workspace_ids:
                url = f"{LOG_ANALYTICS_API}/workspaces/{workspace_id}/query"
                resp = requests.post(
                    url,
                    headers={**headers, "Content-Type": "application/json"},
                    json={"query": kql},
                    timeout=30,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                tables = data.get("tables", [])
                if not tables:
                    continue
                columns = [c["name"] for c in tables[0].get("columns", [])]
                for row_values in tables[0].get("rows", []):
                    row = dict(zip(columns, row_values))
                    model = row.get("model", "")
                    calls = int(row.get("calls", 0))
                    if model:
                        counts[model] = counts.get(model, 0) + calls
            return counts
        except Exception as e:
            logger.debug("Could not count model usage: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Project discovery
    # ------------------------------------------------------------------

    def _list_projects(self) -> List[Dict[str, Any]]:
        """List projects across all discovered accounts (preview API)."""
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)

            all_accounts = self._get_all_accounts(headers)

            projects = []
            for acct in all_accounts:
                resource_id = acct.get("id", "")
                acct_name = acct.get("name", "")
                rg = self._extract_resource_group(resource_id)

                url = (
                    f"{ARM_BASE}{resource_id}/projects"
                    f"?api-version={PROJECTS_API_VERSION}"
                )

                try:
                    raw = self._arm_paginated_get(url, headers)
                    for proj in raw:
                        proj_name = proj.get("name", "")
                        projects.append({
                            "id": proj.get("id", f"{resource_id}/projects/{proj_name}"),
                            "name": proj_name,
                            "type": "project",
                            "account_name": acct_name,
                            "location": proj.get("location", ""),
                            "metadata": {
                                "foundry.subscription_id": self.subscription_id,
                                "foundry.resource_group": rg,
                                "foundry.account_name": acct_name,
                            },
                        })
                except Exception as e:
                    # Preview API may 404 in some regions — degrade gracefully
                    logger.warning(
                        "Failed to list projects for account %s: %s",
                        acct_name, e,
                    )

            self._projects_cache = projects
            logger.info("Discovered %d Azure AI projects", len(projects))
            return projects
        except Exception as e:
            logger.error("Failed to list Azure AI projects: %s", e)
            return []

    # ------------------------------------------------------------------
    # Agent discovery (Foundry service API)
    # ------------------------------------------------------------------

    def _list_agents(self) -> List[Dict[str, Any]]:
        """List AI agents (OpenAI assistants) across all accounts."""
        try:
            service_token = self._get_service_token()
            headers = self._auth_headers(service_token)
            token = self._get_access_token()
            arm_headers = self._auth_headers(token)
            all_accounts = self._get_all_accounts(arm_headers)

            agents = []
            seen_ids: Set[str] = set()
            for acct in all_accounts:
                acct_name = acct.get("name", "")
                service_base = f"https://{acct_name}.services.ai.azure.com"
                url = f"{service_base}/openai/assistants?api-version={ASSISTANTS_API_VERSION}"
                try:
                    resp = requests.get(url, headers=headers, timeout=30)
                    if resp.status_code != 200:
                        logger.warning(
                            "GET assistants for %s returned %d: %s",
                            acct_name, resp.status_code, resp.text[:200],
                        )
                        continue

                    data = resp.json()
                    for asst in data.get("data", []):
                        asst_id = asst.get("id", "")
                        if asst_id in seen_ids:
                            continue
                        seen_ids.add(asst_id)

                        asst_name = asst.get("name", "") or asst_id
                        model = asst.get("model", "")
                        instructions = asst.get("instructions", "") or ""

                        # Extract tool names
                        tools_used = []
                        for t in asst.get("tools", []):
                            tool_type = t.get("type", "")
                            if tool_type == "function":
                                func_name = t.get("function", {}).get("name", "")
                                if func_name:
                                    tools_used.append(func_name)
                            elif tool_type:
                                tools_used.append(tool_type)

                        now = datetime.now(timezone.utc).isoformat()
                        created_at = asst.get("created_at")
                        if isinstance(created_at, (int, float)):
                            first_seen = datetime.fromtimestamp(
                                created_at, tz=timezone.utc,
                            ).isoformat()
                        else:
                            first_seen = now

                        agents.append({
                            "id": asst_id,
                            "name": asst_name,
                            "tools_used": tools_used,
                            "models_used": [model] if model else [],
                            "first_seen": first_seen,
                            "last_seen": now,
                            "metadata": {
                                "foundry.subscription_id": self.subscription_id,
                                "foundry.account_name": acct_name,
                                "foundry.agent_id": asst_id,
                                "foundry.model": model,
                                "foundry.instructions_preview": instructions[:200],
                            },
                        })
                except Exception as e:
                    logger.warning(
                        "Failed to list agents for %s: %s",
                        acct_name, e,
                    )

            self._agents_cache = agents
            logger.info("Discovered %d Azure AI agents", len(agents))
            return agents
        except Exception as e:
            logger.error("Failed to list Azure AI agents: %s", e)
            return []

    # ------------------------------------------------------------------
    # Tool discovery (extracted from agents)
    # ------------------------------------------------------------------

    def _list_tools(self) -> List[Dict[str, Any]]:
        """Extract and deduplicate tools from discovered agents."""
        try:
            # Ensure agents are discovered first
            if not self._agents_cache:
                self._list_agents()

            now = datetime.now(timezone.utc).isoformat()

            # Deduplicate by (tool_type, function_name)
            tool_map: Dict[str, Dict[str, Any]] = {}

            for agent in self._agents_cache:
                account_name = agent.get("metadata", {}).get("foundry.account_name", "")
                agent_models = agent.get("models_used", [])

                for tool_name in agent.get("tools_used", []):
                    # Determine tool type and ID
                    if tool_name in ("code_interpreter", "file_search"):
                        tool_id = tool_name
                        tool_type = tool_name
                    else:
                        tool_id = f"func:{tool_name}"
                        tool_type = "function"

                    if tool_id in tool_map:
                        for m in agent_models:
                            if m not in tool_map[tool_id]["models"]:
                                tool_map[tool_id]["models"].append(m)
                    else:
                        tool_map[tool_id] = {
                            "id": tool_id,
                            "name": tool_name,
                            "models": list(agent_models),
                            "trace_count": 0,
                            "first_seen": now,
                            "last_seen": now,
                            "metadata": {
                                "foundry.subscription_id": self.subscription_id,
                                "foundry.account_name": account_name,
                                "foundry.tool_type": tool_type,
                            },
                        }

            tools = list(tool_map.values())
            self._tools_cache = tools
            logger.info("Discovered %d unique Azure AI tools", len(tools))
            return tools
        except Exception as e:
            logger.error("Failed to list Azure AI tools: %s", e)
            return []

    # ------------------------------------------------------------------
    # Workspace auto-discovery (from diagnostic settings)
    # ------------------------------------------------------------------

    def _discover_workspace_ids(self) -> List[str]:
        """
        Auto-discover Log Analytics workspace IDs from diagnostic settings
        on discovered Cognitive Services accounts.

        Returns a list of unique workspace customer IDs (GUIDs) that can be
        used with the Log Analytics query API.
        """
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)
            all_accounts = self._get_all_accounts(headers)

            workspace_resource_ids: Set[str] = set()

            for acct in all_accounts:
                resource_id = acct.get("id", "")
                url = (
                    f"{ARM_BASE}{resource_id}"
                    f"/providers/Microsoft.Insights/diagnosticSettings"
                    f"?api-version={DIAGNOSTIC_SETTINGS_API_VERSION}"
                )
                try:
                    resp = requests.get(url, headers=headers, timeout=15)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    for setting in data.get("value", []):
                        ws_id = setting.get("properties", {}).get("workspaceId", "")
                        if ws_id:
                            workspace_resource_ids.add(ws_id)
                except Exception as e:
                    logger.debug(
                        "Failed to get diagnostic settings for %s: %s",
                        acct.get("name", ""), e,
                    )

            # Resolve each workspace resource ID to its customerId
            customer_ids = []
            for ws_resource_id in workspace_resource_ids:
                url = f"{ARM_BASE}{ws_resource_id}?api-version={WORKSPACE_API_VERSION}"
                try:
                    resp = requests.get(url, headers=headers, timeout=15)
                    if resp.status_code == 200:
                        cid = resp.json().get("properties", {}).get("customerId", "")
                        if cid:
                            customer_ids.append(cid)
                except Exception as e:
                    logger.debug("Failed to resolve workspace %s: %s", ws_resource_id, e)

            logger.info("Auto-discovered %d Log Analytics workspace(s)", len(customer_ids))
            return customer_ids
        except Exception as e:
            logger.warning("Workspace auto-discovery failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Trace discovery (Log Analytics / Application Insights)
    # ------------------------------------------------------------------

    def _list_traces(self) -> List[Dict[str, Any]]:
        """Query Application Insights traces via Log Analytics API."""
        # Determine workspace IDs: explicit override or auto-discovered
        if self.log_analytics_workspace_id:
            workspace_ids = [self.log_analytics_workspace_id]
        else:
            workspace_ids = self._discover_workspace_ids()

        if not workspace_ids:
            logger.debug("No Log Analytics workspaces found; skipping trace discovery")
            return []

        try:
            monitor_token = self._get_monitor_token()
            headers = self._auth_headers(monitor_token)

            # Use high-water mark if available, otherwise use lookback window
            if self._last_trace_time:
                time_filter = f" | where TimeGenerated > datetime('{self._last_trace_time}')"
            else:
                time_filter = f" | where TimeGenerated > ago({self.lookback_hours}h)"

            kql = (
                "AzureDiagnostics"
                + time_filter
                + " | where Category == 'RequestResponse'"
                " | extend props = parse_json(properties_s)"
                " | project TimeGenerated, OperationId=CorrelationId,"
                "   Name=OperationName, DurationMs, Success=(ResultSignature == '200'),"
                "   ResultCode=ResultSignature, Properties=props"
                " | order by TimeGenerated asc"
            )

            all_traces = []
            all_rows = []
            for workspace_id in workspace_ids:
                traces_batch, rows_batch = self._query_workspace(
                    workspace_id, kql, headers,
                )
                all_traces.extend(traces_batch)
                all_rows.extend(rows_batch)

            # Deduplicate and forward only new traces to webhooks
            new_traces = []
            new_rows = []
            new_high_water = self._last_trace_time
            for trace, row in zip(all_traces, all_rows):
                op_id = trace.get("trace_id", "")
                if op_id in self._seen_trace_ids:
                    continue
                self._seen_trace_ids.add(op_id)
                new_traces.append(trace)
                new_rows.append(row)
                # Track highest TimeGenerated
                tg = row.get("TimeGenerated") or ""
                if tg and (new_high_water is None or tg > new_high_water):
                    new_high_water = tg

            # Advance high-water mark
            if new_high_water is not None:
                self._last_trace_time = new_high_water

            # Forward only new traces as OTLP to webhooks
            if new_rows and self._webhook_urls:
                for row in new_rows:
                    try:
                        otlp_payload = _appinsights_row_to_otlp(row)
                        self._deliver_to_webhooks(otlp_payload)
                    except Exception as e:
                        logger.warning("Failed to convert/forward trace to webhook: %s", e)

            # Append new traces to cache (cumulative)
            self._traces_cache.extend(new_traces)
            if new_traces:
                logger.info("Discovered %d new Azure AI traces (%d total)",
                            len(new_traces), len(self._traces_cache))
            return self._traces_cache
        except Exception as e:
            logger.error("Failed to list Azure AI traces: %s", e)
            return []

    def _query_workspace(
        self,
        workspace_id: str,
        kql: str,
        headers: Dict[str, str],
    ) -> tuple:
        """
        Run a KQL query against a single workspace.

        Returns (traces_list, raw_row_dicts) so the caller can forward
        raw rows to webhooks.
        """
        url = f"{LOG_ANALYTICS_API}/workspaces/{workspace_id}/query"
        resp = requests.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json={"query": kql},
            timeout=30,
        )

        if resp.status_code != 200:
            logger.warning(
                "Log Analytics query on %s returned %d: %s",
                workspace_id, resp.status_code, resp.text[:300],
            )
            return [], []

        data = resp.json()
        tables = data.get("tables", [])
        if not tables:
            return [], []

        columns = [c["name"] for c in tables[0].get("columns", [])]
        rows = tables[0].get("rows", [])

        traces = []
        raw_rows = []
        for row_values in rows:
            row = dict(zip(columns, row_values))
            raw_rows.append(row)
            traces.append({
                "trace_id": row.get("OperationId", ""),
                "name": row.get("Name", ""),
                "type": "trace",
                "timestamp": row.get("TimeGenerated", ""),
                "duration_ms": row.get("DurationMs", 0),
                "success": row.get("Success", True),
                "result_code": str(row.get("ResultCode", "")),
                "metadata": {
                    "foundry.subscription_id": self.subscription_id,
                    "foundry.properties": row.get("Properties", {}),
                },
            })

        return traces, raw_rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_all_accounts(self, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch all Cognitive Services accounts (both AIServices and OpenAI)."""
        if self.resource_group:
            url = (
                f"{ARM_BASE}/subscriptions/{self.subscription_id}"
                f"/resourceGroups/{self.resource_group}"
                f"/providers/Microsoft.CognitiveServices/accounts"
                f"?api-version={ACCOUNTS_API_VERSION}"
            )
        else:
            url = (
                f"{ARM_BASE}/subscriptions/{self.subscription_id}"
                f"/providers/Microsoft.CognitiveServices/accounts"
                f"?api-version={ACCOUNTS_API_VERSION}"
            )

        raw = self._arm_paginated_get(url, headers)

        if self.account_filter:
            raw = [a for a in raw if a.get("name") in self.account_filter]

        return raw

    @staticmethod
    def _extract_resource_group(resource_id: str) -> str:
        """Extract the resource group name from an ARM resource ID."""
        parts = resource_id.split("/")
        for i, part in enumerate(parts):
            if part.lower() == "resourcegroups" and i + 1 < len(parts):
                return parts[i + 1]
        return ""

    # ------------------------------------------------------------------
    # Convenience aggregation
    # ------------------------------------------------------------------

    def refresh_discovery(self):
        """Refresh all cached discovery data."""
        with self._lock:
            logger.info("Refreshing Azure AI Foundry discovery...")
            self._accounts_cache.clear()
            self._deployments_cache.clear()
            self._projects_cache.clear()
            self._agents_cache.clear()
            self._tools_cache.clear()
            self._traces_cache.clear()

            self._list_accounts(kind_filter="AIServices")
            self._list_accounts(kind_filter="OpenAI")
            self._list_deployments()
            self._list_projects()
            self._list_agents()
            self._list_tools()
            self._list_traces()

            logger.info("Azure AI Foundry discovery refresh complete")
