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


def _parse_json_safe(raw: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON string, returning None on failure."""
    if not raw:
        return None
    try:
        import json
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _appinsights_row_to_otlp(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an Application Insights row to an OTLP resourceSpans payload.

    Each row produces TWO spans sharing the same traceId:
      1. A root span (no parentSpanId) — becomes the trace in LangGuard.
      2. A child span (parentSpanId = root) — becomes the observation/generation
         with all gen_ai semantic attributes (model, tokens, input/output, tools).

    The request/response bodies from Azure OpenAI are full API JSON payloads.
    We parse them to extract messages, tools, tool_calls, model, temperature,
    finish_reason, etc. and map them to standard gen_ai semantic convention
    attribute keys that the LangGuard OTLP adapter recognises.
    """
    import json

    operation_id = row.get("OperationId", "")
    trace_id = _generate_span_id(operation_id, 32) if operation_id else _generate_span_id(str(time.time()), 32)
    root_span_id = _generate_span_id(f"{operation_id}-root", 16)
    child_span_id = _generate_span_id(f"{operation_id}-gen", 16)

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

    # Raw fields from KQL
    deployment_name = row.get("Model", "") or ""
    request_uri = row.get("RequestUri", "") or ""
    prompt_tokens = row.get("PromptTokens") or 0
    completion_tokens = row.get("CompletionTokens") or 0
    total_tokens = row.get("TotalTokens") or 0
    request_body = row.get("RequestBody", "") or ""
    response_body = row.get("ResponseBody", "") or ""
    api_version = row.get("ApiVersion", "") or ""
    stream_type = row.get("StreamType", "") or ""

    # Parse the Azure OpenAI request/response JSON for rich details
    req = _parse_json_safe(request_body)
    resp = _parse_json_safe(response_body)

    # Extract deployment name from requestUri as fallback:
    #   /openai/deployments/<deployment-name>/chat/completions?api-version=...
    uri_deployment = ""
    if request_uri and "/deployments/" in request_uri:
        try:
            uri_deployment = request_uri.split("/deployments/")[1].split("/")[0]
        except (IndexError, AttributeError):
            pass

    # Resolve model: response.model > request.model > KQL prop > URI deployment > operation name
    model = ""
    if resp:
        model = resp.get("model", "") or ""
    if not model and req:
        model = req.get("model", "") or ""
    if not model:
        model = deployment_name
    if not model:
        model = uri_deployment

    # Extract tokens from response.usage (more reliable than KQL props)
    if resp and isinstance(resp.get("usage"), dict):
        usage = resp["usage"]
        prompt_tokens = prompt_tokens or usage.get("prompt_tokens", 0) or 0
        completion_tokens = completion_tokens or usage.get("completion_tokens", 0) or 0
        total_tokens = total_tokens or usage.get("total_tokens", 0) or 0

    # ── Build child span attributes (the observation/generation) ──

    child_attrs = [
        _make_attr("http.status_code", result_code),
        _make_attr("gen_ai.provider.name", "azure"),
        _make_attr("gen_ai.system", "azure"),
        _make_attr("gen_ai.operation.name", "chat"),
    ]

    if model:
        child_attrs.append(_make_attr("gen_ai.request.model", model))

    # Tokens — use the attribute keys the adapter actually looks for
    if prompt_tokens:
        child_attrs.append(_make_attr("gen_ai.usage.input_tokens", str(prompt_tokens)))
    if completion_tokens:
        child_attrs.append(_make_attr("gen_ai.usage.output_tokens", str(completion_tokens)))
    if total_tokens:
        child_attrs.append(_make_attr("gen_ai.usage.total_tokens", str(total_tokens)))

    # Input: emit indexed gen_ai.prompt.N.role / gen_ai.prompt.N.content
    if req and isinstance(req.get("messages"), list):
        for i, msg in enumerate(req["messages"]):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role:
                    child_attrs.append(_make_attr(f"gen_ai.prompt.{i}.role", str(role)))
                if content:
                    child_attrs.append(_make_attr(
                        f"gen_ai.prompt.{i}.content",
                        str(content)[:4096],
                    ))
    elif request_body:
        # Fallback: dump the raw request as a flat prompt attribute
        child_attrs.append(_make_attr("gen_ai.prompt", request_body[:4096]))

    # Output: emit indexed gen_ai.completion.N.role / gen_ai.completion.N.content
    choices = resp.get("choices", []) if resp else []
    if isinstance(choices, list):
        for i, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message") or choice.get("delta") or {}
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            if role:
                child_attrs.append(_make_attr(f"gen_ai.completion.{i}.role", str(role)))
            if content:
                child_attrs.append(_make_attr(
                    f"gen_ai.completion.{i}.content",
                    str(content)[:4096],
                ))

            # Tool calls in the response
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                child_attrs.append(_make_attr(
                    "gen_ai.tool_calls",
                    json.dumps(tool_calls)[:4096],
                ))
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        if isinstance(fn, dict) and fn.get("name"):
                            child_attrs.append(_make_attr(
                                "gen_ai.tool.name", fn["name"],
                            ))

        # Finish reason from first choice
        if choices and isinstance(choices[0], dict):
            finish = choices[0].get("finish_reason", "")
            if finish:
                child_attrs.append(_make_attr(
                    "gen_ai.response.finish_reasons", str(finish),
                ))
    elif response_body:
        # Fallback: dump raw response as flat completion attribute
        child_attrs.append(_make_attr("gen_ai.completion", response_body[:4096]))

    # Tool definitions from the request
    if req and isinstance(req.get("tools"), list) and req["tools"]:
        child_attrs.append(_make_attr(
            "gen_ai.request.tools",
            json.dumps(req["tools"])[:4096],
        ))

    # Request config (temperature, max_tokens, top_p, etc.)
    if req:
        for param, attr_key in [
            ("temperature", "gen_ai.request.temperature"),
            ("max_tokens", "gen_ai.request.max_tokens"),
            ("top_p", "gen_ai.request.top_p"),
            ("frequency_penalty", "gen_ai.request.frequency_penalty"),
            ("presence_penalty", "gen_ai.request.presence_penalty"),
            ("seed", "gen_ai.request.seed"),
        ]:
            val = req.get(param)
            if val is not None:
                child_attrs.append(_make_attr(attr_key, str(val)))

        stop = req.get("stop")
        if stop is not None:
            child_attrs.append(_make_attr(
                "gen_ai.request.stop_sequences",
                json.dumps(stop) if isinstance(stop, list) else str(stop),
            ))

    # Response ID
    if resp and resp.get("id"):
        child_attrs.append(_make_attr("gen_ai.response.id", str(resp["id"])))

    if api_version:
        child_attrs.append(_make_attr("azure.api_version", api_version))
    if stream_type:
        child_attrs.append(_make_attr("gen_ai.request.stream", stream_type))

    # ── Build root span attributes (the trace) ──

    root_attrs = [
        _make_attr("gen_ai.provider.name", "azure"),
        _make_attr("gen_ai.system", "azure"),
    ]
    if model:
        root_attrs.append(_make_attr("gen_ai.request.model", model))
        root_attrs.append(_make_attr("gen_ai.agent.name", model))
    else:
        root_attrs.append(_make_attr("gen_ai.agent.name", name))

    # ── Resource attributes ──

    resource_attrs = [
        _make_attr("service.name", "azure-ai-foundry"),
        _make_attr("gen_ai.provider.name", "azure"),
        _make_attr("opencite.discovery_source", "azure_ai_foundry"),
    ]

    root_span = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": name,
        "kind": 2,  # SERVER
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "status": {"code": status_code},
        "attributes": root_attrs,
    }

    child_span = {
        "traceId": trace_id,
        "spanId": child_span_id,
        "parentSpanId": root_span_id,
        "name": model or name,
        "kind": 3,  # CLIENT
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "status": {"code": status_code},
        "attributes": child_attrs,
    }

    return {
        "resourceSpans": [{
            "resource": {"attributes": resource_attrs},
            "scopeSpans": [{
                "scope": {"name": "open_cite.azure_ai_foundry"},
                "spans": [root_span, child_span],
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
        self._lock = threading.Lock()

        # High-water mark for incremental trace queries
        self._last_query_time: Optional[str] = None  # ISO 8601

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
                # Log full response details for debugging persistent errors
                resp_headers = dict(resp.headers)
                logger.warning(
                    "GET %s returned %d\n"
                    "  Response body: %s\n"
                    "  x-ms-request-id: %s\n"
                    "  x-ms-correlation-request-id: %s\n"
                    "  x-ms-routing-request-id: %s\n"
                    "  Date: %s",
                    url,
                    resp.status_code,
                    resp.text[:1000],
                    resp_headers.get("x-ms-request-id", "n/a"),
                    resp_headers.get("x-ms-correlation-request-id", "n/a"),
                    resp_headers.get("x-ms-routing-request-id", "n/a"),
                    resp_headers.get("Date", "n/a"),
                )
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
                            "agent_source_name": acct_name,
                            "agent_source_id": asst_id,
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
                                "agent_source_name": acct_name,
                                "agent_source_id": asst_id,
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

            # Use high-water mark for incremental queries, fall back to lookback window
            if self._last_query_time:
                time_filter = f" | where TimeGenerated > datetime('{self._last_query_time}')"
            else:
                time_filter = f" | where TimeGenerated > ago({self.lookback_hours}h)"

            kql = (
                "AzureDiagnostics"
                + time_filter
                + " | where Category == 'RequestResponse'"
                " | extend props = parse_json(properties_s)"
                # Try multiple property names for model (varies by service version)
                " | extend Model=coalesce("
                "     tostring(props.modelDeploymentName),"
                "     tostring(props.ModelDeploymentName),"
                "     tostring(props.model_deployment_name),"
                "     tostring(props.model),"
                "     tostring(props.modelName)"
                "   ),"
                # Extract deployment name from requestUri as fallback
                #   /openai/deployments/<name>/chat/completions
                "   RequestUri=tostring(props.requestUri),"
                "   StreamType=coalesce(tostring(props.stream), tostring(props.Stream)),"
                "   ApiVersion=coalesce(tostring(props.apiVersion), tostring(props.ApiVersion)),"
                "   ObjectType=coalesce(tostring(props.objectType), tostring(props.ObjectType)),"
                "   PromptTokens=coalesce(toint(props.promptTokens), toint(props.PromptTokens)),"
                "   CompletionTokens=coalesce(toint(props.completionTokens), toint(props.CompletionTokens)),"
                "   TotalTokens=coalesce(toint(props.totalTokens), toint(props.TotalTokens)),"
                "   RequestBody=coalesce(tostring(props.request), tostring(props.Request)),"
                "   ResponseBody=coalesce(tostring(props.response), tostring(props.Response))"
                " | project TimeGenerated, OperationId=CorrelationId,"
                "   Name=OperationName, DurationMs, Success=(ResultSignature == '200'),"
                "   ResultCode=ResultSignature, Properties=props,"
                "   Model, RequestUri, StreamType, ApiVersion, ObjectType,"
                "   PromptTokens, CompletionTokens, TotalTokens,"
                "   RequestBody, ResponseBody"
                " | order by TimeGenerated desc"
                " | limit 500"
            )

            all_traces = []
            all_rows = []
            for workspace_id in workspace_ids:
                traces_batch, rows_batch = self._query_workspace(
                    workspace_id, kql, headers,
                )
                all_traces.extend(traces_batch)
                all_rows.extend(rows_batch)

            # Forward traces as OTLP to webhooks in chunks to avoid
            # overwhelming the receiver (large batches cause timeouts and
            # the webhook handler attaches rawPayload to every trace).
            WEBHOOK_CHUNK_SIZE = 50
            if all_traces and self._webhook_urls:
                all_resource_spans = []
                for row in all_rows:
                    try:
                        otlp_payload = _appinsights_row_to_otlp(row)
                        all_resource_spans.extend(otlp_payload.get("resourceSpans", []))
                    except Exception as e:
                        logger.warning("Failed to convert trace to OTLP: %s", e)
                # Deliver in chunks
                for i in range(0, len(all_resource_spans), WEBHOOK_CHUNK_SIZE):
                    chunk = all_resource_spans[i:i + WEBHOOK_CHUNK_SIZE]
                    self._deliver_to_webhooks({"resourceSpans": chunk})
                logger.info(
                    "Delivered %d resourceSpans in %d chunk(s) to webhooks",
                    len(all_resource_spans),
                    (len(all_resource_spans) + WEBHOOK_CHUNK_SIZE - 1) // WEBHOOK_CHUNK_SIZE,
                )

            # Advance high-water mark to the newest trace timestamp
            if all_rows:
                newest = max(
                    row.get("TimeGenerated", "") for row in all_rows
                )
                if newest:
                    self._last_query_time = newest

            self._traces_cache = all_traces
            logger.info("Discovered %d Azure AI traces", len(all_traces))
            return all_traces
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
        for idx, row_values in enumerate(rows):
            row = dict(zip(columns, row_values))
            # Log the first row so we can see what KQL actually returns
            if idx == 0:
                props = row.get("Properties")
                prop_keys = sorted(props.keys()) if isinstance(props, dict) else type(props).__name__
                logger.info(
                    "Sample KQL row — columns: %s | Model=%r | RequestBody=%r (len=%d) | "
                    "ResponseBody=%r (len=%d) | PromptTokens=%r | Properties keys=%s",
                    columns,
                    row.get("Model"),
                    (row.get("RequestBody") or "")[:100],
                    len(row.get("RequestBody") or ""),
                    (row.get("ResponseBody") or "")[:100],
                    len(row.get("ResponseBody") or ""),
                    row.get("PromptTokens"),
                    prop_keys,
                )
            raw_rows.append(row)
            traces.append({
                "trace_id": row.get("OperationId", ""),
                "name": row.get("Name", ""),
                "type": "trace",
                "timestamp": row.get("TimeGenerated", ""),
                "duration_ms": row.get("DurationMs", 0),
                "success": row.get("Success", True),
                "result_code": str(row.get("ResultCode", "")),
                "model": row.get("Model", ""),
                "prompt_tokens": row.get("PromptTokens", 0),
                "completion_tokens": row.get("CompletionTokens", 0),
                "total_tokens": row.get("TotalTokens", 0),
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
