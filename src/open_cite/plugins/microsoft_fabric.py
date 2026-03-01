"""
Microsoft Fabric discovery plugin for Open-CITE.

This plugin discovers AI/ML and data analytics assets in Microsoft Fabric:
- Workspaces
- Lakehouses (data engineering)
- Warehouses (data warehousing)
- Notebooks
- Pipelines (data integration)
- ML Models and Experiments
- Reports and Semantic Models (Power BI)
- Event Streams (real-time analytics)
- KQL Databases (real-time analytics)
- Capacities

Authentication is via Microsoft Entra ID (Azure AD) using either:
- Service principal (client_id + client_secret + tenant_id)
- A pre-acquired access token
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import requests

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)

FABRIC_API_BASE = "https://api.fabric.microsoft.com/v1"
ENTRA_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"


class MicrosoftFabricPlugin(BaseDiscoveryPlugin):
    """
    Microsoft Fabric discovery plugin.

    Discovers data analytics and AI/ML assets across Microsoft Fabric
    workspaces including lakehouses, warehouses, notebooks, pipelines,
    ML models, reports, semantic models, event streams, and KQL databases.
    """

    plugin_type = "microsoft_fabric"

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        workspace_ids: Optional[List[str]] = None,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the Microsoft Fabric plugin.

        Provide either (tenant_id + client_id + client_secret) for service
        principal auth, or a pre-acquired access_token.

        Args:
            tenant_id: Azure AD / Entra ID tenant ID.
            client_id: Application (client) ID of the registered app.
            client_secret: Client secret for the registered app.
            access_token: Pre-acquired bearer token (skips OAuth flow).
            workspace_ids: Optional list of workspace IDs to scope discovery.
                           If empty/None, all accessible workspaces are discovered.
            instance_id: Unique identifier for this plugin instance.
            display_name: Human-readable name for this instance.
        """
        super().__init__(instance_id=instance_id, display_name=display_name)

        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token = access_token
        self._token_expires_at: float = 0
        self.workspace_ids = workspace_ids or []

        # Caches
        self._workspaces_cache: List[Dict[str, Any]] = []
        self._capacities_cache: List[Dict[str, Any]] = []
        self._items_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Plugin metadata & factory
    # ------------------------------------------------------------------

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        return {
            "name": "Microsoft Fabric",
            "description": (
                "Discovers workspaces, lakehouses, warehouses, notebooks, "
                "pipelines, ML models, reports, and more in Microsoft Fabric"
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
                "workspace_ids": {
                    "label": "Workspace IDs (comma-separated, leave blank for all)",
                    "default": "",
                    "required": False,
                },
            },
            "env_vars": [
                "FABRIC_TENANT_ID",
                "FABRIC_CLIENT_ID",
                "FABRIC_CLIENT_SECRET",
            ],
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        workspace_ids_raw = config.get("workspace_ids", "")
        if isinstance(workspace_ids_raw, str):
            workspace_ids = [
                ws.strip() for ws in workspace_ids_raw.split(",") if ws.strip()
            ]
        else:
            workspace_ids = list(workspace_ids_raw or [])

        return cls(
            tenant_id=config.get("tenant_id"),
            client_id=config.get("client_id"),
            client_secret=config.get("client_secret"),
            access_token=config.get("access_token"),
            workspace_ids=workspace_ids,
            instance_id=instance_id,
            display_name=display_name,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @property
    def supported_asset_types(self) -> Set[str]:
        return {
            "workspace",
            "lakehouse",
            "warehouse",
            "notebook",
            "pipeline",
            "ml_model",
            "ml_experiment",
            "report",
            "semantic_model",
            "event_stream",
            "kql_database",
            "capacity",
        }

    def get_identification_attributes(self) -> List[str]:
        return [
            "fabric.tenant_id",
            "fabric.workspace_id",
            "fabric.item_id",
            "fabric.item_type",
            "fabric.capacity_id",
        ]

    def verify_connection(self) -> Dict[str, Any]:
        """Verify connection by listing workspaces."""
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)
            resp = requests.get(
                f"{FABRIC_API_BASE}/workspaces",
                headers=headers,
                params={"$top": 1},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "tenant_id": self.tenant_id,
                    "message": "Successfully connected to Microsoft Fabric",
                    "workspace_count_hint": len(data.get("value", [])),
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                "message": "Failed to connect to Microsoft Fabric",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to connect to Microsoft Fabric",
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List Microsoft Fabric assets.

        Supported asset_types: workspace, lakehouse, warehouse, notebook,
        pipeline, ml_model, ml_experiment, report, semantic_model,
        event_stream, kql_database, capacity.
        """
        if asset_type not in self.supported_asset_types:
            raise ValueError(
                f"Unsupported asset type: {asset_type}. "
                f"Supported: {', '.join(sorted(self.supported_asset_types))}"
            )

        with self._lock:
            if asset_type == "workspace":
                return self._list_workspaces()
            elif asset_type == "capacity":
                return self._list_capacities()
            else:
                return self._list_workspace_items(asset_type, **kwargs)

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "client_secret": "****" if self.client_secret else None,
            "access_token": "****" if self._access_token else None,
            "workspace_ids": self.workspace_ids,
        }

    def export_assets(self) -> Dict[str, Any]:
        return {
            "fabric_workspaces": self.list_assets("workspace"),
            "fabric_lakehouses": self.list_assets("lakehouse"),
            "fabric_warehouses": self.list_assets("warehouse"),
            "fabric_notebooks": self.list_assets("notebook"),
            "fabric_pipelines": self.list_assets("pipeline"),
            "fabric_ml_models": self.list_assets("ml_model"),
            "fabric_ml_experiments": self.list_assets("ml_experiment"),
            "fabric_reports": self.list_assets("report"),
            "fabric_semantic_models": self.list_assets("semantic_model"),
            "fabric_event_streams": self.list_assets("event_stream"),
            "fabric_kql_databases": self.list_assets("kql_database"),
            "fabric_capacities": self.list_assets("capacity"),
        }

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
            # Pre-acquired token with no expiry tracking – use as-is.
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
            "scope": FABRIC_SCOPE,
        }
        resp = requests.post(url, data=data, timeout=15)
        resp.raise_for_status()
        token_data = resp.json()
        self._access_token = token_data["access_token"]
        # Refresh a bit early to avoid edge-case expiry.
        self._token_expires_at = time.time() + token_data.get("expires_in", 3600) - 120
        logger.debug("Acquired new Fabric access token (expires in %ss)", token_data.get("expires_in"))
        return self._access_token

    @staticmethod
    def _auth_headers(token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Paginated GET helper
    # ------------------------------------------------------------------

    def _paginated_get(
        self,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        max_items: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all pages from a Fabric REST API list endpoint.

        The Fabric API uses a ``continuationToken`` in the response body to
        signal more pages.
        """
        items: List[Dict[str, Any]] = []
        params = dict(params or {})

        while True:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning("GET %s returned %d: %s", url, resp.status_code, resp.text[:200])
                break

            data = resp.json()
            page_items = data.get("value", [])
            items.extend(page_items)

            if len(items) >= max_items:
                items = items[:max_items]
                break

            continuation = data.get("continuationToken")
            if not continuation:
                break
            params["continuationToken"] = continuation

        return items

    # ------------------------------------------------------------------
    # Workspace discovery
    # ------------------------------------------------------------------

    def _get_target_workspaces(self) -> List[Dict[str, Any]]:
        """Return the workspaces to scan, honouring workspace_ids filter."""
        workspaces = self._list_workspaces()
        if self.workspace_ids:
            workspaces = [
                ws for ws in workspaces if ws.get("id") in self.workspace_ids
            ]
        return workspaces

    def _list_workspaces(self) -> List[Dict[str, Any]]:
        """List all accessible workspaces."""
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)
            raw = self._paginated_get(f"{FABRIC_API_BASE}/workspaces", headers)

            workspaces = []
            for ws in raw:
                workspaces.append({
                    "id": ws.get("id"),
                    "name": ws.get("displayName", ws.get("id")),
                    "description": ws.get("description", ""),
                    "type": "workspace",
                    "discovery_source": self.instance_id,
                    "capacity_id": ws.get("capacityId"),
                    "state": ws.get("state"),
                    "metadata": {
                        "fabric.tenant_id": self.tenant_id,
                        "fabric.workspace_id": ws.get("id"),
                    },
                })
            self._workspaces_cache = workspaces
            logger.info("Discovered %d Fabric workspaces", len(workspaces))
            return workspaces
        except Exception as e:
            logger.error("Failed to list Fabric workspaces: %s", e)
            return []

    # ------------------------------------------------------------------
    # Capacity discovery
    # ------------------------------------------------------------------

    def _list_capacities(self) -> List[Dict[str, Any]]:
        """List all accessible capacities."""
        try:
            token = self._get_access_token()
            headers = self._auth_headers(token)
            raw = self._paginated_get(f"{FABRIC_API_BASE}/capacities", headers)

            capacities = []
            for cap in raw:
                capacities.append({
                    "id": cap.get("id"),
                    "name": cap.get("displayName", cap.get("id")),
                    "type": "capacity",
                    "discovery_source": self.instance_id,
                    "sku": cap.get("sku"),
                    "state": cap.get("state"),
                    "region": cap.get("region"),
                    "metadata": {
                        "fabric.tenant_id": self.tenant_id,
                        "fabric.capacity_id": cap.get("id"),
                    },
                })
            self._capacities_cache = capacities
            logger.info("Discovered %d Fabric capacities", len(capacities))
            return capacities
        except Exception as e:
            logger.error("Failed to list Fabric capacities: %s", e)
            return []

    # ------------------------------------------------------------------
    # Generic workspace-item discovery
    # ------------------------------------------------------------------

    # Maps Open-CITE asset_type -> Fabric API segment name
    _ITEM_TYPE_MAP: Dict[str, str] = {
        "lakehouse": "lakehouses",
        "warehouse": "warehouses",
        "notebook": "notebooks",
        "pipeline": "dataPipelines",
        "ml_model": "mlModels",
        "ml_experiment": "mlExperiments",
        "report": "reports",
        "semantic_model": "semanticModels",
        "event_stream": "eventstreams",
        "kql_database": "kqlDatabases",
    }

    def _list_workspace_items(
        self,
        asset_type: str,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        List items of a given type across target workspaces.

        Args:
            asset_type: One of the keys in ``_ITEM_TYPE_MAP``.
            workspace_id: Optionally scope to a single workspace.
        """
        api_segment = self._ITEM_TYPE_MAP.get(asset_type)
        if not api_segment:
            logger.error("No API segment for asset type: %s", asset_type)
            return []

        if workspace_id:
            workspaces = [{"id": workspace_id}]
        else:
            workspaces = self._get_target_workspaces()

        all_items: List[Dict[str, Any]] = []
        token = self._get_access_token()
        headers = self._auth_headers(token)

        for ws in workspaces:
            ws_id = ws["id"]
            url = f"{FABRIC_API_BASE}/workspaces/{ws_id}/{api_segment}"
            try:
                raw = self._paginated_get(url, headers)
                for item in raw:
                    all_items.append({
                        "id": item.get("id"),
                        "name": item.get("displayName", item.get("id")),
                        "description": item.get("description", ""),
                        "type": asset_type,
                        "discovery_source": self.instance_id,
                        "workspace_id": ws_id,
                        "workspace_name": ws.get("name", ws_id),
                        "metadata": {
                            "fabric.tenant_id": self.tenant_id,
                            "fabric.workspace_id": ws_id,
                            "fabric.item_id": item.get("id"),
                            "fabric.item_type": asset_type,
                        },
                    })
            except Exception as e:
                logger.warning(
                    "Failed to list %s in workspace %s: %s",
                    asset_type, ws_id, e,
                )

        self._items_cache[asset_type] = all_items
        logger.info("Discovered %d Fabric %s(s)", len(all_items), asset_type)
        return all_items

    # ------------------------------------------------------------------
    # Convenience aggregation
    # ------------------------------------------------------------------

    def refresh_discovery(self):
        """Refresh all cached discovery data."""
        with self._lock:
            logger.info("Refreshing Microsoft Fabric discovery...")
            self._workspaces_cache.clear()
            self._capacities_cache.clear()
            self._items_cache.clear()

            self._list_workspaces()
            self._list_capacities()
            for asset_type in self._ITEM_TYPE_MAP:
                self._list_workspace_items(asset_type)

            logger.info("Microsoft Fabric discovery refresh complete")
