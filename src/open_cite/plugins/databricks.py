import os
import re
import logging
import time
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow import MlflowClient
from ..core import BaseDiscoveryPlugin, LoggingDefaultDict

logger = logging.getLogger(__name__)

# MLflow span type → normalized entity category (used for trace-level discovery)
_SPAN_TYPE_CATEGORY = {
    "LLM": "model",
    "CHAT_MODEL": "model",
    "EMBEDDING": "model",
    "TOOL": "tool",
    "AGENT": "agent",
    "RETRIEVER": "tool",
}

# Lock to serialize env-var mutations when multiple plugin instances call MLflow
_mlflow_env_lock = threading.Lock()


@contextmanager
def _databricks_mlflow_env(host: str, token: Optional[str] = None):
    """Context manager that sets DATABRICKS_HOST/TOKEN for MLflow calls.

    Uses a lock so concurrent plugin instances don't clobber each other.
    When token is None, DATABRICKS_TOKEN is left untouched so MLflow can
    auto-detect OAuth credentials from the environment.
    """
    with _mlflow_env_lock:
        old_host = os.environ.get("DATABRICKS_HOST")
        old_token = os.environ.get("DATABRICKS_TOKEN")
        token_was_set = False
        try:
            os.environ["DATABRICKS_HOST"] = host
            if token:
                os.environ["DATABRICKS_TOKEN"] = token
                token_was_set = True
                logger.debug("[mlflow_env] Set DATABRICKS_TOKEN (length=%d)", len(token))
            else:
                logger.warning("[mlflow_env] No token provided — MLflow may trigger OAuth")
            yield
        finally:
            if old_host is not None:
                os.environ["DATABRICKS_HOST"] = old_host
            else:
                os.environ.pop("DATABRICKS_HOST", None)
            if token_was_set:
                if old_token is not None:
                    os.environ["DATABRICKS_TOKEN"] = old_token
                else:
                    os.environ.pop("DATABRICKS_TOKEN", None)


class _GenieClient:
    """Genie REST API client backed by the Databricks SDK's ApiClient (inherits auth)."""

    def __init__(self, api_client):
        self._api = api_client

    def _get(self, path: str, query: Optional[Dict] = None) -> Any:
        return self._api.do("GET", path, query=query)

    def list_spaces(self) -> List[Dict]:
        data = self._get("/api/2.0/genie/spaces")
        return (data or {}).get("spaces", [])

    def get_space(self, space_id: str) -> Dict:
        return self._get(f"/api/2.0/genie/spaces/{space_id}") or {}

    def list_conversations(self, space_id: str, page_token: Optional[str] = None) -> Dict:
        query = {"page_token": page_token} if page_token else None
        return self._get(f"/api/2.0/genie/spaces/{space_id}/conversations", query=query) or {}

    def list_all_conversations(
        self,
        space_id: str,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """Paginate through all conversations, respecting since date and limit."""
        since_ms = int(since.timestamp() * 1000) if since else None
        all_conversations: List[Dict] = []
        page_token: Optional[str] = None

        while True:
            data = self.list_conversations(space_id, page_token=page_token)
            conversations = data.get("conversations", [])
            if not conversations:
                break

            for conv in conversations:
                if len(all_conversations) >= limit:
                    return all_conversations
                created = conv.get("created_timestamp", 0)
                if since_ms is not None and created < since_ms:
                    return all_conversations
                all_conversations.append(conv)

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return all_conversations

    def list_messages(self, space_id: str, conversation_id: str) -> List[Dict]:
        data = self._get(
            f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages",
        )
        return (data or {}).get("messages", [])

    def validate_connection(self) -> Dict:
        """Test API access by listing spaces."""
        try:
            spaces = self.list_spaces()
            return {"valid": True, "message": "OK", "spaces_count": len(spaces)}
        except Exception as e:
            return {"valid": False, "message": str(e)}


class DatabricksPlugin(BaseDiscoveryPlugin):
    """
    Databricks discovery plugin.
    """

    plugin_type = "databricks"

    @classmethod
    def plugin_metadata(cls):
        return {
            "name": "Databricks",
            "description": "Discovers AI/ML assets from MLflow experiments, Genie, and Unity Catalog",
            "required_fields": {
                "host": {"label": "Host", "default": os.environ.get("DATABRICKS_HOST", "https://dbc-xxx.cloud.databricks.com"), "required": False, "help": "Leave blank to auto-detect from environment"},
                "token": {"label": "Personal Access Token", "default": "", "required": False, "type": "password", "help": "Leave blank for OAuth/service principal auth"},
                "warehouse_id": {"label": "Warehouse ID (optional)", "default": "", "required": False},
                "lookback_days": {"label": "Initial Lookback (days)", "default": "90", "required": False, "type": "number", "min": 1, "max": 180},
                "ai_gateway_usage_table": {"label": "AI Gateway Usage Table", "default": os.environ.get("OPENCITE_AI_GATEWAY_USAGE_TABLE", ""), "required": False, "help": "e.g. system.ai_gateway.usage — leave blank to disable"},
                "gateway_poll_interval": {"label": "Gateway Poll Interval (seconds)", "default": "10", "required": False, "type": "number", "min": 5, "max": 300},
            },
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        dependencies = dependencies or {}
        lookback = config.get('lookback_days')
        try:
            lookback = int(lookback) if lookback else 90
        except (ValueError, TypeError):
            lookback = 90
        poll_interval = config.get('gateway_poll_interval')
        try:
            poll_interval = int(poll_interval) if poll_interval else 10
        except (ValueError, TypeError):
            poll_interval = 10
        return cls(
            host=config.get('host') or None,
            token=config.get('token') or None,
            warehouse_id=config.get('warehouse_id'),
            lookback_days=lookback,
            http_client=dependencies.get('http_client'),
            instance_id=instance_id,
            display_name=display_name,
            ai_gateway_usage_table=config.get('ai_gateway_usage_table') or None,
            gateway_poll_interval=poll_interval,
        )

    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        lookback_days: int = 90,
        http_client: Any = None,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
        ai_gateway_usage_table: Optional[str] = None,
        gateway_poll_interval: int = 10,
    ):
        super().__init__(instance_id=instance_id, display_name=display_name)
        self.host = host
        self.token = token
        self.warehouse_id = warehouse_id
        self.lookback_days = lookback_days
        self._last_query_time: Optional[datetime] = None

        # Build WorkspaceClient
        print(f"  [DatabricksPlugin] __init__ host={self.host!r}, token={'set' if self.token else 'None'}")
        if self.host:
            # Normalize host to always include https:// scheme
            self.host = self.host.rstrip("/")
            if not self.host.startswith(("https://", "http://")):
                self.host = f"https://{self.host}"

            if self.token:
                print(f"  [DatabricksPlugin] Creating WorkspaceClient(host={self.host}, token=***)")
                self.workspace_client = WorkspaceClient(host=self.host, token=self.token)
            else:
                print(f"  [DatabricksPlugin] Creating WorkspaceClient(host={self.host}) — no token, SDK auto-detect auth")
                self.workspace_client = WorkspaceClient(host=self.host)
        else:
            # No host — let the SDK auto-detect everything from environment.
            # In Databricks Apps the SDK reads DATABRICKS_HOST and resolves
            # the workspace URL + OAuth credentials automatically.
            print("  [DatabricksPlugin] Creating WorkspaceClient() — full SDK auto-detect from env")
            self.workspace_client = WorkspaceClient()
            self.host = self.workspace_client.config.host
            print(f"  [DatabricksPlugin] SDK resolved host: {self.host}")

        print(f"  [DatabricksPlugin] WorkspaceClient created, auth_type={self.workspace_client.config.auth_type}")

        # MLflow client pointed at Databricks tracking server
        # Extract token from SDK if not explicitly provided (Databricks App OAuth flow)
        import sys
        sys.stderr.write(f"[DatabricksPlugin MLflow setup] self.token={self.token!r}\n")
        sys.stderr.flush()
        mlflow_token = self.token
        sys.stderr.write(f"[DatabricksPlugin MLflow setup] mlflow_token={mlflow_token!r}, entering token extraction\n")
        sys.stderr.flush()
        if not mlflow_token:
            sys.stderr.write("[DatabricksPlugin MLflow setup] Token is None, extracting from SDK...\n")
            sys.stderr.flush()
            print("  [DatabricksPlugin] No static token, extracting from SDK for MLflow...")
            try:
                # SDK API varies: some versions take a headers dict, others return headers
                try:
                    headers = {}
                    self.workspace_client.config.authenticate(headers)
                except TypeError:
                    result = self.workspace_client.config.authenticate()
                    if callable(result):
                        headers = result() or {}
                    else:
                        headers = result or {}
                auth = headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    mlflow_token = auth[7:]
                    # Store in self.token so subsequent MLflow operations have it
                    self.token = mlflow_token
                    print(f"  [DatabricksPlugin] Extracted token from SDK (length={len(mlflow_token)})")
                else:
                    print(f"  [DatabricksPlugin] WARNING: SDK did not return Bearer token: {repr(auth[:50] if auth else auth)}")
            except Exception as exc:
                print(f"  [DatabricksPlugin] WARNING: Failed to extract token from SDK: {exc}")

        if not mlflow_token:
            import traceback
            logger.error("[DatabricksPlugin] MLflow client creation with NO TOKEN — will trigger OAuth! Stack:\n%s",
                        ''.join(traceback.format_stack()))

        print(f"  [DatabricksPlugin] Creating MLflow client (host={self.host}, token={'set' if mlflow_token else 'MISSING'})...")
        with _databricks_mlflow_env(self.host, mlflow_token):
            mlflow.set_tracking_uri("databricks")
            self.mlflow_client = MlflowClient("databricks")
        print("  [DatabricksPlugin] MLflow client created")

        # Disable urllib3 connection retries on the MLflow client's HTTP session
        # to avoid retry spam when artifact storage is unreachable.
        try:
            from urllib3.util.retry import Retry
            from requests.adapters import HTTPAdapter
            no_connect_retry = HTTPAdapter(max_retries=Retry(connect=0, read=2, status=2))
            store = getattr(self.mlflow_client, "_tracking_client", None)
            store = getattr(store, "store", None) if store else None
            sess = getattr(store, "_session", None) if store else None
            if sess and hasattr(sess, "mount"):
                sess.mount("https://", no_connect_retry)
                sess.mount("http://", no_connect_retry)
                print("  [DatabricksPlugin] Patched MLflow session: connect retries disabled")
            else:
                print("  [DatabricksPlugin] Could not patch MLflow session (structure changed)")
        except Exception as exc:
            print(f"  [DatabricksPlugin] Could not patch MLflow session retries: {exc}")

        # Inject custom session if provided
        if http_client:
            self.http_client = http_client
            if hasattr(self.workspace_client, "api_client") and hasattr(self.workspace_client.api_client, "_session"):
                 self.workspace_client.api_client._session = http_client._session
            elif hasattr(self.workspace_client, "_api_client") and hasattr(self.workspace_client._api_client, "_session"):
                 self.workspace_client._api_client._session = http_client._session
            else:
                logger.warning("Could not inject custom HTTP session into Databricks WorkspaceClient")

        # Genie client (uses SDK's api_client for auth)
        self.genie_client = _GenieClient(self.workspace_client.api_client)
        self.genie_spaces: Dict[str, Dict] = {}       # space_id → space metadata
        self.genie_traces: List[Dict] = []              # Converted Genie traces
        self._user_cache: Dict[str, Optional[str]] = {}  # databricks_user_id → email (or None if unresolvable)

        # MCP discovery stores (matching OTLP plugin pattern)
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}

        # MLflow trace-based discoveries (matching OTLP plugin pattern)
        _pid = self._instance_id
        self.discovered_agents: Dict[str, Dict[str, Any]] = LoggingDefaultDict(
            lambda: {
                "tools_used": set(),
                "models_used": set(),
                "first_seen": None,
                "last_seen": None,
                "metadata": {},
            },
            asset_type="agent", plugin_id=_pid,
        )
        self.discovered_tools: Dict[str, Dict[str, Any]] = LoggingDefaultDict(
            lambda: {"models": set(), "traces": [], "metadata": {}},
            asset_type="tool", plugin_id=_pid,
        )
        self.discovered_models: Dict[str, Dict[str, Any]] = LoggingDefaultDict(
            lambda: {
                "provider": "unknown",
                "tools_using": set(),
                "agents_using": set(),
                "usage_count": 0,
                "first_seen": None,
                "last_seen": None,
            },
            asset_type="model", plugin_id=_pid,
        )
        self.discovered_downstream: Dict[str, Dict[str, Any]] = LoggingDefaultDict(
            lambda: {
                "name": "",
                "type": "unknown",
                "endpoint": None,
                "tools_connecting": set(),
                "first_seen": None,
                "last_seen": None,
                "metadata": {},
            },
            asset_type="downstream_system", plugin_id=_pid,
        )
        self.lineage: Dict[str, Dict[str, Any]] = {}
        self.model_token_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0}
        )

        # AI Gateway usage table monitoring
        self._ai_gateway_table = (
            ai_gateway_usage_table
            or os.environ.get("OPENCITE_AI_GATEWAY_USAGE_TABLE")
            or None
        )
        self._gateway_poll_interval = gateway_poll_interval
        self._gateway_poll_stop = threading.Event()
        self._gateway_poll_thread: Optional[threading.Thread] = None
        self._last_gateway_event_time: Optional[datetime] = None

    def export_assets(self) -> Dict[str, Any]:
        """Export Databricks data assets used by AI/ML tools."""
        from open_cite.schema import DataAssetFormatter

        data_assets: List[Dict[str, Any]] = []
        try:
            used_tables: Dict[str, Dict[str, Any]] = {}

            try:
                ai_workload_usage = self.get_tables_used_by_ai_workloads(days=30)
                if ai_workload_usage:
                    logger.info(f"Found {len(ai_workload_usage)} tables from AI/ML workloads")
                    for table_name, usage in ai_workload_usage.items():
                        used_tables[table_name] = {
                            "ai_users": usage.get("ai_users", []),
                            "ai_experiments": usage.get("ai_experiments", []),
                            "ai_experiment_names": usage.get("ai_experiment_names", []),
                            "access_count": usage.get("access_count", 0),
                            "first_seen": usage.get("first_seen"),
                            "last_seen": usage.get("last_seen"),
                            "discovery_method": "mlflow_experiments",
                        }
            except Exception as e:
                logger.warning(f"Could not fetch AI workload table usage: {e}")

            try:
                genie_usage = self.get_tables_used_by_genie(days=30)
                if genie_usage:
                    logger.info(f"Found {len(genie_usage)} tables from Genie spaces")
                    for table_name, usage in genie_usage.items():
                        if table_name not in used_tables:
                            used_tables[table_name] = {"discovery_method": "genie_spaces"}
                        info = used_tables[table_name]
                        info["genie_users"] = usage.get("genie_users", [])
                        info["genie_spaces"] = usage.get("genie_spaces", [])
                        info.setdefault("access_count", 0)
                        info["access_count"] += usage.get("access_count", 0)
                        if usage.get("first_seen"):
                            if not info.get("first_seen") or usage["first_seen"] < info["first_seen"]:
                                info["first_seen"] = usage["first_seen"]
                        if usage.get("last_seen"):
                            if not info.get("last_seen") or usage["last_seen"] > info["last_seen"]:
                                info["last_seen"] = usage["last_seen"]
            except Exception as e:
                logger.warning(f"Could not fetch Genie table usage: {e}")

            if not used_tables:
                return {"data_assets": []}

            catalogs_needed: set = set()
            schemas_needed: set = set()
            for table_full_name in used_tables:
                parts = table_full_name.split(".")
                if len(parts) == 3:
                    catalogs_needed.add(parts[0])
                    schemas_needed.add((parts[0], parts[1]))

            for catalog_name in catalogs_needed:
                try:
                    catalogs = self.list_assets("catalog")
                    cat = next((c for c in catalogs if c["name"].lower() == catalog_name.lower()), None)
                    if cat:
                        data_assets.append(DataAssetFormatter.format_data_asset(
                            asset_id=cat["id"], name=cat["name"], asset_type="catalog",
                            discovery_source="databricks",
                            hierarchy={"catalog": cat["name"]},
                            metadata={"owner": cat.get("owner"), "comment": cat.get("comment"),
                                      "ai_usage": "Contains tables accessed by AI/ML tools"},
                        ))
                except Exception as e:
                    logger.warning(f"Failed to export catalog {catalog_name}: {e}")

            for catalog_name, schema_name in schemas_needed:
                try:
                    db_schemas = self.list_assets("schema", catalog_name=catalog_name)
                    sch = next((s for s in db_schemas if s["name"].lower() == schema_name.lower()), None)
                    if sch:
                        data_assets.append(DataAssetFormatter.format_data_asset(
                            asset_id=f"{catalog_name}.{schema_name}", name=sch["name"],
                            asset_type="schema", discovery_source="databricks",
                            full_name=f"{catalog_name}.{schema_name}",
                            hierarchy={"catalog": catalog_name, "schema": schema_name},
                            metadata={"owner": sch.get("owner"), "comment": sch.get("comment"),
                                      "ai_usage": "Contains tables accessed by AI/ML tools"},
                        ))
                except Exception as e:
                    logger.warning(f"Failed to export schema {catalog_name}.{schema_name}: {e}")

            for table_full_name, usage_info in used_tables.items():
                parts = table_full_name.split(".")
                if len(parts) != 3:
                    continue
                cat_n, sch_n, tbl_n = parts
                try:
                    tables = self.list_assets("table", catalog_name=cat_n, schema_name=sch_n)
                    tbl = next((t for t in tables if t["name"].lower() == tbl_n.lower()), None)
                    if tbl:
                        data_assets.append(DataAssetFormatter.format_data_asset(
                            asset_id=table_full_name, name=tbl["name"],
                            asset_type="table", discovery_source="databricks",
                            full_name=table_full_name,
                            hierarchy={"catalog": cat_n, "schema": sch_n, "table": tbl_n},
                            metadata={
                                "table_type": tbl.get("table_type"), "owner": tbl.get("owner"),
                                "comment": tbl.get("comment"),
                                "ai_users": usage_info.get("ai_users", []),
                                "ai_experiments": usage_info.get("ai_experiments", []),
                                "ai_experiment_names": usage_info.get("ai_experiment_names", []),
                                "ai_access_count": usage_info.get("access_count", 0),
                                "ai_first_seen": usage_info.get("first_seen"),
                                "ai_last_seen": usage_info.get("last_seen"),
                                "ai_discovery_method": usage_info.get("discovery_method", "unknown"),
                                "genie_users": usage_info.get("genie_users", []),
                                "genie_spaces": usage_info.get("genie_spaces", []),
                            },
                        ))
                except Exception as e:
                    logger.warning(f"Failed to export table {table_full_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to export Databricks assets: {e}")

        return {"data_assets": data_assets}

    def get_config(self) -> Dict[str, Any]:
        """Return plugin configuration (sensitive values masked)."""
        return {
            "host": self.host,
            "token": "****" if self.token else "",
            "warehouse_id": self.warehouse_id,
            "lookback_days": self.lookback_days,
        }

    def update_token(self, token: str):
        """Hot-swap the OAuth token used by all Databricks clients.

        Called when a new ``x-forwarded-access-token`` arrives from the
        Databricks App proxy.  Reference assignment is atomic under the
        GIL so in-flight calls on old clients complete safely.
        """
        logger.info("[DatabricksPlugin] Updating token (length=%d)", len(token))
        self.token = token
        self.workspace_client = WorkspaceClient(host=self.host, token=token)
        self.genie_client = _GenieClient(self.workspace_client.api_client)
        with _databricks_mlflow_env(self.host, token):
            mlflow.set_tracking_uri("databricks")
            self.mlflow_client = MlflowClient("databricks")
        logger.info("[DatabricksPlugin] All clients recreated with new token")

    @property
    def supported_asset_types(self) -> Set[str]:
        return {"catalog", "schema", "table", "volume", "model", "function", "tool", "agent", "downstream_system", "mcp_server", "mcp_tool"}

    def get_identification_attributes(self) -> List[str]:
        return ["databricks.workspace_id", "databricks.catalog", "databricks.schema"]

    def _get_warehouse_id(self) -> Optional[str]:
        """
        Get SQL warehouse ID, auto-discovering if not configured.

        Returns:
            Warehouse ID or None if none available
        """
        # Return configured warehouse if available
        if self.warehouse_id:
            return self.warehouse_id

        # Try to auto-discover a running warehouse
        try:
            warehouses = list(self.workspace_client.warehouses.list())

            # Prefer running warehouses
            for wh in warehouses:
                if wh.state and wh.state.value == "RUNNING":
                    logger.info(f"Auto-discovered running SQL warehouse: {wh.name} ({wh.id})")
                    self.warehouse_id = wh.id
                    return wh.id

            # Fall back to any available warehouse
            if warehouses:
                wh = warehouses[0]
                logger.info(f"Auto-discovered SQL warehouse: {wh.name} ({wh.id}) - may need to start it")
                self.warehouse_id = wh.id
                return wh.id

            logger.warning("No SQL warehouses found in workspace")
            return None

        except Exception as e:
            logger.warning(f"Could not auto-discover SQL warehouse: {e}")
            return None

    def _sql_connect_kwargs(self, http_path: str) -> Dict[str, Any]:
        """Build kwargs for databricks.sql.connect()."""
        kwargs = {
            "server_hostname": self.host.replace("https://", "").replace("http://", ""),
            "http_path": http_path,
        }
        if self.token:
            kwargs["access_token"] = self.token
        return kwargs

    def verify_connection(self) -> Dict[str, Any]:
        try:
            user = self.workspace_client.current_user.me()
            return {
                "success": True,
                "user": user.user_name,
                "active": user.active
            }
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        if asset_type == "catalog":
            return self._list_catalogs()
        elif asset_type == "schema":
            return self._list_schemas(kwargs.get("catalog_name"))
        elif asset_type == "table":
            return self._list_tables(kwargs.get("catalog_name"), kwargs.get("schema_name"))
        elif asset_type == "volume":
            return self._list_volumes(kwargs.get("catalog_name"), kwargs.get("schema_name"))
        elif asset_type == "model":
            # Combine Unity Catalog registered models with trace-discovered models
            uc_models = self._list_models(kwargs.get("catalog_name"), kwargs.get("schema_name"))
            return uc_models + self._list_discovered_models()
        elif asset_type == "function":
            return self._list_functions(kwargs.get("catalog_name"), kwargs.get("schema_name"))
        elif asset_type == "tool":
            return self._list_discovered_tools()
        elif asset_type == "agent":
            return self._list_discovered_agents()
        elif asset_type == "downstream_system":
            return self._list_discovered_downstream()
        elif asset_type == "mcp_server":
            return self._list_mcp_servers(**kwargs)
        elif asset_type == "mcp_tool":
            return self._list_mcp_tools(**kwargs)
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

    def _list_catalogs(self) -> List[Dict[str, Any]]:
        catalogs = []
        for catalog in self.workspace_client.catalogs.list():
            catalogs.append({
                "name": catalog.name,
                "id": catalog.name,  # Databricks uses name as unique identifier
                "type": "catalog",   # Add type for Open-CITE schema compliance
                "owner": catalog.owner,
                "comment": catalog.comment
            })
        return catalogs

    def _list_schemas(self, catalog_name: str) -> List[Dict[str, Any]]:
        if not catalog_name:
            raise ValueError("catalog_name is required for listing schemas")
        schemas = []
        for schema in self.workspace_client.schemas.list(catalog_name=catalog_name):
            schemas.append({
                "name": schema.name,
                "catalog": schema.catalog_name,  # Normalize to 'catalog' for Open-CITE schema
                "owner": schema.owner,
                "comment": schema.comment
            })
        return schemas

    def _list_tables(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            raise ValueError("catalog_name and schema_name are required for listing tables")
        tables = []
        for table in self.workspace_client.tables.list(catalog_name=catalog_name, schema_name=schema_name):
            tables.append({
                "name": table.name,
                "catalog": table.catalog_name,  # Normalize to 'catalog' for Open-CITE schema
                "schema": table.schema_name,    # Normalize to 'schema' for Open-CITE schema
                "table_type": table.table_type.value if table.table_type else None,
                "owner": table.owner,
                "comment": table.comment
            })
        return tables

    def _list_volumes(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            raise ValueError("catalog_name and schema_name are required for listing volumes")
        volumes = []
        for volume in self.workspace_client.volumes.list(catalog_name=catalog_name, schema_name=schema_name):
            volumes.append({
                "name": volume.name,
                "catalog": volume.catalog_name,  # Normalize to 'catalog' for Open-CITE schema
                "schema": volume.schema_name,    # Normalize to 'schema' for Open-CITE schema
                "volume_type": volume.volume_type.value if volume.volume_type else None,
                "owner": volume.owner,
                "comment": volume.comment
            })
        return volumes

    def _list_models(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            return []
        models = []
        for model in self.workspace_client.registered_models.list(catalog_name=catalog_name, schema_name=schema_name):
            models.append({
                "name": model.name,
                "catalog": model.catalog_name,  # Normalize to 'catalog' for Open-CITE schema
                "schema": model.schema_name,    # Normalize to 'schema' for Open-CITE schema
                "owner": model.owner,
                "comment": model.comment
            })
        return models

    def _list_functions(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            raise ValueError("catalog_name and schema_name are required for listing functions")
        functions = []
        for func in self.workspace_client.functions.list(catalog_name=catalog_name, schema_name=schema_name):
            functions.append({
                "name": func.name,
                "catalog": func.catalog_name,  # Normalize to 'catalog' for Open-CITE schema
                "schema": func.schema_name,    # Normalize to 'schema' for Open-CITE schema
                "owner": func.owner,
                "comment": func.comment,
                "routine_definition": func.routine_definition
            })
        return functions

    def get_table_usage_from_audit_logs(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Query Databricks system tables to discover which tables have been accessed.

        This queries system.access.audit for table access events to determine
        which tables are actually being used (particularly by AI/ML workloads).

        Args:
            days: Number of days to look back for audit logs (default: 30)

        Returns:
            Dict mapping table full names to usage information:
            {
                "catalog.schema.table": {
                    "access_count": int,
                    "users": [list of users],
                    "notebooks": [list of notebooks],
                    "first_seen": str,
                    "last_seen": str
                }
            }
        """
        table_usage = {}

        try:
            # Query system.access.audit for table access events
            # This requires access to system tables (typically available in Databricks workspaces)
            query = f"""
            SELECT
                request_params.table_full_name as table_name,
                user_identity.email as user,
                request_params.notebook_id as notebook,
                MIN(event_date) as first_access,
                MAX(event_date) as last_access,
                COUNT(*) as access_count
            FROM system.access.audit
            WHERE
                event_date >= DATE_SUB(CURRENT_DATE(), {days})
                AND action_name IN ('getTable', 'readTable', 'scanTable')
                AND request_params.table_full_name IS NOT NULL
            GROUP BY
                request_params.table_full_name,
                user_identity.email,
                request_params.notebook_id
            """

            # Execute query using Databricks SQL
            from databricks import sql

            with sql.connect(
                **self._sql_connect_kwargs("/sql/1.0/warehouses/...")  # Would need actual warehouse path
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)

                    for row in cursor.fetchall():
                        table_name = row[0]
                        user = row[1]
                        notebook = row[2]
                        first_access = row[3]
                        last_access = row[4]
                        access_count = row[5]

                        if table_name not in table_usage:
                            table_usage[table_name] = {
                                "access_count": 0,
                                "users": set(),
                                "notebooks": set(),
                                "first_seen": None,
                                "last_seen": None
                            }

                        table_usage[table_name]["access_count"] += access_count
                        table_usage[table_name]["users"].add(user)
                        if notebook:
                            table_usage[table_name]["notebooks"].add(notebook)

                        if not table_usage[table_name]["first_seen"] or first_access < table_usage[table_name]["first_seen"]:
                            table_usage[table_name]["first_seen"] = str(first_access)

                        if not table_usage[table_name]["last_seen"] or last_access > table_usage[table_name]["last_seen"]:
                            table_usage[table_name]["last_seen"] = str(last_access)

        except Exception as e:
            logger.warning(f"Could not query audit logs (may not have access): {e}")
            logger.info("Falling back to trace-based discovery only")

        # Convert sets to lists for JSON serialization
        for table_name in table_usage:
            table_usage[table_name]["users"] = list(table_usage[table_name]["users"])
            table_usage[table_name]["notebooks"] = list(table_usage[table_name]["notebooks"])

        return table_usage

    def get_tables_used_by_ai_workloads(self, days: int = 30, warehouse_id: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get tables that are used by AI/ML workloads specifically.

        This queries system tables to find tables accessed by:
        1. MLflow experiments/runs
        2. Model serving endpoints
        3. Notebooks running AI/ML workloads

        Args:
            days: Number of days to look back
            warehouse_id: SQL warehouse ID for executing queries (optional, will auto-discover if not provided)

        Returns:
            Dict mapping table full names to AI usage information:
            {
                "catalog.schema.table": {
                    "access_count": int,
                    "ai_users": [list of users],
                    "ai_experiments": [list of MLflow experiment IDs],
                    "ai_models": [list of model names],
                    "first_seen": str,
                    "last_seen": str
                }
            }
        """
        ai_table_usage = {}

        try:
            from databricks import sql

            # Get warehouse ID (use provided, configured, or auto-discover)
            wh_id = warehouse_id or self._get_warehouse_id()

            if not wh_id:
                logger.warning("No SQL warehouse available. Cannot query AI workload table usage.")
                logger.info("Please set DATABRICKS_WAREHOUSE_ID or ensure a SQL warehouse exists in your workspace.")
                return ai_table_usage

            http_path = f"/sql/1.0/warehouses/{wh_id}"

            # Query for tables used in MLflow experiments and runs
            query = f"""
            WITH mlflow_table_access AS (
                -- Tables accessed during MLflow experiment runs
                SELECT DISTINCT
                    audit.request_params.table_full_name as table_name,
                    audit.user_identity.email as user,
                    runs.experiment_id,
                    experiments.name as experiment_name,
                    runs.run_id,
                    MIN(audit.event_date) as first_access,
                    MAX(audit.event_date) as last_access,
                    COUNT(*) as access_count
                FROM system.access.audit AS audit
                INNER JOIN (
                    SELECT run_id, experiment_id, artifact_uri
                    FROM system.mlflow.runs
                    WHERE status = 'FINISHED'
                ) AS runs ON audit.request_params.notebook_id LIKE CONCAT('%', runs.run_id, '%')
                INNER JOIN (
                    SELECT experiment_id, name
                    FROM system.mlflow.experiments
                ) AS experiments ON runs.experiment_id = experiments.experiment_id
                WHERE
                    audit.event_date >= DATE_SUB(CURRENT_DATE(), {days})
                    AND audit.action_name IN ('getTable', 'readTable', 'scanTable')
                    AND audit.request_params.table_full_name IS NOT NULL
                GROUP BY
                    audit.request_params.table_full_name,
                    audit.user_identity.email,
                    runs.experiment_id,
                    experiments.name,
                    runs.run_id
            )
            SELECT
                table_name,
                user,
                experiment_id,
                experiment_name,
                MIN(first_access) as first_access,
                MAX(last_access) as last_access,
                SUM(access_count) as total_access_count
            FROM mlflow_table_access
            GROUP BY table_name, user, experiment_id, experiment_name
            ORDER BY table_name, total_access_count DESC
            """

            with sql.connect(
                **self._sql_connect_kwargs(http_path)
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)

                    for row in cursor.fetchall():
                        table_name = row[0]
                        user = row[1]
                        experiment_id = row[2]
                        experiment_name = row[3]
                        first_access = row[4]
                        last_access = row[5]
                        access_count = row[6]

                        if table_name not in ai_table_usage:
                            ai_table_usage[table_name] = {
                                "access_count": 0,
                                "ai_users": set(),
                                "ai_experiments": set(),
                                "ai_experiment_names": set(),
                                "first_seen": None,
                                "last_seen": None
                            }

                        ai_table_usage[table_name]["access_count"] += access_count
                        ai_table_usage[table_name]["ai_users"].add(user)
                        ai_table_usage[table_name]["ai_experiments"].add(experiment_id)
                        ai_table_usage[table_name]["ai_experiment_names"].add(experiment_name)

                        if not ai_table_usage[table_name]["first_seen"] or first_access < ai_table_usage[table_name]["first_seen"]:
                            ai_table_usage[table_name]["first_seen"] = str(first_access)

                        if not ai_table_usage[table_name]["last_seen"] or last_access > ai_table_usage[table_name]["last_seen"]:
                            ai_table_usage[table_name]["last_seen"] = str(last_access)

            logger.info(f"Found {len(ai_table_usage)} tables accessed by AI/ML workloads (MLflow experiments)")

        except ImportError:
            logger.warning("databricks-sql-connector not installed. Cannot query AI workload table usage.")
        except Exception as e:
            logger.warning(f"Could not query AI workload table usage: {e}")
            logger.info("This requires SQL warehouse access and system.mlflow tables")

        # Convert sets to lists for JSON serialization
        for table_name in ai_table_usage:
            ai_table_usage[table_name]["ai_users"] = list(ai_table_usage[table_name]["ai_users"])
            ai_table_usage[table_name]["ai_experiments"] = list(ai_table_usage[table_name]["ai_experiments"])
            ai_table_usage[table_name]["ai_experiment_names"] = list(ai_table_usage[table_name]["ai_experiment_names"])

        return ai_table_usage

    def get_tables_used_by_genie(self, days: int = 30, warehouse_id: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get tables that are used by Databricks Genie spaces.

        This queries system tables to find tables accessed by queries originating from Genie.
        It joins system.query.history with system.access.table_lineage.

        Args:
            days: Number of days to look back
            warehouse_id: SQL warehouse ID for executing queries (optional)

        Returns:
            Dict mapping table full names to Genie usage information:
            {
                "catalog.schema.table": {
                    "access_count": int,
                    "genie_users": [list of users],
                    "genie_spaces": [list of space IDs],
                    "first_seen": str,
                    "last_seen": str
                }
            }
        """
        genie_table_usage = {}

        try:
            from databricks import sql

            wh_id = warehouse_id or self._get_warehouse_id()
            if not wh_id:
                logger.warning("No SQL warehouse available. Cannot query Genie table usage.")
                return genie_table_usage

            http_path = f"/sql/1.0/warehouses/{wh_id}"

            # Query for tables touched by Genie-originated queries
            query = f"""
            SELECT
                lineage.source_table_full_name as table_name,
                history.executed_by as user,
                history.query_source.genie_space_id as space_id,
                MIN(history.start_time) as first_access,
                MAX(history.start_time) as last_access,
                COUNT(*) as access_count
            FROM system.access.table_lineage lineage
            JOIN system.query.history history ON lineage.entity_id = history.statement_id
            WHERE
                lineage.entity_type = 'QUERY'
                AND history.query_source.genie_space_id IS NOT NULL
                AND history.start_time >= DATE_SUB(CURRENT_DATE(), {days})
            GROUP BY
                lineage.source_table_full_name,
                history.executed_by,
                history.query_source.genie_space_id
            """

            with sql.connect(
                **self._sql_connect_kwargs(http_path)
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)

                    for row in cursor.fetchall():
                        table_name = row[0]
                        user = row[1]
                        space_id = row[2]
                        first_access = row[3]
                        last_access = row[4]
                        access_count = row[5]

                        if table_name not in genie_table_usage:
                            genie_table_usage[table_name] = {
                                "access_count": 0,
                                "genie_users": set(),
                                "genie_spaces": set(),
                                "first_seen": None,
                                "last_seen": None
                            }

                        genie_table_usage[table_name]["access_count"] += access_count
                        genie_table_usage[table_name]["genie_users"].add(user)
                        genie_table_usage[table_name]["genie_spaces"].add(space_id)

                        if not genie_table_usage[table_name]["first_seen"] or first_access < genie_table_usage[table_name]["first_seen"]:
                            genie_table_usage[table_name]["first_seen"] = str(first_access)

                        if not genie_table_usage[table_name]["last_seen"] or last_access > genie_table_usage[table_name]["last_seen"]:
                            genie_table_usage[table_name]["last_seen"] = str(last_access)

            logger.info(f"Found {len(genie_table_usage)} tables accessed by Genie spaces")

        except ImportError:
            logger.warning("databricks-sql-connector not installed. Cannot query Genie usage.")
        except Exception as e:
            logger.warning(f"Could not query Genie table usage: {e}")
            logger.info("This requires SQL warehouse access and system tables (query.history, access.table_lineage)")

        # Convert sets to lists
        for table_name in genie_table_usage:
            genie_table_usage[table_name]["genie_users"] = list(genie_table_usage[table_name]["genie_users"])
            genie_table_usage[table_name]["genie_spaces"] = list(genie_table_usage[table_name]["genie_spaces"])

        return genie_table_usage

    # =========================================================================
    # Lifecycle overrides
    # =========================================================================

    def _run_discovery(self, days: int, label: str = "start"):
        """Run all discovery steps. Intended to be called from a background thread."""
        try:
            self.discover_from_traces(days=days)
        except Exception as e:
            logger.warning(f"Trace discovery failed ({label}): {e}")
        try:
            self.discover_from_genie(days=days)
        except Exception as e:
            logger.warning(f"Genie discovery failed ({label}): {e}")
        try:
            self.discover_mcp_servers()
        except Exception as e:
            logger.warning(f"MCP discovery failed ({label}): {e}")
        logger.info(f"Discovery complete ({label}) for {self.instance_id}")

    def _forward_to_otel(self, otlp: dict):
        """Forward an OTLP payload to the embedded OTel plugin for discovery.

        This ensures that Databricks-sourced traces (MLflow, Genie) go through
        the OTel plugin's model/agent detection pipeline.
        """
        try:
            from open_cite.api import app as api_app
            otel = api_app._default_otel_plugin
            if otel is not None:
                otel._ingest_traces(otlp)
        except Exception as e:
            logger.debug("Could not forward to OTel plugin: %s", e)

    # =========================================================================
    # AI Gateway Usage Table Monitoring
    # =========================================================================

    def _poll_ai_gateway_usage(self):
        """Background loop that polls the AI Gateway usage table for new rows."""
        logger.info("AI Gateway usage polling thread started (interval=%ds)", self._gateway_poll_interval)
        poll_interval = self._gateway_poll_interval

        # Initial lookback: use plugin's lookback_days for first fetch
        if self._last_gateway_event_time is None:
            from datetime import timedelta, timezone
            self._last_gateway_event_time = (
                datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
            )

        while not self._gateway_poll_stop.is_set():
            try:
                self._fetch_and_process_gateway_usage()
            except Exception as e:
                logger.warning(f"AI Gateway usage poll error: {e}")

            # Sleep in small increments to allow fast shutdown
            self._gateway_poll_stop.wait(timeout=poll_interval)

        logger.info("AI Gateway usage polling thread stopped")

    def _fetch_and_process_gateway_usage(self):
        """Execute a single poll: query new rows and convert to OTLP traces."""
        from databricks import sql
        from open_cite.plugins.databricks_otlp_converter import ai_gateway_usage_to_otlp

        wh_id = self._get_warehouse_id()
        if not wh_id:
            logger.debug("No SQL warehouse available for AI Gateway usage polling")
            return

        http_path = f"/sql/1.0/warehouses/{wh_id}"
        table = self._ai_gateway_table

        query = f"""
        SELECT *
        FROM {table}
        WHERE event_time > :high_water
        ORDER BY event_time ASC
        LIMIT 500
        """

        try:
            with sql.connect(**self._sql_connect_kwargs(http_path)) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        query,
                        {"high_water": self._last_gateway_event_time},
                    )
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
        except Exception as e:
            logger.warning(f"AI Gateway usage query failed: {e}")
            return

        if not rows:
            return

        logger.info(f"[AI Gateway] Fetched {len(rows)} new usage rows")

        new_high_water = self._last_gateway_event_time
        forwarded = 0
        # Batch all rows into a single OTLP payload for one _ingest_traces call
        batched_resource_spans = []

        for raw_row in rows:
            row = dict(zip(columns, raw_row))

            # Update high-water mark
            evt = row.get("event_time")
            if evt is not None and (new_high_water is None or evt > new_high_water):
                new_high_water = evt

            try:
                otlp = ai_gateway_usage_to_otlp(row)
                batched_resource_spans.extend(otlp.get("resourceSpans", []))
                forwarded += 1
            except Exception as e:
                logger.warning(
                    f"Failed to convert AI Gateway row {row.get('request_id')}: {e}"
                )

        if batched_resource_spans:
            batched_otlp = {"resourceSpans": batched_resource_spans}
            self._forward_to_otel(batched_otlp)
            self._deliver_to_webhooks(batched_otlp)

        # Advance high-water mark by at least 1 second since event_time
        # has only second-level granularity — without this, rows at the
        # exact same second would be re-fetched on every poll.
        if new_high_water is not None and new_high_water != self._last_gateway_event_time:
            from datetime import timedelta
            new_high_water = new_high_water + timedelta(seconds=1)
        self._last_gateway_event_time = new_high_water
        logger.info(
            f"[AI Gateway] Forwarded {forwarded}/{len(rows)} rows as OTLP traces"
        )

    def start(self):
        """Start the plugin — runs initial discovery in a background thread
        so the API remains responsive (gunicorn single-worker safe)."""
        self._status = "running"
        thread = threading.Thread(
            target=self._run_discovery,
            args=(self.lookback_days, "start"),
            daemon=True,
        )
        thread.start()

        # Start AI Gateway usage table polling if configured
        if self._ai_gateway_table:
            self._gateway_poll_stop.clear()
            self._gateway_poll_thread = threading.Thread(
                target=self._poll_ai_gateway_usage,
                daemon=True,
                name=f"gw-poll-{self.instance_id}",
            )
            self._gateway_poll_thread.start()
            logger.info(
                f"Started AI Gateway usage polling: table={self._ai_gateway_table}"
            )

        logger.info(f"Started Databricks plugin {self.instance_id} (discovery running in background)")

    def stop(self):
        """Stop the plugin and any background threads."""
        # Stop AI Gateway polling thread
        if self._gateway_poll_thread is not None:
            self._gateway_poll_stop.set()
            self._gateway_poll_thread.join(timeout=5.0)
            self._gateway_poll_thread = None
            logger.info("AI Gateway usage polling stopped")

        self._status = "stopped"
        if self._webhook_executor:
            self._webhook_executor.shutdown(wait=False)
            self._webhook_executor = None
        logger.info(f"Stopped plugin {self.instance_id}")

    # =========================================================================
    # MLflow Trace Discovery
    # =========================================================================

    def discover_from_traces(self, days: int = 90, max_per_experiment: int = 100):
        """
        Discover AI entities from MLflow experiments. Attempts trace-level
        discovery first (richer span data), then falls back to run-level
        discovery if the Traces API is unavailable in the workspace.

        All calls go through the Databricks SDK (no mlflow package needed).

        Args:
            days: Number of days to look back
            max_per_experiment: Maximum traces/runs to fetch per experiment
        """
        logger.info(f"Starting MLflow discovery (lookback={days} days)")

        try:
            experiments = list(self.workspace_client.experiments.search_experiments())
        except Exception as e:
            logger.error(f"Failed to search experiments: {e}")
            return

        if not experiments:
            logger.warning("No experiments found for discovery")
            return

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 24 * 60 * 60 * 1000)

        # Try trace-level discovery first (richer data from spans)
        traces_available = self._discover_from_traces(
            experiments, start_ms, max_per_experiment
        )

        # Always run run-level discovery (complementary: tags, params, metrics, model_inputs)
        self._discover_from_runs(experiments, start_ms, max_per_experiment)

        if not traces_available:
            logger.debug(
                "Traces API unavailable — discovery used runs only. "
                "Run-level discovery extracts models, tools, and agents from "
                "tags, params, metrics, and model inputs."
            )

        logger.info(
            f"MLflow discovery complete. "
            f"Discovered {len(self.discovered_agents)} agents, "
            f"{len(self.discovered_tools)} tools, "
            f"{len(self.discovered_models)} models, "
            f"{len(self.discovered_downstream)} downstream systems"
        )

        self._last_query_time = datetime.utcnow()
        self.notify_data_changed()

    def _discover_from_traces(
        self,
        experiments: List,
        start_ms: int,
        max_per_experiment: int,
    ) -> bool:
        """
        Fetch MLflow traces via the MLflow SDK and process their spans.

        Returns True if the Traces API is available, False if not.
        """
        succeeded = 0
        failed = 0
        total_traces = 0

        batched_resource_spans = []

        with _databricks_mlflow_env(self.host, self.token):
            for exp in experiments:
                exp_id = exp.experiment_id
                exp_name = getattr(exp, "name", "unknown")

                try:
                    page_token = None
                    exp_traces = 0
                    while True:
                        results = self.mlflow_client.search_traces(
                            locations=[exp_id],
                            filter_string=f"attributes.timestamp_ms >= {start_ms}",
                            max_results=min(max_per_experiment - exp_traces, 100),
                            page_token=page_token,
                        )

                        for trace in results:
                            if trace.data.spans:
                                self._process_trace_object(trace, exp_name)
                                from open_cite.plugins.databricks_otlp_converter import mlflow_trace_to_otlp
                                otlp = mlflow_trace_to_otlp(trace, exp_name)
                                batched_resource_spans.extend(otlp.get("resourceSpans", []))
                                total_traces += 1
                            exp_traces += 1

                        if not results.token or exp_traces >= max_per_experiment:
                            break
                        page_token = results.token

                    succeeded += 1

                except Exception as e:
                    error_str = str(e)
                    if "ENDPOINT_NOT_FOUND" in error_str or "404" in error_str:
                        logger.debug("MLflow Traces API not available — using run-level discovery")
                        return False
                    if "INVALID_PARAMETER_VALUE" in error_str and "trace storage schema" in error_str.lower():
                        logger.debug(
                            "Experiment %s (%s) has no trace storage schema — skipping trace search",
                            exp_id, exp_name,
                        )
                        continue
                    if "Connection refused" in error_str or "NewConnectionError" in error_str:
                        logger.warning(
                            "MLflow trace fetch hit connection refused — stopping trace discovery: %s", e
                        )
                        break
                    failed += 1
                    logger.warning(f"Error fetching traces for experiment {exp_id} ({exp_name}): {e}")

        if batched_resource_spans:
            batched_otlp = {"resourceSpans": batched_resource_spans}
            self._forward_to_otel(batched_otlp)
            if self._webhook_urls:
                self._deliver_to_webhooks(batched_otlp)

        logger.info(
            f"Trace discovery: {succeeded} experiments succeeded, "
            f"{failed} failed, {total_traces} traces processed"
        )
        return True

    def _discover_from_runs(
        self,
        experiments: List,
        start_ms: int,
        max_per_experiment: int,
    ):
        """Fetch MLflow runs via the SDK and extract entities from tags/params/metrics."""
        filter_string = f"attributes.start_time >= {start_ms}"
        succeeded = 0
        failed = 0
        total_runs = 0

        for exp in experiments:
            exp_id = exp.experiment_id
            exp_name = getattr(exp, "name", "unknown")

            try:
                runs = self.workspace_client.experiments.search_runs(
                    experiment_ids=[exp_id],
                    filter=filter_string,
                    max_results=max_per_experiment,
                )

                for run in runs:
                    self._process_mlflow_run(run, exp_name)
                    total_runs += 1

                succeeded += 1

            except Exception as e:
                failed += 1
                logger.warning(f"Error querying runs for experiment {exp_id} ({exp_name}): {e}")

        logger.info(
            f"Run discovery: {succeeded} experiments succeeded, "
            f"{failed} failed, {total_runs} runs processed"
        )

    def refresh_traces(self, days: Optional[int] = None):
        """
        Public method to trigger on-demand trace re-discovery.

        If _last_query_time is set, only looks back to that time (incremental).
        Otherwise falls back to lookback_days or the explicit days parameter.

        Runs in a background thread so the API endpoint returns immediately.

        Args:
            days: Override number of days to look back (None = auto from last query)
        """
        if days is None:
            if self._last_query_time:
                elapsed = datetime.utcnow() - self._last_query_time
                days = max(1, int(elapsed.total_seconds() / 86400) + 1)
            else:
                days = self.lookback_days
        logger.info(f"Refreshing traces (lookback={days} days) for {self.instance_id}")
        thread = threading.Thread(
            target=self._run_discovery,
            args=(days, f"refresh lookback={days}d"),
            daemon=True,
        )
        thread.start()

    # =========================================================================
    # Genie Conversation Discovery
    # =========================================================================

    CHARS_PER_TOKEN_TEXT = 4
    CHARS_PER_TOKEN_SQL = 3.5

    @staticmethod
    def _extract_tables_from_sql(sql: str) -> List[str]:
        """Extract table names from SQL (FROM/JOIN clauses).

        Handles both plain identifiers and backtick-quoted identifiers
        (e.g. `catalog`.`schema`.`table`).
        """
        # Match an identifier: either `backtick-quoted` or plain
        ident = r'`[^`]+`|[a-zA-Z_]\w*'
        pattern = rf'(?:FROM|JOIN)\s+((?:{ident})(?:\.(?:{ident}))*)'
        raw = set(re.findall(pattern, sql, re.IGNORECASE))
        # Strip backticks from results
        return [t.replace('`', '') for t in raw]

    @staticmethod
    def _estimate_tokens(text: Optional[str], chars_per_token: float = 4) -> int:
        """Estimate token count from text length."""
        if not text:
            return 0
        return int(len(text) / chars_per_token + 0.999)

    def discover_from_genie(
        self,
        days: int = 90,
        max_messages: int = 100,
        space_ids: Optional[List[str]] = None,
    ):
        """
        Fetch Genie space conversations and discover entities.

        Discovers agents (per-space Genie agent), the databricks-genie model,
        and downstream systems (SQL tables referenced in generated queries).

        Args:
            days: Number of days to look back for conversations
            max_messages: Maximum messages to process across all spaces
            space_ids: Optional list of space IDs to filter (None = all)
        """
        logger.info(f"Starting Genie conversation discovery (lookback={days} days)")

        validation = self.genie_client.validate_connection()
        if not validation.get("valid"):
            logger.warning(f"Genie discovery skipped - validation failed: {validation.get('message')}")
            return

        logger.info(f"Genie access validated: {validation.get('spaces_count', 0)} space(s)")

        try:
            if space_ids:
                spaces = [self.genie_client.get_space(sid) for sid in space_ids]
            else:
                spaces = self.genie_client.list_spaces()
        except Exception as e:
            logger.error(f"Failed to list Genie spaces: {e}")
            return

        if not spaces:
            logger.info("No Genie spaces found")
            return

        since_date = datetime.utcfromtimestamp(
            datetime.utcnow().timestamp() - days * 24 * 60 * 60
        )

        messages_processed = 0
        now = datetime.utcnow().isoformat()
        batched_resource_spans = []

        for space in spaces:
            if messages_processed >= max_messages:
                break

            space_id = space.get("space_id", "")
            space_title = space.get("title", space_id)

            # Store space metadata
            self.genie_spaces[space_id] = {
                "space_id": space_id,
                "title": space_title,
                "description": space.get("description"),
                "warehouse_id": space.get("warehouse_id"),
                "created_timestamp": space.get("created_timestamp"),
                "last_updated_timestamp": space.get("last_updated_timestamp"),
            }

            # Register space as an agent
            agent_name = f"Genie ({space_title})"
            agent = self.discovered_agents[agent_name]
            if not agent["first_seen"]:
                agent["first_seen"] = now
            agent["last_seen"] = now
            agent["metadata"]["discovery_source"] = "databricks/genie"
            agent["metadata"]["genie_space_id"] = space_id
            agent["metadata"]["genie_space_name"] = space_title
            agent["metadata"]["agent_type"] = "sql-assistant"
            if space.get("description"):
                agent["metadata"]["description"] = space["description"]
            if space.get("warehouse_id"):
                agent["metadata"]["warehouse_id"] = space["warehouse_id"]
            table_ids = space.get("table_identifiers") or []
            if table_ids:
                agent["metadata"]["configured_tables"] = ", ".join(table_ids)

            # Register configured tables as downstream data assets
            for table_ref in (space.get("table_identifiers") or []):
                ds_id = table_ref.lower().replace(" ", "_")
                ds = self.discovered_downstream[ds_id]
                ds["name"] = table_ref
                ds["type"] = "database"
                ds["tools_connecting"].add(agent_name)
                if not ds["first_seen"]:
                    ds["first_seen"] = now
                ds["last_seen"] = now
                ds["metadata"]["discovery_source"] = "databricks/genie"
                ds["metadata"]["genie_space_id"] = space_id
                self._add_lineage(agent_name, "agent", ds_id, "downstream", "queries")

            try:
                conversations = self.genie_client.list_all_conversations(
                    space_id,
                    since=since_date,
                    limit=max_messages - messages_processed,
                )
            except Exception as e:
                logger.warning(f"Failed to list conversations for space {space_title}: {e}")
                continue

            for conv in conversations:
                if messages_processed >= max_messages:
                    break

                conv_id = conv.get("conversation_id") or conv.get("id", "")
                if not conv_id:
                    continue
                try:
                    messages = self.genie_client.list_messages(space_id, conv_id)
                except Exception as e:
                    logger.warning(f"Failed to list messages for conversation {conv_id}: {e}")
                    continue

                for message in messages:
                    if messages_processed >= max_messages:
                        break
                    otlp = self._process_genie_message(message, space, conv, now)
                    if otlp:
                        batched_resource_spans.extend(otlp.get("resourceSpans", []))
                    messages_processed += 1

        if batched_resource_spans:
            batched_otlp = {"resourceSpans": batched_resource_spans}
            self._forward_to_otel(batched_otlp)
            if self._webhook_urls:
                self._deliver_to_webhooks(batched_otlp)

        logger.info(
            f"Genie discovery complete: {len(self.genie_spaces)} spaces, "
            f"{messages_processed} messages processed, "
            f"{len(self.genie_traces)} traces created"
        )

        self._last_query_time = datetime.utcnow()
        self.notify_data_changed()

    def _resolve_user_email(self, user_id: str) -> str:
        """Resolve a Databricks numeric user ID to an email via SCIM API.

        Returns the email if found, otherwise returns the original user_id.
        Results are cached for the lifetime of the plugin instance.
        """
        if not user_id:
            return user_id

        # Return cached result
        if user_id in self._user_cache:
            cached = self._user_cache[user_id]
            return cached if cached else user_id

        try:
            data = self.workspace_client.api_client.do(
                "GET",
                f"/api/2.0/preview/scim/v2/Users/{user_id}",
            )
            # SCIM response: emails is an array of {value, type, primary}
            emails = data.get("emails", [])
            email = None
            for e in emails:
                if e.get("primary"):
                    email = e.get("value")
                    break
            if not email and emails:
                email = emails[0].get("value")

            # Fall back to userName (usually the email/UPN)
            if not email:
                email = data.get("userName")

            self._user_cache[user_id] = email
            return email if email else user_id
        except Exception as e:
            logger.debug(f"SCIM lookup failed for user_id={user_id}: {e}")
            self._user_cache[user_id] = None  # Cache the failure to avoid retrying
            return user_id

    def _process_genie_message(
        self,
        message: Dict,
        space: Dict,
        conversation: Dict,
        now: str,
    ):
        """
        Process a single Genie message: extract entities, build lineage, create trace.

        Args:
            message: Genie message dict from the API
            space: Genie space dict
            conversation: Genie conversation dict
            now: Current ISO timestamp string
        """
        space_id = space.get("space_id", "")
        space_title = space.get("title", space_id)
        conv_id = conversation.get("conversation_id") or conversation.get("id", "")
        msg_id = message.get("message_id") or message.get("id", "")
        user_prompt = message.get("content", "")

        # Extract SQL and text from attachments
        generated_sql: Optional[str] = None
        text_response: Optional[str] = None
        attachments = message.get("attachments") or []
        for att in attachments:
            if not generated_sql and att.get("query", {}).get("query"):
                generated_sql = att["query"]["query"]
            if not text_response and att.get("text", {}).get("content"):
                text_response = att["text"]["content"]

        # Register Genie agent (per-space)
        agent_name = f"Genie ({space_title})"
        agent = self.discovered_agents[agent_name]
        if not agent["first_seen"]:
            agent["first_seen"] = now
        agent["last_seen"] = now
        agent["metadata"]["discovery_source"] = "databricks/genie"
        agent["metadata"]["genie_space_id"] = space_id
        agent["metadata"]["genie_space_name"] = space_title
        user_email = self._resolve_user_email(str(message.get("user_id", "")))
        if user_email:
            agent["metadata"]["last_user"] = user_email
        agent["metadata"]["last_conversation_id"] = conv_id
        agent["metadata"]["message_count"] = agent["metadata"].get("message_count", 0) + 1

        # Register databricks-genie model with token estimation
        model_name = "databricks-genie"
        model = self.discovered_models[model_name]
        model["provider"] = "databricks"
        model["usage_count"] += 1
        model["agents_using"].add(agent_name)
        if not model["first_seen"]:
            model["first_seen"] = now
        model["last_seen"] = now

        # Token estimation (matching LangGuard: 4 chars/token text, 3.5 chars/token SQL)
        input_tokens = self._estimate_tokens(user_prompt, self.CHARS_PER_TOKEN_TEXT)
        output_tokens = (
            self._estimate_tokens(text_response, self.CHARS_PER_TOKEN_TEXT)
            + self._estimate_tokens(generated_sql, self.CHARS_PER_TOKEN_SQL)
        )
        self.model_token_usage[model_name]["input_tokens"] += input_tokens
        self.model_token_usage[model_name]["output_tokens"] += output_tokens

        # Agent → model lineage
        self.discovered_agents[agent_name]["models_used"].add(model_name)
        self._add_lineage(agent_name, "agent", model_name, "model", "uses")

        # Extract table references from SQL → downstream systems
        if generated_sql:
            tables = self._extract_tables_from_sql(generated_sql)
            for table_ref in tables:
                ds_id = table_ref.lower().replace(" ", "_")
                ds = self.discovered_downstream[ds_id]
                ds["name"] = table_ref
                ds["type"] = "database"
                ds["endpoint"] = None
                ds["tools_connecting"].add(agent_name)
                if not ds["first_seen"]:
                    ds["first_seen"] = now
                ds["last_seen"] = now
                ds["metadata"]["discovery_source"] = "databricks/genie"
                ds["metadata"]["genie_space_id"] = space_id
                self._add_lineage(agent_name, "agent", ds_id, "downstream", "queries")

        # Compute latency (timestamps are in milliseconds)
        created_ts = message.get("created_timestamp", 0)
        updated_ts = message.get("last_updated_timestamp")
        latency_ms = (updated_ts - created_ts) if updated_ts else None

        # Status mapping
        status_raw = message.get("status", "")
        if status_raw == "COMPLETED":
            status = "success"
        elif status_raw == "FAILED":
            status = "error"
        else:
            status = "pending"

        # Create trace dict
        trace_id = f"genie-{conv_id}-{msg_id}"
        trace_dict = {
            "id": trace_id,
            "name": f"Genie: {user_prompt[:50]}{'...' if len(user_prompt) > 50 else ''}",
            "timestamp": datetime.utcfromtimestamp(created_ts / 1000) if created_ts else datetime.utcnow(),
            "source": "databricks",
            "agent_name": agent_name,
            "agent_type": "sql-assistant",
            "latency_ms": latency_ms,
            "status": status,
            "model": model_name,
            "provider": "databricks",
            "user_id": user_email,
            "session_id": conv_id,
            "input": {"prompt": user_prompt},
            "output": {"generated_sql": generated_sql, "text_response": text_response},
            "raw_response": message,
            "metadata": {
                "genie_space_id": space_id,
                "genie_space_name": space_title,
                "genie_conversation_id": conv_id,
                "genie_message_id": msg_id,
                "genie_status": status_raw,
                "source": "databricks-genie",
            },
            "token_estimate": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "is_estimated": True,
            },
        }
        self.genie_traces.append(trace_dict)

        # Convert to OTLP and return for batched delivery by the caller.
        from open_cite.plugins.databricks_otlp_converter import genie_trace_to_otlp
        return genie_trace_to_otlp(trace_dict)

    # =========================================================================
    # MLflow Trace/Span Processing (via MLflow SDK)
    # =========================================================================

    def _process_trace_object(self, trace, experiment_name: str):
        """
        Process an MLflow Trace object (from mlflow.MlflowClient.search_traces)
        to discover agents, tools, models, and downstream systems.

        Only registers an agent when there is actual evidence of agentic behavior:
        explicit AGENT span, explicit agent_name metadata, or tool+model co-occurrence.

        Args:
            trace: mlflow.entities.trace.Trace object
            experiment_name: Name of the parent experiment
        """
        spans = trace.data.spans
        if not spans:
            return

        now = datetime.utcnow().isoformat()
        request_metadata = trace.info.request_metadata or {}

        # --- Scan spans for evidence of agentic behavior ---
        has_agent_span = False
        has_model_span = False
        has_tool_span = False
        agent_span_name = None

        for span in spans:
            span_type = (span.span_type or "").upper()
            if span_type == "AGENT":
                has_agent_span = True
                agent_span_name = span.name
            elif _SPAN_TYPE_CATEGORY.get(span_type) == "model":
                has_model_span = True
            elif _SPAN_TYPE_CATEGORY.get(span_type) == "tool":
                has_tool_span = True

        # --- Determine if this trace represents an agent ---
        has_explicit_agent_name = bool(request_metadata.get("agent_name"))
        is_agent = has_agent_span or has_explicit_agent_name or (has_model_span and has_tool_span)

        # --- Determine agent name (only used if is_agent) ---
        agent_name = None
        if is_agent:
            # Priority: metadata > AGENT span > root span > experiment
            agent_name = request_metadata.get("agent_name")
            if not agent_name and agent_span_name:
                agent_name = agent_span_name
            if not agent_name:
                for span in spans:
                    if span.parent_id is None:
                        agent_name = span.name
                        break
            if not agent_name:
                agent_name = experiment_name
            if not agent_name or agent_name == "Unknown Agent":
                agent_name = None

        # --- Register agent only with evidence ---
        if agent_name:
            agent = self.discovered_agents[agent_name]
            if not agent["first_seen"]:
                agent["first_seen"] = now
            agent["last_seen"] = now
            agent["metadata"]["discovery_source"] = "databricks/mlflow"

        # --- Process each span ---
        for span in spans:
            span_type = (span.span_type or "").upper()
            span_name = span.name or ""
            attributes = span.attributes or {}
            category = _SPAN_TYPE_CATEGORY.get(span_type)

            if category == "model":
                self._process_model_span(span_name, attributes, agent_name, now)

            elif category == "tool":
                # Build a span dict for _process_tool_span (needs 'outputs' for RETRIEVER)
                span_dict = {"outputs": span.outputs}
                self._process_tool_span(span_dict, span_type, span_name, attributes, agent_name, now)

            elif category == "agent" and span_name and span_name != agent_name:
                sub_agent = self.discovered_agents[span_name]
                if not sub_agent["first_seen"]:
                    sub_agent["first_seen"] = now
                sub_agent["last_seen"] = now
                sub_agent["metadata"]["discovery_source"] = "databricks/mlflow"
                if agent_name:
                    self._add_lineage(agent_name, "agent", span_name, "agent", "calls")

    def _process_model_span(
        self,
        span_name: str,
        attributes: Dict[str, Any],
        agent_name: Optional[str],
        now: str,
    ):
        """Process an LLM/CHAT_MODEL/EMBEDDING span to discover models."""
        model_name = (
            attributes.get("llm.model")
            or attributes.get("ai.model.name")
            or attributes.get("mlflow.chat.model")
            or attributes.get("model")
            or span_name
        )
        if not model_name:
            return

        provider = attributes.get("ai.model.provider") or "unknown"

        model = self.discovered_models[model_name]
        model["provider"] = provider
        model["usage_count"] += 1
        if not model["first_seen"]:
            model["first_seen"] = now
        model["last_seen"] = now

        # Track token usage
        input_tokens = attributes.get("llm.token_usage.input_tokens")
        output_tokens = attributes.get("llm.token_usage.output_tokens")
        if input_tokens is not None:
            try:
                self.model_token_usage[model_name]["input_tokens"] += int(input_tokens)
            except (ValueError, TypeError):
                pass
        if output_tokens is not None:
            try:
                self.model_token_usage[model_name]["output_tokens"] += int(output_tokens)
            except (ValueError, TypeError):
                pass

        # Create agent→model lineage
        if agent_name and agent_name != "Unknown Agent":
            model["agents_using"].add(agent_name)
            self.discovered_agents[agent_name]["models_used"].add(model_name)
            self._add_lineage(agent_name, "agent", model_name, "model", "uses")

    def _process_tool_span(
        self,
        span: Dict[str, Any],
        span_type: str,
        span_name: str,
        attributes: Dict[str, Any],
        agent_name: Optional[str],
        now: str,
    ):
        """Process a TOOL or RETRIEVER span to discover tools and downstream systems."""
        tool_name = (
            attributes.get("tool_name")
            or attributes.get("mlflow.tool.name")
            or span_name
        )
        if not tool_name:
            return

        tool = self.discovered_tools[tool_name]
        tool["metadata"]["discovery_source"] = "databricks/mlflow"
        tool["metadata"]["last_seen"] = now
        if "first_seen" not in tool["metadata"]:
            tool["metadata"]["first_seen"] = now

        # Create agent→tool lineage
        if agent_name and agent_name != "Unknown Agent":
            self._add_lineage(agent_name, "agent", tool_name, "tool", "uses")
            self.discovered_agents[agent_name]["tools_used"].add(tool_name)

        # Check for downstream system in tool attributes
        downstream = (
            attributes.get("tool.target")
            or attributes.get("service")
            or attributes.get("http.host")
            or attributes.get("db.system")
        )
        if downstream:
            ds_id = downstream.lower().replace(" ", "_").replace(":", "_")
            ds = self.discovered_downstream[ds_id]
            ds["name"] = downstream
            ds["type"] = "database" if attributes.get("db.system") else "api"
            ds["tools_connecting"].add(tool_name)
            if not ds["first_seen"]:
                ds["first_seen"] = now
            ds["last_seen"] = now
            ds["metadata"]["discovery_source"] = "databricks/mlflow"
            self._add_lineage(tool_name, "tool", ds_id, "downstream", "connects_to")

        # RETRIEVER: extract document sources as downstream systems
        if span_type == "RETRIEVER":
            outputs = span.get("outputs", [])
            if isinstance(outputs, list):
                seen_sources: Set[str] = set()
                for output in outputs:
                    if isinstance(output, dict):
                        output_metadata = output.get("metadata", {}) or {}
                        doc_source = (
                            output_metadata.get("source")
                            or output_metadata.get("doc_uri")
                            or output.get("doc_uri")
                        )
                        if doc_source and doc_source not in seen_sources:
                            seen_sources.add(doc_source)
                            ds_id = doc_source.lower().replace(" ", "_").replace(":", "_").replace("/", "_")
                            ds = self.discovered_downstream[ds_id]
                            ds["name"] = doc_source
                            ds["type"] = "document_store"
                            ds["tools_connecting"].add(tool_name)
                            if not ds["first_seen"]:
                                ds["first_seen"] = now
                            ds["last_seen"] = now
                            ds["metadata"]["discovery_source"] = "databricks/mlflow"
                            self._add_lineage(tool_name, "tool", ds_id, "downstream", "connects_to")

    # =========================================================================
    # MLflow Run Processing (via Databricks SDK)
    # =========================================================================

    def _process_mlflow_run(self, run, experiment_name: str = "unknown"):
        """
        Extract agents, tools, models, and downstream systems from an MLflow Run.

        Only registers an agent when there is actual evidence of agentic behavior:
        explicit agent_name tag, or both model and tool evidence in the run.

        Uses run tags, params, and metrics available via the Databricks SDK
        (no dependency on the mlflow package).

        Args:
            run: databricks.sdk.service.ml.Run object
            experiment_name: Name of the parent experiment
        """
        now = datetime.utcnow().isoformat()
        info = run.info or {}
        data = run.data or {}
        inputs = run.inputs or {}

        # Convert tags/params lists to dicts for easy lookup
        tags = {t.key: t.value for t in (data.tags or [])}
        params = {p.key: p.value for p in (data.params or [])}

        # --- Check for agent evidence ---
        explicit_agent_name = tags.get("agent_name")
        has_model = bool(
            tags.get("mlflow.model.name")
            or tags.get("model_name")
            or params.get("model_name")
            or params.get("model")
            or tags.get("mlflow.log-model.history")
            or getattr(inputs, "model_inputs", None)
        )
        has_tools = any(
            k.startswith("tool.") or k.startswith("mlflow.tool.")
            for k in tags
        )
        is_agent = bool(explicit_agent_name) or (has_model and has_tools)

        # --- Determine agent name (only used if is_agent) ---
        agent_name = None
        if is_agent:
            agent_name = (
                explicit_agent_name
                or tags.get("mlflow.runName")
                or getattr(info, "run_name", None)
                or experiment_name
            )
            if not agent_name or agent_name == "Unknown Agent":
                agent_name = None

        # Register agent
        if agent_name:
            agent = self.discovered_agents[agent_name]
            if not agent["first_seen"]:
                agent["first_seen"] = now
            agent["last_seen"] = now
            agent["metadata"]["discovery_source"] = "databricks/mlflow"
            agent["metadata"]["experiment_name"] = experiment_name

        # --- Discover model from tags/params ---
        model_name = (
            tags.get("mlflow.model.name")
            or tags.get("model_name")
            or params.get("model_name")
            or params.get("model")
        )
        # If a model was logged but no explicit name, use experiment name
        if not model_name and tags.get("mlflow.log-model.history"):
            model_name = experiment_name
        if model_name:
            provider = (
                tags.get("model_provider")
                or params.get("provider")
                or "unknown"
            )
            model = self.discovered_models[model_name]
            model["provider"] = provider
            model["usage_count"] += 1
            if not model["first_seen"]:
                model["first_seen"] = now
            model["last_seen"] = now

            if agent_name and agent_name != "Unknown Agent":
                model["agents_using"].add(agent_name)
                self.discovered_agents[agent_name]["models_used"].add(model_name)
                self._add_lineage(agent_name, "agent", model_name, "model", "uses")

        # --- Discover models from run.inputs.model_inputs ---
        for model_input in (getattr(inputs, "model_inputs", None) or []):
            mid = getattr(model_input, "model_id", None)
            if mid:
                m = self.discovered_models[mid]
                m["provider"] = "databricks"
                m["usage_count"] += 1
                if not m["first_seen"]:
                    m["first_seen"] = now
                m["last_seen"] = now
                if agent_name and agent_name != "Unknown Agent":
                    m["agents_using"].add(agent_name)
                    self.discovered_agents[agent_name]["models_used"].add(mid)
                    self._add_lineage(agent_name, "agent", mid, "model", "uses")

        # --- Discover tools from tags ---
        # Convention: tags like "tool.<name>" or params like "tools"
        for key, value in tags.items():
            if key.startswith("tool.") or key.startswith("mlflow.tool."):
                tool_name = key.split(".")[-1]
                tool = self.discovered_tools[tool_name]
                tool["metadata"]["discovery_source"] = "databricks/mlflow"
                tool["metadata"]["last_seen"] = now
                if "first_seen" not in tool["metadata"]:
                    tool["metadata"]["first_seen"] = now
                if agent_name and agent_name != "Unknown Agent":
                    self._add_lineage(agent_name, "agent", tool_name, "tool", "uses")
                    self.discovered_agents[agent_name]["tools_used"].add(tool_name)

        # --- Discover datasets as downstream systems ---
        for ds_input in (getattr(inputs, "dataset_inputs", None) or []):
            dataset = getattr(ds_input, "dataset", None)
            if not dataset:
                continue
            ds_name = getattr(dataset, "name", None)
            if not ds_name:
                continue
            ds_id = ds_name.lower().replace(" ", "_").replace(":", "_")
            ds = self.discovered_downstream[ds_id]
            ds["name"] = ds_name
            ds["type"] = "dataset"
            if agent_name and agent_name != "Unknown Agent":
                ds["tools_connecting"].add(agent_name)
            if not ds["first_seen"]:
                ds["first_seen"] = now
            ds["last_seen"] = now
            ds["metadata"]["discovery_source"] = "databricks/mlflow"
            self._add_lineage(agent_name or ds_name, "agent", ds_id, "downstream", "uses")

        # --- Track token usage from metrics ---
        for metric in (data.metrics or []):
            key = metric.key or ""
            val = metric.value
            if val is None:
                continue
            if "input_token" in key and model_name:
                self.model_token_usage[model_name]["input_tokens"] += int(val)
            elif "output_token" in key and model_name:
                self.model_token_usage[model_name]["output_tokens"] += int(val)

    # =========================================================================
    # MCP Server & Tool Discovery
    # =========================================================================

    def discover_mcp_servers(self):
        """
        Discover MCP servers from Databricks workspace sources.

        Sources:
        1. Vector Search Indexes - each unique catalog.schema → one MCP server
        2. Databricks Apps - each app is a potential MCP server

        For each discovered server, fetches tools via JSON-RPC tools/list.
        Adapted from webapp/server/python/mlflow_client.py list_mcp_servers().
        """
        logger.info("Starting MCP server discovery")
        now = datetime.utcnow().isoformat()
        host = self.host.rstrip("/")
        servers_found: List[Dict[str, Any]] = []

        # --- Source 1: Vector Search Indexes ---
        try:
            if hasattr(self.workspace_client, "vector_search_endpoints"):
                seen_urls: Set[str] = set()
                for ep in self.workspace_client.vector_search_endpoints.list_endpoints():
                    ep_name = getattr(ep, "name", None)
                    if not ep_name:
                        continue
                    try:
                        for idx in self.workspace_client.vector_search_indexes.list_indexes(
                            endpoint_name=ep_name
                        ):
                            parts = idx.name.split(".")
                            if len(parts) >= 3:
                                catalog, schema = parts[0], parts[1]
                                server_url = f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}"

                                if server_url in seen_urls:
                                    continue
                                seen_urls.add(server_url)

                                servers_found.append({
                                    "name": f"Vector Search ({schema})",
                                    "server_type": "vector-search",
                                    "url": server_url,
                                    "metadata": {
                                        "catalog": catalog,
                                        "schema": schema,
                                        "source_entity": idx.name,
                                        "endpoint": ep_name,
                                    },
                                })
                    except Exception as e:
                        logger.debug(f"Failed to list indexes for endpoint {ep_name}: {e}")
            else:
                logger.debug("Vector Search SDK not available")
        except Exception as e:
            logger.debug(f"Failed to list vector search endpoints: {e}")

        # --- Source 2: Databricks Apps ---
        try:
            if hasattr(self.workspace_client, "apps"):
                for app in self.workspace_client.apps.list():
                    app_url = getattr(app, "url", None)
                    servers_found.append({
                        "name": app.name,
                        "server_type": "app",
                        "url": app_url,
                        "metadata": {
                            "source_entity": app.name,
                        },
                    })
        except Exception as e:
            logger.debug(f"Failed to list Databricks apps: {e}")

        if not servers_found:
            logger.info("No MCP servers discovered")
            return

        # --- Fetch tools in parallel ---
        server_tools: Dict[int, List[Dict]] = {}
        servers_with_urls = [
            (i, s) for i, s in enumerate(servers_found) if s.get("url")
        ]

        if servers_with_urls:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_idx = {
                    executor.submit(self._fetch_mcp_tools, s["url"]): i
                    for i, s in servers_with_urls
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        server_tools[idx] = future.result()
                    except Exception as e:
                        logger.debug(f"Tool fetch failed for server {idx}: {e}")
                        server_tools[idx] = []

        # --- Register servers and tools ---
        for i, server_info in enumerate(servers_found):
            server_url = server_info.get("url") or server_info["name"]
            server_id = f"mcp:{server_url}"
            tools = server_tools.get(i, [])

            # Upsert server
            if server_id in self.mcp_servers:
                existing = self.mcp_servers[server_id]
                existing["last_seen"] = now
                existing["tools_count"] = len(tools)
            else:
                self.mcp_servers[server_id] = {
                    "id": server_id,
                    "name": server_info["name"],
                    "server_type": server_info["server_type"],
                    "url": server_info.get("url"),
                    "transport": "http",
                    "discovery_source": "databricks",
                    "first_seen": now,
                    "last_seen": now,
                    "tools_count": len(tools),
                    "metadata": server_info.get("metadata", {}),
                }

            # Register each tool
            for tool in tools:
                tool_name = tool.get("name", "")
                if not tool_name:
                    continue
                tool_id = f"{server_id}-{tool_name}"

                if tool_id in self.mcp_tools:
                    self.mcp_tools[tool_id]["last_seen"] = now
                else:
                    self.mcp_tools[tool_id] = {
                        "id": tool_id,
                        "name": tool_name,
                        "server_id": server_id,
                        "description": tool.get("description"),
                        "input_schema": tool.get("inputSchema"),
                        "discovery_source": "databricks",
                        "first_seen": now,
                        "last_seen": now,
                    }

                # Lineage: mcp_server --contains--> mcp_tool
                self._add_lineage(server_id, "mcp_server", tool_id, "mcp_tool", "contains")

        logger.info(
            f"MCP discovery complete: {len(self.mcp_servers)} servers, "
            f"{len(self.mcp_tools)} tools"
        )
        self.notify_data_changed()

    def _fetch_mcp_tools(self, server_url: str) -> List[Dict]:
        """Fetch tools from an MCP server via JSON-RPC tools/list."""
        try:
            data = self.workspace_client.api_client.do(
                "POST",
                url=server_url,
                body={"jsonrpc": "2.0", "method": "tools/list", "id": 1, "params": {}},
            )
            return (data or {}).get("result", {}).get("tools", [])
        except Exception as e:
            logger.debug(f"Failed to fetch tools from {server_url}: {e}")
            return []

    def _list_mcp_servers(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered MCP servers."""
        servers = []
        for server_id, server_info in self.mcp_servers.items():
            servers.append({
                "id": server_info["id"],
                "name": server_info["name"],
                "type": "mcp_server",
                "server_type": server_info.get("server_type"),
                "url": server_info.get("url"),
                "transport": server_info.get("transport", "http"),
                "discovery_source": server_info.get("discovery_source"),
                "first_seen": server_info.get("first_seen"),
                "last_seen": server_info.get("last_seen"),
                "tools_count": server_info.get("tools_count", 0),
                "metadata": server_info.get("metadata", {}),
            })
        return servers

    def _list_mcp_tools(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered MCP tools, optionally filtered by server_id."""
        server_id = kwargs.get("server_id")
        tools = []
        for tool_id, tool_info in self.mcp_tools.items():
            if server_id and tool_info.get("server_id") != server_id:
                continue
            tools.append({
                "id": tool_info["id"],
                "name": tool_info["name"],
                "type": "mcp_tool",
                "server_id": tool_info["server_id"],
                "description": tool_info.get("description"),
                "input_schema": tool_info.get("input_schema"),
                "discovery_source": tool_info.get("discovery_source"),
                "first_seen": tool_info.get("first_seen"),
                "last_seen": tool_info.get("last_seen"),
            })
        return tools

    # =========================================================================
    # Lineage
    # =========================================================================

    def _add_lineage(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        relationship_type: str,
    ):
        """Add or update a lineage relationship (same pattern as OTLP plugin)."""
        key = f"{source_id}:{target_id}:{relationship_type}"
        now = datetime.utcnow().isoformat()

        if key in self.lineage:
            self.lineage[key]["weight"] += 1
            self.lineage[key]["last_seen"] = now
        else:
            self.lineage[key] = {
                "source_id": source_id,
                "source_type": source_type,
                "target_id": target_id,
                "target_type": target_type,
                "relationship_type": relationship_type,
                "weight": 1,
                "first_seen": now,
                "last_seen": now,
            }

    def list_lineage(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List lineage relationships discovered from MLflow traces.

        Args:
            source_id: Optional filter to only return relationships involving this entity

        Returns:
            List of lineage relationship dicts
        """
        if source_id:
            return [
                rel for rel in self.lineage.values()
                if rel["source_id"] == source_id or rel["target_id"] == source_id
            ]
        return list(self.lineage.values())

    # =========================================================================
    # Asset Listing for Discovered Entities
    # =========================================================================

    def _list_discovered_tools(self) -> List[Dict[str, Any]]:
        """List tools discovered from MLflow traces."""
        tools = []
        for tool_name, tool_data in self.discovered_tools.items():
            metadata = tool_data.get("metadata", {})
            tools.append({
                "id": tool_name,
                "name": tool_name,
                "type": "tool",
                "discovery_source": metadata.get("discovery_source", "databricks"),
                "models": list(tool_data["models"]),
                "trace_count": len(tool_data["traces"]),
                "first_seen": metadata.get("first_seen"),
                "last_seen": metadata.get("last_seen"),
                "metadata": metadata,
            })
        return tools

    def _list_discovered_agents(self) -> List[Dict[str, Any]]:
        """List agents discovered from MLflow traces."""
        agents = []
        for agent_name, agent_data in self.discovered_agents.items():
            agents.append({
                "id": agent_name,
                "name": agent_name,
                "type": "agent",
                "discovery_source": agent_data.get("metadata", {}).get("discovery_source", "databricks"),
                "tools_used": list(agent_data["tools_used"]),
                "models_used": list(agent_data["models_used"]),
                "first_seen": agent_data["first_seen"],
                "last_seen": agent_data["last_seen"],
                "metadata": agent_data["metadata"],
            })
        return agents

    def _list_discovered_models(self) -> List[Dict[str, Any]]:
        """List models discovered from MLflow traces (not UC registered models)."""
        models = []
        for model_name, model_data in self.discovered_models.items():
            token_data = self.model_token_usage.get(model_name, {})
            models.append({
                "id": f"trace:{model_name}",
                "name": model_name,
                "type": "model",
                "discovery_source": "databricks",
                "source": "mlflow_trace",
                "provider": model_data["provider"],
                "agents_using": list(model_data["agents_using"]),
                "tools_using": list(model_data["tools_using"]),
                "usage_count": model_data["usage_count"],
                "total_input_tokens": token_data.get("input_tokens", 0),
                "total_output_tokens": token_data.get("output_tokens", 0),
                "first_seen": model_data["first_seen"],
                "last_seen": model_data["last_seen"],
            })
        return models

    def _list_discovered_downstream(self) -> List[Dict[str, Any]]:
        """List downstream systems discovered from MLflow traces."""
        systems = []
        for system_id, system_data in self.discovered_downstream.items():
            systems.append({
                "id": system_id,
                "name": system_data["name"],
                "type": system_data["type"],
                "discovery_source": system_data.get("metadata", {}).get("discovery_source", "databricks"),
                "endpoint": system_data["endpoint"],
                "tools_connecting": list(system_data["tools_connecting"]),
                "first_seen": system_data["first_seen"],
                "last_seen": system_data["last_seen"],
                "metadata": system_data["metadata"],
            })
        return systems
