"""
LiteLLM MCP Proxy discovery plugin for Open-CITE.

This plugin discovers MCP tools and identity-to-tool authorization mappings
from a remote LiteLLM proxy instance:
- MCP tools available through the proxy
- MCP servers (grouped from tool list by server_name)
- Teams with MCP access permissions
- Users with team memberships

Authentication is via LiteLLM API key (admin or regular).
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)


class LiteLLMPlugin(BaseDiscoveryPlugin):
    """
    LiteLLM MCP Proxy discovery plugin.

    Discovers MCP tools, servers, teams, and users from a LiteLLM proxy
    and builds lineage relationships showing the authorization graph.
    """

    plugin_type = "litellm"

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        api_key: Optional[str] = None,
        poll_interval: int = 60,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
        auto_poll: bool = True,
    ):
        super().__init__(instance_id=instance_id, display_name=display_name)

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.poll_interval = max(10, int(poll_interval))
        self._auto_poll = auto_poll

        # Caches
        self._mcp_tools_cache: List[Dict[str, Any]] = []
        self._mcp_servers_cache: List[Dict[str, Any]] = []
        self._teams_cache: List[Dict[str, Any]] = []
        self._users_cache: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Persistence dicts (auto-persisted by _save_current_state in app.py)
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}
        self.lineage: Dict[str, Dict[str, Any]] = {}

        # Polling thread state
        self._poll_stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Plugin metadata & factory
    # ------------------------------------------------------------------

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        return {
            "name": "LiteLLM MCP Proxy",
            "description": (
                "Discovers MCP tools and identity-to-tool authorization "
                "mappings from a LiteLLM proxy instance"
            ),
            "required_fields": {
                "base_url": {
                    "label": "LiteLLM Proxy URL",
                    "default": "http://localhost:4000",
                    "required": True,
                },
                "api_key": {
                    "label": "API Key",
                    "default": "",
                    "required": True,
                    "type": "password",
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
                "LITELLM_BASE_URL",
                "LITELLM_API_KEY",
            ],
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        import os

        base_url = (
            config.get("base_url")
            or os.environ.get("LITELLM_BASE_URL")
            or "http://localhost:4000"
        )
        api_key = (
            config.get("api_key")
            or os.environ.get("LITELLM_API_KEY")
        )
        poll_interval = int(config.get("poll_interval") or 60)
        auto_poll = config.get("auto_poll", True)
        if isinstance(auto_poll, str):
            auto_poll = auto_poll.lower() not in ("false", "0", "no")

        return cls(
            base_url=base_url,
            api_key=api_key,
            poll_interval=poll_interval,
            instance_id=instance_id,
            display_name=display_name,
            auto_poll=bool(auto_poll),
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @property
    def supported_asset_types(self) -> Set[str]:
        return {"mcp_server", "mcp_tool", "identity"}

    def get_identification_attributes(self) -> List[str]:
        return [
            "litellm.base_url",
            "litellm.server_name",
            "litellm.team_id",
            "litellm.user_id",
        ]

    def verify_connection(self) -> Dict[str, Any]:
        """Verify connection by listing MCP tools."""
        try:
            resp = requests.get(
                f"{self.base_url}/mcp-rest/tools/list",
                headers=self._auth_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                tool_count = len(data) if isinstance(data, list) else 0
                return {
                    "success": True,
                    "base_url": self.base_url,
                    "message": "Successfully connected to LiteLLM proxy",
                    "tool_count_hint": tool_count,
                }
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                "message": "Failed to connect to LiteLLM proxy",
            }
        except requests.ConnectionError as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to connect to LiteLLM proxy",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to connect to LiteLLM proxy",
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        if asset_type not in self.supported_asset_types:
            raise ValueError(
                f"Unsupported asset type: {asset_type}. "
                f"Supported: {', '.join(sorted(self.supported_asset_types))}"
            )

        with self._lock:
            if asset_type == "mcp_server":
                return list(self._mcp_servers_cache)
            elif asset_type == "mcp_tool":
                return list(self._mcp_tools_cache)
            elif asset_type == "identity":
                return self._list_identities()
            return []

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": "****" if self.api_key else None,
            "poll_interval": self.poll_interval,
        }

    def export_assets(self) -> Dict[str, Any]:
        return {
            "mcp_servers": self.list_assets("mcp_server"),
            "mcp_tools": self.list_assets("mcp_tool"),
            "identities": self.list_assets("identity"),
        }

    # ------------------------------------------------------------------
    # Lifecycle — background polling
    # ------------------------------------------------------------------

    def start(self):
        """Start background polling."""
        super().start()
        # One-time initial discovery
        threading.Thread(
            target=self._discover_all, daemon=True,
            name=f"litellm-init-{self.instance_id[:8]}",
        ).start()
        if self._auto_poll:
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop, daemon=True,
                name=f"litellm-poll-{self.instance_id[:8]}",
            )
            self._poll_thread.start()
            logger.info(
                "LiteLLM polling started (interval=%ds)",
                self.poll_interval,
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
            self._poll_stop.wait(timeout=self.poll_interval)
            if self._poll_stop.is_set():
                break
            try:
                self._discover_all()
            except Exception as e:
                logger.warning("LiteLLM poll error: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        """Build auth headers for LiteLLM API requests."""
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _list_identities(self) -> List[Dict[str, Any]]:
        """Return teams and users as identity assets."""
        identities: List[Dict[str, Any]] = []
        for team in self._teams_cache:
            identities.append(team)
        for user in self._users_cache:
            identities.append(user)
        return identities

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_all(self):
        """Run full discovery cycle: servers → tools → teams → users → lineage."""
        self._fetch_mcp_servers()
        self._fetch_mcp_tools()
        self._fetch_teams()
        self._fetch_users()
        self._build_lineage()
        self._sync_mcp_dicts()
        self.notify_data_changed()

    @staticmethod
    def _extract_server_name(item: Dict[str, Any]) -> str:
        """Extract MCP server name from a tool record.

        The LiteLLM tool list nests ``server_name`` inside ``mcp_info``. Older
        responses placed it at the top level. Fall back to ``unknown`` so we
        still produce stable ids.
        """
        mcp_info = item.get("mcp_info") or {}
        if isinstance(mcp_info, dict) and mcp_info.get("server_name"):
            return mcp_info["server_name"]
        return item.get("server_name") or item.get("server") or "unknown"

    def _fetch_mcp_tools(self):
        """Fetch MCP tools from GET /mcp-rest/tools/list."""
        try:
            resp = requests.get(
                f"{self.base_url}/mcp-rest/tools/list",
                headers=self._auth_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle multiple response formats:
            #   - list of tools (legacy)
            #   - {"tools": [...]} or {"data": [...]}
            #   - {"result": {"tools": [...]}} (MCP standard)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if isinstance(data.get("result"), dict):
                    items = data["result"].get("tools", []) or []
                else:
                    items = data.get("tools") or data.get("data") or []
                if not isinstance(items, list):
                    logger.warning(
                        "LiteLLM /mcp-rest/tools/list returned dict with non-list payload: %s",
                        list(data.keys()),
                    )
                    return
            else:
                logger.warning(
                    "LiteLLM /mcp-rest/tools/list returned unexpected type: %s",
                    type(data),
                )
                return

            now = datetime.now(timezone.utc).isoformat()
            tools = []
            for item in items:
                tool_name = item.get("name", "")
                server_name = self._extract_server_name(item)
                tool_id = f"litellm:{server_name}:{tool_name}"
                tools.append({
                    "id": tool_id,
                    "name": tool_name,
                    "server_id": f"litellm:{server_name}",
                    "server_name": server_name,
                    "discovery_source": self.instance_id,
                    "description": item.get("description", ""),
                    "schema": item.get("inputSchema") or item.get("input_schema"),
                    "metadata": {
                        "litellm.server_name": server_name,
                    },
                    "last_seen": now,
                })

            with self._lock:
                self._mcp_tools_cache = tools
            logger.debug("LiteLLM: discovered %d MCP tools", len(tools))

        except Exception as e:
            logger.warning("LiteLLM: failed to fetch MCP tools, using stale cache: %s", e)

    def _fetch_mcp_servers(self):
        """Fetch MCP servers from GET /v1/mcp/server.

        Falls back to deriving servers from the tools cache if the dedicated
        endpoint is unavailable (older LiteLLM versions).
        """
        servers_from_api: List[Dict[str, Any]] = []
        api_ok = False
        try:
            resp = requests.get(
                f"{self.base_url}/v1/mcp/server",
                headers=self._auth_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    servers_from_api = data
                elif isinstance(data, dict):
                    servers_from_api = (
                        data.get("servers") or data.get("data") or []
                    )
                api_ok = True
            else:
                logger.warning(
                    "LiteLLM /v1/mcp/server returned HTTP %d, falling back to tool-derived servers",
                    resp.status_code,
                )
        except Exception as e:
            logger.warning(
                "LiteLLM: failed to fetch MCP servers from /v1/mcp/server: %s", e
            )

        now = datetime.now(timezone.utc).isoformat()
        servers_map: Dict[str, Dict[str, Any]] = {}

        if api_ok:
            for item in servers_from_api:
                server_name = (
                    item.get("server_name")
                    or item.get("alias")
                    or item.get("server_id")
                    or "unknown"
                )
                server_id = f"litellm:{server_name}"
                transport = item.get("transport") or "http"
                endpoint = (
                    item.get("url")
                    or (
                        f"{item.get('command', '')} {' '.join(item.get('args') or [])}".strip()
                        if item.get("command")
                        else None
                    )
                    or f"{self.base_url}/mcp-rest"
                )
                # Permission fields from LiteLLM
                allowed_teams = item.get("teams") or []
                access_groups = item.get("mcp_access_groups") or []
                allowed_tools = item.get("allowed_tools") or []
                allow_all_keys = bool(item.get("allow_all_keys"))

                servers_map[server_id] = {
                    "id": server_id,
                    "name": server_name,
                    "discovery_source": self.instance_id,
                    "transport": transport,
                    "endpoint": endpoint,
                    "description": item.get("description", ""),
                    "tools_provided": [],
                    "tools_count": 0,
                    "metadata": {
                        "litellm.base_url": self.base_url,
                        "litellm.server_id": item.get("server_id"),
                        "litellm.alias": item.get("alias"),
                        "litellm.auth_type": item.get("auth_type"),
                        "litellm.status": item.get("status"),
                        "litellm.allowed_teams": allowed_teams,
                        "litellm.mcp_access_groups": access_groups,
                        "litellm.allowed_tools": allowed_tools,
                        "litellm.allow_all_keys": allow_all_keys,
                        "litellm.available_on_public_internet": item.get(
                            "available_on_public_internet"
                        ),
                    },
                    "last_seen": now,
                }

        # Always merge in tool-derived servers (covers tools whose server is
        # not yet listed by /v1/mcp/server, and provides the legacy fallback)
        with self._lock:
            tools = list(self._mcp_tools_cache)

        for tool in tools:
            server_name = tool.get("server_name", "unknown")
            server_id = f"litellm:{server_name}"
            if server_id not in servers_map:
                servers_map[server_id] = {
                    "id": server_id,
                    "name": server_name,
                    "discovery_source": self.instance_id,
                    "transport": "http",
                    "endpoint": f"{self.base_url}/mcp-rest",
                    "tools_provided": [],
                    "tools_count": 0,
                    "metadata": {
                        "litellm.base_url": self.base_url,
                    },
                    "last_seen": now,
                }
            if tool["id"] not in servers_map[server_id]["tools_provided"]:
                servers_map[server_id]["tools_provided"].append(tool["id"])
            servers_map[server_id]["tools_count"] = len(
                servers_map[server_id]["tools_provided"]
            )

        servers = list(servers_map.values())
        with self._lock:
            self._mcp_servers_cache = servers
        logger.debug("LiteLLM: discovered %d MCP servers", len(servers))

    def _fetch_teams(self):
        """Fetch teams from GET /team/list."""
        try:
            resp = requests.get(
                f"{self.base_url}/team/list",
                headers=self._auth_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle both list and dict-with-list responses
            if isinstance(data, dict):
                items = data.get("teams", data.get("data", []))
            elif isinstance(data, list):
                items = data
            else:
                logger.warning("LiteLLM /team/list returned unexpected type: %s", type(data))
                return

            now = datetime.now(timezone.utc).isoformat()
            teams = []
            for item in items:
                team_id = item.get("team_id", "")
                team_alias = item.get("team_alias", "") or team_id

                # Extract MCP permissions
                obj_perm = item.get("object_permission", {}) or {}
                mcp_servers = obj_perm.get("mcp_servers", []) or []
                mcp_tools = obj_perm.get("mcp_tools", []) or []

                # Extract member user IDs
                members = []
                for m in (item.get("members_with_roles", []) or []):
                    uid = m.get("user_id", "")
                    if uid:
                        members.append(uid)

                teams.append({
                    "id": f"litellm:team:{team_id}",
                    "name": team_alias,
                    "type": "identity",
                    "identity_type": "team",
                    "discovery_source": self.instance_id,
                    "metadata": {
                        "litellm.team_id": team_id,
                        "litellm.mcp_servers": mcp_servers,
                        "litellm.mcp_tools": mcp_tools,
                        "litellm.members": members,
                    },
                    "last_seen": now,
                })

            with self._lock:
                self._teams_cache = teams
            logger.debug("LiteLLM: discovered %d teams", len(teams))

        except Exception as e:
            logger.warning("LiteLLM: failed to fetch teams, using stale cache: %s", e)

    def _fetch_users(self):
        """Fetch users from GET /user/list."""
        try:
            resp = requests.get(
                f"{self.base_url}/user/list",
                headers=self._auth_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle both list and dict-with-list responses
            if isinstance(data, dict):
                items = data.get("users", data.get("data", []))
            elif isinstance(data, list):
                items = data
            else:
                logger.warning("LiteLLM /user/list returned unexpected type: %s", type(data))
                return

            # Build team_id → mcp_servers mapping from teams cache
            with self._lock:
                teams = list(self._teams_cache)

            team_mcp_map: Dict[str, List[str]] = {}
            for team in teams:
                tid = team.get("metadata", {}).get("litellm.team_id", "")
                srvs = team.get("metadata", {}).get("litellm.mcp_servers", [])
                if tid:
                    team_mcp_map[tid] = srvs

            now = datetime.now(timezone.utc).isoformat()
            users = []
            for item in items:
                user_id = item.get("user_id", "")
                user_email = item.get("user_email", "") or ""
                user_name = user_email or user_id

                # Team memberships
                user_teams = item.get("teams", []) or []

                # Compute effective MCP access from teams
                effective_mcp_servers: List[str] = []
                for tid in user_teams:
                    for srv in team_mcp_map.get(tid, []):
                        if srv not in effective_mcp_servers:
                            effective_mcp_servers.append(srv)

                users.append({
                    "id": f"litellm:user:{user_id}",
                    "name": user_name,
                    "type": "identity",
                    "identity_type": "user",
                    "discovery_source": self.instance_id,
                    "metadata": {
                        "litellm.user_id": user_id,
                        "litellm.user_email": user_email,
                        "litellm.teams": user_teams,
                        "litellm.effective_mcp_servers": effective_mcp_servers,
                    },
                    "last_seen": now,
                })

            with self._lock:
                self._users_cache = users
            logger.debug("LiteLLM: discovered %d users", len(users))

        except Exception as e:
            logger.warning("LiteLLM: failed to fetch users, using stale cache: %s", e)

    def _build_lineage(self):
        """Build lineage relationships from cached data."""
        with self._lock:
            servers = list(self._mcp_servers_cache)
            tools = list(self._mcp_tools_cache)
            teams = list(self._teams_cache)
            users = list(self._users_cache)

        new_lineage: Dict[str, Dict[str, Any]] = {}

        # server → tool (contains)
        for tool in tools:
            server_id = tool.get("server_id", "")
            if server_id:
                self._add_lineage_to(
                    new_lineage, server_id, "mcp_server",
                    tool["id"], "mcp_tool", "contains",
                )

        # team → server (has_access_to) — from team object_permission.mcp_servers
        for team in teams:
            team_id = team["id"]
            meta = team.get("metadata", {})
            for srv_name in meta.get("litellm.mcp_servers", []):
                server_id = f"litellm:{srv_name}"
                self._add_lineage_to(
                    new_lineage, team_id, "identity",
                    server_id, "mcp_server", "has_access_to",
                )

        # team → server (has_access_to) — from server-side `teams` allowlist
        for server in servers:
            server_id = server["id"]
            allowed_team_ids = (server.get("metadata", {}) or {}).get(
                "litellm.allowed_teams", []
            )
            for tid in allowed_team_ids:
                team_asset_id = f"litellm:team:{tid}"
                self._add_lineage_to(
                    new_lineage, team_asset_id, "identity",
                    server_id, "mcp_server", "has_access_to",
                )

            # team → tool (has_access_to, explicit allowlist)
            for tool_name in meta.get("litellm.mcp_tools", []):
                # Find the matching tool ID
                for tool in tools:
                    if tool.get("name") == tool_name:
                        self._add_lineage_to(
                            new_lineage, team_id, "identity",
                            tool["id"], "mcp_tool", "has_access_to",
                        )

        # user → team (member_of)
        for user in users:
            user_id = user["id"]
            meta = user.get("metadata", {})
            for tid in meta.get("litellm.teams", []):
                team_asset_id = f"litellm:team:{tid}"
                self._add_lineage_to(
                    new_lineage, user_id, "identity",
                    team_asset_id, "identity", "member_of",
                )

            # user → server (has_access_to, derived from teams)
            for srv_name in meta.get("litellm.effective_mcp_servers", []):
                server_id = f"litellm:{srv_name}"
                self._add_lineage_to(
                    new_lineage, user_id, "identity",
                    server_id, "mcp_server", "has_access_to",
                )

        self.lineage = new_lineage

    @staticmethod
    def _add_lineage_to(
        lineage_dict: Dict[str, Dict[str, Any]],
        source_id: str, source_type: str,
        target_id: str, target_type: str,
        relationship_type: str,
    ):
        """Add a lineage relationship to the given dict."""
        key = f"{source_id}:{target_id}:{relationship_type}"
        now = datetime.now(timezone.utc).isoformat()

        if key in lineage_dict:
            lineage_dict[key]["weight"] += 1
            lineage_dict[key]["last_seen"] = now
        else:
            lineage_dict[key] = {
                "source_id": source_id,
                "source_type": source_type,
                "target_id": target_id,
                "target_type": target_type,
                "relationship_type": relationship_type,
                "weight": 1,
                "first_seen": now,
                "last_seen": now,
            }

    def _sync_mcp_dicts(self):
        """Populate mcp_servers/mcp_tools dicts for auto-persistence."""
        with self._lock:
            servers = list(self._mcp_servers_cache)
            tools = list(self._mcp_tools_cache)

        new_servers: Dict[str, Dict[str, Any]] = {}
        for srv in servers:
            new_servers[srv["id"]] = {
                "name": srv["name"],
                "transport": srv.get("transport", "http"),
                "endpoint": srv.get("endpoint"),
                "description": srv.get("description", ""),
                "tools_provided": srv.get("tools_provided", []),
                "metadata": srv.get("metadata", {}),
            }

        new_tools: Dict[str, Dict[str, Any]] = {}
        for tool in tools:
            new_tools[tool["id"]] = {
                "name": tool["name"],
                "server_id": tool.get("server_id", ""),
                "description": tool.get("description", ""),
                "schema": tool.get("schema"),
                "metadata": tool.get("metadata", {}),
            }

        self.mcp_servers = new_servers
        self.mcp_tools = new_tools
