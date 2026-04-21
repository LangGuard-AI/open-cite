"""
Unit tests for the LiteLLM MCP Proxy discovery plugin.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from open_cite.plugins.litellm import LiteLLMPlugin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plugin():
    """Plugin configured with a test API key."""
    return LiteLLMPlugin(
        base_url="http://litellm.test:4000",
        api_key="sk-test-key",
        instance_id="litellm-test",
        display_name="LiteLLM Test",
        auto_poll=False,
    )


@pytest.fixture
def sample_tools():
    return [
        {
            "name": "read_file",
            "server_name": "filesystem",
            "description": "Read contents of a file",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "server_name": "filesystem",
            "description": "Write contents to a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "search",
            "server_name": "brave-search",
            "description": "Search the web",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ]


@pytest.fixture
def sample_teams():
    return [
        {
            "team_id": "team-alpha",
            "team_alias": "Alpha Team",
            "object_permission": {
                "mcp_servers": ["filesystem", "brave-search"],
                "mcp_tools": ["read_file"],
            },
            "members_with_roles": [
                {"user_id": "user-001", "role": "admin"},
                {"user_id": "user-002", "role": "user"},
            ],
        },
        {
            "team_id": "team-beta",
            "team_alias": "Beta Team",
            "object_permission": {
                "mcp_servers": ["brave-search"],
            },
            "members_with_roles": [
                {"user_id": "user-002", "role": "user"},
            ],
        },
    ]


@pytest.fixture
def sample_users():
    return [
        {
            "user_id": "user-001",
            "user_email": "alice@example.com",
            "teams": ["team-alpha"],
        },
        {
            "user_id": "user-002",
            "user_email": "bob@example.com",
            "teams": ["team-alpha", "team-beta"],
        },
    ]


@pytest.fixture
def sample_servers():
    """Simulated /v1/mcp/server response for the LiteLLM proxy."""
    return [
        {
            "server_id": "srv-fs",
            "server_name": "filesystem",
            "alias": "filesystem",
            "description": "Local filesystem MCP server",
            "transport": "stdio",
            "url": None,
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "teams": ["team-alpha"],
            "mcp_access_groups": [],
            "allowed_tools": [],
            "allow_all_keys": True,
            "available_on_public_internet": False,
        },
        {
            "server_id": "srv-search",
            "server_name": "brave-search",
            "alias": "brave-search",
            "description": "Brave web search MCP server",
            "transport": "http",
            "url": "https://brave.example.com/mcp",
            "teams": [],
            "mcp_access_groups": ["Engineering"],
            "allowed_tools": ["search"],
            "allow_all_keys": False,
        },
    ]


def _mock_resp(payload, status=200):
    """Build a MagicMock that mimics a successful requests.Response."""
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status = MagicMock()
    return r


# ---------------------------------------------------------------------------
# TestPluginMetadata
# ---------------------------------------------------------------------------

class TestPluginMetadata:
    def test_plugin_type(self):
        assert LiteLLMPlugin.plugin_type == "litellm"

    def test_supported_asset_types(self, plugin):
        assert plugin.supported_asset_types == {"mcp_server", "mcp_tool", "identity"}

    def test_identification_attributes(self, plugin):
        attrs = plugin.get_identification_attributes()
        assert "litellm.base_url" in attrs
        assert "litellm.server_name" in attrs
        assert "litellm.team_id" in attrs
        assert "litellm.user_id" in attrs

    def test_metadata_fields(self):
        meta = LiteLLMPlugin.plugin_metadata()
        assert meta["name"] == "LiteLLM MCP Proxy"
        assert "required_fields" in meta
        assert "base_url" in meta["required_fields"]
        assert "api_key" in meta["required_fields"]
        assert "poll_interval" in meta["required_fields"]
        assert meta["required_fields"]["api_key"]["type"] == "password"

    def test_config_masking(self, plugin):
        config = plugin.get_config()
        assert config["api_key"] == "****"
        assert config["base_url"] == "http://litellm.test:4000"
        assert config["poll_interval"] == 60

    def test_config_masking_no_key(self):
        p = LiteLLMPlugin(base_url="http://localhost:4000")
        config = p.get_config()
        assert config["api_key"] is None

    def test_env_vars(self):
        meta = LiteLLMPlugin.plugin_metadata()
        assert "LITELLM_BASE_URL" in meta["env_vars"]
        assert "LITELLM_API_KEY" in meta["env_vars"]


# ---------------------------------------------------------------------------
# TestFromConfig
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_factory_basic(self):
        p = LiteLLMPlugin.from_config(
            config={
                "base_url": "http://my-proxy:4000",
                "api_key": "sk-admin",
                "poll_interval": "120",
            },
            instance_id="my-litellm",
            display_name="My LiteLLM",
        )
        assert p.base_url == "http://my-proxy:4000"
        assert p.api_key == "sk-admin"
        assert p.poll_interval == 120
        assert p.instance_id == "my-litellm"
        assert p.display_name == "My LiteLLM"

    def test_factory_defaults(self):
        p = LiteLLMPlugin.from_config(config={})
        assert p.base_url == "http://localhost:4000"
        assert p.poll_interval == 60

    @patch.dict("os.environ", {
        "LITELLM_BASE_URL": "http://env-proxy:4000",
        "LITELLM_API_KEY": "sk-env-key",
    })
    def test_factory_env_var_fallback(self):
        p = LiteLLMPlugin.from_config(config={})
        assert p.base_url == "http://env-proxy:4000"
        assert p.api_key == "sk-env-key"

    def test_factory_config_overrides_env(self):
        import os
        with patch.dict("os.environ", {"LITELLM_BASE_URL": "http://env:4000"}):
            p = LiteLLMPlugin.from_config(config={"base_url": "http://config:4000"})
            assert p.base_url == "http://config:4000"

    def test_trailing_slash_stripped(self):
        p = LiteLLMPlugin(base_url="http://localhost:4000/")
        assert p.base_url == "http://localhost:4000"

    def test_poll_interval_minimum(self):
        p = LiteLLMPlugin(poll_interval=1)
        assert p.poll_interval == 10

    def test_auto_poll_string_false(self):
        p = LiteLLMPlugin.from_config(config={"auto_poll": "false"})
        assert p._auto_poll is False


# ---------------------------------------------------------------------------
# TestVerifyConnection
# ---------------------------------------------------------------------------

class TestVerifyConnection:
    @patch("open_cite.plugins.litellm.requests.get")
    def test_success(self, mock_get, plugin):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "tool1", "server_name": "srv1"}]
        mock_get.return_value = mock_resp

        result = plugin.verify_connection()
        assert result["success"] is True
        assert result["tool_count_hint"] == 1
        mock_get.assert_called_once()

    @patch("open_cite.plugins.litellm.requests.get")
    def test_http_error(self, mock_get, plugin):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_get.return_value = mock_resp

        result = plugin.verify_connection()
        assert result["success"] is False
        assert "401" in result["error"]

    @patch("open_cite.plugins.litellm.requests.get")
    def test_network_error(self, mock_get, plugin):
        from requests import ConnectionError
        mock_get.side_effect = ConnectionError("Connection refused")

        result = plugin.verify_connection()
        assert result["success"] is False
        assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# TestListMCPTools
# ---------------------------------------------------------------------------

class TestListMCPTools:
    @patch("open_cite.plugins.litellm.requests.get")
    def test_tools_discovered(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        tools = plugin.list_assets("mcp_tool")

        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"read_file", "write_file", "search"}

    @patch("open_cite.plugins.litellm.requests.get")
    def test_schema_included(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        tools = plugin.list_assets("mcp_tool")

        read_file = next(t for t in tools if t["name"] == "read_file")
        assert read_file["schema"]["type"] == "object"
        assert "path" in read_file["schema"]["properties"]

    @patch("open_cite.plugins.litellm.requests.get")
    def test_server_name_extraction(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        tools = plugin.list_assets("mcp_tool")

        read_file = next(t for t in tools if t["name"] == "read_file")
        assert read_file["server_name"] == "filesystem"
        assert read_file["server_id"] == "litellm:filesystem"

    @patch("open_cite.plugins.litellm.requests.get")
    def test_stale_cache_on_error(self, mock_get, plugin, sample_tools):
        # First successful fetch
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        plugin._fetch_mcp_tools()

        # Second fetch fails
        mock_get.side_effect = Exception("timeout")
        plugin._fetch_mcp_tools()

        # Stale cache still available
        tools = plugin.list_assets("mcp_tool")
        assert len(tools) == 3

    @patch("open_cite.plugins.litellm.requests.get")
    def test_dict_response_with_tools_key(self, mock_get, plugin, sample_tools):
        """LiteLLM proxies that wrap tools in {'tools': [...]} should work."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"tools": sample_tools}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        tools = plugin.list_assets("mcp_tool")
        assert len(tools) == 3

    @patch("open_cite.plugins.litellm.requests.get")
    def test_mcp_standard_envelope(self, mock_get, plugin, sample_tools):
        """MCP standard {'result': {'tools': [...]}} format should work."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"tools": sample_tools}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        tools = plugin.list_assets("mcp_tool")
        assert len(tools) == 3

    @patch("open_cite.plugins.litellm.requests.get")
    def test_tool_id_format(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        tools = plugin.list_assets("mcp_tool")

        read_file = next(t for t in tools if t["name"] == "read_file")
        assert read_file["id"] == "litellm:filesystem:read_file"

    @patch("open_cite.plugins.litellm.requests.get")
    def test_server_name_from_mcp_info(self, mock_get, plugin):
        """Real LiteLLM responses nest server_name under mcp_info."""
        nested_tools = [
            {
                "name": "slack_post_message",
                "description": "Post a message",
                "inputSchema": {"type": "object"},
                "mcp_info": {"server_name": "slack"},
            },
        ]
        mock_get.return_value = _mock_resp({"tools": nested_tools})

        plugin._fetch_mcp_tools()
        tools = plugin.list_assets("mcp_tool")
        assert len(tools) == 1
        assert tools[0]["server_name"] == "slack"
        assert tools[0]["server_id"] == "litellm:slack"
        assert tools[0]["id"] == "litellm:slack:slack_post_message"


# ---------------------------------------------------------------------------
# TestListMCPServers
# ---------------------------------------------------------------------------

class TestListMCPServers:
    @patch("open_cite.plugins.litellm.requests.get")
    def test_servers_derived_from_tools(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        plugin._fetch_mcp_servers()
        servers = plugin.list_assets("mcp_server")

        assert len(servers) == 2
        names = {s["name"] for s in servers}
        assert names == {"filesystem", "brave-search"}

    @patch("open_cite.plugins.litellm.requests.get")
    def test_tools_provided_list(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        plugin._fetch_mcp_servers()
        servers = plugin.list_assets("mcp_server")

        fs = next(s for s in servers if s["name"] == "filesystem")
        assert len(fs["tools_provided"]) == 2
        assert fs["tools_count"] == 2

    @patch("open_cite.plugins.litellm.requests.get")
    def test_endpoint_format(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        plugin._fetch_mcp_servers()
        servers = plugin.list_assets("mcp_server")

        for srv in servers:
            assert srv["endpoint"] == "http://litellm.test:4000/mcp-rest"
            assert srv["transport"] == "http"

    @patch("open_cite.plugins.litellm.requests.get")
    def test_servers_from_api_with_permissions(
        self, mock_get, plugin, sample_servers, sample_tools,
    ):
        """`/v1/mcp/server` is fetched and server-level permissions are surfaced."""
        # First call: server endpoint. Second call: tool endpoint.
        mock_get.side_effect = [
            _mock_resp(sample_servers),
            _mock_resp(sample_tools),
        ]
        plugin._fetch_mcp_servers()  # populates cache from API
        plugin._fetch_mcp_tools()
        # Re-fetch servers to merge in tool counts (mock side_effect must
        # be re-set since it's exhausted).
        mock_get.side_effect = [_mock_resp(sample_servers)]
        plugin._fetch_mcp_servers()

        servers = plugin.list_assets("mcp_server")
        names = {s["name"] for s in servers}
        assert names == {"filesystem", "brave-search"}

        fs = next(s for s in servers if s["name"] == "filesystem")
        assert fs["transport"] == "stdio"
        assert fs["metadata"]["litellm.allowed_teams"] == ["team-alpha"]
        assert fs["metadata"]["litellm.allow_all_keys"] is True

        bs = next(s for s in servers if s["name"] == "brave-search")
        assert bs["endpoint"] == "https://brave.example.com/mcp"
        assert bs["metadata"]["litellm.mcp_access_groups"] == ["Engineering"]
        assert bs["metadata"]["litellm.allowed_tools"] == ["search"]
        assert bs["metadata"]["litellm.allow_all_keys"] is False

    @patch("open_cite.plugins.litellm.requests.get")
    def test_server_endpoint_404_falls_back_to_tools(
        self, mock_get, plugin, sample_tools,
    ):
        """If `/v1/mcp/server` is unavailable, derive servers from the tools cache."""
        not_found = _mock_resp({"detail": "Not Found"}, status=404)
        tools_ok = _mock_resp(sample_tools)
        mock_get.side_effect = [not_found, tools_ok, _mock_resp(sample_tools)]

        plugin._fetch_mcp_servers()  # 404 -> empty
        plugin._fetch_mcp_tools()    # populates tools cache
        plugin._fetch_mcp_servers()  # 404 again -> falls back to tools

        servers = plugin.list_assets("mcp_server")
        names = {s["name"] for s in servers}
        assert names == {"filesystem", "brave-search"}


# ---------------------------------------------------------------------------
# TestListTeams
# ---------------------------------------------------------------------------

class TestListTeams:
    @patch("open_cite.plugins.litellm.requests.get")
    def test_teams_discovered(self, mock_get, plugin, sample_teams):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_teams
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_teams()

        with plugin._lock:
            teams = list(plugin._teams_cache)
        assert len(teams) == 2
        assert teams[0]["name"] == "Alpha Team"
        assert teams[0]["identity_type"] == "team"

    @patch("open_cite.plugins.litellm.requests.get")
    def test_mcp_permissions(self, mock_get, plugin, sample_teams):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_teams
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_teams()

        with plugin._lock:
            teams = list(plugin._teams_cache)
        alpha = next(t for t in teams if t["name"] == "Alpha Team")
        assert alpha["metadata"]["litellm.mcp_servers"] == ["filesystem", "brave-search"]
        assert alpha["metadata"]["litellm.mcp_tools"] == ["read_file"]

    @patch("open_cite.plugins.litellm.requests.get")
    def test_members_extracted(self, mock_get, plugin, sample_teams):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_teams
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_teams()

        with plugin._lock:
            teams = list(plugin._teams_cache)
        alpha = next(t for t in teams if t["name"] == "Alpha Team")
        assert "user-001" in alpha["metadata"]["litellm.members"]
        assert "user-002" in alpha["metadata"]["litellm.members"]

    @patch("open_cite.plugins.litellm.requests.get")
    def test_stale_cache_on_error(self, mock_get, plugin, sample_teams):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_teams
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        plugin._fetch_teams()

        mock_get.side_effect = Exception("timeout")
        plugin._fetch_teams()

        with plugin._lock:
            assert len(plugin._teams_cache) == 2

    @patch("open_cite.plugins.litellm.requests.get")
    def test_dict_response_format(self, mock_get, plugin, sample_teams):
        """Test that dict responses with 'teams' key work."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"teams": sample_teams}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_teams()

        with plugin._lock:
            assert len(plugin._teams_cache) == 2


# ---------------------------------------------------------------------------
# TestListUsers
# ---------------------------------------------------------------------------

class TestListUsers:
    @patch("open_cite.plugins.litellm.requests.get")
    def test_users_discovered(self, mock_get, plugin, sample_users, sample_teams):
        # Setup teams first
        teams_resp = MagicMock()
        teams_resp.status_code = 200
        teams_resp.json.return_value = sample_teams
        teams_resp.raise_for_status = MagicMock()

        users_resp = MagicMock()
        users_resp.status_code = 200
        users_resp.json.return_value = sample_users
        users_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [teams_resp, users_resp]

        plugin._fetch_teams()
        plugin._fetch_users()

        with plugin._lock:
            users = list(plugin._users_cache)
        assert len(users) == 2
        assert users[0]["name"] == "alice@example.com"
        assert users[0]["identity_type"] == "user"

    @patch("open_cite.plugins.litellm.requests.get")
    def test_team_membership(self, mock_get, plugin, sample_users, sample_teams):
        teams_resp = MagicMock()
        teams_resp.status_code = 200
        teams_resp.json.return_value = sample_teams
        teams_resp.raise_for_status = MagicMock()

        users_resp = MagicMock()
        users_resp.status_code = 200
        users_resp.json.return_value = sample_users
        users_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [teams_resp, users_resp]

        plugin._fetch_teams()
        plugin._fetch_users()

        with plugin._lock:
            users = list(plugin._users_cache)
        bob = next(u for u in users if "bob" in u["name"])
        assert "team-alpha" in bob["metadata"]["litellm.teams"]
        assert "team-beta" in bob["metadata"]["litellm.teams"]

    @patch("open_cite.plugins.litellm.requests.get")
    def test_effective_mcp_access(self, mock_get, plugin, sample_users, sample_teams):
        teams_resp = MagicMock()
        teams_resp.status_code = 200
        teams_resp.json.return_value = sample_teams
        teams_resp.raise_for_status = MagicMock()

        users_resp = MagicMock()
        users_resp.status_code = 200
        users_resp.json.return_value = sample_users
        users_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [teams_resp, users_resp]

        plugin._fetch_teams()
        plugin._fetch_users()

        with plugin._lock:
            users = list(plugin._users_cache)
        bob = next(u for u in users if "bob" in u["name"])
        effective = bob["metadata"]["litellm.effective_mcp_servers"]
        assert "filesystem" in effective
        assert "brave-search" in effective

    @patch("open_cite.plugins.litellm.requests.get")
    def test_stale_cache_on_error(self, mock_get, plugin, sample_users, sample_teams):
        teams_resp = MagicMock()
        teams_resp.status_code = 200
        teams_resp.json.return_value = sample_teams
        teams_resp.raise_for_status = MagicMock()

        users_resp = MagicMock()
        users_resp.status_code = 200
        users_resp.json.return_value = sample_users
        users_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [teams_resp, users_resp]
        plugin._fetch_teams()
        plugin._fetch_users()

        mock_get.side_effect = Exception("timeout")
        plugin._fetch_users()

        with plugin._lock:
            assert len(plugin._users_cache) == 2


# ---------------------------------------------------------------------------
# TestLineage
# ---------------------------------------------------------------------------

class TestLineage:
    def _setup_caches(
        self, plugin, mock_get, sample_tools, sample_teams, sample_users,
        sample_servers=None,
    ):
        """Helper to populate all caches.

        Mocks the four endpoints used by ``_discover_all`` in the same order
        the plugin calls them: servers → tools → teams → users.
        """
        servers_payload = sample_servers if sample_servers is not None else []
        mock_get.side_effect = [
            _mock_resp(servers_payload),
            _mock_resp(sample_tools),
            _mock_resp(sample_teams),
            _mock_resp(sample_users),
        ]

        plugin._fetch_mcp_servers()
        plugin._fetch_mcp_tools()
        plugin._fetch_teams()
        plugin._fetch_users()
        plugin._build_lineage()

    @patch("open_cite.plugins.litellm.requests.get")
    def test_server_tool_contains(self, mock_get, plugin, sample_tools, sample_teams, sample_users):
        self._setup_caches(plugin, mock_get, sample_tools, sample_teams, sample_users)

        contains = [
            r for r in plugin.lineage.values()
            if r["relationship_type"] == "contains"
        ]
        assert len(contains) == 3  # 3 tools across 2 servers
        source_types = {r["source_type"] for r in contains}
        target_types = {r["target_type"] for r in contains}
        assert source_types == {"mcp_server"}
        assert target_types == {"mcp_tool"}

    @patch("open_cite.plugins.litellm.requests.get")
    def test_team_server_has_access_to(self, mock_get, plugin, sample_tools, sample_teams, sample_users):
        self._setup_caches(plugin, mock_get, sample_tools, sample_teams, sample_users)

        team_server = [
            r for r in plugin.lineage.values()
            if r["relationship_type"] == "has_access_to"
            and r["source_type"] == "identity"
            and r["target_type"] == "mcp_server"
            and ":team:" in r["source_id"]
        ]
        # Alpha -> filesystem, brave-search; Beta -> brave-search
        assert len(team_server) == 3

    @patch("open_cite.plugins.litellm.requests.get")
    def test_user_team_member_of(self, mock_get, plugin, sample_tools, sample_teams, sample_users):
        self._setup_caches(plugin, mock_get, sample_tools, sample_teams, sample_users)

        member_of = [
            r for r in plugin.lineage.values()
            if r["relationship_type"] == "member_of"
        ]
        # alice -> alpha; bob -> alpha, beta
        assert len(member_of) == 3

    @patch("open_cite.plugins.litellm.requests.get")
    def test_user_server_derived_access(self, mock_get, plugin, sample_tools, sample_teams, sample_users):
        self._setup_caches(plugin, mock_get, sample_tools, sample_teams, sample_users)

        user_server = [
            r for r in plugin.lineage.values()
            if r["relationship_type"] == "has_access_to"
            and r["source_type"] == "identity"
            and r["target_type"] == "mcp_server"
            and ":user:" in r["source_id"]
        ]
        # alice -> filesystem, brave-search (from alpha)
        # bob -> filesystem, brave-search (from alpha+beta, deduplicated)
        assert len(user_server) == 4

    @patch("open_cite.plugins.litellm.requests.get")
    def test_lineage_rebuild(self, mock_get, plugin, sample_tools, sample_teams, sample_users):
        """Lineage is fully rebuilt on each cycle (not appended)."""
        self._setup_caches(plugin, mock_get, sample_tools, sample_teams, sample_users)
        first_count = len(plugin.lineage)

        # Rebuild
        plugin._build_lineage()
        assert len(plugin.lineage) == first_count

    @patch("open_cite.plugins.litellm.requests.get")
    def test_server_team_allowlist_creates_lineage(
        self, mock_get, plugin, sample_servers, sample_tools, sample_teams, sample_users,
    ):
        """Server-side `teams` allowlist creates team→server has_access_to edges."""
        self._setup_caches(
            plugin, mock_get, sample_tools, sample_teams, sample_users,
            sample_servers=sample_servers,
        )

        # filesystem server has teams=["team-alpha"]
        edge = next(
            (e for e in plugin.lineage.values()
             if e["source_id"] == "litellm:team:team-alpha"
             and e["target_id"] == "litellm:filesystem"
             and e["relationship_type"] == "has_access_to"),
            None,
        )
        assert edge is not None
        assert edge["source_type"] == "identity"
        assert edge["target_type"] == "mcp_server"


# ---------------------------------------------------------------------------
# TestIdentityListing
# ---------------------------------------------------------------------------

class TestIdentityListing:
    @patch("open_cite.plugins.litellm.requests.get")
    def test_list_identities_includes_teams_and_users(self, mock_get, plugin, sample_teams, sample_users):
        teams_resp = MagicMock()
        teams_resp.status_code = 200
        teams_resp.json.return_value = sample_teams
        teams_resp.raise_for_status = MagicMock()

        users_resp = MagicMock()
        users_resp.status_code = 200
        users_resp.json.return_value = sample_users
        users_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [teams_resp, users_resp]
        plugin._fetch_teams()
        plugin._fetch_users()

        identities = plugin.list_assets("identity")
        assert len(identities) == 4  # 2 teams + 2 users
        types = {i["identity_type"] for i in identities}
        assert types == {"team", "user"}

    def test_unsupported_asset_type_raises(self, plugin):
        with pytest.raises(ValueError, match="Unsupported asset type"):
            plugin.list_assets("catalog")


# ---------------------------------------------------------------------------
# TestSyncMCPDicts
# ---------------------------------------------------------------------------

class TestSyncMCPDicts:
    @patch("open_cite.plugins.litellm.requests.get")
    def test_mcp_dicts_populated(self, mock_get, plugin, sample_tools):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_tools
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        plugin._fetch_mcp_tools()
        plugin._fetch_mcp_servers()
        plugin._sync_mcp_dicts()

        assert len(plugin.mcp_servers) == 2
        assert len(plugin.mcp_tools) == 3

        # Check server dict format matches what _save_current_state expects
        fs = plugin.mcp_servers["litellm:filesystem"]
        assert fs["name"] == "filesystem"
        assert fs["transport"] == "http"
        assert "tools_provided" in fs

        # Check tool dict format
        rf = plugin.mcp_tools["litellm:filesystem:read_file"]
        assert rf["name"] == "read_file"
        assert rf["server_id"] == "litellm:filesystem"


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_sets_running(self, plugin):
        plugin.start()
        assert plugin.status == "running"
        plugin.stop()

    def test_stop_sets_stopped(self, plugin):
        plugin.start()
        plugin.stop()
        assert plugin.status == "stopped"

    @patch("open_cite.plugins.litellm.requests.get")
    def test_discover_all_calls(
        self, mock_get, plugin, sample_servers, sample_tools, sample_teams, sample_users,
    ):
        """_discover_all populates all caches by calling the four endpoints in order."""
        mock_get.side_effect = [
            _mock_resp(sample_servers),
            _mock_resp(sample_tools),
            _mock_resp(sample_teams),
            _mock_resp(sample_users),
        ]

        plugin._discover_all()

        assert len(plugin.list_assets("mcp_tool")) == 3
        assert len(plugin.list_assets("mcp_server")) == 2
        assert len(plugin.list_assets("identity")) == 4
        assert len(plugin.lineage) > 0
        assert len(plugin.mcp_servers) == 2
        assert len(plugin.mcp_tools) == 3

    def test_export_assets(self, plugin):
        export = plugin.export_assets()
        assert "mcp_servers" in export
        assert "mcp_tools" in export
        assert "identities" in export


# ---------------------------------------------------------------------------
# TestAuthHeaders
# ---------------------------------------------------------------------------

class TestAuthHeaders:
    def test_with_api_key(self, plugin):
        headers = plugin._auth_headers()
        assert headers["Authorization"] == "Bearer sk-test-key"

    def test_without_api_key(self):
        p = LiteLLMPlugin(base_url="http://localhost:4000")
        headers = p._auth_headers()
        assert "Authorization" not in headers
