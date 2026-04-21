# LiteLLM MCP Proxy Plugin

The LiteLLM plugin discovers MCP tools and identity-to-tool authorization
mappings from a remote LiteLLM proxy instance.

## Prerequisites

- LiteLLM proxy with `enable_mcp_registry: true` in its config
- An API key (admin key sees all resources; regular key sees its authorized subset)

## Discovered Asset Types

| Asset Type | LiteLLM API Source | Description |
|---|---|---|
| `mcp_server` | `GET /v1/mcp/server` (preferred) or derived from tool list | Registered MCP servers with permission metadata |
| `mcp_tool` | `GET /mcp-rest/tools/list` | MCP tools available through the proxy |
| `identity` | `GET /team/list`, `GET /user/list` | Teams and users with MCP access |

The plugin extracts each tool's `server_name` from the nested `mcp_info`
field used by recent LiteLLM versions, falling back to a top-level
`server_name` for older releases.

Each `mcp_server` carries permission metadata in its `metadata` map:

| Metadata Key | Source | Meaning |
|---|---|---|
| `litellm.allowed_teams` | server `teams` | Team IDs explicitly granted access |
| `litellm.mcp_access_groups` | server `mcp_access_groups` | Named access groups granted to the server |
| `litellm.allowed_tools` | server `allowed_tools` | Tool allowlist (empty = all tools allowed) |
| `litellm.allow_all_keys` | server `allow_all_keys` | Whether any API key may use the server |
| `litellm.auth_type` | server `auth_type` | Server-side auth type |

## Lineage Relationships

| Source | Target | Relationship | Source Field |
|---|---|---|---|
| `mcp_server` | `mcp_tool` | `contains` | derived |
| `identity` (team) | `mcp_server` | `has_access_to` | team `object_permission.mcp_servers` and server `teams` |
| `identity` (team) | `mcp_tool` | `has_access_to` | team `object_permission.mcp_tools` |
| `identity` (user) | `identity` (team) | `member_of` | user `teams` |
| `identity` (user) | `mcp_server` | `has_access_to` | derived from team memberships |

If your proxy has no teams configured (the default), no
`has_access_to` / `member_of` edges will be produced. Configure teams in
LiteLLM and assign MCP server access via `object_permission.mcp_servers`
on each team (or set `teams: [<team_id>]` on the MCP server itself) to
populate the permission graph.

## Configuration

| Field | Required | Default | Description |
|---|---|---|---|
| `base_url` | Yes | `http://localhost:4000` | LiteLLM proxy URL |
| `api_key` | Yes | | Admin or regular API key |
| `poll_interval` | No | 60 | Seconds between polls (min: 10) |

### Environment Variables

| Variable | Description |
|---|---|
| `LITELLM_BASE_URL` | Proxy URL (fallback if not in config) |
| `LITELLM_API_KEY` | API key (fallback if not in config) |

## Programmatic Usage

```python
from open_cite.plugins.litellm import LiteLLMPlugin

plugin = LiteLLMPlugin(
    base_url="http://litellm-proxy:4000",
    api_key="sk-your-admin-key",
    poll_interval=120,
)

result = plugin.verify_connection()

tools = plugin.list_assets("mcp_tool")
servers = plugin.list_assets("mcp_server")
identities = plugin.list_assets("identity")
```

## Running Tests

```bash
pytest tests/test_litellm.py -v
```
