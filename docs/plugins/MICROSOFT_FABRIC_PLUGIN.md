# Microsoft Fabric Plugin

The Microsoft Fabric plugin discovers data analytics and AI/ML assets across
Microsoft Fabric workspaces.

## Discovered Asset Types

| Asset Type | Fabric API Endpoint | Description |
|---|---|---|
| `workspace` | `/v1/workspaces` | Fabric workspaces |
| `lakehouse` | `/v1/workspaces/{id}/lakehouses` | Data engineering lakehouses |
| `warehouse` | `/v1/workspaces/{id}/warehouses` | Data warehouses |
| `notebook` | `/v1/workspaces/{id}/notebooks` | Notebooks |
| `pipeline` | `/v1/workspaces/{id}/pipelines` | Data integration pipelines |
| `ml_model` | `/v1/workspaces/{id}/mlModels` | Machine learning models |
| `ml_experiment` | `/v1/workspaces/{id}/mlExperiments` | ML experiments |
| `report` | `/v1/workspaces/{id}/reports` | Power BI reports |
| `semantic_model` | `/v1/workspaces/{id}/semanticModels` | Semantic models (datasets) |
| `event_stream` | `/v1/workspaces/{id}/eventstreams` | Real-time event streams |
| `kql_database` | `/v1/workspaces/{id}/kqlDatabases` | KQL databases (real-time analytics) |
| `capacity` | `/v1/capacities` | Fabric capacities |

## Authentication

The plugin authenticates via **Microsoft Entra ID** (Azure AD) using the
OAuth 2.0 client-credentials flow. You need:

1. An **App Registration** in Azure AD with the appropriate Fabric API
   permissions.
2. The **Tenant ID**, **Client (Application) ID**, and a **Client Secret**.

### Required API Permissions

Grant the following application permissions to your app registration in the
Azure portal:

| Permission | Description |
|---|---|
| `Workspace.ReadWrite.All` | Read/list workspaces |
| `Item.ReadWrite.All` | Read/list items within workspaces |
| `Capacity.ReadWrite.All` | Read/list capacities |

Alternatively, you can provide a pre-acquired **access token** if you prefer
to handle token acquisition outside of the plugin.

### Environment Variables

| Variable | Description |
|---|---|
| `FABRIC_TENANT_ID` | Azure AD tenant ID |
| `FABRIC_CLIENT_ID` | App registration client ID |
| `FABRIC_CLIENT_SECRET` | App registration client secret |

## Configuration

### GUI / API Configuration Fields

| Field | Required | Description |
|---|---|---|
| `tenant_id` | Yes | Azure AD tenant ID |
| `client_id` | Yes | Application (client) ID |
| `client_secret` | Yes | Client secret (stored securely) |
| `workspace_ids` | No | Comma-separated workspace IDs to scope discovery. Leave blank to discover all accessible workspaces. |

### Programmatic Usage

```python
from open_cite.plugins.microsoft_fabric import MicrosoftFabricPlugin

plugin = MicrosoftFabricPlugin(
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret",
    workspace_ids=["ws-id-1", "ws-id-2"],  # optional
)

# Verify connectivity
result = plugin.verify_connection()
print(result)

# List workspaces
workspaces = plugin.list_assets("workspace")

# List lakehouses across all target workspaces
lakehouses = plugin.list_assets("lakehouse")

# List items in a specific workspace
notebooks = plugin.list_assets("notebook", workspace_id="ws-id-1")
```

### Using a Pre-Acquired Token

```python
plugin = MicrosoftFabricPlugin(access_token="eyJ0eXAi...")
```

## API Details

- **Base URL:** `https://api.fabric.microsoft.com/v1`
- **Token endpoint:** `https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token`
- **Scope:** `https://api.fabric.microsoft.com/.default`
- **Pagination:** The plugin automatically handles `continuationToken`-based
  pagination across all list endpoints.

## Workspace Scoping

By default the plugin discovers all workspaces the service principal has
access to. You can limit discovery to specific workspaces by providing a
comma-separated list of workspace IDs in the `workspace_ids` configuration
field (or passing a list programmatically).

## Running Tests

```bash
pytest tests/test_microsoft_fabric.py -v
```
