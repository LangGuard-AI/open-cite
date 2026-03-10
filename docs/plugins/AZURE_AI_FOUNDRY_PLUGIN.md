# Azure AI Foundry Plugin

The Azure AI Foundry plugin discovers AI resources, model deployments, projects,
agents, tools, and traces across an Azure subscription.

## Discovered Asset Types

| Asset Type | API | Description |
|---|---|---|
| `foundry_resource` | ARM: `GET .../accounts` (kind=AIServices) | AI Foundry resources |
| `openai_resource` | ARM: same endpoint (kind=OpenAI) | Azure OpenAI resources |
| `deployment` | ARM: `GET .../{account}/deployments` | Model deployments (GPT, embedding, etc.) |
| `model` | ARM + Log Analytics | Deployments with usage counts from traces |
| `project` | ARM: `GET .../{account}/projects` (preview) | AI Foundry projects |
| `agent` | Service: `GET /openai/assistants` | AI agents (OpenAI assistants) |
| `tool` | Derived from agents | Tools used by agents (code_interpreter, file_search, functions) |
| `trace` | Log Analytics: KQL query on `AzureDiagnostics` | Traces from diagnostic settings |

## APIs Used

| API | Base URL | Token Scope | Discovers |
|---|---|---|---|
| ARM | `management.azure.com` | `management.azure.com/.default` | Accounts, deployments, projects |
| Foundry Service | `<account>.services.ai.azure.com` | `cognitiveservices.azure.com/.default` | Agents, tools |
| Log Analytics | `api.loganalytics.azure.com` | `api.loganalytics.io/.default` | Traces, model usage counts |

## Authentication

The plugin authenticates via **Microsoft Entra ID** (Azure AD) using the
OAuth 2.0 client-credentials flow. You need:

1. An **App Registration** in Azure AD.
2. The **Tenant ID**, **Client (Application) ID**, and a **Client Secret**.
3. The service principal must have the following roles:

| Role | Scope | Required For |
|---|---|---|
| **Reader** | Subscription or resource group | Account, deployment, project discovery |
| **Cognitive Services User** | Cognitive Services account | Agent and tool discovery |
| **Log Analytics Reader** | Log Analytics workspace | Trace discovery |

### Creating an App Registration

```bash
# 1. Create the app registration
az ad app create --display-name "open-cite"

# 2. Note the appId from the output, then create a service principal
az ad sp create --id <app-id>

# 3. Generate a client secret
az ad app credential reset --id <app-id> --display-name "open-cite-key"
```

The output of step 3 gives you `appId` (Client ID), `password` (Client Secret),
and `tenant` (Tenant ID). Your Subscription ID is available via
`az account show --query id -o tsv`.

### Granting Access

```bash
# Reader role on the subscription (ARM discovery)
az role assignment create \
    --assignee <client-id> \
    --role Reader \
    --scope /subscriptions/<subscription-id>

# Cognitive Services User (agent discovery)
az role assignment create \
    --assignee <client-id> \
    --role "Cognitive Services User" \
    --scope /subscriptions/<subscription-id>

# Log Analytics Reader (trace discovery)
az role assignment create \
    --assignee <client-id> \
    --role "Log Analytics Reader" \
    --scope /subscriptions/<subscription-id>
```

### Environment Variables

| Variable | Description |
|---|---|
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_CLIENT_ID` | App registration client ID |
| `AZURE_CLIENT_SECRET` | App registration client secret |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |

## Configuration

### GUI / API Configuration Fields

| Field | Required | Default | Description |
|---|---|---|---|
| `tenant_id` | Yes | | Azure AD tenant ID |
| `client_id` | Yes | | Application (client) ID |
| `client_secret` | Yes | | Client secret (stored securely) |
| `subscription_id` | Yes | | Azure subscription to scan |
| `resource_group` | No | all | Narrow discovery to one resource group |
| `account_filter` | No | all | Comma-separated account names to include |
| `lookback_hours` | No | 24 | How far back to query logs (1–720) |
| `poll_interval` | No | 60 | Seconds between background polling cycles (min: 10) |

### Trace Discovery

The plugin **auto-discovers** Log Analytics workspaces by reading the diagnostic
settings on each Cognitive Services account. No manual configuration is needed
if diagnostic settings are already in place.

Trace discovery requires three Azure resources wired together:

1. **Log Analytics workspace** — stores the trace data
2. **Application Insights** — instruments the AI Foundry account, backed by
   the workspace
3. **Diagnostic settings** — on the Cognitive Services account, routes
   `Audit`, `RequestResponse`, and `Trace` logs to the workspace

The populate script can create all three automatically:

```bash
python scripts/populate_foundry_test_data.py \
    --skip-deployments --skip-projects --skip-agents --skip-threads \
    --setup-tracing
```

If you need to override auto-discovery (e.g. querying a workspace that isn't
linked via diagnostic settings), pass `log_analytics_workspace_id` in the
programmatic constructor. This is not exposed in the GUI — it's an advanced
override only.

### Programmatic Usage

```python
from open_cite.plugins.azure_ai_foundry import AzureAIFoundryPlugin

plugin = AzureAIFoundryPlugin(
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret",
    subscription_id="your-subscription-id",
    resource_group="my-rg",          # optional — narrow to one RG
    account_filter=["my-foundry"],   # optional — narrow to specific accounts
    lookback_hours=48,               # optional — query last 48h of logs
    poll_interval=120,               # optional — poll every 2 minutes
)

# Verify connectivity
result = plugin.verify_connection()

# List AI Foundry resources
resources = plugin.list_assets("foundry_resource")

# List model deployments across all accounts
deployments = plugin.list_assets("deployment")

# List models with usage counts
models = plugin.list_assets("model")

# List projects (preview API)
projects = plugin.list_assets("project")

# List agents (OpenAI assistants)
agents = plugin.list_assets("agent")

# List tools (extracted from agents)
tools = plugin.list_assets("tool")

# List traces (auto-discovers Log Analytics workspace from diagnostic settings)
traces = plugin.list_assets("trace")
```

### Using a Pre-Acquired Token

```python
plugin = AzureAIFoundryPlugin(
    access_token="eyJ0eXAi...",
    subscription_id="your-subscription-id",
)
```

Note: Pre-acquired tokens only work for ARM discovery. Agent, tool, and trace
discovery require `tenant_id` + `client_id` + `client_secret` to acquire
scope-specific tokens.

## Deployment Metadata

Each discovered deployment includes:

| Metadata Key | Description |
|---|---|
| `foundry.model_name` | Model name (e.g., `gpt-4o`) |
| `foundry.model_format` | Format (e.g., `OpenAI`) |
| `foundry.model_version` | Model version |
| `foundry.model_publisher` | Publisher |
| `foundry.sku_name` | SKU tier (e.g., `GlobalStandard`) |
| `foundry.sku_capacity` | Provisioned capacity |

## Agent Metadata

| Metadata Key | Description |
|---|---|
| `foundry.agent_id` | Assistant ID (e.g., `asst_abc123`) |
| `foundry.model` | Deployment model used |
| `foundry.account_name` | Parent Cognitive Services account |
| `foundry.instructions_preview` | First 200 chars of instructions |

## Trace Webhook Forwarding

When traces are discovered and webhooks are configured, each trace is converted
to OTLP format and forwarded to subscribed webhooks. The OTLP payload includes:
- `service.name`: `azure-ai-foundry`
- `opencite.discovery_source`: `azure_ai_foundry`
- Span attributes from Application Insights `Properties`

## Test Data Script

A helper script creates test deployments, projects, agents, and conversations:

```bash
python scripts/populate_foundry_test_data.py \
    --subscription-id <sub> \
    --resource-group <rg> \
    --account-name <account>
```

Flags: `--skip-deployments`, `--skip-projects`, `--skip-agents`, `--skip-threads`, `--setup-tracing`, `--dry-run`

## Running Tests

```bash
pytest tests/test_azure_ai_foundry.py -v
```
