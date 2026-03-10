# Open-CITE JSON Schema Documentation

## Overview

The Open-CITE JSON schema provides a standardized format for storing and exchanging discovery data about AI tools, models, data assets, MCP servers, and traces. This schema enables:

- **Interoperability**: Share discovery data between systems
- **Persistence**: Store discoveries for historical analysis
- **Integration**: Feed discovery data into other tools and dashboards
- **Compliance**: Track AI usage for governance and auditing

## Schema Version

Current version: **1.0.0**

Schema location: `schema/opencite-schema.json`

## Root Structure

The programmatic export (`OpenCiteExporter.export_discovery()`) produces:

```json
{
  "opencite_version": "1.0.0",
  "export_timestamp": "2026-03-10T14:30:00.000Z",
  "tools": [],
  "models": [],
  "data_assets": [],
  "mcp_servers": [],
  "mcp_tools": [],
  "mcp_resources": [],
  "plugins": []
}
```

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `opencite_version` | string | Schema version (semver) |
| `export_timestamp` | string (ISO 8601) | When the export was generated |
| `tools` | array | Discovered AI tools and applications |
| `models` | array | Discovered AI models |
| `data_assets` | array | Data assets (catalogs, tables, deployments, etc.) |
| `mcp_servers` | array | Discovered MCP servers |
| `mcp_tools` | array | Tools provided by MCP servers |
| `mcp_resources` | array | Resources provided by MCP servers |
| `plugins` | array | Plugins that contributed to this export |
| `traces` | array | Collected traces (when included) |
| `metadata` | object | Metadata about the discovery session (full export format) |

## Discovery Source Values

The `discovery_source` field identifies which plugin or subsystem discovered an asset. Values vary by plugin:

| Plugin | Discovery Source Values |
|--------|----------------------|
| OpenTelemetry | `opentelemetry`, `trace_analysis` |
| Databricks | `databricks`, `databricks/mlflow`, `databricks/genie`, `databricks/ai-gateway` |
| Azure AI Foundry | `azure_ai_foundry` |
| Google Cloud | `google_cloud_api`, `gcp_compute_labels`, `gcp_port_scan` |
| AWS Bedrock | `aws_bedrock_api`, `cloudtrail`, `cloudwatch_logs` |
| AWS SageMaker | `aws_sagemaker_api` |
| Splunk | Plugin instance ID (dynamic) |
| Manual | `manual` |

## Tools

Represents AI tools and applications that use models.

### Tool Schema

```json
{
  "id": "customer-service-bot",
  "name": "customer-service-bot",
  "type": "application",
  "description": "AI-powered customer service chatbot",
  "discovery_source": "opentelemetry",
  "models_used": [
    {
      "model_id": "openai/gpt-4",
      "usage_count": 3,
      "first_seen": "2026-03-10T14:25:10.000Z",
      "last_seen": "2026-03-10T14:29:45.000Z"
    }
  ],
  "provider": "openrouter",
  "trace_count": 3,
  "traces": ["trace-001", "trace-002"],
  "first_seen": "2026-03-10T14:25:10.000Z",
  "last_seen": "2026-03-10T14:29:45.000Z",
  "metadata": {
    "owner": "customer-experience-team",
    "environment": "production"
  },
  "tags": ["production", "customer-facing"]
}
```

### Tool Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `name` | string | Yes | Display name |
| `type` | string | No | application, service, library, agent, chatbot, unknown |
| `description` | string | No | Human-readable description |
| `discovery_source` | string | Yes | Plugin/subsystem that discovered this tool |
| `models_used` | array | No | Models used by this tool |
| `provider` | string | No | LLM provider/gateway |
| `trace_count` | integer | No | Number of traces collected |
| `traces` | array[string] | No | References to trace IDs |
| `first_seen` | string (ISO 8601) | No | First discovery timestamp |
| `last_seen` | string (ISO 8601) | No | Last observation timestamp |
| `metadata` | object | No | Additional metadata |
| `endpoints` | array | No | Associated API endpoints |
| `tags` | array[string] | No | User-defined tags |

### Models Used Structure

```json
{
  "model_id": "openai/gpt-4",
  "usage_count": 3,
  "first_seen": "2026-03-10T14:25:10.000Z",
  "last_seen": "2026-03-10T14:29:45.000Z"
}
```

## Models

Represents AI models discovered across tools.

### Model Schema

```json
{
  "id": "openai/gpt-4",
  "name": "openai/gpt-4",
  "provider": "openai",
  "model_family": "gpt-4",
  "model_version": null,
  "modality": ["text"],
  "discovery_source": "opentelemetry",
  "usage": {
    "total_calls": 4,
    "unique_tools": 2,
    "tools_using": ["customer-service-bot", "code-assistant"],
    "first_seen": "2026-03-10T14:25:10.000Z",
    "last_seen": "2026-03-10T14:29:45.000Z"
  },
  "metadata": {
    "context_window": 8192,
    "max_output_tokens": 4096,
    "capabilities": ["function-calling", "streaming"]
  },
  "tags": ["production", "high-performance"]
}
```

### Model Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (e.g., "openai/gpt-4") |
| `name` | string | Yes | Display name |
| `provider` | string | No | Model provider |
| `model_family` | string | No | Model family/series |
| `model_version` | string | No | Specific version |
| `modality` | array[string] | No | text, image, audio, video, multimodal |
| `discovery_source` | string | Yes | Plugin/subsystem that discovered this model |
| `usage` | object | No | Usage statistics |
| `metadata` | object | No | Additional model metadata |
| `catalog_info` | object | No | Unity Catalog registration info |
| `tags` | array[string] | No | User-defined tags |

## Data Assets

Represents data assets from Unity Catalog, Azure AI Foundry, AWS, Google Cloud, or other sources.

### Data Asset Schema

```json
{
  "id": "main.default.users",
  "name": "users",
  "type": "table",
  "full_name": "main.default.users",
  "discovery_source": "databricks",
  "hierarchy": {
    "catalog_name": "main",
    "schema_name": "default"
  },
  "metadata": {
    "owner": "data-team",
    "comment": "User information table",
    "table_type": "MANAGED",
    "data_source_format": "DELTA",
    "row_count": 1000000,
    "size_bytes": 52428800
  }
}
```

### Data Asset Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `name` | string | Yes | Asset name |
| `type` | string | Yes | Asset type (see below) |
| `full_name` | string | No | Fully qualified name |
| `discovery_source` | string | Yes | Plugin/subsystem that discovered this asset |
| `hierarchy` | object | No | Catalog and schema location |
| `metadata` | object | No | Asset metadata |
| `schema_details` | object | No | Schema information for tables |
| `function_details` | object | No | Details for functions |
| `tags` | array[string] | No | User-defined tags |

### Asset Type Values

| Type | Source |
|------|--------|
| `catalog`, `schema`, `table`, `view`, `volume`, `function` | Databricks Unity Catalog |
| `model` | Multiple plugins |
| `endpoint`, `deployment`, `generative_model` | Google Cloud, AWS, Splunk |
| `foundry_resource`, `openai_resource`, `project` | Azure AI Foundry |
| `agent`, `tool`, `trace` | Azure AI Foundry, Databricks |
| `model_package`, `training_job` | AWS SageMaker |
| `custom_model`, `provisioned_throughput`, `invocation` | AWS Bedrock |

## MCP Servers

Represents discovered MCP (Model Context Protocol) servers.

### MCP Server Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `name` | string | Yes | Server name |
| `discovery_source` | string | Yes | config_file, environment, network, trace, manual |
| `transport` | string | No | stdio, http, sse, unknown |
| `endpoint` | string | No | HTTP/SSE endpoint URL |
| `command` | string | No | Command to start the server (stdio) |
| `args` | array[string] | No | Command-line arguments |
| `tools_provided` | array[string] | No | IDs of tools provided |
| `resources_provided` | array[string] | No | IDs of resources provided |
| `tools_count` | integer | No | Number of tools provided |
| `resources_count` | integer | No | Number of resources provided |
| `first_seen` | string (ISO 8601) | No | First discovery timestamp |
| `last_seen` | string (ISO 8601) | No | Last observation timestamp |
| `metadata` | object | No | Additional metadata |
| `tags` | array[string] | No | User-defined tags |

## MCP Tools

Represents tools provided by MCP servers.

### MCP Tool Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `name` | string | Yes | Tool name |
| `server_id` | string | Yes | ID of the MCP server providing this tool |
| `discovery_source` | string | No | introspection, trace, config, manual |
| `description` | string | No | Human-readable description |
| `schema` | object | No | JSON Schema for the tool's parameters |
| `usage` | object | No | Usage statistics (call_count, success_count, error_count) |
| `metadata` | object | No | Additional metadata |
| `tags` | array[string] | No | User-defined tags |

## MCP Resources

Represents resources provided by MCP servers.

### MCP Resource Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `uri` | string | Yes | Resource URI |
| `server_id` | string | Yes | ID of the MCP server providing this resource |
| `name` | string | No | Display name |
| `discovery_source` | string | No | introspection, trace, config, manual |
| `type` | string | No | file, database, api, document |
| `mime_type` | string | No | MIME type |
| `description` | string | No | Human-readable description |
| `usage` | object | No | Usage statistics (access_count, first/last_accessed) |
| `metadata` | object | No | Additional metadata |
| `tags` | array[string] | No | User-defined tags |

## Traces

Represents execution traces from OpenTelemetry or MLflow.

### Trace Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trace_id` | string | Yes | Unique trace identifier |
| `source` | string | Yes | opentelemetry or mlflow |
| `tool_id` | string | No | Reference to tool ID |
| `timestamp` | string (ISO 8601) | Yes | Trace timestamp |
| `duration_ms` | integer | No | Total duration in milliseconds |
| `status` | string | No | OK, ERROR, or UNSET |
| `models_called` | array[string] | No | Models called during trace |
| `spans` | array | No | Spans within the trace |
| `attributes` | object | No | Trace-level attributes |
| `mlflow_info` | object | No | MLflow-specific information |

### Span Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `span_id` | string | Yes | Unique span identifier |
| `parent_span_id` | string | No | Parent span for nesting |
| `name` | string | Yes | Span name |
| `kind` | string | No | INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER |
| `start_time` | string (ISO 8601) | Yes | Start timestamp |
| `end_time` | string (ISO 8601) | No | End timestamp |
| `duration_ms` | integer | No | Duration in milliseconds |
| `status` | string | No | OK, ERROR, or UNSET |
| `attributes` | object | No | Span attributes (key-value pairs) |
| `events` | array | No | Span events |

### Common Span Attributes

OpenTelemetry semantic conventions for LLM operations:

| Attribute | Type | Description |
|-----------|------|-------------|
| `http.url` | string | Request URL |
| `http.method` | string | HTTP method |
| `http.status_code` | integer | HTTP status code |
| `gen_ai.request.model` | string | Model requested |
| `gen_ai.response.model` | string | Model that responded |
| `gen_ai.request.temperature` | number | Temperature parameter |
| `gen_ai.request.max_tokens` | integer | Max tokens requested |
| `gen_ai.usage.input_tokens` | integer | Input tokens consumed |
| `gen_ai.usage.output_tokens` | integer | Output tokens generated |

## Usage Examples

### Basic Export

```python
from open_cite import OpenCiteClient

client = OpenCiteClient(enable_otel=True)

# Export to dictionary
data = client.export_to_json()

# Export to file
client.export_to_json(filepath="discoveries.json")
```

### Using the Schema Module

```python
from open_cite.schema import (
    OpenCiteExporter,
    ToolFormatter,
    ModelFormatter
)

# Create formatted tool
tool = ToolFormatter.format_tool(
    tool_id="my-tool",
    name="my-tool",
    discovery_source="opentelemetry",
    models_used=[
        {
            "model_id": "openai/gpt-4",
            "usage_count": 10
        }
    ]
)

# Export
exporter = OpenCiteExporter()
export_data = exporter.export_discovery(
    tools=[tool],
    models=[]
)

# Save
exporter.save_to_file(export_data, "export.json")
```

### Validation

```python
from open_cite.schema import OpenCiteExporter

# Load schema for validation
exporter = OpenCiteExporter(
    schema_path="schema/opencite-schema.json"
)

# Validate data
is_valid = exporter.validate(export_data)

if not is_valid:
    print("Validation failed!")
```

Note: Requires `jsonschema` package:
```bash
pip install jsonschema
```

### Parsing Model IDs

```python
from open_cite.schema import parse_model_id

info = parse_model_id("openai/gpt-4-turbo")
# Returns:
# {
#   "provider": "openai",
#   "model_family": "gpt-4",
#   "model_version": "turbo"
# }
```

## Integration

### Importing into Other Tools

The JSON format can be imported into:

- **Dashboards**: Grafana, Kibana, custom dashboards
- **Databases**: Store in MongoDB, PostgreSQL (JSONB)
- **Analytics**: Process with pandas, Spark
- **Governance tools**: Feed into compliance systems

### Example: Load in Python

```python
import json

with open("discoveries.json", "r") as f:
    data = json.load(f)

# Access tools
for tool in data["tools"]:
    print(f"Tool: {tool['name']}")
    print(f"  Models: {[m['model_id'] for m in tool['models_used']]}")
```

### Example: Load in pandas

```python
import pandas as pd

tools_df = pd.DataFrame(data["tools"])
models_df = pd.DataFrame(data["models"])

# Analyze
print(tools_df.groupby("provider").size())
```

## Related Documentation

- [OpenTelemetry Plugin Documentation](plugins/OPENTELEMETRY_PLUGIN.md)
- [Plugin Authoring Guide](PLUGINS.md)
- [REST API Documentation](REST_API.md)
- [Development Guide](DEVELOPMENT.md)
