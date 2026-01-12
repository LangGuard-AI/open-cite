# OpenCITE JSON Schema Documentation

## Overview

The OpenCITE JSON schema provides a standardized format for storing and exchanging discovery data about AI tools, models, data assets, and traces. This schema enables:

- **Interoperability**: Share discovery data between systems
- **Persistence**: Store discoveries for historical analysis
- **Integration**: Feed discovery data into other tools and dashboards
- **Compliance**: Track AI usage for governance and auditing

## Schema Version

Current version: **1.0.0**

Schema location: `schema/opencite-schema.json`

## Root Structure

```json
{
  "version": "1.0.0",
  "metadata": { },
  "tools": [ ],
  "models": [ ],
  "data_assets": [ ],
  "traces": [ ]
}
```

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Schema version (semver format) |
| `metadata` | object | Yes | Metadata about the discovery session |
| `tools` | array | No | Discovered AI tools and applications |
| `models` | array | No | Discovered AI models |
| `data_assets` | array | No | Data assets from catalogs |
| `traces` | array | No | Collected traces |

## Metadata

Contains information about when and how the data was discovered.

```json
{
  "metadata": {
    "generated_at": "2025-12-08T14:30:00.000Z",
    "generated_by": "opencite-client",
    "discovery_window": {
      "start": "2025-12-08T14:00:00.000Z",
      "end": "2025-12-08T14:30:00.000Z"
    },
    "plugins": [
      {
        "name": "opentelemetry",
        "version": "1.0.0"
      }
    ],
    "statistics": {
      "total_tools": 3,
      "total_models": 3,
      "by_provider": {
        "openai": 2,
        "anthropic": 1
      }
    }
  }
}
```

### Metadata Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `generated_at` | string (ISO 8601) | Yes | When the export was created |
| `generated_by` | string | Yes | Tool that generated the export |
| `discovery_window` | object | No | Time range of discovery |
| `plugins` | array | No | Plugins that contributed data |
| `statistics` | object | No | Summary statistics |

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
      "first_seen": "2025-12-08T14:25:10.000Z",
      "last_seen": "2025-12-08T14:29:45.000Z"
    }
  ],
  "provider": "openrouter",
  "trace_count": 3,
  "traces": ["trace-001", "trace-002"],
  "first_seen": "2025-12-08T14:25:10.000Z",
  "last_seen": "2025-12-08T14:29:45.000Z",
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
| `type` | enum | No | Type: application, service, library, agent, chatbot, unknown |
| `description` | string | No | Human-readable description |
| `discovery_source` | enum | Yes | opentelemetry, databricks, or manual |
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
  "first_seen": "2025-12-08T14:25:10.000Z",
  "last_seen": "2025-12-08T14:29:45.000Z"
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
    "first_seen": "2025-12-08T14:25:10.000Z",
    "last_seen": "2025-12-08T14:29:45.000Z"
  },
  "metadata": {
    "context_window": 8192,
    "max_output_tokens": 4096,
    "cost_per_1k_input_tokens": 0.03,
    "cost_per_1k_output_tokens": 0.06,
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
| `modality` | array[enum] | No | text, image, audio, video, multimodal |
| `discovery_source` | enum | Yes | opentelemetry, databricks, or manual |
| `usage` | object | No | Usage statistics |
| `metadata` | object | No | Additional model metadata |
| `catalog_info` | object | No | Unity Catalog registration info |
| `tags` | array[string] | No | User-defined tags |

### Usage Structure

```json
{
  "total_calls": 4,
  "unique_tools": 2,
  "tools_using": ["tool-1", "tool-2"],
  "first_seen": "2025-12-08T14:25:10.000Z",
  "last_seen": "2025-12-08T14:29:45.000Z"
}
```

## Data Assets

Represents data assets from Unity Catalog or other sources.

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
  },
  "schema_details": {
    "columns": [
      {
        "name": "user_id",
        "type": "bigint",
        "nullable": false,
        "comment": "Unique user identifier"
      },
      {
        "name": "email",
        "type": "string",
        "nullable": false
      }
    ]
  }
}
```

### Data Asset Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `name` | string | Yes | Asset name |
| `type` | enum | Yes | catalog, schema, table, view, volume, function, model |
| `full_name` | string | No | Fully qualified name |
| `discovery_source` | enum | Yes | databricks or manual |
| `hierarchy` | object | No | Catalog and schema location |
| `metadata` | object | No | Asset metadata |
| `schema_details` | object | No | Schema information for tables |
| `function_details` | object | No | Details for functions |
| `tags` | array[string] | No | User-defined tags |

## Traces

Represents execution traces from OpenTelemetry or MLflow.

### Trace Schema

```json
{
  "trace_id": "trace-001",
  "source": "opentelemetry",
  "tool_id": "customer-service-bot",
  "timestamp": "2025-12-08T14:25:10.000Z",
  "duration_ms": 1250,
  "status": "OK",
  "models_called": ["openai/gpt-4"],
  "spans": [
    {
      "span_id": "span-001-01",
      "name": "handle_customer_query",
      "kind": "INTERNAL",
      "start_time": "2025-12-08T14:25:10.000Z",
      "end_time": "2025-12-08T14:25:11.250Z",
      "duration_ms": 1250,
      "status": "OK",
      "attributes": {
        "http.url": "https://openrouter.ai/api/v1/chat/completions",
        "gen_ai.request.model": "openai/gpt-4",
        "gen_ai.usage.input_tokens": 125,
        "gen_ai.usage.output_tokens": 450
      }
    }
  ]
}
```

### Trace Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trace_id` | string | Yes | Unique trace identifier |
| `source` | enum | Yes | opentelemetry or mlflow |
| `tool_id` | string | No | Reference to tool ID |
| `timestamp` | string (ISO 8601) | Yes | Trace timestamp |
| `duration_ms` | integer | No | Total duration in milliseconds |
| `status` | enum | No | OK, ERROR, or UNSET |
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
| `kind` | enum | No | INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER |
| `start_time` | string (ISO 8601) | Yes | Start timestamp |
| `end_time` | string (ISO 8601) | No | End timestamp |
| `duration_ms` | integer | No | Duration in milliseconds |
| `status` | enum | No | OK, ERROR, or UNSET |
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

## Best Practices

### 1. Always Include Timestamps

Use ISO 8601 format with UTC timezone:
```python
from datetime import datetime
timestamp = datetime.utcnow().isoformat() + "Z"
```

### 2. Use Consistent IDs

- **Tools**: Use service name or unique identifier
- **Models**: Use provider/model format (e.g., "openai/gpt-4")
- **Traces**: Use hexadecimal trace IDs from OTLP

### 3. Add Metadata

Include ownership, environment, and tags for better organization:
```json
{
  "metadata": {
    "owner": "team-name",
    "environment": "production",
    "version": "1.2.3"
  },
  "tags": ["critical", "customer-facing"]
}
```

### 4. Track Usage Statistics

Always include first_seen and last_seen timestamps for temporal analysis.

### 5. Validate Before Sharing

Use the schema validation to ensure data quality:
```python
exporter.validate(data)
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

## Schema Evolution

Future versions may add:

- **v1.1.0**: Cost tracking fields
- **v1.2.0**: Performance metrics
- **v2.0.0**: Breaking changes (backward incompatible)

Version history will be maintained in the schema file.

## Support

For schema questions or suggestions:
- File an issue on GitHub
- See example exports in `examples/example_export.json`
- Review the JSON Schema: `schema/opencite-schema.json`

## Related Documentation

- [OpenTelemetry Plugin Documentation](OPENTELEMETRY_PLUGIN.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Quick Start Guide](QUICKSTART_OTEL.md)
