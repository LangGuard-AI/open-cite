# Creating an OpenCITE Plugin

Plugins are the primary extension point for OpenCITE. Each plugin discovers AI assets from a specific source (Databricks, AWS, GCP, OpenTelemetry, etc.) and exposes them through a unified interface.

## Auto-Discovery

Plugins are auto-discovered at startup. Place a `.py` file in `src/open_cite/plugins/` containing a concrete `BaseDiscoveryPlugin` subclass, and the registry picks it up automatically. No other files need editing.

The registry (`plugins/registry.py`) walks the `open_cite.plugins` package, finds every non-abstract `BaseDiscoveryPlugin` subclass, and indexes it by `plugin_type`. The GUI, API, and client all use the registry to list, create, and manage plugin instances.

## Minimal Plugin

```python
# src/open_cite/plugins/my_source.py
from typing import List, Dict, Any, Set
from ..core import BaseDiscoveryPlugin

class MySourcePlugin(BaseDiscoveryPlugin):
    plugin_type = "my_source"

    @classmethod
    def plugin_metadata(cls):
        return {
            "name": "My Source",
            "description": "Discovers AI assets from My Source",
            "required_fields": {
                "api_key": {
                    "label": "API Key",
                    "default": "",
                    "required": True,
                    "type": "password",   # renders as password input in GUI
                },
                "region": {
                    "label": "Region",
                    "default": "us-east-1",
                    "required": False,
                },
            },
            "env_vars": ["MY_SOURCE_API_KEY"],
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        return cls(
            api_key=config.get("api_key"),
            region=config.get("region", "us-east-1"),
            instance_id=instance_id,
            display_name=display_name,
        )

    def __init__(self, api_key=None, region="us-east-1",
                 instance_id=None, display_name=None):
        super().__init__(instance_id=instance_id, display_name=display_name)
        self.api_key = api_key
        self.region = region

    @property
    def supported_asset_types(self) -> Set[str]:
        return {"model", "tool"}

    def verify_connection(self) -> Dict[str, Any]:
        # Check connectivity; return {"success": True} or {"success": False, "error": "..."}
        return {"success": True}

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        if asset_type == "model":
            return self._list_models()
        elif asset_type == "tool":
            return self._list_tools()
        return []

    def get_identification_attributes(self) -> List[str]:
        return ["my_source.model_id"]

    def _list_models(self):
        # Your discovery logic here
        return [{"id": "m1", "name": "my-model", "discovery_source": self.instance_id}]

    def _list_tools(self):
        return []
```

That's it. Start the GUI (`opencite gui`) and "My Source" appears as a configurable plugin.

## Required Interface

Every plugin must implement these abstract members from `BaseDiscoveryPlugin`:

| Member | Type | Purpose |
|--------|------|---------|
| `plugin_type` | class attribute | Unique string identifier (e.g. `"databricks"`) |
| `supported_asset_types` | `@property` | Set of asset type strings this plugin can discover |
| `verify_connection()` | method | Test connectivity, return `{"success": bool, ...}` |
| `list_assets(asset_type, **kwargs)` | method | Return list of asset dicts for the given type |
| `get_identification_attributes()` | method | Return attribute keys used for tool identification |

### Standard Asset Types

Use these when applicable so assets display correctly in the GUI:

`tool`, `model`, `agent`, `endpoint`, `catalog`, `schema`, `table`, `volume`, `function`, `deployment`, `generative_model`, `downstream_system`, `mcp_server`, `mcp_tool`, `mcp_resource`

## Classmethods

### `plugin_metadata()`

Returns a dict describing the plugin for the GUI and API:

```python
{
    "name": "Human-Readable Name",
    "description": "What this plugin discovers",
    "required_fields": {
        "field_name": {
            "label": "Display Label",
            "default": "default_value",
            "required": True,
            "type": "password",  # optional, omit for plain text
        }
    },
    "env_vars": ["ENV_VAR_NAME"],  # environment variables the plugin reads
}
```

### `from_config(config, instance_id, display_name, dependencies)`

Factory method called by the registry. Receives the config dict from the GUI/API, plus optional `instance_id`, `display_name`, and `dependencies` (e.g. `{"http_client": ...}`). Must return a new plugin instance.

## Lifecycle

### `start()`

Called when the user starts the plugin. Override to initialize receivers, connections, or background threads. Always call `super().start()` or set `self._status = "running"`.

```python
def start(self):
    self._status = "running"
    self._connect_to_source()
    logger.info(f"Started {self.instance_id}")
```

### `stop()`

Called when the user stops the plugin or the server shuts down. Clean up connections, threads, and background resources. The base implementation shuts down the webhook executor automatically, so always call `super().stop()` if you override.

```python
def stop(self):
    self._disconnect()
    super().stop()  # sets _status="stopped", shuts down webhook executor
```

## Configuration Serialization

### `get_config()`

Override to return the current config. Mask sensitive values:

```python
def get_config(self):
    return {
        "region": self.region,
        "api_key": "****" if self.api_key else None,
    }
```

### `to_dict()`

Returns full plugin metadata including `instance_id`, `plugin_type`, `status`, `supported_asset_types`, `config`, and `webhooks`. Generally no need to override.

## Data Change Notifications

When your plugin discovers new data asynchronously (e.g. a trace arrives), call `self.notify_data_changed()`. This triggers a WebSocket push to all connected GUI clients so they see updates in real time.

```python
def _on_new_trace(self, trace):
    self._store_trace(trace)
    self.notify_data_changed()  # push to GUI
```

The GUI sets the callback via `plugin.on_data_changed = lambda p: push_update(p)` when starting the plugin.

## Webhook Trace Forwarding

Plugins inherit built-in webhook support from `BaseDiscoveryPlugin`. Users subscribe webhook URLs via REST, and the plugin forwards raw OTLP JSON payloads to those URLs as traces are discovered.

### How It Works

1. User subscribes a URL:
   ```bash
   curl -X POST http://localhost:5000/api/instances/<id>/webhooks \
     -H 'Content-Type: application/json' \
     -d '{"url":"http://collector:4318/v1/traces"}'
   ```

2. The plugin converts discovered traces to OTLP JSON and calls `self._deliver_to_webhooks(otlp_payload)`.

3. Each subscribed URL receives an HTTP POST with the OTLP payload (fire-and-forget, 3 attempts with backoff).

### Inherited Methods

These are available on every plugin via `BaseDiscoveryPlugin`:

| Method | Description |
|--------|-------------|
| `subscribe_webhook(url)` | Add URL, returns `True` if newly added |
| `unsubscribe_webhook(url)` | Remove URL, returns `True` if found |
| `list_webhooks()` | Return list of subscribed URLs |
| `_deliver_to_webhooks(otlp_payload)` | Submit payload to all URLs via thread pool |

### Adding Webhook Forwarding to Your Plugin

If your plugin discovers trace-like data that should be forwardable, convert it to OTLP JSON and deliver:

```python
def _process_trace(self, raw_trace):
    # 1. Normal discovery logic
    self._extract_entities(raw_trace)

    # 2. Forward to webhooks (only runs if any are subscribed)
    if self._webhook_urls:
        otlp_payload = self._convert_to_otlp(raw_trace)
        self._deliver_to_webhooks(otlp_payload)
```

Guard with `if self._webhook_urls:` to avoid conversion overhead when no webhooks are subscribed. Use lazy imports for converter modules to keep startup fast.

### OTLP Payload Format

The payload must follow the [OTLP JSON Traces](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) structure:

```json
{
  "resourceSpans": [{
    "resource": {
      "attributes": [
        {"key": "service.name", "value": {"stringValue": "my-service"}}
      ]
    },
    "scopeSpans": [{
      "scope": {"name": "open_cite.plugins.databricks_otlp_converter"},
      "spans": [{
        "traceId": "abc123...",
        "spanId": "def456...",
        "name": "span-name",
        "kind": 1,
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000001000000000",
        "attributes": [
          {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4"}}
        ],
        "status": {}
      }]
    }]
  }]
}
```

### OTLP Converter Utilities

Each plugin keeps its own OTLP converters alongside its plugin file. For example, the Databricks plugin uses `plugins/databricks_otlp_converter.py` with helpers for building OTLP attribute dicts:

```python
from open_cite.plugins.databricks_otlp_converter import _make_attr, _make_attr_int

attrs = [
    _make_attr("gen_ai.request.model", "gpt-4"),
    _make_attr_int("gen_ai.usage.input_tokens", 150),
]
```

The Databricks converter includes:
- `mlflow_trace_to_otlp(trace, experiment_name)` -- MLflow Trace object to OTLP
- `genie_trace_to_otlp(trace_dict)` -- Genie message trace dict to OTLP
- `ai_gateway_usage_to_otlp(row)` -- AI Gateway usage table row to OTLP

When writing a new plugin, create your own converter module (e.g. `plugins/my_source_otlp_converter.py`) to keep source-specific code inside the plugin.

### Webhook REST Endpoints

These endpoints are available for all plugin instances:

**GUI** (`/api`):

| Method | Path | Body | Description |
|--------|------|------|-------------|
| GET | `/api/instances/<id>/webhooks` | -- | List subscribed URLs |
| POST | `/api/instances/<id>/webhooks` | `{"url": "..."}` | Subscribe a URL |
| DELETE | `/api/instances/<id>/webhooks` | `{"url": "..."}` | Unsubscribe a URL |

**Headless API** (`/api/v1`):

| Method | Path | Body | Description |
|--------|------|------|-------------|
| GET | `/api/v1/instances/<id>/webhooks` | -- | List subscribed URLs |
| POST | `/api/v1/instances/<id>/webhooks` | `{"url": "..."}` | Subscribe a URL |
| DELETE | `/api/v1/instances/<id>/webhooks` | `{"url": "..."}` | Unsubscribe a URL |

POST validates that the URL starts with `http://` or `https://`.

## GenAI Semantic Conventions

When building OTLP payloads for AI workloads, use these attribute keys for interoperability:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.request.model` | Model name (e.g. `gpt-4`) |
| `gen_ai.system` | Provider (e.g. `openai`, `databricks`) |
| `gen_ai.agent.name` | Agent name |
| `gen_ai.tool.name` | Tool name |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.prompt` | User prompt text |

## File Layout

```
src/open_cite/
  core.py                   # BaseDiscoveryPlugin (the base class)
  plugins/
    registry.py             # Auto-discovery and factory
    opentelemetry.py         # OTLP receiver plugin
    databricks.py            # Databricks + MLflow + Genie
    databricks_otlp_converter.py  # OTLP builders for Databricks data
    google_cloud.py          # Vertex AI + Compute Engine
    aws/
      base.py                # Shared AWS auth mixin
      bedrock.py             # AWS Bedrock
      sagemaker.py           # AWS SageMaker
    zscaler.py               # Zscaler ZIA + NSS
```

## Databricks Plugin Requirements

The Databricks plugin discovers assets from three subsystems. Each has its own permission requirements.

### Genie

Genie discovery queries Databricks system tables for table lineage and query history. The service principal or user running OpenCITE needs:

- **`USE SCHEMA`** on `system.access` — required for querying `system.access.table_lineage`
- **`USE SCHEMA`** on `system.query` — required for querying `system.query.history`
- **SQL warehouse access** — queries run via the configured SQL warehouse

Without these permissions, Genie spaces and conversations are still discovered, but table usage lineage will be unavailable. The log will show:

```
WARNING - Could not query Genie table usage: [INSUFFICIENT_PERMISSIONS]
  User does not have USE SCHEMA on Schema 'system.access'. SQLSTATE: 42501
```

Grant access in Unity Catalog:

```sql
GRANT USE SCHEMA ON SCHEMA system.access TO `<principal>`;
GRANT USE SCHEMA ON SCHEMA system.query TO `<principal>`;
GRANT SELECT ON TABLE system.access.table_lineage TO `<principal>`;
GRANT SELECT ON TABLE system.query.history TO `<principal>`;
```

### AI Gateway

Requires access to the AI Gateway usage system table (default `system.ai_gateway.usage`). Set via `OPENCITE_AI_GATEWAY_USAGE_TABLE`.

### MLflow

Requires access to MLflow experiment tracking APIs. The configured Databricks credentials must have at least read access to the experiments being monitored.

## Testing Your Plugin

```python
from open_cite.plugins.registry import create_plugin_instance

plugin = create_plugin_instance("my_source", {
    "api_key": "test-key",
    "region": "us-east-1",
})

# Verify connection
print(plugin.verify_connection())

# List assets
models = plugin.list_assets("model")
print(f"Found {len(models)} models")

# Test webhook forwarding
plugin.subscribe_webhook("http://localhost:9999/test")
print(plugin.list_webhooks())  # ["http://localhost:9999/test"]
plugin.unsubscribe_webhook("http://localhost:9999/test")

# Serialization
print(plugin.to_dict())
```
