# Open-CITE Documentation

Open-CITE (**C**ataloging **I**ntelligent **T**ools in the **E**nterprise), pronounced like "Open-Sight", is a Python library, service, and application for discovering and managing AI/ML assets across multiple cloud platforms and protocols.

## Features by Plugin

### Databricks Plugin
- Unity Catalog integration for catalogs, schemas, tables, and volumes
- MLflow trace search and retrieval for observability
- AI Gateway usage monitoring and agent discovery
- Genie space and conversation discovery with table lineage
- Data lineage and metadata management

### Google Cloud Plugin
- Vertex AI model and endpoint discovery
- Generative AI model listing (Gemini, PaLM, etc.)
- Model deployment tracking
- MCP server discovery via:
  - Label-based discovery (tags on compute instances)
  - Port scanning for active MCP servers
- Requires: Google Cloud credentials and project configuration

### OpenTelemetry Plugin
- OTLP/HTTP receiver for collecting traces
- Automatic tool and model discovery from trace data
- Works with any LLM provider (OpenRouter, OpenAI, Anthropic, etc.)
- **OpenRouter Broadcast support**: Zero-code integration via OpenRouter's built-in trace broadcasting
- Real-time analytics on tool and model usage
- Requires: OTLP endpoint configuration

### Azure AI Foundry Plugin
- Foundry resource and project discovery via Azure Resource Manager
- OpenAI resource and deployment discovery
- Agent and tool discovery via Assistants API
- Trace discovery via Log Analytics (Application Insights)
- OTLP conversion for webhook forwarding of traces
- Requires: Azure credentials and subscription configuration

### AWS Plugins (Bedrock & SageMaker)
- AWS Bedrock model and endpoint discovery
- AWS SageMaker model and endpoint discovery
- Shared AWS auth mixin for credential management
- Requires: AWS credentials

### Zscaler Plugin
- ZIA DLP and NSS shadow MCP detection
- Requires: Zscaler credentials

### Splunk Plugin
- Splunk-based asset discovery
- Requires: Splunk credentials and configuration

### MCP Plugin (Model Context Protocol)
- Trace-based MCP server discovery
- MCP tool and resource cataloging
- Usage pattern analysis
- Integration with OpenTelemetry traces
- Enabled by default when OpenTelemetry is active

## Architecture

Open-CITE uses a plugin-based architecture that allows you to:
- Enable only the discovery sources you need
- Add custom plugins for proprietary systems
- Combine multiple discovery methods for comprehensive coverage
- Export unified results regardless of source

## Usage

### Web GUI

Open-CITE includes a web-based GUI for easy discovery and visualization:

```bash
# Start the GUI server
opencite gui          # or: python -m open_cite.gui.app
# Access at http://localhost:5000

# With debug mode
opencite gui --debug
```

**Features:**
- **Visual plugin configuration**: Select and configure plugins with a simple UI
- **Real-time discovery**: See assets appear automatically as traces arrive (WebSocket push, 3-second polling fallback)
- **OpenTelemetry integration**: Built-in OTLP receiver with ngrok support for remote traces
- **Export functionality**: Download discovered assets as JSON directly from the browser
- **Multi-plugin support**: Enable all plugins simultaneously

The GUI automatically stops any running discovery when you refresh the page, ensuring a clean state on each load.

### Headless API

Run Open-CITE as a REST API service (for Kubernetes, CI, or headless use):

```bash
opencite api          # or: python -m open_cite.api.app
# Access at http://0.0.0.0:8080
```

### Python API

#### Databricks Plugin

```python
from open_cite import OpenCiteClient

# Initialize client (uses DATABRICKS_HOST and DATABRICKS_TOKEN env vars)
client = OpenCiteClient()

# Verify connection
print(client.verify_connection())

# List catalogs
catalogs = client.list_catalogs()
for cat in catalogs:
    print(cat['name'])

    # List schemas in the first catalog
    schemas = client.list_schemas(cat['name'])
    for schema in schemas:
        print(f"  - {schema['name']}")

# Search for traces
traces = client.search_traces(max_results=5)
for trace in traces:
    print(f"Trace {trace['request_id']}: {trace['status']}")
```

#### OpenTelemetry Plugin

```python
from open_cite import OpenCiteClient

# Initialize client with OpenTelemetry plugin enabled
client = OpenCiteClient(enable_otel=True, otel_host="0.0.0.0", otel_port=4318)

# Verify OTLP receiver is running
status = client.verify_otel_connection()
print(f"OTLP Endpoint: {status['endpoint']}")
print(f"Traces received: {status['traces_received']}")

# List discovered tools from traces
tools = client.list_otel_tools()
for tool in tools:
    print(f"Tool: {tool['name']}")
    print(f"  Models: {tool['models']}")
    print(f"  Traces: {tool['trace_count']}")

# List all models discovered
models = client.list_otel_models()
for model in models:
    print(f"Model: {model['name']} - Used by {len(model['tools'])} tools")
```

For detailed OpenTelemetry plugin documentation, see [plugins/OPENTELEMETRY_PLUGIN.md](plugins/OPENTELEMETRY_PLUGIN.md).

#### Google Cloud Plugin

```python
from open_cite import OpenCiteClient

# Initialize client with Google Cloud plugin
client = OpenCiteClient(
    enable_google_cloud=True,
    gcp_project_id="my-project-id",
    gcp_location="us-central1"
)

# List Vertex AI models
models = client.list_gcp_models()
for model in models:
    print(f"Model: {model['name']} ({model['type']})")

# List endpoints
endpoints = client.list_gcp_endpoints()
for endpoint in endpoints:
    print(f"Endpoint: {endpoint['name']}")
    print(f"  Deployed models: {len(endpoint['deployed_models'])}")

# List generative AI models
gen_models = client.list_gcp_generative_models()
for model in gen_models:
    print(f"{model['name']}: {model['capabilities']}")

# Verify connection
status = client.verify_gcp_connection()
print(f"Connected to project: {status['project_id']}")
```

For detailed Google Cloud plugin documentation, see [plugins/GOOGLE_CLOUD_PLUGIN.md](plugins/GOOGLE_CLOUD_PLUGIN.md).

### Exporting to JSON

```python
from open_cite import OpenCiteClient

client = OpenCiteClient(enable_otel=True)

# ... collect some traces ...

# Export to JSON format (Open-CITE schema)
data = client.export_to_json()

# Or save directly to file
client.export_to_json(filepath="discoveries.json")
```

For schema documentation, see [SCHEMA_DOCUMENTATION.md](SCHEMA_DOCUMENTATION.md).

## Project Structure

```
open-cite/
├── src/open_cite/              # Main library code
│   ├── client.py               # Unified client interface
│   ├── core.py                 # BaseDiscoveryPlugin base class
│   ├── identifier.py           # Tool identification / mapping
│   ├── schema.py               # Export schema definitions
│   ├── plugins/                # Discovery plugins
│   │   ├── registry.py         # Auto-discovery and factory
│   │   ├── opentelemetry.py    # OTLP trace receiver
│   │   ├── databricks.py       # Databricks MLflow + Genie + Unity Catalog
│   │   ├── databricks_otlp_converter.py  # OTLP builders for Databricks data
│   │   ├── google_cloud.py     # Vertex AI + Compute Engine MCP
│   │   ├── azure_ai_foundry.py # Azure AI Foundry discovery
│   │   ├── splunk.py           # Splunk discovery
│   │   ├── zscaler.py          # ZIA DLP + NSS shadow MCP detection
│   │   ├── logs_adapter.py     # Logs adapter
│   │   └── aws/                # AWS plugins
│   │       ├── base.py         # Shared AWS auth mixin
│   │       ├── bedrock.py      # AWS Bedrock
│   │       └── sagemaker.py    # AWS SageMaker
│   ├── gui/                    # Web GUI application
│   │   ├── app.py              # Flask + SocketIO backend
│   │   ├── templates/          # HTML templates
│   │   └── static/             # Static assets
│   └── api/                    # Headless REST API
│       ├── app.py              # Flask API server
│       ├── config.py           # Environment-based configuration
│       ├── health.py           # /healthz and /readyz endpoints
│       ├── shutdown.py         # Graceful shutdown handler
│       └── persistence.py      # SQLite persistence manager
├── docs/                       # Documentation
├── website/                    # Project website (open-cite.org)
└── README.md                   # Quick start guide
```

## Guides

- [Deployment (Docker & Kubernetes)](DEPLOYMENT.md) -- running Open-CITE headless in containers
- [REST API Reference](REST_API.md) -- all headless API endpoints
- [Sending Traces to Open-CITE](SENDING_TRACES.md) -- configure Cloudflare AI Gateway, OpenRouter, and other OTLP sources

## Plugin Documentation

- [Creating a Plugin](PLUGINS.md) -- full plugin authoring guide
- [OpenTelemetry Plugin](plugins/OPENTELEMETRY_PLUGIN.md) -- OTLP receiver setup and trace format
- [Azure AI Foundry Plugin](plugins/AZURE_AI_FOUNDRY_PLUGIN.md) -- Azure discovery
- [AWS Plugins](plugins/AWS_PLUGINS.md) -- Bedrock and SageMaker discovery
- [Google Cloud Plugin](plugins/GOOGLE_CLOUD_PLUGIN.md) -- Vertex AI and Compute Engine discovery
- [Microsoft Fabric Plugin](plugins/MICROSOFT_FABRIC_PLUGIN.md) -- Microsoft Fabric discovery
- [Schema Documentation](SCHEMA_DOCUMENTATION.md) -- JSON export format
- [Development Guide](DEVELOPMENT.md) -- development setup, debugging, testing

## Contributing

Contributions are welcome! The plugin architecture makes it easy to add support for new platforms:

1. Create a new plugin in `src/open_cite/plugins/`
2. Implement the `BaseDiscoveryPlugin` interface
3. The plugin is auto-discovered at startup -- no other files need editing

See [PLUGINS.md](PLUGINS.md) for the full plugin authoring guide.
