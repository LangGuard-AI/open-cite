# OpenCITE

OpenCITE (**C**ataloging **I**ntelligent **T**ools in the **E**nterprise) is a Python library and application designed to facilitate the discovery and management of AI/ML Assets (including tools, models, and infrastructure) across multiple cloud platforms and protocols.

## Overview

OpenCITE provides a unified interface for discovering and cataloging AI/ML resources across different platforms. Whether you're managing models in Databricks, tracking AI agent usage via OpenTelemetry, discovering MCP servers, or working with Google Cloud's Vertex AI, OpenCITE brings everything together under one roof.

## Key Capabilities

- **Multi-Platform Discovery**: Automatic discovery of AI/ML resources across Databricks, Google Cloud, and custom infrastructure
- **Protocol Support**: Native support for OpenTelemetry, MCP (Model Context Protocol), and major cloud APIs
- **Trace Analysis**: Collect and analyze traces from AI agents, tools, and model invocations
- **Model Cataloging**: Track models, endpoints, deployments, and usage patterns
- **Infrastructure Discovery**: Find MCP servers, compute instances, and AI services via labels and port scanning
- **Unified Schema**: Export discoveries in a standardized JSON format for downstream processing

## Features by Plugin

### Databricks Plugin (Default)
- Unity Catalog integration for catalogs, schemas, tables, and volumes
- MLflow trace search and retrieval for observability
- Data lineage and metadata management
- Requires: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables

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

### MCP Plugin (Model Context Protocol)
- Trace-based MCP server discovery
- MCP tool and resource cataloging
- Usage pattern analysis
- Integration with OpenTelemetry traces
- Enabled by default when OpenTelemetry is active

## Architecture

Open Cite uses a plugin-based architecture that allows you to:
- Enable only the discovery sources you need
- Add custom plugins for proprietary systems
- Combine multiple discovery methods for comprehensive coverage
- Export unified results regardless of source

## Installation

```bash
pip install -e .
```

## Usage

### Databricks Plugin (Default)

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

### OpenTelemetry Plugin

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

For detailed OpenTelemetry plugin documentation, see [docs/OPENTELEMETRY_PLUGIN.md](docs/OPENTELEMETRY_PLUGIN.md).

### MCP Plugin

```python
from open_cite import OpenCiteClient

# Initialize client (MCP discovery enabled by default)
client = OpenCiteClient(enable_mcp=True)

# List discovered MCP servers
servers = client.list_mcp_servers()
for server in servers:
    print(f"Server: {server['name']} ({server['transport']})")

# List MCP tools
tools = client.list_mcp_tools()

# Verify discovery
status = client.verify_mcp_discovery()
print(f"Discovered {status['servers_discovered']} MCP servers")
```

For detailed MCP plugin documentation, see [docs/MCP_PLUGIN.md](docs/MCP_PLUGIN.md).

### Google Cloud Plugin

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

For detailed Google Cloud plugin documentation, see [docs/GOOGLE_CLOUD_PLUGIN.md](docs/GOOGLE_CLOUD_PLUGIN.md).

### Exporting to JSON

```python
from open_cite import OpenCiteClient

client = OpenCiteClient(enable_otel=True)

# ... collect some traces ...

# Export to JSON format (OpenCITE schema)
data = client.export_to_json()

# Or save directly to file
client.export_to_json(filepath="discoveries.json")
```

For schema documentation, see [docs/SCHEMA_DOCUMENTATION.md](docs/SCHEMA_DOCUMENTATION.md).

## Testing

Open Cite includes comprehensive integration tests for all plugins. See [tests/TESTING_QUICKSTART.md](tests/TESTING_QUICKSTART.md) for details on:
- Running the test suite
- Setting up test environments
- GCP test data automation
- CI/CD integration

## Documentation

- **[docs/](docs/)** - Plugin documentation and implementation guides
- **[tests/](tests/)** - Test suite documentation and setup guides

## Project Structure

```
open-cite/
├── src/open_cite/          # Main library code
│   ├── client.py           # Unified client interface
│   ├── plugins/            # Discovery plugins
│   │   ├── databricks.py   # Databricks/Unity Catalog plugin
│   │   ├── google_cloud.py # Google Cloud/Vertex AI plugin
│   │   ├── opentelemetry.py # OpenTelemetry plugin
│   │   └── mcp.py          # MCP plugin
│   └── schema.py           # Export schema definitions
├── tests/                  # Integration tests and test utilities
│   ├── integration/        # Integration test suite
│   ├── setup_gcp_test_data.py    # GCP test data automation
│   ├── cleanup_gcp_test_data.py  # GCP resource cleanup
│   └── *.md                # Test documentation
├── docs/                   # Documentation
└── README.md               # This file
```

## Contributing

Contributions are welcome! The plugin architecture makes it easy to add support for new platforms:

1. Create a new plugin in `src/open_cite/plugins/`
2. Implement the discovery interface
3. Add integration tests in `tests/integration/`
4. Update the client to expose your plugin's methods

## License

[Add your license information here]
