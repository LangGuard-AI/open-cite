# Open-CITE

Open-CITE (**C**ataloging **I**ntelligent **T**ools in the **E**nterprise), pronounced like "Open-Sight", is a Python library, service, and application designed to facilitate the discovery and management of AI/ML Assets (including tools, models, and infrastructure) across multiple cloud platforms and protocols.

## Overview

Open-CITE provides a unified interface for discovering and cataloging AI/ML resources across different platforms. Whether you're managing models in Databricks, tracking AI agent usage via OpenTelemetry, discovering resources in Azure AI Foundry, or working with Google Cloud's Vertex AI, Open-CITE brings everything together under one roof.

## Key Capabilities

- **Multi-Platform Discovery**: Automatic discovery of AI/ML resources across Databricks, AWS, Azure, Google Cloud, and more
- **Protocol Support**: Native support for OpenTelemetry, MCP (Model Context Protocol), and major cloud APIs
- **Trace Analysis**: Collect and analyze traces from AI agents, tools, and model invocations
- **Model Cataloging**: Track models, endpoints, deployments, and usage patterns
- **Infrastructure Discovery**: Find MCP servers, compute instances, and AI services
- **Unified Schema**: Export discoveries in a standardized JSON format for downstream processing
- **Runs as a library or service**: Open-CITE can be run with or without the GUI, in a docker container or Kubernetes to provide a headless AI asset discovery service, or leveraged as a library in your own Python application

## Powered by LangGuard

Open-CITE is provided to the community by the team at [LangGuard.AI](https://langguard.ai?utm=Open-CITE), home of the AI Control Plane for enterprise AI governance and monitoring. LangGuard leverages Open-CITE for internal AI Asset discovery.

## Quick Start

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Start the GUI
opencite gui
# Access at http://localhost:5000

# Or start the headless API
opencite api
# Access at http://0.0.0.0:8080
```

## Documentation

Full documentation is available in the [docs/](docs/) folder:

- **[docs/README.md](docs/README.md)** -- Full usage guide, Python API examples, and project structure
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** -- Docker and Kubernetes deployment
- **[docs/REST_API.md](docs/REST_API.md)** -- REST API reference
- **[docs/SENDING_TRACES.md](docs/SENDING_TRACES.md)** -- Configure Cloudflare AI Gateway, OpenRouter, and other sources to send traces
- **[docs/PLUGINS.md](docs/PLUGINS.md)** -- Plugin authoring guide
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** -- Development setup and debugging
- **[docs/SCHEMA_DOCUMENTATION.md](docs/SCHEMA_DOCUMENTATION.md)** -- JSON export schema

### Plugin Documentation

- [OpenTelemetry Plugin](docs/plugins/OPENTELEMETRY_PLUGIN.md)
- [Azure AI Foundry Plugin](docs/plugins/AZURE_AI_FOUNDRY_PLUGIN.md)
- [AWS Plugins (Bedrock & SageMaker)](docs/plugins/AWS_PLUGINS.md)
- [Google Cloud Plugin](docs/plugins/GOOGLE_CLOUD_PLUGIN.md)
- [Microsoft Fabric Plugin](docs/plugins/MICROSOFT_FABRIC_PLUGIN.md)

## Contributing

Contributions are welcome! The plugin architecture makes it easy to add support for new platforms. See [docs/PLUGINS.md](docs/PLUGINS.md) for the full plugin authoring guide.
