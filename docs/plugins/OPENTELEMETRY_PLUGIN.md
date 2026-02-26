# OpenTelemetry Plugin for Open-CITE

The OpenTelemetry plugin enables automatic discovery of AI tools, models, agents, downstream systems, and MCP servers by receiving and analyzing OpenTelemetry traces.

## Overview

This plugin acts as an OTLP (OpenTelemetry Protocol) receiver that:
1. Receives traces via HTTP (OTLP/HTTP protocol)
2. Analyzes traces to detect AI model usage from any LLM provider
3. Discovers tools, models, agents, downstream systems, and MCP usage
4. Pushes updates to the GUI in real time via WebSocket

## Installation

No additional dependencies are required. The plugin uses only Python standard library components.

## Quick Start

```python
from open_cite import OpenCiteClient

# Initialize client with OpenTelemetry plugin enabled
client = OpenCiteClient(enable_otel=True, otel_host="0.0.0.0", otel_port=4318)

# Verify the receiver is running
status = client.verify_otel_connection()
print(f"OTLP Receiver: {status['endpoint']}")
print(f"Traces received: {status['traces_received']}")

# List discovered tools
tools = client.list_otel_tools()
for tool in tools:
    print(f"Tool: {tool['name']}")
    print(f"  Models: {tool['models']}")
    print(f"  Traces: {tool['trace_count']}")

# List all models discovered
models = client.list_otel_models()
for model in models:
    print(f"Model: {model['name']}")
    print(f"  Used by: {model['tools']}")
    print(f"  Usage count: {model['usage_count']}")
```

## Configuration

### Client Initialization

```python
client = OpenCiteClient(
    enable_otel=True,        # Enable the OpenTelemetry plugin
    otel_host="0.0.0.0",     # Bind to all interfaces
    otel_port=4318           # Standard OTLP/HTTP port
)
```

### Manual Plugin Registration

```python
from open_cite import OpenCiteClient
from open_cite.plugins.opentelemetry import OpenTelemetryPlugin

client = OpenCiteClient()
otel_plugin = OpenTelemetryPlugin(host="localhost", port=4318)
client.register_plugin(otel_plugin)
otel_plugin.start_receiver()
```

## Sending Traces to the Plugin

### Configure Your Application

Configure your application to send OTLP traces to the receiver:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Set up the tracer
resource = Resource(attributes={
    "service.name": "my-ai-tool"
})

tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# Configure OTLP exporter to send to Open Cite
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces",
    headers={}
)

tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Get a tracer
tracer = trace.get_tracer(__name__)
```

### Using OpenRouter's Broadcast Feature

**OpenRouter has built-in OpenTelemetry trace broadcasting** - you don't need to instrument your code!

If you're using OpenRouter, you can enable their Broadcast feature to automatically send traces to OpenCite:

1. **Enable Broadcast in OpenRouter Settings**
   - Go to https://openrouter.ai/settings/broadcast
   - Add "OTel Collector" as a destination
   - Configure the endpoint URL to point to your OpenCite instance:
     - **Same machine**: `http://localhost:4318/v1/traces`
     - **Remote machine**: `http://YOUR_IP:4318/v1/traces` (see GUI for your IP)

2. **That's it!** OpenRouter will automatically send traces for all your API calls to OpenCite.

**Benefits of using OpenRouter Broadcast:**
- ✅ Zero code changes required
- ✅ Automatic trace generation for all OpenRouter calls
- ✅ Works with any OpenRouter client (OpenAI SDK, curl, etc.)
- ✅ Centralized configuration in OpenRouter dashboard

For more details, see [OpenRouter's Broadcast documentation](https://openrouter.ai/docs/guides/features/broadcast/overview).

### Instrument OpenRouter Calls (Manual Alternative)

```python
import requests

def call_openrouter(prompt, model="openai/gpt-4"):
    with tracer.start_as_current_span("openrouter_call") as span:
        # Add attributes for detection
        span.set_attribute("http.url", "https://openrouter.ai/api/v1/chat/completions")
        span.set_attribute("http.host", "openrouter.ai")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("llm.model", model)

        # Make the actual API call
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            headers={
                "Authorization": f"Bearer {YOUR_API_KEY}"
            }
        )

        return response.json()
```

### Using OpenAI SDK with OpenRouter

```python
from openai import OpenAI

# Point OpenAI SDK to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key"
)

def call_model(prompt, model="openai/gpt-4"):
    with tracer.start_as_current_span("openrouter_call") as span:
        span.set_attribute("http.host", "openrouter.ai")
        span.set_attribute("gen_ai.request.model", model)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
```

## API Reference

### Client Methods

#### `list_otel_tools()`
List all discovered tools that use OpenRouter models.

```python
tools = client.list_otel_tools()
# Returns: [
#     {
#         "name": "my-ai-tool",
#         "models": ["openai/gpt-4", "anthropic/claude-3-opus"],
#         "trace_count": 42,
#         "metadata": {
#             "last_seen": "2025-12-08T12:34:56",
#             "url": "https://openrouter.ai/api/v1/chat/completions"
#         }
#     }
# ]
```

#### `list_otel_models()`
List all models discovered across all tools.

```python
models = client.list_otel_models()
# Returns: [
#     {
#         "name": "openai/gpt-4",
#         "tools": ["my-ai-tool", "another-tool"],
#         "usage_count": 150
#     }
# ]
```

#### `list_otel_traces(trace_id=None)`
List all traces or get a specific trace by ID.

```python
# List all traces
traces = client.list_otel_traces()

# Get specific trace
trace = client.list_otel_traces(trace_id="abc123")
```

#### `verify_otel_connection()`
Verify the OTLP receiver is running and get statistics.

```python
status = client.verify_otel_connection()
# Returns: {
#     "success": True,
#     "receiver_running": True,
#     "host": "0.0.0.0",
#     "port": 4318,
#     "endpoint": "http://0.0.0.0:4318/v1/traces",
#     "traces_received": 1234,
#     "tools_discovered": 5
# }
```

#### `start_otel_receiver()` / `stop_otel_receiver()`
Manually start or stop the OTLP receiver.

```python
client.start_otel_receiver()
# Receiver is now listening for traces

client.stop_otel_receiver()
# Receiver has been stopped
```

## Detection Logic

The plugin analyzes span attributes to discover AI usage from any LLM provider:

### Model Detection
The plugin looks for a model name in (checked in order):
- `gen_ai.request.model`
- `gen_ai.response.model`
- `llm.model`
- `model`

### Provider Detection
Provider is extracted from:
- `gen_ai.system` (e.g., `"openai"`, `"anthropic"`)
- `gen_ai.provider.name`
- Falls back to parsing the model name (e.g., `"openai/gpt-4"` → `"openai"`)

### Tool/Service Name
Extracted from (in priority order):
1. `gen_ai.tool.name` or `tool.name`
2. Span name when the span has `gen_ai.tool.call.id`
3. `service.name` or `app.name` (but not when it equals the agent name)
4. Span name as fallback

### Agent Detection
Agents are detected from:
- `gen_ai.agent.name`, `agent_name`, or `agent.name`

### Downstream System Detection
HTTP spans to external hosts (not LLM providers) are tracked as downstream systems.

### MCP Detection
MCP servers, tools, and resources are detected from `mcp.*` attributes. See [MCP Discovery](MCP_PLUGIN.md).

### Recommended Span Attributes

For best results, include these attributes in your spans:

```python
span.set_attribute("gen_ai.request.model", "gpt-4")
span.set_attribute("gen_ai.system", "openai")
```

And in your resource:

```python
Resource(attributes={
    "service.name": "my-ai-tool"
})
```

## OTLP Format Support

Currently, the plugin supports:
- **JSON format**: `Content-Type: application/json` (OTLP/JSON)
- **HTTP protocol**: POST to `/v1/traces`

Future versions may add:
- Protobuf format support
- gRPC protocol support

## Architecture

```
┌─────────────────┐
│  Your AI Tool   │
│  (with OTEL)    │
└────────┬────────┘
         │ OTLP/HTTP
         │ (traces)
         ▼
┌─────────────────────────┐
│  Open Cite              │
│  ┌──────────────────┐   │
│  │ OTLP Receiver    │   │
│  │ (port 4318)      │   │
│  └────────┬─────────┘   │
│           │             │
│           ▼             │
│  ┌──────────────────┐   │
│  │ Trace Analyzer   │   │
│  │ (OpenRouter      │   │
│  │  detection)      │   │
│  └────────┬─────────┘   │
│           │             │
│           ▼             │
│  ┌──────────────────┐   │
│  │ Tool Discovery   │   │
│  │ Database         │   │
│  └──────────────────┘   │
└─────────────────────────┘
```

## Use Cases

### 1. Model Usage Auditing
Track which models are being used across your organization:

```python
models = client.list_otel_models()
for model in models:
    print(f"{model['name']}: {model['usage_count']} calls")
```

### 2. Tool Discovery
Discover which tools are using AI models:

```python
tools = client.list_otel_tools()
for tool in tools:
    print(f"{tool['name']} uses {len(tool['models'])} different models")
```

### 3. Cost Optimization
Identify high-usage tools for cost optimization:

```python
tools = client.list_otel_tools()
high_usage = [t for t in tools if t['trace_count'] > 1000]
print(f"High-usage tools: {[t['name'] for t in high_usage]}")
```

### 4. Model Standardization
Find tools that might benefit from model consolidation:

```python
models = client.list_otel_models()
underused = [m for m in models if len(m['tools']) == 1 and m['usage_count'] < 10]
print(f"Underutilized models: {[m['name'] for m in underused]}")
```

## Troubleshooting

### No traces being received

1. Check the receiver is running:
   ```python
   status = client.verify_otel_connection()
   print(status['receiver_running'])
   ```

2. Verify your application is sending to the correct endpoint:
   ```python
   print(f"Send traces to: {status['endpoint']}")
   ```

3. Check firewall rules allow traffic on port 4318

### Tools not being discovered

1. Verify your spans include the required attributes:
   - `http.url` or `http.host` containing `openrouter.ai`
   - `gen_ai.request.model` or `llm.model` or `model`

2. Check the raw traces:
   ```python
   traces = client.list_otel_traces()
   print(traces)
   ```

### Memory Usage

The plugin stores traces in memory. For production use with high trace volume:
- Implement trace rotation/cleanup
- Consider adding persistent storage
- Monitor memory usage

## Limitations

- **In-memory storage**: Traces are stored in memory and lost on restart
- **JSON only**: Currently only supports OTLP/JSON format
- **No authentication**: The receiver accepts all incoming traces

## Future Enhancements

- [ ] Persistent storage (SQLite/PostgreSQL)
- [ ] Protobuf format support
- [ ] Authentication/authorization
- [ ] Trace retention policies
- [ ] Metrics and alerting
