# MCP Discovery Plugin for OpenCITE

## Overview

The MCP (Model Context Protocol) discovery plugin automatically discovers MCP servers through **OpenTelemetry trace analysis**. It detects when applications use MCP tools and resources by analyzing traces sent to the OpenTelemetry receiver.

**Key Principle**: The MCP plugin discovers infrastructure through observability, not configuration scanning. It reveals what MCP servers, tools, and resources are *actually being used* in production.

## Features

- ✅ **Trace-based discovery**: Discovers MCP usage from OpenTelemetry traces
- ✅ **Server detection**: Identifies MCP servers from trace attributes
- ✅ **Tool tracking**: Tracks which MCP tools are being called
- ✅ **Resource monitoring**: Monitors MCP resource access patterns
- ✅ **Usage statistics**: Call counts, success/error rates, access patterns
- ✅ **JSON export**: Full schema support for MCP entities
- ✅ **Thread-safe**: Lock-based synchronization for concurrent trace ingestion

## Installation

No additional dependencies required beyond OpenCITE core.

```bash
pip install -e .
```

## Quick Start

```python
from open_cite import OpenCiteClient

# Initialize with both OpenTelemetry and MCP plugins
client = OpenCiteClient(
    enable_otel=True,  # Required for trace ingestion
    enable_mcp=True    # Enabled by default
)

# As your application sends traces with MCP attributes,
# the plugin will discover servers, tools, and resources

# List discovered servers
servers = client.list_mcp_servers()
for server in servers:
    print(f"{server['name']}: {server['tools_count']} tools, {server['resources_count']} resources")

# List tools with usage statistics
tools = client.list_mcp_tools()
for tool in tools:
    print(f"{tool['name']}: {tool['usage']['call_count']} calls")

# Export to JSON
client.export_to_json(include_mcp=True, filepath="mcp_discoveries.json")
```

## How It Works

### Discovery Flow

```
┌─────────────────────────────────────────┐
│    Your Application                     │
│  (Using MCP tools/resources)            │
└─────────────────┬───────────────────────┘
                  │
                  │ OpenTelemetry traces with
                  │ mcp.* attributes
                  ▼
┌─────────────────────────────────────────┐
│  OpenTelemetry Plugin                   │
│  (OTLP/HTTP receiver on port 4318)      │
└─────────────────┬───────────────────────┘
                  │
                  │ _detect_mcp_usage()
                  ▼
┌─────────────────────────────────────────┐
│  MCP Plugin                             │
│  • register_server_from_trace()         │
│  • register_tool()                      │
│  • register_resource()                  │
└─────────────────────────────────────────┘
```

### Integration with OpenTelemetry

The MCP plugin is tightly integrated with the OpenTelemetry plugin:

1. **OpenTelemetry plugin** receives traces via OTLP/HTTP on port 4318
2. For each span, it calls `_detect_mcp_usage()` to check for MCP attributes
3. When MCP attributes are found, it calls the **MCP plugin** to register the entities
4. The **MCP plugin** stores and tracks MCP servers, tools, and resources

## Instrumenting Your Application

To enable MCP discovery, your application must send OpenTelemetry traces with MCP semantic attributes.

### Required Span Attributes

The plugin looks for these attributes in spans:

| Attribute | Purpose | Example |
|-----------|---------|---------|
| `mcp.server.name` or `mcp.server` | MCP server identifier | `"filesystem"` |
| `mcp.tool.name` or `mcp.tool` | Tool being called | `"read_file"` |
| `mcp.resource.uri` or `mcp.resource` | Resource URI | `"file:///path/to/file.txt"` |
| `mcp.server.version` | Server version (optional) | `"1.0.0"` |
| `mcp.server.protocol_version` | MCP protocol version (optional) | `"1.0"` |
| `mcp.tool.status` | Tool call status (optional) | `"success"` or `"error"` |
| `mcp.resource.type` | Resource type (optional) | `"file"` |
| `mcp.resource.mime_type` | Resource MIME type (optional) | `"text/plain"` |

### Example Instrumentation

#### Python with OpenTelemetry SDK

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OTLP exporter
tracer_provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces"
)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer(__name__)

# Instrument MCP tool call
def call_mcp_tool(server_name, tool_name, params):
    with tracer.start_as_current_span("mcp_tool_call") as span:
        # Set MCP attributes
        span.set_attribute("mcp.server.name", server_name)
        span.set_attribute("mcp.tool.name", tool_name)

        try:
            # Call the actual MCP tool
            result = mcp_client.call_tool(server_name, tool_name, params)

            span.set_attribute("mcp.tool.status", "success")
            return result
        except Exception as e:
            span.set_attribute("mcp.tool.status", "error")
            span.record_exception(e)
            raise

# Instrument MCP resource access
def read_mcp_resource(server_name, resource_uri):
    with tracer.start_as_current_span("mcp_resource_read") as span:
        # Set MCP attributes
        span.set_attribute("mcp.server.name", server_name)
        span.set_attribute("mcp.resource.uri", resource_uri)
        span.set_attribute("mcp.resource.type", "file")

        # Read the resource
        content = mcp_client.read_resource(server_name, resource_uri)
        return content
```

#### TypeScript/JavaScript with OpenTelemetry

```typescript
import { trace } from '@opentelemetry/api';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';

// Configure OTLP exporter
const provider = new NodeTracerProvider();
const exporter = new OTLPTraceExporter({
  url: 'http://localhost:4318/v1/traces'
});
provider.addSpanProcessor(new BatchSpanProcessor(exporter));
provider.register();

const tracer = trace.getTracer('my-app');

// Instrument MCP tool call
async function callMcpTool(serverName: string, toolName: string, params: any) {
  return tracer.startActiveSpan('mcp_tool_call', async (span) => {
    span.setAttribute('mcp.server.name', serverName);
    span.setAttribute('mcp.tool.name', toolName);

    try {
      const result = await mcpClient.callTool(serverName, toolName, params);
      span.setAttribute('mcp.tool.status', 'success');
      return result;
    } catch (error) {
      span.setAttribute('mcp.tool.status', 'error');
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  });
}
```

## API Reference

### Client Methods

#### `list_mcp_servers()`

List all MCP servers discovered from traces.

```python
servers = client.list_mcp_servers()

# Returns:
# [
#   {
#     "id": "filesystem",
#     "name": "filesystem",
#     "discovery_source": "trace",
#     "transport": "stdio",
#     "first_seen": "2025-12-08T10:30:00.123Z",
#     "last_seen": "2025-12-08T10:35:00.456Z",
#     "tools_count": 5,
#     "resources_count": 10,
#     "trace_count": 42,
#     "usage_stats": {
#       "tool_calls": 42,
#       "resource_accesses": 15
#     }
#   }
# ]
```

#### `list_mcp_tools(server_id=None)`

List MCP tools discovered from traces, optionally filtered by server.

```python
# All tools
tools = client.list_mcp_tools()

# Tools for a specific server
tools = client.list_mcp_tools(server_id="filesystem")

# Returns:
# [
#   {
#     "id": "filesystem-read_file",
#     "name": "read_file",
#     "server_id": "filesystem",
#     "discovery_source": "trace",
#     "first_used": "2025-12-08T10:30:00.123Z",
#     "last_used": "2025-12-08T10:35:00.456Z",
#     "usage": {
#       "call_count": 25,
#       "success_count": 24,
#       "error_count": 1
#     },
#     "trace_count": 25
#   }
# ]
```

#### `list_mcp_resources(server_id=None)`

List MCP resources discovered from traces, optionally filtered by server.

```python
# All resources
resources = client.list_mcp_resources()

# Resources for a specific server
resources = client.list_mcp_resources(server_id="filesystem")

# Returns:
# [
#   {
#     "id": "filesystem-5678",
#     "uri": "file:///path/to/file.txt",
#     "server_id": "filesystem",
#     "discovery_source": "trace",
#     "type": "file",
#     "mime_type": "text/plain",
#     "first_accessed": "2025-12-08T10:30:00.123Z",
#     "last_accessed": "2025-12-08T10:35:00.456Z",
#     "usage": {
#       "access_count": 10
#     },
#     "trace_count": 10
#   }
# ]
```

#### `verify_mcp_discovery()`

Check MCP discovery status.

```python
status = client.verify_mcp_discovery()

# Returns:
# {
#     "success": True,
#     "discovery_method": "trace_analysis",
#     "servers_discovered": 3,
#     "tools_discovered": 12,
#     "resources_discovered": 8
# }
```

### Plugin Methods

Direct access to MCP plugin for advanced use cases:

```python
mcp_plugin = client.get_plugin("mcp")

# Manually register a server (typically not needed)
mcp_plugin.register_server_from_trace(
    server_name="my-custom-server",
    trace_id="trace123",
    span_id="span456",
    attributes={
        "mcp.server.version": "1.0.0"
    }
)

# Manually register a tool
mcp_plugin.register_tool(
    server_id="my-custom-server",
    tool_name="custom_tool",
    trace_id="trace123",
    span_id="span789",
    tool_schema={
        "description": "My custom tool",
        "parameters": { ... }
    },
    status="success"
)

# Manually register a resource
mcp_plugin.register_resource(
    server_id="my-custom-server",
    resource_uri="custom://resource/path",
    trace_id="trace123",
    span_id="span012",
    resource_type="custom",
    mime_type="application/json"
)
```

## Transport Detection

The plugin attempts to detect the transport type from trace attributes:

| Transport | Detection Method |
|-----------|------------------|
| `http` | Presence of `http.url` or `mcp.server.endpoint` attribute |
| `stdio` | Default when no endpoint is specified |
| `unknown` | When transport cannot be determined |

## JSON Schema

MCP entities in exports follow the OpenCITE schema:

```json
{
  "version": "1.0.0",
  "metadata": {
    "generated_at": "2025-12-08T10:00:00Z",
    "plugins": [
      {"name": "mcp", "version": "1.0.0"}
    ]
  },
  "mcp_servers": [
    {
      "id": "filesystem",
      "name": "filesystem",
      "discovery_source": "trace",
      "transport": "stdio",
      "first_seen": "2025-12-08T10:30:00.123Z",
      "last_seen": "2025-12-08T10:35:00.456Z",
      "tools_count": 5,
      "resources_count": 10,
      "trace_count": 42,
      "usage_stats": {
        "tool_calls": 42,
        "resource_accesses": 15
      }
    }
  ],
  "mcp_tools": [
    {
      "id": "filesystem-read_file",
      "name": "read_file",
      "server_id": "filesystem",
      "discovery_source": "trace",
      "usage": {
        "call_count": 25,
        "success_count": 24,
        "error_count": 1
      }
    }
  ],
  "mcp_resources": [
    {
      "id": "filesystem-5678",
      "uri": "file:///path/to/file.txt",
      "server_id": "filesystem",
      "discovery_source": "trace",
      "type": "file"
    }
  ]
}
```

## Use Cases

### 1. MCP Usage Analytics

Track which MCP servers and tools are actually being used:

```python
client = OpenCiteClient(enable_otel=True, enable_mcp=True)

# After collecting traces...
servers = client.list_mcp_servers()

print("MCP Server Usage:")
for server in servers:
    stats = server.get('usage_stats', {})
    print(f"{server['name']}:")
    print(f"  Tool calls: {stats.get('tool_calls', 0)}")
    print(f"  Resource accesses: {stats.get('resource_accesses', 0)}")

    # Show tools for this server
    tools = client.list_mcp_tools(server_id=server['id'])
    for tool in tools:
        usage = tool['usage']
        success_rate = (usage['success_count'] / usage['call_count'] * 100) if usage['call_count'] > 0 else 0
        print(f"    {tool['name']}: {usage['call_count']} calls ({success_rate:.1f}% success)")
```

### 2. Production Observability

Monitor MCP infrastructure in production:

```python
# Continuous monitoring
def check_mcp_health():
    status = client.verify_mcp_discovery()

    if status['servers_discovered'] == 0:
        print("WARNING: No MCP servers detected in traces")
        return

    servers = client.list_mcp_servers()
    for server in servers:
        # Check when server was last seen
        last_seen = datetime.fromisoformat(server['last_seen'])
        age = datetime.utcnow() - last_seen

        if age.total_seconds() > 300:  # 5 minutes
            print(f"WARNING: Server {server['name']} not seen in 5+ minutes")
```

### 3. Tool Usage Tracking

Identify which MCP tools are most commonly used:

```python
tools = client.list_mcp_tools()

# Sort by usage
by_usage = sorted(tools, key=lambda t: t['usage']['call_count'], reverse=True)

print("Top 10 Most Used MCP Tools:")
for i, tool in enumerate(by_usage[:10], 1):
    print(f"{i}. {tool['name']}: {tool['usage']['call_count']} calls")
```

### 4. Error Rate Monitoring

Track error rates for MCP tool calls:

```python
tools = client.list_mcp_tools()

print("MCP Tools with Errors:")
for tool in tools:
    usage = tool['usage']
    if usage['error_count'] > 0:
        error_rate = usage['error_count'] / usage['call_count'] * 100
        print(f"{tool['name']}: {usage['error_count']} errors ({error_rate:.1f}% error rate)")
```

## Troubleshooting

### No servers discovered

**Cause**: No traces with MCP attributes have been received.

**Solutions**:
1. Verify OpenTelemetry plugin is enabled and receiving traces:
   ```python
   status = client.verify_otel_connection()
   print(f"Traces received: {status['traces_received']}")
   ```

2. Check that your application is sending traces with MCP attributes:
   ```python
   # Verify traces contain mcp.* attributes
   traces = client.list_otel_traces()
   for trace in traces:
       for span in trace.get('spans', []):
           attrs = {attr['key']: attr.get('value') for attr in span.get('attributes', [])}
           if any(k.startswith('mcp.') for k in attrs):
               print(f"Found MCP attributes in trace {trace['trace_id']}")
   ```

3. Ensure your application is instrumented correctly with MCP semantic attributes

### Tools/resources empty but servers discovered

**Cause**: Traces contain `mcp.server.name` but not `mcp.tool.name` or `mcp.resource.uri`.

**Solution**: Ensure your instrumentation includes all relevant MCP attributes:
```python
# Include tool name when calling tools
span.set_attribute("mcp.tool.name", tool_name)

# Include resource URI when accessing resources
span.set_attribute("mcp.resource.uri", resource_uri)
```

### Incorrect usage statistics

**Cause**: Tool status not being set correctly.

**Solution**: Always set `mcp.tool.status` attribute:
```python
try:
    result = call_tool()
    span.set_attribute("mcp.tool.status", "success")
except Exception:
    span.set_attribute("mcp.tool.status", "error")
    raise
```

## Best Practices

1. **Consistent naming**: Use consistent server names across your application
2. **Status tracking**: Always set `mcp.tool.status` to track success/error rates
3. **Resource types**: Include `mcp.resource.type` for better categorization
4. **Span names**: Use descriptive span names like `mcp_tool_call` or `mcp_resource_read`
5. **Error handling**: Record exceptions in spans for better debugging

## Limitations

- **Passive discovery**: Only discovers MCP usage from traces (no active probing)
- **Requires instrumentation**: Applications must send traces with MCP attributes
- **No schema introspection**: Cannot automatically discover tool schemas (unless sent in traces)
- **In-memory storage**: Discovery data is stored in memory (not persisted)

## Comparison with Config-Based Discovery

| Aspect | Trace-Based (Current) | Config-Based (Alternative) |
|--------|----------------------|----------------------------|
| **What it discovers** | Servers *actually being used* | Servers *configured* |
| **Requires** | OpenTelemetry traces | Config file access |
| **Provides** | Usage statistics | Static configuration |
| **Accuracy** | 100% (what's used is what's discovered) | May find unused servers |
| **Privacy** | Respects boundaries (no local scanning) | Requires file system access |
| **Real-time** | Yes (as traces arrive) | Requires periodic scanning |

## Related Documentation

- [OpenTelemetry Plugin](OPENTELEMETRY_PLUGIN.md) - Required for trace ingestion
- [Schema Documentation](SCHEMA_DOCUMENTATION.md) - JSON export format
- [MCP Protocol Specification](https://modelcontextprotocol.io) - MCP semantic attributes

## Examples

See the [examples](examples/) directory for complete examples:
- `examples/mcp_trace_example.py` - How to instrument MCP usage
- `examples/export_example.py` - Exporting MCP discoveries

## Support

For MCP plugin questions:
- Ensure OpenTelemetry plugin is enabled and receiving traces
- Verify your application sends traces with MCP semantic attributes
- Check the [OpenTelemetry documentation](OPENTELEMETRY_PLUGIN.md) for OTLP configuration
