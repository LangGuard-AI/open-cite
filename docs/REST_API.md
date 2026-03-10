# Open-CITE REST API Reference

The headless REST API provides programmatic access to Open-CITE discovery, plugin management, and trace ingestion. Start the API server with:

```bash
opencite api          # default: http://0.0.0.0:8080
```

All endpoints return JSON (`application/json`) unless otherwise noted.

## Health Checks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe -- always returns 200 |
| GET | `/readyz` | Readiness probe -- checks client initialized and OTLP receiver running |

**`/readyz` response (200 or 503):**
```json
{
  "status": "ready",
  "checks": {
    "client_initialized": true,
    "otlp_receiver": true
  }
}
```

---

## OTLP Trace Ingestion

These endpoints accept OpenTelemetry traces and logs for asset discovery.

| Method | Path | Content-Type | Description |
|--------|------|--------------|-------------|
| POST | `/v1/traces` | `application/json` or `application/x-protobuf` | Ingest OTLP trace payloads |
| POST | `/v1/logs` | `application/json` or `application/x-protobuf` | Ingest OTLP log payloads (converted to synthetic traces) |

**Request:** Standard OTLP ExportTraceServiceRequest / ExportLogsServiceRequest body.

**Response (200):**
```json
{"status": "success"}
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| 415 | Unsupported Content-Type |
| 500 | Processing error |
| 503 | Embedded OTLP receiver not configured |

**Header forwarding:** Inbound headers are forwarded to subscribed webhooks. Hop-by-hop headers (`content-length`, `transfer-encoding`, `connection`, `keep-alive`) are filtered. The `host` header is renamed to `OTEL-HOST`.

---

## Discovery Status

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/status` | Get current discovery status |

**Response:**
```json
{
  "running": true,
  "plugins_enabled": ["opentelemetry", "databricks"],
  "last_updated": "2026-03-10T14:30:00Z",
  "error": null,
  "current_status": "Discovering...",
  "progress": [
    {"step": "otel_start", "message": "OTLP receiver started", "status": "success"}
  ]
}
```

---

## Plugin Configuration (Legacy)

These endpoints configure plugins in bulk. For per-instance management, use the [Instance Management](#instance-management) endpoints below.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/plugins` | List available plugin types and requirements |
| POST | `/api/v1/plugins/configure` | Configure and start plugins |
| POST | `/api/v1/stop` | Stop all discovery and clean up |

### POST `/api/v1/plugins/configure`

**Request:**
```json
{
  "plugins": [
    {
      "name": "opentelemetry",
      "config": {"host": "0.0.0.0", "port": 4318}
    }
  ]
}
```

**Response (200):**
```json
{"success": true, "plugins_enabled": ["opentelemetry"]}
```

---

## Assets

| Method | Path | Query Parameters | Description |
|--------|------|------------------|-------------|
| GET | `/api/v1/assets` | `type` (optional) | Get all discovered assets (cached 30s) |
| GET | `/api/v1/tools` | -- | List discovered tools |
| GET | `/api/v1/models` | -- | List discovered models |
| GET | `/api/v1/agents` | -- | List discovered agents |
| GET | `/api/v1/downstream` | -- | List discovered downstream systems |
| GET | `/api/v1/mcp/servers` | -- | List MCP servers |
| GET | `/api/v1/mcp/tools` | `server_id` (optional) | List MCP tools |
| GET | `/api/v1/lineage` | `source_id` (optional) | List lineage relationships |

### GET `/api/v1/assets`

**Query parameters:**
- `type` -- Filter by asset type: `all`, `tool`, `model`, `agent`, `downstream_system`, `mcp_server`, `mcp_tool`, `mcp_resource`

**Response:**
```json
{
  "assets": {
    "tools": [...],
    "models": [...],
    "agents": [...],
    "downstream_systems": [...],
    "mcp_servers": [...],
    "mcp_tools": [...],
    "mcp_resources": [...],
    "data_assets": [...]
  },
  "totals": {
    "tools": 5,
    "models": 3,
    "agents": 2
  },
  "timestamp": "2026-03-10T14:30:00Z",
  "discovering": true
}
```

### GET `/api/v1/tools` (and similar)

**Response:**
```json
{
  "tools": [...],
  "count": 5
}
```

### GET `/api/v1/lineage`

**Query parameters:**
- `source_id` -- Filter relationships by source asset ID

**Response:**
```json
{
  "relationships": [...],
  "count": 12
}
```

---

## Export

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/export` | Export discovered data in Open-CITE JSON format |

**Request:**
```json
{
  "plugins": ["opentelemetry", "databricks"]
}
```

**Response:** Complete export in [Open-CITE JSON schema](SCHEMA_DOCUMENTATION.md) format.

---

## Tool Mapping

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/map-tool` | Save a tool identification mapping |

**Request:**
```json
{
  "plugin_name": "opentelemetry",
  "attributes": {
    "service.name": "my-app"
  },
  "identity": {
    "tool_name": "My Application",
    "tool_type": "application"
  },
  "match_type": "all"
}
```

**Response (200):**
```json
{"success": true}
```

---

## Instance Management

Manage individual plugin instances (create, configure, start, stop, delete).

### Plugin Types

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/plugin-types` | List available plugin types and metadata |

**Response:**
```json
{
  "plugin_types": {
    "opentelemetry": {
      "name": "OpenTelemetry",
      "description": "OTLP trace receiver",
      "required_fields": {...},
      "env_vars": [...]
    },
    "databricks": {...}
  }
}
```

### Instances

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/instances` | List all plugin instances |
| POST | `/api/v1/instances` | Create a new plugin instance |
| GET | `/api/v1/instances/{id}` | Get instance details |
| PUT | `/api/v1/instances/{id}` | Update instance configuration |
| DELETE | `/api/v1/instances/{id}` | Delete an instance |

#### GET `/api/v1/instances`

**Query parameters:**
- `plugin_type` -- Filter by plugin type

**Response:**
```json
{
  "instances": [
    {
      "instance_id": "abc-123",
      "plugin_type": "opentelemetry",
      "display_name": "OpenTelemetry",
      "status": "running",
      "config": {...},
      "supported_asset_types": ["tool", "model", "agent"],
      "webhooks": []
    }
  ],
  "count": 1
}
```

#### POST `/api/v1/instances`

**Request:**
```json
{
  "plugin_type": "databricks",
  "instance_id": "databricks-prod",
  "display_name": "Production Databricks",
  "config": {
    "host": "https://dbc-xxx.cloud.databricks.com",
    "token": "dapi..."
  },
  "auto_start": true
}
```

- `plugin_type` (required): Plugin type identifier
- `instance_id` (optional): Custom ID, auto-generated (UUIDv5) if omitted
- `display_name` (optional): Human-readable name
- `config` (optional): Plugin-specific configuration
- `auto_start` (optional): Start immediately after creation

**Response (201):**
```json
{
  "success": true,
  "instance": {...}
}
```

#### PUT `/api/v1/instances/{id}`

**Request:**
```json
{
  "display_name": "New Name",
  "config": {"host": "..."},
  "auto_start": true
}
```

**Response (200):**
```json
{
  "success": true,
  "instance": {...}
}
```

#### DELETE `/api/v1/instances/{id}`

**Response (200):**
```json
{
  "success": true,
  "message": "Instance deleted"
}
```

### Instance Actions

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/instances/{id}/start` | Start a plugin instance |
| POST | `/api/v1/instances/{id}/stop` | Stop a plugin instance |
| POST | `/api/v1/instances/{id}/refresh` | Trigger re-discovery |
| POST | `/api/v1/instances/{id}/verify` | Verify connection to data source |

#### POST `/api/v1/instances/{id}/start`

**Response (200):**
```json
{
  "success": true,
  "message": "Instance started",
  "already_running": false
}
```

#### POST `/api/v1/instances/{id}/refresh`

**Request (optional):**
```json
{"days": 7}
```

**Response (200):**
```json
{
  "success": true,
  "message": "Refresh initiated",
  "method": "refresh_traces",
  "days": 7
}
```

#### POST `/api/v1/instances/{id}/verify`

**Response (200):**
```json
{
  "success": true,
  "verification": {
    "success": true,
    "host": "https://dbc-xxx.cloud.databricks.com"
  }
}
```

---

## Webhooks

Subscribe URLs to receive OTLP trace payloads as they are discovered by a plugin instance.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/instances/{id}/webhooks` | List subscribed webhook URLs |
| POST | `/api/v1/instances/{id}/webhooks` | Subscribe a webhook URL |
| DELETE | `/api/v1/instances/{id}/webhooks` | Unsubscribe a webhook URL |

#### POST `/api/v1/instances/{id}/webhooks`

**Request:**
```json
{
  "url": "https://collector:4318/v1/traces",
  "headers": {"Authorization": "Bearer ..."}
}
```

- `url` (required): Must start with `http://` or `https://`
- `headers` (optional): Custom headers sent with each webhook delivery

**Response (200):**
```json
{
  "success": true,
  "added": true,
  "webhooks": ["https://collector:4318/v1/traces"]
}
```

#### DELETE `/api/v1/instances/{id}/webhooks`

**Request:**
```json
{"url": "https://collector:4318/v1/traces"}
```

**Response (200):**
```json
{
  "success": true,
  "removed": true,
  "webhooks": []
}
```

---

## Persistence

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/persistence/status` | Get persistence status and asset counts |
| GET | `/api/v1/stats` | Lightweight stats (in-memory, no DB query) |
| POST | `/api/v1/persistence/save` | Manually trigger a save to the database |
| POST | `/api/v1/persistence/load` | No-op (DB is source of truth) |
| GET | `/api/v1/persistence/export` | Export all persisted data as JSON |
| POST | `/api/v1/reset-discoveries` | Clear all discovered assets from DB and plugin memory |

### GET `/api/v1/persistence/status`

**Response (200):**
```json
{
  "enabled": true,
  "stats": {
    "tools": 15,
    "models": 8,
    "agents": 3,
    "downstream_systems": 2,
    "mcp_servers": 1,
    "lineage": 24
  }
}
```

### GET `/api/v1/stats`

**Response (200):**
```json
{
  "last_modified": "2026-03-10T14:30:00Z"
}
```

---

## Visualization

| Method | Path | Query Parameters | Description |
|--------|------|------------------|-------------|
| GET | `/api/v1/lineage-graph` | `theme` | Interactive lineage graph (HTML) |

**Query parameters:**
- `theme` -- `opencite` (default), `databricks-light`, or `databricks-dark`

**Response:** HTML page (`text/html`) with an interactive pyvis network graph.

---

## Common Error Responses

| Status | Response | Condition |
|--------|----------|-----------|
| 400 | `{"error": "..."}` | Missing required fields or invalid input |
| 403 | `{"error": "No client initialized..."}` | Plugins not configured |
| 404 | `{"error": "Instance not found"}` | Resource not found |
| 415 | `{"error": "Unsupported Content-Type: ..."}` | Invalid Content-Type (OTLP endpoints) |
| 500 | `{"error": "..."}` | Server error |
| 503 | `{"status": "not_ready", ...}` | Service not ready |

## Environment Variables

```bash
OPENCITE_HOST="0.0.0.0"           # Bind address (default: 0.0.0.0)
OPENCITE_PORT="8080"              # API port (default: 8080)
OPENCITE_LOG_LEVEL="INFO"         # Log level (default: INFO)
OPENCITE_AUTO_START="true"        # Auto-configure plugins on startup

OPENCITE_OTLP_HOST="0.0.0.0"     # OTLP receiver bind address
OPENCITE_OTLP_PORT="4318"        # OTLP receiver port (default: 4318)
OPENCITE_ENABLE_OTEL="true"      # Enable embedded OTLP receiver

OPENCITE_PERSISTENCE_ENABLED="true"
OPENCITE_DB_PATH="./opencite.db"  # SQLite database path
```

## Related Documentation

- [Schema Documentation](SCHEMA_DOCUMENTATION.md) -- JSON export format
- [Sending Traces to Open-CITE](SENDING_TRACES.md) -- Configure external sources
- [Plugin Authoring Guide](PLUGINS.md) -- Creating plugins
- [Development Guide](DEVELOPMENT.md) -- Development setup
