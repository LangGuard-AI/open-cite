# Open-CITE Development Guide

## Quick Start

```bash
# GUI (with browser UI)
./run_dev.sh            # or: opencite gui --debug
# → http://127.0.0.1:5000

# Headless API (REST only, for Kubernetes / CI)
opencite api            # or: python -m open_cite.api.app
# → http://0.0.0.0:8080
```

## Development Mode Features

### Auto-Reload
Flask automatically restarts when you save code changes:
- Edit `src/open_cite/gui/app.py` → Flask restarts
- Edit `src/open_cite/gui/templates/index.html` → Just refresh browser

### Detailed Error Pages
Debug mode shows full stack traces in the browser when errors occur:
- Exception details
- Local variables
- Code context

### Verbose Logging
See detailed logs in terminal:
```
2025-12-11 13:15:23 [INFO] Started OpenTelemetry plugin on localhost:4318
2025-12-11 13:15:23 [DEBUG] Trace ingested: trace_id=abc123...
```

## Project Structure

```
src/open_cite/
├── core.py                 # BaseDiscoveryPlugin base class
├── client.py               # OpenCiteClient (plugin-agnostic orchestrator)
├── otlp_converter.py       # MLflow/Genie → OTLP JSON converters
├── identifier.py           # Tool identification / mapping
├── schema.py               # Open-CITE JSON export schema
│
├── gui/
│   ├── app.py              # Flask + SocketIO backend (WebSocket push)
│   └── templates/
│       └── index.html      # Single-page frontend (HTML/CSS/JS)
│
├── api/
│   ├── app.py              # Headless REST API (Flask, no GUI)
│   ├── config.py           # Environment-based configuration
│   ├── health.py           # /healthz and /readyz endpoints
│   ├── shutdown.py         # Graceful shutdown handler
│   └── persistence.py      # SQLite persistence manager
│
└── plugins/
    ├── registry.py          # Auto-discovery and factory
    ├── opentelemetry.py     # OTLP trace receiver
    ├── databricks.py        # Databricks MLflow + Genie + Unity Catalog
    ├── google_cloud.py      # Vertex AI + Compute Engine MCP
    ├── zscaler.py           # ZIA DLP + NSS shadow MCP detection
    └── aws/
        ├── base.py          # Shared AWS auth mixin
        ├── bedrock.py       # AWS Bedrock
        └── sagemaker.py     # AWS SageMaker
```

## Common Development Tasks

### 1. Add a New Plugin

See **[docs/PLUGINS.md](PLUGINS.md)** for the full plugin authoring guide, including the minimal template, required interface, lifecycle methods, webhook trace forwarding, and OTLP conventions.

**Short version**: create a single `.py` file in `src/open_cite/plugins/` with a concrete `BaseDiscoveryPlugin` subclass. The registry auto-discovers it and the GUI/API expose it automatically.

### 2. Debug Issues

**Backend debugging (Python):**
```python
# Add debug prints in app.py
logger.debug(f"Current state: {discovery_status}")

# Or use Python debugger
import pdb; pdb.set_trace()  # Breakpoint
```

**Frontend debugging (JavaScript):**
```javascript
// Add console logs
console.log('Assets:', assets);

// Use debugger statement
debugger;  // Browser will pause here

// Inspect network requests
// Open DevTools (F12) → Network tab
```

**Check Flask logs:**
```bash
# Terminal shows:
127.0.0.1 - - [11/Dec/2025 13:15:25] "GET /api/assets HTTP/1.1" 200 -
```

## Testing During Development

### Manual Testing (GUI)

1. **Start GUI in debug mode**
   ```bash
   opencite gui --debug
   ```

2. **Create an OpenTelemetry plugin instance** via the GUI
   - Select "OpenTelemetry" plugin type
   - Use defaults: host=0.0.0.0, port=4318
   - Start the instance

3. **Send test trace**
   ```bash
   curl -X POST http://localhost:4318/v1/traces \
     -H "Content-Type: application/json" \
     -d '{
       "resourceSpans": [{
         "resource": {
           "attributes": [
             {"key": "service.name", "value": {"stringValue": "test-tool"}}
           ]
         },
         "scopeSpans": [{
           "scope": {"name": "openai"},
           "spans": [{
             "traceId": "abc123def456",
             "spanId": "123456",
             "name": "chat.completions",
             "attributes": [
               {"key": "gen_ai.request.model", "value": {"stringValue": "openai/gpt-4"}},
               {"key": "gen_ai.system", "value": {"stringValue": "openai"}}
             ]
           }]
         }]
       }]
     }'
   ```

4. **Verify in GUI** -- "test-tool" should appear in the Tools tab and "openai/gpt-4" in Models. Updates arrive via WebSocket (no refresh needed).

### Manual Testing (Headless API)

```bash
# Start API
opencite api

# Create an instance
curl -X POST http://localhost:8080/api/v1/instances \
  -H 'Content-Type: application/json' \
  -d '{"plugin_type":"opentelemetry","auto_start":true}'

# List instances
curl http://localhost:8080/api/v1/instances

# Get assets
curl http://localhost:8080/api/v1/assets
```

### Testing Webhook Forwarding

```bash
# 1. Find your plugin instance ID
curl http://localhost:5000/api/instances
# → note the instance_id

# 2. Subscribe a webhook (e.g. a local OTLP collector or a request bin)
curl -X POST http://localhost:5000/api/instances/<instance_id>/webhooks \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://localhost:4318/v1/traces"}'

# 3. List subscribed webhooks
curl http://localhost:5000/api/instances/<instance_id>/webhooks

# 4. Trigger discovery (start the plugin or send traces)
# → OTLP payloads are POSTed to the webhook URL as traces are processed

# 5. Unsubscribe
curl -X DELETE http://localhost:5000/api/instances/<instance_id>/webhooks \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://localhost:4318/v1/traces"}'
```

### Automated Testing

```bash
source venv/bin/activate
pytest tests/integration/ -v
```

## Common Issues

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill it
kill -9 <PID>

# Or use different port
opencite gui --debug --port 8080
```

### Changes Not Appearing
- **Backend changes**: Wait for Flask to restart (watch terminal)
- **Frontend changes**: Hard refresh browser (Ctrl+Shift+R)
- **CSS not updating**: Clear browser cache

### Auto-Reload Not Working
```bash
# Make sure debug mode is enabled
opencite gui --debug  # ← Must have --debug flag

# Check terminal for "Restarting with stat" message
```

### Plugin Configuration Fails
- Check browser console (F12) for error details
- Check Flask terminal for backend errors
- Verify credentials are correct
- Test plugin manually first:
  ```python
  from open_cite.plugins.registry import create_plugin_instance
  plugin = create_plugin_instance("databricks", {"host": "...", "token": "..."})
  print(plugin.verify_connection())
  ```

## Development Workflow Example

```bash
# 1. Start development server
./run_dev.sh

# 2. Open browser
# → http://127.0.0.1:5000

# 3. Make code changes
# Edit src/open_cite/gui/app.py or templates/index.html

# 4. See changes
# - Backend: Flask auto-restarts
# - Frontend: Refresh browser

# 5. Test
# - Manual testing in browser
# - Check terminal logs
# - Use browser DevTools

# 6. Stop server
# Ctrl+C in terminal
```

## Hot Reload Demo

**Try this:**

1. Start GUI: `opencite gui --debug`
2. Open browser to http://127.0.0.1:5000
3. Edit `src/open_cite/gui/templates/index.html`
4. Change line 13:
   ```html
   <!-- Before -->
   <h1>🔍 Open-CITE</h1>

   <!-- After -->
   <h1>🔍 Open-CITE [DEV MODE]</h1>
   ```
5. Save file
6. Refresh browser (F5)
7. See "[DEV MODE]" appear!

## Performance Tips

### Real-Time Updates
The GUI uses WebSocket (Flask-SocketIO) for push-based updates. When traces arrive, assets appear immediately. If WebSocket is unavailable, the browser falls back to 3-second polling automatically.

### Limit Plugin Output
For development, start with just one plugin to reduce noise.

## Environment Variables

### GUI development

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
```

### Headless API configuration

These control the API service when run via `opencite api` or gunicorn:

```bash
# Server
export OPENCITE_HOST="0.0.0.0"           # default: 0.0.0.0
export OPENCITE_PORT="8080"              # default: 8080
export OPENCITE_LOG_LEVEL="DEBUG"        # default: INFO
export OPENCITE_AUTO_START="true"        # auto-configure plugins on startup

# OTLP receiver
export OPENCITE_OTLP_HOST="0.0.0.0"     # default: 0.0.0.0
export OPENCITE_OTLP_PORT="4318"        # default: 4318
export OPENCITE_ENABLE_OTEL="true"       # default: true

# Persistence
export OPENCITE_PERSISTENCE_ENABLED="true"
export OPENCITE_DB_PATH="./opencite.db"  # default: /data/opencite.db
```

### Plugin credentials (optional)

```bash
# Databricks
export DATABRICKS_HOST="https://dbc-xxx.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_WAREHOUSE_ID="..."
export OPENCITE_ENABLE_DATABRICKS="true"

# Google Cloud
export GCP_PROJECT_ID="your-project"
export GCP_LOCATION="us-central1"
export OPENCITE_ENABLE_GOOGLE_CLOUD="true"

# AWS (uses standard AWS credential chain)
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

## IDE Setup

### VS Code

**Install extensions:**
- Python
- Flask Snippets
- Prettier (for HTML/CSS/JS)

**Debug configuration** (`.vscode/launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Open-CITE GUI",
            "type": "python",
            "request": "launch",
            "module": "open_cite.gui.app",
            "env": { "FLASK_DEBUG": "1" }
        },
        {
            "name": "Open-CITE API",
            "type": "python",
            "request": "launch",
            "module": "open_cite.api.app",
            "env": {
                "OPENCITE_LOG_LEVEL": "DEBUG",
                "OPENCITE_ENABLE_OTEL": "true"
            }
        }
    ]
}
```

### PyCharm

1. Right-click `src/open_cite/gui/app.py`
2. Select "Run 'app'"
3. Add `--debug` to run configuration

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/gui-improvement

# Make changes
# ... edit files ...

# Test changes
opencite gui --debug

# Commit
git add src/open_cite/gui/
git commit -m "Add new feature to GUI"

# Push
git push origin feature/gui-improvement
```

## Related Documentation

- [Plugin Authoring Guide](PLUGINS.md) -- creating a new plugin, webhook forwarding, OTLP conventions
- [OpenTelemetry Plugin](plugins/OPENTELEMETRY_PLUGIN.md) -- OTLP receiver setup and trace format
- [AWS Plugins](plugins/AWS_PLUGINS.md) -- Bedrock and SageMaker discovery
- [Google Cloud Plugin](plugins/GOOGLE_CLOUD_PLUGIN.md) -- Vertex AI and Compute Engine discovery
- [Schema Documentation](SCHEMA_DOCUMENTATION.md) -- JSON export format

## External Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [Browser DevTools Guide](https://developer.chrome.com/docs/devtools/)
