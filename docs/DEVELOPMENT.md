# OpenCITE GUI Development Guide

## Quick Start

```bash
# Simple way - use the dev script
./run_dev.sh

# Or manually
source venv/bin/activate
opencite gui --debug
```

Open browser to: **http://127.0.0.1:5000**

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
src/open_cite/gui/
├── app.py              # Flask backend
│   ├── REST API endpoints
│   ├── Plugin management
│   ├── State management
│   └── Discovery orchestration
│
└── templates/
    └── index.html      # Frontend UI
        ├── HTML structure
        ├── CSS styling
        └── JavaScript logic
```

## Common Development Tasks

### 1. Add a New Plugin

Plugins are auto-discovered via `plugins/registry.py`. Create a single file in `src/open_cite/plugins/` and the GUI, API, and client pick it up automatically — no other files need editing.

**Step 1: Create `src/open_cite/plugins/your_plugin.py`**

```python
from open_cite.core import BaseDiscoveryPlugin

class YourPlugin(BaseDiscoveryPlugin):
    plugin_type = "your_plugin"

    @classmethod
    def plugin_metadata(cls):
        return {
            "name": "Your Plugin",
            "description": "What your plugin discovers",
            "required_fields": {
                "api_key": {
                    "label": "API Key",
                    "default": "",
                    "required": True,
                    "type": "password"
                }
            },
            "env_vars": ["YOUR_PLUGIN_API_KEY"],
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        return cls(
            api_key=config.get("api_key"),
            instance_id=instance_id,
            display_name=display_name,
        )

    def __init__(self, api_key=None, instance_id=None, display_name=None):
        super().__init__(instance_id=instance_id, display_name=display_name)
        self.api_key = api_key

    @property
    def supported_asset_types(self):
        return {"your_asset"}

    @property
    def supports_multiple_instances(self):
        return True

    def verify_connection(self):
        return {"success": True}

    def list_assets(self, asset_type=None):
        # Return discovered assets
        return []

    def get_identification_attributes(self):
        return []
```

That's it. The registry auto-discovers the class, the GUI shows it as a configurable plugin, and the API serves its assets.

### 3. Debug Issues

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

### Manual Testing

1. **Start GUI in debug mode**
   ```bash
   opencite gui --debug
   ```

2. **Configure test plugin** (e.g., OpenTelemetry)
   - Check "OpenTelemetry" box
   - Use defaults: host=localhost, port=4318
   - Click "Start Discovery"

3. **Send test trace**
   ```bash
   # In another terminal
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
               {"key": "gen_ai.system", "value": {"stringValue": "openrouter"}}
             ]
           }]
         }]
       }]
     }'
   ```

4. **Verify in GUI**
   - Should see "test-tool" appear in Tools tab
   - Should see "openai/gpt-4" appear in Models tab
   - Stats should update

### Automated Testing

Run integration tests while GUI is running:
```bash
# In another terminal
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
   <h1>🔍 OpenCITE</h1>

   <!-- After -->
   <h1>🔍 OpenCITE [DEV MODE]</h1>
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

Set these for development:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1

# Plugin credentials (optional)
export DATABRICKS_HOST="https://..."
export DATABRICKS_TOKEN="dapi..."
export GCP_PROJECT_ID="your-project"
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
            "name": "OpenCITE GUI",
            "type": "python",
            "request": "launch",
            "module": "open_cite.gui.app",
            "env": {
                "FLASK_DEBUG": "1"
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

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Debug Mode](https://flask.palletsprojects.com/en/latest/debugging/)
- [Browser DevTools Guide](https://developer.chrome.com/docs/devtools/)
