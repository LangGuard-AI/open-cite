"""
OpenCITE Web GUI - Flask Application

A web-based interface for OpenCITE discovery and visualization.

All REST API routes are shared with the headless API service via
``register_api_routes()`` from ``open_cite.api.app``.  This module only
adds the template route (``GET /``), WebSocket handlers, and the
``run_gui()`` entry-point.
"""

import logging
import os
import time
from datetime import datetime

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Shared state & route registration from the API module
import open_cite.api.app as api_app
from open_cite.api.app import (
    init_opencite_state,
    register_api_routes,
    _restore_saved_plugins,
    _reclassify_downstream,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Flask + SocketIO setup
# =========================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.json.sort_keys = False  # Preserve dict insertion order (e.g. plugin config fields)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('open_cite.plugins.opentelemetry').setLevel(logging.DEBUG)

# =========================================================================
# Debounced WebSocket push helpers
# =========================================================================

_last_assets_push = 0.0
_PUSH_MIN_INTERVAL = 0.5  # seconds


def _push_assets_update(source_plugin=None):
    """Push current assets to all connected WebSocket clients (debounced)."""
    global _last_assets_push

    now = time.monotonic()
    if now - _last_assets_push < _PUSH_MIN_INTERVAL:
        return
    _last_assets_push = now

    try:
        if not api_app.client:
            return

        assets = {
            "tools": [],
            "models": [],
            "agents": [],
            "downstream_systems": [],
            "mcp_servers": [],
            "mcp_tools": [],
            "mcp_resources": [],
            "data_assets": [],
            "gcp_models": [],
            "gcp_endpoints": []
        }

        c = api_app.client
        assets["tools"] = c.list_tools()
        assets["models"] = c.list_models()
        assets["agents"] = c.list_agents()
        assets["downstream_systems"] = c.list_downstream_systems()
        assets["mcp_servers"] = c.list_mcp_servers()
        assets["mcp_tools"] = c.list_mcp_tools()
        assets["mcp_resources"] = c.list_mcp_resources()

        # Use cached values for expensive Databricks/GCP calls
        if api_app.asset_cache:
            plugins_enabled = api_app.discovery_status.get("plugins_enabled", [])
            if "databricks" in plugins_enabled:
                assets["data_assets"] = api_app.asset_cache.get("assets", {}).get("data_assets", [])
            if "google_cloud" in plugins_enabled:
                assets["gcp_models"] = api_app.asset_cache.get("assets", {}).get("gcp_models", [])
                assets["gcp_endpoints"] = api_app.asset_cache.get("assets", {}).get("gcp_endpoints", [])

        _reclassify_downstream(assets)
        totals = {k: len(v) for k, v in assets.items()}

        result = {
            "assets": assets,
            "totals": totals,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update shared cache as side effect
        api_app.asset_cache = result
        api_app.asset_cache_time = datetime.utcnow()

        socketio.emit('assets_update', result)
    except Exception as e:
        logger.error(f"Error pushing assets update: {e}")


def _push_status_update():
    """Push current discovery status to all connected WebSocket clients."""
    try:
        socketio.emit('status_update', api_app.discovery_status)
    except Exception as e:
        logger.error(f"Error pushing status update: {e}")


# =========================================================================
# GUI integration hooks (set on the api_app module so shared routes use them)
# =========================================================================

def _gui_start_plugin(plugin):
    """Start a plugin in a background thread and push WebSocket updates."""
    from threading import Thread

    plugin.on_data_changed = lambda p: _push_assets_update(p)

    def _run():
        try:
            plugin.start()
        except Exception as e:
            logger.error(f"Background start failed for {plugin.instance_id}: {e}")
            plugin.status = "error"
        _push_status_update()
        _push_assets_update(plugin)

    Thread(target=_run, daemon=True).start()


# Wire the hooks into the shared API module
api_app._on_plugin_start = _gui_start_plugin
api_app._on_status_changed = _push_status_update


# =========================================================================
# WebSocket handlers
# =========================================================================

@socketio.on('connect')
def handle_connect():
    """Send current state to newly connected client."""
    logger.info("WebSocket client connected")
    emit('status_update', api_app.discovery_status)


@socketio.on('disconnect')
def handle_disconnect():
    """Log client disconnection."""
    logger.info("WebSocket client disconnected")


# =========================================================================
# GUI-only route
# =========================================================================

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


# =========================================================================
# Initialisation: shared state + shared API routes
# =========================================================================

init_opencite_state(app)
register_api_routes(app)
_restore_saved_plugins()


# =========================================================================
# Entry-point
# =========================================================================

def run_gui(host='127.0.0.1', port=5000, debug=False):
    """
    Run the OpenCITE GUI.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 5000)
        debug: Enable debug mode
    """
    print(f"\n{'='*60}")
    print(f"  OpenCITE Web GUI")
    print(f"{'='*60}")
    print(f"  Access the GUI at: http://{host}:{port}")
    print(f"{'='*60}\n")

    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_gui(debug=True)
