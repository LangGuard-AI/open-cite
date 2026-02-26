"""
OpenCITE Web GUI - Flask Application

A web-based interface for OpenCITE discovery and visualization.

All REST API routes are shared with the headless API service via
``register_api_routes()`` from ``open_cite.api.app``.  This module only
adds the template route (``GET /``), WebSocket handlers, and the
``run_gui()`` entry-point.
"""

import asyncio
import logging
import os
import signal
import threading
import time
from datetime import datetime

from flask import Flask, render_template
import socketio as sio_module

# Shared state & route registration from the API module
import open_cite.api.app as api_app
from open_cite.api.app import (
    init_opencite_state,
    register_api_routes,
    _restore_saved_plugins,
    _auto_configure_databricks_app,
    _reclassify_downstream,
    _otlp_ingest,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Flask + python-socketio setup
# =========================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.json.sort_keys = False  # Preserve dict insertion order (e.g. plugin config fields)

# AsyncServer is required for the ASGI layer (Hypercorn).
# Sync emit calls from plugin threads use _sync_emit() below.
sio = sio_module.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Reference to the running asyncio event loop, set by run_gui().
# Used by _sync_emit() to schedule async emits from sync threads.
_event_loop: asyncio.AbstractEventLoop | None = None

# Configure logging — honour OPENCITE_LOG_LEVEL (default INFO)
_log_level = getattr(logging, os.environ.get('OPENCITE_LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('open_cite.plugins.opentelemetry').setLevel(_log_level)
logging.getLogger('hpack').setLevel(logging.WARNING)
for _ul in ('urllib3', 'urllib3.util.retry', 'urllib3.connectionpool'):
    _ulg = logging.getLogger(_ul)
    _ulg.setLevel(logging.ERROR)
    _ulg.propagate = False
logging.getLogger('geventwebsocket.handler').setLevel(logging.ERROR)
logging.getLogger('databricks.sql.auth').setLevel(logging.WARNING)


# =========================================================================
# Sync-to-async emit bridge
# =========================================================================

def _sync_emit(event, data, **kwargs):
    """Emit a socket.io event from a sync context (e.g. plugin callback thread).

    Schedules the coroutine on the main event loop via
    ``run_coroutine_threadsafe`` (fire-and-forget).
    """
    loop = _event_loop
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(sio.emit(event, data, **kwargs), loop)


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

        _sync_emit('assets_update', result)
    except Exception as e:
        logger.error(f"Error pushing assets update: {e}")


def _push_status_update():
    """Push current discovery status to all connected WebSocket clients."""
    try:
        _sync_emit('status_update', api_app.discovery_status)
    except Exception as e:
        logger.error(f"Error pushing status update: {e}")


# =========================================================================
# GUI integration hooks (set on the api_app module so shared routes use them)
# =========================================================================

def _gui_start_plugin(plugin):
    """Start a plugin in a background thread and push WebSocket updates."""
    def _on_data(p):
        api_app._maybe_save_state(p)   # persist first so DB is fresh
        _push_assets_update(p)          # then read and push to clients
    plugin.on_data_changed = _on_data

    def _run():
        try:
            plugin.start()
        except Exception as e:
            logger.error(f"Background start failed for {plugin.instance_id}: {e}")
            plugin.status = "error"
        _push_status_update()
        _push_assets_update(plugin)

    threading.Thread(target=_run, daemon=True).start()


# Wire the hooks into the shared API module
api_app._on_plugin_start = _gui_start_plugin
api_app._on_status_changed = _push_status_update


# =========================================================================
# WebSocket handlers
# =========================================================================

@sio.on('connect')
async def handle_connect(sid, environ):
    """Send current state to newly connected client."""
    logger.info("WebSocket client connected: %s", sid)
    await sio.emit('status_update', api_app.discovery_status, to=sid)


@sio.on('disconnect')
async def handle_disconnect(sid):
    """Log client disconnection."""
    logger.info("WebSocket client disconnected: %s", sid)


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

print("  [gui init] Initializing OpenCITE state...")
init_opencite_state(app)
# Wire up WebSocket push for the embedded OTel plugin (created during init,
# bypasses _gui_start_plugin which only handles user-started plugins)
if api_app._default_otel_plugin is not None:
    def _default_otel_on_data(p):
        api_app._maybe_save_state(p)   # persist first so DB is fresh
        _push_assets_update(p)          # then read and push to clients
    api_app._default_otel_plugin.on_data_changed = _default_otel_on_data
print("  [gui init] Registering API routes...")
register_api_routes(app)
print("  [gui init] Restoring saved plugins...")
_restore_saved_plugins()
logger.info("Checking Databricks App auto-configure...")
_auto_configure_databricks_app()
logger.info("GUI initialization complete.")


# =========================================================================
# Entry-point
# =========================================================================

def run_gui(host=None, port=None, debug=False):
    """
    Run the OpenCITE GUI.

    Args:
        host: Host to bind to (default from OPENCITE_HOST env or 127.0.0.1)
        port: Port to bind to (default from DATABRICKS_APP_PORT or OPENCITE_PORT env or 5000)
        debug: Enable debug mode
    """
    from open_cite.asgi.app import create_asgi_app
    from hypercorn.config import Config as HyperConfig
    from hypercorn.asyncio import serve

    if host is None:
        host = os.environ.get('OPENCITE_HOST', '127.0.0.1')
    if port is None:
        port = int(os.environ.get('DATABRICKS_APP_PORT',
                   os.environ.get('OPENCITE_PORT', '5000')))

    ingest_fn = _otlp_ingest if api_app._default_otel_plugin else None
    from open_cite.api.app import _otlp_ingest_logs
    logs_ingest_fn = _otlp_ingest_logs if api_app._default_otel_plugin else None
    asgi_app = create_asgi_app(
        app, sio_server=sio, ingest_fn=ingest_fn,
        logs_ingest_fn=logs_ingest_fn,
    )

    print(f"\n{'='*60}")
    print(f"  OpenCITE Web GUI (HTTP/2 via Hypercorn)")
    print(f"{'='*60}")
    print(f"  Access the GUI at: http://{host}:{port}")
    if api_app._default_otel_plugin:
        print(f"  OTLP (JSON): http://{host}:{port}/v1/traces")
        print(f"  OTLP (gRPC): http://{host}:{port} (HTTP/2)")
    print(f"{'='*60}\n")

    hconfig = HyperConfig()
    hconfig.bind = [f"{host}:{port}"]

    async def _serve():
        global _event_loop
        _event_loop = asyncio.get_running_loop()

        shutdown_event = asyncio.Event()

        def _signal_handler():
            if shutdown_event.is_set():
                # Second signal — force exit immediately
                logger.warning("Received second signal, forcing exit")
                os._exit(1)
            logger.info("Received shutdown signal, shutting down gracefully... (press Ctrl+C again to force)")
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        try:
            await serve(asgi_app, hconfig, shutdown_trigger=shutdown_event.wait)
        finally:
            _event_loop = None
            api_app.shutdown_cleanup()

    asyncio.run(_serve())


if __name__ == '__main__':
    run_gui(debug=True)
