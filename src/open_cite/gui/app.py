"""
OpenCITE Web GUI - Flask Application

A web-based interface for OpenCITE discovery and visualization.
"""

import os
import json
import logging
import socket
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from threading import Thread, Lock

from open_cite.client import OpenCiteClient
from open_cite.identifier import ToolIdentifier
from open_cite.plugins.registry import get_all_plugin_metadata, create_plugin_instance as registry_create_plugin

logger = logging.getLogger(__name__)

# Downstream types that are really data assets, not generic downstream systems
_DATA_ASSET_TYPES = {"database", "dataset", "document_store"}


def _reclassify_downstream(assets):
    """Move data-oriented downstream items into data_assets."""
    remaining = []
    for item in assets.get("downstream_systems", []):
        if item.get("type") in _DATA_ASSET_TYPES:
            assets["data_assets"].append(item)
        else:
            remaining.append(item)
    assets["downstream_systems"] = remaining


def get_local_ip():
    """
    Discover the local IP address of this machine.

    Returns the primary IP address that can be used by other machines
    to connect to this server.
    """
    try:
        # Create a socket to determine the local IP
        # This doesn't actually connect, just determines routing
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # Connect to a public DNS server (doesn't actually send data)
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        except Exception:
            # Fallback to localhost
            local_ip = '127.0.0.1'
        finally:
            s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP: {e}")
        return '127.0.0.1'

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.json.sort_keys = False  # Preserve dict insertion order (e.g. plugin config fields)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# Enable DEBUG logging for this session
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set DEBUG level for opentelemetry plugin specifically
logging.getLogger('open_cite.plugins.opentelemetry').setLevel(logging.DEBUG)
logger.warning("[DEBUG MODE ENABLED] Raw OTLP traces will be logged")

# Global state
client: Optional[OpenCiteClient] = None
discovery_status = {
    "running": False,
    "plugins_enabled": [],
    "last_updated": None,
    "error": None,
    "current_status": "Idle",
    "progress": []
}
state_lock = Lock()
discovering_assets = False  # Flag to prevent concurrent asset discovery
asset_cache = None  # Cache for discovered assets
asset_cache_time = None  # Last time assets were discovered
tool_identifier = ToolIdentifier()  # Shared identifier for tool/agent mapping


# Debounce state for WebSocket pushes (max 2 pushes/second)
_last_assets_push = 0.0
_PUSH_MIN_INTERVAL = 0.5  # seconds


def _push_assets_update(source_plugin=None):
    """Push current assets to all connected WebSocket clients (debounced)."""
    global _last_assets_push, asset_cache, asset_cache_time

    now = time.monotonic()
    if now - _last_assets_push < _PUSH_MIN_INTERVAL:
        return
    _last_assets_push = now

    try:
        if not client:
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

        # Collect assets from ALL plugins (fast, in-memory aggregation)
        assets["tools"] = client.list_tools()
        assets["models"] = client.list_models()
        assets["agents"] = client.list_agents()
        assets["downstream_systems"] = client.list_downstream_systems()
        assets["mcp_servers"] = client.list_mcp_servers()
        assets["mcp_tools"] = client.list_mcp_tools()
        assets["mcp_resources"] = client.list_mcp_resources()

        # Use cached values for expensive Databricks/GCP calls
        if asset_cache:
            if "databricks" in discovery_status.get("plugins_enabled", []):
                assets["data_assets"] = asset_cache.get("assets", {}).get("data_assets", [])
            if "google_cloud" in discovery_status.get("plugins_enabled", []):
                assets["gcp_models"] = asset_cache.get("assets", {}).get("gcp_models", [])
                assets["gcp_endpoints"] = asset_cache.get("assets", {}).get("gcp_endpoints", [])

        _reclassify_downstream(assets)
        totals = {k: len(v) for k, v in assets.items()}

        result = {
            "assets": assets,
            "totals": totals,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update cache as side effect
        asset_cache = result
        asset_cache_time = datetime.utcnow()

        socketio.emit('assets_update', result)
    except Exception as e:
        logger.error(f"Error pushing assets update: {e}")


def _push_status_update():
    """Push current discovery status to all connected WebSocket clients."""
    try:
        socketio.emit('status_update', discovery_status)
    except Exception as e:
        logger.error(f"Error pushing status update: {e}")


@socketio.on('connect')
def handle_connect():
    """Send current state to newly connected client."""
    logger.info("WebSocket client connected")
    emit('status_update', discovery_status)


@socketio.on('disconnect')
def handle_disconnect():
    """Log client disconnection."""
    logger.info("WebSocket client disconnected")


def discover_available_plugins():
    """Dynamically discover all available plugins via the registry."""
    return get_all_plugin_metadata()


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current discovery status."""
    with state_lock:
        return jsonify(discovery_status)


@app.route('/api/plugins', methods=['GET'])
def list_available_plugins():
    """List available plugins and their requirements (dynamically discovered)."""
    try:
        plugins = discover_available_plugins()
        return jsonify(plugins)
    except Exception as e:
        logger.error(f"Failed to discover plugins: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/instances', methods=['GET'])
def list_plugin_instances():
    """List all active plugin instances."""
    if not client:
        return jsonify({"instances": [], "count": 0})

    try:
        instances = client.list_plugin_instances()
        return jsonify({"instances": instances, "count": len(instances)})
    except Exception as e:
        logger.error(f"Failed to list plugin instances: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/instances', methods=['POST'])
def create_plugin_instance():
    """Create a new plugin instance."""
    global client

    try:
        data = request.json
        plugin_type = data.get('plugin_type')
        instance_id = data.get('instance_id')
        display_name = data.get('display_name')
        config = data.get('config', {})
        auto_start = data.get('auto_start', False)

        if not plugin_type:
            return jsonify({"error": "plugin_type is required"}), 400

        # Initialize client if not already running
        if not client:
            client = OpenCiteClient()

        # Always generate instance_id as UUIDv5 (namespace: plugin_type, name: display_name)
        OPENCITE_NS = uuid.UUID('a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d')
        id_name = f"{plugin_type}:{display_name or plugin_type}"
        instance_id = str(uuid.uuid5(OPENCITE_NS, id_name))

        # Auto-generate display_name if not provided
        if not display_name:
            display_name = plugin_type.replace('_', ' ').title()

        # Create the plugin instance
        plugin_instance = _create_plugin_instance(plugin_type, instance_id, display_name, config)

        # Register the plugin
        client.register_plugin(plugin_instance)

        # Start the plugin if auto_start is enabled
        if auto_start:
            _start_plugin_instance(plugin_instance)
            with state_lock:
                if plugin_type not in discovery_status["plugins_enabled"]:
                    discovery_status["plugins_enabled"].append(plugin_type)
                discovery_status["running"] = True
            _push_status_update()

        return jsonify({
            "success": True,
            "instance": plugin_instance.to_dict()
        }), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to create plugin instance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/instances/<instance_id>', methods=['DELETE'])
def delete_plugin_instance(instance_id: str):
    """Delete a plugin instance."""
    global client

    try:
        if not client or instance_id not in client.plugins:
            return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

        plugin = client.plugins[instance_id]
        _stop_plugin_instance(plugin)
        client.unregister_plugin(instance_id)

        return jsonify({"success": True, "message": f"Instance '{instance_id}' deleted"})

    except Exception as e:
        logger.error(f"Failed to delete instance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/instances/<instance_id>/start', methods=['POST'])
def start_plugin_instance(instance_id: str):
    """Start a plugin instance."""
    try:
        if not client:
            return jsonify({"error": "No client initialized"}), 400

        if instance_id not in client.plugins:
            return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

        plugin = client.plugins[instance_id]
        _start_plugin_instance(plugin)

        # Update discovery status
        with state_lock:
            plugin_type = plugin.plugin_type
            if plugin_type not in discovery_status["plugins_enabled"]:
                discovery_status["plugins_enabled"].append(plugin_type)
            discovery_status["running"] = True
        _push_status_update()

        return jsonify({"success": True, "message": f"Instance '{instance_id}' started"})

    except Exception as e:
        logger.error(f"Failed to start instance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/instances/<instance_id>/stop', methods=['POST'])
def stop_plugin_instance(instance_id: str):
    """Stop a plugin instance."""
    try:
        if not client:
            return jsonify({"error": "No client initialized"}), 400

        if instance_id not in client.plugins:
            return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

        plugin = client.plugins[instance_id]
        _stop_plugin_instance(plugin)
        _push_status_update()

        return jsonify({"success": True, "message": f"Instance '{instance_id}' stopped"})

    except Exception as e:
        logger.error(f"Failed to stop instance: {e}")
        return jsonify({"error": str(e)}), 500


def _create_plugin_instance(plugin_type_name: str, instance_id: str, display_name: str, config: dict):
    """Create a new plugin instance of the specified type via the registry."""
    return registry_create_plugin(
        plugin_type=plugin_type_name,
        config=config,
        instance_id=instance_id,
        display_name=display_name,
    )


def _start_plugin_instance(plugin):
    """Start a plugin instance in a background thread.

    Sets the data-changed callback and launches plugin.start() off the
    request thread so the HTTP response returns immediately.  Status
    updates are pushed over WebSocket when the background work finishes
    (or fails).
    """
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


def _stop_plugin_instance(plugin):
    """Stop a plugin instance via its lifecycle method."""
    plugin.stop()


@app.route('/api/plugins/configure', methods=['POST'])
def configure_plugins():
    """Configure and start plugins."""
    global client, discovery_status

    data = request.json
    selected_plugins = data.get('plugins', [])

    try:
        # Initialize client and status
        with state_lock:
            client = OpenCiteClient()
            discovery_status["error"] = None
            discovery_status["plugins_enabled"] = []
            discovery_status["progress"] = []
            discovery_status["current_status"] = "Initializing OpenCITE client..."
            discovery_status["progress"].append({"step": "init", "message": "OpenCITE client created", "status": "success"})

        # Configure each selected plugin (outside lock so status can be polled)
        for plugin_config in selected_plugins:
            plugin_name = plugin_config.get('name')
            config = plugin_config.get('config', {})

            try:
                with state_lock:
                    discovery_status["current_status"] = f"Configuring {plugin_name} plugin..."
                    discovery_status["progress"].append({
                        "step": plugin_name,
                        "message": f"Initializing {plugin_name} plugin",
                        "status": "in_progress",
                    })
                _push_status_update()

                # Check if plugin is already registered (reuse existing instance)
                existing_plugin = client.plugins.get(plugin_name)
                if existing_plugin is not None:
                    logger.info(f"Reusing existing {plugin_name} plugin")
                else:
                    # Unregister any auto-registered bare instance (e.g. from OpenCiteClient())
                    if plugin_name in client.plugins:
                        client.unregister_plugin(plugin_name)

                    # Create and register via registry
                    plugin_instance = registry_create_plugin(
                        plugin_type=plugin_name,
                        config=config,
                    )
                    client.register_plugin(plugin_instance)

                    # Start the plugin (background thread for heavy plugins)
                    _start_plugin_instance(plugin_instance)
                    logger.info(f"Configured and started {plugin_name} plugin")

                # Build success message with plugin-specific details
                success_msg = f"{plugin_name.replace('_', ' ').title()} plugin configured"
                plugin_obj = client.plugins.get(plugin_name)
                if plugin_obj and plugin_obj.plugin_type == 'opentelemetry':
                    local_ip = get_local_ip()
                    port = getattr(plugin_obj, 'port', 4318)
                    if local_ip != '127.0.0.1':
                        success_msg = (
                            f"Trace receiver ready - Send to http://localhost:{port}/v1/traces "
                            f"(same machine) or http://{local_ip}:{port}/v1/traces (other machines)"
                        )
                    else:
                        success_msg = f"Trace receiver ready at http://localhost:{port}/v1/traces"

                with state_lock:
                    discovery_status["progress"][-1]["status"] = "success"
                    discovery_status["progress"][-1]["message"] = success_msg
                    discovery_status["plugins_enabled"].append(plugin_name)
                _push_status_update()

            except Exception as e:
                logger.error(f"Failed to configure {plugin_name}: {e}")
                with state_lock:
                    discovery_status["progress"].append({"step": plugin_name, "message": f"Failed: {str(e)}", "status": "error"})
                    discovery_status["error"] = f"Failed to configure {plugin_name}: {str(e)}"
                    discovery_status["current_status"] = f"Error configuring {plugin_name}"
                _push_status_update()
                return jsonify({"error": f"Failed to configure {plugin_name}: {str(e)}"}), 400

        with state_lock:
            discovery_status["last_updated"] = datetime.utcnow().isoformat()
            discovery_status["running"] = True
            discovery_status["current_status"] = "Plugins configured - ready to discover assets"
            discovery_status["progress"].append({"step": "complete", "message": f"All plugins configured successfully ({len(discovery_status['plugins_enabled'])} active)", "status": "success"})
        _push_status_update()

        return jsonify({
            "success": True,
            "plugins_enabled": discovery_status["plugins_enabled"]
        })

    except Exception as e:
        logger.error(f"Failed to configure plugins: {e}")
        with state_lock:
            discovery_status["error"] = str(e)
            discovery_status["running"] = False
        return jsonify({"error": str(e)}), 500


@app.route('/api/assets', methods=['GET'])
def get_assets():
    """Get discovered assets."""
    global discovering_assets, asset_cache, asset_cache_time

    if not client:
        return jsonify({"error": "No client initialized. Please configure plugins first."}), 400

    try:
        # Get asset type filter
        asset_type = request.args.get('type', 'all')

        # Check if discovery is in progress
        if discovering_assets:
            # Return cached assets if available, otherwise empty
            if asset_cache:
                return jsonify(asset_cache)
            else:
                return jsonify({
                    "assets": {
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
                    },
                    "totals": {},
                    "timestamp": datetime.utcnow().isoformat(),
                    "discovering": True
                })

        # Check cache (WebSocket clients get instant pushes; REST fallback uses cache)
        if asset_cache and asset_cache_time:
            if datetime.utcnow() - asset_cache_time < timedelta(seconds=30):
                return jsonify(asset_cache)

        # Mark as discovering
        discovering_assets = True

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

        # Collect assets from all plugins (aggregated)
        if asset_type in ['all', 'tools']:
            assets["tools"] = client.list_tools()
        if asset_type in ['all', 'models']:
            assets["models"] = client.list_models()
        if asset_type in ['all', 'agents']:
            assets["agents"] = client.list_agents()
        if asset_type in ['all', 'downstream_systems']:
            assets["downstream_systems"] = client.list_downstream_systems()
        if asset_type in ['all', 'mcp_servers']:
            assets["mcp_servers"] = client.list_mcp_servers()
        if asset_type in ['all', 'mcp_tools']:
            assets["mcp_tools"] = client.list_mcp_tools()
        if asset_type in ['all', 'mcp_resources']:
            assets["mcp_resources"] = client.list_mcp_resources()

        # Databricks data assets - use export to get what will actually be exported
        if "databricks" in discovery_status["plugins_enabled"]:
            if asset_type in ['all', 'data_assets']:
                try:
                    # Update status to show we're discovering Databricks assets
                    with state_lock:
                        discovery_status["current_status"] = "Discovering Databricks assets (this may take a minute)..."
                        # Add or update progress item
                        progress_items = [p for p in discovery_status["progress"] if p["step"] != "discovery"]
                        progress_items.append({"step": "discovery", "message": "Querying MLflow experiments for AI-used tables...", "status": "in_progress"})
                        discovery_status["progress"] = progress_items

                    # Get what will actually be exported (AI-used tables via MLflow)
                    export_data = client._export_databricks_assets()
                    assets["data_assets"] = export_data

                    with state_lock:
                        # Remove discovery progress item after completion
                        discovery_status["progress"] = [p for p in discovery_status["progress"] if p["step"] != "discovery"]
                        discovery_status["current_status"] = "Discovery complete"

                except Exception as e:
                    logger.warning(f"Could not list Databricks data assets: {e}")
                    with state_lock:
                        # Update progress to show error
                        for p in discovery_status["progress"]:
                            if p["step"] == "discovery":
                                p["status"] = "error"
                                p["message"] = f"Failed: {str(e)}"
                        discovery_status["current_status"] = "Error discovering Databricks assets"

        # Google Cloud assets
        if "google_cloud" in discovery_status["plugins_enabled"]:
            if asset_type in ['all', 'gcp_models']:
                try:
                    models = client.list_gcp_models()
                    assets["gcp_models"] = models
                except Exception as e:
                    logger.warning(f"Could not list GCP models: {e}")
            if asset_type in ['all', 'gcp_endpoints']:
                try:
                    endpoints = client.list_gcp_endpoints()
                    assets["gcp_endpoints"] = endpoints
                except Exception as e:
                    logger.warning(f"Could not list GCP endpoints: {e}")


        # Move data-oriented downstream items into data_assets
        _reclassify_downstream(assets)

        # Count totals
        totals = {k: len(v) for k, v in assets.items()}

        # Cache the results
        result = {
            "assets": assets,
            "totals": totals,
            "timestamp": datetime.utcnow().isoformat()
        }
        asset_cache = result
        asset_cache_time = datetime.utcnow()

        # Reset discovery flag
        discovering_assets = False

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting assets: {e}")
        discovering_assets = False  # Reset flag on error
        return jsonify({"error": str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_data():
    """Export discovered data to OpenCITE JSON format."""
    if not client:
        return jsonify({"error": "No client initialized"}), 400

    try:
        data = request.json
        include_plugins = data.get('plugins', [])

        export_data = client.export_to_json(
            include_otel="opentelemetry" in include_plugins,
            include_mcp="mcp" in include_plugins,
            include_databricks="databricks" in include_plugins,
            include_google_cloud="google_cloud" in include_plugins
        )


        return jsonify(export_data)

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/map-tool', methods=['POST'])
def map_tool():
    """Save a new tool mapping."""
    try:
        data = request.json
        plugin_name = data.get('plugin_name')
        attributes = data.get('attributes')
        identity = data.get('identity')
        match_type = data.get('match_type', 'all')

        if not all([plugin_name, attributes, identity]):
            return jsonify({"error": "Missing required fields"}), 400

        # Generate a stable source_id as UUIDv5
        OPENCITE_NS = uuid.UUID('a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d')
        source_name = identity.get('source_name', '')
        identity['source_id'] = str(uuid.uuid5(OPENCITE_NS, f"{plugin_name}:{source_name}"))

        success = tool_identifier.add_mapping(plugin_name, attributes, identity, match_type=match_type)

        if success:
            # Clear asset cache to show the updated identification immediately
            global asset_cache, asset_cache_time
            asset_cache = None
            asset_cache_time = None
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save mapping"}), 500

    except Exception as e:
        logger.error(f"Failed to map tool: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/instances/<instance_id>/webhooks', methods=['GET'])
def list_webhooks(instance_id: str):
    """List subscribed webhook URLs for a plugin instance."""
    if not client or instance_id not in client.plugins:
        return jsonify({"error": f"Instance '{instance_id}' not found"}), 404
    plugin = client.plugins[instance_id]
    return jsonify({"webhooks": plugin.list_webhooks()})


@app.route('/api/instances/<instance_id>/webhooks', methods=['POST'])
def subscribe_webhook(instance_id: str):
    """Subscribe a webhook URL to receive OTLP trace payloads."""
    if not client or instance_id not in client.plugins:
        return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

    data = request.json or {}
    url = data.get("url", "").strip()
    if not url or not url.startswith(("http://", "https://")):
        return jsonify({"error": "A valid http:// or https:// URL is required"}), 400

    plugin = client.plugins[instance_id]
    added = plugin.subscribe_webhook(url)
    return jsonify({"success": True, "added": added, "webhooks": plugin.list_webhooks()})


@app.route('/api/instances/<instance_id>/webhooks', methods=['DELETE'])
def unsubscribe_webhook(instance_id: str):
    """Unsubscribe a webhook URL."""
    if not client or instance_id not in client.plugins:
        return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    plugin = client.plugins[instance_id]
    removed = plugin.unsubscribe_webhook(url)
    return jsonify({"success": True, "removed": removed, "webhooks": plugin.list_webhooks()})


@app.route('/api/lineage-graph')
def lineage_graph():
    """Generate a pyvis lineage graph as embeddable HTML."""
    from pyvis.network import Network

    if not client:
        return "<html><body><p>No data yet</p></body></html>", 200, {'Content-Type': 'text/html'}

    net = Network(
        height="350",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
    )

    net.set_options("""{
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 250,
                "nodeSpacing": 120
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 150
            }
        },
        "edges": {
            "arrows": { "to": { "enabled": true } },
            "smooth": { "type": "cubicBezier" }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }""")

    # Collect current assets
    agents = client.list_agents()
    tools = client.list_tools()
    models = client.list_models()
    downstream = client.list_downstream_systems()

    data_assets_list = []
    if asset_cache:
        data_assets_list = asset_cache.get("assets", {}).get("data_assets", [])

    temp = {"downstream_systems": downstream, "data_assets": data_assets_list}
    _reclassify_downstream(temp)
    downstream = temp["downstream_systems"]
    data_assets_list = temp["data_assets"]

    added = set()

    # type -> (color, shape, level)
    STYLES = {
        "agent":      ("#8b5cf6", "diamond",  0),
        "tool":       ("#3b82f6", "dot",       1),
        "model":      ("#10b981", "star",      2),
        "data_asset": ("#f59e0b", "database",  3),
        "downstream": ("#ef4444", "triangle",  3),
    }

    def add_node(nid, label, ntype, title=None):
        if nid in added:
            return
        added.add(nid)
        color, shape, level = STYLES.get(ntype, ("#6b7280", "dot", 1))
        net.add_node(nid, label=label, color=color, shape=shape, level=level,
                     title=title or f"{ntype}: {label}", size=20)

    # Agent nodes + edges
    for a in agents:
        nid = f"agent:{a['name']}"
        add_node(nid, a["name"], "agent", f"Agent: {a['name']}")
        for t in (a.get("tools_used") or []):
            tid = f"tool:{t}"
            add_node(tid, t, "tool")
            net.add_edge(nid, tid)
        for m in (a.get("models_used") or []):
            mid = f"model:{m}"
            add_node(mid, m, "model")
            net.add_edge(nid, mid)

    # Tool nodes + model edges
    for t in tools:
        tid = f"tool:{t['name']}"
        add_node(tid, t["name"], "tool")
        for m in (t.get("models") or []):
            mid = f"model:{m}"
            add_node(mid, m, "model")
            net.add_edge(tid, mid)

    # Model nodes (ensure present even if no edges)
    for m in models:
        mid = f"model:{m['name']}"
        add_node(mid, m["name"], "model", f"Model: {m['name']}\nProvider: {m.get('provider','')}")

    # Data assets + edges from tools_connecting
    for d in data_assets_list:
        did = f"data:{d['name']}"
        add_node(did, d["name"], "data_asset", f"Data: {d['name']}\nType: {d.get('type','')}")
        for t in (d.get("tools_connecting") or []):
            tid = f"tool:{t}"
            add_node(tid, t, "tool")
            net.add_edge(tid, did)

    # Downstream + edges from tools_connecting
    for d in downstream:
        did = f"ds:{d['name']}"
        add_node(did, d["name"], "downstream", f"Downstream: {d['name']}\nType: {d.get('type','')}")
        for t in (d.get("tools_connecting") or []):
            tid = f"tool:{t}"
            add_node(tid, t, "tool")
            net.add_edge(tid, did)

    if not added:
        return ("<html><body style='display:flex;align-items:center;justify-content:center;"
                "height:100%;color:#6b7280;font-family:sans-serif'>"
                "<p>No lineage data yet</p></body></html>"), 200, {'Content-Type': 'text/html'}

    html = net.generate_html()

    # Inject postMessage listener so parent page can focus/select nodes
    focus_script = """<script>
window.addEventListener('message', function(e) {
    if (e.data && e.data.type === 'focusNode' && typeof network !== 'undefined') {
        var nid = e.data.nodeId;
        try {
            var ids = network.body.data.nodes.getIds();
            if (ids.indexOf(nid) !== -1) {
                network.focus(nid, {scale: 1.5, animation: {duration: 800, easingFunction: 'easeInOutQuad'}});
                network.selectNodes([nid]);
            }
        } catch(err) {}
    }
});
</script>"""
    html = html.replace('</body>', focus_script + '</body>')

    return html, 200, {'Content-Type': 'text/html'}


@app.route('/api/stop', methods=['POST'])
def stop_discovery():
    """Stop discovery and clean up."""
    global client, discovery_status

    try:
        if client:
            # Stop all running plugins via lifecycle methods
            for plugin in list(client.plugins.values()):
                if plugin.status == 'running':
                    try:
                        plugin.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping plugin {plugin.instance_id}: {e}")

            client = None

        with state_lock:
            discovery_status["running"] = False
            discovery_status["plugins_enabled"] = []
            discovery_status["last_updated"] = datetime.utcnow().isoformat()
            discovery_status["current_status"] = "Idle"
            discovery_status["progress"] = []
            discovery_status["error"] = None
        _push_status_update()

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
