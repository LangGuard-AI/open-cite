"""
OpenCITE Web GUI - Flask Application

A web-based interface for OpenCITE discovery and visualization.
"""

import os
import json
import logging
import socket
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, send_from_directory
from threading import Thread, Lock

from open_cite.client import OpenCiteClient
import importlib
import inspect
import pkgutil
import open_cite.plugins

logger = logging.getLogger(__name__)


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


def discover_available_plugins():
    """
    Dynamically discover all available plugins.

    Returns a dictionary mapping plugin names to their metadata.
    """
    plugins = {}

    # Discover all plugin modules
    plugin_package = open_cite.plugins
    for importer, modname, ispkg in pkgutil.iter_modules(plugin_package.__path__):
        try:
            # Import the plugin module
            module = importlib.import_module(f'open_cite.plugins.{modname}')

            # Find plugin classes (classes that inherit from BaseDiscoveryPlugin)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip imported base classes
                if obj.__module__ != module.__name__:
                    continue

                # Check if it's a plugin (has 'name' property and inherits from BaseDiscoveryPlugin)
                if hasattr(obj, 'name') and hasattr(obj, 'list_assets'):
                    # Get plugin metadata
                    plugin_info = get_plugin_metadata(modname, obj)
                    if plugin_info:
                        plugins[modname] = plugin_info
                        logger.info(f"Discovered plugin: {modname}")
                    break  # Only take first plugin class from module

        except Exception as e:
            logger.warning(f"Failed to load plugin module {modname}: {e}")
            continue

    return plugins


def get_plugin_metadata(plugin_name, plugin_class):
    """
    Extract metadata for a plugin based on its name and class.

    Returns plugin configuration including required fields and env vars.
    """
    metadata = {
        "name": plugin_name.replace('_', ' ').title(),
        "description": "Plugin for discovering assets",
        "required_fields": {},
        "env_vars": []
    }

    # Customize metadata based on plugin name
    if plugin_name == "opentelemetry":
        local_ip = get_local_ip()
        metadata["name"] = "OpenTelemetry"
        metadata["description"] = "Discovers AI tools using models via OTLP traces"
        # No configuration needed - trace receiver always starts when plugin is enabled
        metadata["required_fields"] = {}
        # Store endpoint info for display
        metadata["trace_endpoints"] = {
            "localhost": "http://localhost:4318/v1/traces",
            "network": f"http://{local_ip}:4318/v1/traces" if local_ip != '127.0.0.1' else None
        }

    elif plugin_name == "mcp":
        metadata["name"] = "MCP (Model Context Protocol)"
        metadata["description"] = "Discovers MCP servers, tools, and resources from OTLP traces"

    elif plugin_name == "databricks":
        metadata["name"] = "Databricks"
        metadata["description"] = "Discovers AI/ML tables from MLflow experiments"
        metadata["required_fields"] = {
            "host": {"label": "Host", "default": "https://dbc-xxx.cloud.databricks.com", "required": True},
            "token": {"label": "Token", "default": "", "required": True, "type": "password"},
            "warehouse_id": {"label": "Warehouse ID (optional)", "default": "", "required": False}
        }
        metadata["env_vars"] = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_WAREHOUSE_ID"]

    elif plugin_name == "google_cloud":
        metadata["name"] = "Google Cloud"
        metadata["description"] = "Discovers Vertex AI models and endpoints"
        metadata["required_fields"] = {
            "project_id": {"label": "Project ID", "default": "", "required": True},
            "location": {"label": "Location", "default": "us-central1", "required": False}
        }
        metadata["env_vars"] = ["GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"]

    return metadata


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
                if plugin_name == 'opentelemetry':
                    with state_lock:
                        discovery_status["current_status"] = f"Configuring OpenTelemetry plugin..."
                        discovery_status["progress"].append({"step": plugin_name, "message": "Initializing OpenTelemetry plugin", "status": "in_progress"})

                    # Always bind to 0.0.0.0 to accept connections from any interface
                    host = '0.0.0.0'
                    port = 4318

                    # Get local IP for display
                    local_ip = get_local_ip()

                    # Get MCP plugin if already registered for integration
                    mcp_plugin = client.plugins.get('mcp')

                    # Check if OpenTelemetry plugin is already registered and running
                    otel_plugin = client.plugins.get('opentelemetry')
                    if otel_plugin is None:
                        # Create new plugin instance only if one doesn't exist
                        from open_cite.plugins.opentelemetry import OpenTelemetryPlugin
                        otel_plugin = OpenTelemetryPlugin(
                            host=host,
                            port=port,
                            mcp_plugin=mcp_plugin
                        )
                        client.register_plugin(otel_plugin)

                        with state_lock:
                            if local_ip != '127.0.0.1':
                                discovery_status["progress"][-1]["message"] = f"Starting trace receiver (accessible from http://localhost:{port}/v1/traces or http://{local_ip}:{port}/v1/traces)..."
                            else:
                                discovery_status["progress"][-1]["message"] = f"Starting trace receiver (http://localhost:{port}/v1/traces)..."

                        otel_plugin.start_receiver()
                        logger.info(f"Started OpenTelemetry plugin on {host}:{port}")
                    else:
                        # Reuse existing plugin instance
                        logger.info(f"Reusing existing OpenTelemetry plugin on {host}:{port}")

                    with state_lock:
                        discovery_status["progress"][-1]["status"] = "success"
                        if local_ip != '127.0.0.1':
                            discovery_status["progress"][-1]["message"] = f"✓ Trace receiver ready - Send to http://localhost:{port}/v1/traces (same machine) or http://{local_ip}:{port}/v1/traces (other machines)"
                        else:
                            discovery_status["progress"][-1]["message"] = f"✓ Trace receiver ready at http://localhost:{port}/v1/traces"

                elif plugin_name == 'mcp':
                    with state_lock:
                        discovery_status["current_status"] = f"Configuring MCP plugin..."
                        discovery_status["progress"].append({"step": plugin_name, "message": "Registering MCP plugin", "status": "in_progress"})

                    # Dynamically import MCPPlugin
                    from open_cite.plugins.mcp import MCPPlugin
                    mcp_plugin = MCPPlugin()
                    client.register_plugin(mcp_plugin)
                    logger.info("Registered MCP plugin")

                    with state_lock:
                        discovery_status["progress"][-1]["status"] = "success"
                        discovery_status["progress"][-1]["message"] = "MCP plugin registered successfully"

                elif plugin_name == 'databricks':
                    with state_lock:
                        discovery_status["current_status"] = f"Configuring Databricks plugin..."
                        discovery_status["progress"].append({"step": plugin_name, "message": "Initializing Databricks plugin", "status": "in_progress"})

                    # Get config values (None if not provided, allowing fallback to env vars)
                    host = config.get('host') or None
                    token = config.get('token') or None
                    warehouse_id = config.get('warehouse_id') or None

                    # Debug logging
                    logger.info(f"[Databricks Config] host={'SET' if host else 'NONE'}, token={'SET' if token else 'NONE'}, warehouse_id={'SET' if warehouse_id else 'NONE'}")
                    logger.info(f"[Databricks Config] config keys: {list(config.keys())}")

                    with state_lock:
                        discovery_status["progress"][-1]["message"] = "Connecting to Databricks workspace..."

                    # Dynamically import DatabricksPlugin
                    # Plugin will validate and fall back to environment variables if needed
                    from open_cite.plugins.databricks import DatabricksPlugin
                    databricks_plugin = DatabricksPlugin(
                        host=host,
                        token=token,
                        warehouse_id=warehouse_id
                    )

                    with state_lock:
                        discovery_status["progress"][-1]["message"] = "Registering Databricks plugin..."

                    client.register_plugin(databricks_plugin)
                    logger.info("Registered Databricks plugin")

                    with state_lock:
                        discovery_status["progress"][-1]["status"] = "success"
                        if warehouse_id:
                            discovery_status["progress"][-1]["message"] = f"Databricks plugin configured with warehouse {warehouse_id}"
                        else:
                            discovery_status["progress"][-1]["message"] = "Databricks plugin configured (warehouse will be auto-discovered)"

                elif plugin_name == 'google_cloud':
                    with state_lock:
                        discovery_status["current_status"] = f"Configuring Google Cloud plugin..."
                        discovery_status["progress"].append({"step": plugin_name, "message": "Initializing Google Cloud plugin", "status": "in_progress"})

                    project_id = config.get('project_id')
                    location = config.get('location', 'us-central1')

                    if not project_id:
                        raise ValueError("Google Cloud requires project_id")

                    with state_lock:
                        discovery_status["progress"][-1]["message"] = f"Connecting to GCP project {project_id}..."

                    # Dynamically import GoogleCloudPlugin
                    from open_cite.plugins.google_cloud import GoogleCloudPlugin
                    gcp_plugin = GoogleCloudPlugin(
                        project_id=project_id,
                        location=location
                    )
                    client.register_plugin(gcp_plugin)
                    logger.info("Registered Google Cloud plugin")

                    with state_lock:
                        discovery_status["progress"][-1]["status"] = "success"
                        discovery_status["progress"][-1]["message"] = f"Google Cloud plugin configured for project {project_id}"


                with state_lock:
                    discovery_status["plugins_enabled"].append(plugin_name)

            except Exception as e:
                logger.error(f"Failed to configure {plugin_name}: {e}")
                with state_lock:
                    discovery_status["progress"].append({"step": plugin_name, "message": f"Failed: {str(e)}", "status": "error"})
                    discovery_status["error"] = f"Failed to configure {plugin_name}: {str(e)}"
                    discovery_status["current_status"] = f"Error configuring {plugin_name}"
                return jsonify({"error": f"Failed to configure {plugin_name}: {str(e)}"}), 400

        with state_lock:
            discovery_status["last_updated"] = datetime.utcnow().isoformat()
            discovery_status["running"] = True
            discovery_status["current_status"] = "Plugins configured - ready to discover assets"
            discovery_status["progress"].append({"step": "complete", "message": f"All plugins configured successfully ({len(discovery_status['plugins_enabled'])} active)", "status": "success"})

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

        # Check cache (refresh every 2 seconds for real-time OTLP discoveries)
        from datetime import timedelta
        if asset_cache and asset_cache_time:
            if datetime.utcnow() - asset_cache_time < timedelta(seconds=2):
                return jsonify(asset_cache)

        # Mark as discovering
        discovering_assets = True

        assets = {
            "tools": [],
            "models": [],
            "mcp_servers": [],
            "mcp_tools": [],
            "mcp_resources": [],
            "data_assets": [],
            "gcp_models": [],
            "gcp_endpoints": []
        }

        # OpenTelemetry assets
        if "opentelemetry" in discovery_status["plugins_enabled"]:
            if asset_type in ['all', 'tools']:
                assets["tools"] = client.list_otel_tools()
            if asset_type in ['all', 'models']:
                assets["models"] = client.list_otel_models()

        # MCP assets
        if "mcp" in discovery_status["plugins_enabled"]:
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


@app.route('/api/stop', methods=['POST'])
def stop_discovery():
    """Stop discovery and clean up."""
    global client, discovery_status

    try:
        if client:
            # Stop OpenTelemetry receiver if running
            if 'opentelemetry' in discovery_status.get("plugins_enabled", []):
                otel_plugin = client.plugins.get('opentelemetry')
                if otel_plugin and hasattr(otel_plugin, 'stop_receiver'):
                    try:
                        otel_plugin.stop_receiver()
                    except Exception as e:
                        logger.warning(f"Error stopping OpenTelemetry receiver: {e}")


            client = None

        with state_lock:
            discovery_status["running"] = False
            discovery_status["plugins_enabled"] = []
            discovery_status["last_updated"] = datetime.utcnow().isoformat()
            discovery_status["current_status"] = "Idle"
            discovery_status["progress"] = []
            discovery_status["error"] = None

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

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_gui(debug=True)
