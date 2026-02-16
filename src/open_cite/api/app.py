"""
OpenCITE Headless API Service.

A REST API for OpenCITE discovery and inventory capabilities,
designed for deployment in Kubernetes without a GUI.
"""

import os
import json
import logging
import socket
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from threading import Lock

from open_cite.client import OpenCiteClient
from .config import OpenCiteConfig
from .health import health_bp, init_health
from .shutdown import register_shutdown_handler
from .persistence import PersistenceManager

from open_cite.plugins.registry import get_all_plugin_metadata, create_plugin_instance

logger = logging.getLogger(__name__)

# Global state
client: Optional[OpenCiteClient] = None
persistence: Optional[PersistenceManager] = None
discovery_status = {
    "running": False,
    "plugins_enabled": [],
    "last_updated": None,
    "error": None,
    "current_status": "Idle",
    "progress": []
}
state_lock = Lock()
discovering_assets = False
asset_cache = None
asset_cache_time = None


def get_persistence() -> Optional[PersistenceManager]:
    """Get the persistence manager instance."""
    return persistence


def get_client() -> Optional[OpenCiteClient]:
    """Get the current client instance."""
    return client


def get_status() -> Dict[str, Any]:
    """Get the current discovery status."""
    with state_lock:
        return discovery_status.copy()


def get_local_ip():
    """Discover the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP: {e}")
        return '127.0.0.1'


def discover_available_plugins():
    """Dynamically discover all available plugins via the registry."""
    return get_all_plugin_metadata()


def create_app(config: Optional[OpenCiteConfig] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config: Optional OpenCiteConfig instance. If None, loads from environment.

    Returns:
        Configured Flask application
    """
    global persistence

    if config is None:
        config = OpenCiteConfig.from_env()

    # Configure the open_cite logger hierarchy directly instead of
    # logging.basicConfig() which touches the root logger.  Gunicorn adds
    # its own handler to the root logger *after* create_app() returns, so
    # using basicConfig causes every log line to be emitted twice (once by
    # our handler, once by gunicorn's).  Setting propagate=False on the
    # open_cite logger prevents this duplication.
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    oc_logger = logging.getLogger('open_cite')
    oc_logger.setLevel(log_level)
    if not oc_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        oc_logger.addHandler(handler)
    oc_logger.propagate = False

    # Initialize persistence if enabled
    if config.persistence_enabled:
        persistence = PersistenceManager(config.db_path)
        logger.info(f"Persistence enabled: {config.db_path}")
    else:
        logger.info("Persistence disabled (in-memory only)")

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)

    # Register health check blueprint
    app.register_blueprint(health_bp)

    # Initialize health check dependencies
    init_health(get_client, get_status)

    # Register API routes
    register_api_routes(app)

    # Store config in app context
    app.opencite_config = config

    # Auto-configure plugins from environment when auto_start is enabled.
    # This is needed for gunicorn which calls create_app() directly
    # rather than going through run_api().
    if config.auto_start:
        auto_configure_plugins(app)

    return app


def register_api_routes(app: Flask):
    """Register all API routes on the Flask app."""

    @app.route('/api/v1/status')
    def api_get_status():
        """Get current discovery status."""
        with state_lock:
            return jsonify(discovery_status)

    @app.route('/api/v1/plugins', methods=['GET'])
    def api_list_plugins():
        """List available plugins and their requirements."""
        try:
            plugins = discover_available_plugins()
            return jsonify(plugins)
        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/plugins/configure', methods=['POST'])
    def api_configure_plugins():
        """Configure and start plugins."""
        global client, discovery_status

        data = request.json
        selected_plugins = data.get('plugins', [])

        try:
            with state_lock:
                client = OpenCiteClient()
                discovery_status["error"] = None
                discovery_status["plugins_enabled"] = []
                discovery_status["progress"] = []
                discovery_status["current_status"] = "Initializing OpenCITE client..."
                discovery_status["progress"].append({
                    "step": "init",
                    "message": "OpenCITE client created",
                    "status": "success"
                })

            for plugin_config in selected_plugins:
                plugin_name = plugin_config.get('name')
                config = plugin_config.get('config', {})

                try:
                    _configure_plugin(plugin_name, config)
                    with state_lock:
                        discovery_status["plugins_enabled"].append(plugin_name)
                except Exception as e:
                    logger.error(f"Failed to configure {plugin_name}: {e}")
                    with state_lock:
                        discovery_status["progress"].append({
                            "step": plugin_name,
                            "message": f"Failed: {str(e)}",
                            "status": "error"
                        })
                        discovery_status["error"] = f"Failed to configure {plugin_name}: {str(e)}"
                        discovery_status["current_status"] = f"Error configuring {plugin_name}"
                    return jsonify({"error": f"Failed to configure {plugin_name}: {str(e)}"}), 400

            with state_lock:
                discovery_status["last_updated"] = datetime.utcnow().isoformat()
                discovery_status["running"] = True
                discovery_status["current_status"] = "Plugins configured - ready to discover assets"
                discovery_status["progress"].append({
                    "step": "complete",
                    "message": f"All plugins configured successfully ({len(discovery_status['plugins_enabled'])} active)",
                    "status": "success"
                })

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

    @app.route('/api/v1/assets', methods=['GET'])
    def api_get_assets():
        """Get discovered assets."""
        global discovering_assets, asset_cache, asset_cache_time

        if not client:
            return jsonify({"error": "No client initialized. Please configure plugins first."}), 400

        try:
            asset_type = request.args.get('type', 'all')

            if discovering_assets:
                if asset_cache:
                    return jsonify(asset_cache)
                else:
                    return jsonify({
                        "assets": _empty_assets(),
                        "totals": {},
                        "timestamp": datetime.utcnow().isoformat(),
                        "discovering": True
                    })

            if asset_cache and asset_cache_time:
                if datetime.utcnow() - asset_cache_time < timedelta(seconds=2):
                    return jsonify(asset_cache)

            discovering_assets = True
            assets = _collect_assets(asset_type)
            totals = {k: len(v) for k, v in assets.items()}

            result = {
                "assets": assets,
                "totals": totals,
                "timestamp": datetime.utcnow().isoformat()
            }
            asset_cache = result
            asset_cache_time = datetime.utcnow()
            discovering_assets = False

            return jsonify(result)

        except Exception as e:
            logger.error(f"Error getting assets: {e}")
            discovering_assets = False
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/tools', methods=['GET'])
    def api_list_tools():
        """List discovered tools."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400
        try:
            tools = client.list_tools()
            return jsonify({"tools": tools, "count": len(tools)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/models', methods=['GET'])
    def api_list_models():
        """List discovered models."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400
        try:
            models = client.list_models()
            return jsonify({"models": models, "count": len(models)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/mcp/servers', methods=['GET'])
    def api_list_mcp_servers():
        """List MCP servers."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400
        try:
            servers = client.list_mcp_servers()
            return jsonify({"servers": servers, "count": len(servers)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/mcp/tools', methods=['GET'])
    def api_list_mcp_tools():
        """List MCP tools."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400
        try:
            server_id = request.args.get('server_id')
            tools = client.list_mcp_tools(server_id=server_id)
            return jsonify({"tools": tools, "count": len(tools)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/agents', methods=['GET'])
    def api_list_agents():
        """List discovered agents."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400
        try:
            agents = client.list_agents()
            return jsonify({"agents": agents, "count": len(agents)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/downstream', methods=['GET'])
    def api_list_downstream():
        """List discovered downstream systems."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400
        try:
            systems = client.list_downstream_systems()
            return jsonify({"downstream_systems": systems, "count": len(systems)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/lineage', methods=['GET'])
    def api_list_lineage():
        """List lineage relationships."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400
        try:
            source_id = request.args.get('source_id')
            relationships = client.list_lineage(source_id=source_id)
            return jsonify({"relationships": relationships, "count": len(relationships)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/export', methods=['POST'])
    def api_export():
        """Export discovered data to OpenCITE JSON format."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400

        try:
            data = request.json or {}
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

    @app.route('/api/v1/map-tool', methods=['POST'])
    def api_map_tool():
        """Save a new tool mapping."""
        if not client:
            return jsonify({"error": "No client initialized"}), 400

        try:
            data = request.json
            plugin_name = data.get('plugin_name')
            attributes = data.get('attributes')
            identity = data.get('identity')
            match_type = data.get('match_type', 'all')

            if not all([plugin_name, attributes, identity]):
                return jsonify({"error": "Missing required fields"}), 400

            plugin = client.plugins.get(plugin_name)
            if not plugin:
                return jsonify({"error": f"Plugin {plugin_name} not found"}), 404

            if not hasattr(plugin, 'identifier'):
                return jsonify({"error": f"Plugin {plugin_name} does not support mapping"}), 400

            success = plugin.identifier.add_mapping(
                plugin_name, attributes, identity, match_type=match_type
            )

            if success:
                global asset_cache, asset_cache_time
                asset_cache = None
                asset_cache_time = None
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Failed to save mapping"}), 500

        except Exception as e:
            logger.error(f"Failed to map tool: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/stop', methods=['POST'])
    def api_stop():
        """Stop discovery and clean up."""
        global client, discovery_status

        try:
            # Save state before stopping if persistence is enabled
            if persistence:
                _save_current_state()

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

            return jsonify({"success": True})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Persistence API endpoints
    # =========================================================================

    @app.route('/api/v1/persistence/status', methods=['GET'])
    def api_persistence_status():
        """Get persistence status."""
        if not persistence:
            return jsonify({
                "enabled": False,
                "message": "Persistence is disabled"
            })

        try:
            tools = persistence.load_tools()
            models = persistence.load_models()
            agents = persistence.load_agents()
            downstream = persistence.load_downstream_systems()
            lineage = persistence.load_lineage()
            mcp_servers = persistence.load_mcp_servers()
            mcp_tools = persistence.load_mcp_tools()
            mcp_resources = persistence.load_mcp_resources()
            mappings = persistence.load_mappings()

            return jsonify({
                "enabled": True,
                "db_path": str(persistence.db_path),
                "stats": {
                    "tools": len(tools),
                    "models": len(models),
                    "agents": len(agents),
                    "downstream_systems": len(downstream),
                    "lineage_relationships": len(lineage),
                    "mcp_servers": len(mcp_servers),
                    "mcp_tools": len(mcp_tools),
                    "mcp_resources": len(mcp_resources),
                    "mappings": len(mappings),
                }
            })
        except Exception as e:
            return jsonify({"enabled": True, "error": str(e)}), 500

    @app.route('/api/v1/persistence/save', methods=['POST'])
    def api_persistence_save():
        """Manually save current state to persistence."""
        if not persistence:
            return jsonify({"error": "Persistence is disabled"}), 400

        try:
            _save_current_state()
            return jsonify({"success": True, "message": "State saved to persistence"})
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/persistence/load', methods=['POST'])
    def api_persistence_load():
        """Manually load persisted data into current state."""
        if not persistence:
            return jsonify({"error": "Persistence is disabled"}), 400

        if not client:
            return jsonify({"error": "No client initialized"}), 400

        try:
            _load_persisted_data()
            return jsonify({"success": True, "message": "State loaded from persistence"})
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/persistence/export', methods=['GET'])
    def api_persistence_export():
        """Export all persisted data as JSON."""
        if not persistence:
            return jsonify({"error": "Persistence is disabled"}), 400

        try:
            data = persistence.export_all()
            return jsonify(data)
        except Exception as e:
            logger.error(f"Failed to export persistence data: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/persistence/clear', methods=['POST'])
    def api_persistence_clear():
        """Clear all persisted data."""
        if not persistence:
            return jsonify({"error": "Persistence is disabled"}), 400

        try:
            persistence.clear_all()
            return jsonify({"success": True, "message": "Persistence data cleared"})
        except Exception as e:
            logger.error(f"Failed to clear persistence data: {e}")
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Plugin Instance Management API endpoints (Multi-instance support)
    # =========================================================================

    @app.route('/api/v1/plugin-types', methods=['GET'])
    def api_list_plugin_types():
        """
        List available plugin types and their metadata.

        Returns plugin type info including:
        - name, description, required fields
        - supports_multiple_instances flag
        """
        try:
            plugins = discover_available_plugins()
            # Add supports_multiple_instances info
            for plugin_name, plugin_info in plugins.items():
                plugin_info['supports_multiple_instances'] = True  # All plugins support multi-instance now
                plugin_info['plugin_type'] = plugin_name
            return jsonify({"plugin_types": plugins})
        except Exception as e:
            logger.error(f"Failed to list plugin types: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/instances', methods=['GET'])
    def api_list_instances():
        """
        List all configured plugin instances.

        Query params:
        - plugin_type: Filter by plugin type (optional)
        """
        plugin_type = request.args.get('plugin_type')

        # If client is running, get live instances
        if client:
            instances = client.list_plugin_instances()
            if plugin_type:
                instances = [i for i in instances if i['plugin_type'] == plugin_type]
            return jsonify({"instances": instances, "count": len(instances)})

        # Fall back to persisted instances if no client
        if persistence:
            try:
                instances = persistence.load_plugin_instances(plugin_type)
                return jsonify({"instances": instances, "count": len(instances)})
            except Exception as e:
                logger.error(f"Failed to load instances from persistence: {e}")
                return jsonify({"error": str(e)}), 500

        return jsonify({"instances": [], "count": 0})

    @app.route('/api/v1/instances', methods=['POST'])
    def api_create_instance():
        """
        Create a new plugin instance.

        Request body:
        {
            "plugin_type": "databricks",
            "instance_id": "databricks-prod",  // Optional, auto-generated if not provided
            "display_name": "Production Databricks",
            "config": {
                "host": "https://dbc-xxx.cloud.databricks.com",
                "token": "dapi...",
                ...
            },
            "auto_start": true
        }
        """
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

            # Generate instance_id as UUIDv5 if not provided
            if not instance_id:
                OPENCITE_NS = uuid.UUID('a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d')
                id_name = f"{plugin_type}:{display_name or plugin_type}"
                instance_id = str(uuid.uuid5(OPENCITE_NS, id_name))

            # Auto-generate display_name if not provided
            if not display_name:
                display_name = plugin_type.replace('_', ' ').title()

            # Create and register the plugin instance
            plugin_instance = _create_plugin_instance(
                plugin_type, instance_id, display_name, config
            )

            if not client:
                # Initialize client if not already running
                client = OpenCiteClient()

            # Register the plugin
            client.register_plugin(plugin_instance)

            # Save to persistence if enabled
            # Persist the raw config (not plugin.get_config() which masks tokens)
            # so that the instance can be restored from persistence after restart.
            if persistence:
                persistence.save_plugin_instance(
                    instance_id=instance_id,
                    plugin_type=plugin_type,
                    display_name=display_name,
                    config=config,
                    status='running' if auto_start else 'stopped',
                    auto_start=auto_start,
                )

            # Start the plugin if auto_start is enabled
            if auto_start:
                _start_plugin_instance(plugin_instance)

            return jsonify({
                "success": True,
                "instance": plugin_instance.to_dict()
            }), 201

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Failed to create plugin instance: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/instances/<instance_id>', methods=['GET'])
    def api_get_instance(instance_id: str):
        """Get details of a specific plugin instance."""
        # Try live instance first
        if client and instance_id in client.plugins:
            plugin = client.plugins[instance_id]
            return jsonify({"instance": plugin.to_dict()})

        # Fall back to persistence
        if persistence:
            instance = persistence.get_plugin_instance(instance_id)
            if instance:
                return jsonify({"instance": instance})

        return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

    @app.route('/api/v1/instances/<instance_id>', methods=['PUT'])
    def api_update_instance(instance_id: str):
        """
        Update a plugin instance configuration.

        Request body:
        {
            "display_name": "New Name",  // Optional
            "config": {...},  // Optional, merged with existing
            "auto_start": true  // Optional
        }
        """
        try:
            data = request.json

            # Get existing instance
            if client and instance_id in client.plugins:
                plugin = client.plugins[instance_id]

                # Update display_name if provided
                if 'display_name' in data:
                    plugin.display_name = data['display_name']

                # Update persistence if enabled
                if persistence:
                    existing = persistence.get_plugin_instance(instance_id)
                    if existing:
                        new_display_name = data.get('display_name', existing['display_name'])
                        new_config = {**existing['config'], **data.get('config', {})}
                        new_auto_start = data.get('auto_start', existing['auto_start'])

                        persistence.save_plugin_instance(
                            instance_id=instance_id,
                            plugin_type=existing['plugin_type'],
                            display_name=new_display_name,
                            config=new_config,
                            status=existing['status'],
                            auto_start=new_auto_start,
                        )

                return jsonify({"success": True, "instance": plugin.to_dict()})

            elif persistence:
                existing = persistence.get_plugin_instance(instance_id)
                if existing:
                    new_display_name = data.get('display_name', existing['display_name'])
                    new_config = {**existing['config'], **data.get('config', {})}
                    new_auto_start = data.get('auto_start', existing['auto_start'])

                    persistence.save_plugin_instance(
                        instance_id=instance_id,
                        plugin_type=existing['plugin_type'],
                        display_name=new_display_name,
                        config=new_config,
                        status=existing['status'],
                        auto_start=new_auto_start,
                    )
                    updated = persistence.get_plugin_instance(instance_id)
                    return jsonify({"success": True, "instance": updated})

            return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

        except Exception as e:
            logger.error(f"Failed to update instance: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/instances/<instance_id>', methods=['DELETE'])
    def api_delete_instance(instance_id: str):
        """Delete a plugin instance."""
        try:
            # Stop and unregister from client if running
            if client and instance_id in client.plugins:
                plugin = client.plugins[instance_id]
                _stop_plugin_instance(plugin)
                client.unregister_plugin(instance_id)

            # Remove from persistence
            if persistence:
                persistence.delete_plugin_instance(instance_id)

            return jsonify({"success": True, "message": f"Instance '{instance_id}' deleted"})

        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            logger.error(f"Failed to delete instance: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/instances/<instance_id>/start', methods=['POST'])
    def api_start_instance(instance_id: str):
        """Start/enable a plugin instance. Auto-restores from persistence if needed.
        No-op if already running (avoids re-triggering full discovery)."""
        try:
            plugin = _ensure_instance(instance_id)
            if not plugin:
                return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

            # No-op if already running — avoids re-triggering full 90-day discovery
            if plugin.status == 'running':
                return jsonify({
                    "success": True,
                    "message": f"Instance '{instance_id}' already running",
                    "already_running": True,
                })

            _start_plugin_instance(plugin)

            # Update status in persistence
            if persistence:
                persistence.update_instance_status(instance_id, 'running')

            return jsonify({"success": True, "message": f"Instance '{instance_id}' started"})

        except Exception as e:
            logger.error(f"Failed to start instance: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/instances/<instance_id>/stop', methods=['POST'])
    def api_stop_instance(instance_id: str):
        """Stop/disable a plugin instance. Auto-restores from persistence if needed."""
        try:
            plugin = _ensure_instance(instance_id)
            if not plugin:
                return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

            _stop_plugin_instance(plugin)

            # Update status in persistence
            if persistence:
                persistence.update_instance_status(instance_id, 'stopped')

            return jsonify({"success": True, "message": f"Instance '{instance_id}' stopped"})

        except Exception as e:
            logger.error(f"Failed to stop instance: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/instances/<instance_id>/refresh', methods=['POST'])
    def api_refresh_instance(instance_id: str):
        """
        Trigger incremental trace re-discovery for a plugin instance.

        Uses plugin.refresh_traces(days=N) if available (incremental fetch
        based on _last_query_time), otherwise falls back to plugin.start()
        (full discovery).

        Request body (optional):
        {
            "days": 7  // Override lookback days (default: auto from last query)
        }
        """
        try:
            plugin = _ensure_instance(instance_id)
            if not plugin:
                return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

            data = request.json or {}
            days = data.get('days')

            if hasattr(plugin, 'refresh_traces') and callable(plugin.refresh_traces):
                # Incremental refresh — uses _last_query_time internally
                if days is not None:
                    plugin.refresh_traces(days=int(days))
                else:
                    plugin.refresh_traces()
                method = 'refresh_traces'
            else:
                # Fallback: full discovery via start()
                _start_plugin_instance(plugin)
                method = 'start'

            # Update status in persistence
            if persistence:
                persistence.update_instance_status(instance_id, 'running')

            return jsonify({
                "success": True,
                "message": f"Instance '{instance_id}' refreshed via {method}",
                "method": method,
                "days": days,
            })

        except Exception as e:
            logger.error(f"Failed to refresh instance '{instance_id}': {e}")
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Webhook Management API endpoints
    # =========================================================================

    @app.route('/api/v1/instances/<instance_id>/webhooks', methods=['GET'])
    def api_list_webhooks(instance_id: str):
        """List subscribed webhook URLs for a plugin instance.
        Auto-restores from persistence if needed."""
        plugin = _ensure_instance(instance_id)
        if not plugin:
            return jsonify({"error": f"Instance '{instance_id}' not found"}), 404
        return jsonify({"webhooks": plugin.list_webhooks()})

    @app.route('/api/v1/instances/<instance_id>/webhooks', methods=['POST'])
    def api_subscribe_webhook(instance_id: str):
        """Subscribe a webhook URL to receive OTLP trace payloads.
        Auto-restores from persistence if needed."""
        plugin = _ensure_instance(instance_id)
        if not plugin:
            return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

        data = request.json or {}
        url = data.get("url", "").strip()
        if not url or not url.startswith(("http://", "https://")):
            return jsonify({"error": "A valid http:// or https:// URL is required"}), 400

        added = plugin.subscribe_webhook(url)
        return jsonify({"success": True, "added": added, "webhooks": plugin.list_webhooks()})

    @app.route('/api/v1/instances/<instance_id>/webhooks', methods=['DELETE'])
    def api_unsubscribe_webhook(instance_id: str):
        """Unsubscribe a webhook URL.
        Auto-restores from persistence if needed."""
        plugin = _ensure_instance(instance_id)
        if not plugin:
            return jsonify({"error": f"Instance '{instance_id}' not found"}), 404

        data = request.json or {}
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"error": "url is required"}), 400

        removed = plugin.unsubscribe_webhook(url)
        return jsonify({"success": True, "removed": removed, "webhooks": plugin.list_webhooks()})


def _create_plugin_instance(
    plugin_type_name: str,
    instance_id: str,
    display_name: str,
    config: Dict[str, Any],
) -> 'BaseDiscoveryPlugin':
    """Create a new plugin instance of the specified type via the registry."""
    return create_plugin_instance(
        plugin_type=plugin_type_name,
        config=config,
        instance_id=instance_id,
        display_name=display_name,
    )


def _start_plugin_instance(plugin: 'BaseDiscoveryPlugin'):
    """Start a plugin instance via its lifecycle method."""
    plugin.start()


def _stop_plugin_instance(plugin: 'BaseDiscoveryPlugin'):
    """Stop a plugin instance via its lifecycle method."""
    plugin.stop()


def _ensure_instance(instance_id: str) -> Optional['BaseDiscoveryPlugin']:
    """
    Ensure a plugin instance is in memory and return it.

    Lookup order:
    1. Already in client.plugins (in-memory) → return it
    2. Found in persistence → restore (create from persisted config, register) → return it
    3. Not found anywhere → return None
    """
    global client

    # 1. Check in-memory
    if client and instance_id in client.plugins:
        return client.plugins[instance_id]

    # 2. Try restoring from persistence
    if persistence:
        persisted = persistence.get_plugin_instance(instance_id)
        if persisted:
            try:
                if not client:
                    client = OpenCiteClient()

                plugin_instance = _create_plugin_instance(
                    plugin_type_name=persisted['plugin_type'],
                    instance_id=instance_id,
                    display_name=persisted['display_name'],
                    config=persisted['config'],
                )
                client.register_plugin(plugin_instance)
                logger.info(f"Restored instance '{instance_id}' from persistence")
                return plugin_instance
            except Exception as e:
                logger.error(f"Failed to restore instance '{instance_id}' from persistence: {e}")
                return None

    # 3. Not found
    return None


def _configure_plugin(plugin_name: str, config: Dict[str, Any]):
    """Configure a single plugin via the registry."""
    global client, discovery_status

    with state_lock:
        discovery_status["current_status"] = f"Configuring {plugin_name} plugin..."
        discovery_status["progress"].append({
            "step": plugin_name,
            "message": f"Initializing {plugin_name} plugin",
            "status": "in_progress",
        })

    # Check if plugin is already registered (reuse existing instance)
    existing_plugin = client.plugins.get(plugin_name)
    if existing_plugin is not None:
        logger.info(f"Reusing existing {plugin_name} plugin")
    else:
        # Unregister any auto-registered bare instance
        if plugin_name in client.plugins:
            client.unregister_plugin(plugin_name)

        # Create and register via registry
        plugin_instance = create_plugin_instance(
            plugin_type=plugin_name,
            config=config,
        )
        client.register_plugin(plugin_instance)

        # Start the plugin
        plugin_instance.start()
        logger.info(f"Configured and started {plugin_name} plugin")

    # Build success message with plugin-specific details
    success_msg = f"{plugin_name.replace('_', ' ').title()} plugin configured"
    plugin_obj = client.plugins.get(plugin_name)
    if plugin_obj and plugin_obj.plugin_type == 'opentelemetry':
        local_ip = get_local_ip()
        port = getattr(plugin_obj, 'port', 4318)
        if local_ip != '127.0.0.1':
            success_msg = (
                f"Trace receiver ready - Send to http://localhost:{port}/v1/traces (same machine) "
                f"or http://{local_ip}:{port}/v1/traces (other machines)"
            )
        else:
            success_msg = f"Trace receiver ready at http://localhost:{port}/v1/traces"

    with state_lock:
        discovery_status["progress"][-1]["status"] = "success"
        discovery_status["progress"][-1]["message"] = success_msg


def _empty_assets() -> Dict[str, List]:
    """Return empty assets structure."""
    return {
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


def _load_persisted_data():
    """Load persisted data into plugin state."""
    global client, persistence

    if not persistence or not client:
        return

    logger.info("Loading persisted data...")

    # Load OpenTelemetry plugin data (discovered tools only, not raw traces)
    otel_plugin = client.plugins.get('opentelemetry')
    if otel_plugin:
        try:
            # Load discovered tools
            tools = persistence.load_tools()
            if tools:
                otel_plugin.discovered_tools = tools
                logger.info(f"Loaded {len(tools)} tools from persistence")
        except Exception as e:
            logger.warning(f"Failed to load OpenTelemetry data: {e}")

    # Load MCP data into OTel plugin (MCP is now part of OTel)
    if otel_plugin:
        try:
            servers = persistence.load_mcp_servers()
            if servers:
                otel_plugin.mcp_servers = servers
                logger.info(f"Loaded {len(servers)} MCP servers from persistence")

            tools = persistence.load_mcp_tools()
            if tools:
                otel_plugin.mcp_tools = tools
                logger.info(f"Loaded {len(tools)} MCP tools from persistence")

            resources = persistence.load_mcp_resources()
            if resources:
                otel_plugin.mcp_resources = resources
                logger.info(f"Loaded {len(resources)} MCP resources from persistence")
        except Exception as e:
            logger.warning(f"Failed to load MCP data: {e}")

    # Load agents and downstream systems from persistence
    if otel_plugin and persistence:
        try:
            agents = persistence.load_agents()
            if agents:
                for agent_id, agent_data in agents.items():
                    otel_plugin.discovered_agents[agent_id] = {
                        "tools_used": set(agent_data.get("tools_used", [])),
                        "models_used": set(agent_data.get("models_used", [])),
                        "confidence": agent_data.get("confidence", "low"),
                        "first_seen": agent_data.get("first_seen"),
                        "last_seen": agent_data.get("last_seen"),
                        "metadata": agent_data.get("metadata", {}),
                    }
                logger.info(f"Loaded {len(agents)} agents from persistence")
        except Exception as e:
            logger.warning(f"Failed to load agent data: {e}")

        try:
            downstream = persistence.load_downstream_systems()
            if downstream:
                for sys_id, sys_data in downstream.items():
                    otel_plugin.discovered_downstream[sys_id] = {
                        "name": sys_data.get("name", ""),
                        "type": sys_data.get("type", "unknown"),
                        "endpoint": sys_data.get("endpoint"),
                        "tools_connecting": set(sys_data.get("tools_connecting", [])),
                        "first_seen": sys_data.get("first_seen"),
                        "last_seen": sys_data.get("last_seen"),
                        "metadata": sys_data.get("metadata", {}),
                    }
                logger.info(f"Loaded {len(downstream)} downstream systems from persistence")
        except Exception as e:
            logger.warning(f"Failed to load downstream system data: {e}")

    # Load tool mappings into identifier
    if otel_plugin and hasattr(otel_plugin, 'identifier'):
        try:
            mappings = persistence.load_mappings('opentelemetry')
            for mapping in mappings:
                otel_plugin.identifier.add_mapping(
                    mapping['plugin_name'],
                    mapping['attributes'],
                    mapping['identity'],
                    match_type=mapping.get('match_type', 'all')
                )
            if mappings:
                logger.info(f"Loaded {len(mappings)} tool mappings from persistence")
        except Exception as e:
            logger.warning(f"Failed to load tool mappings: {e}")


def _save_current_state():
    """Save current plugin state to persistence."""
    global client, persistence

    if not persistence or not client:
        return

    logger.info("Saving state to persistence...")

    # Save OpenTelemetry plugin data (discovered tools only, not raw traces)
    otel_plugin = client.plugins.get('opentelemetry')
    if otel_plugin:
        try:
            # Save discovered tools
            for tool_name, tool_data in otel_plugin.discovered_tools.items():
                models = list(tool_data.get('models', set()))
                trace_count = len(tool_data.get('traces', []))
                metadata = tool_data.get('metadata', {})
                persistence.save_tool(tool_name, models, trace_count, metadata)

            logger.info(f"Saved {len(otel_plugin.discovered_tools)} tools to persistence")
        except Exception as e:
            logger.warning(f"Failed to save OpenTelemetry data: {e}")

    # Save agents, downstream systems, and lineage
    if otel_plugin and persistence:
        try:
            for agent_name, agent_data in otel_plugin.discovered_agents.items():
                persistence.save_agent(
                    agent_id=agent_name,
                    name=agent_name,
                    confidence=agent_data.get("confidence", "low"),
                    tools_used=list(agent_data.get("tools_used", set())),
                    models_used=list(agent_data.get("models_used", set())),
                    first_seen=agent_data.get("first_seen"),
                    metadata=agent_data.get("metadata", {}),
                )
            logger.info(f"Saved {len(otel_plugin.discovered_agents)} agents to persistence")
        except Exception as e:
            logger.warning(f"Failed to save agent data: {e}")

        try:
            for sys_id, sys_data in otel_plugin.discovered_downstream.items():
                persistence.save_downstream_system(
                    system_id=sys_id,
                    name=sys_data.get("name", ""),
                    system_type=sys_data.get("type", "unknown"),
                    endpoint=sys_data.get("endpoint"),
                    tools_connecting=list(sys_data.get("tools_connecting", set())),
                    first_seen=sys_data.get("first_seen"),
                    metadata=sys_data.get("metadata", {}),
                )
            logger.info(f"Saved {len(otel_plugin.discovered_downstream)} downstream systems to persistence")
        except Exception as e:
            logger.warning(f"Failed to save downstream system data: {e}")

        try:
            for rel in otel_plugin.lineage.values():
                persistence.save_lineage(
                    source_id=rel["source_id"],
                    source_type=rel["source_type"],
                    target_id=rel["target_id"],
                    target_type=rel["target_type"],
                    relationship_type=rel["relationship_type"],
                    weight=rel.get("weight", 1),
                )
            logger.info(f"Saved {len(otel_plugin.lineage)} lineage relationships to persistence")
        except Exception as e:
            logger.warning(f"Failed to save lineage data: {e}")

    # Save MCP data from OTel plugin
    if otel_plugin:
        try:
            servers = getattr(otel_plugin, 'mcp_servers', {})
            for server_id, server_data in servers.items():
                persistence.save_mcp_server(
                    server_id=server_id,
                    name=server_data.get('name', ''),
                    transport=server_data.get('transport'),
                    endpoint=server_data.get('endpoint'),
                    command=server_data.get('command'),
                    args=server_data.get('args'),
                    env=server_data.get('env'),
                    source_file=server_data.get('source_file'),
                    source_env_var=server_data.get('source_env_var'),
                    metadata=server_data.get('metadata', {}),
                )

            tools = getattr(otel_plugin, 'mcp_tools', {})
            for tool_id, tool_data in tools.items():
                persistence.save_mcp_tool(
                    tool_id=tool_id,
                    server_id=tool_data.get('server_id', ''),
                    name=tool_data.get('name', ''),
                    description=tool_data.get('description'),
                    schema=tool_data.get('schema'),
                    usage=tool_data.get('usage'),
                    metadata=tool_data.get('metadata', {}),
                )

            resources = getattr(otel_plugin, 'mcp_resources', {})
            for resource_id, resource_data in resources.items():
                persistence.save_mcp_resource(
                    resource_id=resource_id,
                    server_id=resource_data.get('server_id', ''),
                    uri=resource_data.get('uri', ''),
                    name=resource_data.get('name'),
                    type=resource_data.get('type'),
                    mime_type=resource_data.get('mime_type'),
                    description=resource_data.get('description'),
                    usage=resource_data.get('usage'),
                    metadata=resource_data.get('metadata', {}),
                )

            logger.info(f"Saved {len(servers)} MCP servers, {len(tools)} MCP tools, {len(resources)} MCP resources")
        except Exception as e:
            logger.warning(f"Failed to save MCP data: {e}")


def _collect_assets(asset_type: str) -> Dict[str, List]:
    """Collect assets from all enabled plugins."""
    global discovery_status

    assets = _empty_assets()

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

    if "databricks" in discovery_status["plugins_enabled"]:
        if asset_type in ['all', 'data_assets']:
            try:
                with state_lock:
                    discovery_status["current_status"] = "Discovering Databricks assets..."
                export_data = client._export_databricks_assets()
                assets["data_assets"] = export_data
                with state_lock:
                    discovery_status["current_status"] = "Discovery complete"
            except Exception as e:
                logger.warning(f"Could not list Databricks data assets: {e}")

    if "google_cloud" in discovery_status["plugins_enabled"]:
        if asset_type in ['all', 'gcp_models']:
            try:
                assets["gcp_models"] = client.list_gcp_models()
            except Exception as e:
                logger.warning(f"Could not list GCP models: {e}")
        if asset_type in ['all', 'gcp_endpoints']:
            try:
                assets["gcp_endpoints"] = client.list_gcp_endpoints()
            except Exception as e:
                logger.warning(f"Could not list GCP endpoints: {e}")

    return assets


def shutdown_cleanup():
    """Clean up resources on shutdown."""
    global client, persistence
    logger.info("Running shutdown cleanup...")

    # Save state to persistence before shutdown
    if persistence and client:
        try:
            _save_current_state()
            logger.info("Saved state to persistence before shutdown")
        except Exception as e:
            logger.warning(f"Error saving state during shutdown: {e}")

    if client:
        # Stop all running plugins via lifecycle methods
        for plugin in list(client.plugins.values()):
            if plugin.status == 'running':
                try:
                    plugin.stop()
                    logger.info(f"Stopped plugin {plugin.instance_id}")
                except Exception as e:
                    logger.warning(f"Error stopping plugin {plugin.instance_id}: {e}")

    # Close persistence connection
    if persistence:
        try:
            persistence.close()
            logger.info("Closed persistence connection")
        except Exception as e:
            logger.warning(f"Error closing persistence: {e}")


def auto_configure_plugins(app: Flask):
    """Auto-configure plugins from environment configuration."""
    global client, discovery_status

    config = app.opencite_config
    plugins_to_enable = config.get_enabled_plugins()

    if not plugins_to_enable:
        logger.info("No plugins enabled via environment configuration")
        return

    logger.info(f"Auto-configuring plugins: {[p['name'] for p in plugins_to_enable]}")

    with state_lock:
        client = OpenCiteClient()
        discovery_status["error"] = None
        discovery_status["plugins_enabled"] = []
        discovery_status["progress"] = []
        discovery_status["current_status"] = "Auto-initializing OpenCITE client..."
        discovery_status["progress"].append({
            "step": "init",
            "message": "OpenCITE client created",
            "status": "success"
        })

    # Unregister any auto-registered plugins that we are about to configure
    # with proper credentials.  The OpenCiteClient constructor auto-registers
    # bare plugin instances (e.g. DatabricksPlugin with no host/token).
    # _configure_plugin() creates a new instance with the real config and
    # calls register_plugin(), which raises ValueError if the instance_id
    # is already taken.  Unregistering first avoids this conflict.
    for plugin_config in plugins_to_enable:
        plugin_name = plugin_config['name']
        if plugin_name in client.plugins:
            try:
                client.unregister_plugin(plugin_name)
            except ValueError:
                pass

    for plugin_config in plugins_to_enable:
        plugin_name = plugin_config['name']
        config_dict = plugin_config['config']

        try:
            _configure_plugin(plugin_name, config_dict)
            with state_lock:
                discovery_status["plugins_enabled"].append(plugin_name)
        except Exception as e:
            logger.error(f"Failed to auto-configure {plugin_name}: {e}")
            with state_lock:
                discovery_status["progress"].append({
                    "step": plugin_name,
                    "message": f"Failed: {str(e)}",
                    "status": "error"
                })

    # Load persisted data if persistence is enabled
    if persistence:
        with state_lock:
            discovery_status["current_status"] = "Loading persisted data..."
            discovery_status["progress"].append({
                "step": "persistence",
                "message": "Loading persisted data...",
                "status": "in_progress"
            })

        try:
            _load_persisted_data()
            with state_lock:
                discovery_status["progress"][-1]["status"] = "success"
                discovery_status["progress"][-1]["message"] = "Persisted data loaded successfully"
        except Exception as e:
            logger.warning(f"Failed to load persisted data: {e}")
            with state_lock:
                discovery_status["progress"][-1]["status"] = "warning"
                discovery_status["progress"][-1]["message"] = f"Could not load persisted data: {e}"

    with state_lock:
        discovery_status["last_updated"] = datetime.utcnow().isoformat()
        discovery_status["running"] = True
        discovery_status["current_status"] = "Auto-configuration complete - ready to discover assets"
        discovery_status["progress"].append({
            "step": "complete",
            "message": f"All plugins configured ({len(discovery_status['plugins_enabled'])} active)",
            "status": "success"
        })


def run_api(host: str = "0.0.0.0", port: int = 8080, auto_start: bool = True):
    """
    Run the OpenCITE API server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8080)
        auto_start: Auto-configure plugins from environment (default: True)
    """
    config = OpenCiteConfig.from_env()
    config.host = host
    config.port = port
    config.auto_start = auto_start

    app = create_app(config)

    # Register shutdown handler
    register_shutdown_handler(shutdown_cleanup)

    # Note: auto_configure_plugins is already called inside create_app()
    # when config.auto_start is True — no need to call it again here.

    print(f"\n{'='*60}")
    print(f"  OpenCITE API Service")
    print(f"{'='*60}")
    print(f"  API: http://{host}:{port}")
    print(f"  Health: http://{host}:{port}/healthz")
    print(f"  Readiness: http://{host}:{port}/readyz")
    if config.enable_otel:
        print(f"  OTLP Receiver: http://{config.otlp_host}:{config.otlp_port}/v1/traces")
    if config.persistence_enabled:
        print(f"  Persistence: {config.db_path}")
    else:
        print(f"  Persistence: disabled (in-memory only)")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=False, threaded=True)


# For gunicorn: gunicorn "open_cite.api.app:create_app()"
if __name__ == '__main__':
    run_api()
