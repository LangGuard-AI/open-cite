"""
OpenCITE Headless API Service.

A REST API for OpenCITE discovery and inventory capabilities,
designed for deployment in Kubernetes without a GUI.

Shared route registration:
  - ``register_api_routes(app)`` mounts all ``/api/v1/`` endpoints.
  - The GUI app imports and calls it so both entry-points share one
    implementation.
"""

import os
import json
import logging
import socket
import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Any, Optional
from flask import Flask, request, jsonify
from threading import Lock, Thread

from open_cite.client import OpenCiteClient
from .config import OpenCiteConfig
from .health import health_bp, init_health
from .persistence import PersistenceManager
from open_cite.plugin_store import PluginConfigStore

from open_cite.plugins.registry import get_all_plugin_metadata, create_plugin_instance

logger = logging.getLogger(__name__)

# =========================================================================
# Module-level shared state
#
# These globals are initialised by ``init_opencite_state()`` (called by
# both ``create_app()`` for the API and by the GUI's startup code).
# They are intentionally module-level so that route handlers and helper
# functions can access them without coupling to Flask's ``current_app``.
# Only one entry-point (API *or* GUI) runs per process, so there is no
# conflict.
# =========================================================================

client: Optional[OpenCiteClient] = None
persistence: Optional[PersistenceManager] = None
plugin_store: Optional[PluginConfigStore] = None
_config: Optional[OpenCiteConfig] = None
_default_otel_plugin: Optional[Any] = None
"""Default embedded OpenTelemetry plugin instance (created when otlp_embedded=True)."""
discovery_status: Dict[str, Any] = {
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
_last_save_time: float = 0.0
_SAVE_INTERVAL: float = 5.0  # minimum seconds between persistence saves

# Tracks fingerprints of persisted items so we only write new/changed data.
# Keys are like "tool:Bash", "agent:claude", "lineage:src->tgt:rel", etc.
_persisted_fingerprints: Dict[str, Any] = {}

# Last seen x-forwarded-access-token from the Databricks App proxy.
# Used to deduplicate so we only recreate clients when the token changes.
_forwarded_access_token: Optional[str] = None

# Downstream types that are really data assets, not generic downstream systems
_DATA_ASSET_TYPES = {"database", "dataset", "document_store"}

# =========================================================================
# GUI integration hooks
#
# When the GUI entry-point is active it sets these callbacks so that the
# shared API route logic can trigger WebSocket pushes and background-
# threaded plugin starts without importing GUI-specific modules.
# =========================================================================

_on_plugin_start: Optional[Callable] = None
"""If set, called instead of ``plugin.start()`` – the GUI sets this to run
start() in a background thread and push WebSocket updates."""

_on_status_changed: Optional[Callable] = None
"""Called after any discovery_status mutation so the GUI can push via WS."""


# =========================================================================
# Public helpers (used by GUI init and health checks)
# =========================================================================

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


def _notify_status_changed():
    """Notify listeners (GUI WebSocket) that discovery_status changed."""
    if _on_status_changed:
        try:
            _on_status_changed()
        except Exception as e:
            logger.debug(f"_on_status_changed error: {e}")


def _reclassify_downstream(assets):
    """Move data-oriented downstream items into data_assets."""
    remaining = []
    for item in assets.get("downstream_systems", []):
        if item.get("type") in _DATA_ASSET_TYPES:
            assets["data_assets"].append(item)
        else:
            remaining.append(item)
    assets["downstream_systems"] = remaining


# =========================================================================
# State initialisation
# =========================================================================

def init_opencite_state(app: Flask, config: Optional[OpenCiteConfig] = None):
    """Initialise module-level shared state and attach to *app*.

    Called by ``create_app()`` (API) and by the GUI's startup code so that
    both entry-points share the same state layer.
    """
    global persistence, plugin_store, _config, client, _default_otel_plugin

    if config is None:
        config = OpenCiteConfig.from_env()

    _config = config

    # Store config on the app
    app.opencite_config = config

    # Propagate database URL / path to env so the db package can pick it up.
    # Skip db_path when on Databricks — let the engine auto-detect the
    # SQL warehouse instead of falling back to SQLite.
    if config.database_url:
        os.environ.setdefault("OPENCITE_DATABASE_URL", config.database_url)
    elif config.db_path and not os.getenv("DATABRICKS_HOST"):
        os.environ.setdefault("OPENCITE_DB_PATH", config.db_path)

    # Initialise shared database if any persistence category is enabled
    any_persistence = (
        config.persistence_enabled
        or config.persist_plugins
        or config.persist_mappings
    )
    if any_persistence:
        from open_cite.db import init_db
        init_db()

    # Asset persistence (disabled by default)
    if config.persistence_enabled:
        persistence = PersistenceManager()
        logger.info("Asset persistence (SQLAlchemy) enabled")
    else:
        logger.info("Asset persistence disabled — in-memory only")

    # Plugin config persistence (enabled by default)
    plugin_store = PluginConfigStore(
        enabled=config.persist_plugins,
    )

    # Create client early so we can wire persistence
    if not client:
        client = OpenCiteClient()
    if persistence:
        client.persistence = persistence

    # Create the default embedded OTel plugin if enabled
    if config.otlp_embedded:
        otel_config = {
            "embedded_receiver": True,
            "persist_mappings": config.persist_mappings,
            "mapping_store_path": config.mapping_store_path,
        }
        _default_otel_plugin = _create_plugin_instance(
            plugin_type_name="opentelemetry",
            instance_id="opentelemetry",
            display_name="OpenTelemetry (Embedded)",
            config=otel_config,
        )
        client.register_plugin(_default_otel_plugin)
        _default_otel_plugin.start()
        logger.info("Embedded OTLP receiver active on main web port")


# =========================================================================
# Flask app factory (API entry-point)
# =========================================================================

def create_app(config: Optional[OpenCiteConfig] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config: Optional OpenCiteConfig instance. If None, loads from environment.

    Returns:
        Configured Flask application
    """
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

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)

    # Initialise shared state (persistence, plugin_store)
    init_opencite_state(app, config)

    # Register health check blueprint
    app.register_blueprint(health_bp)

    # Initialize health check dependencies
    init_health(get_client, get_status)

    # Register API routes
    register_api_routes(app)

    # Auto-configure plugins from environment when auto_start is enabled.
    # This is needed for gunicorn which calls create_app() directly
    # rather than going through run_api().
    if config.auto_start:
        auto_configure_plugins(app)

    # Restore saved plugin instances from JSON store
    _restore_saved_plugins()

    # Auto-configure Databricks plugin when running as a Databricks App
    _auto_configure_databricks_app()

    return app


def export_to_json(include_plugins: List[str]) -> Dict[str, Any]:
    """Export all discovered data to JSON format according to OpenCITE schema.

    Reads from the DB (via the client) for persisted asset types, and falls
    back to plugin ``export_assets()`` for plugin-specific extras (e.g.
    Databricks data_assets, GCP deployments).

    Args:
        include_plugins: List of plugin type names to include
                         (e.g. ``["opentelemetry", "databricks"]``).

    Returns:
        JSON-serializable dictionary with all discovered data.
    """
    from open_cite.schema import OpenCiteExporter

    # Core asset types — read from DB via client
    tools = client.list_tools()
    models = client.list_models()
    agents = client.list_agents()
    downstream = client.list_downstream_systems()
    mcp_servers = client.list_mcp_servers()
    mcp_tools = client.list_mcp_tools()
    mcp_resources = client.list_mcp_resources()

    # Reclassify downstream → data_assets
    temp = {"downstream_systems": downstream, "data_assets": []}
    _reclassify_downstream(temp)
    downstream = temp["downstream_systems"]
    data_assets = temp["data_assets"]

    # Collect plugin-specific extras (e.g. Databricks catalogs, GCP deployments)
    extras: Dict[str, Any] = {}
    plugins_info: List[Dict[str, str]] = []
    for plugin in client.plugins.values():
        if plugin.plugin_type not in include_plugins:
            continue
        plugins_info.append({"name": plugin.plugin_type, "version": "1.0.0"})
        try:
            plugin_assets = plugin.export_assets()
        except Exception as e:
            logger.error(f"Export failed for plugin {plugin.instance_id}: {e}")
            continue
        if not plugin_assets:
            continue
        # Only keep keys that aren't already covered by the DB read above
        for key, value in plugin_assets.items():
            if key in ("tools", "models", "agents", "downstream_systems",
                       "mcp_servers", "mcp_tools", "mcp_resources"):
                continue
            if key == "data_assets" and isinstance(value, list):
                data_assets.extend(value)
            elif isinstance(value, list):
                extras.setdefault(key, []).extend(value)
            elif isinstance(value, dict):
                extras.setdefault(key, {}).update(value)
            else:
                extras[key] = value

    exporter = OpenCiteExporter()
    export_data = exporter.export_discovery(
        tools=tools,
        models=models,
        data_assets=data_assets,
        mcp_servers=mcp_servers,
        mcp_tools=mcp_tools,
        mcp_resources=mcp_resources,
        metadata={"generated_by": "opencite", "plugins": plugins_info},
    )
    # Add non-standard keys from plugins
    export_data.update(extras)
    # Include agents and downstream in export
    export_data["agents"] = agents
    export_data["downstream_systems"] = downstream

    return export_data


# =========================================================================
# Route registration (shared by API and GUI Flask apps)
# =========================================================================

def _otlp_ingest(data: dict, headers: dict):
    """Ingest OTLP trace data into the default embedded OTel plugin.

    Used as the callback for both the ``POST /v1/traces`` Flask route and
    the gRPC ASGI handler.
    """
    if _default_otel_plugin is None:
        raise RuntimeError("No embedded OTLP receiver configured")
    _default_otel_plugin._ingest_traces(data)
    _default_otel_plugin._deliver_to_webhooks(data, inbound_headers=headers)


def _otlp_ingest_logs(data: dict, headers: dict):
    """Convert OTLP logs to synthetic traces, then ingest + forward via webhooks.

    Used as the callback for both the ``POST /v1/logs`` Flask route and
    the gRPC ASGI handler.
    """
    if _default_otel_plugin is None:
        raise RuntimeError("No embedded OTLP receiver configured")
    from open_cite.plugins.logs_adapter import convert_logs_to_traces
    synthetic_traces = convert_logs_to_traces(data)
    if synthetic_traces.get("resourceSpans"):
        _default_otel_plugin._ingest_traces(synthetic_traces)
        _default_otel_plugin._deliver_to_webhooks(synthetic_traces, inbound_headers=headers)


def register_api_routes(app: Flask):
    """Register all API routes on the Flask app."""

    # -----------------------------------------------------------------
    # Databricks App proxy token forwarding
    # -----------------------------------------------------------------
    @app.before_request
    def _forward_databricks_token():
        """Pick up ``x-forwarded-access-token`` from the Databricks App proxy.

        The proxy injects this header on every browser request.  When the
        token changes (periodic refresh) we hot-swap it into all running
        Databricks plugin instances and the DB connection pool.
        """
        global _forwarded_access_token
        token = request.headers.get("x-forwarded-access-token")
        if not token:
            return
        if token == _forwarded_access_token:
            return
        _forwarded_access_token = token
        logger.info("[token-fwd] New x-forwarded-access-token received (length=%d)", len(token))

        # Update running Databricks plugin instances
        if client:
            for plugin in client.get_plugins_by_type("databricks"):
                try:
                    plugin.update_token(token)
                except Exception as exc:
                    logger.warning("[token-fwd] Failed to update plugin %s: %s", plugin.instance_id, exc)

        # Update DB engine so new connections use the forwarded token
        from open_cite.db import engine as db_engine
        db_engine.set_forwarded_token(token)

    # -----------------------------------------------------------------
    # OTLP HTTP trace ingestion (JSON + protobuf)
    # -----------------------------------------------------------------
    @app.route('/v1/traces', methods=['POST'])
    def otlp_ingest_traces():
        """Accept OTLP trace payloads (JSON or protobuf) on the main port."""
        if _default_otel_plugin is None:
            return jsonify({"error": "Embedded OTLP receiver not configured"}), 503

        content_type = request.headers.get("Content-Type", "")

        # Extract forwarded headers (skip hop-by-hop)
        _SKIP_HEADERS = {"content-length", "transfer-encoding", "connection",
                         "keep-alive", "te", "trailers", "upgrade"}
        inbound_headers = {}
        for key, value in request.headers:
            if key.lower() in _SKIP_HEADERS:
                continue
            if key.lower() == "host":
                inbound_headers["OTEL-HOST"] = value
            else:
                inbound_headers[key] = value

        try:
            if "application/json" in content_type:
                data = request.get_json(force=True)
            elif "application/x-protobuf" in content_type:
                from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                    ExportTraceServiceRequest,
                )
                from google.protobuf.json_format import MessageToDict

                req = ExportTraceServiceRequest()
                req.ParseFromString(request.get_data())
                data = MessageToDict(req)
            else:
                return jsonify({"error": f"Unsupported Content-Type: {content_type}"}), 415

            _default_otel_plugin._ingest_traces(data)
            _default_otel_plugin._deliver_to_webhooks(data, inbound_headers=inbound_headers)

            num_spans = sum(
                len(rs.get("scopeSpans", []))
                for rs in data.get("resourceSpans", [])
            )
            logger.info(f"[OTLP/embedded] Received trace with {num_spans} scope spans")

            return jsonify({"status": "success"})

        except Exception as e:
            logger.error(f"Error processing OTLP traces: {e}")
            return jsonify({"error": str(e)}), 500

    # -----------------------------------------------------------------
    # OTLP HTTP log ingestion (JSON + protobuf) — converts to traces
    # -----------------------------------------------------------------
    @app.route('/v1/logs', methods=['POST'])
    def otlp_ingest_logs():
        """Accept OTLP log payloads (JSON or protobuf), convert to traces, ingest."""
        if _default_otel_plugin is None:
            return jsonify({"error": "Embedded OTLP receiver not configured"}), 503

        content_type = request.headers.get("Content-Type", "")

        # Extract forwarded headers (skip hop-by-hop)
        _SKIP_HEADERS = {"content-length", "transfer-encoding", "connection",
                         "keep-alive", "te", "trailers", "upgrade"}
        inbound_headers = {}
        for key, value in request.headers:
            if key.lower() in _SKIP_HEADERS:
                continue
            if key.lower() == "host":
                inbound_headers["OTEL-HOST"] = value
            else:
                inbound_headers[key] = value

        try:
            if "application/json" in content_type:
                data = request.get_json(force=True)
            elif "application/x-protobuf" in content_type:
                from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
                    ExportLogsServiceRequest,
                )
                from google.protobuf.json_format import MessageToDict

                req = ExportLogsServiceRequest()
                req.ParseFromString(request.get_data())
                data = MessageToDict(req)
            else:
                return jsonify({"error": f"Unsupported Content-Type: {content_type}"}), 415

            from open_cite.plugins.logs_adapter import convert_logs_to_traces
            synthetic_traces = convert_logs_to_traces(data)

            num_log_records = sum(
                len(lr)
                for rl in data.get("resourceLogs", [])
                for sl in rl.get("scopeLogs", [])
                for lr in [sl.get("logRecords", [])]
            )

            if synthetic_traces.get("resourceSpans"):
                _default_otel_plugin._ingest_traces(synthetic_traces)
                _default_otel_plugin._deliver_to_webhooks(
                    synthetic_traces, inbound_headers=inbound_headers
                )

            num_spans = sum(
                len(ss.get("spans", []))
                for rs in synthetic_traces.get("resourceSpans", [])
                for ss in rs.get("scopeSpans", [])
            )
            logger.info(
                f"[OTLP/logs] Received {num_log_records} log records, "
                f"converted to {num_spans} synthetic spans"
            )

            return jsonify({"status": "success"})

        except Exception as e:
            logger.error(f"Error processing OTLP logs: {e}")
            return jsonify({"error": str(e)}), 500

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
                    _notify_status_changed()
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
            _notify_status_changed()

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

            with state_lock:
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
                    if datetime.utcnow() - asset_cache_time < timedelta(seconds=30):
                        return jsonify(asset_cache)

                discovering_assets = True

            try:
                assets = _collect_assets(asset_type)
                _reclassify_downstream(assets)
                totals = {k: len(v) for k, v in assets.items()}

                result = {
                    "assets": assets,
                    "totals": totals,
                    "timestamp": datetime.utcnow().isoformat()
                }
                with state_lock:
                    asset_cache = result
                    asset_cache_time = datetime.utcnow()
                    discovering_assets = False

                return jsonify(result)
            except Exception:
                with state_lock:
                    discovering_assets = False
                raise

        except Exception as e:
            logger.error(f"Error getting assets: {e}")
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
            return jsonify(export_to_json(include_plugins))
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

            # Generate a stable source_id as UUIDv5
            OPENCITE_NS = uuid.UUID('a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d')
            source_name = identity.get('source_name', '')
            identity['source_id'] = str(uuid.uuid5(OPENCITE_NS, f"{plugin_name}:{source_name}"))

            # Try plugin's own identifier first, fall back to finding any plugin with one
            plugin = client.plugins.get(plugin_name)
            if plugin and hasattr(plugin, 'identifier'):
                success = plugin.identifier.add_mapping(
                    plugin_name, attributes, identity, match_type=match_type
                )
            else:
                # Find first plugin with an identifier
                success = False
                for p in client.plugins.values():
                    if hasattr(p, 'identifier'):
                        success = p.identifier.add_mapping(
                            plugin_name, attributes, identity, match_type=match_type
                        )
                        break
                if not success:
                    return jsonify({"error": "No plugin with identifier support found"}), 400

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
            _notify_status_changed()

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

            return jsonify({
                "enabled": True,
                "stats": {
                    "tools": len(tools),
                    "models": len(models),
                    "agents": len(agents),
                    "downstream_systems": len(downstream),
                    "lineage_relationships": len(lineage),
                    "mcp_servers": len(mcp_servers),
                    "mcp_tools": len(mcp_tools),
                    "mcp_resources": len(mcp_resources),
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
        """No-op: the DB is now the source of truth for asset reads."""
        if not persistence:
            return jsonify({"error": "Persistence is disabled"}), 400

        return jsonify({"success": True, "message": "DB is the source of truth; no load needed"})

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

    @app.route('/api/v1/reset-discoveries', methods=['POST'])
    def api_reset_discoveries():
        """Clear all discovered assets from DB and plugin memory.

        Does NOT touch plugin configurations — only discovered data.
        """
        global asset_cache, asset_cache_time, _persisted_fingerprints

        errors = []

        # 1. Clear DB
        if persistence:
            try:
                persistence.clear_all()
            except Exception as e:
                errors.append(f"DB clear failed: {e}")

        # 2. Clear plugin in-memory discovered data
        if client:
            for plugin in client.plugins.values():
                for attr in ('discovered_tools', 'discovered_agents',
                             'discovered_downstream', 'lineage',
                             'model_providers', 'model_call_count',
                             'model_token_usage', 'mcp_servers',
                             'mcp_tools', 'mcp_resources', 'traces'):
                    store = getattr(plugin, attr, None)
                    if isinstance(store, dict):
                        store.clear()

        # 3. Clear caches and fingerprints
        _persisted_fingerprints.clear()
        asset_cache = None
        asset_cache_time = None

        if errors:
            logger.warning("Reset completed with errors: %s", errors)
            return jsonify({"success": True, "warnings": errors})

        logger.info("All discovered data reset")
        return jsonify({"success": True})

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

        # Fall back to plugin store
        if plugin_store:
            try:
                instances = plugin_store.load_all()
                if plugin_type:
                    instances = [i for i in instances if i['plugin_type'] == plugin_type]
                return jsonify({"instances": instances, "count": len(instances)})
            except Exception as e:
                logger.error(f"Failed to load instances from plugin store: {e}")
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

            # Save to plugin config store
            if plugin_store:
                plugin_store.save(
                    instance_id=instance_id,
                    plugin_type=plugin_type,
                    display_name=display_name,
                    config=config,
                    auto_start=auto_start,
                )

            # Start the plugin if auto_start is enabled
            if auto_start:
                _start_plugin_instance(plugin_instance)
                with state_lock:
                    if plugin_type not in discovery_status["plugins_enabled"]:
                        discovery_status["plugins_enabled"].append(plugin_type)
                    discovery_status["running"] = True
                _notify_status_changed()

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

        # Fall back to plugin store
        if plugin_store:
            for saved in plugin_store.load_all():
                if saved['instance_id'] == instance_id:
                    return jsonify({"instance": saved})

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

                # Update plugin store
                if plugin_store:
                    saved = next(
                        (s for s in plugin_store.load_all() if s['instance_id'] == instance_id),
                        None
                    )
                    if saved:
                        plugin_store.save(
                            instance_id=instance_id,
                            plugin_type=saved['plugin_type'],
                            display_name=data.get('display_name', saved['display_name']),
                            config={**saved['config'], **data.get('config', {})},
                            auto_start=data.get('auto_start', saved['auto_start']),
                        )

                return jsonify({"success": True, "instance": plugin.to_dict()})

            elif plugin_store:
                saved = next(
                    (s for s in plugin_store.load_all() if s['instance_id'] == instance_id),
                    None
                )
                if saved:
                    plugin_store.save(
                        instance_id=instance_id,
                        plugin_type=saved['plugin_type'],
                        display_name=data.get('display_name', saved['display_name']),
                        config={**saved['config'], **data.get('config', {})},
                        auto_start=data.get('auto_start', saved['auto_start']),
                    )
                    updated = next(
                        (s for s in plugin_store.load_all() if s['instance_id'] == instance_id),
                        None
                    )
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

            # Remove from plugin config store
            if plugin_store:
                plugin_store.delete(instance_id)

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

            # Update discovery status
            with state_lock:
                plugin_type = plugin.plugin_type
                if plugin_type not in discovery_status["plugins_enabled"]:
                    discovery_status["plugins_enabled"].append(plugin_type)
                discovery_status["running"] = True
            _notify_status_changed()

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
            _notify_status_changed()

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

    # =========================================================================
    # Lineage Graph (visual HTML – used by GUI iframe)
    # =========================================================================

    # Theme palettes for the lineage graph (keyed by data-theme value)
    _LINEAGE_THEMES = {
        "opencite": {
            "bgcolor": "#ffffff",
            "font_color": "#333333",
            "edge_color": "#cccccc",
            "node_colors": {
                "agent":      "#8b5cf6",
                "tool":       "#3b82f6",
                "model":      "#10b981",
                "data_asset": "#f59e0b",
                "downstream": "#ef4444",
                "default":    "#6b7280",
            },
            "empty_bg": "#ffffff",
            "empty_text": "#6b7280",
        },
        "databricks-light": {
            "bgcolor": "#ffffff",
            "font_color": "#0B2026",
            "edge_color": "#C8C3BC",
            "node_colors": {
                "agent":      "#8b5cf6",
                "tool":       "#1A56DB",
                "model":      "#059669",
                "data_asset": "#D97706",
                "downstream": "#FF3621",
                "default":    "#7A7A7A",
            },
            "empty_bg": "#ffffff",
            "empty_text": "#7A7A7A",
        },
        "databricks-dark": {
            "bgcolor": "#243B44",
            "font_color": "#E8E8E8",
            "edge_color": "#4A6070",
            "node_colors": {
                "agent":      "#C4B5FD",
                "tool":       "#93C5FD",
                "model":      "#6EE7B7",
                "data_asset": "#FCD34D",
                "downstream": "#FF5642",
                "default":    "#7A8E96",
            },
            "empty_bg": "#243B44",
            "empty_text": "#7A8E96",
        },
    }

    @app.route('/api/v1/lineage-graph')
    def api_lineage_graph():
        """Generate a pyvis lineage graph as embeddable HTML."""
        try:
            from pyvis.network import Network
        except ImportError:
            return ("<html><body><p>pyvis not installed</p></body></html>",
                    200, {'Content-Type': 'text/html'})

        if not client:
            return ("<html><body><p>No data yet</p></body></html>",
                    200, {'Content-Type': 'text/html'})

        theme_name = request.args.get('theme', 'opencite')
        theme = _LINEAGE_THEMES.get(theme_name, _LINEAGE_THEMES["opencite"])

        net = Network(
            height="350", width="100%", directed=True,
            bgcolor=theme["bgcolor"], font_color=theme["font_color"],
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
                "hierarchicalRepulsion": { "nodeDistance": 150 }
            },
            "edges": {
                "color": { "color": "%s", "highlight": "%s" },
                "arrows": { "to": { "enabled": true } },
                "smooth": { "type": "cubicBezier" }
            },
            "interaction": { "hover": true, "tooltipDelay": 100 }
        }""" % (theme["edge_color"], theme["font_color"]))

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

        node_colors = theme["node_colors"]
        added = set()

        # Emoji icons matching the GUI asset cards
        _NODE_EMOJIS = {
            "agent":      "\U0001F916",  # 🤖
            "tool":       "\U0001F527",  # 🔧
            "model":      "\u2728",      # ✨
            "data_asset": "\U0001F4CA",  # 📊
            "downstream": "\U0001F517",  # 🔗
        }

        def _emoji_svg_uri(emoji):
            """Build an inline SVG data URI with just an emoji (no background)."""
            from urllib.parse import quote
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48">'
                f'<text x="24" y="24" text-anchor="middle" dominant-baseline="central"'
                f' font-size="36">{emoji}</text>'
                '</svg>'
            )
            return "data:image/svg+xml," + quote(svg)

        # Pre-build the image URIs per node type
        STYLES = {}
        for ntype, level in (("agent", 0), ("tool", 1), ("model", 2),
                             ("data_asset", 3), ("downstream", 3)):
            color = node_colors.get(ntype, node_colors["default"])
            emoji = _NODE_EMOJIS.get(ntype, "")
            STYLES[ntype] = (color, _emoji_svg_uri(emoji), level)

        def add_node(nid, label, ntype, title=None):
            if nid in added:
                return
            added.add(nid)
            color, image_url, level = STYLES.get(
                ntype,
                (node_colors["default"],
                 _emoji_svg_uri(""),
                 1),
            )
            net.add_node(
                nid, label=label, shape="image", image=image_url,
                color=color,
                level=level, title=title or f"{ntype}: {label}", size=30,
            )

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

        for t in tools:
            tid = f"tool:{t['name']}"
            add_node(tid, t["name"], "tool")
            for m in (t.get("models") or []):
                mid = f"model:{m}"
                add_node(mid, m, "model")
                net.add_edge(tid, mid)

        for m in models:
            mid = f"model:{m['name']}"
            add_node(mid, m["name"], "model",
                     f"Model: {m['name']}\nProvider: {m.get('provider','')}")

        for d in data_assets_list:
            did = f"data:{d['name']}"
            add_node(did, d["name"], "data_asset",
                     f"Data: {d['name']}\nType: {d.get('type','')}")
            for t in (d.get("tools_connecting") or []):
                tid = f"tool:{t}"
                add_node(tid, t, "tool")
                net.add_edge(tid, did)

        for d in downstream:
            did = f"ds:{d['name']}"
            add_node(did, d["name"], "downstream",
                     f"Downstream: {d['name']}\nType: {d.get('type','')}")
            for t in (d.get("tools_connecting") or []):
                tid = f"tool:{t}"
                add_node(tid, t, "tool")
                net.add_edge(tid, did)

        if not added:
            return ("<html><body style='display:flex;align-items:center;justify-content:center;"
                    "height:100%%;background:%s;color:%s;font-family:sans-serif'>"
                    "<p>No lineage data yet</p></body></html>"
                    % (theme["empty_bg"], theme["empty_text"])), 200, {'Content-Type': 'text/html'}

        html = net.generate_html()

        # Override pyvis/Bootstrap default styles so the graph blends
        # seamlessly into the parent card (no white border or background).
        theme_css = (
            "<style>"
            "body { background: %s !important; margin: 0; padding: 0; }"
            ".card { background: transparent !important; border: none !important;"
            "  box-shadow: none !important; margin: 0 !important; padding: 0 !important; }"
            "#mynetwork { border: none !important; }"
            "</style>"
        ) % theme["bgcolor"]
        html = html.replace('</head>', theme_css + '</head>')

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
if (typeof network !== 'undefined') {
    network.on('selectNode', function(params) {
        if (params.nodes.length === 1) {
            window.parent.postMessage({ type: 'openDetail', nodeId: params.nodes[0] }, '*');
        }
    });
}
</script>"""
        html = html.replace('</body>', focus_script + '</body>')
        return html, 200, {'Content-Type': 'text/html'}


# =========================================================================
# Plugin lifecycle helpers
# =========================================================================

def _create_plugin_instance(
    plugin_type_name: str,
    instance_id: str,
    display_name: str,
    config: Dict[str, Any],
) -> 'BaseDiscoveryPlugin':
    """Create a new plugin instance of the specified type via the registry."""
    # Inject identity mapping persistence settings for OpenTelemetry plugins
    if plugin_type_name == "opentelemetry" and _config:
        config = {**config}  # shallow copy to avoid mutating caller's dict
        config.setdefault("persist_mappings", _config.persist_mappings)
        config.setdefault("mapping_store_path", _config.mapping_store_path)
        # Don't inject embedded_receiver for user-created instances —
        # they should run their own standalone receivers unless explicitly set.

    return create_plugin_instance(
        plugin_type=plugin_type_name,
        config=config,
        instance_id=instance_id,
        display_name=display_name,
    )


def _start_plugin_instance(plugin: 'BaseDiscoveryPlugin'):
    """Start a plugin instance.

    If the GUI has set ``_on_plugin_start`` this delegates to that callback
    (which runs start() in a background thread with WebSocket notifications).
    Otherwise starts synchronously and wires up persistence via
    ``on_data_changed``.
    """
    if _on_plugin_start:
        _on_plugin_start(plugin)
    else:
        plugin.on_data_changed = _maybe_save_state
        plugin.start()


def _stop_plugin_instance(plugin: 'BaseDiscoveryPlugin'):
    """Stop a plugin instance via its lifecycle method."""
    plugin.stop()


def _ensure_instance(instance_id: str) -> Optional['BaseDiscoveryPlugin']:
    """
    Ensure a plugin instance is in memory and return it.

    Lookup order:
    1. Already in client.plugins (in-memory) -> return it
    2. Found in plugin_store or persistence -> restore -> return it
    3. Not found anywhere -> return None
    """
    global client

    # 1. Check in-memory
    if client and instance_id in client.plugins:
        return client.plugins[instance_id]

    # 2. Try restoring from plugin_store
    persisted = None
    if plugin_store:
        for saved in plugin_store.load_all():
            if saved['instance_id'] == instance_id:
                persisted = saved
                break

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


def _restore_saved_plugins():
    """Restore plugin instances from the JSON plugin store on startup."""
    global client

    if not plugin_store or not plugin_store.enabled:
        return

    saved = plugin_store.load_all()
    if not saved:
        return

    logger.info(f"Restoring {len(saved)} saved plugin instance(s) from JSON store")

    if not client:
        client = OpenCiteClient()

    for entry in saved:
        iid = entry['instance_id']
        # Skip if already registered (e.g. by auto_configure_plugins)
        if iid in client.plugins:
            continue

        try:
            plugin_instance = _create_plugin_instance(
                plugin_type_name=entry['plugin_type'],
                instance_id=iid,
                display_name=entry['display_name'],
                config=entry['config'],
            )
            client.register_plugin(plugin_instance)

            if entry.get('auto_start', False):
                _start_plugin_instance(plugin_instance)
                with state_lock:
                    pt = entry['plugin_type']
                    if pt not in discovery_status["plugins_enabled"]:
                        discovery_status["plugins_enabled"].append(pt)
                    discovery_status["running"] = True

            logger.info(f"Restored plugin instance '{iid}' ({entry['plugin_type']})")
        except Exception as e:
            logger.error(f"Failed to restore plugin instance '{iid}': {e}")

    _notify_status_changed()


def _is_databricks_app() -> bool:
    """Return True when running inside a Databricks App container.

    Detection: both ``DATABRICKS_CLIENT_ID`` and ``DATABRICKS_APP_PORT``
    are present in the environment.  These are auto-injected by the
    Databricks Apps platform.
    """
    return bool(os.environ.get("DATABRICKS_CLIENT_ID")
                and os.environ.get("DATABRICKS_APP_PORT"))


def _auto_configure_databricks_app():
    """Auto-configure a Databricks plugin when running as a Databricks App.

    Detection: both ``DATABRICKS_CLIENT_ID`` and ``DATABRICKS_APP_PORT``
    are set (auto-injected by the Databricks Apps platform).

    The Databricks SDK auto-detects the workspace URL and OAuth
    credentials from the environment, so no explicit host or token is
    needed.  Skips if a Databricks instance already exists (e.g.
    restored from the plugin store).
    """
    global client

    # -- Dump all DATABRICKS_* env vars for troubleshooting ----------------
    db_env = {k: ("****" if "SECRET" in k or "TOKEN" in k else v)
              for k, v in os.environ.items() if k.startswith("DATABRICKS_")}
    print(f"  [auto-configure] DATABRICKS_* env vars: {db_env}")
    logger.info("Databricks App auto-configure check — env vars: %s", db_env)

    if not _is_databricks_app():
        print(f"  [auto-configure] Skipping — not a Databricks App "
              f"(DATABRICKS_CLIENT_ID={'set' if os.environ.get('DATABRICKS_CLIENT_ID') else 'UNSET'}, "
              f"DATABRICKS_APP_PORT={'set' if os.environ.get('DATABRICKS_APP_PORT') else 'UNSET'})")
        return

    print("  [auto-configure] Databricks App environment detected")

    # Skip if a Databricks plugin instance is already registered
    if client:
        registered = {iid: getattr(p, 'plugin_type', '?') for iid, p in client.plugins.items()}
        print(f"  [auto-configure] Already registered plugins: {registered}")
        for plugin in client.plugins.values():
            if getattr(plugin, "plugin_type", None) == "databricks":
                print("  [auto-configure] Databricks instance already registered — skipping")
                return

    if not client:
        print("  [auto-configure] Creating OpenCiteClient")
        client = OpenCiteClient()

    instance_id = "databricks-app-auto"
    display_name = "Databricks (App Auto-configured)"

    try:
        print("  [auto-configure] Creating Databricks plugin instance (empty config, SDK auto-detect)...")
        # Pass empty config — the SDK auto-detects host + OAuth from env
        plugin_instance = _create_plugin_instance(
            plugin_type_name="databricks",
            instance_id=instance_id,
            display_name=display_name,
            config={},
        )

        resolved_host = getattr(plugin_instance, "host", "unknown")
        print(f"  [auto-configure] Plugin created — resolved host: {resolved_host}")

        client.register_plugin(plugin_instance)
        print(f"  [auto-configure] Plugin registered as '{instance_id}'")

        # Persist so it survives restarts (auto_start=True)
        if plugin_store:
            plugin_store.save(
                instance_id=instance_id,
                plugin_type="databricks",
                display_name=display_name,
                config={},
                auto_start=True,
            )
            print("  [auto-configure] Plugin config saved to store")

        print("  [auto-configure] Starting plugin...")
        _start_plugin_instance(plugin_instance)

        with state_lock:
            if "databricks" not in discovery_status["plugins_enabled"]:
                discovery_status["plugins_enabled"].append("databricks")
            discovery_status["running"] = True
        _notify_status_changed()

        print(f"  [auto-configure] SUCCESS — Databricks plugin running for {resolved_host}")
    except Exception as e:
        import traceback
        print(f"  [auto-configure] FAILED: {e}")
        print(f"  [auto-configure] Traceback:\n{traceback.format_exc()}")
        logger.error(f"Failed to auto-configure Databricks plugin: {e}", exc_info=True)


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
    _notify_status_changed()

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
                f"Trace receiver ready - Send to http://localhost:{port}/v1/traces (same machine) "
                f"or http://{local_ip}:{port}/v1/traces (other machines)"
            )
        else:
            success_msg = f"Trace receiver ready at http://localhost:{port}/v1/traces"

    with state_lock:
        discovery_status["progress"][-1]["status"] = "success"
        discovery_status["progress"][-1]["message"] = success_msg
    _notify_status_changed()


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
    }


def _maybe_save_state(plugin=None):
    """Throttled persistence save — called from on_data_changed callbacks.

    Saves at most once per ``_SAVE_INTERVAL`` seconds to avoid hammering the
    database during rapid discovery bursts.
    """
    global _last_save_time
    now = time.time()
    if now - _last_save_time < _SAVE_INTERVAL:
        return
    _last_save_time = now
    try:
        _save_current_state()
        logger.debug("Throttled state save completed")
    except Exception as e:
        logger.warning(f"Throttled state save failed: {e}")


def _save_current_state():
    """Save *new and modified* plugin data to persistence (incremental).

    Compares each item's fingerprint to the last-saved snapshot and only
    writes items that are new or have changed, avoiding redundant DB
    round-trips.
    """
    global client, persistence, _persisted_fingerprints

    if not persistence or not client:
        return

    saved_counts: Dict[str, int] = {}

    for plugin_name, plugin in client.plugins.items():
        count = 0

        # --- Tools ---
        discovered_tools = getattr(plugin, 'discovered_tools', {})
        for tool_name, tool_data in discovered_tools.items():
            models = sorted(tool_data.get('models', set()))
            trace_count = len(tool_data.get('traces', []))
            metadata = tool_data.get('metadata', {})
            source_name = metadata.get('tool_source_name', '')
            source_id = metadata.get('tool_source_id', '')
            fp = ("tool", tool_name, tuple(models), trace_count,
                  source_name, source_id)
            if _persisted_fingerprints.get(f"tool:{tool_name}") == fp:
                continue
            try:
                persistence.save_tool(tool_name, models, trace_count, metadata)
                _persisted_fingerprints[f"tool:{tool_name}"] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save tool {tool_name}: {e}")

        # --- Agents ---
        discovered_agents = getattr(plugin, 'discovered_agents', {})
        for agent_name, agent_data in discovered_agents.items():
            tools_used = sorted(agent_data.get("tools_used", set()))
            models_used = sorted(agent_data.get("models_used", set()))
            fp = ("agent", agent_name, tuple(tools_used), tuple(models_used))
            if _persisted_fingerprints.get(f"agent:{agent_name}") == fp:
                continue
            try:
                persistence.save_agent(
                    agent_id=agent_name,
                    name=agent_name,
                    tools_used=tools_used,
                    models_used=models_used,
                    first_seen=agent_data.get("first_seen"),
                    metadata=agent_data.get("metadata", {}),
                )
                _persisted_fingerprints[f"agent:{agent_name}"] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save agent {agent_name}: {e}")

        # --- Downstream systems ---
        discovered_downstream = getattr(plugin, 'discovered_downstream', {})
        for sys_id, sys_data in discovered_downstream.items():
            tools_conn = sorted(sys_data.get("tools_connecting", set()))
            fp = ("ds", sys_id, sys_data.get("type", "unknown"), tuple(tools_conn))
            if _persisted_fingerprints.get(f"ds:{sys_id}") == fp:
                continue
            try:
                persistence.save_downstream_system(
                    system_id=sys_id,
                    name=sys_data.get("name", ""),
                    system_type=sys_data.get("type", "unknown"),
                    endpoint=sys_data.get("endpoint"),
                    tools_connecting=tools_conn,
                    first_seen=sys_data.get("first_seen"),
                    metadata=sys_data.get("metadata", {}),
                )
                _persisted_fingerprints[f"ds:{sys_id}"] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save downstream {sys_id}: {e}")

        # --- Lineage ---
        lineage = getattr(plugin, 'lineage', {})
        for rel_key, rel in lineage.items():
            weight = rel.get("weight", 1)
            fp = ("lin", rel["source_id"], rel["target_id"],
                  rel["relationship_type"], weight)
            fk = f"lin:{rel['source_id']}->{rel['target_id']}:{rel['relationship_type']}"
            if _persisted_fingerprints.get(fk) == fp:
                continue
            try:
                persistence.save_lineage(
                    source_id=rel["source_id"],
                    source_type=rel["source_type"],
                    target_id=rel["target_id"],
                    target_type=rel["target_type"],
                    relationship_type=rel["relationship_type"],
                    weight=weight,
                )
                _persisted_fingerprints[fk] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save lineage {fk}: {e}")

        # --- Models ---
        # Collect model names from all three sources (matching the
        # plugin's _list_models): model_providers, discovered_tools,
        # and discovered_agents.
        model_providers = getattr(plugin, 'model_providers', {})
        model_names: set = set(model_providers.keys())
        for td in discovered_tools.values():
            model_names.update(td.get('models', set()))
        for ad in discovered_agents.values():
            model_names.update(ad.get('models_used', set()))

        for model_name in model_names:
            provider = model_providers.get(model_name) or (
                model_name.split("/")[0] if "/" in model_name else "unknown"
            )
            tools_using = sorted(
                t for t, td in discovered_tools.items()
                if model_name in td.get('models', set())
            )
            # Prefer model_call_count (counts all spans) over tool-trace count
            model_call_count = getattr(plugin, 'model_call_count', {})
            usage_count = model_call_count.get(model_name, 0) or sum(
                len([tr for tr in td.get('traces', []) if tr.get('model') == model_name])
                for td in discovered_tools.values()
            )
            fp = ("model", model_name, provider, tuple(tools_using), usage_count)
            if _persisted_fingerprints.get(f"model:{model_name}") == fp:
                continue
            try:
                persistence.save_model(model_name, provider, tools_using, usage_count)
                _persisted_fingerprints[f"model:{model_name}"] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save model {model_name}: {e}")

        # --- MCP entities ---
        mcp_servers = getattr(plugin, 'mcp_servers', {})
        for server_id, sd in mcp_servers.items():
            fp = ("mcp_srv", server_id, sd.get('name', ''), sd.get('transport'))
            if _persisted_fingerprints.get(f"mcp_srv:{server_id}") == fp:
                continue
            try:
                persistence.save_mcp_server(
                    server_id=server_id, name=sd.get('name', ''),
                    transport=sd.get('transport'), endpoint=sd.get('endpoint'),
                    command=sd.get('command'), args=sd.get('args'),
                    env=sd.get('env'), source_file=sd.get('source_file'),
                    source_env_var=sd.get('source_env_var'),
                    metadata=sd.get('metadata', {}),
                )
                _persisted_fingerprints[f"mcp_srv:{server_id}"] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save MCP server {server_id}: {e}")

        mcp_tools = getattr(plugin, 'mcp_tools', {})
        for tool_id, td in mcp_tools.items():
            fp = ("mcp_tool", tool_id, td.get('name', ''), td.get('server_id', ''))
            if _persisted_fingerprints.get(f"mcp_tool:{tool_id}") == fp:
                continue
            try:
                persistence.save_mcp_tool(
                    tool_id=tool_id, server_id=td.get('server_id', ''),
                    name=td.get('name', ''), description=td.get('description'),
                    schema=td.get('schema'), usage=td.get('usage'),
                    metadata=td.get('metadata', {}),
                )
                _persisted_fingerprints[f"mcp_tool:{tool_id}"] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save MCP tool {tool_id}: {e}")

        mcp_resources = getattr(plugin, 'mcp_resources', {})
        for res_id, rd in mcp_resources.items():
            fp = ("mcp_res", res_id, rd.get('uri', ''), rd.get('server_id', ''))
            if _persisted_fingerprints.get(f"mcp_res:{res_id}") == fp:
                continue
            try:
                persistence.save_mcp_resource(
                    resource_id=res_id, server_id=rd.get('server_id', ''),
                    uri=rd.get('uri', ''), name=rd.get('name'),
                    type=rd.get('type'), mime_type=rd.get('mime_type'),
                    description=rd.get('description'), usage=rd.get('usage'),
                    metadata=rd.get('metadata', {}),
                )
                _persisted_fingerprints[f"mcp_res:{res_id}"] = fp
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save MCP resource {res_id}: {e}")

        if count:
            saved_counts[plugin_name] = count

    if saved_counts:
        parts = [f"{c} from {p}" for p, c in saved_counts.items()]
        logger.info("Persisted %s", ", ".join(parts))


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
                for p in client.plugins.values():
                    if getattr(p, 'plugin_type', None) == 'databricks':
                        export_data = p.export_assets()
                        assets["data_assets"] = export_data.get("data_assets", [])
                        break
            except Exception as e:
                logger.warning(f"Could not list Databricks data assets: {e}")

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

    # Close database engine
    try:
        from open_cite.db import close_db
        close_db()
        logger.info("Closed database connection")
    except Exception as e:
        logger.warning(f"Error closing database: {e}")


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
        if persistence:
            client.persistence = persistence
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

    # Note: auto_configure_plugins is already called inside create_app()
    # when config.auto_start is True — no need to call it again here.

    print(f"\n{'='*60}")
    print(f"  OpenCITE API Service (HTTP/2 via Hypercorn)")
    print(f"{'='*60}")
    print(f"  API: http://{host}:{port}")
    print(f"  Health: http://{host}:{port}/healthz")
    print(f"  Readiness: http://{host}:{port}/readyz")
    if config.otlp_embedded:
        print(f"  OTLP Traces (JSON): http://{host}:{port}/v1/traces")
        print(f"  OTLP Logs   (JSON): http://{host}:{port}/v1/logs")
        print(f"  OTLP (gRPC): http://{host}:{port} (HTTP/2)")
    elif config.enable_otel:
        print(f"  OTLP Receiver: http://{config.otlp_host}:{config.otlp_port}/v1/traces")
    if config.persistence_enabled:
        db_url = config.database_url or f"sqlite:///{config.db_path}"
        # Mask credentials in display
        display_url = db_url.split("@")[-1] if "@" in db_url else db_url
        print(f"  Persistence: {display_url}")
    else:
        print(f"  Persistence: disabled (in-memory only)")
    if plugin_store and plugin_store.enabled:
        print(f"  Plugin Store: SQLAlchemy (shared database)")
    print(f"{'='*60}\n")

    import asyncio
    import signal
    from open_cite.asgi.app import create_asgi_app
    from hypercorn.config import Config as HyperConfig
    from hypercorn.asyncio import serve

    ingest_fn = _otlp_ingest if _default_otel_plugin else None
    logs_ingest_fn = _otlp_ingest_logs if _default_otel_plugin else None
    asgi_app = create_asgi_app(
        app, sio_server=None, ingest_fn=ingest_fn,
        logs_ingest_fn=logs_ingest_fn,
    )

    hconfig = HyperConfig()
    hconfig.bind = [f"{host}:{port}"]

    async def _serve():
        shutdown_event = asyncio.Event()

        def _signal_handler():
            if shutdown_event.is_set():
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
            shutdown_cleanup()

    asyncio.run(_serve())


# For gunicorn: gunicorn "open_cite.api.app:create_app()"
if __name__ == '__main__':
    run_api()
