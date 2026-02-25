"""
OpenTelemetry discovery plugin for Open Cite.

This plugin receives OpenTelemetry traces via OTLP/HTTP protocol and discovers
tools that use models through OpenRouter.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
import re

from open_cite.core import BaseDiscoveryPlugin, LoggingDefaultDict

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    """Get the local IP address for display purposes."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
        except Exception:
            return '127.0.0.1'
        finally:
            s.close()
    except Exception:
        return '127.0.0.1'

# Configure regex patterns here to capture additional attributes
# Example: [r"^custom\..*", r"^app\.metadata\..*"]
DEFAULT_ATTRIBUTE_PATTERNS = []


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP Server that handles each request in a separate thread."""
    daemon_threads = True  # Don't wait for threads to finish on shutdown


class OTLPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OTLP protocol."""

    def do_POST(self):
        """Handle POST requests for trace and log ingestion."""
        if self.path in ("/v1/traces", "/v1/traces/"):
            self._handle_traces()
        elif self.path in ("/v1/logs", "/v1/logs/"):
            self._handle_logs()
        else:
            self.send_response(404)
            self.end_headers()

    def _read_body_and_headers(self):
        """Read request body and extract forwarded headers."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        _SKIP_HEADERS = {"content-length", "transfer-encoding", "connection",
                         "keep-alive", "te", "trailers", "upgrade"}
        inbound_headers = {}
        for key, value in self.headers.items():
            if key.lower() in _SKIP_HEADERS:
                continue
            if key.lower() == "host":
                inbound_headers["OTEL-HOST"] = value
            else:
                inbound_headers[key] = value

        return body, inbound_headers

    def _handle_traces(self):
        """Handle POST /v1/traces for trace ingestion."""
        body, inbound_headers = self._read_body_and_headers()
        plugin = self.server.plugin

        try:
            content_type = self.headers.get("Content-Type", "")

            if "application/json" in content_type:
                data = json.loads(body.decode("utf-8"))
                plugin._ingest_traces(data)
                plugin._deliver_to_webhooks(data, inbound_headers=inbound_headers)

                num_spans = sum(len(rs.get("scopeSpans", [])) for rs in data.get("resourceSpans", []))
                logger.warning(f"[OTLP] Received trace with {num_spans} scope spans")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
            else:
                self.send_response(415)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Only JSON format supported"}).encode()
                )
        except Exception as e:
            logger.error(f"Error processing traces: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_logs(self):
        """Handle POST /v1/logs — convert OTLP logs to synthetic traces."""
        body, inbound_headers = self._read_body_and_headers()
        plugin = self.server.plugin

        try:
            content_type = self.headers.get("Content-Type", "")

            if "application/json" in content_type:
                data = json.loads(body.decode("utf-8"))

                from open_cite.plugins.logs_adapter import convert_logs_to_traces
                synthetic_traces = convert_logs_to_traces(data)

                if synthetic_traces.get("resourceSpans"):
                    plugin._ingest_traces(synthetic_traces)
                    plugin._deliver_to_webhooks(synthetic_traces, inbound_headers=inbound_headers)

                num_log_records = sum(
                    len(sl.get("logRecords", []))
                    for rl in data.get("resourceLogs", [])
                    for sl in rl.get("scopeLogs", [])
                )
                logger.warning(f"[OTLP] Received {num_log_records} log records, converted to traces")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
            else:
                self.send_response(415)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Only JSON format supported"}).encode()
                )
        except Exception as e:
            logger.error(f"Error processing logs: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        """Override to use Python logging."""
        # Log at WARNING level so it shows up in default logging
        logger.warning(f"[OTLP] {self.address_string()} - {format % args}")


class OpenTelemetryPlugin(BaseDiscoveryPlugin):
    """
    OpenTelemetry discovery plugin.

    Receives OTLP traces via HTTP and discovers tools using AI models from any provider.
    Supports standard GenAI semantic conventions for LLM observability.
    """

    plugin_type = "opentelemetry"

    @classmethod
    def plugin_metadata(cls):
        import socket
        local_ip = _get_local_ip()
        return {
            "name": "OpenTelemetry",
            "description": "Discovers AI tools using models via OTLP traces",
            "required_fields": {
                "port": {"label": "OTLP HTTP Port", "default": 4318, "required": False, "type": "number"},
                "host": {"label": "Bind Address", "default": "0.0.0.0", "required": False, "type": "text"},
            },
            "trace_endpoints": {
                "localhost": "http://localhost:4318/v1/traces",
                "network": f"http://{local_ip}:4318/v1/traces" if local_ip != '127.0.0.1' else None,
            },
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        import socket as _socket

        embedded_receiver = config.get('embedded_receiver', False)
        host = config.get('host', '0.0.0.0')
        port = int(config.get('port', 4318))

        # Only check port availability for standalone (non-embedded) instances
        if not embedded_receiver:
            try:
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                sock.close()
            except OSError:
                raise ValueError(
                    f"Port {port} is already in use. "
                    f"Choose a different port for this OpenTelemetry instance."
                )

        return cls(
            host=host,
            port=port,
            instance_id=instance_id,
            display_name=display_name,
            persist_mappings=config.get('persist_mappings', True),
            mapping_store_path=config.get('mapping_store_path'),
            embedded_receiver=embedded_receiver,
        )

    def start(self):
        """Start the OTLP trace receiver."""
        if self._embedded_receiver:
            self._status = "running"
            logger.info(f"Started OpenTelemetry plugin {self.instance_id} (embedded receiver — no standalone server)")
        else:
            self.start_receiver()
            self._status = "running"
            logger.info(f"Started OpenTelemetry plugin {self.instance_id}")

    def stop(self):
        """Stop the OTLP trace receiver."""
        if self._embedded_receiver:
            self._status = "stopped"
            logger.info(f"Stopped OpenTelemetry plugin {self.instance_id} (embedded receiver)")
        else:
            self.stop_receiver()
            self._status = "stopped"
            logger.info(f"Stopped OpenTelemetry plugin {self.instance_id}")

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4318,
        attribute_patterns: List[str] = None,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
        persist_mappings: bool = True,
        mapping_store_path: Optional[str] = None,
        embedded_receiver: bool = False,
    ):
        """
        Initialize the OpenTelemetry plugin.

        Args:
            host: Host to bind the OTLP receiver to
            port: Port to bind the OTLP receiver to (default: 4318, standard OTLP/HTTP)
            attribute_patterns: Optional list of regex patterns to match attributes to collect
            instance_id: Unique identifier for this plugin instance
            display_name: Human-readable name for this instance
            persist_mappings: Whether to persist identity mappings to disk
            mapping_store_path: Path to the mapping JSON file (None = default)
            embedded_receiver: If True, skip standalone HTTP server (traces arrive
                via the main app's /v1/traces and gRPC routes instead)
        """
        super().__init__(instance_id=instance_id, display_name=display_name)
        self.host = host
        self.port = port
        self._embedded_receiver = embedded_receiver
        self._persist_mappings = persist_mappings
        self._mapping_store_path = mapping_store_path
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

        # Use provided patterns or fallback to default configuration
        patterns = attribute_patterns if attribute_patterns is not None else DEFAULT_ATTRIBUTE_PATTERNS
        self.attribute_patterns = [re.compile(p) for p in patterns]

        # Trace storage: {trace_id: {resource_spans, scope_spans, etc.}}
        self.traces: Dict[str, Dict[str, Any]] = {}

        # Discovered tools: {tool_name: {models: set(), traces: list(), metadata: dict()}}
        _pid = self._instance_id
        self.discovered_tools: Dict[str, Dict[str, Any]] = LoggingDefaultDict(
            lambda: {"models": set(), "traces": [], "metadata": {}},
            asset_type="tool", plugin_id=_pid,
        )

        # Discovered agents: {agent_name: {tools_used: set(), models_used: set(), ...}}
        self.discovered_agents: Dict[str, Dict[str, Any]] = LoggingDefaultDict(
            lambda: {
                "tools_used": set(),
                "models_used": set(),
                "first_seen": None,
                "last_seen": None,
                "metadata": {},
            },
            asset_type="agent", plugin_id=_pid,
        )

        # Discovered downstream systems: {system_id: {...}}
        self.discovered_downstream: Dict[str, Dict[str, Any]] = LoggingDefaultDict(
            lambda: {
                "name": "",
                "type": "unknown",
                "endpoint": None,
                "tools_connecting": set(),
                "first_seen": None,
                "last_seen": None,
                "metadata": {},
            },
            asset_type="downstream_system", plugin_id=_pid,
        )

        # Lineage relationships: list of (source_id, source_type, target_id, target_type, rel_type)
        self.lineage: Dict[str, Dict[str, Any]] = {}

        # Token usage tracking per model: {model_name: {input_tokens, output_tokens}}
        self.model_token_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0}
        )

        # Per-model span call count (every span where a model is seen)
        self.model_call_count: Dict[str, int] = defaultdict(int)

        # Provider per model: {model_name: provider_string}
        self.model_providers: Dict[str, str] = {}

        # MCP storage (discovered from traces)
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}
        self.mcp_resources: Dict[str, Dict[str, Any]] = {}
        self.mcp_usage_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"tool_calls": 0, "resource_accesses": 0}
        )

        # Per-session cache of user.* attributes so that every span in the
        # same session inherits user context seen on earlier spans.
        # {session_id: {"user.email": "...", "user.account_uuid": "...", ...}}
        self._session_user_attrs: Dict[str, Dict[str, Any]] = {}

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Periodic save throttling (once per 60 seconds)
        self._last_save_time = 0
        self._save_interval = 60  # seconds

        # Identifier for tool source identification
        from ..identifier import ToolIdentifier
        self.identifier = ToolIdentifier(
            mapping_path=self._mapping_store_path,
            persist=self._persist_mappings,
        )

    def _maybe_save_state(self):
        """Periodically save state to persistence (throttled to once per interval)."""
        now = time.time()
        if now - self._last_save_time < self._save_interval:
            return

        self._last_save_time = now

        # Trigger save via the API app's _save_current_state function
        try:
            from open_cite.api import app as api_app
            if api_app.persistence and api_app.client:
                api_app._save_current_state()
                logger.debug("Periodic state save completed")
        except Exception as e:
            logger.warning(f"Periodic state save failed: {e}")

    def export_assets(self) -> Dict[str, Any]:
        """Export OTel-discovered assets in OpenCITE schema format."""
        from open_cite.schema import (
            ToolFormatter, ModelFormatter, parse_model_id,
            MCPServerFormatter, MCPToolFormatter, MCPResourceFormatter,
        )

        tools = []
        for tool in self.list_assets("tool"):
            models_used = []
            for model_name in tool.get("models", []):
                models_used.append({
                    "model_id": model_name,
                    "usage_count": len([
                        t for t in tool.get("traces", [])
                        if t.get("model") == model_name
                    ]),
                })
            tools.append(ToolFormatter.format_tool(
                tool_id=tool["name"],
                name=tool["name"],
                discovery_source=tool.get("metadata", {}).get("discovery_source", "opentelemetry"),
                type="application",
                models_used=models_used,
                provider=None,
                last_seen=tool.get("metadata", {}).get("last_seen"),
                metadata=tool.get("metadata", {}),
            ))

        models = []
        for model in self.list_assets("model"):
            model_id = model["name"]
            parsed = parse_model_id(model_id)
            models.append(ModelFormatter.format_model(
                model_id=model_id,
                name=model_id,
                discovery_source="opentelemetry",
                provider=parsed["provider"],
                model_family=parsed["model_family"],
                model_version=parsed["model_version"],
                usage={
                    "total_calls": model.get("usage_count", 0),
                    "unique_tools": len(model.get("tools", [])),
                    "tools_using": model.get("tools", []),
                },
            ))

        mcp_servers = []
        for server in self.list_assets("mcp_server"):
            mcp_servers.append(MCPServerFormatter.format_mcp_server(
                server_id=server["id"],
                name=server["name"],
                discovery_source=server.get("discovery_source", self.instance_id),
                transport=server.get("transport", "unknown"),
                endpoint=server.get("endpoint"),
                command=server.get("command"),
                args=server.get("args"),
                env=server.get("env"),
                tools_count=server.get("tools_count", 0),
                resources_count=server.get("resources_count", 0),
                source_file=server.get("source_file"),
                source_env_var=server.get("source_env_var"),
                metadata=server.get("metadata", {}),
            ))

        mcp_tools = []
        for tool in self.list_assets("mcp_tool"):
            mcp_tools.append(MCPToolFormatter.format_mcp_tool(
                tool_id=tool["id"],
                name=tool["name"],
                server_id=tool["server_id"],
                discovery_source=tool.get("discovery_source"),
                description=tool.get("description"),
                schema=tool.get("schema"),
                usage=tool.get("usage"),
                metadata=tool.get("metadata"),
            ))

        mcp_resources = []
        for resource in self.list_assets("mcp_resource"):
            mcp_resources.append(MCPResourceFormatter.format_mcp_resource(
                resource_id=resource["id"],
                uri=resource["uri"],
                server_id=resource["server_id"],
                name=resource.get("name"),
                discovery_source=resource.get("discovery_source"),
                type=resource.get("type"),
                mime_type=resource.get("mime_type"),
                description=resource.get("description"),
                usage=resource.get("usage"),
                metadata=resource.get("metadata"),
            ))

        return {
            "tools": tools,
            "models": models,
            "mcp_servers": mcp_servers,
            "mcp_tools": mcp_tools,
            "mcp_resources": mcp_resources,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return plugin configuration."""
        if self._embedded_receiver:
            return {
                "embedded": True,
            }
        return {
            "host": self.host,
            "port": self.port,
            "endpoint": f"http://{self.host}:{self.port}/v1/[logs/traces]",
        }

    @property
    def supported_asset_types(self) -> Set[str]:
        """Asset types supported by this plugin."""
        return {"tool", "model", "agent", "downstream_system", "mcp_server", "mcp_tool", "mcp_resource"}

    def get_identification_attributes(self) -> List[str]:
        """Return a list of attribute keys used for tool identification."""
        return [
            "trace.metadata.openrouter.entity_id",
            "trace.metadata.openrouter.api_key_name",
            "trace.metadata.openrouter.creator_user_id",
            "mcp.server.name",
            "mcp.server.endpoint",
        ]

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify the OTLP receiver is running.

        Returns:
            Dict with connection status
        """
        if self._embedded_receiver:
            is_running = self._status == "running"
            return {
                "success": is_running,
                "receiver_running": is_running,
                "embedded": True,
                "traces_received": len(self.traces),
                "tools_discovered": len(self.discovered_tools),
                "mcp_servers_discovered": len(self.mcp_servers),
                "mcp_tools_discovered": len(self.mcp_tools),
                "mcp_resources_discovered": len(self.mcp_resources),
            }

        is_running = self.server_thread is not None and self.server_thread.is_alive()
        return {
            "success": is_running,
            "receiver_running": is_running,
            "host": self.host,
            "port": self.port,
            "endpoint": f"http://{self.host}:{self.port}/v1/traces",
            "traces_received": len(self.traces),
            "tools_discovered": len(self.discovered_tools),
            "mcp_servers_discovered": len(self.mcp_servers),
            "mcp_tools_discovered": len(self.mcp_tools),
            "mcp_resources_discovered": len(self.mcp_resources),
        }

    def start_receiver(self):
        """Start the OTLP HTTP receiver in a background thread."""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("OTLP receiver is already running")
            return

        # Event to signal when server is ready (or failed)
        server_ready = threading.Event()
        server_error: list = []

        def run_server():
            try:
                self.server = ThreadingHTTPServer((self.host, self.port), OTLPRequestHandler)
                self.server.plugin = self  # Pass plugin reference to handler
                logger.info(f"OTLP receiver started on {self.host}:{self.port}")
                server_ready.set()  # Signal that server is ready
                self.server.serve_forever()
            except OSError as e:
                server_error.append(e)
                server_ready.set()  # Unblock the waiting thread immediately

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to actually start (timeout after 5 seconds)
        if not server_ready.wait(timeout=5.0):
            self.server_thread = None
            logger.error("OTLP receiver failed to start within 5 seconds")
            raise RuntimeError("OTLP receiver failed to start")

        if server_error:
            self.server_thread = None
            err = server_error[0]
            logger.error(f"OTLP receiver failed to bind port {self.port}: {err}")
            raise RuntimeError(
                f"Port {self.port} is already in use. "
                f"Choose a different port for this OpenTelemetry instance."
            )

        logger.info("OTLP receiver thread started")

    def stop_receiver(self):
        """Stop the OTLP HTTP receiver."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()  # Close the socket
            self.server = None

            # Wait for the server thread to stop
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=2.0)
            self.server_thread = None

            logger.info("OTLP receiver stopped")

    def _enrich_session_user_attrs(
        self,
        attributes: List[Dict[str, Any]],
        resource: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Collect ``user.*`` attributes per ``session.id`` and inject cached values.

        When any span in a session carries user attributes (e.g.
        ``user.email``, ``user.account_uuid``), they are cached by
        ``session.id``.  Subsequent spans in the same session that lack
        those attributes get them injected so they propagate to
        tool/agent metadata and stored traces.
        """
        attr_dict = self._attrs_to_dict(attributes)
        res_dict = self._attrs_to_dict(resource.get("attributes", []))
        merged = {**res_dict, **attr_dict}

        session_id = merged.get("session.id")
        if not session_id:
            return attributes

        # Collect user.* attributes from this span into the session cache
        user_attrs = {k: v for k, v in merged.items() if k.startswith("user.")}
        if user_attrs:
            if session_id not in self._session_user_attrs:
                self._session_user_attrs[session_id] = {}
            self._session_user_attrs[session_id].update(user_attrs)

        # Inject any cached user attributes not already on this span
        cached = self._session_user_attrs.get(session_id)
        if not cached:
            return attributes

        existing_keys = {a.get("key") for a in attributes}
        missing = {k: v for k, v in cached.items() if k not in existing_keys}
        if not missing:
            return attributes

        injected = list(attributes)  # shallow copy
        for key, value in missing.items():
            if isinstance(value, bool):
                injected.append({"key": key, "value": {"boolValue": value}})
            elif isinstance(value, int):
                injected.append({"key": key, "value": {"intValue": value}})
            elif isinstance(value, float):
                injected.append({"key": key, "value": {"doubleValue": value}})
            else:
                injected.append({"key": key, "value": {"stringValue": str(value)}})

        return injected

    def _ingest_traces(self, otlp_data: Dict[str, Any]):
        """
        Ingest OTLP trace data and extract tool/model information.

        Args:
            otlp_data: OTLP JSON payload
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[OTLP] Raw ingested data: {json.dumps(otlp_data, indent=2)}")

        try:
            with self._lock:
                # OTLP format: {"resourceSpans": [...]}
                resource_spans = otlp_data.get("resourceSpans", [])

                # Track per-span detections for parent-child correlation
                span_agents: Dict[str, str] = {}   # span_id -> agent_name
                span_tools: Dict[str, str] = {}     # span_id -> tool_name
                span_models: Dict[str, str] = {}    # span_id -> model_name
                span_parents: Dict[str, str] = {}   # span_id -> parent_span_id
                span_attrs: Dict[str, Dict[str, Any]] = {}  # span_id -> merged attributes
                span_traces: Dict[str, str] = {}   # span_id -> trace_id
                span_times: Dict[str, str] = {}    # span_id -> ISO timestamp from span

                for resource_span in resource_spans:
                    resource = resource_span.get("resource", {})
                    scope_spans = resource_span.get("scopeSpans", [])

                    for scope_span in scope_spans:
                        spans = scope_span.get("spans", [])

                        for span in spans:
                            trace_id = span.get("traceId", "")
                            span_id = span.get("spanId", "")
                            parent_span_id = span.get("parentSpanId", "")
                            span_name = span.get("name", "")
                            attributes = span.get("attributes", [])

                            # Extract span timestamp (prefer startTimeUnixNano)
                            _start_ns = span.get("startTimeUnixNano")
                            if _start_ns:
                                try:
                                    _ts_sec = int(_start_ns) / 1_000_000_000
                                    span_times[span_id] = datetime.utcfromtimestamp(_ts_sec).isoformat()
                                except (ValueError, TypeError, OSError):
                                    pass

                            # Enrich with session-level user attributes
                            attributes = self._enrich_session_user_attrs(
                                attributes, resource)

                            if parent_span_id:
                                span_parents[span_id] = parent_span_id

                            # Store the trace
                            if trace_id not in self.traces:
                                self.traces[trace_id] = {
                                    "trace_id": trace_id,
                                    "spans": [],
                                    "first_seen": span_times.get(span_id) or datetime.utcnow().isoformat(),
                                }

                            self.traces[trace_id]["spans"].append({
                                "span_id": span_id,
                                "span_name": span_name,
                                "attributes": attributes,
                                "resource": resource,
                            })

                            # Detect OpenRouter usage (tools/models)
                            tool_name = self._detect_tool_usage(
                                trace_id, span_id, span_name, attributes, resource,
                                span_time=span_times.get(span_id),
                            )
                            if tool_name:
                                span_tools[span_id] = tool_name

                            # Track model per span for agent-model correlation
                            attr_dict = self._attrs_to_dict(attributes)
                            res_dict = self._attrs_to_dict(resource.get("attributes", []))
                            merged = {**res_dict, **attr_dict}
                            span_model = (
                                merged.get("gen_ai.request.model")
                                or merged.get("gen_ai.response.model")
                                or merged.get("llm.model")
                                or merged.get("model")
                            )
                            if span_model:
                                span_models[span_id] = span_model
                                self.model_call_count[span_model] += 1
                            span_attrs[span_id] = merged
                            span_traces[span_id] = trace_id

                            # Detect agents
                            agent_name = self._detect_agent(
                                trace_id, span_id, span_name, attributes, resource,
                                span_time=span_times.get(span_id),
                            )
                            if agent_name:
                                span_agents[span_id] = agent_name

                            # Detect downstream systems
                            self._detect_downstream_system(
                                trace_id, span_id, span_name, attributes, resource
                            )

                            # Extract token usage
                            self._extract_token_usage(attributes)

                            # Detect MCP tool/resource usage
                            self._detect_mcp_usage(
                                trace_id, span_id, span_name, attributes, resource
                            )

                            # Detect tools from span events (e.g. Claude Code
                            # tool_result / tool_use events)
                            event_tools = self._detect_tools_from_events(
                                trace_id, span_id, span_name, span, resource,
                                span_time=span_times.get(span_id),
                            )

                            # When a span emits tool-use events it is
                            # orchestrating tool calls (e.g. Claude Code
                            # calling Read, Write, Bash).  Auto-detect
                            # the span as an agent if no explicit agent
                            # attribute was present, and create direct
                            # lineage links on the same span.
                            if event_tools:
                                _agent_for_span = span_agents.get(span_id)
                                if not _agent_for_span:
                                    _ev_res = self._attrs_to_dict(
                                        resource.get("attributes", []))
                                    _ev_spa = self._attrs_to_dict(
                                        span.get("attributes", []))
                                    _ev_m = {**_ev_res, **_ev_spa}
                                    _auto_agent = (
                                        _ev_m.get("service.name")
                                        or _ev_m.get("app.name")
                                    )
                                    if _auto_agent:
                                        _ag = self.discovered_agents[_auto_agent]
                                        _now = span_times.get(span_id) or datetime.utcnow().isoformat()
                                        if not _ag["first_seen"] or _now < _ag["first_seen"]:
                                            _ag["first_seen"] = _now
                                        if not _ag["last_seen"] or _now > _ag["last_seen"]:
                                            _ag["last_seen"] = _now
                                        _ag["metadata"].update({
                                            "last_seen": _ag["last_seen"],
                                            "discovery_source": _ev_m.get("opencite.discovery_source", "opentelemetry"),
                                        })
                                        for _k in ("service.name",
                                                    "service.version",
                                                    "gen_ai.system",
                                                    "enduser.id",
                                                    "net.peer.ip"):
                                            if _k in _ev_m:
                                                _ag["metadata"][_k] = _ev_m[_k]
                                        for _k, _v in _ev_m.items():
                                            if _v is not None and _k.startswith(
                                                ("user.", "ai_gateway.")
                                            ):
                                                _ag["metadata"][_k] = _v
                                        span_agents[span_id] = _auto_agent
                                        _agent_for_span = _auto_agent

                                        # Reclassify: remove from tools if
                                        # _detect_tool_usage registered
                                        # the service name as a tool.
                                        self.discovered_tools.pop(
                                            _auto_agent, None)
                                        if span_tools.get(span_id) == _auto_agent:
                                            del span_tools[span_id]

                                        # Link agent -> model on same span
                                        if span_id in span_models:
                                            _model = span_models[span_id]
                                            _ag["models_used"].add(_model)
                                            self._add_lineage(
                                                _auto_agent, "agent",
                                                _model, "model", "uses")

                                # Direct agent -> tool lineage (same span)
                                if _agent_for_span:
                                    for _et in event_tools:
                                        if _et != _agent_for_span:
                                            self.discovered_agents[
                                                _agent_for_span
                                            ]["tools_used"].add(_et)
                                            self._add_lineage(
                                                _agent_for_span, "agent",
                                                _et, "tool", "uses")

                            for et in event_tools:
                                span_tools.setdefault(span_id, et)

                # ----------------------------------------------------------
                # Post-processing stage 1: auto-detect agents.
                #
                # Spans that have model usage but were NOT registered as
                # tools (no explicit gen_ai.tool.name / tool.call.id) are
                # applications making LLM calls — classify as agents via
                # their service.name.
                # ----------------------------------------------------------
                for _sid, _model in span_models.items():
                    if _sid in span_agents or _sid in span_tools:
                        continue
                    # Skip child spans whose parent is already an agent —
                    # these are LLM calls made by the agent, not agents
                    # themselves (e.g. AI Gateway child LLM span).
                    _parent = span_parents.get(_sid)
                    if _parent and _parent in span_agents:
                        continue
                    _m = span_attrs.get(_sid, {})
                    _svc = _m.get("service.name") or _m.get("app.name")
                    if not _svc:
                        continue
                    _ag = self.discovered_agents[_svc]
                    _now = span_times.get(_sid) or datetime.utcnow().isoformat()
                    if not _ag["first_seen"] or _now < _ag["first_seen"]:
                        _ag["first_seen"] = _now
                    if not _ag["last_seen"] or _now > _ag["last_seen"]:
                        _ag["last_seen"] = _now
                    _ag["metadata"].update({
                        "last_seen": _ag["last_seen"],
                        "discovery_source": _m.get("opencite.discovery_source", "opentelemetry"),
                    })
                    for _k in ("service.name", "service.version",
                                "gen_ai.system", "gen_ai.operation.name",
                                "enduser.id", "net.peer.ip",
                                "http.url", "http.user_agent",
                                "http.status_code"):
                        if _k in _m:
                            _ag["metadata"][_k] = _m[_k]
                    for _k, _v in _m.items():
                        if _v is not None and _k.startswith(
                            ("user.", "ai_gateway.", "trace.metadata.")
                        ):
                            _ag["metadata"][_k] = _v
                    span_agents[_sid] = _svc
                    _ag["models_used"].add(_model)
                    self._add_lineage(_svc, "agent", _model, "model", "uses")
                    # Clean up if it was previously registered as a tool
                    self.discovered_tools.pop(_svc, None)

                # ----------------------------------------------------------
                # Post-processing stage 2: trace-level correlation.
                #
                # Link agents to every tool and model that appears in the
                # same trace.  This works regardless of parentSpanId and
                # regardless of whether spans arrived in the same batch.
                # ----------------------------------------------------------
                _trace_agents: Dict[str, Set[str]] = defaultdict(set)
                _trace_tools: Dict[str, Set[str]] = defaultdict(set)
                _trace_models: Dict[str, Set[str]] = defaultdict(set)

                for _sid, _aname in span_agents.items():
                    _tid = span_traces.get(_sid)
                    if _tid:
                        _trace_agents[_tid].add(_aname)
                for _sid, _tname in span_tools.items():
                    _tid = span_traces.get(_sid)
                    if _tid:
                        _trace_tools[_tid].add(_tname)
                for _sid, _mname in span_models.items():
                    _tid = span_traces.get(_sid)
                    if _tid:
                        _trace_models[_tid].add(_mname)

                for _tid, _agents in _trace_agents.items():
                    for _aname in _agents:
                        _ag = self.discovered_agents[_aname]
                        for _tname in _trace_tools.get(_tid, ()):
                            if _tname != _aname:
                                _ag["tools_used"].add(_tname)
                                self._add_lineage(
                                    _aname, "agent", _tname, "tool", "uses")
                        for _mname in _trace_models.get(_tid, ()):
                            _ag["models_used"].add(_mname)
                            self._add_lineage(
                                _aname, "agent", _mname, "model", "uses")

                # Correlate agents with tools via parent-child span relationships
                self._correlate_agent_tools(span_agents, span_tools, span_parents)

                # Correlate agents with models via parent-child span relationships
                self._correlate_agent_models(span_agents, span_models, span_parents)

                logger.info(f"Ingested traces. Total traces: {len(self.traces)}")

            # Notify after releasing the lock to avoid holding it during push
            self.notify_data_changed()

            # Periodically save state (throttled to once per minute)
            self._maybe_save_state()
        except Exception as e:
            logger.error(f"Error ingesting traces: {e}")

    def _detect_tool_usage(
        self,
        trace_id: str,
        span_id: str,
        span_name: str,
        attributes: List[Dict[str, Any]],
        resource: Dict[str, Any],
        span_time: Optional[str] = None,
    ) -> Optional[str]:
        """
        Detect if a span represents tool or LLM usage.

        Looks for GenAI semantic convention attributes:
        - gen_ai.request.model or gen_ai.response.model
        - llm.model or model attributes
        - service.name for tool identification

        Returns:
            The detected tool name, or None if no tool was detected.
        """
        # Convert attributes list to dict for easier access
        attr_dict = {}
        for attr in attributes:
            key = attr.get("key", "")
            value = attr.get("value", {})
            if "stringValue" in value:
                attr_dict[key] = value["stringValue"]
            elif "intValue" in value:
                attr_dict[key] = value["intValue"]
            elif "boolValue" in value:
                attr_dict[key] = value["boolValue"]
            elif "doubleValue" in value:
                attr_dict[key] = value["doubleValue"]

        # Extract resource attributes as well
        resource_dict = {}
        for attr in resource.get("attributes", []):
            key = attr.get("key", "")
            value = attr.get("value", {})
            if "stringValue" in value:
                resource_dict[key] = value["stringValue"]
            elif "intValue" in value:
                resource_dict[key] = value["intValue"]
            elif "boolValue" in value:
                resource_dict[key] = value["boolValue"]
            elif "doubleValue" in value:
                resource_dict[key] = value["doubleValue"]

        # Merged attributes for identification and metadata
        merged_attrs = {**resource_dict, **attr_dict}

        # Check for model in attributes (using GenAI semantic conventions)
        model_name = (
            merged_attrs.get("gen_ai.request.model")
            or merged_attrs.get("gen_ai.response.model")
            or merged_attrs.get("llm.model")
            or merged_attrs.get("model")
        )

        # Extract tool/service name
        # Priority:
        #   1. Explicit tool attribute (gen_ai.tool.name, tool.name)
        #   2. Span name when span is a tool call (has gen_ai.tool.call.id)
        #   3. service.name / app.name (but NOT when it equals the agent name)
        #   4. Span name as last resort
        is_tool_call = "gen_ai.tool.call.id" in attr_dict
        agent_name = (
            merged_attrs.get("gen_ai.agent.name")
            or merged_attrs.get("agent_name")
            or merged_attrs.get("agent.name")
        )

        # Explicit tool identity: gen_ai.tool.name, tool.name, or
        # gen_ai.tool.call.id (marks a tool-call span).
        has_explicit_tool = bool(
            attr_dict.get("gen_ai.tool.name")
            or attr_dict.get("tool.name")
            or is_tool_call
        )

        tool_name = (
            attr_dict.get("gen_ai.tool.name")
            or attr_dict.get("tool.name")
        )

        # For tool-call spans, prefer the span name (e.g. "CodeExecution")
        if not tool_name and is_tool_call and span_name:
            tool_name = span_name

        # Only register spans with explicit tool identity as tools.
        # Spans without gen_ai.tool.name / tool.name / gen_ai.tool.call.id
        # are either agents making LLM calls (has model) or plain service
        # spans (no model) — neither should be classified as tools.
        if not has_explicit_tool:
            if model_name:
                provider = (
                    merged_attrs.get("gen_ai.system")
                    or merged_attrs.get("gen_ai.provider.name")
                    or ""
                )
                if provider:
                    self.model_providers[model_name] = provider
            return None

        # Fall back to service.name, but skip if it matches the agent name
        if not tool_name:
            service_name = (
                merged_attrs.get("service.name")
                or merged_attrs.get("app.name")
            )
            if service_name and service_name != agent_name:
                tool_name = service_name

        # If no explicit service name, use span name or infer from trace
        if not tool_name:
            tool_name = span_name or f"tool_{trace_id[:8]}"

        # Skip recording this span as a tool if the resolved name is just the agent name
        if tool_name == agent_name:
            return None

        # Debug logging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Detection] trace_id={trace_id[:8]}... model_name={model_name}, "
                        f"tool_name={tool_name}")

        if not tool_name:
            return None

        # Touch the defaultdict entry so the tool is registered
        _ = self.discovered_tools[tool_name]

        # Always record the trace (the span itself is evidence of a call)
        self.discovered_tools[tool_name]["traces"].append({
            "trace_id": trace_id,
            "span_id": span_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name or "",
        })

        # Model-specific bookkeeping
        if model_name:
            if model_name not in self.discovered_tools[tool_name]["models"]:
                self.discovered_tools[tool_name]["models"].add(model_name)
                logger.info(f"Discovered tool '{tool_name}' using model '{model_name}'")

        # Store metadata (always, not only when a model is present)
        url = (
            merged_attrs.get("http.url")
            or merged_attrs.get("url.full")
            or merged_attrs.get("http.host")
            or ""
        )
        provider = (
            merged_attrs.get("gen_ai.system")
            or merged_attrs.get("gen_ai.provider.name")
            or ""
        )
        if provider and model_name:
            self.model_providers[model_name] = provider
        source = (
            merged_attrs.get("trace.metadata.source")
            or merged_attrs.get("source")
            or ""
        )

        # Perform tool identification
        identification = self.identifier.identify("opentelemetry", merged_attrs)
        if identification:
            self.discovered_tools[tool_name]["metadata"].update({
                "tool_source_name": identification.get("source_name"),
                "tool_source_id": identification.get("source_id")
            })

        # Store useful attributes in metadata for identification mapping
        _TOOL_METADATA_KEYS = (
            # OpenTelemetry semantic conventions
            "service.name", "service.version", "service.namespace",
            "gen_ai.system", "gen_ai.request.model", "gen_ai.response.model",
            "gen_ai.tool.name", "gen_ai.tool.call.id",
            # Tool input/output/parameters (Claude Code, generic GenAI)
            "gen_ai.tool.input", "gen_ai.tool.output",
            "gen_ai.tool.arguments", "gen_ai.tool.result",
            "input", "output", "input_json", "arguments", "command",
            "tool_input", "tool_output", "tool_result", "tool_parameters",
            # HTTP context
            "http.host", "http.url", "url.full", "server.address",
            # OpenRouter trace metadata
            "trace.metadata.openrouter.entity_id",
            "trace.metadata.openrouter.api_key_name",
            "trace.metadata.openrouter.creator_user_id",
            # MCP context
            "mcp.server.name", "mcp.server.endpoint",
            # Operation context
            "gen_ai.operation.name",
            # Common app-level attributes
            "app.name", "app.version",
        )
        for attr_key in _TOOL_METADATA_KEYS:
            if attr_key in merged_attrs:
                val = merged_attrs[attr_key]
                if isinstance(val, (dict, list)):
                    self.discovered_tools[tool_name]["metadata"][attr_key] = json.dumps(val, default=str)
                else:
                    self.discovered_tools[tool_name]["metadata"][attr_key] = val

        # Store span name
        if span_name:
            self.discovered_tools[tool_name]["metadata"]["span.name"] = span_name

        # Also store any trace.metadata.* and user.* keys
        for attr_key, attr_val in merged_attrs.items():
            if (attr_key.startswith("trace.metadata.") or attr_key.startswith("user.")) and attr_val is not None:
                if isinstance(attr_val, (dict, list)):
                    self.discovered_tools[tool_name]["metadata"][attr_key] = json.dumps(attr_val, default=str)
                else:
                    self.discovered_tools[tool_name]["metadata"][attr_key] = attr_val

        _tool_time = span_time or datetime.utcnow().isoformat()
        _prev_last = self.discovered_tools[tool_name]["metadata"].get("last_seen")
        if not _prev_last or _tool_time > _prev_last:
            self.discovered_tools[tool_name]["metadata"]["last_seen"] = _tool_time
        self.discovered_tools[tool_name]["metadata"].update({
            "url": url,
            "provider": provider,
            "source": source,
            "discovery_source": merged_attrs.get("opencite.discovery_source", "opentelemetry"),
        })

        # Check for attributes matching configured patterns
        if self.attribute_patterns:
            for key, value in merged_attrs.items():
                if key in self.discovered_tools[tool_name]["metadata"]:
                    continue

                for pattern in self.attribute_patterns:
                    if pattern.match(key):
                        if isinstance(value, (dict, list)):
                            self.discovered_tools[tool_name]["metadata"][key] = json.dumps(value, default=str)
                        else:
                            self.discovered_tools[tool_name]["metadata"][key] = value
                        break

        return tool_name

    # Tool-event attribute names that indicate a tool invocation
    _TOOL_EVENT_NAMES = frozenset({
        "tool_result", "tool_use", "tool_decision",
        "claude_code.tool_result", "claude_code.tool_use",
        "claude_code.tool_decision",
        "genai.tool_result", "genai.tool_use",
    })

    # Attribute keys that carry the tool name inside an event
    _EVENT_TOOL_NAME_KEYS = (
        "tool_name", "tool.name", "gen_ai.tool.name",
        "llm.tool.name", "ai.tool.name", "name",
    )

    def _detect_tools_from_events(
        self,
        trace_id: str,
        span_id: str,
        span_name: str,
        span: Dict[str, Any],
        resource: Dict[str, Any],
        span_time: Optional[str] = None,
    ) -> List[str]:
        """Extract tool names from span events.

        Handles telemetry patterns where tool invocations are recorded as
        span events (e.g. Claude Code emits ``tool_result`` /
        ``tool_use`` events with ``tool_name`` attributes).

        Returns a list of tool names discovered from the events.
        """
        events = span.get("events", [])
        if not events:
            return []

        res_dict = self._attrs_to_dict(resource.get("attributes", []))
        span_attr_dict = self._attrs_to_dict(span.get("attributes", []))
        merged_span = {**res_dict, **span_attr_dict}

        discovered: List[str] = []
        for event in events:
            event_name = event.get("name", "")
            # Strip common prefixes
            normalized = event_name
            for prefix in ("claude_code.", "genai."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]

            if event_name not in self._TOOL_EVENT_NAMES and normalized not in (
                "tool_result", "tool_use", "tool_decision",
            ):
                continue

            evt_attrs = self._attrs_to_dict(event.get("attributes", []))

            # Resolve tool name from event attributes, then span attributes
            tool_name = None
            for key in self._EVENT_TOOL_NAME_KEYS:
                tool_name = evt_attrs.get(key) or span_attr_dict.get(key)
                if tool_name:
                    break

            # Fall back to span name (many instrumentations name the span
            # after the tool, e.g. "Read", "Bash", "Write")
            if not tool_name:
                tool_name = span_name or f"tool_{trace_id[:8]}"

            if not tool_name:
                continue

            # Register the tool
            tool = self.discovered_tools[tool_name]
            _evt_time = span_time or datetime.utcnow().isoformat()
            _prev = tool["metadata"].get("last_seen")
            if not _prev or _evt_time > _prev:
                tool["metadata"]["last_seen"] = _evt_time
            tool["metadata"].update({
                "discovery_source": merged_span.get("opencite.discovery_source", "opentelemetry"),
            })

            # Carry over useful event attributes as metadata
            for k, v in evt_attrs.items():
                if k not in self._EVENT_TOOL_NAME_KEYS and v is not None:
                    if isinstance(v, (dict, list)):
                        tool["metadata"][k] = json.dumps(v, default=str)
                    else:
                        tool["metadata"][k] = v

            # Carry over service.name
            service = merged_span.get("service.name") or merged_span.get("app.name")
            if service:
                tool["metadata"]["service.name"] = service

            # Associate model if present on the span
            model_name = (
                merged_span.get("gen_ai.request.model")
                or merged_span.get("gen_ai.response.model")
                or merged_span.get("llm.model")
                or merged_span.get("model")
            )
            if model_name and model_name not in tool["models"]:
                tool["models"].add(model_name)

            tool["traces"].append({
                "trace_id": trace_id,
                "span_id": span_id,
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name or "",
                "event_name": event_name,
            })

            discovered.append(tool_name)

        return discovered

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List assets discovered from OpenTelemetry traces.

        Supported asset types:
        - "tool": Tools using AI models
        - "model": Models discovered across all tools
        - "agent": Agents detected from trace patterns
        - "downstream_system": Downstream systems (databases, APIs, etc.)

        Args:
            asset_type: Type of asset to list
            **kwargs: Additional filters

        Returns:
            List of assets
        """
        with self._lock:
            if asset_type == "tool":
                return self._list_tools(**kwargs)
            elif asset_type == "model":
                return self._list_models(**kwargs)
            elif asset_type == "agent":
                return self._list_agents(**kwargs)
            elif asset_type == "downstream_system":
                return self._list_downstream_systems(**kwargs)
            elif asset_type == "mcp_server":
                return self._list_mcp_servers(**kwargs)
            elif asset_type == "mcp_tool":
                return self._list_mcp_tools(**kwargs)
            elif asset_type == "mcp_resource":
                return self._list_mcp_resources(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: tool, model, agent, downstream_system, "
                    f"mcp_server, mcp_tool, mcp_resource"
                )

    def _list_tools(self, **kwargs) -> List[Dict[str, Any]]:
        """List discovered tools."""
        tools = []
        for tool_name, tool_data in self.discovered_tools.items():
            metadata = tool_data.get("metadata", {})
            
            # Re-identify if currently unknown to pick up new mappings immediately
            if not metadata.get("tool_source_name"):
                identification = self.identifier.identify("opentelemetry", metadata)
                if identification:
                    metadata.update({
                        "tool_source_name": identification.get("source_name"),
                        "tool_source_id": identification.get("source_id")
                    })

            tools.append({
                "id": tool_name,  # Use name as ID
                "name": tool_name,
                "type": "llm_client",  # OpenCITE schema compliance
                "discovery_source": metadata.get("discovery_source", "opentelemetry"),
                "models": list(tool_data["models"]),
                "trace_count": len(tool_data["traces"]),
                "tool_source_name": metadata.get("tool_source_name"),
                "tool_source_id": metadata.get("tool_source_id"),
                "metadata": metadata,
            })
        return tools

    def _list_traces(self, **kwargs) -> List[Dict[str, Any]]:
        """List all traces."""
        trace_id = kwargs.get("trace_id")
        if trace_id:
            trace = self.traces.get(trace_id)
            return [trace] if trace else []
        return list(self.traces.values())

    def _list_models(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered models.

        Collects model names from three sources:
        1. ``discovered_tools[*]["models"]`` — models seen in tool-call spans
        2. ``model_providers`` — models seen in non-tool spans (e.g. agent LLM calls)
        3. ``discovered_agents[*]["models_used"]`` — models correlated to agents
        """
        model_usage = defaultdict(lambda: {"tools": set(), "usage_count": 0})

        # 1. Models referenced by tools
        for tool_name, tool_data in self.discovered_tools.items():
            for model in tool_data["models"]:
                model_usage[model]["tools"].add(tool_name)
                model_usage[model]["usage_count"] += len(
                    [t for t in tool_data["traces"] if t["model"] == model]
                )

        # 2. Models from model_providers (non-tool spans with gen_ai.system)
        for model_name in self.model_providers:
            if model_name not in model_usage:
                model_usage[model_name]  # creates default entry

        # 3. Models referenced by agents
        for agent_data in self.discovered_agents.values():
            for model_name in agent_data.get("models_used", set()):
                if model_name not in model_usage:
                    model_usage[model_name]  # creates default entry

        models = []
        for model_name, data in model_usage.items():
            # Use explicit gen_ai.system provider, fall back to model name prefix
            provider = (
                self.model_providers.get(model_name)
                or (model_name.split("/")[0] if "/" in model_name else "unknown")
            )

            # Use model_call_count (all spans) as usage_count; fall back to
            # tool-trace count for models only seen in tool spans.
            usage = self.model_call_count.get(model_name, 0) or data["usage_count"]

            token_data = self.model_token_usage.get(model_name, {})
            models.append({
                "name": model_name,
                "provider": provider,
                "tools": list(data["tools"]),
                "usage_count": usage,
                "total_input_tokens": token_data.get("input_tokens", 0),
                "total_output_tokens": token_data.get("output_tokens", 0),
            })

        return models

    def _list_agents(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered agents."""
        agents = []
        for agent_name, agent_data in self.discovered_agents.items():
            metadata = agent_data.get("metadata", {})

            # Re-identify if currently unknown to pick up new mappings immediately
            if not metadata.get("agent_source_name"):
                identification = self.identifier.identify("opentelemetry", metadata)
                if identification:
                    metadata.update({
                        "agent_source_name": identification.get("source_name"),
                        "agent_source_id": identification.get("source_id"),
                    })

            agents.append({
                "id": agent_name,
                "name": agent_name,
                "discovery_source": metadata.get("discovery_source", "opentelemetry"),
                "tools_used": list(agent_data["tools_used"]),
                "models_used": list(agent_data["models_used"]),
                "first_seen": agent_data["first_seen"],
                "last_seen": agent_data["last_seen"],
                "agent_source_name": metadata.get("agent_source_name"),
                "agent_source_id": metadata.get("agent_source_id"),
                "metadata": metadata,
            })
        return agents

    def _list_downstream_systems(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered downstream systems."""
        systems = []
        for system_id, system_data in self.discovered_downstream.items():
            systems.append({
                "id": system_id,
                "name": system_data["name"],
                "type": system_data["type"],
                "endpoint": system_data["endpoint"],
                "tools_connecting": list(system_data["tools_connecting"]),
                "first_seen": system_data["first_seen"],
                "last_seen": system_data["last_seen"],
                "metadata": system_data["metadata"],
            })
        return systems

    def list_lineage(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List lineage relationships."""
        with self._lock:
            if source_id:
                return [
                    rel for rel in self.lineage.values()
                    if rel["source_id"] == source_id or rel["target_id"] == source_id
                ]
            return list(self.lineage.values())

    def _detect_agent(
        self,
        trace_id: str,
        span_id: str,
        span_name: str,
        attributes: List[Dict[str, Any]],
        resource: Dict[str, Any],
        span_time: Optional[str] = None,
    ) -> Optional[str]:
        """
        Detect if a span represents an agent.

        Detection is based on explicit agent attributes only:
        - gen_ai.agent.name, agent_name, agent.name

        Returns:
            The detected agent name, or None if no agent was detected.
        """
        attr_dict = self._attrs_to_dict(attributes)
        resource_dict = self._attrs_to_dict(resource.get("attributes", []))
        merged = {**resource_dict, **attr_dict}
        now = span_time or datetime.utcnow().isoformat()

        # Only detect agents via explicit attributes
        agent_name = (
            merged.get("gen_ai.agent.name")
            or merged.get("agent_name")
            or merged.get("agent.name")
        )

        if not agent_name:
            return None

        agent = self.discovered_agents[agent_name]
        if not agent["first_seen"] or now < agent["first_seen"]:
            agent["first_seen"] = now
        if not agent["last_seen"] or now > agent["last_seen"]:
            agent["last_seen"] = now

        # Track tools and models used by this agent
        # Only use explicit tool attributes — do NOT fall back to service.name
        # (service.name is the service identity, not a tool; using it here
        # causes endpoint/model names like "databricks-claude-sonnet-4-5"
        # to appear in tools_used).
        is_tool_call = "gen_ai.tool.call.id" in merged
        tool_name = (
            merged.get("gen_ai.tool.name")
            or merged.get("tool.name")
        )
        if not tool_name and is_tool_call and span_name:
            tool_name = span_name

        model_name = (
            merged.get("gen_ai.request.model")
            or merged.get("gen_ai.response.model")
        )

        if tool_name:
            agent["tools_used"].add(tool_name)
            self._add_lineage(agent_name, "agent", tool_name, "tool", "uses")

        if model_name:
            agent["models_used"].add(model_name)
            self._add_lineage(agent_name, "agent", model_name, "model", "uses")

        # Perform agent identification
        identification = self.identifier.identify("opentelemetry", merged)
        if identification:
            agent["metadata"].update({
                "agent_source_name": identification.get("source_name"),
                "agent_source_id": identification.get("source_id"),
            })

        # Store identification attributes in metadata for mapping modal
        id_attrs = self.get_identification_attributes()
        for attr_key in id_attrs:
            if attr_key in merged:
                agent["metadata"][attr_key] = merged[attr_key]

        # Store additional useful attributes
        agent["metadata"].update({
            "last_seen": agent["last_seen"],
            "discovery_source": merged.get("opencite.discovery_source", "opentelemetry"),
        })
        for key in ("service.name", "service.version", "gen_ai.system",
                     "gen_ai.operation.name", "enduser.id",
                     "net.peer.ip", "http.url", "http.user_agent",
                     "http.status_code", "http.response.content_type"):
            if key in merged:
                agent["metadata"][key] = merged[key]

        # Store user.*, ai_gateway.*, and trace.metadata.* attributes
        for key, val in merged.items():
            if val is None:
                continue
            if key.startswith(("user.", "ai_gateway.", "trace.metadata.")):
                if isinstance(val, (dict, list)):
                    agent["metadata"][key] = json.dumps(val, default=str)
                else:
                    agent["metadata"][key] = val

        return agent_name

    def _correlate_agent_tools(
        self,
        span_agents: Dict[str, str],
        span_tools: Dict[str, str],
        span_parents: Dict[str, str],
    ):
        """
        Correlate agents with tools via parent-child span relationships.

        In OTEL traces, agent spans and tool spans are typically separate:
        the agent span is the parent and tool/LLM spans are children.
        This method walks up from each tool span to find an ancestor agent span
        and creates the agent->tool lineage relationship.
        """
        if not span_agents or not span_tools:
            return

        for tool_span_id, tool_name in span_tools.items():
            # Same-span: agent and tool detected on the same span
            if tool_span_id in span_agents:
                agent_name = span_agents[tool_span_id]
                if agent_name != tool_name:
                    self.discovered_agents[agent_name]["tools_used"].add(tool_name)
                    self._add_lineage(agent_name, "agent", tool_name, "tool", "uses")
                    logger.info(
                        f"Correlated agent '{agent_name}' -> tool '{tool_name}' via same span"
                    )
                continue

            # Walk up the parent chain from this tool span to find an agent
            current = tool_span_id
            visited = set()
            while current in span_parents and current not in visited:
                visited.add(current)
                parent = span_parents[current]
                if parent in span_agents:
                    agent_name = span_agents[parent]
                    if agent_name != tool_name:
                        self.discovered_agents[agent_name]["tools_used"].add(tool_name)
                        self._add_lineage(agent_name, "agent", tool_name, "tool", "uses")
                        logger.info(
                            f"Correlated agent '{agent_name}' -> tool '{tool_name}' via span hierarchy"
                        )
                    break
                current = parent

    def _correlate_agent_models(
        self,
        span_agents: Dict[str, str],
        span_models: Dict[str, str],
        span_parents: Dict[str, str],
    ):
        """
        Correlate agents with models via parent-child span relationships.

        In OTEL traces, agent spans and LLM call spans are typically separate:
        the agent span is the parent and the LLM call span (with gen_ai.request.model)
        is a child or grandchild. This method walks up from each model-bearing span
        to find an ancestor agent span and creates the agent->model lineage relationship.
        """
        if not span_agents or not span_models:
            return

        for model_span_id, model_name in span_models.items():
            # Same-span: agent and model on the same span — link directly
            if model_span_id in span_agents:
                agent_name = span_agents[model_span_id]
                self.discovered_agents[agent_name]["models_used"].add(model_name)
                self._add_lineage(agent_name, "agent", model_name, "model", "uses")
                continue

            # Walk up the parent chain from this model span to find an agent
            current = model_span_id
            visited = set()
            while current in span_parents and current not in visited:
                visited.add(current)
                parent = span_parents[current]
                if parent in span_agents:
                    agent_name = span_agents[parent]
                    self.discovered_agents[agent_name]["models_used"].add(model_name)
                    self._add_lineage(agent_name, "agent", model_name, "model", "uses")
                    logger.info(
                        f"Correlated agent '{agent_name}' -> model '{model_name}' via span hierarchy"
                    )
                    break
                current = parent

    def _detect_downstream_system(
        self,
        trace_id: str,
        span_id: str,
        span_name: str,
        attributes: List[Dict[str, Any]],
        resource: Dict[str, Any],
    ):
        """
        Detect downstream systems (databases, external APIs, message queues, etc.)
        from OTEL span attributes.
        """
        attr_dict = self._attrs_to_dict(attributes)
        resource_dict = self._attrs_to_dict(resource.get("attributes", []))
        merged = {**resource_dict, **attr_dict}
        now = datetime.utcnow().isoformat()

        system_name = None
        system_type = "unknown"
        endpoint = None

        # Database systems
        db_system = merged.get("db.system")
        if db_system:
            db_name = merged.get("db.name") or merged.get("db.namespace") or db_system
            system_name = f"{db_system}:{db_name}" if db_name != db_system else db_system
            system_type = "database"
            endpoint = merged.get("server.address") or merged.get("net.peer.name")

        # HTTP/API calls (only CLIENT spans to external services)
        if not system_name:
            span_kind = merged.get("span.kind") or attr_dict.get("span.kind")
            http_url = merged.get("http.url") or merged.get("url.full")
            server_addr = merged.get("server.address") or merged.get("net.peer.name")

            if http_url and server_addr:
                system_name = server_addr
                system_type = "api"
                endpoint = http_url

        # Messaging systems
        if not system_name:
            messaging_system = merged.get("messaging.system")
            if messaging_system:
                destination = merged.get("messaging.destination.name") or messaging_system
                system_name = f"{messaging_system}:{destination}"
                system_type = "messaging"

        # RPC systems
        if not system_name:
            rpc_system = merged.get("rpc.system")
            if rpc_system:
                rpc_service = merged.get("rpc.service") or rpc_system
                system_name = f"{rpc_system}:{rpc_service}"
                system_type = "rpc"

        if not system_name:
            return

        # Generate a stable ID
        system_id = system_name.lower().replace(" ", "_").replace(":", "_")

        system = self.discovered_downstream[system_id]
        system["name"] = system_name
        system["type"] = system_type
        if endpoint:
            system["endpoint"] = endpoint
        if not system["first_seen"]:
            system["first_seen"] = now
        system["last_seen"] = now

        # Track which tools connect to this system
        tool_name = (
            merged.get("service.name")
            or merged.get("tool.name")
            or resource_dict.get("service.name")
        )
        if tool_name:
            system["tools_connecting"].add(tool_name)
            self._add_lineage(tool_name, "tool", system_id, "downstream", "connects_to")

    def _extract_token_usage(self, attributes: List[Dict[str, Any]]):
        """Extract token usage statistics from span attributes."""
        attr_dict = self._attrs_to_dict(attributes)

        model_name = (
            attr_dict.get("gen_ai.request.model")
            or attr_dict.get("gen_ai.response.model")
        )
        if not model_name:
            return

        input_tokens = attr_dict.get("gen_ai.usage.input_tokens") or attr_dict.get("gen_ai.usage.prompt_tokens")
        output_tokens = attr_dict.get("gen_ai.usage.output_tokens") or attr_dict.get("gen_ai.usage.completion_tokens")

        if input_tokens:
            try:
                self.model_token_usage[model_name]["input_tokens"] += int(input_tokens)
            except (ValueError, TypeError):
                pass
        if output_tokens:
            try:
                self.model_token_usage[model_name]["output_tokens"] += int(output_tokens)
            except (ValueError, TypeError):
                pass

    def _add_lineage(self, source_id: str, source_type: str,
                     target_id: str, target_type: str,
                     relationship_type: str):
        """Add or update a lineage relationship."""
        key = f"{source_id}:{target_id}:{relationship_type}"
        now = datetime.utcnow().isoformat()

        if key in self.lineage:
            self.lineage[key]["weight"] += 1
            self.lineage[key]["last_seen"] = now
        else:
            self.lineage[key] = {
                "source_id": source_id,
                "source_type": source_type,
                "target_id": target_id,
                "target_type": target_type,
                "relationship_type": relationship_type,
                "weight": 1,
                "first_seen": now,
                "last_seen": now,
            }

    @staticmethod
    @staticmethod
    def _extract_otlp_value(value: Dict[str, Any]) -> Any:
        """Extract a Python value from an OTLP AnyValue dict."""
        if "stringValue" in value:
            return value["stringValue"]
        if "intValue" in value:
            return value["intValue"]
        if "boolValue" in value:
            return value["boolValue"]
        if "doubleValue" in value:
            return value["doubleValue"]
        if "arrayValue" in value:
            return [
                OpenTelemetryPlugin._extract_otlp_value(v)
                for v in value["arrayValue"].get("values", [])
            ]
        if "kvlistValue" in value:
            return {
                kv.get("key", ""): OpenTelemetryPlugin._extract_otlp_value(kv.get("value", {}))
                for kv in value["kvlistValue"].get("values", [])
            }
        if "bytesValue" in value:
            return value["bytesValue"]
        return None

    @staticmethod
    def _attrs_to_dict(attributes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert OTLP attributes list to a flat dictionary."""
        result = {}
        for attr in attributes:
            key = attr.get("key", "")
            value = attr.get("value", {})
            extracted = OpenTelemetryPlugin._extract_otlp_value(value)
            if extracted is not None:
                result[key] = extracted
        return result

    def _detect_mcp_usage(
        self,
        trace_id: str,
        span_id: str,
        span_name: str,
        attributes: List[Dict[str, Any]],
        resource: Dict[str, Any],
    ):
        """
        Detect if a span represents MCP tool or resource usage.

        Looks for:
        - mcp.* attributes
        - mcp_tool or mcp_resource span names
        - tools://... or resource://... URIs
        """
        # Convert attributes list to dict
        attr_dict = {}
        for attr in attributes:
            key = attr.get("key", "")
            value = attr.get("value", {})
            if "stringValue" in value:
                attr_dict[key] = value["stringValue"]
            elif "intValue" in value:
                attr_dict[key] = value["intValue"]
            elif "boolValue" in value:
                attr_dict[key] = value["boolValue"]

        # Check for MCP-specific attributes
        mcp_server = attr_dict.get("mcp.server.name") or attr_dict.get("mcp.server")
        mcp_tool = attr_dict.get("mcp.tool.name") or attr_dict.get("mcp.tool")
        mcp_resource = attr_dict.get("mcp.resource.uri") or attr_dict.get("mcp.resource")

        # Check span name for MCP patterns
        if not mcp_tool and ("mcp_tool" in span_name.lower() or "call_tool" in span_name.lower()):
            # Try to extract tool name from span name
            mcp_tool = span_name.split(":")[-1].strip() if ":" in span_name else None

        if not mcp_resource and ("mcp_resource" in span_name.lower() or "read_resource" in span_name.lower()):
            # Try to extract resource URI from attributes
            for key, val in attr_dict.items():
                if isinstance(val, str) and ("://" in val or val.startswith("resource/")):
                    mcp_resource = val
                    break

        # If we detected MCP server usage, register it
        if mcp_server:
            self._register_mcp_server(
                server_name=mcp_server,
                trace_id=trace_id,
                span_id=span_id,
                attributes=attr_dict,
            )

            server_id = self._generate_mcp_server_id(mcp_server)

            # Register tool if present
            if mcp_tool:
                # Get status from span
                status = attr_dict.get("mcp.tool.status", "success")
                if attr_dict.get("error") or attr_dict.get("exception"):
                    status = "error"

                self._register_mcp_tool(
                    server_id=server_id,
                    tool_name=mcp_tool,
                    trace_id=trace_id,
                    span_id=span_id,
                    status=status,
                )

            # Register resource if present
            if mcp_resource:
                resource_type = attr_dict.get("mcp.resource.type")
                mime_type = attr_dict.get("mcp.resource.mime_type")

                self._register_mcp_resource(
                    server_id=server_id,
                    resource_uri=mcp_resource,
                    trace_id=trace_id,
                    span_id=span_id,
                    resource_type=resource_type,
                    mime_type=mime_type,
                )

            logger.info(
                f"Registered MCP usage: server={mcp_server}, "
                f"tool={mcp_tool}, resource={mcp_resource}"
            )

    # =========================================================================
    # MCP registration and listing methods (absorbed from former MCP plugin)
    # =========================================================================

    def _generate_mcp_server_id(self, name: str) -> str:
        """Generate a unique server ID from name."""
        safe_name = re.sub(r"[^a-z0-9-]", "-", name.lower())
        return safe_name

    def _register_mcp_server(
        self,
        server_name: str,
        trace_id: str,
        span_id: str,
        attributes: Dict[str, Any],
    ):
        """Register an MCP server discovered from a trace."""
        server_id = self._generate_mcp_server_id(server_name)

        if server_id not in self.mcp_servers:
            self.mcp_servers[server_id] = {
                "id": server_id,
                "name": server_name,
                "discovery_source": "trace_analysis",
                "first_seen": datetime.utcnow().isoformat(),
                "traces": [],
                "metadata": {},
            }
            logger.info(f"Discovered MCP server from trace: {server_name}")

        server = self.mcp_servers[server_id]
        server["last_seen"] = datetime.utcnow().isoformat()

        if trace_id not in [t["trace_id"] for t in server["traces"]]:
            server["traces"].append({
                "trace_id": trace_id,
                "span_id": span_id,
                "timestamp": datetime.utcnow().isoformat(),
            })

        if "mcp.server.version" in attributes:
            server["metadata"]["version"] = attributes["mcp.server.version"]
        if "mcp.server.protocol_version" in attributes:
            server["metadata"]["protocol_version"] = attributes["mcp.server.protocol_version"]

        if "mcp.server.transport" in attributes:
            server["transport"] = attributes["mcp.server.transport"]
        elif "http.url" in attributes or "mcp.server.endpoint" in attributes:
            endpoint = attributes.get("mcp.server.endpoint") or attributes.get("http.url")
            server["transport"] = "http" if endpoint else "unknown"
            if endpoint:
                server["endpoint"] = endpoint
        else:
            server["transport"] = "stdio"

        if "mcp.server.endpoint" in attributes:
            server["endpoint"] = attributes["mcp.server.endpoint"]

    def _register_mcp_tool(
        self,
        server_id: str,
        tool_name: str,
        trace_id: str,
        span_id: str,
        tool_schema: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ):
        """Register an MCP tool discovered from a trace."""
        tool_id = f"{server_id}-{tool_name}"

        if tool_id not in self.mcp_tools:
            self.mcp_tools[tool_id] = {
                "id": tool_id,
                "name": tool_name,
                "server_id": server_id,
                "discovery_source": "trace_analysis",
                "first_used": datetime.utcnow().isoformat(),
                "usage": {
                    "call_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                },
                "traces": [],
            }

            if tool_schema:
                self.mcp_tools[tool_id]["schema"] = tool_schema
                if "description" in tool_schema:
                    self.mcp_tools[tool_id]["description"] = tool_schema["description"]

            logger.info(f"Discovered MCP tool: {tool_name} on server {server_id}")

        tool = self.mcp_tools[tool_id]
        tool["last_used"] = datetime.utcnow().isoformat()
        tool["usage"]["call_count"] += 1

        if status == "success":
            tool["usage"]["success_count"] += 1
        else:
            tool["usage"]["error_count"] += 1

        tool["traces"].append({
            "trace_id": trace_id,
            "span_id": span_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
        })

        self.mcp_usage_stats[server_id]["tool_calls"] += 1

    def _register_mcp_resource(
        self,
        server_id: str,
        resource_uri: str,
        trace_id: str,
        span_id: str,
        resource_type: Optional[str] = None,
        mime_type: Optional[str] = None,
    ):
        """Register an MCP resource discovered from a trace."""
        resource_id = f"{server_id}-{abs(hash(resource_uri)) % 10000}"

        if resource_id not in self.mcp_resources:
            self.mcp_resources[resource_id] = {
                "id": resource_id,
                "uri": resource_uri,
                "server_id": server_id,
                "discovery_source": "trace_analysis",
                "first_accessed": datetime.utcnow().isoformat(),
                "usage": {
                    "access_count": 0,
                },
                "traces": [],
            }

            if resource_type:
                self.mcp_resources[resource_id]["type"] = resource_type
            if mime_type:
                self.mcp_resources[resource_id]["mime_type"] = mime_type

            logger.info(f"Discovered MCP resource: {resource_uri} on server {server_id}")

        resource = self.mcp_resources[resource_id]
        resource["last_accessed"] = datetime.utcnow().isoformat()
        resource["usage"]["access_count"] += 1

        resource["traces"].append({
            "trace_id": trace_id,
            "span_id": span_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        self.mcp_usage_stats[server_id]["resource_accesses"] += 1

    def _list_mcp_servers(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered MCP servers."""
        servers = []

        for server_id, server_info in self.mcp_servers.items():
            server = {
                "id": server_info["id"],
                "name": server_info["name"],
                "discovery_source": server_info["discovery_source"],
                "transport": server_info.get("transport", "unknown"),
                "first_seen": server_info["first_seen"],
                "last_seen": server_info.get("last_seen"),
                "trace_count": len(server_info["traces"]),
            }

            if "endpoint" in server_info:
                server["endpoint"] = server_info["endpoint"]
            if server_info.get("metadata"):
                server["metadata"] = server_info["metadata"]

            server["tools_count"] = len([
                t for t in self.mcp_tools.values()
                if t.get("server_id") == server_id
            ])
            server["resources_count"] = len([
                r for r in self.mcp_resources.values()
                if r.get("server_id") == server_id
            ])

            if server_id in self.mcp_usage_stats:
                server["usage_stats"] = self.mcp_usage_stats[server_id]

            servers.append(server)

        return servers

    def _list_mcp_tools(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered MCP tools."""
        server_id = kwargs.get("server_id")

        tools = []
        for tool_id, tool_info in self.mcp_tools.items():
            if server_id and tool_info.get("server_id") != server_id:
                continue

            tool = {
                "id": tool_info["id"],
                "name": tool_info["name"],
                "server_id": tool_info["server_id"],
                "discovery_source": tool_info["discovery_source"],
                "first_used": tool_info["first_used"],
                "last_used": tool_info.get("last_used"),
                "usage": tool_info["usage"],
                "call_count": tool_info["usage"]["call_count"],
                "trace_count": len(tool_info["traces"]),
            }

            if "description" in tool_info:
                tool["description"] = tool_info["description"]
            if "schema" in tool_info:
                tool["schema"] = tool_info["schema"]

            tools.append(tool)

        return tools

    def _list_mcp_resources(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered MCP resources."""
        server_id = kwargs.get("server_id")

        resources = []
        for resource_id, resource_info in self.mcp_resources.items():
            if server_id and resource_info.get("server_id") != server_id:
                continue

            resource = {
                "id": resource_info["id"],
                "uri": resource_info["uri"],
                "server_id": resource_info["server_id"],
                "discovery_source": resource_info["discovery_source"],
                "first_accessed": resource_info["first_accessed"],
                "last_accessed": resource_info.get("last_accessed"),
                "usage": resource_info["usage"],
                "access_count": resource_info["usage"]["access_count"],
                "trace_count": len(resource_info["traces"]),
            }

            if "type" in resource_info:
                resource["type"] = resource_info["type"]
            if "mime_type" in resource_info:
                resource["mime_type"] = resource_info["mime_type"]

            resources.append(resource)

        return resources

