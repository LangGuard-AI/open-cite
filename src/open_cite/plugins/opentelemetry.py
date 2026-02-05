"""
OpenTelemetry discovery plugin for Open Cite.

This plugin receives OpenTelemetry traces via OTLP/HTTP protocol and discovers
tools that use models through OpenRouter.
"""

import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
import re

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)

# Configure regex patterns here to capture additional attributes
# Example: [r"^custom\..*", r"^app\.metadata\..*"]
DEFAULT_ATTRIBUTE_PATTERNS = []


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP Server that handles each request in a separate thread."""
    daemon_threads = True  # Don't wait for threads to finish on shutdown


class OTLPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OTLP protocol."""

    def do_POST(self):
        """Handle POST requests for trace ingestion."""
        # Accept both /v1/traces and /v1/traces/ (with trailing slash)
        if self.path == "/v1/traces" or self.path == "/v1/traces/":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Store reference to the plugin's trace store
            plugin = self.server.plugin

            try:
                # Parse OTLP JSON payload
                content_type = self.headers.get("Content-Type", "")

                if "application/json" in content_type:
                    data = json.loads(body.decode("utf-8"))

                    # Log raw OTLP content if DEBUG is enabled
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[OTLP] Raw trace data: {json.dumps(data, indent=2)}")

                    plugin._ingest_traces(data)

                    # Log successful trace ingestion
                    num_spans = sum(len(rs.get("scopeSpans", [])) for rs in data.get("resourceSpans", []))
                    logger.warning(f"[OTLP] Received trace with {num_spans} scope spans")

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success"}).encode())
                else:
                    # For protobuf support, we'd need the opentelemetry-proto package
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
        else:
            self.send_response(404)
            self.end_headers()

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

    def __init__(self, host: str = "localhost", port: int = 4318, mcp_plugin=None, attribute_patterns: List[str] = None):
        """
        Initialize the OpenTelemetry plugin.

        Args:
            host: Host to bind the OTLP receiver to
            port: Port to bind the OTLP receiver to (default: 4318, standard OTLP/HTTP)
            mcp_plugin: Optional MCP plugin instance for MCP discovery integration
            attribute_patterns: Optional list of regex patterns to match attributes to collect
        """
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.mcp_plugin = mcp_plugin
        
        # Use provided patterns or fallback to default configuration
        patterns = attribute_patterns if attribute_patterns is not None else DEFAULT_ATTRIBUTE_PATTERNS
        self.attribute_patterns = [re.compile(p) for p in patterns]

        # Trace storage: {trace_id: {resource_spans, scope_spans, etc.}}
        self.traces: Dict[str, Dict[str, Any]] = {}

        # Discovered tools: {tool_name: {models: set(), traces: list(), metadata: dict()}}
        self.discovered_tools: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"models": set(), "traces": [], "metadata": {}}
        )

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Identifier for tool source identification
        from ..identifier import ToolIdentifier
        self.identifier = ToolIdentifier()

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "opentelemetry"

    @property
    def supported_asset_types(self) -> Set[str]:
        """Asset types supported by this plugin."""
        return {"tool", "model"}

    def get_identification_attributes(self) -> List[str]:
        """Return a list of attribute keys used for tool identification."""
        return [
            "trace.metadata.openrouter.entity_id",
            "trace.metadata.openrouter.api_key_name",
            "trace.metadata.openrouter.creator_user_id"
        ]

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify the OTLP receiver is running.

        Returns:
            Dict with connection status
        """
        is_running = self.server_thread is not None and self.server_thread.is_alive()
        return {
            "success": is_running,
            "receiver_running": is_running,
            "host": self.host,
            "port": self.port,
            "endpoint": f"http://{self.host}:{self.port}/v1/traces",
            "traces_received": len(self.traces),
            "tools_discovered": len(self.discovered_tools),
        }

    def start_receiver(self):
        """Start the OTLP HTTP receiver in a background thread."""
        import time
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("OTLP receiver is already running")
            return

        # Event to signal when server is ready
        server_ready = threading.Event()

        def run_server():
            self.server = ThreadingHTTPServer((self.host, self.port), OTLPRequestHandler)
            self.server.plugin = self  # Pass plugin reference to handler
            logger.info(f"OTLP receiver started on {self.host}:{self.port}")
            server_ready.set()  # Signal that server is ready
            self.server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to actually start (timeout after 5 seconds)
        if not server_ready.wait(timeout=5.0):
            logger.error("OTLP receiver failed to start within 5 seconds")
            raise RuntimeError("OTLP receiver failed to start")

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

    def _ingest_traces(self, otlp_data: Dict[str, Any]):
        """
        Ingest OTLP trace data and extract tool/model information.

        Args:
            otlp_data: OTLP JSON payload
        """
        with self._lock:
            try:
                # OTLP format: {"resourceSpans": [...]}
                resource_spans = otlp_data.get("resourceSpans", [])

                for resource_span in resource_spans:
                    resource = resource_span.get("resource", {})
                    scope_spans = resource_span.get("scopeSpans", [])

                    for scope_span in scope_spans:
                        spans = scope_span.get("spans", [])

                        for span in spans:
                            trace_id = span.get("traceId", "")
                            span_id = span.get("spanId", "")
                            span_name = span.get("name", "")
                            attributes = span.get("attributes", [])

                            # Store the trace
                            if trace_id not in self.traces:
                                self.traces[trace_id] = {
                                    "trace_id": trace_id,
                                    "spans": [],
                                    "first_seen": datetime.utcnow().isoformat(),
                                }

                            self.traces[trace_id]["spans"].append({
                                "span_id": span_id,
                                "span_name": span_name,
                                "attributes": attributes,
                                "resource": resource,
                            })

                            # Detect OpenRouter usage
                            self._detect_openrouter_usage(
                                trace_id, span_id, span_name, attributes, resource
                            )

                            # Detect MCP tool/resource usage
                            self._detect_mcp_usage(
                                trace_id, span_id, span_name, attributes, resource
                            )

                logger.info(f"Ingested traces. Total traces: {len(self.traces)}")
            except Exception as e:
                logger.error(f"Error ingesting traces: {e}")

    def _detect_openrouter_usage(
        self,
        trace_id: str,
        span_id: str,
        span_name: str,
        attributes: List[Dict[str, Any]],
        resource: Dict[str, Any],
    ):
        """
        Detect if a span represents LLM/AI model usage.

        Looks for GenAI semantic convention attributes:
        - gen_ai.request.model or gen_ai.response.model
        - llm.model or model attributes
        - service.name for tool identification
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
        tool_name = (
            merged_attrs.get("service.name")
            or merged_attrs.get("tool.name")
            or merged_attrs.get("app.name")
        )

        # If no explicit service name, use span name or infer from trace
        if not tool_name:
            tool_name = span_name or f"tool_{trace_id[:8]}"

        # Debug logging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Detection] trace_id={trace_id[:8]}... model_name={model_name}, "
                        f"tool_name={tool_name}")

        # If we detected model usage, record it (regardless of provider)
        if model_name:
            if model_name not in self.discovered_tools[tool_name]["models"]:
                self.discovered_tools[tool_name]["models"].add(model_name)
                logger.info(f"Discovered tool '{tool_name}' using model '{model_name}'")

            self.discovered_tools[tool_name]["traces"].append({
                "trace_id": trace_id,
                "span_id": span_id,
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name,
            })

            # Store metadata
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

            # Always store identification attributes in metadata so they are visible in GUI mapping modal
            id_attrs = self.get_identification_attributes()
            for attr_key in id_attrs:
                if attr_key in merged_attrs:
                    self.discovered_tools[tool_name]["metadata"][attr_key] = merged_attrs[attr_key]

            self.discovered_tools[tool_name]["metadata"].update({
                "last_seen": datetime.utcnow().isoformat(),
                "url": url,
                "provider": provider,
                "source": source,
                "discovery_source": "opentelemetry"
            })
            
            # Check for attributes matching configured patterns
            if self.attribute_patterns:
                for key, value in merged_attrs.items():
                    # Skip if already in metadata
                    if key in self.discovered_tools[tool_name]["metadata"]:
                        continue
                        
                    for pattern in self.attribute_patterns:
                        if pattern.match(key):
                            self.discovered_tools[tool_name]["metadata"][key] = value
                            break

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List assets discovered from OpenTelemetry traces.

        Supported asset types:
        - "tool": Tools using OpenRouter models
        - "trace": Raw traces received
        - "model": Models discovered across all tools

        Args:
            asset_type: Type of asset to list
            **kwargs: Additional filters (e.g., tool_name, model_name)

        Returns:
            List of assets
        """
        with self._lock:
            if asset_type == "tool":
                return self._list_tools(**kwargs)
            elif asset_type == "model":
                return self._list_models(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: tool, model"
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
                "discovery_source": "opentelemetry",  # OpenCITE schema compliance
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
        """List all discovered models across tools."""
        model_usage = defaultdict(lambda: {"tools": set(), "usage_count": 0})

        for tool_name, tool_data in self.discovered_tools.items():
            for model in tool_data["models"]:
                model_usage[model]["tools"].add(tool_name)
                model_usage[model]["usage_count"] += len(
                    [t for t in tool_data["traces"] if t["model"] == model]
                )

        models = []
        for model_name, data in model_usage.items():
            # Extract provider from model name (e.g., "openai/gpt-4" -> "openai")
            provider = model_name.split("/")[0] if "/" in model_name else "unknown"

            models.append({
                "name": model_name,
                "provider": provider,
                "tools": list(data["tools"]),
                "usage_count": data["usage_count"],
            })

        return models

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
        if not self.mcp_plugin:
            return  # MCP plugin not available

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
            self.mcp_plugin.register_server_from_trace(
                server_name=mcp_server,
                trace_id=trace_id,
                span_id=span_id,
                attributes=attr_dict,
            )

            server_id = self.mcp_plugin._generate_server_id(mcp_server)

            # Register tool if present
            if mcp_tool:
                # Get status from span
                status = attr_dict.get("mcp.tool.status", "success")
                if attr_dict.get("error") or attr_dict.get("exception"):
                    status = "error"

                self.mcp_plugin.register_tool(
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

                self.mcp_plugin.register_resource(
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

