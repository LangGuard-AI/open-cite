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

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4318,
        mcp_plugin=None,
        attribute_patterns: List[str] = None,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the OpenTelemetry plugin.

        Args:
            host: Host to bind the OTLP receiver to
            port: Port to bind the OTLP receiver to (default: 4318, standard OTLP/HTTP)
            mcp_plugin: Optional MCP plugin instance for MCP discovery integration
            attribute_patterns: Optional list of regex patterns to match attributes to collect
            instance_id: Unique identifier for this plugin instance
            display_name: Human-readable name for this instance
        """
        super().__init__(instance_id=instance_id, display_name=display_name)
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

        # Discovered agents: {agent_name: {tools_used: set(), models_used: set(), ...}}
        self.discovered_agents: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "tools_used": set(),
                "models_used": set(),
                "confidence": "low",
                "first_seen": None,
                "last_seen": None,
                "metadata": {},
            }
        )

        # Discovered downstream systems: {system_id: {...}}
        self.discovered_downstream: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "name": "",
                "type": "unknown",
                "endpoint": None,
                "tools_connecting": set(),
                "first_seen": None,
                "last_seen": None,
                "metadata": {},
            }
        )

        # Lineage relationships: list of (source_id, source_type, target_id, target_type, rel_type)
        self.lineage: Dict[str, Dict[str, Any]] = {}

        # Token usage tracking per model: {model_name: {input_tokens, output_tokens}}
        self.model_token_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0}
        )

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Identifier for tool source identification
        from ..identifier import ToolIdentifier
        self.identifier = ToolIdentifier()

    @property
    def plugin_type(self) -> str:
        """Type identifier for this plugin."""
        return "opentelemetry"

    def get_config(self) -> Dict[str, Any]:
        """Return plugin configuration."""
        return {
            "host": self.host,
            "port": self.port,
        }

    @property
    def supported_asset_types(self) -> Set[str]:
        """Asset types supported by this plugin."""
        return {"tool", "model", "agent", "downstream_system"}

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

                # Track per-span detections for parent-child correlation
                span_agents: Dict[str, str] = {}   # span_id -> agent_name
                span_tools: Dict[str, str] = {}     # span_id -> tool_name
                span_parents: Dict[str, str] = {}   # span_id -> parent_span_id

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

                            if parent_span_id:
                                span_parents[span_id] = parent_span_id

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

                            # Detect OpenRouter usage (tools/models)
                            tool_name = self._detect_openrouter_usage(
                                trace_id, span_id, span_name, attributes, resource
                            )
                            if tool_name:
                                span_tools[span_id] = tool_name

                            # Detect agents
                            agent_name = self._detect_agent(
                                trace_id, span_id, span_name, attributes, resource
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

                # Correlate agents with tools via parent-child span relationships
                self._correlate_agent_tools(span_agents, span_tools, span_parents)

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
    ) -> Optional[str]:
        """
        Detect if a span represents LLM/AI model usage.

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

        tool_name = (
            attr_dict.get("gen_ai.tool.name")
            or attr_dict.get("tool.name")
        )

        # For tool-call spans, prefer the span name (e.g. "CodeExecution")
        if not tool_name and is_tool_call and span_name:
            tool_name = span_name

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

        # If we detected model usage, record it (regardless of provider)
        if not model_name:
            return tool_name if tool_name else None

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

            return tool_name
        return None

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
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: tool, model, agent, downstream_system"
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

            token_data = self.model_token_usage.get(model_name, {})
            models.append({
                "name": model_name,
                "provider": provider,
                "tools": list(data["tools"]),
                "usage_count": data["usage_count"],
                "total_input_tokens": token_data.get("input_tokens", 0),
                "total_output_tokens": token_data.get("output_tokens", 0),
            })

        return models

    def _list_agents(self, **kwargs) -> List[Dict[str, Any]]:
        """List all discovered agents."""
        agents = []
        for agent_name, agent_data in self.discovered_agents.items():
            agents.append({
                "id": agent_name,
                "name": agent_name,
                "confidence": agent_data["confidence"],
                "tools_used": list(agent_data["tools_used"]),
                "models_used": list(agent_data["models_used"]),
                "first_seen": agent_data["first_seen"],
                "last_seen": agent_data["last_seen"],
                "metadata": agent_data["metadata"],
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

    # Agent detection patterns
    AGENT_SPAN_PATTERNS = re.compile(
        r'(agent|bot|assistant|copilot|workflow|pipeline|orchestrator)',
        re.IGNORECASE
    )

    def _detect_agent(
        self,
        trace_id: str,
        span_id: str,
        span_name: str,
        attributes: List[Dict[str, Any]],
        resource: Dict[str, Any],
    ) -> Optional[str]:
        """
        Detect if a span represents an agent.

        Detection strategy:
        - Explicit agent attributes (gen_ai.agent.name, agent_name) -> high confidence
        - Span name pattern matching (agent, bot, assistant, etc.) -> medium confidence
        - Fall back to service.name when span structure indicates agentic behavior -> low confidence

        Returns:
            The detected agent name, or None if no agent was detected.
        """
        attr_dict = self._attrs_to_dict(attributes)
        resource_dict = self._attrs_to_dict(resource.get("attributes", []))
        merged = {**resource_dict, **attr_dict}
        now = datetime.utcnow().isoformat()

        # Check for explicit agent attributes (high confidence)
        agent_name = (
            merged.get("gen_ai.agent.name")
            or merged.get("agent_name")
            or merged.get("agent.name")
        )
        confidence = "high" if agent_name else None

        # Check span name patterns (medium confidence)
        if not agent_name and self.AGENT_SPAN_PATTERNS.search(span_name):
            agent_name = span_name
            confidence = "medium"

        if not agent_name:
            return None

        agent = self.discovered_agents[agent_name]
        if not agent["first_seen"]:
            agent["first_seen"] = now
        agent["last_seen"] = now
        agent["confidence"] = confidence

        # Track tools and models used by this agent
        # Prefer explicit tool attributes; fall back to service.name only if distinct from agent
        is_tool_call = "gen_ai.tool.call.id" in merged
        tool_name = (
            merged.get("gen_ai.tool.name")
            or merged.get("tool.name")
        )
        if not tool_name and is_tool_call and span_name:
            tool_name = span_name
        if not tool_name:
            svc = merged.get("service.name")
            if svc and svc != agent_name:
                tool_name = svc

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
    def _attrs_to_dict(attributes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert OTLP attributes list to a flat dictionary."""
        result = {}
        for attr in attributes:
            key = attr.get("key", "")
            value = attr.get("value", {})
            if "stringValue" in value:
                result[key] = value["stringValue"]
            elif "intValue" in value:
                result[key] = value["intValue"]
            elif "boolValue" in value:
                result[key] = value["boolValue"]
            elif "doubleValue" in value:
                result[key] = value["doubleValue"]
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

