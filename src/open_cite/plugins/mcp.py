"""
MCP (Model Context Protocol) discovery plugin for Open Cite.

This plugin discovers MCP servers through OpenTelemetry trace analysis,
detecting which MCP servers, tools, and resources are actually being used
by AI applications in production.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from collections import defaultdict

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)


class MCPPlugin(BaseDiscoveryPlugin):
    """
    MCP server discovery plugin.

    Discovers MCP servers, tools, and resources from OpenTelemetry traces,
    providing visibility into actual MCP usage in production.
    """

    def __init__(
        self,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the MCP plugin.

        Args:
            instance_id: Unique identifier for this plugin instance
            display_name: Human-readable name for this instance
        """
        super().__init__(instance_id=instance_id, display_name=display_name)
        # Storage for discovered entities
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}
        self.mcp_resources: Dict[str, Dict[str, Any]] = {}

        # Track usage statistics
        self.usage_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"tool_calls": 0, "resource_accesses": 0}
        )

        # Lock for thread-safe operations
        import threading
        self._lock = threading.Lock()

    @property
    def plugin_type(self) -> str:
        """Type identifier for this plugin."""
        return "mcp"

    def get_config(self) -> Dict[str, Any]:
        """Return plugin configuration."""
        return {}

    @property
    def supported_asset_types(self) -> Set[str]:
        """Asset types supported by this plugin."""
        return {"mcp_server", "mcp_tool", "mcp_resource"}

    def get_identification_attributes(self) -> List[str]:
        return ["mcp.server.name", "mcp.server.endpoint"]

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify MCP discovery status.

        Returns:
            Dict with discovery status
        """
        with self._lock:
            return {
                "success": True,
                "discovery_method": "trace_analysis",
                "servers_discovered": len(self.mcp_servers),
                "tools_discovered": len(self.mcp_tools),
                "resources_discovered": len(self.mcp_resources),
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List MCP assets discovered from traces.

        Supported asset types:
        - "mcp_server": MCP servers observed in traces
        - "mcp_tool": MCP tools called in traces
        - "mcp_resource": MCP resources accessed in traces

        Args:
            asset_type: Type of asset to list
            **kwargs: Additional filters (e.g., server_id)

        Returns:
            List of assets
        """
        with self._lock:
            if asset_type == "mcp_server":
                return self._list_mcp_servers(**kwargs)
            elif asset_type == "mcp_tool":
                return self._list_mcp_tools(**kwargs)
            elif asset_type == "mcp_resource":
                return self._list_mcp_resources(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: mcp_server, mcp_tool, mcp_resource"
                )

    def register_server_from_trace(
        self,
        server_name: str,
        trace_id: str,
        span_id: str,
        attributes: Dict[str, Any],
    ):
        """
        Register an MCP server discovered from a trace.

        Args:
            server_name: Name of the MCP server
            trace_id: Trace ID where server was observed
            span_id: Span ID where server was observed
            attributes: Span attributes containing server metadata
        """
        with self._lock:
            server_id = self._generate_server_id(server_name)

            if server_id not in self.mcp_servers:
                self.mcp_servers[server_id] = {
                    "id": server_id,
                    "name": server_name,
                    "discovery_source": "trace_analysis",  # Discovered via trace analysis
                    "first_seen": datetime.utcnow().isoformat(),
                    "traces": [],
                    "metadata": {},
                }

                logger.info(f"Discovered MCP server from trace: {server_name}")

            # Update server info
            server = self.mcp_servers[server_id]
            server["last_seen"] = datetime.utcnow().isoformat()

            # Track trace
            if trace_id not in [t["trace_id"] for t in server["traces"]]:
                server["traces"].append({
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            # Extract metadata from attributes
            if "mcp.server.version" in attributes:
                server["metadata"]["version"] = attributes["mcp.server.version"]

            if "mcp.server.protocol_version" in attributes:
                server["metadata"]["protocol_version"] = attributes["mcp.server.protocol_version"]

            # Detect transport from attributes
            if "mcp.server.transport" in attributes:
                # Explicit transport attribute from trace
                server["transport"] = attributes["mcp.server.transport"]
            elif "http.url" in attributes or "mcp.server.endpoint" in attributes:
                # Infer from endpoint
                endpoint = attributes.get("mcp.server.endpoint") or attributes.get("http.url")
                server["transport"] = "http" if endpoint else "unknown"
                if endpoint:
                    server["endpoint"] = endpoint
            else:
                server["transport"] = "stdio"  # Default for MCP

            # Also check for endpoint attribute
            if "mcp.server.endpoint" in attributes:
                server["endpoint"] = attributes["mcp.server.endpoint"]

    def register_tool(
        self,
        server_id: str,
        tool_name: str,
        trace_id: str,
        span_id: str,
        tool_schema: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ):
        """
        Register an MCP tool discovered from a trace.

        Args:
            server_id: MCP server ID
            tool_name: Name of the tool
            trace_id: Trace ID where tool was called
            span_id: Span ID where tool was called
            tool_schema: Optional JSON schema for the tool
            status: Call status (success/error)
        """
        with self._lock:
            tool_id = f"{server_id}-{tool_name}"

            if tool_id not in self.mcp_tools:
                self.mcp_tools[tool_id] = {
                    "id": tool_id,
                    "name": tool_name,
                    "server_id": server_id,
                    "discovery_source": "trace_analysis",  # Discovered via trace analysis
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

            # Update tool usage
            tool = self.mcp_tools[tool_id]
            tool["last_used"] = datetime.utcnow().isoformat()
            tool["usage"]["call_count"] += 1

            if status == "success":
                tool["usage"]["success_count"] += 1
            else:
                tool["usage"]["error_count"] += 1

            # Track trace
            tool["traces"].append({
                "trace_id": trace_id,
                "span_id": span_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": status,
            })

            # Update server stats
            self.usage_stats[server_id]["tool_calls"] += 1

    def register_resource(
        self,
        server_id: str,
        resource_uri: str,
        trace_id: str,
        span_id: str,
        resource_type: Optional[str] = None,
        mime_type: Optional[str] = None,
    ):
        """
        Register an MCP resource discovered from a trace.

        Args:
            server_id: MCP server ID
            resource_uri: URI of the resource
            trace_id: Trace ID where resource was accessed
            span_id: Span ID where resource was accessed
            resource_type: Optional type of resource
            mime_type: Optional MIME type
        """
        with self._lock:
            resource_id = f"{server_id}-{abs(hash(resource_uri)) % 10000}"

            if resource_id not in self.mcp_resources:
                self.mcp_resources[resource_id] = {
                    "id": resource_id,
                    "uri": resource_uri,
                    "server_id": server_id,
                    "discovery_source": "trace_analysis",  # Discovered via trace analysis
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

            # Update resource usage
            resource = self.mcp_resources[resource_id]
            resource["last_accessed"] = datetime.utcnow().isoformat()
            resource["usage"]["access_count"] += 1

            # Track trace
            resource["traces"].append({
                "trace_id": trace_id,
                "span_id": span_id,
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Update server stats
            self.usage_stats[server_id]["resource_accesses"] += 1

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

            # Add tool/resource counts
            server["tools_count"] = len([
                t for t in self.mcp_tools.values()
                if t.get("server_id") == server_id
            ])
            server["resources_count"] = len([
                r for r in self.mcp_resources.values()
                if r.get("server_id") == server_id
            ])

            # Add usage stats
            if server_id in self.usage_stats:
                server["usage_stats"] = self.usage_stats[server_id]

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
                "call_count": tool_info["usage"]["call_count"],  # Top-level for compatibility
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
                "access_count": resource_info["usage"]["access_count"],  # Top-level for compatibility
                "trace_count": len(resource_info["traces"]),
            }

            if "type" in resource_info:
                resource["type"] = resource_info["type"]

            if "mime_type" in resource_info:
                resource["mime_type"] = resource_info["mime_type"]

            resources.append(resource)

        return resources

    def _generate_server_id(self, name: str) -> str:
        """Generate a unique server ID from name."""
        import re
        # Sanitize name
        safe_name = re.sub(r"[^a-z0-9-]", "-", name.lower())
        return safe_name

    def clear(self):
        """Clear all discovered data."""
        with self._lock:
            self.mcp_servers.clear()
            self.mcp_tools.clear()
            self.mcp_resources.clear()
            self.usage_stats.clear()
            logger.info("Cleared all MCP discovery data")
