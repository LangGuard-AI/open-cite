import logging
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Type, Set
from .core import BaseDiscoveryPlugin
from .plugins.databricks import DatabricksPlugin
from .plugins.opentelemetry import OpenTelemetryPlugin
from .plugins.mcp import MCPPlugin
from .plugins.google_cloud import GoogleCloudPlugin
from .plugins.zscaler import ZscalerPlugin
from .http_client import get_http_client, OpenCiteHttpClient
from .schema import (
    OpenCiteExporter,
    ToolFormatter,
    ModelFormatter,
    DataAssetFormatter,
    MCPServerFormatter,
    MCPToolFormatter,
    MCPResourceFormatter,
    GoogleCloudModelFormatter,
    GoogleCloudEndpointFormatter,
    GoogleCloudDeploymentFormatter,
    GoogleCloudGenerativeModelFormatter,
    parse_model_id,
)

logger = logging.getLogger(__name__)


def _deprecated(message: str):
    """Emit a deprecation warning."""
    warnings.warn(message, DeprecationWarning, stacklevel=3)

class OpenCiteClient:
    """
    Client for Open Cite. Manages discovery plugins.
    """

    def __init__(
        self,
        enable_otel: bool = False,
        otel_host: str = "localhost",
        otel_port: int = 4318,
        enable_mcp: bool = True,
        enable_google_cloud: bool = False,
        gcp_project_id: Optional[str] = None,
        gcp_location: str = "us-central1",
        gcp_credentials: Optional[Any] = None,
    ):
        """
        Initialize the OpenCite client.

        Args:
            enable_otel: If True, register and start the OpenTelemetry plugin
            otel_host: Host for the OTLP receiver (default: localhost)
            otel_port: Port for the OTLP receiver (default: 4318)
            enable_mcp: If True, register the MCP discovery plugin (default: True)
            enable_google_cloud: If True, register the Google Cloud plugin
            gcp_project_id: Google Cloud project ID (if None, uses default)
            gcp_location: Google Cloud location (default: us-central1)
            gcp_credentials: Google Cloud credentials (if None, uses default)
        """
        self.plugins: Dict[str, BaseDiscoveryPlugin] = {}
        
        # Initialize central HTTP client
        self.http_client = get_http_client()

        # Auto-register default plugins
        # In a real plugin system, this might use entry points
        try:
            # Pass central http_client
            self.register_plugin(DatabricksPlugin(http_client=self.http_client))
        except Exception as e:
            logger.warning(f"Failed to auto-register Databricks plugin: {e}")

        # Register MCP plugin first (if enabled)
        mcp_plugin = None
        if enable_mcp:
            try:
                mcp_plugin = MCPPlugin()
                self.register_plugin(mcp_plugin)
                logger.info(f"MCP plugin registered (trace-based discovery)")
            except Exception as e:
                logger.warning(f"Failed to auto-register MCP plugin: {e}")

        # Register OpenTelemetry plugin with MCP integration
        if enable_otel:
            try:
                otel_plugin = OpenTelemetryPlugin(
                    host=otel_host,
                    port=otel_port,
                    mcp_plugin=mcp_plugin  # Pass MCP plugin for integration
                )
                self.register_plugin(otel_plugin)
                otel_plugin.start_receiver()
                logger.info(f"OpenTelemetry plugin started on {otel_host}:{otel_port}")
            except Exception as e:
                logger.warning(f"Failed to auto-register OpenTelemetry plugin: {e}")

        # Register Google Cloud plugin (if enabled)
        if enable_google_cloud:
            try:
                gcp_plugin = GoogleCloudPlugin(
                    project_id=gcp_project_id,
                    location=gcp_location,
                    credentials=gcp_credentials,
                )
                self.register_plugin(gcp_plugin)
                logger.info(f"Google Cloud plugin registered for project {gcp_project_id or 'default'}")
            except Exception as e:
                logger.warning(f"Failed to auto-register Google Cloud plugin: {e}")

        # Register Zscaler plugin (default auto-register if env vars present)
        try:
            # Pass http_client dependency injection
            zscaler_plugin = ZscalerPlugin(http_client=self.http_client)
            self.register_plugin(zscaler_plugin)
            logger.debug("Zscaler plugin registered")
            
            # Auto-start NSS receiver if port is configured
            import os
            nss_port = os.getenv("ZSCALER_NSS_PORT")
            if nss_port:
                try:
                    port_int = int(nss_port)
                    zscaler_plugin.start_nss_receiver(port=port_int)
                except ValueError:
                    logger.warning(f"Invalid ZSCALER_NSS_PORT: {nss_port}")
                    
        except Exception as e:
            logger.debug(f"Zscaler plugin skipped: {e}")

    def register_plugin(self, plugin: BaseDiscoveryPlugin):
        """
        Register a discovery plugin.
        """
        self.plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name}")

    def get_plugin(self, name: str) -> BaseDiscoveryPlugin:
        """
        Get a registered plugin by name.
        """
        if name not in self.plugins:
            raise ValueError(f"Plugin '{name}' not found. Available plugins: {list(self.plugins.keys())}")
        return self.plugins[name]

    def get_plugins_for_asset_type(self, asset_type: str) -> List[BaseDiscoveryPlugin]:
        """
        Get all plugins that support a given asset type.

        Args:
            asset_type: The asset type to query for

        Returns:
            List of plugins supporting that asset type
        """
        return [
            plugin for plugin in self.plugins.values()
            if asset_type in plugin.supported_asset_types
        ]

    def get_all_supported_asset_types(self) -> Set[str]:
        """
        Get all asset types supported across all registered plugins.

        Returns:
            Set of all supported asset types
        """
        all_types: Set[str] = set()
        for plugin in self.plugins.values():
            all_types.update(plugin.supported_asset_types)
        return all_types

    # =========================================================================
    # Generic Asset Listing Methods
    # These aggregate results from all registered plugins that support each type
    # =========================================================================

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List assets of a specific type from all registered plugins.

        This is the primary method for querying assets across all plugins.
        It aggregates results from every plugin that supports the given asset type.

        Args:
            asset_type: Type of asset to list (e.g., 'tool', 'model', 'mcp_server')
            **kwargs: Additional filters passed to each plugin

        Returns:
            Aggregated list of assets from all supporting plugins

        Example:
            # List all tools from all plugins
            tools = client.list_assets("tool")

            # List all MCP servers from all plugins
            servers = client.list_assets("mcp_server")

            # List with filters (passed to plugins)
            tools = client.list_assets("mcp_tool", server_id="my-server")
        """
        results = []
        plugins = self.get_plugins_for_asset_type(asset_type)

        for plugin in plugins:
            try:
                assets = plugin.list_assets(asset_type, **kwargs)
                # Ensure each asset has discovery_source set to the plugin name
                for asset in assets:
                    if "discovery_source" not in asset:
                        asset["discovery_source"] = plugin.name
                results.extend(assets)
            except Exception as e:
                logger.warning(f"Failed to list {asset_type} from plugin {plugin.name}: {e}")

        return results

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all discovered tools from all plugins.

        Returns:
            Aggregated list of tools
        """
        return self.list_assets("tool")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all discovered models from all plugins.

        Note: This includes both AI models discovered via traces (OpenTelemetry)
        and Vertex AI models (Google Cloud). Use the discovery_source field
        to distinguish between sources.

        Returns:
            Aggregated list of models
        """
        return self.list_assets("model")

    def list_mcp_servers(self) -> List[Dict[str, Any]]:
        """
        List all discovered MCP servers from all plugins.

        This aggregates MCP servers from:
        - MCP plugin (trace-based discovery)
        - Google Cloud plugin (Compute Engine instance labels)

        Returns:
            Aggregated list of MCP servers
        """
        return self.list_assets("mcp_server")

    def list_mcp_tools(self, server_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all discovered MCP tools from all plugins.

        Args:
            server_id: Optional server ID to filter by

        Returns:
            Aggregated list of MCP tools
        """
        return self.list_assets("mcp_tool", server_id=server_id)

    def list_mcp_resources(self, server_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all discovered MCP resources from all plugins.

        Args:
            server_id: Optional server ID to filter by

        Returns:
            Aggregated list of MCP resources
        """
        return self.list_assets("mcp_resource", server_id=server_id)

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all discovered endpoints from all plugins.

        Returns:
            Aggregated list of endpoints (e.g., Vertex AI endpoints)
        """
        return self.list_assets("endpoint")

    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all discovered deployments from all plugins.

        Returns:
            Aggregated list of deployments
        """
        return self.list_assets("deployment")

    def list_generative_models(self) -> List[Dict[str, Any]]:
        """
        List all discovered generative AI models from all plugins.

        Returns:
            Aggregated list of generative models
        """
        return self.list_assets("generative_model")

    def list_catalogs(self) -> List[Dict[str, Any]]:
        """
        List all discovered catalogs from all plugins.

        Returns:
            Aggregated list of catalogs
        """
        return self.list_assets("catalog")

    def list_schemas(self, catalog_name: str) -> List[Dict[str, Any]]:
        """
        List all discovered schemas from all plugins.

        Args:
            catalog_name: Catalog name to filter by

        Returns:
            Aggregated list of schemas
        """
        return self.list_assets("schema", catalog_name=catalog_name)

    def list_tables(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        """
        List all discovered tables from all plugins.

        Args:
            catalog_name: Catalog name to filter by
            schema_name: Schema name to filter by

        Returns:
            Aggregated list of tables
        """
        return self.list_assets("table", catalog_name=catalog_name, schema_name=schema_name)

    def list_volumes(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        """
        List all discovered volumes from all plugins.

        Args:
            catalog_name: Catalog name to filter by
            schema_name: Schema name to filter by

        Returns:
            Aggregated list of volumes
        """
        return self.list_assets("volume", catalog_name=catalog_name, schema_name=schema_name)

    def list_functions(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        """
        List all discovered functions from all plugins.

        Args:
            catalog_name: Catalog name to filter by
            schema_name: Schema name to filter by

        Returns:
            Aggregated list of functions
        """
        return self.list_assets("function", catalog_name=catalog_name, schema_name=schema_name)

    # =========================================================================
    # Connection Verification
    # =========================================================================

    def verify_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        Verify connections for all registered plugins.

        Returns:
            Dict mapping plugin names to their connection status
        """
        results = {}
        for name, plugin in self.plugins.items():
            try:
                results[name] = plugin.verify_connection()
            except Exception as e:
                results[name] = {"success": False, "error": str(e)}
        return results

    # =========================================================================
    # Plugin-specific helper methods (for operations not covered by list_assets)
    # =========================================================================

    @property
    def _opentelemetry(self) -> OpenTelemetryPlugin:
        """Get the OpenTelemetry plugin."""
        return self.get_plugin("opentelemetry")  # type: ignore

    def start_otel_receiver(self):
        """Start the OpenTelemetry receiver."""
        self._opentelemetry.start_receiver()

    def stop_otel_receiver(self):
        """Stop the OpenTelemetry receiver."""
        self._opentelemetry.stop_receiver()

    @property
    def _google_cloud(self) -> GoogleCloudPlugin:
        """Get the Google Cloud plugin."""
        return self.get_plugin("google_cloud")  # type: ignore

    def scan_gcp_mcp_servers(
        self,
        zones: Optional[List[str]] = None,
        ports: Optional[List[int]] = None,
        min_ports_open: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Discover MCP servers by scanning open ports on GCP Compute Engine instances.

        This method scans instances for open ports that might indicate MCP servers.
        It complements label-based discovery by finding servers that may not be
        properly labeled.

        Default ports scanned: 3000, 3001, 3002, 8000, 8080, 8888, 5000, 9000

        Args:
            zones: Optional list of zones to scan
            ports: Optional list of ports to scan
            min_ports_open: Minimum number of open ports to consider

        Returns:
            List of instances with open ports that might be MCP servers
        """
        return self._google_cloud.discover_mcp_servers_by_port_scan(
            zones=zones, ports=ports, min_ports_open=min_ports_open
        )

    def refresh_gcp_discovery(self):
        """Refresh Google Cloud discovery data."""
        self._google_cloud.refresh_discovery()

    # =========================================================================
    # Deprecated methods (for backward compatibility)
    # =========================================================================

    @property
    def _databricks(self) -> DatabricksPlugin:
        """Get the Databricks plugin."""
        return self.get_plugin("databricks")  # type: ignore

    @property
    def _mcp(self) -> MCPPlugin:
        """Get the MCP plugin."""
        return self.get_plugin("mcp")  # type: ignore

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify connection to Databricks.

        Deprecated: Use verify_all_connections() instead.
        """
        _deprecated("verify_connection() is deprecated. Use verify_all_connections() instead.")
        return self._databricks.verify_connection()

    def verify_otel_connection(self) -> Dict[str, Any]:
        """
        Verify the OpenTelemetry receiver is running.

        Deprecated: Use verify_all_connections()['opentelemetry'] instead.
        """
        _deprecated("verify_otel_connection() is deprecated. Use verify_all_connections()['opentelemetry'] instead.")
        return self._opentelemetry.verify_connection()

    def verify_mcp_discovery(self) -> Dict[str, Any]:
        """
        Verify MCP discovery is working.

        Deprecated: Use verify_all_connections()['mcp'] instead.
        """
        _deprecated("verify_mcp_discovery() is deprecated. Use verify_all_connections()['mcp'] instead.")
        return self._mcp.verify_connection()

    def verify_gcp_connection(self) -> Dict[str, Any]:
        """
        Verify connection to Google Cloud.

        Deprecated: Use verify_all_connections()['google_cloud'] instead.
        """
        _deprecated("verify_gcp_connection() is deprecated. Use verify_all_connections()['google_cloud'] instead.")
        return self._google_cloud.verify_connection()

    def list_otel_tools(self) -> List[Dict[str, Any]]:
        """
        List tools discovered via OpenTelemetry traces.

        Deprecated: Use list_tools() to aggregate from all plugins,
        or list_assets('tool') with filtering.
        """
        _deprecated("list_otel_tools() is deprecated. Use list_tools() instead.")
        return self._opentelemetry.list_assets("tool")

    def list_otel_models(self) -> List[Dict[str, Any]]:
        """
        List models discovered via OpenTelemetry traces.

        Deprecated: Use list_models() to aggregate from all plugins.
        """
        _deprecated("list_otel_models() is deprecated. Use list_models() instead.")
        return self._opentelemetry.list_assets("model")

    def list_gcp_models(self) -> List[Dict[str, Any]]:
        """
        List Vertex AI models.

        Deprecated: Use list_models() to aggregate from all plugins.
        """
        _deprecated("list_gcp_models() is deprecated. Use list_models() instead.")
        return self._google_cloud.list_assets("model")

    def list_gcp_endpoints(self) -> List[Dict[str, Any]]:
        """
        List Vertex AI endpoints.

        Deprecated: Use list_endpoints() to aggregate from all plugins.
        """
        _deprecated("list_gcp_endpoints() is deprecated. Use list_endpoints() instead.")
        return self._google_cloud.list_assets("endpoint")

    def list_gcp_deployments(self) -> List[Dict[str, Any]]:
        """
        List model deployments.

        Deprecated: Use list_deployments() to aggregate from all plugins.
        """
        _deprecated("list_gcp_deployments() is deprecated. Use list_deployments() instead.")
        return self._google_cloud.list_assets("deployment")

    def list_gcp_generative_models(self) -> List[Dict[str, Any]]:
        """
        List available generative AI models.

        Deprecated: Use list_generative_models() to aggregate from all plugins.
        """
        _deprecated("list_gcp_generative_models() is deprecated. Use list_generative_models() instead.")
        return self._google_cloud.list_assets("generative_model")

    def list_gcp_mcp_servers(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List MCP servers running on GCP Compute Engine instances.

        Deprecated: Use list_mcp_servers() to aggregate from all plugins.
        """
        _deprecated("list_gcp_mcp_servers() is deprecated. Use list_mcp_servers() instead.")
        return self._google_cloud.list_assets("mcp_server", **kwargs)


    # Export methods
    def export_to_json(
        self,
        include_otel: bool = True,
        include_databricks: bool = False,
        include_mcp: bool = True,
        include_google_cloud: bool = False,
        filepath: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export all discovered data to JSON format according to OpenCITE schema.

        Args:
            include_otel: Include OpenTelemetry discoveries
            include_databricks: Include Databricks discoveries (experimental)
            include_mcp: Include MCP discoveries (default: True)
            include_google_cloud: Include Google Cloud discoveries
            filepath: Optional path to save JSON file

        Returns:
            JSON-serializable dictionary with all discovered data
        """
        exporter = OpenCiteExporter()

        tools = []
        models = []
        data_assets = []
        mcp_servers = []
        mcp_tools = []
        mcp_resources = []
        gcp_models = []
        gcp_endpoints = []
        gcp_deployments = []
        gcp_generative_models = []
        gcp_mcp_servers = []
        plugins_info = []

        # Gather OpenTelemetry data
        if include_otel and "opentelemetry" in self.plugins:
            otel_tools = self._export_otel_tools()
            otel_models = self._export_otel_models()

            tools.extend(otel_tools)
            models.extend(otel_models)

            plugins_info.append({
                "name": "opentelemetry",
                "version": "1.0.0",
            })

        # Gather MCP data (aggregate from all plugins that support MCP)
        if include_mcp:
            mcp_servers = self._export_mcp_servers()
            mcp_tools = self._export_mcp_tools()
            mcp_resources = self._export_mcp_resources()

            if "mcp" in self.plugins:
                plugins_info.append({
                    "name": "mcp",
                    "version": "1.0.0",
                })

        # Gather Databricks data (if requested)
        if include_databricks and "databricks" in self.plugins:
            databricks_assets = self._export_databricks_assets()
            data_assets.extend(databricks_assets)

            plugins_info.append({
                "name": "databricks",
                "version": "1.0.0",
            })

        # Gather Google Cloud data
        if include_google_cloud and "google_cloud" in self.plugins:
            # Use generic list methods which aggregate from all plugins
            gcp_plugin = self._google_cloud
            gcp_models = gcp_plugin.list_assets("model")
            gcp_endpoints = gcp_plugin.list_assets("endpoint")
            gcp_deployments = gcp_plugin.list_assets("deployment")
            gcp_generative_models = gcp_plugin.list_assets("generative_model")

            # GCP MCP servers are already included via list_mcp_servers aggregation
            # but we keep them separate for the export schema
            gcp_mcp_servers = gcp_plugin.list_assets("mcp_server")

            plugins_info.append({
                "name": "google_cloud",
                "version": "1.0.0",
            })

        # Build metadata
        metadata = {
            "generated_by": "opencite-client",
            "plugins": plugins_info,
        }

        # Export with MCP entities
        export_data = exporter.export_discovery(
            tools=tools,
            models=models,
            data_assets=data_assets,
            mcp_servers=mcp_servers,
            mcp_tools=mcp_tools,
            mcp_resources=mcp_resources,
            metadata=metadata,
        )

        # Add Google Cloud entities
        export_data["gcp_models"] = gcp_models
        export_data["gcp_endpoints"] = gcp_endpoints
        export_data["gcp_deployments"] = gcp_deployments
        export_data["gcp_generative_models"] = gcp_generative_models
        export_data["gcp_mcp_servers"] = gcp_mcp_servers

        # Save to file if requested
        if filepath:
            exporter.save_to_file(export_data, filepath, validate=False)
            logger.info(f"Exported data to {filepath}")

        return export_data

    def _export_otel_tools(self) -> List[Dict[str, Any]]:
        """Export OpenTelemetry tools to schema format."""
        otel_tools = self._opentelemetry.list_assets("tool")
        formatted_tools = []

        for tool in otel_tools:
            # Build models_used list
            models_used = []
            for model_name in tool.get("models", []):
                models_used.append({
                    "model_id": model_name,
                    "usage_count": len([
                        t for t in tool.get("traces", [])
                        if t.get("model") == model_name
                    ]),
                })

            formatted_tool = ToolFormatter.format_tool(
                tool_id=tool["name"],
                name=tool["name"],
                discovery_source="opentelemetry",
                type="application",
                models_used=models_used,
                provider=None,  # Provider can be determined from the model or trace data
                last_seen=tool.get("metadata", {}).get("last_seen"),
                metadata=tool.get("metadata", {}),
            )

            formatted_tools.append(formatted_tool)

        return formatted_tools

    def _export_otel_models(self) -> List[Dict[str, Any]]:
        """Export OpenTelemetry models to schema format."""
        otel_models = self._opentelemetry.list_assets("model")
        formatted_models = []

        for model in otel_models:
            model_id = model["name"]
            parsed = parse_model_id(model_id)

            formatted_model = ModelFormatter.format_model(
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
            )

            formatted_models.append(formatted_model)

        return formatted_models


    def _export_mcp_servers(self) -> List[Dict[str, Any]]:
        """Export MCP servers to schema format."""
        mcp_servers = self.list_mcp_servers()
        formatted_servers = []

        for server in mcp_servers:
            formatted_server = MCPServerFormatter.format_mcp_server(
                server_id=server["id"],
                name=server["name"],
                discovery_source=server["discovery_source"],
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
            )

            formatted_servers.append(formatted_server)

        return formatted_servers

    def _export_mcp_tools(self) -> List[Dict[str, Any]]:
        """Export MCP tools to schema format."""
        mcp_tools = self.list_mcp_tools()
        formatted_tools = []

        for tool in mcp_tools:
            formatted_tool = MCPToolFormatter.format_mcp_tool(
                tool_id=tool["id"],
                name=tool["name"],
                server_id=tool["server_id"],
                discovery_source=tool.get("discovery_source"),
                description=tool.get("description"),
                schema=tool.get("schema"),
                usage=tool.get("usage"),
                metadata=tool.get("metadata"),
            )

            formatted_tools.append(formatted_tool)

        return formatted_tools

    def _export_mcp_resources(self) -> List[Dict[str, Any]]:
        """Export MCP resources to schema format."""
        mcp_resources = self.list_mcp_resources()
        formatted_resources = []

        for resource in mcp_resources:
            formatted_resource = MCPResourceFormatter.format_mcp_resource(
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
            )

            formatted_resources.append(formatted_resource)

        return formatted_resources

    def _export_databricks_assets(self) -> List[Dict[str, Any]]:
        """
        Export Databricks data assets that have been used by AI/ML tools.

        Uses the Databricks plugin to query audit logs for table usage.
        Only exports tables with clear ties to AI/ML usage.
        """
        data_assets = []

        try:
            # Get tables used by AI/ML workloads from Databricks
            used_tables = {}
            if "databricks" in self.plugins:
                databricks_plugin = self.plugins["databricks"]
                try:
                    # Query for tables accessed by MLflow experiments (AI/ML workloads)
                    ai_workload_usage = databricks_plugin.get_tables_used_by_ai_workloads(days=30)
                    if ai_workload_usage:
                        logger.info(f"Found {len(ai_workload_usage)} tables from AI/ML workloads (MLflow experiments)")

                        # Format for export - these are confirmed AI/ML tables
                        for table_name, usage in ai_workload_usage.items():
                            used_tables[table_name] = {
                                "ai_users": usage.get("ai_users", []),  # Users running AI experiments
                                "ai_experiments": usage.get("ai_experiments", []),  # MLflow experiment IDs
                                "ai_experiment_names": usage.get("ai_experiment_names", []),  # Experiment names
                                "access_count": usage.get("access_count", 0),
                                "first_seen": usage.get("first_seen"),
                                "last_seen": usage.get("last_seen"),
                                "discovery_method": "mlflow_experiments"  # Clearly indicate this is AI-specific
                            }
                except Exception as e:
                    logger.warning(f"Could not fetch AI workload table usage: {e}")

                try:
                    # Query for tables accessed by Genie spaces
                    genie_usage = databricks_plugin.get_tables_used_by_genie(days=30)
                    if genie_usage:
                        logger.info(f"Found {len(genie_usage)} tables from Genie spaces")
                        
                        # Merge into used_tables
                        for table_name, usage in genie_usage.items():
                            if table_name not in used_tables:
                                used_tables[table_name] = {
                                    "discovery_method": "genie_spaces"
                                }
                            
                            # Merge usage info
                            table_info = used_tables[table_name]
                            table_info["genie_users"] = usage.get("genie_users", [])
                            table_info["genie_spaces"] = usage.get("genie_spaces", [])
                            table_info.setdefault("access_count", 0)
                            table_info["access_count"] += usage.get("access_count", 0)
                            
                            # Update timestamps
                            if usage.get("first_seen"):
                                if not table_info.get("first_seen") or usage["first_seen"] < table_info["first_seen"]:
                                    table_info["first_seen"] = usage["first_seen"]
                                    
                            if usage.get("last_seen"):
                                if not table_info.get("last_seen") or usage["last_seen"] > table_info["last_seen"]:
                                    table_info["last_seen"] = usage["last_seen"]

                except Exception as e:
                    logger.warning(f"Could not fetch Genie table usage: {e}")

            if not used_tables:
                logger.info("No Databricks tables with AI/ML usage detected. Skipping Databricks export.")
                return data_assets

            logger.info(f"Total: {len(used_tables)} Databricks tables with AI/ML usage detected")

            # Parse table names to get unique catalogs and schemas
            catalogs_needed = set()
            schemas_needed = set()  # (catalog, schema) tuples

            for table_full_name in used_tables.keys():
                parts = table_full_name.split(".")
                if len(parts) == 3:
                    catalog_name, schema_name, table_name = parts
                    catalogs_needed.add(catalog_name)
                    schemas_needed.add((catalog_name, schema_name))

            # Export only catalogs that contain used tables
            for catalog_name in catalogs_needed:
                try:
                    # Fetch catalog details
                    catalogs = self.list_catalogs()
                    catalog = next((c for c in catalogs if c["name"].lower() == catalog_name.lower()), None)

                    if catalog:
                        catalog_asset = DataAssetFormatter.format_data_asset(
                            asset_id=catalog["id"],
                            name=catalog["name"],
                            asset_type="catalog",
                            discovery_source="databricks",
                            hierarchy={
                                "catalog": catalog["name"]
                            },
                            metadata={
                                "owner": catalog.get("owner"),
                                "comment": catalog.get("comment"),
                                "ai_usage": "Contains tables accessed by AI/ML tools"
                            }
                        )
                        data_assets.append(catalog_asset)
                except Exception as e:
                    logger.warning(f"Failed to export catalog {catalog_name}: {e}")

            # Export only schemas that contain used tables
            for catalog_name, schema_name in schemas_needed:
                try:
                    schemas = self.list_schemas(catalog_name)
                    schema = next((s for s in schemas if s["name"].lower() == schema_name.lower()), None)

                    if schema:
                        schema_asset = DataAssetFormatter.format_data_asset(
                            asset_id=f"{catalog_name}.{schema_name}",
                            name=schema["name"],
                            asset_type="schema",
                            discovery_source="databricks",
                            full_name=f"{catalog_name}.{schema_name}",
                            hierarchy={
                                "catalog": catalog_name,
                                "schema": schema_name
                            },
                            metadata={
                                "owner": schema.get("owner"),
                                "comment": schema.get("comment"),
                                "ai_usage": "Contains tables accessed by AI/ML tools"
                            }
                        )
                        data_assets.append(schema_asset)
                except Exception as e:
                    logger.warning(f"Failed to export schema {catalog_name}.{schema_name}: {e}")

            # Export only used tables with usage metadata
            for table_full_name, usage_info in used_tables.items():
                parts = table_full_name.split(".")
                if len(parts) != 3:
                    logger.warning(f"Invalid table name format: {table_full_name}")
                    continue

                catalog_name, schema_name, table_name = parts

                try:
                    # Fetch table details from Databricks
                    tables = self.list_tables(catalog_name, schema_name)
                    table = next((t for t in tables if t["name"].lower() == table_name.lower()), None)

                    if table:
                        table_asset = DataAssetFormatter.format_data_asset(
                            asset_id=table_full_name,
                            name=table["name"],
                            asset_type="table",
                            discovery_source="databricks",
                            full_name=table_full_name,
                            hierarchy={
                                "catalog": catalog_name,
                                "schema": schema_name,
                                "table": table_name
                            },
                            metadata={
                                "table_type": table.get("table_type"),
                                "owner": table.get("owner"),
                                "comment": table.get("comment"),
                                # Add AI/ML usage metadata (confirmed via MLflow experiments)
                                "ai_users": usage_info.get("ai_users", []),  # Users running AI experiments
                                "ai_experiments": usage_info.get("ai_experiments", []),  # MLflow experiment IDs
                                "ai_experiment_names": usage_info.get("ai_experiment_names", []),  # Experiment names
                                "ai_access_count": usage_info.get("access_count", 0),
                                "ai_first_seen": usage_info.get("first_seen"),
                                "ai_last_seen": usage_info.get("last_seen"),
                                "ai_discovery_method": usage_info.get("discovery_method", "unknown"),
                                # Add Genie usage metadata
                                "genie_users": usage_info.get("genie_users", []),
                                "genie_spaces": usage_info.get("genie_spaces", []),
                            }
                        )
                        data_assets.append(table_asset)
                    else:
                        logger.warning(f"Table {table_full_name} not found in Databricks (may have been deleted)")
                except Exception as e:
                    logger.warning(f"Failed to export table {table_full_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to export Databricks assets: {e}")

        return data_assets
