import logging
import warnings
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Type, Set
from .core import BaseDiscoveryPlugin
from .plugins.registry import create_plugin_instance
from .http_client import get_http_client, OpenCiteHttpClient


logger = logging.getLogger(__name__)

# Asset types that are read from the DB (when persistence is available)
# rather than from plugin in-memory structures.
_PERSISTED_ASSET_TYPES = {
    "tool", "model", "agent", "downstream_system",
    "mcp_server", "mcp_tool", "mcp_resource",
}


def _deprecated(message: str):
    """Emit a deprecation warning."""
    warnings.warn(message, DeprecationWarning, stacklevel=3)

class OpenCiteClient:
    """
    Client for Open Cite. Manages discovery plugins.
    """

    def __init__(self):
        """
        Initialize the OpenCite client.

        Plugins are registered explicitly via register_plugin() or through the GUI.
        """
        # Plugin registry: instance_id -> plugin instance
        self.plugins: Dict[str, BaseDiscoveryPlugin] = {}
        # Track which instances belong to which plugin type
        self._plugin_types: Dict[str, List[str]] = defaultdict(list)

        # Optional persistence layer — when set, list methods merge
        # persisted data with live plugin data so assets survive restarts.
        self.persistence = None

        # Initialize central HTTP client
        self.http_client = get_http_client()

    def register_plugin(self, plugin: BaseDiscoveryPlugin):
        """
        Register a discovery plugin instance.

        Args:
            plugin: The plugin instance to register

        Raises:
            ValueError: If a plugin with the same instance_id is already registered
        """
        instance_id = plugin.instance_id

        if instance_id in self.plugins:
            raise ValueError(f"Plugin instance '{instance_id}' already registered")

        self.plugins[instance_id] = plugin
        self._plugin_types[plugin.plugin_type].append(instance_id)
        logger.info(f"Registered plugin: {instance_id} (type: {plugin.plugin_type})")

    def unregister_plugin(self, instance_id: str):
        """
        Unregister a plugin instance.

        Args:
            instance_id: The instance ID of the plugin to unregister

        Raises:
            ValueError: If the plugin instance is not found
        """
        if instance_id not in self.plugins:
            raise ValueError(f"Plugin instance '{instance_id}' not found")

        plugin = self.plugins[instance_id]
        del self.plugins[instance_id]
        self._plugin_types[plugin.plugin_type].remove(instance_id)
        logger.info(f"Unregistered plugin: {instance_id}")

    def get_plugins_by_type(self, plugin_type: str) -> List[BaseDiscoveryPlugin]:
        """
        Get all instances of a plugin type.

        Args:
            plugin_type: The plugin type to query for (e.g., 'databricks', 'opentelemetry')

        Returns:
            List of plugin instances of that type
        """
        return [self.plugins[iid] for iid in self._plugin_types.get(plugin_type, [])]

    def list_plugin_instances(self) -> List[Dict[str, Any]]:
        """
        List all plugin instances with metadata.

        Returns:
            List of plugin instance information dictionaries
        """
        return [plugin.to_dict() for plugin in self.plugins.values()]

    def get_plugin(self, instance_id: str) -> BaseDiscoveryPlugin:
        """
        Get a registered plugin by instance ID.

        For backward compatibility, if instance_id matches a plugin_type and only
        one instance of that type exists, returns that instance.

        Args:
            instance_id: The instance ID (or plugin type for single instances)

        Returns:
            The plugin instance

        Raises:
            ValueError: If the plugin instance is not found
        """
        # Direct lookup by instance_id
        if instance_id in self.plugins:
            return self.plugins[instance_id]

        # Backward compatibility: try looking up by plugin_type
        # if there's exactly one instance of that type
        if instance_id in self._plugin_types:
            instances = self._plugin_types[instance_id]
            if len(instances) == 1:
                return self.plugins[instances[0]]
            elif len(instances) > 1:
                raise ValueError(
                    f"Multiple instances of plugin type '{instance_id}' exist. "
                    f"Specify instance_id: {instances}"
                )

        raise ValueError(
            f"Plugin '{instance_id}' not found. "
            f"Available plugins: {list(self.plugins.keys())}"
        )

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
        List assets of a specific type.

        For persisted asset types (tool, model, agent, downstream_system,
        mcp_server, mcp_tool, mcp_resource), reads directly from the database
        when persistence is available.  For all other types, aggregates results
        from registered plugins.

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
        # For persisted types, merge DB rows with live plugin data so that
        # newly-discovered items appear immediately (from plugin memory)
        # while previously-saved items survive restarts (from DB).
        if self.persistence and asset_type in _PERSISTED_ASSET_TYPES:
            merged: Dict[str, Dict[str, Any]] = {}

            # 1. DB data (primary — survives restarts)
            try:
                for item in self._list_from_db(asset_type, **kwargs):
                    key = item.get("name") or item.get("id") or item.get("uri", "")
                    merged[key] = item
            except Exception as e:
                logger.warning(
                    f"DB read failed for {asset_type}: {e}"
                )

            # 2. Live plugin data (fills gaps for items not yet saved)
            for plugin in self.get_plugins_for_asset_type(asset_type):
                try:
                    for item in plugin.list_assets(asset_type, **kwargs):
                        if "discovery_source" not in item:
                            item["discovery_source"] = plugin.display_name
                        key = item.get("name") or item.get("id") or item.get("uri", "")
                        if key not in merged:
                            merged[key] = item
                except Exception as e:
                    logger.warning(f"Failed to list {asset_type} from plugin {plugin.instance_id}: {e}")

            return list(merged.values())

        # Non-persisted types: aggregate from plugins only
        results = []
        plugins = self.get_plugins_for_asset_type(asset_type)

        for plugin in plugins:
            try:
                assets = plugin.list_assets(asset_type, **kwargs)
                # Ensure each asset has discovery_source set to a readable name
                for asset in assets:
                    if "discovery_source" not in asset:
                        asset["discovery_source"] = plugin.display_name
                results.extend(assets)
            except Exception as e:
                logger.warning(f"Failed to list {asset_type} from plugin {plugin.instance_id}: {e}")

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

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all discovered agents from all plugins.

        Returns:
            Aggregated list of agents
        """
        return self.list_assets("agent")

    def list_downstream_systems(self) -> List[Dict[str, Any]]:
        """
        List all discovered downstream systems from all plugins.

        Returns:
            Aggregated list of downstream systems
        """
        return self.list_assets("downstream_system")

    def list_lineage(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all lineage relationships.

        Reads from the database when persistence is available, otherwise
        aggregates from plugins.

        Args:
            source_id: Optional asset ID to filter by

        Returns:
            List of lineage relationships
        """
        if self.persistence:
            try:
                return self.persistence.load_lineage(source_id=source_id)
            except Exception as e:
                logger.warning(f"DB read failed for lineage, falling back to plugins: {e}")

        results = []
        for plugin in self.plugins.values():
            if hasattr(plugin, 'list_lineage'):
                try:
                    relationships = plugin.list_lineage(source_id=source_id)
                    results.extend(relationships)
                except Exception as e:
                    logger.warning(f"Failed to get lineage from {plugin.instance_id}: {e}")
        return results

    # =========================================================================
    # Private DB read methods (used when persistence is available)
    # =========================================================================

    def _list_from_db(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Dispatch a DB read for a persisted asset type."""
        dispatch = {
            "tool": self._list_tools_from_db,
            "model": self._list_models_from_db,
            "agent": self._list_agents_from_db,
            "downstream_system": self._list_downstream_from_db,
            "mcp_server": self._list_mcp_servers_from_db,
            "mcp_tool": self._list_mcp_tools_from_db,
            "mcp_resource": self._list_mcp_resources_from_db,
        }
        handler = dispatch[asset_type]
        return handler(**kwargs)

    def _list_tools_from_db(self, **kwargs) -> List[Dict[str, Any]]:
        """Read tools from the database."""
        rows = self.persistence.load_tools()
        tools = []
        for name, data in rows.items():
            metadata = data.get("metadata", {})
            tool = {
                "id": name,
                "name": name,
                "type": "llm_client",
                "discovery_source": metadata.get("discovery_source", "opentelemetry"),
                "models": list(data.get("models", set())),
                "trace_count": data.get("trace_count", 0),
                "metadata": metadata,
            }
            # Surface identification fields from metadata as top-level keys
            if metadata.get("tool_source_name"):
                tool["tool_source_name"] = metadata["tool_source_name"]
            if metadata.get("tool_source_id"):
                tool["tool_source_id"] = metadata["tool_source_id"]
            # Surface any other metadata fields that the plugin would
            # have exposed at the top level
            for key in ("source", "service_name", "http_method",
                        "http_url", "db_system", "db_statement"):
                if key in metadata:
                    tool[key] = metadata[key]
            tools.append(tool)
        return tools

    def _list_models_from_db(self, **kwargs) -> List[Dict[str, Any]]:
        """Read models from the database."""
        rows = self.persistence.load_models()
        models = []
        for name, data in rows.items():
            models.append({
                "name": name,
                "provider": data.get("provider", "unknown"),
                "tools": list(data.get("tools", set())),
                "usage_count": data.get("usage_count", 0),
            })
        return models

    def _list_agents_from_db(self, **kwargs) -> List[Dict[str, Any]]:
        """Read agents from the database."""
        rows = self.persistence.load_agents()
        agents = []
        for agent_id, data in rows.items():
            metadata = data.get("metadata", {})
            agents.append({
                "id": agent_id,
                "name": data.get("name", agent_id),
                "discovery_source": metadata.get("discovery_source", "opentelemetry"),
                "tools_used": list(data.get("tools_used", [])),
                "models_used": list(data.get("models_used", [])),
                "first_seen": data.get("first_seen"),
                "last_seen": data.get("last_seen"),
                "agent_source_name": metadata.get("agent_source_name"),
                "agent_source_id": metadata.get("agent_source_id"),
                "metadata": metadata,
            })
        return agents

    def _list_downstream_from_db(self, **kwargs) -> List[Dict[str, Any]]:
        """Read downstream systems from the database."""
        rows = self.persistence.load_downstream_systems()
        systems = []
        for sys_id, data in rows.items():
            systems.append({
                "id": sys_id,
                "name": data.get("name", ""),
                "type": data.get("type", "unknown"),
                "endpoint": data.get("endpoint"),
                "tools_connecting": list(data.get("tools_connecting", [])),
                "first_seen": data.get("first_seen"),
                "last_seen": data.get("last_seen"),
                "metadata": data.get("metadata", {}),
            })
        return systems

    def _list_mcp_servers_from_db(self, **kwargs) -> List[Dict[str, Any]]:
        """Read MCP servers from the database."""
        rows = self.persistence.load_mcp_servers()
        servers = []
        for server_id, data in rows.items():
            server = {
                "id": data.get("id", server_id),
                "name": data.get("name", ""),
                "discovery_source": "opentelemetry",
                "transport": data.get("transport", "unknown"),
                "metadata": data.get("metadata", {}),
            }
            if data.get("endpoint"):
                server["endpoint"] = data["endpoint"]
            if data.get("command"):
                server["command"] = data["command"]
            if data.get("args"):
                server["args"] = data["args"]
            if data.get("env"):
                server["env"] = data["env"]
            if data.get("source_file"):
                server["source_file"] = data["source_file"]
            if data.get("source_env_var"):
                server["source_env_var"] = data["source_env_var"]
            servers.append(server)
        return servers

    def _list_mcp_tools_from_db(self, **kwargs) -> List[Dict[str, Any]]:
        """Read MCP tools from the database."""
        server_id = kwargs.get("server_id")
        rows = self.persistence.load_mcp_tools()
        tools = []
        for tool_id, data in rows.items():
            if server_id and data.get("server_id") != server_id:
                continue
            usage = data.get("usage") or {}
            tool = {
                "id": data.get("id", tool_id),
                "name": data.get("name", ""),
                "server_id": data.get("server_id", ""),
                "discovery_source": "opentelemetry",
                "usage": usage,
                "call_count": usage.get("call_count", 0),
                "metadata": data.get("metadata", {}),
            }
            if data.get("description"):
                tool["description"] = data["description"]
            if data.get("schema"):
                tool["schema"] = data["schema"]
            tools.append(tool)
        return tools

    def _list_mcp_resources_from_db(self, **kwargs) -> List[Dict[str, Any]]:
        """Read MCP resources from the database."""
        server_id = kwargs.get("server_id")
        rows = self.persistence.load_mcp_resources()
        resources = []
        for resource_id, data in rows.items():
            if server_id and data.get("server_id") != server_id:
                continue
            usage = data.get("usage") or {}
            resource = {
                "id": data.get("id", resource_id),
                "uri": data.get("uri", ""),
                "server_id": data.get("server_id", ""),
                "discovery_source": "opentelemetry",
                "usage": usage,
                "access_count": usage.get("access_count", 0),
                "metadata": data.get("metadata", {}),
            }
            if data.get("name"):
                resource["name"] = data["name"]
            if data.get("type"):
                resource["type"] = data["type"]
            if data.get("mime_type"):
                resource["mime_type"] = data["mime_type"]
            if data.get("description"):
                resource["description"] = data["description"]
            resources.append(resource)
        return resources

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
    def _opentelemetry(self) -> 'OpenTelemetryPlugin':
        """Get the OpenTelemetry plugin."""
        return self.get_plugin("opentelemetry")  # type: ignore

    def start_otel_receiver(self):
        """Start the OpenTelemetry receiver."""
        self._opentelemetry.start_receiver()

    def stop_otel_receiver(self):
        """Stop the OpenTelemetry receiver."""
        self._opentelemetry.stop_receiver()

    @property
    def _google_cloud(self) -> 'GoogleCloudPlugin':
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
    def _databricks(self) -> 'DatabricksPlugin':
        """Get the Databricks plugin."""
        return self.get_plugin("databricks")  # type: ignore

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


    # Convenience methods for the AWS Bedrock plugin
    @property
    def _aws_bedrock(self) -> 'AWSBedrockPlugin':
        """Get the AWS Bedrock plugin."""
        return self.get_plugin("aws_bedrock")  # type: ignore

    def list_bedrock_models(self) -> List[Dict[str, Any]]:
        """
        List available Bedrock foundation models.

        Returns:
            List of foundation models (Claude, Llama, Titan, etc.)
        """
        return self._aws_bedrock.list_assets("model")

    def list_bedrock_custom_models(self) -> List[Dict[str, Any]]:
        """
        List custom/fine-tuned Bedrock models.

        Returns:
            List of custom models
        """
        return self._aws_bedrock.list_assets("custom_model")

    def list_bedrock_provisioned_throughput(self) -> List[Dict[str, Any]]:
        """
        List Bedrock provisioned throughput configurations.

        Returns:
            List of provisioned throughput configs
        """
        return self._aws_bedrock.list_assets("provisioned_throughput")

    def list_bedrock_invocations(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        List Bedrock model invocations from CloudTrail.

        This shows which Bedrock models are actually being USED.

        Args:
            days: Number of days to look back (default: 7)

        Returns:
            List of invocation events
        """
        return self._aws_bedrock.list_assets("invocation", days=days)

    def get_bedrock_usage_by_model(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Get Bedrock usage statistics aggregated by model.

        Args:
            days: Number of days to analyze

        Returns:
            Dict mapping model_id to usage stats (invocation count, users, etc.)
        """
        return self._aws_bedrock.get_usage_by_model(days=days)

    def get_bedrock_usage_by_user(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Get Bedrock usage statistics aggregated by user/role.

        Args:
            days: Number of days to analyze

        Returns:
            Dict mapping user ARN to usage stats
        """
        return self._aws_bedrock.get_usage_by_user(days=days)

    def verify_bedrock_connection(self) -> Dict[str, Any]:
        """
        Verify connection to AWS Bedrock.

        Returns:
            Dict with connection status
        """
        return self._aws_bedrock.verify_connection()

    def refresh_bedrock_discovery(self):
        """Refresh AWS Bedrock discovery data."""
        self._aws_bedrock.refresh_discovery()

    # Convenience methods for the AWS SageMaker plugin
    @property
    def _aws_sagemaker(self) -> 'AWSSageMakerPlugin':
        """Get the AWS SageMaker plugin."""
        return self.get_plugin("aws_sagemaker")  # type: ignore

    def list_sagemaker_endpoints(self) -> List[Dict[str, Any]]:
        """
        List SageMaker endpoints.

        Returns:
            List of deployed model endpoints
        """
        return self._aws_sagemaker.list_assets("endpoint")

    def list_sagemaker_models(self) -> List[Dict[str, Any]]:
        """
        List SageMaker models.

        Returns:
            List of registered models
        """
        return self._aws_sagemaker.list_assets("model")

    def list_sagemaker_model_packages(self) -> List[Dict[str, Any]]:
        """
        List SageMaker model registry packages.

        Returns:
            List of model packages in the registry
        """
        return self._aws_sagemaker.list_assets("model_package")

    def list_sagemaker_training_jobs(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        List recent SageMaker training jobs.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            List of training jobs
        """
        return self._aws_sagemaker.list_assets("training_job", days=days)

    def get_sagemaker_endpoint_metrics(
        self,
        endpoint_name: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get invocation metrics for a specific SageMaker endpoint.

        Args:
            endpoint_name: Name of the endpoint
            days: Number of days to look back

        Returns:
            Dict with invocation count, latency, errors, etc.
        """
        return self._aws_sagemaker.get_endpoint_invocation_metrics(endpoint_name, days)

    def get_all_sagemaker_endpoint_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get invocation metrics for all SageMaker endpoints.

        Args:
            days: Number of days to look back

        Returns:
            List of endpoint metrics, sorted by invocation count
        """
        return self._aws_sagemaker.get_all_endpoint_metrics(days=days)

    def get_sagemaker_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get a summary of SageMaker usage.

        Args:
            days: Number of days to analyze

        Returns:
            Summary with counts, top endpoints, etc.
        """
        return self._aws_sagemaker.get_usage_summary(days=days)

    def verify_sagemaker_connection(self) -> Dict[str, Any]:
        """
        Verify connection to AWS SageMaker.

        Returns:
            Dict with connection status
        """
        return self._aws_sagemaker.verify_connection()

    def refresh_sagemaker_discovery(self):
        """Refresh AWS SageMaker discovery data."""
        self._aws_sagemaker.refresh_discovery()
