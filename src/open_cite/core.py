"""
OpenCITE Core - Base classes and interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import warnings


class BaseDiscoveryPlugin(ABC):
    """
    Abstract base class for Open Cite discovery plugins.

    Each plugin must implement:
    - plugin_type: Type identifier (e.g., 'databricks')
    - supported_asset_types: Set of asset types this plugin can discover
    - supports_multiple_instances: Whether multiple instances are allowed
    - verify_connection: Check connectivity to the data source
    - list_assets: List assets of a given type
    - get_identification_attributes: Attributes used for tool identification
    """

    def __init__(
        self,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the plugin instance.

        Args:
            instance_id: Unique identifier for this instance. Defaults to plugin_type.
            display_name: Human-readable name. Defaults to formatted plugin_type.
        """
        self._instance_id = instance_id or self.plugin_type
        self._display_name = display_name or self.plugin_type.replace('_', ' ').title()
        self._status = "stopped"  # running, stopped, error

    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """
        Type identifier for this plugin (e.g., 'databricks').

        This is immutable and the same for all instances of a plugin class.
        """
        pass

    @property
    def instance_id(self) -> str:
        """
        Unique identifier for this plugin instance.

        Defaults to plugin_type if not specified. Must be unique across all
        registered plugin instances.
        """
        return self._instance_id

    @instance_id.setter
    def instance_id(self, value: str):
        self._instance_id = value

    @property
    def display_name(self) -> str:
        """
        Human-readable display name for this instance.

        Shown in the UI to help users identify different instances.
        """
        return self._display_name

    @display_name.setter
    def display_name(self, value: str):
        self._display_name = value

    @property
    def status(self) -> str:
        """
        Current status of the plugin instance.

        Values: 'running', 'stopped', 'error'
        """
        return self._status

    @status.setter
    def status(self, value: str):
        if value not in ('running', 'stopped', 'error'):
            raise ValueError(f"Invalid status: {value}")
        self._status = value

    @property
    def name(self) -> str:
        """
        Backward-compatible name property.

        Deprecated: Use instance_id for unique identification or plugin_type
        for the type of plugin.
        """
        return self._instance_id

    @property
    @abstractmethod
    def supported_asset_types(self) -> Set[str]:
        """
        Return the set of asset types this plugin supports.

        Standard asset types:
        - tool: Discovered tools/applications using AI
        - model: AI models
        - endpoint: API endpoints
        - mcp_server: MCP servers
        - mcp_tool: MCP tools
        - mcp_resource: MCP resources
        - catalog: Data catalogs
        - schema: Data schemas
        - table: Data tables
        - volume: Data volumes
        - function: Functions (UDFs, etc.)
        - deployment: Model deployments
        - generative_model: Generative AI models
        """
        pass

    @property
    def supports_multiple_instances(self) -> bool:
        """
        Whether this plugin type supports multiple simultaneous instances.

        Override in subclass to return True if the plugin can be instantiated
        multiple times (e.g., connecting to different Databricks workspaces).

        Default is True for most plugins.
        """
        return True

    @abstractmethod
    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify connection to the source.

        Returns:
            Dict with at minimum:
            - success: bool indicating if connection is valid
            - error: str with error message if success is False
        """
        pass

    @abstractmethod
    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List assets of a specific type (e.g., 'catalog', 'schema', 'table').

        Args:
            asset_type: Type of asset to list (must be in supported_asset_types)
            **kwargs: Additional filters (e.g., catalog_name, schema_name)

        Returns:
            List of asset dictionaries with at minimum:
            - id: Unique identifier
            - name: Display name
            - discovery_source: Instance ID that discovered this asset
        """
        pass

    @abstractmethod
    def get_identification_attributes(self) -> List[str]:
        """
        Return a list of attribute keys used for tool identification.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of this plugin instance.

        Override in subclass to return plugin-specific configuration.
        Sensitive values (tokens, passwords) should be masked or omitted.

        Returns:
            Dict with configuration key-value pairs
        """
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize plugin instance metadata to a dictionary.

        Returns:
            Dict with instance metadata
        """
        return {
            "instance_id": self.instance_id,
            "plugin_type": self.plugin_type,
            "display_name": self.display_name,
            "status": self.status,
            "supports_multiple_instances": self.supports_multiple_instances,
            "supported_asset_types": list(self.supported_asset_types),
            "config": self.get_config(),
        }
