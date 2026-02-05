from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set


class BaseDiscoveryPlugin(ABC):
    """
    Abstract base class for Open Cite discovery plugins.

    Each plugin must implement:
    - name: Unique plugin identifier
    - supported_asset_types: Set of asset types this plugin can discover
    - verify_connection: Check connectivity to the data source
    - list_assets: List assets of a given type
    - get_identification_attributes: Attributes used for tool identification
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the plugin (e.g., 'databricks').
        """
        pass

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

    @abstractmethod
    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify connection to the source.
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
            - discovery_source: Plugin name that discovered this asset
        """
        pass

    @abstractmethod
    def get_identification_attributes(self) -> List[str]:
        """
        Return a list of attribute keys used for tool identification.
        """
        pass
