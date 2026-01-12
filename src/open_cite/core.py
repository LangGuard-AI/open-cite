from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDiscoveryPlugin(ABC):
    """
    Abstract base class for Open Cite discovery plugins.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the plugin (e.g., 'databricks').
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
        """
        pass
