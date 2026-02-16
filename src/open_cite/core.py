"""
OpenCITE Core - Base classes and interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import warnings

import requests

logger = logging.getLogger(__name__)


class BaseDiscoveryPlugin(ABC):
    """
    Abstract base class for Open Cite discovery plugins.

    Each plugin must implement:
    - plugin_type: Type identifier (e.g., 'databricks') - class attribute
    - supported_asset_types: Set of asset types this plugin can discover
    - supports_multiple_instances: Whether multiple instances are allowed
    - verify_connection: Check connectivity to the data source
    - list_assets: List assets of a given type
    - get_identification_attributes: Attributes used for tool identification

    Plugins should also provide:
    - plugin_metadata(): classmethod returning display name, description, config fields
    - from_config(): classmethod factory to create instance from config dict
    - start() / stop(): lifecycle methods for plugins that need initialization/cleanup
    """

    # Subclasses should set this as a class attribute, e.g. plugin_type = "databricks"
    # For backward compat, it can also be a @property.
    plugin_type: str = ""

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
        self._on_data_changed = None
        self._webhook_urls: Set[str] = set()
        self._webhook_executor: Optional[ThreadPoolExecutor] = None

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        """
        Return metadata describing this plugin type.

        Subclasses should override to provide specific metadata.

        Returns:
            Dict with keys:
            - name: Human-readable plugin name
            - description: What this plugin discovers
            - required_fields: Dict of config field name -> {label, default, required, type}
            - env_vars: List of environment variable names used
            - (optional) trace_endpoints: endpoint info for trace-based plugins
        """
        return {
            "name": cls.plugin_type.replace('_', ' ').title() if cls.plugin_type else cls.__name__,
            "description": "Discovery plugin",
            "required_fields": {},
            "env_vars": [],
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> 'BaseDiscoveryPlugin':
        """
        Factory method to create an instance from a config dict.

        Subclasses should override to handle their specific config fields.

        Args:
            config: Plugin-specific configuration dict
            instance_id: Unique instance identifier
            display_name: Human-readable name
            dependencies: Optional dict of dependencies (e.g., mcp_plugin, http_client)

        Returns:
            New plugin instance
        """
        return cls(instance_id=instance_id, display_name=display_name)

    def start(self):
        """
        Start the plugin (lifecycle method).

        Override in subclasses that need initialization (e.g., start a receiver).
        Default implementation just sets status to 'running'.
        """
        self._status = "running"
        logger.info(f"Started plugin {self.instance_id}")

    def stop(self):
        """
        Stop the plugin (lifecycle method).

        Override in subclasses that need cleanup (e.g., stop a receiver).
        Default implementation just sets status to 'stopped'.
        """
        self._status = "stopped"
        if self._webhook_executor:
            self._webhook_executor.shutdown(wait=False)
            self._webhook_executor = None
        logger.info(f"Stopped plugin {self.instance_id}")

    def notify_data_changed(self):
        """Notify that plugin data has changed (e.g., new traces ingested)."""
        if self._on_data_changed:
            try:
                self._on_data_changed(self)
            except Exception:
                pass

    @property
    def on_data_changed(self):
        """Callback invoked when plugin data changes."""
        return self._on_data_changed

    @on_data_changed.setter
    def on_data_changed(self, callback):
        self._on_data_changed = callback

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
            "webhooks": self.list_webhooks(),
        }

    # =========================================================================
    # Webhook Trace Forwarding
    # =========================================================================

    def subscribe_webhook(self, url: str) -> bool:
        """
        Subscribe a webhook URL to receive OTLP trace payloads.

        Args:
            url: HTTP(S) URL to POST OTLP JSON to

        Returns:
            True if newly added, False if already subscribed
        """
        if url in self._webhook_urls:
            return False
        self._webhook_urls.add(url)
        if self._webhook_executor is None:
            self._webhook_executor = ThreadPoolExecutor(max_workers=2)
        logger.info(f"Webhook subscribed: {url} (plugin={self.instance_id})")
        return True

    def unsubscribe_webhook(self, url: str) -> bool:
        """
        Unsubscribe a webhook URL.

        Args:
            url: URL to remove

        Returns:
            True if found and removed, False if not found
        """
        if url not in self._webhook_urls:
            return False
        self._webhook_urls.discard(url)
        logger.info(f"Webhook unsubscribed: {url} (plugin={self.instance_id})")
        return True

    def list_webhooks(self) -> List[str]:
        """Return list of subscribed webhook URLs."""
        return list(self._webhook_urls)

    def _deliver_to_webhooks(self, otlp_payload: dict):
        """
        Deliver an OTLP payload to all subscribed webhooks (fire-and-forget).

        Each delivery is submitted to a thread pool so it doesn't block
        the discovery loop.
        """
        if not self._webhook_urls:
            return
        span_count = sum(
            len(ss.get("spans", []))
            for rs in otlp_payload.get("resourceSpans", [])
            for ss in rs.get("scopeSpans", [])
        )
        logger.debug(
            "Delivering OTLP payload to %d webhook(s): %d resourceSpans, %d spans (plugin=%s)",
            len(self._webhook_urls),
            len(otlp_payload.get("resourceSpans", [])),
            span_count,
            self.instance_id,
        )
        if self._webhook_executor is None:
            self._webhook_executor = ThreadPoolExecutor(max_workers=2)
        for url in list(self._webhook_urls):
            self._webhook_executor.submit(self._send_webhook, url, otlp_payload)

    def _send_webhook(self, url: str, otlp_payload: dict):
        """POST an OTLP JSON payload to a webhook URL with retries."""
        # Mask token in debug output
        masked_url = url.split("?")[0] + ("?token=***" if "token=" in url else "")
        backoffs = [0.5, 1.0]
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    json=otlp_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                if resp.status_code < 400:
                    logger.debug(
                        "Webhook delivered successfully: %s status=%d (plugin=%s)",
                        masked_url,
                        resp.status_code,
                        self.instance_id,
                    )
                    return
                logger.warning(
                    "Webhook %s returned %d (attempt %d/3, plugin=%s)",
                    masked_url,
                    resp.status_code,
                    attempt + 1,
                    self.instance_id,
                )
            except Exception as e:
                logger.warning(
                    "Webhook %s failed (attempt %d/3, plugin=%s): %s",
                    masked_url,
                    attempt + 1,
                    self.instance_id,
                    e,
                )
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
