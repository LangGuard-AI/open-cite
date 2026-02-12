"""
OpenCITE Plugin Registry - Dynamic plugin discovery and factory.

Discovers all concrete BaseDiscoveryPlugin subclasses from the plugins package
and provides factory functions for creating instances.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, List, Optional, Type

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)

# Cache of discovered plugin classes: {plugin_type: PluginClass}
_plugin_classes: Optional[Dict[str, Type[BaseDiscoveryPlugin]]] = None


def discover_plugin_classes() -> Dict[str, Type[BaseDiscoveryPlugin]]:
    """
    Walk the open_cite.plugins package recursively and find all concrete
    BaseDiscoveryPlugin subclasses.

    Returns:
        Dict mapping plugin_type string to the plugin class.
    """
    global _plugin_classes
    if _plugin_classes is not None:
        return _plugin_classes

    import open_cite.plugins as plugins_pkg

    classes: Dict[str, Type[BaseDiscoveryPlugin]] = {}

    for importer, modname, ispkg in pkgutil.walk_packages(
        plugins_pkg.__path__,
        prefix="open_cite.plugins.",
    ):
        # Skip registry itself
        if modname == "open_cite.plugins.registry":
            continue

        try:
            module = importlib.import_module(modname)
        except Exception as e:
            logger.debug(f"Could not import plugin module {modname}: {e}")
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Must be a subclass of BaseDiscoveryPlugin but not the base itself
            if not issubclass(obj, BaseDiscoveryPlugin):
                continue
            if obj is BaseDiscoveryPlugin:
                continue

            # Must be defined in this module (skip re-exports)
            if obj.__module__ != module.__name__:
                continue

            # Must be concrete (not abstract)
            if inspect.isabstract(obj):
                continue

            # Get plugin_type - works for both class attribute and @property
            plugin_type = getattr(obj, 'plugin_type', None)

            # For classes where plugin_type is a property, we can't read it
            # from the class directly. Check if it's a property descriptor.
            if isinstance(inspect.getattr_static(obj, 'plugin_type', None), property):
                # Try to instantiate to get the value - skip if it fails
                # Instead, use a heuristic: check if the class has a known pattern
                try:
                    # Create a temporary instance just to read plugin_type
                    # This won't work for plugins requiring args, so fall back
                    # to module-based naming
                    plugin_type = None
                except Exception:
                    plugin_type = None

            if not plugin_type:
                # Fallback: derive from module name
                plugin_type = modname.rsplit('.', 1)[-1]
                logger.debug(
                    f"Plugin class {obj.__name__} has no plugin_type class attribute, "
                    f"using module name: {plugin_type}"
                )

            if plugin_type in classes:
                logger.warning(
                    f"Duplicate plugin_type '{plugin_type}': "
                    f"{classes[plugin_type].__name__} and {obj.__name__}"
                )
                continue

            classes[plugin_type] = obj
            logger.debug(f"Discovered plugin: {plugin_type} -> {obj.__name__}")

    _plugin_classes = classes
    logger.info(f"Discovered {len(classes)} plugins: {list(classes.keys())}")
    return classes


def get_all_plugin_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all discovered plugins.

    Returns:
        Dict mapping plugin_type to metadata dict.
    """
    classes = discover_plugin_classes()
    metadata = {}

    for plugin_type, cls in classes.items():
        try:
            meta = cls.plugin_metadata()
            meta['plugin_type'] = plugin_type
            meta['supports_multiple_instances'] = True
            metadata[plugin_type] = meta
        except Exception as e:
            logger.warning(f"Failed to get metadata for plugin {plugin_type}: {e}")

    return metadata


def create_plugin_instance(
    plugin_type: str,
    config: Optional[Dict[str, Any]] = None,
    instance_id: Optional[str] = None,
    display_name: Optional[str] = None,
    dependencies: Optional[Dict[str, Any]] = None,
) -> BaseDiscoveryPlugin:
    """
    Create a plugin instance by plugin_type.

    Args:
        plugin_type: The plugin type identifier
        config: Plugin-specific configuration dict
        instance_id: Optional unique instance ID
        display_name: Optional display name
        dependencies: Optional dependencies (e.g., mcp_plugin, http_client)

    Returns:
        New plugin instance

    Raises:
        ValueError: If plugin_type is unknown
    """
    classes = discover_plugin_classes()

    if plugin_type not in classes:
        raise ValueError(
            f"Unknown plugin type: {plugin_type}. "
            f"Available types: {list(classes.keys())}"
        )

    cls = classes[plugin_type]
    return cls.from_config(
        config=config or {},
        instance_id=instance_id,
        display_name=display_name,
        dependencies=dependencies,
    )


def reset_cache():
    """Reset the discovered plugin classes cache (for testing)."""
    global _plugin_classes
    _plugin_classes = None
