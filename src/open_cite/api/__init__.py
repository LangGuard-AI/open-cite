"""
OpenCITE Headless API Module.

Provides a REST API for OpenCITE discovery and inventory capabilities,
designed for deployment in Kubernetes without a GUI.
"""

from .app import create_app, run_api
from .config import OpenCiteConfig
from .health import health_bp, init_health
from .shutdown import register_shutdown_handler
from .persistence import PersistenceManager

__all__ = [
    "create_app",
    "run_api",
    "OpenCiteConfig",
    "health_bp",
    "init_health",
    "register_shutdown_handler",
    "PersistenceManager",
]
