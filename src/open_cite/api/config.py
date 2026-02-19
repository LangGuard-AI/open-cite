"""
OpenCITE API Configuration.

Environment-based configuration for the headless API service.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _resolve_persist_default(env_var: str) -> bool:
    """Resolve a persistence toggle: True unless in Kubernetes, overridable by *env_var*."""
    env = os.getenv(env_var)
    if env is not None:
        return env.lower() == "true"
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return False
    return True


@dataclass
class OpenCiteConfig:
    """
    Configuration for OpenCITE API service.

    All settings can be configured via environment variables.
    """

    # API Server settings
    host: str = field(default_factory=lambda: os.getenv("OPENCITE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("OPENCITE_PORT", "8080")))

    # OTLP Receiver settings
    otlp_host: str = field(default_factory=lambda: os.getenv("OPENCITE_OTLP_HOST", "0.0.0.0"))
    otlp_port: int = field(default_factory=lambda: int(os.getenv("OPENCITE_OTLP_PORT", "4318")))

    # Plugin toggles
    enable_otel: bool = field(default_factory=lambda: os.getenv("OPENCITE_ENABLE_OTEL", "true").lower() == "true")
    enable_mcp: bool = field(default_factory=lambda: os.getenv("OPENCITE_ENABLE_MCP", "true").lower() == "true")
    enable_databricks: bool = field(default_factory=lambda: os.getenv("OPENCITE_ENABLE_DATABRICKS", "false").lower() == "true")
    enable_google_cloud: bool = field(default_factory=lambda: os.getenv("OPENCITE_ENABLE_GOOGLE_CLOUD", "false").lower() == "true")

    # Databricks settings (passed through to plugin)
    databricks_host: Optional[str] = field(default_factory=lambda: os.getenv("DATABRICKS_HOST"))
    databricks_token: Optional[str] = field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN"))
    databricks_warehouse_id: Optional[str] = field(default_factory=lambda: os.getenv("DATABRICKS_WAREHOUSE_ID"))

    # Google Cloud settings (passed through to plugin)
    gcp_project_id: Optional[str] = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID"))
    gcp_location: str = field(default_factory=lambda: os.getenv("GCP_LOCATION", "us-central1"))

    # Auto-start discovery on startup
    auto_start: bool = field(default_factory=lambda: os.getenv("OPENCITE_AUTO_START", "true").lower() == "true")

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("OPENCITE_LOG_LEVEL", "INFO"))

    # Persistence (SQLite for discovered assets)
    persistence_enabled: bool = field(
        default_factory=lambda: os.getenv("OPENCITE_PERSISTENCE_ENABLED", "false").lower() == "true"
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("OPENCITE_DB_PATH", "/data/opencite.db")
    )

    # Plugin config persistence (JSON file)
    persist_plugins: bool = field(
        default_factory=lambda: _resolve_persist_default("OPENCITE_PERSIST_PLUGINS")
    )
    plugin_store_path: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENCITE_PLUGIN_STORE_PATH")
    )

    # Identity mapping persistence (JSON file)
    persist_mappings: bool = field(
        default_factory=lambda: _resolve_persist_default("OPENCITE_PERSIST_MAPPINGS")
    )
    mapping_store_path: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENCITE_MAPPING_STORE_PATH")
    )

    @classmethod
    def from_env(cls) -> "OpenCiteConfig":
        """Create configuration from environment variables."""
        return cls()

    def get_enabled_plugins(self) -> list:
        """Return list of enabled plugin names based on configuration."""
        plugins = []

        # MCP should be registered first (OpenTelemetry integrates with it)
        if self.enable_mcp:
            plugins.append({"name": "mcp", "config": {}})

        if self.enable_otel:
            plugins.append({
                "name": "opentelemetry",
                "config": {
                    "host": self.otlp_host,
                    "port": self.otlp_port,
                }
            })

        if self.enable_databricks:
            plugins.append({
                "name": "databricks",
                "config": {
                    "host": self.databricks_host,
                    "token": self.databricks_token,
                    "warehouse_id": self.databricks_warehouse_id,
                }
            })

        if self.enable_google_cloud:
            plugins.append({
                "name": "google_cloud",
                "config": {
                    "project_id": self.gcp_project_id,
                    "location": self.gcp_location,
                }
            })

        return plugins
