"""
SQLAlchemy-based plugin configuration persistence.

Stores plugin instance configurations to the ``plugin_configs`` table so they
survive restarts.  Enabled by default; disabled when running as a Kubernetes
sidecar.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from open_cite.db import get_session, init_db, PluginConfig

logger = logging.getLogger(__name__)


class PluginConfigStore:
    """Thread-safe store for plugin instance configurations (SQLAlchemy)."""

    def __init__(self, path: Optional[str] = None, enabled: Optional[bool] = None):
        """
        Args:
            path: Ignored (kept for backward compatibility).
            enabled: Explicitly enable/disable. Defaults to True unless running
                     in Kubernetes (KUBERNETES_SERVICE_HOST is set). Can be
                     overridden via OPENCITE_PERSIST_PLUGINS env var.
        """
        if enabled is not None:
            self._enabled = enabled
        else:
            env_override = os.getenv("OPENCITE_PERSIST_PLUGINS")
            if env_override is not None:
                self._enabled = env_override.lower() == "true"
            elif os.getenv("KUBERNETES_SERVICE_HOST"):
                self._enabled = False
            else:
                self._enabled = True

        if self._enabled:
            init_db()
            logger.info("Plugin config persistence (SQLAlchemy) enabled")
        else:
            logger.info("Plugin config persistence disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def save(
        self,
        instance_id: str,
        plugin_type: str,
        display_name: str,
        config: Dict[str, Any],
        auto_start: bool = False,
    ) -> None:
        """Save or update a plugin instance configuration."""
        if not self._enabled:
            return

        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(PluginConfig, instance_id)
            if existing:
                existing.plugin_type = plugin_type
                existing.display_name = display_name
                existing.config = config
                existing.auto_start = auto_start
                existing.updated_at = now
            else:
                session.add(PluginConfig(
                    instance_id=instance_id,
                    plugin_type=plugin_type,
                    display_name=display_name,
                    config=config,
                    auto_start=auto_start,
                    created_at=now,
                    updated_at=now,
                ))
            session.commit()
            logger.debug("Saved plugin instance: %s", instance_id)
        except Exception:
            session.rollback()
            raise

    def delete(self, instance_id: str) -> None:
        """Remove a plugin instance configuration."""
        if not self._enabled:
            return

        session = get_session()
        try:
            row = session.get(PluginConfig, instance_id)
            if row:
                session.delete(row)
                session.commit()
                logger.info("Deleted plugin instance: %s", instance_id)
        except Exception:
            session.rollback()
            raise

    def load_all(self) -> List[Dict[str, Any]]:
        """Return all saved instance configs as a list of dicts.

        Each dict contains: instance_id, plugin_type, display_name, config,
        auto_start, created_at, updated_at.
        """
        if not self._enabled:
            return []

        session = get_session()
        rows = session.query(PluginConfig).all()
        return [
            {
                "instance_id": r.instance_id,
                "plugin_type": r.plugin_type,
                "display_name": r.display_name,
                "config": r.config,
                "auto_start": r.auto_start,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]
