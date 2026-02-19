"""
JSON-based plugin configuration persistence.

Stores plugin instance configurations to a JSON file so they survive restarts.
Enabled by default; disabled when running as a Kubernetes sidecar.
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "resources" / "plugin_instances.json"


class PluginConfigStore:
    """Thread-safe JSON store for plugin instance configurations."""

    def __init__(self, path: Optional[str] = None, enabled: Optional[bool] = None):
        """
        Args:
            path: Path to JSON file. Defaults to resources/plugin_instances.json.
            enabled: Explicitly enable/disable. Defaults to True unless running
                     in Kubernetes (KUBERNETES_SERVICE_HOST is set). Can be
                     overridden via OPENCITE_PERSIST_PLUGINS env var.
        """
        self._path = Path(path) if path else _DEFAULT_PATH
        self._lock = threading.Lock()

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
            self._path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Plugin config persistence (JSON) enabled: {self._path}")
        else:
            logger.info("Plugin config persistence (JSON) disabled")

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

        with self._lock:
            data = self._read()
            now = datetime.utcnow().isoformat()
            existing = data["instances"].get(instance_id)
            data["instances"][instance_id] = {
                "plugin_type": plugin_type,
                "display_name": display_name,
                "config": config,
                "auto_start": auto_start,
                "created_at": existing["created_at"] if existing else now,
                "updated_at": now,
            }
            self._write(data)
            logger.debug(f"Saved plugin instance: {instance_id}")

    def delete(self, instance_id: str) -> None:
        """Remove a plugin instance configuration."""
        if not self._enabled:
            return

        with self._lock:
            data = self._read()
            if instance_id in data["instances"]:
                del data["instances"][instance_id]
                self._write(data)
                logger.info(f"Deleted plugin instance: {instance_id}")

    def load_all(self) -> List[Dict[str, Any]]:
        """Return all saved instance configs as a list of dicts.

        Each dict contains: instance_id, plugin_type, display_name, config,
        auto_start, created_at, updated_at.
        """
        if not self._enabled:
            return []

        with self._lock:
            data = self._read()

        return [
            {"instance_id": iid, **entry}
            for iid, entry in data["instances"].items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> Dict[str, Any]:
        """Read the JSON file (caller must hold self._lock)."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read {self._path}: {e}")
        return {"instances": {}}

    def _write(self, data: Dict[str, Any]) -> None:
        """Write the JSON file (caller must hold self._lock)."""
        try:
            self._path.write_text(
                json.dumps(data, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to write {self._path}: {e}")
