"""
Tool identity mapping — matches trace attributes to known tool identities.

When *persist* is True, mappings are stored in the ``asset_id_mappings`` table
via SQLAlchemy and loaded into memory on startup. The ``identify()`` hot path
is purely in-memory (no DB query).
"""

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolIdentifier:
    """
    Identifies tools based on attribute combinations from traces.

    Mappings are held in memory for fast lookup.  When *persist* is True,
    ``add_mapping`` also writes to the database so changes survive restarts.
    """

    def __init__(self, mapping_path: Optional[str] = None, persist: bool = True):
        """
        Args:
            mapping_path: Ignored (kept for backward compatibility).
            persist: When True, read/write mappings from/to the database.
        """
        self._persist = persist
        self._lock = threading.Lock()
        self.mapping: Dict[str, List[Dict[str, Any]]] = {}

        if self._persist:
            self._load_from_db()
            logger.info("Identity mapping persistence (SQLAlchemy) enabled")
        else:
            logger.info("Identity mapping persistence disabled — in-memory only")

    def _load_from_db(self):
        """Load all AssetIdMapping rows into self.mapping."""
        from open_cite.db import get_session, init_db, AssetIdMapping

        init_db()
        session = get_session()
        rows = session.query(AssetIdMapping).all()
        with self._lock:
            self.mapping = {}
            for row in rows:
                plugin = row.plugin_name
                if plugin not in self.mapping:
                    self.mapping[plugin] = []
                self.mapping[plugin].append({
                    "attributes": row.attributes,
                    "identity": row.identity,
                    "match_type": row.match_type or "all",
                })
        if rows:
            logger.info("Loaded %d identity mappings from database", len(rows))

    def identify(self, plugin_name: str, attributes: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Identify a tool based on plugin name and a dictionary of attributes.

        Returns a dictionary with 'source_name' and 'source_id' if matched, else None.
        Purely in-memory — no DB query on the hot path.
        """
        plugin_mappings = self.mapping.get(plugin_name, [])
        for entry in plugin_mappings:
            target_attrs = entry.get("attributes", {})
            if not target_attrs:
                continue

            match_type = entry.get("match_type", "all")

            if match_type == "any":
                match = False
                for key, expected_value in target_attrs.items():
                    if attributes.get(key) == expected_value:
                        match = True
                        break
            else:
                match = True
                for key, expected_value in target_attrs.items():
                    if attributes.get(key) != expected_value:
                        match = False
                        break

            if match:
                return entry.get("identity")

        return None

    def add_mapping(
        self,
        plugin_name: str,
        attributes: Dict[str, Any],
        identity: Dict[str, str],
        match_type: str = "all",
    ):
        """
        Add a new tool mapping.

        The mapping is always added in memory (so it takes effect immediately).
        It is only written to the database when persistence is enabled.
        """
        with self._lock:
            if plugin_name not in self.mapping:
                self.mapping[plugin_name] = []

            self.mapping[plugin_name].append({
                "attributes": attributes,
                "identity": identity,
                "match_type": match_type,
            })

        if not self._persist:
            return True

        try:
            from open_cite.db import get_session, AssetIdMapping

            session = get_session()
            session.add(AssetIdMapping(
                plugin_name=plugin_name,
                attributes=attributes,
                identity=identity,
                match_type=match_type,
                created_at=datetime.utcnow().isoformat(),
            ))
            session.commit()
            logger.info("Saved new identity mapping for plugin '%s'", plugin_name)
            return True
        except Exception as e:
            logger.error("Failed to save identity mapping: %s", e)
            return False
