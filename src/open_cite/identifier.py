import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ToolIdentifier:
    """
    Identifies tools based on attribute combinations from traces.
    Uses a mapping JSON file to match attributes to identities.
    Supports hot-reloading if the file is edited on the filesystem.
    """

    def __init__(self, mapping_path: Optional[str] = None):
        if not mapping_path:
            # Default to the one in resources
            mapping_path = str(Path(__file__).parent / "resources" / "tool_mapping.json")
        
        self.mapping_path = mapping_path
        self._lock = threading.Lock()
        self.last_mtime = 0
        self.mapping = {}
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Check if the mapping file has changed and reload if necessary."""
        try:
            current_mtime = os.path.getmtime(self.mapping_path)
            if current_mtime > self.last_mtime:
                with self._lock:
                    # Double check under lock
                    if current_mtime > self.last_mtime:
                        self.mapping = self._load_mapping()
                        self.last_mtime = current_mtime
                        logger.info(f"Reloaded tool mapping from {self.mapping_path}")
        except Exception as e:
            # If file doesn't exist yet or other error, just log it once if it was previously loaded
            if self.last_mtime > 0:
                logger.error(f"Error checking tool mapping mtime: {e}")

    def _load_mapping(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.mapping_path):
                return {}
            with open(self.mapping_path, 'r') as f:
                # Basic cleanup in case of comments or other issues
                content = f.read()
                # Simple comment removal if present (naive)
                lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
                return json.loads("\n".join(lines))
        except Exception as e:
            logger.error(f"Failed to load tool mapping from {self.mapping_path}: {e}")
            return {}

    def identify(self, plugin_name: str, attributes: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Identify a tool based on plugin name and a dictionary of attributes.
        
        Returns a dictionary with 'source_name' and 'source_id' if matched, else None.
        """
        self._ensure_loaded()
        plugin_mappings = self.mapping.get(plugin_name, [])
        for entry in plugin_mappings:
            target_attrs = entry.get("attributes", {})
            if not target_attrs:
                continue
            
            match_type = entry.get("match_type", "all")
            
            if match_type == "any":
                # Check if at least one target attribute matches the provided attributes
                match = False
                for key, expected_value in target_attrs.items():
                    if attributes.get(key) == expected_value:
                        match = True
                        break
            else:
                # Default to "all" - Check if all target attributes match the provided attributes
                match = True
                for key, expected_value in target_attrs.items():
                    if attributes.get(key) != expected_value:
                        match = False
                        break
            
            if match:
                return entry.get("identity")
        
        return None

    def add_mapping(self, plugin_name: str, attributes: Dict[str, Any], identity: Dict[str, str], match_type: str = "all"):
        """
        Add a new tool mapping and save it to the mapping file.
        """
        if plugin_name not in self.mapping:
            self.mapping[plugin_name] = []
        
        # Add the new mapping entry
        self.mapping[plugin_name].append({
            "attributes": attributes,
            "identity": identity,
            "match_type": match_type
        })
        
        # Persist to file
        try:
            with self._lock:
                with open(self.mapping_path, 'w') as f:
                    json.dump(self.mapping, f, indent=4)
                # Update mtime so we don't immediately reload what we just saved
                try:
                    self.last_mtime = os.path.getmtime(self.mapping_path)
                except:
                    pass 
            logger.info(f"Successfully saved new tool mapping to {self.mapping_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save tool mapping to {self.mapping_path}: {e}")
            return False
