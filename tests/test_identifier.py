import unittest
import os
import json
import sys
from unittest.mock import MagicMock

# Mock dependencies that might be imported by the package
sys.modules["mlflow"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.aiplatform"] = MagicMock()
sys.modules["google.api_core"] = MagicMock()
sys.modules["databricks"] = MagicMock()
sys.modules["databricks.sdk"] = MagicMock()
sys.modules["open_cite.client"] = MagicMock()

# Add src to sys.path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_cite.identifier import ToolIdentifier

class TestToolIdentifier(unittest.TestCase):
    def setUp(self):
        # Create a temporary mapping file for testing
        self.test_mapping = {
            "test_plugin": [
                {
                    "attributes": {
                        "attr1": "val1",
                        "attr2": "val2"
                    },
                    "identity": {
                        "source_name": "Test Source",
                        "source_id": "ts-001"
                    }
                },
                {
                    "attributes": {
                        "attr1": "val3"
                    },
                    "identity": {
                        "source_name": "Partial Match Source",
                        "source_id": "ts-002"
                    }
                }
            ]
        }
        self.mapping_path = "test_tool_mapping.json"
        with open(self.mapping_path, "w") as f:
            json.dump(self.test_mapping, f)
        
        self.identifier = ToolIdentifier(mapping_path=self.mapping_path)

    def tearDown(self):
        if os.path.exists(self.mapping_path):
            os.remove(self.mapping_path)

    def test_exact_match(self):
        attrs = {"attr1": "val1", "attr2": "val2", "other": "random"}
        result = self.identifier.identify("test_plugin", attrs)
        self.assertIsNotNone(result)
        self.assertEqual(result["source_name"], "Test Source")
        self.assertEqual(result["source_id"], "ts-001")

    def test_partial_match(self):
        # Match against the second entry which only requires attr1
        attrs = {"attr1": "val3", "something": "else"}
        result = self.identifier.identify("test_plugin", attrs)
        self.assertIsNotNone(result)
        self.assertEqual(result["source_name"], "Partial Match Source")

    def test_no_match(self):
        attrs = {"attr1": "wrong"}
        result = self.identifier.identify("test_plugin", attrs)
        self.assertIsNone(result)

    def test_wrong_plugin(self):
        attrs = {"attr1": "val1", "attr2": "val2"}
        result = self.identifier.identify("other_plugin", attrs)
        self.assertIsNone(result)

    def test_add_mapping(self):
        # Add a new mapping
        plugin = "new_plugin"
        attrs = {"id": "123"}
        identity = {"source_name": "New Tool", "source_id": "nt-001"}
        
        success = self.identifier.add_mapping(plugin, attrs, identity)
        self.assertTrue(success)
        
        # Verify it can be identified immediately
        match = self.identifier.identify(plugin, attrs)
        self.assertIsNotNone(match)
        self.assertEqual(match["source_name"], "New Tool")
        
        # Verify it was persisted by reloading
        new_identifier = ToolIdentifier(mapping_path=self.mapping_path)
        match_reloaded = new_identifier.identify(plugin, attrs)
        self.assertIsNotNone(match_reloaded)
        self.assertEqual(match_reloaded["source_name"], "New Tool")

    def test_hot_reload(self):
        # Initial identification
        plugin = "test_plugin"
        attrs = {"attr1": "val1", "attr2": "val2"}
        match = self.identifier.identify(plugin, attrs)
        self.assertIsNotNone(match)
        self.assertEqual(match["source_name"], "Test Source")
        
        # Modify the file directly on filesystem
        with open(self.mapping_path, "r") as f:
            data = json.load(f)
        
        data["test_plugin"][0]["identity"]["source_name"] = "Modified Name"
        
        # We need to wait a tiny bit to ensure mtime changes if the filesystem resolution is low,
        # but in most modern systems it should be fine. To be safe we can force it.
        import time
        time.sleep(0.01) 
        
        with open(self.mapping_path, "w") as f:
            json.dump(data, f)
        
        # Verify identification reflects the change automatically
        match_after = self.identifier.identify(plugin, attrs)
        self.assertIsNotNone(match_after)
        self.assertEqual(match_after["source_name"], "Modified Name")

if __name__ == "__main__":
    # Mocking external imports in open_cite if any (in this case identifier.py is isolated)
    unittest.main()
