"""Databricks App entry point for Open-CITE."""
import os
from open_cite.gui.app import run_gui

if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", "5000"))
    run_gui(host="0.0.0.0", port=port, debug=False)
