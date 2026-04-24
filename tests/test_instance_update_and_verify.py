"""
Tests for PUT /api/v1/instances/<instance_id> (update with config recreation
and rollback) and POST /api/v1/instances/<instance_id>/verify (success field
passthrough).
"""

import json
import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch

import open_cite.api.app as app_module
from open_cite.api.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_plugin(instance_id="test-inst", plugin_type="opentelemetry",
                      display_name="Test", status="running"):
    """Return a MagicMock that behaves like a BaseDiscoveryPlugin."""
    plugin = MagicMock()
    plugin.instance_id = instance_id
    plugin.plugin_type = plugin_type
    plugin.display_name = display_name
    plugin.status = status
    plugin.to_dict.return_value = {
        "instance_id": instance_id,
        "plugin_type": plugin_type,
        "display_name": display_name,
        "status": status,
    }
    return plugin


@pytest.fixture(scope="module")
def flask_app():
    """Create the Flask app once per module, resetting module-level state
    before and after so other test modules are not affected.

    Env vars are set inside the fixture (not at module level) to avoid
    overwriting values that other test files set during collection.
    """
    _tmpdir = tempfile.mkdtemp()
    _env_overrides = {
        "OPENCITE_PERSISTENCE_ENABLED": "false",
        "OPENCITE_AUTO_START": "false",
        "OPENCITE_ENABLE_OTEL": "false",
        "OPENCITE_ENABLE_MCP": "false",
        "OPENCITE_ENABLE_DATABRICKS": "false",
        "OPENCITE_ENABLE_GOOGLE_CLOUD": "false",
        "OPENCITE_DB_PATH": os.path.join(_tmpdir, "test_instance.db"),
    }
    _saved_env = {k: os.environ.get(k) for k in _env_overrides}
    os.environ.update(_env_overrides)

    # Reset state that a previously-collected test module may have set
    _saved_client = app_module.client
    _saved_plugin_store = app_module.plugin_store
    _saved_otel = app_module._default_otel_plugin
    app_module.client = None
    app_module.plugin_store = None
    app_module._default_otel_plugin = None

    app = create_app()
    app.config["TESTING"] = True
    yield app

    # Restore module-level state and env vars
    app_module.client = _saved_client
    app_module.plugin_store = _saved_plugin_store
    app_module._default_otel_plugin = _saved_otel
    for k, v in _saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture()
def test_app(flask_app):
    """Provide a test client with mocked shared state per test."""
    # Build a mock client with a real dict for .plugins
    mock_client = MagicMock()
    mock_client.plugins = {}

    mock_store = MagicMock()
    mock_store.enabled = True
    # Default: no persisted entries
    mock_store.load_all.return_value = []

    with patch.object(app_module, "client", mock_client), \
         patch.object(app_module, "plugin_store", mock_store):
        yield flask_app.test_client(), mock_client, mock_store


# ---------------------------------------------------------------------------
# PUT /api/v1/instances/<id> — display_name only (no recreation)
# ---------------------------------------------------------------------------

class TestUpdateInstanceDisplayNameOnly:
    def test_updates_display_name_in_place(self, test_app):
        flask_client, mock_client, mock_store = test_app

        plugin = _make_mock_plugin(status="running")
        mock_client.plugins["test-inst"] = plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "Old Name",
            "config": {"port": 4318},
            "auto_start": True,
        }]

        resp = flask_client.put(
            "/api/v1/instances/test-inst",
            data=json.dumps({"display_name": "New Name"}),
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        # display_name should be set directly on the plugin object
        assert plugin.display_name == "New Name"

    def test_persists_display_name_change(self, test_app):
        flask_client, mock_client, mock_store = test_app

        plugin = _make_mock_plugin(status="running")
        mock_client.plugins["test-inst"] = plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "Old Name",
            "config": {"port": 4318},
            "auto_start": True,
        }]

        flask_client.put(
            "/api/v1/instances/test-inst",
            data=json.dumps({"display_name": "New Name"}),
            content_type="application/json",
        )

        mock_store.save.assert_called_once()
        save_kwargs = mock_store.save.call_args[1]
        assert save_kwargs["display_name"] == "New Name"
        assert save_kwargs["config"] == {"port": 4318}


# ---------------------------------------------------------------------------
# PUT /api/v1/instances/<id> — config change triggers recreation
# ---------------------------------------------------------------------------

class TestUpdateInstanceWithConfigChange:
    def test_recreates_plugin_when_config_changes(self, test_app):
        flask_client, mock_client, mock_store = test_app

        old_plugin = _make_mock_plugin(status="running")
        mock_client.plugins["test-inst"] = old_plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "OTel",
            "config": {"port": 4318},
            "auto_start": True,
        }]

        new_plugin = _make_mock_plugin(status="stopped")
        new_plugin.to_dict.return_value = {
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "OTel",
            "status": "running",
        }

        with patch.object(app_module, "_create_plugin_instance", return_value=new_plugin) as mock_create, \
             patch.object(app_module, "_stop_plugin_instance") as mock_stop, \
             patch.object(app_module, "_start_plugin_instance") as mock_start:

            resp = flask_client.put(
                "/api/v1/instances/test-inst",
                data=json.dumps({"config": {"port": 9999}}),
                content_type="application/json",
            )

            assert resp.status_code == 200
            assert resp.get_json()["success"] is True

            # Old plugin should have been stopped and unregistered
            mock_stop.assert_called_once_with(old_plugin)
            mock_client.unregister_plugin.assert_called_once_with("test-inst")

            # New plugin should have been created with same instance_id and merged config
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert call_args[1] == "test-inst"  # same instance_id
            assert call_args[3]["port"] == 9999  # new value wins

            # Was running → should be restarted
            mock_start.assert_called_once_with(new_plugin)
            mock_client.register_plugin.assert_called_once_with(new_plugin)

    def test_does_not_restart_if_was_stopped(self, test_app):
        flask_client, mock_client, mock_store = test_app

        old_plugin = _make_mock_plugin(status="stopped")
        mock_client.plugins["test-inst"] = old_plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "OTel",
            "config": {},
            "auto_start": False,
        }]

        new_plugin = _make_mock_plugin(status="stopped")

        with patch.object(app_module, "_create_plugin_instance", return_value=new_plugin), \
             patch.object(app_module, "_stop_plugin_instance"), \
             patch.object(app_module, "_start_plugin_instance") as mock_start:

            resp = flask_client.put(
                "/api/v1/instances/test-inst",
                data=json.dumps({"config": {"key": "val"}}),
                content_type="application/json",
            )

            assert resp.status_code == 200
            # Was stopped → should NOT be restarted
            mock_start.assert_not_called()

    def test_persists_merged_config(self, test_app):
        flask_client, mock_client, mock_store = test_app

        plugin = _make_mock_plugin(status="stopped")
        mock_client.plugins["test-inst"] = plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "OTel",
            "config": {"host": "localhost", "port": 4318},
            "auto_start": False,
        }]

        new_plugin = _make_mock_plugin(status="stopped")

        with patch.object(app_module, "_create_plugin_instance", return_value=new_plugin), \
             patch.object(app_module, "_stop_plugin_instance"), \
             patch.object(app_module, "_start_plugin_instance"):

            flask_client.put(
                "/api/v1/instances/test-inst",
                data=json.dumps({"config": {"port": 9999}}),
                content_type="application/json",
            )

            # plugin_store.save should have been called with merged config
            mock_store.save.assert_called_once()
            save_kwargs = mock_store.save.call_args[1]
            assert save_kwargs["config"] == {"host": "localhost", "port": 9999}


# ---------------------------------------------------------------------------
# PUT /api/v1/instances/<id> — rollback on creation failure
# ---------------------------------------------------------------------------

class TestUpdateInstanceRollback:
    def test_rolls_back_on_create_failure(self, test_app):
        flask_client, mock_client, mock_store = test_app

        old_plugin = _make_mock_plugin(status="running")
        mock_client.plugins["test-inst"] = old_plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "OTel",
            "config": {"port": 4318},
            "auto_start": True,
        }]

        with patch.object(app_module, "_create_plugin_instance",
                          side_effect=ValueError("bad config")) as mock_create, \
             patch.object(app_module, "_stop_plugin_instance") as mock_stop, \
             patch.object(app_module, "_start_plugin_instance") as mock_start:

            resp = flask_client.put(
                "/api/v1/instances/test-inst",
                data=json.dumps({"config": {"port": "invalid"}}),
                content_type="application/json",
            )

            # Should return 500 (the outer except catches the re-raised error)
            assert resp.status_code == 500

            # Old plugin should have been stopped first (part of normal flow)
            mock_stop.assert_called_once_with(old_plugin)

            # Rollback: old plugin re-registered and restarted (was running)
            mock_client.register_plugin.assert_called_once_with(old_plugin)
            mock_start.assert_called_once_with(old_plugin)

    def test_rollback_does_not_persist_bad_config(self, test_app):
        flask_client, mock_client, mock_store = test_app

        old_plugin = _make_mock_plugin(status="running")
        mock_client.plugins["test-inst"] = old_plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "OTel",
            "config": {"port": 4318},
            "auto_start": True,
        }]

        with patch.object(app_module, "_create_plugin_instance",
                          side_effect=ValueError("bad config")), \
             patch.object(app_module, "_stop_plugin_instance"), \
             patch.object(app_module, "_start_plugin_instance"):

            flask_client.put(
                "/api/v1/instances/test-inst",
                data=json.dumps({"config": {"port": "invalid"}}),
                content_type="application/json",
            )

            # Store should NOT have been updated with the bad config
            mock_store.save.assert_not_called()

    def test_rollback_does_not_restart_if_was_stopped(self, test_app):
        flask_client, mock_client, mock_store = test_app

        old_plugin = _make_mock_plugin(status="stopped")
        mock_client.plugins["test-inst"] = old_plugin
        mock_store.load_all.return_value = [{
            "instance_id": "test-inst",
            "plugin_type": "opentelemetry",
            "display_name": "OTel",
            "config": {},
            "auto_start": False,
        }]

        with patch.object(app_module, "_create_plugin_instance",
                          side_effect=RuntimeError("boom")), \
             patch.object(app_module, "_stop_plugin_instance"), \
             patch.object(app_module, "_start_plugin_instance") as mock_start:

            resp = flask_client.put(
                "/api/v1/instances/test-inst",
                data=json.dumps({"config": {"x": 1}}),
                content_type="application/json",
            )

            assert resp.status_code == 500
            # Was stopped → rollback should NOT restart
            mock_start.assert_not_called()
            # But should still re-register
            mock_client.register_plugin.assert_called_once_with(old_plugin)


# ---------------------------------------------------------------------------
# POST /api/v1/instances/<id>/verify — success passthrough
# ---------------------------------------------------------------------------

class TestVerifyInstance:
    def test_verify_passes_through_success_true(self, test_app):
        flask_client, mock_client, _ = test_app

        plugin = _make_mock_plugin()
        plugin.verify_connection.return_value = {"success": True, "message": "OK"}
        mock_client.plugins["test-inst"] = plugin

        resp = flask_client.post("/api/v1/instances/test-inst/verify")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["verification"]["message"] == "OK"

    def test_verify_passes_through_success_false(self, test_app):
        flask_client, mock_client, _ = test_app

        plugin = _make_mock_plugin()
        plugin.verify_connection.return_value = {"success": False, "error": "bad creds"}
        mock_client.plugins["test-inst"] = plugin

        resp = flask_client.post("/api/v1/instances/test-inst/verify")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is False
        assert data["verification"]["error"] == "bad creds"

    def test_verify_defaults_to_true_when_key_missing(self, test_app):
        flask_client, mock_client, _ = test_app

        plugin = _make_mock_plugin()
        plugin.verify_connection.return_value = {"message": "connected"}
        mock_client.plugins["test-inst"] = plugin

        resp = flask_client.post("/api/v1/instances/test-inst/verify")

        data = resp.get_json()
        assert data["success"] is True

    def test_verify_defaults_to_true_for_non_dict(self, test_app):
        flask_client, mock_client, _ = test_app

        plugin = _make_mock_plugin()
        plugin.verify_connection.return_value = "ok"
        mock_client.plugins["test-inst"] = plugin

        resp = flask_client.post("/api/v1/instances/test-inst/verify")

        data = resp.get_json()
        assert data["success"] is True

    def test_verify_returns_404_for_unknown_instance(self, test_app):
        flask_client, _, _ = test_app

        resp = flask_client.post("/api/v1/instances/nonexistent/verify")

        assert resp.status_code == 404
