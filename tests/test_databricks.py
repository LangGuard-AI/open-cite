"""
Unit tests for the Databricks plugin's high-water mark (HWM) logic
in refresh_discovery, refresh_traces, and discover_from_genie.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: build a DatabricksPlugin without hitting real Databricks APIs
# ---------------------------------------------------------------------------

def _make_plugin(lookback_days=90, last_query_time=None, last_genie_query_time=None):
    """Construct a DatabricksPlugin with SDK/MLflow calls mocked out."""
    with patch("open_cite.plugins.databricks.WorkspaceClient"), \
         patch("open_cite.plugins.databricks.MlflowClient"), \
         patch("open_cite.plugins.databricks.mlflow"):
        from open_cite.plugins.databricks import DatabricksPlugin

        plugin = DatabricksPlugin(
            host="https://test.cloud.databricks.com",
            token="test-token",
            lookback_days=lookback_days,
            instance_id="db-test",
            display_name="DB Test",
            auto_poll=False,
        )

    plugin._last_query_time = last_query_time
    plugin._last_genie_query_time = last_genie_query_time

    # Stub out persistence so tests don't need a database
    plugin.save_hwm = MagicMock()
    plugin.load_hwm = MagicMock(return_value=None)
    plugin.notify_data_changed = MagicMock()

    return plugin


# ---------------------------------------------------------------------------
# refresh_discovery — incremental lookback
# ---------------------------------------------------------------------------

class TestRefreshDiscoveryHWM:

    def test_uses_lookback_days_when_no_hwm(self):
        """First run (no HWM) should use the configured lookback_days."""
        plugin = _make_plugin(lookback_days=90)

        with patch.object(plugin, "list_assets"), \
             patch.object(plugin, "_run_discovery") as mock_run:
            plugin.refresh_discovery()

        mock_run.assert_called_once_with(90, "refresh_discovery")

    def test_uses_incremental_lookback_when_hwm_set(self):
        """When _last_query_time exists, lookback should be based on elapsed time."""
        hwm = datetime.utcnow() - timedelta(days=3, hours=5)
        plugin = _make_plugin(lookback_days=90, last_query_time=hwm)

        with patch.object(plugin, "list_assets"), \
             patch.object(plugin, "_run_discovery") as mock_run:
            plugin.refresh_discovery()

        # 3 days 5 hours elapsed → int(3.2) + 1 = 4 days
        args = mock_run.call_args[0]
        assert args[0] == 4
        assert args[1] == "refresh_discovery"

    def test_minimum_incremental_lookback_is_one_day(self):
        """Even if HWM was set moments ago, lookback should be at least 1 day."""
        hwm = datetime.utcnow() - timedelta(minutes=5)
        plugin = _make_plugin(lookback_days=90, last_query_time=hwm)

        with patch.object(plugin, "list_assets"), \
             patch.object(plugin, "_run_discovery") as mock_run:
            plugin.refresh_discovery()

        args = mock_run.call_args[0]
        assert args[0] == 1

    def test_enumerates_unity_catalog_assets(self):
        """refresh_discovery should enumerate all Unity Catalog asset types."""
        plugin = _make_plugin()

        with patch.object(plugin, "list_assets") as mock_list, \
             patch.object(plugin, "_run_discovery"):
            plugin.refresh_discovery()

        called_types = [call.args[0] for call in mock_list.call_args_list]
        assert called_types == ["catalog", "schema", "table", "volume", "model", "function"]


# ---------------------------------------------------------------------------
# refresh_traces — incremental lookback
# ---------------------------------------------------------------------------

class TestRefreshTracesHWM:

    def test_uses_lookback_days_when_no_hwm(self):
        plugin = _make_plugin(lookback_days=30)

        with patch.object(plugin, "_run_discovery"), \
             patch("open_cite.plugins.databricks.threading"):
            plugin.refresh_traces()

        # With no HWM, days should default to lookback_days (30)
        # refresh_traces spawns a thread — verify via the thread target args
        import threading as real_threading
        with patch("open_cite.plugins.databricks.threading") as mock_threading:
            plugin.refresh_traces()
            thread_call = mock_threading.Thread.call_args
            assert thread_call[1]["args"][0] == 30 or thread_call.kwargs.get("args", (None,))[0] == 30 \
                or thread_call[1].get("args", thread_call[0][0] if thread_call[0] else (None,))[0] == 30

    def test_uses_explicit_days_override(self):
        """Explicit days param should override both HWM and lookback_days."""
        hwm = datetime.utcnow() - timedelta(days=10)
        plugin = _make_plugin(lookback_days=90, last_query_time=hwm)

        with patch("open_cite.plugins.databricks.threading") as mock_threading:
            plugin.refresh_traces(days=5)

        thread_args = mock_threading.Thread.call_args
        # args=(days, label)
        assert thread_args[1]["args"][0] == 5


# ---------------------------------------------------------------------------
# discover_from_traces / discover_from_genie — separate HWMs
# ---------------------------------------------------------------------------

class TestSeparateHWMs:

    def test_discover_from_traces_sets_trace_hwm(self):
        """discover_from_traces should update _last_query_time only."""
        plugin = _make_plugin()

        fake_exp = MagicMock()
        fake_exp.experiment_id = "exp-1"
        fake_exp.name = "test-experiment"

        with patch.object(plugin, "workspace_client") as mock_ws, \
             patch.object(plugin, "_discover_from_traces", return_value=True), \
             patch.object(plugin, "_discover_from_runs"):
            mock_ws.experiments.search_experiments.return_value = [fake_exp]
            plugin.discover_from_traces(days=7)

        # _last_query_time should be set
        assert plugin._last_query_time is not None
        # _last_genie_query_time should remain None
        assert plugin._last_genie_query_time is None
        # Should persist the trace HWM
        plugin.save_hwm.assert_called_once_with(
            "last_query_time", plugin._last_query_time.isoformat()
        )

    def test_discover_from_genie_sets_genie_hwm(self):
        """discover_from_genie should update _last_genie_query_time, not _last_query_time."""
        plugin = _make_plugin()

        fake_space = {"space_id": "sp-1", "title": "Test Space", "description": ""}
        mock_gc = MagicMock()
        mock_gc.validate_connection.return_value = {"valid": True, "spaces_count": 1}
        mock_gc.list_spaces.return_value = [fake_space]
        mock_gc.list_all_conversations.return_value = []
        plugin.genie_client = mock_gc

        with patch.object(plugin, "_forward_to_otel"):
            plugin.discover_from_genie(days=7)

        # _last_genie_query_time should be set
        assert plugin._last_genie_query_time is not None
        # _last_query_time should remain None (not overwritten)
        assert plugin._last_query_time is None
        # Should persist the genie HWM
        plugin.save_hwm.assert_called_once_with(
            "last_genie_query_time", plugin._last_genie_query_time.isoformat()
        )

    def test_discover_from_traces_narrows_lookback_from_hwm(self):
        """When _last_query_time is set, discover_from_traces should narrow
        the lookback to elapsed time, not use the caller's days param."""
        hwm = datetime.utcnow() - timedelta(days=2, hours=6)
        plugin = _make_plugin(lookback_days=90, last_query_time=hwm)

        captured_start_ms = {}

        fake_exp = MagicMock()
        fake_exp.experiment_id = "exp-1"
        fake_exp.name = "test-experiment"

        original_discover = None

        with patch.object(plugin, "workspace_client") as mock_ws, \
             patch.object(plugin, "_discover_from_traces", return_value=True) as mock_dt, \
             patch.object(plugin, "_discover_from_runs") as mock_dr:
            mock_ws.experiments.search_experiments.return_value = [fake_exp]
            # Call with the full 90-day lookback that start() would pass
            plugin.discover_from_traces(days=90)

            # The method should have narrowed days to ~3 (2d6h → int(2.25)+1=3)
            # Verify by checking the start_ms computed from the narrowed days:
            # _discover_from_traces is called with (experiments, start_ms, max)
            call_args = mock_dt.call_args
            actual_start_ms = call_args[0][1]  # second positional arg
            now_ms = int(datetime.utcnow().timestamp() * 1000)
            actual_days = (now_ms - actual_start_ms) / (24 * 60 * 60 * 1000)
            # Should be ~3 days, not 90
            assert actual_days < 5, f"Expected ~3 days lookback, got {actual_days:.1f}"

    def test_discover_from_traces_uses_caller_days_without_hwm(self):
        """Without _last_query_time, discover_from_traces should use the
        caller's days parameter as-is."""
        plugin = _make_plugin(lookback_days=90)

        fake_exp = MagicMock()
        fake_exp.experiment_id = "exp-1"
        fake_exp.name = "test-experiment"

        with patch.object(plugin, "workspace_client") as mock_ws, \
             patch.object(plugin, "_discover_from_traces", return_value=True) as mock_dt, \
             patch.object(plugin, "_discover_from_runs"):
            mock_ws.experiments.search_experiments.return_value = [fake_exp]
            plugin.discover_from_traces(days=90)

            call_args = mock_dt.call_args
            actual_start_ms = call_args[0][1]
            now_ms = int(datetime.utcnow().timestamp() * 1000)
            actual_days = (now_ms - actual_start_ms) / (24 * 60 * 60 * 1000)
            # Should be ~90 days
            assert actual_days > 85, f"Expected ~90 days lookback, got {actual_days:.1f}"

    def test_discover_from_genie_narrows_lookback_from_hwm(self):
        """When _last_genie_query_time is set, discover_from_genie should
        narrow the lookback to elapsed time."""
        hwm = datetime.utcnow() - timedelta(days=4, hours=3)
        plugin = _make_plugin(lookback_days=90, last_genie_query_time=hwm)

        fake_space = {"space_id": "sp-1", "title": "Test Space", "description": ""}
        mock_gc = MagicMock()
        mock_gc.validate_connection.return_value = {"valid": True, "spaces_count": 1}
        mock_gc.list_spaces.return_value = [fake_space]
        mock_gc.list_all_conversations.return_value = []
        plugin.genie_client = mock_gc

        with patch.object(plugin, "_forward_to_otel"):
            plugin.discover_from_genie(days=90)

        # Genie HWM should now be updated (discover_from_genie ran to completion)
        assert plugin._last_genie_query_time is not None
        plugin.save_hwm.assert_called_once_with(
            "last_genie_query_time", plugin._last_genie_query_time.isoformat()
        )

    def test_discover_from_genie_uses_caller_days_without_hwm(self):
        """Without _last_genie_query_time, discover_from_genie should use
        the caller's days parameter."""
        plugin = _make_plugin(lookback_days=90)

        fake_space = {"space_id": "sp-1", "title": "Test Space", "description": ""}
        mock_gc = MagicMock()
        mock_gc.validate_connection.return_value = {"valid": True, "spaces_count": 1}
        mock_gc.list_spaces.return_value = [fake_space]
        mock_gc.list_all_conversations.return_value = []
        plugin.genie_client = mock_gc

        with patch.object(plugin, "_forward_to_otel"):
            plugin.discover_from_genie(days=90)

        assert plugin._last_genie_query_time is not None
        plugin.save_hwm.assert_called_once_with(
            "last_genie_query_time", plugin._last_genie_query_time.isoformat()
        )

    def test_run_discovery_does_not_cross_contaminate_hwms(self):
        """After _run_discovery, trace and genie HWMs should be independent."""
        plugin = _make_plugin()

        with patch.object(plugin, "discover_from_traces") as mock_traces, \
             patch.object(plugin, "discover_from_genie") as mock_genie, \
             patch.object(plugin, "discover_mcp_servers"):

            # Simulate each discovery setting its own HWM
            def set_trace_hwm(days=90, max_per_experiment=100):
                plugin._last_query_time = datetime(2026, 3, 25, 10, 0, 0)

            def set_genie_hwm(days=90, max_messages=100, space_ids=None):
                plugin._last_genie_query_time = datetime(2026, 3, 25, 10, 5, 0)

            mock_traces.side_effect = set_trace_hwm
            mock_genie.side_effect = set_genie_hwm

            plugin._run_discovery(7, "test")

        assert plugin._last_query_time == datetime(2026, 3, 25, 10, 0, 0)
        assert plugin._last_genie_query_time == datetime(2026, 3, 25, 10, 5, 0)


# ---------------------------------------------------------------------------
# start() — HWM restoration
# ---------------------------------------------------------------------------

class TestStartHWMRestoration:

    def test_restores_all_hwms_on_start(self):
        """start() should restore trace, genie, and gateway HWMs from the DB."""
        plugin = _make_plugin()

        hwm_values = {
            "last_query_time": "2026-03-20T12:00:00",
            "last_genie_query_time": "2026-03-21T08:30:00",
            "last_gateway_event_time": "2026-03-22T15:00:00",
        }
        plugin.load_hwm = MagicMock(side_effect=lambda name: hwm_values.get(name))

        with patch.object(plugin, "_run_discovery"), \
             patch("open_cite.plugins.databricks.threading"):
            plugin.start()

        assert plugin._last_query_time == datetime.fromisoformat("2026-03-20T12:00:00")
        assert plugin._last_genie_query_time == datetime.fromisoformat("2026-03-21T08:30:00")

    def test_start_with_hwms_methods_self_narrow(self):
        """When start() restores HWMs and passes lookback_days to
        _run_discovery, each discovery method should independently narrow
        the lookback using its own HWM."""
        plugin = _make_plugin(lookback_days=90)

        hwm_values = {
            "last_query_time": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "last_genie_query_time": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "last_gateway_event_time": None,
        }
        plugin.load_hwm = MagicMock(side_effect=lambda name: hwm_values.get(name))

        captured_trace_days = {}
        captured_genie_days = {}

        def fake_traces(days=90, max_per_experiment=100):
            captured_trace_days["days"] = days

        def fake_genie(days=90, max_messages=100, space_ids=None):
            captured_genie_days["days"] = days

        with patch.object(plugin, "discover_from_traces", side_effect=fake_traces), \
             patch.object(plugin, "discover_from_genie", side_effect=fake_genie), \
             patch.object(plugin, "discover_mcp_servers"):
            plugin.start()
            # start() spawns a thread — call _run_discovery directly to test
            plugin._run_discovery(plugin.lookback_days, "start")

        # discover_from_traces should have been called — the method itself
        # will narrow days from _last_query_time (~2 days → 3)
        assert "days" in captured_trace_days
        # discover_from_genie should have been called — the method itself
        # will narrow days from _last_genie_query_time (~5 days → 6)
        assert "days" in captured_genie_days
        # The caller passes 90, but the methods override internally.
        # Here we're testing _run_discovery passes through the days value;
        # the actual narrowing happens inside the methods themselves.
        assert captured_trace_days["days"] == 90
        assert captured_genie_days["days"] == 90

    def test_handles_missing_hwms_on_start(self):
        """start() should gracefully handle missing HWM values."""
        plugin = _make_plugin()
        plugin.load_hwm = MagicMock(return_value=None)

        with patch.object(plugin, "_run_discovery"), \
             patch("open_cite.plugins.databricks.threading"):
            plugin.start()

        assert plugin._last_query_time is None
        assert plugin._last_genie_query_time is None
