"""
Tests for memory eviction in the OpenTelemetry plugin.

Verifies that:
- Trace eviction uses LRU (last_seen) semantics
- Trace eviction is proportional per integration (no noisy-neighbor starvation)
- Session user-attr eviction is proportional per integration
- Cross-batch span cache eviction fires when threshold is exceeded
- Per-tool trace list trimming keeps only the most recent 100 entries
"""

import uuid
from unittest.mock import MagicMock

import pytest

from open_cite.plugins.opentelemetry import OpenTelemetryPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plugin(**overrides):
    """Build an OpenTelemetryPlugin without starting a receiver."""
    plugin = OpenTelemetryPlugin(
        host="127.0.0.1",
        port=0,
        instance_id="opentelemetry",
        display_name="OTLP",
        persist_mappings=False,
        embedded_receiver=True,
    )
    plugin.notify_data_changed = MagicMock()
    for k, v in overrides.items():
        setattr(plugin, k, v)
    return plugin


def _single_span_payload(
    trace_id: str,
    span_id: str,
    name: str,
    attributes: list | None = None,
    parent_span_id: str = "",
    start_time_ns: str = "1700000000000000000",
):
    """Build an OTLP payload with exactly one span."""
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "name": name,
        "kind": 3,
        "startTimeUnixNano": start_time_ns,
        "endTimeUnixNano": str(int(start_time_ns) + 1_000_000_000),
        "attributes": attributes or [],
    }
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "test-svc"}},
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "test"},
                "spans": [span],
            }],
        }]
    }


def _attr(key: str, value: str):
    return {"key": key, "value": {"stringValue": value}}


def _ingest_n_traces(plugin, n, integration_id=None, start_ns=1_700_000_000_000_000_000,
                     id_offset=0):
    """Ingest *n* distinct single-span traces, each with a unique trace_id.

    Use *id_offset* to avoid trace_id collisions across multiple calls.
    """
    for i in range(n):
        idx = id_offset + i
        tid = f"{idx:032x}"
        sid = f"{idx:016x}"
        ts = str(start_ns + i * 1_000_000_000)  # 1 second apart
        plugin._ingest_traces(
            _single_span_payload(trace_id=tid, span_id=sid, name=f"span-{idx}", start_time_ns=ts),
            integration_id=integration_id,
        )


# ---------------------------------------------------------------------------
# Trace eviction
# ---------------------------------------------------------------------------

class TestTraceEviction:
    """Trace eviction uses LRU semantics and is proportional per integration."""

    def test_traces_stored_with_integration_id(self):
        """Each trace entry records the integration_id it was ingested with."""
        plugin = _make_plugin()
        plugin._ingest_traces(
            _single_span_payload(trace_id="a" * 32, span_id="b" * 16, name="s"),
            integration_id="tenant-x",
        )
        trace = plugin.traces["a" * 32]
        assert trace["integration_id"] == "tenant-x"

    def test_traces_default_integration_id_empty(self):
        """Traces ingested without integration_id get an empty string tag."""
        plugin = _make_plugin()
        plugin._ingest_traces(
            _single_span_payload(trace_id="c" * 32, span_id="d" * 16, name="s"),
        )
        assert plugin.traces["c" * 32]["integration_id"] == ""

    def test_last_seen_updated_on_new_span(self):
        """last_seen is updated when a new span is appended to an existing trace."""
        plugin = _make_plugin()
        tid = "e" * 32
        plugin._ingest_traces(
            _single_span_payload(trace_id=tid, span_id="1" * 16, name="s1",
                                 start_time_ns="1700000000000000000"),
        )
        first_last_seen = plugin.traces[tid]["last_seen"]

        plugin._ingest_traces(
            _single_span_payload(trace_id=tid, span_id="2" * 16, name="s2",
                                 start_time_ns="1700000099000000000"),
        )
        assert plugin.traces[tid]["last_seen"] > first_last_seen

    def test_eviction_removes_least_recently_active(self):
        """When over MAX_TRACES, the least recently active traces are evicted."""
        plugin = _make_plugin(MAX_TRACES=10)
        # Ingest 12 traces — should trigger eviction (keep 80% = 8)
        _ingest_n_traces(plugin, 12)

        assert len(plugin.traces) <= 10
        # The oldest traces (lowest index/timestamp) should be gone
        assert f"{0:032x}" not in plugin.traces
        # The newest traces should survive
        assert f"{11:032x}" in plugin.traces

    def test_eviction_proportional_across_integrations(self):
        """Eviction distributes removals proportionally — no single tenant is starved."""
        plugin = _make_plugin(MAX_TRACES=20)

        # Tenant A: 15 traces (older timestamps)
        _ingest_n_traces(plugin, 15, integration_id="tenant-a",
                         start_ns=1_700_000_000_000_000_000, id_offset=0)
        # Tenant B: 10 traces (newer timestamps, non-overlapping ids)
        _ingest_n_traces(plugin, 10, integration_id="tenant-b",
                         start_ns=1_700_000_100_000_000_000, id_offset=1000)

        # 25 total > 20 MAX_TRACES → evict 5 to keep 16 (80%)
        # Without proportional eviction, all 5 could come from tenant-a (oldest).
        # With proportional eviction, both tenants lose some.
        a_remaining = sum(
            1 for t in plugin.traces.values() if t["integration_id"] == "tenant-a"
        )
        b_remaining = sum(
            1 for t in plugin.traces.values() if t["integration_id"] == "tenant-b"
        )

        # Both tenants should still have traces
        assert a_remaining > 0, "tenant-a was completely starved"
        assert b_remaining > 0, "tenant-b was completely starved"
        # Tenant A (60% of traces) should lose more than tenant B (40%)
        a_evicted = 15 - a_remaining
        b_evicted = 10 - b_remaining
        assert a_evicted >= b_evicted

    def test_no_eviction_under_threshold(self):
        """No traces removed when count is at or below MAX_TRACES."""
        plugin = _make_plugin(MAX_TRACES=20)
        _ingest_n_traces(plugin, 20)
        assert len(plugin.traces) == 20


# ---------------------------------------------------------------------------
# Session user-attr eviction
# ---------------------------------------------------------------------------

class TestSessionEviction:
    """Session user-attr cache eviction is proportional per integration."""

    def _populate_sessions(self, plugin, n, integration_id):
        """Directly populate the session cache with n entries for a given integration."""
        for i in range(n):
            sid = f"session-{integration_id}-{i}"
            plugin._session_user_attrs[sid] = {
                "_integration_id": integration_id,
                "user.email": f"user{i}@{integration_id}.com",
            }

    def test_eviction_triggers_above_threshold(self):
        """Sessions are evicted when the cache exceeds MAX_SESSION_CACHE."""
        plugin = _make_plugin(MAX_SESSION_CACHE=10)
        self._populate_sessions(plugin, 12, "tenant-a")

        # Trigger eviction by running an ingest (eviction runs at end of ingest)
        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
        )
        assert len(plugin._session_user_attrs) < 12

    def test_eviction_proportional_across_integrations(self):
        """Session eviction doesn't starve a single tenant."""
        plugin = _make_plugin(MAX_SESSION_CACHE=10)

        self._populate_sessions(plugin, 8, "tenant-a")
        self._populate_sessions(plugin, 6, "tenant-b")
        # 14 total > 10 threshold

        # Trigger eviction
        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
        )

        a_remaining = sum(
            1 for v in plugin._session_user_attrs.values()
            if v.get("_integration_id") == "tenant-a"
        )
        b_remaining = sum(
            1 for v in plugin._session_user_attrs.values()
            if v.get("_integration_id") == "tenant-b"
        )
        assert a_remaining > 0, "tenant-a sessions completely starved"
        assert b_remaining > 0, "tenant-b sessions completely starved"

    def test_no_eviction_under_threshold(self):
        """No sessions removed when count is at or below MAX_SESSION_CACHE."""
        plugin = _make_plugin(MAX_SESSION_CACHE=20)
        self._populate_sessions(plugin, 20, "tenant-a")

        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
        )
        # 20 original + 0 from ingest (no session.id attr) = 20, not over threshold
        assert len(plugin._session_user_attrs) == 20


# ---------------------------------------------------------------------------
# Cross-batch eviction
# ---------------------------------------------------------------------------

class TestCrossBatchEviction:
    """Cross-batch span cache is evicted when it exceeds MAX_CROSS_BATCH_SPANS."""

    def _populate_cross_batch(self, plugin, n, integration_id=""):
        """Directly populate cross-batch cache with n span entries per sub-dict."""
        cb = plugin._cross_batch[integration_id]
        for i in range(n):
            sid = f"span-{i:08x}"
            cb["agents"][sid] = f"agent-{i}"
            cb["tools"][sid] = f"tool-{i}"
            cb["models"][sid] = f"model-{i}"
            cb["parents"][sid] = f"parent-{i:08x}"
            cb["handoffs"][sid] = {"from": f"agent-{i}", "to": f"agent-{i+1}"}
            cb["traces"][sid] = f"trace-{i:032x}"

    def test_eviction_triggers_above_threshold(self):
        """Cross-batch entries are trimmed when total exceeds MAX_CROSS_BATCH_SPANS."""
        plugin = _make_plugin(MAX_CROSS_BATCH_SPANS=30)
        # 10 entries × 6 sub-dicts = 60 total > 30 threshold
        self._populate_cross_batch(plugin, 10, integration_id="tenant-a")

        before_size = sum(len(v) for v in plugin._cross_batch["tenant-a"].values())
        assert before_size == 60

        # Trigger eviction
        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
            integration_id="tenant-a",
        )

        after_size = sum(len(v) for v in plugin._cross_batch["tenant-a"].values())
        assert after_size < before_size

    def test_eviction_keeps_newest_entries(self):
        """Eviction drops the oldest half, preserving newer entries."""
        plugin = _make_plugin(MAX_CROSS_BATCH_SPANS=30)
        self._populate_cross_batch(plugin, 10, integration_id="tenant-a")

        # Trigger eviction
        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
            integration_id="tenant-a",
        )

        cb = plugin._cross_batch["tenant-a"]
        # The newest entries (highest index) should survive
        assert "span-00000009" in cb["agents"]
        # The oldest entries should be gone
        assert "span-00000000" not in cb["agents"]

    def test_no_cross_integration_eviction(self):
        """Eviction for one integration doesn't affect another."""
        plugin = _make_plugin(MAX_CROSS_BATCH_SPANS=30)
        # tenant-a: over threshold
        self._populate_cross_batch(plugin, 10, integration_id="tenant-a")
        # tenant-b: small, under threshold
        self._populate_cross_batch(plugin, 2, integration_id="tenant-b")

        b_before = sum(len(v) for v in plugin._cross_batch["tenant-b"].values())

        # Trigger eviction (ingesting as tenant-a)
        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
            integration_id="tenant-a",
        )

        b_after = sum(len(v) for v in plugin._cross_batch["tenant-b"].values())
        assert b_after == b_before, "tenant-b cache was affected by tenant-a eviction"

    def test_no_eviction_under_threshold(self):
        """Cross-batch cache left untouched when under MAX_CROSS_BATCH_SPANS."""
        plugin = _make_plugin(MAX_CROSS_BATCH_SPANS=1000)
        self._populate_cross_batch(plugin, 10, integration_id="tenant-a")

        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
            integration_id="tenant-a",
        )

        size = sum(len(v) for v in plugin._cross_batch["tenant-a"].values())
        # 10 original × 6 + 1 new span's entries from the trigger ingest
        assert size >= 60


# ---------------------------------------------------------------------------
# Per-tool trace list trimming
# ---------------------------------------------------------------------------

class TestToolTraceListTrim:
    """Per-tool trace lists are trimmed to the most recent 100 entries."""

    def test_trim_to_100(self):
        """Tool trace lists exceeding 100 are trimmed to the last 100."""
        plugin = _make_plugin()

        # Directly populate a tool with 150 trace entries
        plugin.discovered_tools["my-tool"]["traces"] = [
            {"model": "gpt-4", "trace_id": f"t{i}"} for i in range(150)
        ]

        # Trigger eviction via ingest
        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
        )

        traces = plugin.discovered_tools["my-tool"]["traces"]
        assert len(traces) == 100
        # Should keep the last 100 (newest)
        assert traces[0]["trace_id"] == "t50"
        assert traces[-1]["trace_id"] == "t149"

    def test_no_trim_under_100(self):
        """Tool trace lists at or under 100 are left untouched."""
        plugin = _make_plugin()
        plugin.discovered_tools["my-tool"]["traces"] = [
            {"model": "gpt-4", "trace_id": f"t{i}"} for i in range(100)
        ]

        plugin._ingest_traces(
            _single_span_payload(trace_id="f" * 32, span_id="f" * 16, name="trigger"),
        )

        assert len(plugin.discovered_tools["my-tool"]["traces"]) == 100
