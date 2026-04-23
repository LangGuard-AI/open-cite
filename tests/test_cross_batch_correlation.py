"""
Tests for cross-batch span correlation.

Verifies that agent handoffs and agent->tool/model relationships are
correctly established when spans from the same trace arrive in separate
OTLP batches (as happens with SimpleSpanProcessor in the OpenAI Agents SDK).
"""

import uuid
from unittest.mock import MagicMock

import pytest

from open_cite.plugins.opentelemetry import OpenTelemetryPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plugin():
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
    return plugin


def _single_span_payload(
    trace_id: str,
    span_id: str,
    name: str,
    attributes: list,
    parent_span_id: str = "",
):
    """Build an OTLP payload with exactly one span."""
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "name": name,
        "kind": 3,
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000001000000000",
        "attributes": attributes,
    }
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "test-service"}},
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "opentelemetry.instrumentation.openai_agents"},
                "spans": [span],
            }],
        }]
    }


def _attr(key: str, value: str):
    return {"key": key, "value": {"stringValue": value}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrossBatchHandoffCorrelation:
    """Handoff lineage is created even when spans arrive in separate batches."""

    def test_handoff_across_batches(self):
        """PEConcierge -> InvestorRelationsAgent handoff detected across 4 batches."""
        plugin = _make_plugin()
        iid = "test-integration"
        tid = "aabb001122334455"

        # Batch 1: Root workflow span
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid,
            span_id="root01",
            parent_span_id="",
            name="Agent Workflow",
            attributes=[
                _attr("traceloop.span.kind", "workflow"),
            ],
        ), integration_id=iid)

        # Batch 2: PEConcierge agent span
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid,
            span_id="concierge01",
            parent_span_id="root01",
            name="PEConcierge.agent",
            attributes=[
                _attr("traceloop.span.kind", "agent"),
                _attr("gen_ai.agent.name", "PEConcierge"),
            ],
        ), integration_id=iid)

        # Batch 3: Handoff span (child of PEConcierge)
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid,
            span_id="handoff01",
            parent_span_id="concierge01",
            name="PEConcierge -> unknown.handoff",
            attributes=[
                _attr("traceloop.span.kind", "handoff"),
                _attr("gen_ai.handoff.from_agent", "PEConcierge"),
                _attr("gen_ai.agent.name", "PEConcierge"),
            ],
        ), integration_id=iid)

        # Batch 4: InvestorRelationsAgent span (sibling of PEConcierge under root)
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid,
            span_id="iragent01",
            parent_span_id="root01",
            name="InvestorRelationsAgent.agent",
            attributes=[
                _attr("traceloop.span.kind", "agent"),
                _attr("gen_ai.agent.name", "InvestorRelationsAgent"),
            ],
        ), integration_id=iid)

        # Verify delegates_to lineage was created
        delegates_to = [
            e for e in plugin.lineage.values()
            if e["relationship_type"] == "delegates_to"
        ]
        assert len(delegates_to) >= 1, (
            f"Expected delegates_to lineage, got: {list(plugin.lineage.values())}"
        )
        edge = delegates_to[0]
        assert edge["source_type"] == "agent"
        assert edge["target_type"] == "agent"

    def test_no_cross_integration_pollution(self):
        """Handoff spans from different integrations don't create false edges."""
        plugin = _make_plugin()
        tid_a = "aaaa000011112222"
        tid_b = "bbbb000011112222"

        # Integration A: PEConcierge agent + handoff
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid_a, span_id="root_a", parent_span_id="",
            name="Workflow", attributes=[_attr("traceloop.span.kind", "workflow")],
        ), integration_id="tenant-a")
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid_a, span_id="concierge_a", parent_span_id="root_a",
            name="PEConcierge.agent",
            attributes=[_attr("traceloop.span.kind", "agent"), _attr("gen_ai.agent.name", "PEConcierge")],
        ), integration_id="tenant-a")
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid_a, span_id="handoff_a", parent_span_id="concierge_a",
            name="handoff", attributes=[
                _attr("traceloop.span.kind", "handoff"),
                _attr("gen_ai.handoff.from_agent", "PEConcierge"),
            ],
        ), integration_id="tenant-a")

        # Integration B: TargetAgent (unrelated tenant)
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid_b, span_id="root_b", parent_span_id="",
            name="Workflow", attributes=[_attr("traceloop.span.kind", "workflow")],
        ), integration_id="tenant-b")
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid_b, span_id="target_b", parent_span_id="root_b",
            name="TargetAgent.agent",
            attributes=[_attr("traceloop.span.kind", "agent"), _attr("gen_ai.agent.name", "TargetAgent")],
        ), integration_id="tenant-b")

        # No delegates_to should exist — the target agent is in a different integration
        delegates_to = [
            e for e in plugin.lineage.values()
            if e["relationship_type"] == "delegates_to"
        ]
        assert len(delegates_to) == 0, (
            f"Cross-integration lineage should not exist, got: {delegates_to}"
        )


class TestCrossBatchTraceCorrelation:
    """Agent->tool and agent->model edges created across batches."""

    def test_agent_tool_across_batches(self):
        """Agent and tool in separate batches of the same trace get linked."""
        plugin = _make_plugin()
        iid = "test-integration"
        tid = "ccdd001122334455"

        # Batch 1: Agent span with gen_ai.agent.name
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid,
            span_id="agent01",
            parent_span_id="",
            name="MyAgent.agent",
            attributes=[
                _attr("traceloop.span.kind", "agent"),
                _attr("gen_ai.agent.name", "MyAgent"),
            ],
        ), integration_id=iid)

        # Batch 2: Tool span (child of agent)
        plugin._ingest_traces(_single_span_payload(
            trace_id=tid,
            span_id="tool01",
            parent_span_id="agent01",
            name="search_database",
            attributes=[
                _attr("gen_ai.tool.name", "search_database"),
                _attr("gen_ai.tool.call.id", "call-1"),
            ],
        ), integration_id=iid)

        # Check agent has tool in tools_used
        agents = plugin.list_assets("agent")
        my_agent = next((a for a in agents if a["name"] == "MyAgent"), None)
        assert my_agent is not None, f"MyAgent not found in {[a['name'] for a in agents]}"
        assert "search_database" in my_agent.get("tools_used", []), (
            f"Expected search_database in tools_used, got {my_agent.get('tools_used')}"
        )

        # Check uses lineage exists
        uses = [
            e for e in plugin.lineage.values()
            if e["relationship_type"] == "uses"
            and e["source_type"] == "agent"
            and e["target_type"] == "tool"
        ]
        assert len(uses) >= 1, f"Expected agent->tool uses lineage, got: {list(plugin.lineage.values())}"
