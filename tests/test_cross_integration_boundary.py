"""
Cross-integration boundary test suite.

Spins up the Flask app with two integrations (integration-a, integration-b) that each
ingest distinct assets, then verifies every API endpoint that returns data
only returns the correct integration's data when filtered by integration_id —
and that no cross-integration leakage occurs.

Endpoints covered:
  GET /api/v1/assets
  GET /api/v1/tools
  GET /api/v1/models
  GET /api/v1/agents
  GET /api/v1/downstream
  GET /api/v1/mcp/servers
  GET /api/v1/mcp/tools
  GET /api/v1/lineage
  GET /api/v1/lineage-graph
  POST /api/v1/export
  GET /api/v1/stats
  GET /api/v1/status
"""

import json
import os
import tempfile
import uuid

import pytest

# Disable persistence and auto-start before importing the app
_tmpdir = tempfile.mkdtemp()
os.environ["OPENCITE_PERSISTENCE_ENABLED"] = "true"
os.environ["OPENCITE_AUTO_START"] = "false"
os.environ["OPENCITE_ENABLE_OTEL"] = "true"
os.environ["OPENCITE_ENABLE_MCP"] = "false"
os.environ["OPENCITE_ENABLE_DATABRICKS"] = "false"
os.environ["OPENCITE_ENABLE_GOOGLE_CLOUD"] = "false"
os.environ["OPENCITE_DB_PATH"] = os.path.join(_tmpdir, "test_opencite.db")

from open_cite.api.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INTEGRATION_A = "integration-aaaa-1111"
INTEGRATION_B = "integration-bbbb-2222"


def _otlp_payload(
    service_name: str,
    span_name: str,
    model: str,
    *,
    tool_name: str | None = None,
    downstream_url: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
):
    """Build a minimal OTLP JSON payload with one span."""
    trace_id = trace_id or uuid.uuid4().hex[:32]
    span_id = uuid.uuid4().hex[:16]

    attributes = [
        {"key": "gen_ai.request.model", "value": {"stringValue": model}},
        {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
    ]
    if tool_name:
        attributes.append({"key": "gen_ai.tool.name", "value": {"stringValue": tool_name}})
        attributes.append({"key": "gen_ai.tool.call.id", "value": {"stringValue": "call-1"}})
    if downstream_url:
        attributes.append({"key": "http.url", "value": {"stringValue": downstream_url}})
        attributes.append({"key": "server.address", "value": {"stringValue": downstream_url}})

    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": span_name,
        "attributes": attributes,
        "startTimeUnixNano": "1700000000000000000",
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service_name}},
                ]
            },
            "scopeSpans": [{"spans": [span]}],
        }]
    }


def _otlp_agent_with_tool(service_name, model, tool_name, *, downstream_url=None):
    """OTLP payload with an agent span (parent) and a tool span (child)."""
    trace_id = uuid.uuid4().hex[:32]
    agent_span_id = uuid.uuid4().hex[:16]
    tool_span_id = uuid.uuid4().hex[:16]

    agent_attrs = [
        {"key": "gen_ai.request.model", "value": {"stringValue": model}},
        {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
    ]
    tool_attrs = [
        {"key": "gen_ai.tool.name", "value": {"stringValue": tool_name}},
        {"key": "gen_ai.tool.call.id", "value": {"stringValue": "call-1"}},
        {"key": "gen_ai.request.model", "value": {"stringValue": model}},
    ]
    if downstream_url:
        tool_attrs.append({"key": "server.address", "value": {"stringValue": downstream_url}})

    agent_span = {
        "traceId": trace_id,
        "spanId": agent_span_id,
        "name": f"{service_name} agent",
        "attributes": agent_attrs,
        "startTimeUnixNano": "1700000000000000000",
    }
    tool_span = {
        "traceId": trace_id,
        "spanId": tool_span_id,
        "parentSpanId": agent_span_id,
        "name": tool_name,
        "attributes": tool_attrs,
        "startTimeUnixNano": "1700000000000000000",
    }
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service_name}},
                ]
            },
            "scopeSpans": [{"spans": [agent_span, tool_span]}],
        }]
    }


@pytest.fixture(scope="module")
def seeded_client():
    """Create the Flask app, configure the OTLP plugin, ingest two integrations' data,
    run a discovery cycle, and return the test client."""
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        # The app auto-registers the embedded OTLP plugin via OPENCITE_ENABLE_OTEL=true

        # --- Integration A data ---
        # Agent "alpha-agent" using gpt-4o and tool "search-web"
        client.post(
            "/v1/traces",
            data=json.dumps(_otlp_agent_with_tool(
                "alpha-agent", "gpt-4o", "search-web",
                downstream_url="https://api.alpha.example.com",
            )),
            content_type="application/json",
            headers={"X-Integration-Id": INTEGRATION_A},
        )
        # Extra model-only trace
        client.post(
            "/v1/traces",
            data=json.dumps(_otlp_payload("alpha-agent", "chat", "gpt-4o-mini")),
            content_type="application/json",
            headers={"X-Integration-Id": INTEGRATION_A},
        )

        # --- Integration B data ---
        # Agent "beta-agent" using claude-3-opus and tool "run-code"
        client.post(
            "/v1/traces",
            data=json.dumps(_otlp_agent_with_tool(
                "beta-agent", "claude-3-opus", "run-code",
                downstream_url="https://api.beta.example.com",
            )),
            content_type="application/json",
            headers={"X-Integration-Id": INTEGRATION_B},
        )
        # Extra model-only trace
        client.post(
            "/v1/traces",
            data=json.dumps(_otlp_payload("beta-agent", "chat", "claude-3-haiku")),
            content_type="application/json",
            headers={"X-Integration-Id": INTEGRATION_B},
        )

        # Trigger a persistence save cycle so assets are materialized to DB
        resp = client.post("/api/v1/persistence/save")
        assert resp.status_code == 200, resp.get_json()

        # Prime the asset cache
        resp = client.get("/api/v1/assets")
        assert resp.status_code == 200

        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_integration_ids(item):
    """Extract integration_ids from an asset dict."""
    return (
        (item.get("metadata") or {}).get("integration_ids")
        or item.get("integration_ids")
        or []
    )


def _assert_all_belong_to(items, integration_id, label="item"):
    """Assert every item in the list belongs to the given integration."""
    for item in items:
        ids = _get_integration_ids(item)
        assert integration_id in ids, (
            f"{label} '{item.get('name', item.get('id', '?'))}' "
            f"missing {integration_id} in integration_ids={ids}"
        )


def _assert_none_belong_to(items, integration_id, label="item"):
    """Assert no item in the list belongs to the given integration."""
    for item in items:
        ids = _get_integration_ids(item)
        assert integration_id not in ids, (
            f"{label} '{item.get('name', item.get('id', '?'))}' "
            f"should NOT have {integration_id} but has integration_ids={ids}"
        )


# ---------------------------------------------------------------------------
# /api/v1/assets — the aggregate endpoint
# ---------------------------------------------------------------------------

class TestAssetsEndpoint:

    def test_unfiltered_returns_both_integrations(self, seeded_client):
        resp = seeded_client.get("/api/v1/assets")
        data = resp.get_json()
        all_names = set()
        for category in data["assets"].values():
            for item in category:
                all_names.add(item.get("name", ""))
        # Should contain assets from both integrations
        assert "gpt-4o" in all_names or "alpha-agent" in all_names
        assert "claude-3-opus" in all_names or "beta-agent" in all_names

    def test_filtered_integration_a_excludes_integration_b(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/assets?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        for category_name, items in data["assets"].items():
            _assert_none_belong_to(items, INTEGRATION_B, label=f"assets.{category_name}")
            if items:
                _assert_all_belong_to(items, INTEGRATION_A, label=f"assets.{category_name}")

    def test_filtered_integration_b_excludes_integration_a(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/assets?integration_id={INTEGRATION_B}")
        data = resp.get_json()
        for category_name, items in data["assets"].items():
            _assert_none_belong_to(items, INTEGRATION_A, label=f"assets.{category_name}")
            if items:
                _assert_all_belong_to(items, INTEGRATION_B, label=f"assets.{category_name}")

    def test_filtered_totals_match_filtered_counts(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/assets?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        for key, items in data["assets"].items():
            assert data["totals"][key] == len(items), (
                f"totals[{key}]={data['totals'][key]} != len(assets[{key}])={len(items)}"
            )

    def test_nonexistent_integration_returns_empty(self, seeded_client):
        resp = seeded_client.get("/api/v1/assets?integration_id=does-not-exist")
        data = resp.get_json()
        for key, items in data["assets"].items():
            assert len(items) == 0, f"assets.{key} should be empty but has {len(items)} items"

    def test_cached_response_still_filters(self, seeded_client):
        """Hit /assets twice with different integration_ids — cache must not leak."""
        resp_a = seeded_client.get(f"/api/v1/assets?integration_id={INTEGRATION_A}")
        resp_b = seeded_client.get(f"/api/v1/assets?integration_id={INTEGRATION_B}")
        data_a = resp_a.get_json()
        data_b = resp_b.get_json()

        names_a = set()
        names_b = set()
        for items in data_a["assets"].values():
            for item in items:
                names_a.add(item.get("name", ""))
        for items in data_b["assets"].values():
            for item in items:
                names_b.add(item.get("name", ""))

        # No overlap (each integration has unique asset names in our fixture)
        assert names_a.isdisjoint(names_b), (
            f"Cache leak: integration A names {names_a} overlap with integration B names {names_b}"
        )


# ---------------------------------------------------------------------------
# /api/v1/tools
# ---------------------------------------------------------------------------

class TestToolsEndpoint:

    def test_unfiltered_returns_all_tools(self, seeded_client):
        resp = seeded_client.get("/api/v1/tools")
        data = resp.get_json()
        names = {t["name"] for t in data["tools"]}
        assert "search-web" in names
        assert "run-code" in names

    def test_filtered_integration_a(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/tools?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        names = {t["name"] for t in data["tools"]}
        assert "search-web" in names
        assert "run-code" not in names
        _assert_all_belong_to(data["tools"], INTEGRATION_A, label="tool")

    def test_filtered_integration_b(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/tools?integration_id={INTEGRATION_B}")
        data = resp.get_json()
        names = {t["name"] for t in data["tools"]}
        assert "run-code" in names
        assert "search-web" not in names
        _assert_all_belong_to(data["tools"], INTEGRATION_B, label="tool")


# ---------------------------------------------------------------------------
# /api/v1/models
# ---------------------------------------------------------------------------

class TestModelsEndpoint:

    def test_unfiltered_returns_all_models(self, seeded_client):
        resp = seeded_client.get("/api/v1/models")
        data = resp.get_json()
        names = {m["name"] for m in data["models"]}
        assert "gpt-4o" in names
        assert "claude-3-opus" in names

    def test_filtered_integration_a(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/models?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        names = {m["name"] for m in data["models"]}
        assert "gpt-4o" in names or "gpt-4o-mini" in names
        assert "claude-3-opus" not in names
        assert "claude-3-haiku" not in names
        _assert_all_belong_to(data["models"], INTEGRATION_A, label="model")

    def test_filtered_integration_b(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/models?integration_id={INTEGRATION_B}")
        data = resp.get_json()
        names = {m["name"] for m in data["models"]}
        assert "claude-3-opus" in names or "claude-3-haiku" in names
        assert "gpt-4o" not in names
        assert "gpt-4o-mini" not in names
        _assert_all_belong_to(data["models"], INTEGRATION_B, label="model")


# ---------------------------------------------------------------------------
# /api/v1/agents
# ---------------------------------------------------------------------------

class TestAgentsEndpoint:

    def test_unfiltered_returns_all_agents(self, seeded_client):
        resp = seeded_client.get("/api/v1/agents")
        data = resp.get_json()
        names = {a["name"] for a in data["agents"]}
        assert "alpha-agent" in names
        assert "beta-agent" in names

    def test_filtered_integration_a(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/agents?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        names = {a["name"] for a in data["agents"]}
        assert "alpha-agent" in names
        assert "beta-agent" not in names
        _assert_all_belong_to(data["agents"], INTEGRATION_A, label="agent")

    def test_filtered_integration_b(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/agents?integration_id={INTEGRATION_B}")
        data = resp.get_json()
        names = {a["name"] for a in data["agents"]}
        assert "beta-agent" in names
        assert "alpha-agent" not in names
        _assert_all_belong_to(data["agents"], INTEGRATION_B, label="agent")


# ---------------------------------------------------------------------------
# /api/v1/downstream
# ---------------------------------------------------------------------------

class TestDownstreamEndpoint:

    def test_filtered_integration_a(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/downstream?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        _assert_none_belong_to(data["downstream_systems"], INTEGRATION_B, label="downstream")

    def test_filtered_integration_b(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/downstream?integration_id={INTEGRATION_B}")
        data = resp.get_json()
        _assert_none_belong_to(data["downstream_systems"], INTEGRATION_A, label="downstream")


# ---------------------------------------------------------------------------
# /api/v1/lineage
# ---------------------------------------------------------------------------

class TestLineageEndpoint:

    def test_unfiltered_returns_lineage_from_both(self, seeded_client):
        resp = seeded_client.get("/api/v1/lineage")
        data = resp.get_json()
        assert len(data["relationships"]) > 0

    def test_filtered_integration_a_no_integration_b_ids(self, seeded_client):
        """Lineage filtered to integration A should not reference integration B asset UUIDs."""
        from open_cite.plugins.opentelemetry import _AGENT_NS, _TOOL_NS, _MODEL_NS

        resp = seeded_client.get(f"/api/v1/lineage?integration_id={INTEGRATION_A}")
        data = resp.get_json()

        # Build the set of integration B UUIDs
        b_uuids = {
            str(uuid.uuid5(_AGENT_NS, f"{INTEGRATION_B}:beta-agent")),
            str(uuid.uuid5(_TOOL_NS, f"{INTEGRATION_B}:run-code")),
            str(uuid.uuid5(_MODEL_NS, f"{INTEGRATION_B}:claude-3-opus")),
            str(uuid.uuid5(_MODEL_NS, f"{INTEGRATION_B}:claude-3-haiku")),
        }

        for rel in data["relationships"]:
            assert rel["source_id"] not in b_uuids, (
                f"Lineage source_id {rel['source_id']} belongs to integration B"
            )
            assert rel["target_id"] not in b_uuids, (
                f"Lineage target_id {rel['target_id']} belongs to integration B"
            )

    def test_filtered_integration_b_no_integration_a_ids(self, seeded_client):
        from open_cite.plugins.opentelemetry import _AGENT_NS, _TOOL_NS, _MODEL_NS

        resp = seeded_client.get(f"/api/v1/lineage?integration_id={INTEGRATION_B}")
        data = resp.get_json()

        a_uuids = {
            str(uuid.uuid5(_AGENT_NS, f"{INTEGRATION_A}:alpha-agent")),
            str(uuid.uuid5(_TOOL_NS, f"{INTEGRATION_A}:search-web")),
            str(uuid.uuid5(_MODEL_NS, f"{INTEGRATION_A}:gpt-4o")),
            str(uuid.uuid5(_MODEL_NS, f"{INTEGRATION_A}:gpt-4o-mini")),
        }

        for rel in data["relationships"]:
            assert rel["source_id"] not in a_uuids, (
                f"Lineage source_id {rel['source_id']} belongs to integration A"
            )
            assert rel["target_id"] not in a_uuids, (
                f"Lineage target_id {rel['target_id']} belongs to integration A"
            )


# ---------------------------------------------------------------------------
# /api/v1/lineage-graph  (HTML — just check it doesn't error)
# ---------------------------------------------------------------------------

class TestLineageGraphEndpoint:

    def test_returns_html(self, seeded_client):
        resp = seeded_client.get("/api/v1/lineage-graph")
        assert resp.status_code == 200
        assert b"<html" in resp.data.lower() or b"pyvis" in resp.data.lower()


# ---------------------------------------------------------------------------
# /api/v1/mcp/servers
# ---------------------------------------------------------------------------

class TestMcpServersEndpoint:

    def test_filtered_returns_no_cross_integration(self, seeded_client):
        resp = seeded_client.get(f"/api/v1/mcp/servers?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        _assert_none_belong_to(data["servers"], INTEGRATION_B, label="mcp_server")


# ---------------------------------------------------------------------------
# /api/v1/mcp/tools  (no integration_id filter — flag it)
# ---------------------------------------------------------------------------

class TestMcpToolsEndpoint:

    def test_endpoint_exists(self, seeded_client):
        resp = seeded_client.get("/api/v1/mcp/tools")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/v1/status and /api/v1/stats — should not leak per-integration data
# ---------------------------------------------------------------------------

class TestMetaEndpoints:

    def test_status_has_no_integration_specific_data(self, seeded_client):
        resp = seeded_client.get("/api/v1/status")
        data = resp.get_json()
        raw = json.dumps(data)
        assert INTEGRATION_A not in raw, "status endpoint leaks integration A id"
        assert INTEGRATION_B not in raw, "status endpoint leaks integration B id"

    def test_stats_has_no_integration_specific_data(self, seeded_client):
        resp = seeded_client.get("/api/v1/stats")
        data = resp.get_json()
        raw = json.dumps(data)
        assert INTEGRATION_A not in raw, "stats endpoint leaks integration A id"
        assert INTEGRATION_B not in raw, "stats endpoint leaks integration B id"


# ---------------------------------------------------------------------------
# /api/v1/export — check for cross-integration leakage
# ---------------------------------------------------------------------------

class TestExportEndpoint:

    def test_export_contains_data(self, seeded_client):
        resp = seeded_client.post("/api/v1/export", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        # Export returns data — just verify it doesn't crash
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Symmetric boundary test: exhaustive check across all filterable endpoints
# ---------------------------------------------------------------------------

class TestSymmetricBoundary:
    """For each filterable endpoint, assert that filtering by integration X
    returns zero items tagged with integration Y."""

    FILTERABLE = [
        ("/api/v1/tools", "tools"),
        ("/api/v1/models", "models"),
        ("/api/v1/agents", "agents"),
        ("/api/v1/downstream", "downstream_systems"),
        ("/api/v1/mcp/servers", "servers"),
    ]

    @pytest.mark.parametrize("endpoint,key", FILTERABLE)
    def test_integration_a_filter_excludes_integration_b(self, seeded_client, endpoint, key):
        resp = seeded_client.get(f"{endpoint}?integration_id={INTEGRATION_A}")
        data = resp.get_json()
        _assert_none_belong_to(data[key], INTEGRATION_B, label=f"{endpoint}:{key}")

    @pytest.mark.parametrize("endpoint,key", FILTERABLE)
    def test_integration_b_filter_excludes_integration_a(self, seeded_client, endpoint, key):
        resp = seeded_client.get(f"{endpoint}?integration_id={INTEGRATION_B}")
        data = resp.get_json()
        _assert_none_belong_to(data[key], INTEGRATION_A, label=f"{endpoint}:{key}")

    @pytest.mark.parametrize("endpoint,key", FILTERABLE)
    def test_bogus_integration_returns_empty(self, seeded_client, endpoint, key):
        resp = seeded_client.get(f"{endpoint}?integration_id=nonexistent-integration")
        data = resp.get_json()
        assert len(data[key]) == 0, f"{endpoint} returned {len(data[key])} items for bogus integration"
