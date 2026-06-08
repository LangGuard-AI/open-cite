"""
Integration-id cross-contamination proof tests.

Each test demonstrates a specific scenario where data from one integration_id
leaks into another integration_id's view.  These tests are expected to FAIL
against the current implementation and PASS once the isolation bugs are fixed.

Contamination vectors covered:
  1. discovered_tools shared by name across integrations
  2. discovered_agents shared by name across integrations
  3. model_token_usage / model_call_count unscoped
  4. _session_user_attrs keyed by session.id alone (user data leak)
  5. MCP registries unscoped (servers, tools, resources, usage_stats)
  6. discovered_downstream shared by system_id across integrations
  7. _list_mcp_* methods have no integration_id filtering
  8. model_providers last-writer-wins across integrations
"""

import uuid
from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from open_cite.plugins.opentelemetry import OpenTelemetryPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INT_A = "integration-aaaa"
INT_B = "integration-bbbb"


def _make_plugin():
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


def _otlp_payload(
    service_name: str,
    span_name: str,
    model: str,
    *,
    tool_name: str | None = None,
    agent_name: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    session_id: str | None = None,
    user_email: str | None = None,
    extra_attrs: list | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    mcp_server: str | None = None,
    mcp_tool: str | None = None,
    mcp_resource_uri: str | None = None,
    db_system: str | None = None,
    db_name: str | None = None,
    server_address: str | None = None,
):
    """Build a minimal OTLP JSON payload with one span."""
    trace_id = trace_id or uuid.uuid4().hex[:32]
    span_id = span_id or uuid.uuid4().hex[:16]

    attributes = [
        {"key": "gen_ai.request.model", "value": {"stringValue": model}},
        {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
    ]
    if tool_name:
        attributes.append({"key": "gen_ai.tool.name", "value": {"stringValue": tool_name}})
        attributes.append({"key": "gen_ai.tool.call.id", "value": {"stringValue": "call-1"}})
    if agent_name:
        attributes.append({"key": "gen_ai.agent.name", "value": {"stringValue": agent_name}})
    if session_id:
        attributes.append({"key": "session.id", "value": {"stringValue": session_id}})
    if user_email:
        attributes.append({"key": "user.email", "value": {"stringValue": user_email}})
    if input_tokens is not None:
        attributes.append({"key": "gen_ai.usage.input_tokens", "value": {"intValue": input_tokens}})
    if output_tokens is not None:
        attributes.append({"key": "gen_ai.usage.output_tokens", "value": {"intValue": output_tokens}})
    if mcp_server:
        attributes.append({"key": "mcp.server.name", "value": {"stringValue": mcp_server}})
    if mcp_tool:
        attributes.append({"key": "mcp.tool.name", "value": {"stringValue": mcp_tool}})
    if mcp_resource_uri:
        attributes.append({"key": "mcp.resource.uri", "value": {"stringValue": mcp_resource_uri}})
    if db_system:
        attributes.append({"key": "db.system", "value": {"stringValue": db_system}})
    if db_name:
        attributes.append({"key": "db.name", "value": {"stringValue": db_name}})
    if server_address:
        attributes.append({"key": "server.address", "value": {"stringValue": server_address}})
    if extra_attrs:
        attributes.extend(extra_attrs)

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


# ===================================================================
# 1. discovered_tools: shared by tool name across integrations
# ===================================================================

class TestToolsCrossContamination:
    """Two integrations discover a tool with the same name.
    The tool's models set and trace list must not leak between them."""

    def test_tool_models_isolated(self):
        """Integration A's tool uses model-A, integration B's uses model-B.
        Listing tools for integration A must not include model-B."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc", "call", "gpt-4", tool_name="SharedTool"),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc", "call", "claude-3-opus", tool_name="SharedTool"),
            integration_id=INT_B,
        )

        tools_a = plugin.list_assets("tool", integration_id=INT_A)
        tools_b = plugin.list_assets("tool", integration_id=INT_B)

        # Each integration should see only its own model
        assert len(tools_a) == 1
        assert len(tools_b) == 1
        assert tools_a[0]["models"] == ["gpt-4"], (
            f"Integration A sees models from B: {tools_a[0]['models']}"
        )
        assert tools_b[0]["models"] == ["claude-3-opus"], (
            f"Integration B sees models from A: {tools_b[0]['models']}"
        )

    def test_tool_trace_count_isolated(self):
        """Trace counts should reflect only the queried integration's activity."""
        plugin = _make_plugin()

        # Integration A: 3 calls
        for _ in range(3):
            plugin._ingest_traces(
                _otlp_payload("svc", "call", "gpt-4", tool_name="SharedTool"),
                integration_id=INT_A,
            )
        # Integration B: 1 call
        plugin._ingest_traces(
            _otlp_payload("svc", "call", "gpt-4", tool_name="SharedTool"),
            integration_id=INT_B,
        )

        tools_a = plugin.list_assets("tool", integration_id=INT_A)
        tools_b = plugin.list_assets("tool", integration_id=INT_B)

        assert tools_a[0]["trace_count"] == 3, (
            f"Integration A trace_count should be 3, got {tools_a[0]['trace_count']}"
        )
        assert tools_b[0]["trace_count"] == 1, (
            f"Integration B trace_count should be 1, got {tools_b[0]['trace_count']}"
        )


# ===================================================================
# 2. discovered_agents: shared by name across integrations
# ===================================================================

class TestAgentsCrossContamination:
    """Two integrations discover an agent with the same service.name.
    tools_used and models_used must not leak."""

    def test_agent_tools_used_isolated(self):
        """Agent 'my-agent' in integration A uses tool-A, in B uses tool-B.
        Listing agents for A must not show tool-B."""
        plugin = _make_plugin()

        # Integration A: agent uses tool-alpha
        plugin._ingest_traces(
            _otlp_payload("svc", "call", "gpt-4",
                          agent_name="my-agent", tool_name="tool-alpha"),
            integration_id=INT_A,
        )
        # Integration B: same agent name uses tool-beta
        plugin._ingest_traces(
            _otlp_payload("svc", "call", "gpt-4",
                          agent_name="my-agent", tool_name="tool-beta"),
            integration_id=INT_B,
        )

        agents_a = plugin.list_assets("agent", integration_id=INT_A)
        agents_b = plugin.list_assets("agent", integration_id=INT_B)

        assert len(agents_a) == 1
        assert len(agents_b) == 1
        assert set(agents_a[0]["tools_used"]) == {"tool-alpha"}, (
            f"Integration A agent sees B's tools: {agents_a[0]['tools_used']}"
        )
        assert set(agents_b[0]["tools_used"]) == {"tool-beta"}, (
            f"Integration B agent sees A's tools: {agents_b[0]['tools_used']}"
        )

    def test_agent_models_used_isolated(self):
        """Agent 'my-agent' in integration A uses model-A, in B uses model-B."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc", "call", "gpt-4", agent_name="my-agent"),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc", "call", "claude-3-opus", agent_name="my-agent"),
            integration_id=INT_B,
        )

        agents_a = plugin.list_assets("agent", integration_id=INT_A)
        agents_b = plugin.list_assets("agent", integration_id=INT_B)

        assert set(agents_a[0]["models_used"]) == {"gpt-4"}, (
            f"Integration A agent sees B's models: {agents_a[0]['models_used']}"
        )
        assert set(agents_b[0]["models_used"]) == {"claude-3-opus"}, (
            f"Integration B agent sees A's models: {agents_b[0]['models_used']}"
        )


# ===================================================================
# 3. model_token_usage / model_call_count: completely unscoped
# ===================================================================

class TestModelCountsCrossContamination:
    """Token usage and call counts are global — querying for one
    integration returns totals that include the other's traffic."""

    def test_model_call_count_isolated(self):
        """Each integration calls gpt-4 a different number of times.
        usage_count must reflect only the queried integration."""
        plugin = _make_plugin()

        # Integration A: 5 calls (as agent, not tool — so model gets counted)
        for _ in range(5):
            plugin._ingest_traces(
                _otlp_payload("svc-a", "llm-call", "gpt-4"),
                integration_id=INT_A,
            )
        # Integration B: 2 calls
        for _ in range(2):
            plugin._ingest_traces(
                _otlp_payload("svc-b", "llm-call", "gpt-4"),
                integration_id=INT_B,
            )

        models_a = plugin.list_assets("model", integration_id=INT_A)
        models_b = plugin.list_assets("model", integration_id=INT_B)

        gpt4_a = next((m for m in models_a if m["name"] == "gpt-4"), None)
        gpt4_b = next((m for m in models_b if m["name"] == "gpt-4"), None)

        assert gpt4_a is not None
        assert gpt4_b is not None
        assert gpt4_a["usage_count"] == 5, (
            f"Integration A should see 5 calls, got {gpt4_a['usage_count']}"
        )
        assert gpt4_b["usage_count"] == 2, (
            f"Integration B should see 2 calls, got {gpt4_b['usage_count']}"
        )

    def test_model_token_usage_isolated(self):
        """Token counts must not bleed across integrations."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "llm-call", "gpt-4",
                          input_tokens=100, output_tokens=50),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc-b", "llm-call", "gpt-4",
                          input_tokens=900, output_tokens=450),
            integration_id=INT_B,
        )

        models_a = plugin.list_assets("model", integration_id=INT_A)
        gpt4_a = next((m for m in models_a if m["name"] == "gpt-4"), None)

        assert gpt4_a is not None
        assert gpt4_a["total_input_tokens"] == 100, (
            f"Integration A should see 100 input tokens, got {gpt4_a['total_input_tokens']}"
        )
        assert gpt4_a["total_output_tokens"] == 50, (
            f"Integration A should see 50 output tokens, got {gpt4_a['total_output_tokens']}"
        )


# ===================================================================
# 4. _session_user_attrs: session.id collision leaks user data
# ===================================================================

class TestSessionUserAttrsLeak:
    """If two integrations use the same session.id, user.* attributes
    from one integration's spans get injected into the other's spans."""

    def test_user_email_does_not_leak_across_integrations(self):
        """Integration A sets user.email on session 'shared-session'.
        Integration B's subsequent span in the same session must NOT
        inherit A's user.email."""
        plugin = _make_plugin()
        shared_session = "shared-session-123"

        # Integration A: set user.email on this session
        plugin._ingest_traces(
            _otlp_payload("svc-a", "call", "gpt-4",
                          tool_name="tool-a",
                          session_id=shared_session,
                          user_email="alice@company-a.com"),
            integration_id=INT_A,
        )

        # Integration B: span in the same session, no user.email set
        payload_b = _otlp_payload("svc-b", "call", "gpt-4",
                                  tool_name="tool-b",
                                  session_id=shared_session)
        plugin._ingest_traces(payload_b, integration_id=INT_B)

        # Check: tool-b's metadata should NOT contain alice's email
        tools_b = plugin.list_assets("tool", integration_id=INT_B)
        assert len(tools_b) == 1
        tool_b_meta = tools_b[0].get("metadata", {})
        leaked_email = tool_b_meta.get("user.email")
        assert leaked_email is None or leaked_email != "alice@company-a.com", (
            f"user.email leaked from integration A to B: {leaked_email}"
        )

    def test_session_cache_scoped_by_integration(self):
        """The session user-attr cache must be keyed by (integration_id, session_id),
        not just session_id."""
        plugin = _make_plugin()
        shared_session = "session-xyz"

        # Integration A: cache user attrs
        attrs_a = [
            {"key": "session.id", "value": {"stringValue": shared_session}},
            {"key": "user.email", "value": {"stringValue": "alice@a.com"}},
            {"key": "user.account_uuid", "value": {"stringValue": "uuid-a"}},
        ]
        resource_a = {"attributes": [
            {"key": "service.name", "value": {"stringValue": "svc-a"}},
        ]}
        plugin._enrich_session_user_attrs(attrs_a, resource_a, integration_id=INT_A)

        # Integration B: lookup same session, should get nothing
        attrs_b = [
            {"key": "session.id", "value": {"stringValue": shared_session}},
        ]
        resource_b = {"attributes": [
            {"key": "service.name", "value": {"stringValue": "svc-b"}},
        ]}
        enriched, attr_dict, _ = plugin._enrich_session_user_attrs(
            attrs_b, resource_b, integration_id=INT_B)

        # attr_dict should NOT contain user.email from integration A
        assert "user.email" not in attr_dict, (
            f"Session cache leaked user.email to different integration: {attr_dict}"
        )
        assert "user.account_uuid" not in attr_dict, (
            f"Session cache leaked user.account_uuid to different integration: {attr_dict}"
        )


# ===================================================================
# 5. MCP registries: unscoped by integration_id
# ===================================================================

class TestMCPCrossContamination:
    """MCP servers/tools/resources are keyed by name, not by integration.
    Two integrations with the same MCP server name share all state."""

    def test_mcp_server_usage_isolated(self):
        """Two integrations use an MCP server with the same name.
        Trace counts should not be mixed."""
        plugin = _make_plugin()

        # Integration A: 3 MCP calls
        for _ in range(3):
            plugin._ingest_traces(
                _otlp_payload("svc-a", "mcp-call", "gpt-4",
                              mcp_server="shared-mcp", mcp_tool="search"),
                integration_id=INT_A,
            )
        # Integration B: 1 MCP call
        plugin._ingest_traces(
            _otlp_payload("svc-b", "mcp-call", "gpt-4",
                          mcp_server="shared-mcp", mcp_tool="search"),
            integration_id=INT_B,
        )

        # The MCP server's trace list contains traces from both integrations
        server_id = plugin._generate_mcp_server_id("shared-mcp")
        server = plugin.mcp_servers.get(server_id, {})
        trace_count = len(server.get("traces", []))

        # This SHOULD be filterable per integration, but currently isn't
        # For now, just prove the contamination exists:
        # The server has 4 traces total (3 from A + 1 from B)
        assert trace_count == 4, f"Expected 4 total traces (proving shared state), got {trace_count}"

        # The list method should support integration_id filtering
        servers = plugin._list_mcp_servers(integration_id=INT_A)
        # Currently _list_mcp_servers ignores integration_id entirely
        # so it returns the server with mixed data from both integrations
        for s in servers:
            # If properly isolated, A should see 3 traces, not 4
            if s["id"] == server_id:
                assert s["trace_count"] == 3, (
                    f"MCP server trace_count for integration A should be 3, "
                    f"got {s['trace_count']} (includes B's traces)"
                )

    def test_mcp_tool_call_count_isolated(self):
        """MCP tool call_count must not mix across integrations."""
        plugin = _make_plugin()

        for _ in range(5):
            plugin._ingest_traces(
                _otlp_payload("svc-a", "mcp-call", "gpt-4",
                              mcp_server="shared-mcp", mcp_tool="search"),
                integration_id=INT_A,
            )
        for _ in range(2):
            plugin._ingest_traces(
                _otlp_payload("svc-b", "mcp-call", "gpt-4",
                              mcp_server="shared-mcp", mcp_tool="search"),
                integration_id=INT_B,
            )

        server_id = plugin._generate_mcp_server_id("shared-mcp")
        tool_id = f"{server_id}-search"
        tool = plugin.mcp_tools.get(tool_id, {})

        # The call_count is 7 (5+2) — not isolated
        assert tool["usage"]["call_count"] == 7, "Proving shared state: total is 7"

        # A proper implementation would let us query per-integration
        tools = plugin._list_mcp_tools(integration_id=INT_A)
        for t in tools:
            if t["id"] == tool_id:
                assert t["call_count"] == 5, (
                    f"MCP tool call_count for A should be 5, got {t['call_count']}"
                )

    def test_mcp_usage_stats_isolated(self):
        """Per-integration MCP usage stats are properly isolated."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "mcp-call", "gpt-4",
                          mcp_server="shared-mcp", mcp_tool="search"),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc-b", "mcp-call", "gpt-4",
                          mcp_server="shared-mcp", mcp_tool="query"),
            integration_id=INT_B,
        )

        server_id = plugin._generate_mcp_server_id("shared-mcp")

        # Global stats still sum both
        stats = plugin.mcp_usage_stats[server_id]
        assert stats["tool_calls"] == 2

        # Per-integration tool data is isolated
        tool_id_search = f"{server_id}-search"
        tool_id_query = f"{server_id}-query"
        assert plugin.mcp_tool_integration_data[tool_id_search][INT_A]["call_count"] == 1
        assert INT_B not in plugin.mcp_tool_integration_data[tool_id_search]
        assert plugin.mcp_tool_integration_data[tool_id_query][INT_B]["call_count"] == 1
        assert INT_A not in plugin.mcp_tool_integration_data[tool_id_query]


# ===================================================================
# 6. discovered_downstream: shared by system_id across integrations
# ===================================================================

class TestDownstreamCrossContamination:
    """Two integrations connect to the same downstream DB.
    tools_connecting should not mix."""

    def test_downstream_tools_connecting_isolated(self):
        """Integration A connects svc-a to postgres, B connects svc-b.
        Listing downstream for A should only show svc-a."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "db-query", "gpt-4",
                          tool_name="svc-a-tool",
                          db_system="postgresql", db_name="mydb",
                          server_address="db.internal"),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc-b", "db-query", "gpt-4",
                          tool_name="svc-b-tool",
                          db_system="postgresql", db_name="mydb",
                          server_address="db.internal"),
            integration_id=INT_B,
        )

        # Unfiltered: both services visible
        systems = plugin.list_assets("downstream_system")
        pg_system = next(
            (s for s in systems if "postgresql" in s["name"]), None
        )
        assert pg_system is not None
        assert set(pg_system["tools_connecting"]) == {"svc-a", "svc-b"}

        # Filtered for A: only svc-a visible
        systems_a = plugin.list_assets("downstream_system", integration_id=INT_A)
        pg_a = next(
            (s for s in systems_a if "postgresql" in s["name"]), None
        )
        assert pg_a is not None
        assert set(pg_a["tools_connecting"]) == {"svc-a"}, (
            f"Integration A should only see svc-a, got {pg_a['tools_connecting']}"
        )


# ===================================================================
# 7. _list_mcp_* methods: no integration_id filtering
# ===================================================================

class TestMCPListNoFiltering:
    """_list_mcp_servers/tools/resources ignore integration_id kwarg."""

    def test_list_mcp_servers_ignores_integration_filter(self):
        """Register MCP servers under different integrations.
        Listing with integration_id should filter, but doesn't."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "call", "gpt-4",
                          mcp_server="mcp-only-a", mcp_tool="t1"),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc-b", "call", "gpt-4",
                          mcp_server="mcp-only-b", mcp_tool="t2"),
            integration_id=INT_B,
        )

        # Query for integration A should only return mcp-only-a
        servers_a = plugin._list_mcp_servers(integration_id=INT_A)
        server_names = {s["name"] for s in servers_a}

        assert "mcp-only-b" not in server_names, (
            f"Integration A sees B's MCP server: {server_names}"
        )
        assert server_names == {"mcp-only-a"}, (
            f"Expected only mcp-only-a, got {server_names}"
        )

    def test_list_mcp_tools_ignores_integration_filter(self):
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "call", "gpt-4",
                          mcp_server="shared-mcp", mcp_tool="tool-a-only"),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc-b", "call", "gpt-4",
                          mcp_server="shared-mcp", mcp_tool="tool-b-only"),
            integration_id=INT_B,
        )

        tools_a = plugin._list_mcp_tools(integration_id=INT_A)
        tool_names = {t["name"] for t in tools_a}

        assert "tool-b-only" not in tool_names, (
            f"Integration A sees B's MCP tool: {tool_names}"
        )

    def test_list_mcp_resources_ignores_integration_filter(self):
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "call", "gpt-4",
                          mcp_server="shared-mcp",
                          mcp_resource_uri="resource://a/data"),
            integration_id=INT_A,
        )
        plugin._ingest_traces(
            _otlp_payload("svc-b", "call", "gpt-4",
                          mcp_server="shared-mcp",
                          mcp_resource_uri="resource://b/data"),
            integration_id=INT_B,
        )

        resources_a = plugin._list_mcp_resources(integration_id=INT_A)
        uris = {r["uri"] for r in resources_a}

        assert "resource://b/data" not in uris, (
            f"Integration A sees B's MCP resource: {uris}"
        )


# ===================================================================
# 8. model_providers: last-writer-wins across integrations
# ===================================================================

class TestModelProvidersCrossContamination:
    """model_providers is a flat dict — the last integration to report
    a provider for a model name overwrites the previous one."""

    def test_model_provider_not_overwritten_by_other_integration(self):
        """Integration A sees gpt-4 via 'azure', B sees it via 'openai'.
        Querying models for A should show 'azure', not 'openai'."""
        plugin = _make_plugin()

        # A sees gpt-4 with provider=azure
        payload_a = _otlp_payload("svc-a", "call", "gpt-4")
        payload_a["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"].append(
            {"key": "gen_ai.system", "value": {"stringValue": "azure"}}
        )
        plugin._ingest_traces(payload_a, integration_id=INT_A)

        # B sees gpt-4 with provider=openai (overwrites!)
        payload_b = _otlp_payload("svc-b", "call", "gpt-4")
        payload_b["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"].append(
            {"key": "gen_ai.system", "value": {"stringValue": "openai"}}
        )
        plugin._ingest_traces(payload_b, integration_id=INT_B)

        models_a = plugin.list_assets("model", integration_id=INT_A)
        gpt4_a = next((m for m in models_a if m["name"] == "gpt-4"), None)

        assert gpt4_a is not None
        assert gpt4_a["provider"] == "azure", (
            f"Integration A's provider was overwritten by B: {gpt4_a['provider']}"
        )


# ===================================================================
# 9. Integration A's data visible with no filter (expected)
#    but also visible when filtering for B (not expected)
# ===================================================================

class TestUnfilteredVsFiltered:
    """When an asset exists ONLY under integration A, querying with
    integration_id=B should return nothing for that asset."""

    def test_tool_exclusive_to_a_invisible_to_b(self):
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc", "call", "gpt-4", tool_name="a-only-tool"),
            integration_id=INT_A,
        )

        tools_b = plugin.list_assets("tool", integration_id=INT_B)
        tool_names = {t["name"] for t in tools_b}
        assert "a-only-tool" not in tool_names

    def test_agent_exclusive_to_a_invisible_to_b(self):
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc", "call", "gpt-4", agent_name="a-only-agent"),
            integration_id=INT_A,
        )

        agents_b = plugin.list_assets("agent", integration_id=INT_B)
        agent_names = {a["name"] for a in agents_b}
        assert "a-only-agent" not in agent_names

    def test_model_exclusive_to_a_invisible_to_b(self):
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc", "call", "a-only-model"),
            integration_id=INT_A,
        )

        models_b = plugin.list_assets("model", integration_id=INT_B)
        model_names = {m["name"] for m in models_b}
        assert "a-only-model" not in model_names


# ===================================================================
# 10. list_models filtered response: schema parity with unfiltered path
# ===================================================================

class TestListModelsFilteredParity:
    """The integration-scoped fast path in _list_models must surface the same
    fields as the unfiltered path (tools, id) and must include models that are
    only reachable via tool_integration_data."""

    def test_model_tools_populated_for_integration(self):
        """A model used by a tool under integration A must report that tool
        in its `tools` field when listed with integration_id=A."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "call", "gpt-4", tool_name="web_search"),
            integration_id=INT_A,
        )

        models_a = plugin.list_assets("model", integration_id=INT_A)
        gpt4 = next((m for m in models_a if m["name"] == "gpt-4"), None)

        assert gpt4 is not None, "gpt-4 missing from integration A's model list"
        assert "web_search" in gpt4["tools"], (
            f"expected web_search in model.tools, got {gpt4['tools']}"
        )

    def test_model_id_present_in_filtered_response(self):
        """The filtered fast path must include the deterministic `id` field
        that matches _model_id_for, so lineage edges resolve downstream."""
        plugin = _make_plugin()

        plugin._ingest_traces(
            _otlp_payload("svc-a", "llm-call", "gpt-4"),
            integration_id=INT_A,
        )

        models_a = plugin.list_assets("model", integration_id=INT_A)
        gpt4 = next((m for m in models_a if m["name"] == "gpt-4"), None)

        assert gpt4 is not None
        assert "id" in gpt4, "filtered model entry missing `id` field"
        assert gpt4["id"] == plugin._model_id_for("gpt-4", integration_id=INT_A)

    def test_model_visible_when_only_in_tool_integration_data(self):
        """Defensive: if a model only appears in tool_integration_data
        (e.g. via asymmetric eviction or a future code path that registers
        a tool without touching model_integration_data), it must still
        surface in the filtered list with the tool reverse-mapped."""
        plugin = _make_plugin()

        # Directly seed only the tool-side per-integration dict.
        plugin.tool_integration_data["orphan_tool"][INT_A]["models"].add("orphan-model")
        plugin.tool_integration_data["orphan_tool"][INT_A]["trace_count"] = 1

        models_a = plugin.list_assets("model", integration_id=INT_A)
        orphan = next((m for m in models_a if m["name"] == "orphan-model"), None)

        assert orphan is not None, (
            "model only present in tool_integration_data was dropped from filtered list"
        )
        assert "orphan_tool" in orphan["tools"]
