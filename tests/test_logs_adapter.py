"""
Unit tests for the OTLP logs-to-traces adapter.
"""

import json
import os
import pytest

from open_cite.plugins.logs_adapter import (
    convert_logs_to_traces,
    _parse_otlp_attributes,
    _parse_body,
    _find_value,
    _deterministic_id,
    _coerce_int,
    _coerce_float,
    _coerce_bool,
    _map_event_to_span_name,
    _collect_sessions,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURE_DIR, name)
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Test: empty / minimal input
# -----------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_resource_logs(self):
        result = convert_logs_to_traces({"resourceLogs": []})
        assert result == {"resourceSpans": []}

    def test_missing_resource_logs_key(self):
        result = convert_logs_to_traces({})
        assert result == {"resourceSpans": []}

    def test_empty_dict(self):
        result = convert_logs_to_traces({})
        assert result == {"resourceSpans": []}

    def test_no_log_records(self):
        result = convert_logs_to_traces({
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{"logRecords": []}],
            }]
        })
        assert result == {"resourceSpans": []}


# -----------------------------------------------------------------------
# Test: single session grouping
# -----------------------------------------------------------------------


class TestSingleSessionGrouping:
    def test_single_session_from_fixture(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)

        assert len(result["resourceSpans"]) == 1
        rs = result["resourceSpans"][0]

        # Should have one scopeSpan
        assert len(rs["scopeSpans"]) == 1
        spans = rs["scopeSpans"][0]["spans"]

        # 1 root + 6 child spans
        assert len(spans) == 7

        # Root span should be "Session"
        root = spans[0]
        assert root["name"] == "Session"
        assert root.get("parentSpanId") is None or root.get("parentSpanId") == ""

        # All child spans should have parentSpanId = root spanId
        root_span_id = root["spanId"]
        for child in spans[1:]:
            assert child["parentSpanId"] == root_span_id

    def test_three_events_single_session(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [
                        _make_log_record("s1", "event_a", 100),
                        _make_log_record("s1", "event_b", 200),
                        _make_log_record("s1", "event_c", 300),
                    ]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        assert len(result["resourceSpans"]) == 1
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        # 1 root + 3 children
        assert len(spans) == 4


# -----------------------------------------------------------------------
# Test: multiple sessions
# -----------------------------------------------------------------------


class TestMultipleSessions:
    def test_two_sessions(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [
                        _make_log_record("session-A", "event_1", 100),
                        _make_log_record("session-B", "event_2", 200),
                        _make_log_record("session-A", "event_3", 300),
                    ]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        assert len(result["resourceSpans"]) == 2

        # Each session should have its own trace_id
        trace_ids = set()
        for rs in result["resourceSpans"]:
            spans = rs["scopeSpans"][0]["spans"]
            trace_ids.add(spans[0]["traceId"])
        assert len(trace_ids) == 2


# -----------------------------------------------------------------------
# Test: default session
# -----------------------------------------------------------------------


class TestDefaultSession:
    def test_no_session_id_uses_default(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [
                        {
                            "timeUnixNano": "100000000000",
                            "severityNumber": 9,
                            "attributes": [
                                {"key": "event.name", "value": {"stringValue": "test_event"}}
                            ],
                        }
                    ]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        assert len(result["resourceSpans"]) == 1

        # Verify deterministic ID based on "default-session"
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        expected_trace_id = _deterministic_id("log-session:default-session", 32)
        assert spans[0]["traceId"] == expected_trace_id


# -----------------------------------------------------------------------
# Test: event name normalization
# -----------------------------------------------------------------------


class TestEventNameNormalization:
    def test_claude_code_prefix_stripped(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]

        # Find the api_response child span (event 3, index 3 in spans list)
        api_response_span = spans[3]  # 0=root, 1=user_prompt, 2=api_request, 3=api_response
        assert api_response_span["name"] == "API Response"

        # Check the event_name attribute was stored without prefix
        event_attr = _get_attr(api_response_span["attributes"], "logs.event_name")
        assert event_attr == "api_response"

    def test_known_event_names(self):
        assert _map_event_to_span_name("user_prompt", {}) == "User Prompt"
        assert _map_event_to_span_name("api_request", {}) == "API Request"
        assert _map_event_to_span_name("api_response", {}) == "API Response"
        assert _map_event_to_span_name("api_error", {}) == "API Error"

    def test_tool_use_name(self):
        merged = {"tool_name": "Bash"}
        assert _map_event_to_span_name("tool_use", merged) == "Tool: Bash"

    def test_tool_decision_name(self):
        merged = {"tool_name": "Bash", "decision": "approve"}
        assert _map_event_to_span_name("tool_decision", merged) == "Tool Decision: Bash (approve)"

    def test_unknown_event_titlecased(self):
        assert _map_event_to_span_name("custom_event", {}) == "Custom Event"

    def test_empty_event_name(self):
        assert _map_event_to_span_name("", {}) == "Log Event"


# -----------------------------------------------------------------------
# Test: token aggregation
# -----------------------------------------------------------------------


class TestTokenAggregation:
    def test_root_span_has_summed_tokens(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        input_tokens = _get_attr(root["attributes"], "gen_ai.usage.input_tokens")
        output_tokens = _get_attr(root["attributes"], "gen_ai.usage.output_tokens")
        total_tokens = _get_attr(root["attributes"], "gen_ai.usage.total_tokens")

        # From fixture: 120 + 5 = 125 input, 80 + 2 = 82 output
        assert int(input_tokens) == 125
        assert int(output_tokens) == 82
        assert int(total_tokens) == 207


# -----------------------------------------------------------------------
# Test: cost aggregation
# -----------------------------------------------------------------------


class TestCostAggregation:
    def test_root_span_has_summed_cost(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        cost = _get_attr(root["attributes"], "gen_ai.usage.cost")
        # From fixture: 0.012 + 0.018 + 0.001 = 0.031
        assert abs(cost - 0.031) < 0.0001


# -----------------------------------------------------------------------
# Test: tool name extraction
# -----------------------------------------------------------------------


class TestToolNameExtraction:
    def test_tool_name_on_child_span(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]

        # tool_use span is index 4 (0=root, 1-6=children)
        tool_use_span = spans[4]
        tool_attr = _get_attr(tool_use_span["attributes"], "gen_ai.tool.name")
        assert tool_attr == "Bash"

    def test_tools_used_on_root(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        tools = _get_attr(root["attributes"], "logs.tools_used")
        assert tools == "Bash"


# -----------------------------------------------------------------------
# Test: agent name on root
# -----------------------------------------------------------------------


class TestAgentName:
    def test_agent_name_from_event(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [
                        {
                            "timeUnixNano": "100000000000",
                            "severityNumber": 9,
                            "attributes": [
                                {"key": "event.name", "value": {"stringValue": "api_request"}},
                                {"key": "session.id", "value": {"stringValue": "s1"}},
                                {"key": "gen_ai.agent.name", "value": {"stringValue": "Claude Code"}},
                                {"key": "gen_ai.request.model", "value": {"stringValue": "claude-3.5-sonnet"}},
                            ],
                        }
                    ]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        agent = _get_attr(root["attributes"], "gen_ai.agent.name")
        assert agent == "Claude Code"


# -----------------------------------------------------------------------
# Test: error detection
# -----------------------------------------------------------------------


class TestErrorDetection:
    def test_severity_error(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        # The fixture has severity 17 (ERROR) on the last event + success=false
        assert root.get("status", {}).get("code") == 2

    def test_error_attribute(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [{
                        "timeUnixNano": "100000000000",
                        "severityNumber": 9,
                        "attributes": [
                            {"key": "event.name", "value": {"stringValue": "api_error"}},
                            {"key": "session.id", "value": {"stringValue": "s1"}},
                            {"key": "error", "value": {"stringValue": "timeout"}},
                        ],
                    }]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert root.get("status", {}).get("code") == 2

    def test_no_error_no_status(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [{
                        "timeUnixNano": "100000000000",
                        "severityNumber": 9,
                        "attributes": [
                            {"key": "event.name", "value": {"stringValue": "user_prompt"}},
                            {"key": "session.id", "value": {"stringValue": "s1"}},
                        ],
                    }]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert "status" not in root or root.get("status", {}).get("code") != 2

    def test_success_false_triggers_error(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [{
                        "timeUnixNano": "100000000000",
                        "severityNumber": 9,
                        "attributes": [
                            {"key": "event.name", "value": {"stringValue": "tool_result"}},
                            {"key": "session.id", "value": {"stringValue": "s1"}},
                            {"key": "success", "value": {"boolValue": False}},
                        ],
                    }]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert root.get("status", {}).get("code") == 2


# -----------------------------------------------------------------------
# Test: timestamp parsing
# -----------------------------------------------------------------------


class TestTimestampParsing:
    def test_time_unix_nano(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [{
                        "timeUnixNano": "1700000000000000000",
                        "severityNumber": 9,
                        "attributes": [
                            {"key": "event.name", "value": {"stringValue": "test"}},
                            {"key": "session.id", "value": {"stringValue": "s1"}},
                        ],
                    }]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert root["startTimeUnixNano"] == "1700000000000000000"

    def test_events_ordered_by_timestamp(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [
                        _make_log_record("s1", "late", 300),
                        _make_log_record("s1", "early", 100),
                        _make_log_record("s1", "mid", 200),
                    ]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        # Skip root (index 0), check children are ordered
        child_names = [_get_attr(s["attributes"], "logs.event_name") for s in spans[1:]]
        assert child_names == ["early", "mid", "late"]


# -----------------------------------------------------------------------
# Test: body kvlistValue parsing
# -----------------------------------------------------------------------


class TestBodyParsing:
    def test_kvlist_body_contributes_to_extraction(self):
        data = {
            "resourceLogs": [{
                "resource": {"attributes": []},
                "scopeLogs": [{
                    "logRecords": [{
                        "timeUnixNano": "100000000000",
                        "severityNumber": 9,
                        "attributes": [
                            {"key": "session.id", "value": {"stringValue": "s1"}},
                        ],
                        "body": {
                            "kvlistValue": {
                                "values": [
                                    {"key": "event.name", "value": {"stringValue": "api_request"}},
                                    {"key": "model", "value": {"stringValue": "gpt-4"}},
                                ]
                            }
                        },
                    }]
                }],
            }]
        }
        sessions = _collect_sessions(data["resourceLogs"])
        session = list(sessions.values())[0]
        assert session.events[0].event_name == "api_request"


# -----------------------------------------------------------------------
# Test: model extraction
# -----------------------------------------------------------------------


class TestModelExtraction:
    def test_model_on_root_span(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        model = _get_attr(root["attributes"], "gen_ai.request.model")
        assert model == "claude-3.5-sonnet"

    def test_model_on_child_span(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]

        # api_request span is index 2 (0=root, 1=user_prompt, 2=api_request)
        api_req = spans[2]
        model = _get_attr(api_req["attributes"], "gen_ai.request.model")
        assert model == "claude-3.5-sonnet"


# -----------------------------------------------------------------------
# Test: deterministic IDs
# -----------------------------------------------------------------------


class TestDeterministicIds:
    def test_same_session_same_ids(self):
        id1 = _deterministic_id("log-session:abc", 32)
        id2 = _deterministic_id("log-session:abc", 32)
        assert id1 == id2
        assert len(id1) == 32

    def test_different_sessions_different_ids(self):
        id1 = _deterministic_id("log-session:abc", 32)
        id2 = _deterministic_id("log-session:def", 32)
        assert id1 != id2

    def test_16_char_span_id(self):
        sid = _deterministic_id("log-root:abc", 16)
        assert len(sid) == 16


# -----------------------------------------------------------------------
# Test: roundtrip OTLP structure validation
# -----------------------------------------------------------------------


class TestOtlpStructure:
    def test_output_has_valid_structure(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)

        assert "resourceSpans" in result
        for rs in result["resourceSpans"]:
            assert "resource" in rs
            assert "attributes" in rs["resource"]
            assert "scopeSpans" in rs
            for ss in rs["scopeSpans"]:
                assert "scope" in ss
                assert "spans" in ss
                for span in ss["spans"]:
                    assert "traceId" in span
                    assert "spanId" in span
                    assert "name" in span
                    assert "startTimeUnixNano" in span
                    assert "endTimeUnixNano" in span
                    assert "attributes" in span

    def test_resource_attributes_carried_over(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        rs = result["resourceSpans"][0]

        resource_attrs = {a["key"]: a for a in rs["resource"]["attributes"]}
        assert "service.name" in resource_attrs
        assert resource_attrs["service.name"]["value"]["stringValue"] == "claude-code"

    def test_user_id_on_root(self):
        data = _load_fixture("sample_logs_payload.json")
        result = convert_logs_to_traces(data)
        root = result["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        user_id = _get_attr(root["attributes"], "enduser.id")
        assert user_id == "developer-123"


# -----------------------------------------------------------------------
# Test: helper functions
# -----------------------------------------------------------------------


class TestHelpers:
    def test_parse_otlp_attributes(self):
        attrs = [
            {"key": "k1", "value": {"stringValue": "v1"}},
            {"key": "k2", "value": {"intValue": 42}},
            {"key": "k3", "value": {"doubleValue": 3.14}},
            {"key": "k4", "value": {"boolValue": True}},
        ]
        result = _parse_otlp_attributes(attrs)
        assert result == {"k1": "v1", "k2": 42, "k3": 3.14, "k4": True}

    def test_find_value_priority(self):
        m1 = {"a": 1, "b": 2}
        m2 = {"a": 10, "c": 3}
        assert _find_value(("a",), m1, m2) == 1
        assert _find_value(("c",), m1, m2) == 3
        assert _find_value(("x",), m1, m2) is None

    def test_find_value_skips_empty(self):
        m1 = {"a": ""}
        m2 = {"a": "real"}
        assert _find_value(("a",), m1, m2) == "real"

    def test_coerce_int(self):
        assert _coerce_int("42") == 42
        assert _coerce_int(42) == 42
        assert _coerce_int(None) == 0
        assert _coerce_int("abc") == 0

    def test_coerce_float(self):
        assert _coerce_float("3.14") == 3.14
        assert _coerce_float(None) == 0.0

    def test_coerce_bool(self):
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False
        assert _coerce_bool("true") is True
        assert _coerce_bool("false") is False
        assert _coerce_bool(1) is True
        assert _coerce_bool(0) is False


# -----------------------------------------------------------------------
# Test: session ID from resource attributes
# -----------------------------------------------------------------------


class TestSessionIdFromResource:
    def test_session_id_from_resource(self):
        data = {
            "resourceLogs": [{
                "resource": {
                    "attributes": [
                        {"key": "session.id", "value": {"stringValue": "from-resource"}},
                    ]
                },
                "scopeLogs": [{
                    "logRecords": [{
                        "timeUnixNano": "100000000000",
                        "severityNumber": 9,
                        "attributes": [
                            {"key": "event.name", "value": {"stringValue": "test"}},
                        ],
                    }]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        expected_trace_id = _deterministic_id("log-session:from-resource", 32)
        assert spans[0]["traceId"] == expected_trace_id

    def test_log_attr_session_id_takes_priority(self):
        data = {
            "resourceLogs": [{
                "resource": {
                    "attributes": [
                        {"key": "session.id", "value": {"stringValue": "from-resource"}},
                    ]
                },
                "scopeLogs": [{
                    "logRecords": [{
                        "timeUnixNano": "100000000000",
                        "severityNumber": 9,
                        "attributes": [
                            {"key": "event.name", "value": {"stringValue": "test"}},
                            {"key": "session.id", "value": {"stringValue": "from-log-attr"}},
                        ],
                    }]
                }],
            }]
        }
        result = convert_logs_to_traces(data)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        expected_trace_id = _deterministic_id("log-session:from-log-attr", 32)
        assert spans[0]["traceId"] == expected_trace_id


# -----------------------------------------------------------------------
# Helpers for test data construction
# -----------------------------------------------------------------------


def _make_log_record(session_id: str, event_name: str, timestamp_ns: int, **extra_attrs) -> dict:
    """Create a minimal log record for testing."""
    attrs = [
        {"key": "event.name", "value": {"stringValue": event_name}},
        {"key": "session.id", "value": {"stringValue": session_id}},
    ]
    for k, v in extra_attrs.items():
        if isinstance(v, str):
            attrs.append({"key": k, "value": {"stringValue": v}})
        elif isinstance(v, int):
            attrs.append({"key": k, "value": {"intValue": v}})
        elif isinstance(v, float):
            attrs.append({"key": k, "value": {"doubleValue": v}})
        elif isinstance(v, bool):
            attrs.append({"key": k, "value": {"boolValue": v}})
    return {
        "timeUnixNano": str(timestamp_ns),
        "severityNumber": 9,
        "attributes": attrs,
    }


def _get_attr(attributes: list, key: str):
    """Extract a value from an OTLP attributes list by key."""
    for attr in attributes:
        if attr["key"] == key:
            val = attr["value"]
            if "stringValue" in val:
                return val["stringValue"]
            if "intValue" in val:
                return val["intValue"]
            if "doubleValue" in val:
                return val["doubleValue"]
            if "boolValue" in val:
                return val["boolValue"]
    return None
