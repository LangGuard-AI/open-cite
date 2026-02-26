"""
OTLP Logs-to-Traces adapter for Open-CITE.

Converts OTLP log records (resourceLogs) into synthetic OTLP trace spans
(resourceSpans) so that the existing entity discovery engine can process
telemetry from log-emitting sources like Claude Code.

The public API is a single function:

    convert_logs_to_traces(otlp_logs_data: dict) -> dict

The output is a standard OTLP ``{"resourceSpans": [...]}`` payload that can
be passed directly to ``OpenTelemetryPlugin._ingest_traces()``.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attribute key aliases (mirrors the Go adapter's search strategy)
# ---------------------------------------------------------------------------

SESSION_KEYS = ("session.id", "session_id", "gen_ai.conversation.id")
EVENT_NAME_KEYS = ("event.name", "event_name", "name")
MODEL_KEYS = (
    "model", "gen_ai.request.model", "gen_ai.response.model",
    "llm.model", "gen_ai.model",
)
PROVIDER_KEYS = (
    "provider", "provider_name", "gen_ai.system",
    "gen_ai.provider.name", "llm.provider",
)
INPUT_TOKEN_KEYS = ("input_tokens", "gen_ai.usage.input_tokens")
OUTPUT_TOKEN_KEYS = ("output_tokens", "gen_ai.usage.output_tokens")
TOTAL_TOKEN_KEYS = ("total_tokens", "gen_ai.usage.total_tokens")
COST_KEYS = ("cost_usd", "cost", "gen_ai.usage.cost")
TOOL_NAME_KEYS = (
    "tool_name", "tool.name", "gen_ai.tool.name",
    "llm.tool.name", "ai.tool.name",
)
AGENT_NAME_KEYS = ("gen_ai.agent.name", "agent_name", "agent.name")
USER_KEYS = ("user.id", "user.account_uuid", "enduser.id")

# OTLP severity number threshold for ERROR (matches plog.SeverityNumberError)
_SEVERITY_ERROR = 17

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LogEventContext:
    """Parsed context for a single OTLP log record."""
    log_attributes: Dict[str, Any]
    resource_attributes: Dict[str, Any]
    body: Any
    body_map: Optional[Dict[str, Any]]
    session_id: str
    event_name: str
    timestamp_ns: int
    event_index: int
    severity_number: int = 0
    span_id: str = ""


@dataclass
class SessionGroup:
    """A group of log events sharing the same session ID."""
    session_id: str
    events: List[LogEventContext] = field(default_factory=list)
    resource_attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_logs_to_traces(otlp_logs_data: dict) -> dict:
    """Convert OTLP logs JSON (resourceLogs) to synthetic OTLP traces JSON (resourceSpans).

    Args:
        otlp_logs_data: A dict with a ``resourceLogs`` key containing OTLP
            log records in JSON/dict form.

    Returns:
        A dict with a ``resourceSpans`` key containing synthetic OTLP trace
        spans suitable for ingestion by ``_ingest_traces()``.
    """
    resource_logs = otlp_logs_data.get("resourceLogs", [])
    if not resource_logs:
        return {"resourceSpans": []}

    # 1. Collect and group log records by session
    sessions = _collect_sessions(resource_logs)
    if not sessions:
        return {"resourceSpans": []}

    # 2. Build synthetic resourceSpans for each session
    resource_spans = []
    for session in sessions.values():
        rs = _build_resource_span(session)
        if rs:
            resource_spans.append(rs)

    return {"resourceSpans": resource_spans}


# ---------------------------------------------------------------------------
# Session collection
# ---------------------------------------------------------------------------


def _collect_sessions(resource_logs: list) -> Dict[str, SessionGroup]:
    """Iterate over resourceLogs and group log records by session ID."""
    sessions: Dict[str, SessionGroup] = {}
    global_index = 0

    for rl in resource_logs:
        resource = rl.get("resource", {})
        resource_attrs = _parse_otlp_attributes(resource.get("attributes", []))

        for sl in rl.get("scopeLogs", []):
            for log_record in sl.get("logRecords", []):
                log_attrs = _parse_otlp_attributes(log_record.get("attributes", []))
                body_raw = log_record.get("body")
                body_map = _parse_body(body_raw)

                # Extract session ID
                session_id = _find_value(
                    SESSION_KEYS, log_attrs, body_map, resource_attrs,
                ) or "default-session"

                # Extract event name
                event_name = _find_value(
                    EVENT_NAME_KEYS, log_attrs, body_map,
                ) or ""
                # Strip common prefixes
                for prefix in ("claude_code.", "genai."):
                    if event_name.startswith(prefix):
                        event_name = event_name[len(prefix):]

                # Parse timestamp
                timestamp_ns = _parse_timestamp(log_record, log_attrs, body_map)

                # Build event context
                evt = LogEventContext(
                    log_attributes=log_attrs,
                    resource_attributes=resource_attrs,
                    body=body_raw,
                    body_map=body_map,
                    session_id=session_id,
                    event_name=event_name,
                    timestamp_ns=timestamp_ns,
                    event_index=global_index,
                    severity_number=int(log_record.get("severityNumber", 0)),
                    span_id=log_record.get("spanId", ""),
                )
                global_index += 1

                if session_id not in sessions:
                    sessions[session_id] = SessionGroup(
                        session_id=session_id,
                        resource_attributes=resource_attrs,
                        source=resource_attrs.get("service.name", "opentelemetry-logs"),
                    )
                sessions[session_id].events.append(evt)

    # Sort events within each session by timestamp
    for session in sessions.values():
        session.events.sort(key=lambda e: (e.timestamp_ns, e.event_index))

    return sessions


# ---------------------------------------------------------------------------
# Synthetic span building
# ---------------------------------------------------------------------------


def _build_resource_span(session: SessionGroup) -> Optional[dict]:
    """Build a single ``resourceSpan`` entry for a session group."""
    if not session.events:
        return None

    trace_id = _deterministic_id(f"log-session:{session.session_id}", 32)
    root_span_id = _deterministic_id(f"log-root:{session.session_id}", 16)

    first_ts = session.events[0].timestamp_ns
    last_ts = session.events[-1].timestamp_ns

    # Aggregate metrics across all events
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    has_error = False
    first_model = ""
    first_provider = ""
    agent_name = ""
    tools_seen: set = set()

    child_spans = []

    for evt in session.events:
        merged = {**evt.resource_attributes, **evt.log_attributes}
        if evt.body_map:
            for k, v in evt.body_map.items():
                if k not in merged:
                    merged[k] = v

        # Token aggregation
        inp = _coerce_int(_find_value(INPUT_TOKEN_KEYS, merged))
        out = _coerce_int(_find_value(OUTPUT_TOKEN_KEYS, merged))
        cost = _coerce_float(_find_value(COST_KEYS, merged))
        total_input_tokens += inp
        total_output_tokens += out
        total_cost += cost

        # Model / provider (first non-empty wins)
        model = _find_value(MODEL_KEYS, merged) or ""
        provider = _find_value(PROVIDER_KEYS, merged) or ""
        if model and not first_model:
            first_model = str(model)
        if provider and not first_provider:
            first_provider = str(provider)

        # Agent name
        an = _find_value(AGENT_NAME_KEYS, merged) or ""
        if an and not agent_name:
            agent_name = str(an)

        # Tool name
        tool = _find_value(TOOL_NAME_KEYS, merged) or ""
        if tool:
            tools_seen.add(str(tool))

        # Error detection
        error_msg = merged.get("error") or merged.get("error_message") or ""
        tool_success = merged.get("success")
        is_error = (
            evt.severity_number >= _SEVERITY_ERROR
            or bool(error_msg)
            or "error" in evt.event_name.lower()
            or (tool_success is not None and not _coerce_bool(tool_success))
        )
        if is_error:
            has_error = True

        # Build child span
        child_span_id = evt.span_id or _deterministic_id(
            f"{session.session_id}-{evt.timestamp_ns}-{evt.event_index}", 16
        )

        child_attrs = []

        # Event metadata
        child_attrs.append(_make_attr("logs.event_name", evt.event_name))
        child_attrs.append(_make_attr("logs.adapter", "opencite-logs-adapter"))

        # Model
        if model:
            child_attrs.append(_make_attr("gen_ai.request.model", str(model)))

        # Tool — include tool.call.id so the detection engine recognises
        # these child spans as explicit tool invocations.
        if tool:
            child_attrs.append(_make_attr("gen_ai.tool.name", str(tool)))
            _is_tool_event = evt.event_name in (
                "tool_use", "tool_result", "tool_decision",
            )
            if _is_tool_event:
                _call_id = _deterministic_id(
                    f"{session.session_id}-{tool}-{evt.timestamp_ns}-{evt.event_index}",
                    16,
                )
                child_attrs.append(_make_attr("gen_ai.tool.call.id", _call_id))

        # Tokens
        if inp:
            child_attrs.append(_make_attr_int("gen_ai.usage.input_tokens", inp))
        if out:
            child_attrs.append(_make_attr_int("gen_ai.usage.output_tokens", out))

        # Cost
        if cost:
            child_attrs.append(_make_attr_double("gen_ai.usage.cost", cost))

        # Tool parameters (Claude Code logs include input/parameters for tool calls)
        tool_params = merged.get("tool_parameters") or merged.get("tool_input") or ""
        if tool_params:
            child_attrs.append(_make_attr("tool_parameters", str(tool_params)))

        # Agent
        if an:
            child_attrs.append(_make_attr("gen_ai.agent.name", str(an)))

        # User ID
        user_id = _find_value(USER_KEYS, merged) or ""
        if user_id:
            child_attrs.append(_make_attr("enduser.id", str(user_id)))

        # Duration
        duration_ms = _coerce_int(merged.get("duration_ms"))
        end_time_ns = evt.timestamp_ns + (duration_ms * 1_000_000 if duration_ms else 0)

        # Span status
        status = {}
        if is_error:
            status = {"code": 2, "message": str(error_msg) if error_msg else "error"}

        child_span = {
            "traceId": trace_id,
            "spanId": child_span_id,
            "parentSpanId": root_span_id,
            "name": _map_event_to_span_name(evt.event_name, merged),
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": str(evt.timestamp_ns),
            "endTimeUnixNano": str(end_time_ns),
            "attributes": child_attrs,
        }
        if status:
            child_span["status"] = status

        child_spans.append(child_span)

    # Resolve agent name: explicit from events, else fall back to service.name
    if not agent_name and session.source:
        agent_name = session.source

    # Build root span (named after the agent, not generic "Session")
    root_attrs = [
        _make_attr("logs.adapter", "opencite-logs-adapter"),
        _make_attr("logs.session_id", session.session_id),
        _make_attr_int("logs.event_count", len(session.events)),
    ]

    if agent_name:
        root_attrs.append(_make_attr("gen_ai.agent.name", agent_name))
    if first_model:
        root_attrs.append(_make_attr("gen_ai.request.model", first_model))
    if first_provider:
        root_attrs.append(_make_attr("gen_ai.system", first_provider))
    if total_input_tokens:
        root_attrs.append(_make_attr_int("gen_ai.usage.input_tokens", total_input_tokens))
    if total_output_tokens:
        root_attrs.append(_make_attr_int("gen_ai.usage.output_tokens", total_output_tokens))
    total_tokens = total_input_tokens + total_output_tokens
    if total_tokens:
        root_attrs.append(_make_attr_int("gen_ai.usage.total_tokens", total_tokens))
    if total_cost:
        root_attrs.append(_make_attr_double("gen_ai.usage.cost", total_cost))
    if tools_seen:
        root_attrs.append(_make_attr("logs.tools_used", ",".join(sorted(tools_seen))))

    # User ID on root (from first event with one)
    for evt in session.events:
        merged = {**evt.resource_attributes, **evt.log_attributes}
        if evt.body_map:
            merged.update(evt.body_map)
        uid = _find_value(USER_KEYS, merged)
        if uid:
            root_attrs.append(_make_attr("enduser.id", str(uid)))
            break

    root_status = {}
    if has_error:
        root_status = {"code": 2, "message": "error"}

    root_span_name = agent_name or session.source or "Session"
    root_span = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": root_span_name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": str(first_ts),
        "endTimeUnixNano": str(last_ts),
        "attributes": root_attrs,
    }
    if root_status:
        root_span["status"] = root_status

    # Build resource attributes for the synthetic payload
    resource_attrs_list = []
    if session.source:
        resource_attrs_list.append(_make_attr("service.name", session.source))

    # Carry over resource attributes
    for key, val in session.resource_attributes.items():
        if key == "service.name":
            continue  # already added
        if isinstance(val, str):
            resource_attrs_list.append(_make_attr(key, val))
        elif isinstance(val, int):
            resource_attrs_list.append(_make_attr_int(key, val))
        elif isinstance(val, float):
            resource_attrs_list.append(_make_attr_double(key, val))
        elif isinstance(val, bool):
            resource_attrs_list.append({"key": key, "value": {"boolValue": val}})

    all_spans = [root_span] + child_spans

    return {
        "resource": {
            "attributes": resource_attrs_list,
        },
        "scopeSpans": [
            {
                "scope": {"name": "opencite-logs-adapter", "version": "1.0.0"},
                "spans": all_spans,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Helpers: attribute parsing
# ---------------------------------------------------------------------------


def _parse_otlp_attributes(attrs: list) -> Dict[str, Any]:
    """Convert OTLP attributes list to a flat dict."""
    result = {}
    for attr in attrs:
        key = attr.get("key", "")
        value = attr.get("value", {})
        result[key] = _extract_otlp_value(value)
    return result


def _extract_otlp_value(value: dict) -> Any:
    """Extract a native Python value from an OTLP value wrapper."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return _coerce_int(value["intValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "kvlistValue" in value:
        kvlist = value["kvlistValue"]
        result = {}
        for kv in kvlist.get("values", []):
            k = kv.get("key", "")
            v = kv.get("value", {})
            result[k] = _extract_otlp_value(v)
        return result
    if "arrayValue" in value:
        return [_extract_otlp_value(v) for v in value["arrayValue"].get("values", [])]
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def _parse_body(body: Any) -> Optional[Dict[str, Any]]:
    """Parse a log record body into a flat dict if possible."""
    if body is None:
        return None
    if isinstance(body, dict):
        # kvlistValue body
        if "kvlistValue" in body:
            kvlist = body["kvlistValue"]
            result = {}
            for kv in kvlist.get("values", []):
                k = kv.get("key", "")
                v = kv.get("value", {})
                result[k] = _extract_otlp_value(v)
            return result
        # stringValue body — try JSON parse
        if "stringValue" in body:
            return _parse_json_if_possible(body["stringValue"])
        return body
    if isinstance(body, str):
        return _parse_json_if_possible(body)
    return None


def _parse_json_if_possible(s: str) -> Optional[Dict[str, Any]]:
    """Try to parse a string as JSON, return dict or None."""
    s = s.strip()
    if s.startswith("{"):
        try:
            result = json.loads(s)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _find_value(keys: tuple, *maps: Any) -> Any:
    """Search multiple dicts for the first meaningful value matching any key."""
    for m in maps:
        if m is None or not isinstance(m, dict):
            continue
        for key in keys:
            val = m.get(key)
            if val is not None and val != "" and val != 0:
                return val
    return None


# ---------------------------------------------------------------------------
# Helpers: timestamp parsing
# ---------------------------------------------------------------------------


def _parse_timestamp(log_record: dict, log_attrs: dict, body_map: Optional[dict]) -> int:
    """Parse the timestamp from a log record, returning nanoseconds since epoch."""
    # Priority 1: event.timestamp from attributes/body (ISO 8601 string)
    for key in ("event.timestamp", "timestamp"):
        val = log_attrs.get(key)
        if not val and body_map:
            val = body_map.get(key)
        if val:
            if isinstance(val, str):
                ns = _parse_iso_timestamp(val)
                if ns:
                    return ns
            elif isinstance(val, (int, float)):
                return int(val)

    # Priority 2: timeUnixNano from the log record itself
    time_unix_nano = log_record.get("timeUnixNano")
    if time_unix_nano:
        try:
            return int(time_unix_nano)
        except (ValueError, TypeError):
            pass

    # Priority 3: observedTimeUnixNano
    observed = log_record.get("observedTimeUnixNano")
    if observed:
        try:
            return int(observed)
        except (ValueError, TypeError):
            pass

    # Fallback: current time
    return int(time.time() * 1_000_000_000)


def _parse_iso_timestamp(s: str) -> Optional[int]:
    """Parse an ISO 8601 timestamp string to nanoseconds since epoch."""
    from datetime import datetime, timezone
    try:
        # Handle common ISO formats
        s = s.rstrip("Z").replace("+00:00", "")
        if "T" in s:
            # Try parsing with fractional seconds
            if "." in s:
                dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# Helpers: type coercion
# ---------------------------------------------------------------------------


def _coerce_int(val: Any) -> int:
    """Coerce a value to int, returning 0 on failure."""
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _coerce_float(val: Any) -> float:
    """Coerce a value to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _coerce_bool(val: Any) -> bool:
    """Coerce a value to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    if isinstance(val, (int, float)):
        return val != 0
    return bool(val)


# ---------------------------------------------------------------------------
# Helpers: ID generation
# ---------------------------------------------------------------------------


def _deterministic_id(seed: str, length: int) -> str:
    """Generate a deterministic hex ID from a seed string.

    Args:
        seed: Input string to hash.
        length: Desired length of hex output (16 or 32).

    Returns:
        Hex string of the specified length.
    """
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return h[:length]


# ---------------------------------------------------------------------------
# Helpers: event name mapping
# ---------------------------------------------------------------------------

_EVENT_NAME_MAP = {
    "user_prompt": "User Prompt",
    "api_request": "API Request",
    "api_response": "API Response",
    "api_error": "API Error",
}


def _map_event_to_span_name(event_name: str, merged: dict) -> str:
    """Map a log event name to a human-readable span name."""
    if event_name in _EVENT_NAME_MAP:
        return _EVENT_NAME_MAP[event_name]

    tool_name = _find_value(TOOL_NAME_KEYS, merged) or "unknown"

    if event_name == "tool_use":
        return f"Tool: {tool_name}"

    if event_name == "tool_result":
        return f"Tool: {tool_name}"

    if event_name == "tool_decision":
        decision = merged.get("decision", "")
        return f"Tool Decision: {tool_name} ({decision})" if decision else f"Tool Decision: {tool_name}"

    if event_name:
        return event_name.replace("_", " ").title()

    return "Log Event"


# ---------------------------------------------------------------------------
# Helpers: OTLP attribute construction
# ---------------------------------------------------------------------------


def _make_attr(key: str, value: str) -> dict:
    """Build an OTLP string attribute."""
    return {"key": key, "value": {"stringValue": value}}


def _make_attr_int(key: str, value: int) -> dict:
    """Build an OTLP int attribute."""
    return {"key": key, "value": {"intValue": str(value)}}


def _make_attr_double(key: str, value: float) -> dict:
    """Build an OTLP double attribute."""
    return {"key": key, "value": {"doubleValue": value}}
