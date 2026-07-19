"""
Google Cloud Trace -> OTLP converter.

Converts Cloud Trace v1 traces (as returned by
``cloudtrace.googleapis.com/v1/projects/{p}/traces?view=COMPLETE``) into standard
OpenTelemetry Protocol (OTLP) JSON payloads with GenAI semantic conventions, so a
Vertex agent's activity can flow through the same OTel ingestion/detection path as
any other traced app and land in the trace explorer regardless of allow/block.

A Cloud Trace span carries its OTel attributes as string ``labels``; Vertex/ADK
agents emit gen_ai.* conventions, which survive as labels and are passed through
(plus a few well-known aliases are normalized to gen_ai.*).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# GenAI semantic-convention aliases seen in Vertex/ADK/GCP Cloud Trace labels.
_MODEL_ALIASES = {
    "gen_ai.request.model": "gen_ai.request.model",
    "gen_ai.response.model": "gen_ai.request.model",
    "llm.model": "gen_ai.request.model",
    "model": "gen_ai.request.model",
    "vertex.model": "gen_ai.request.model",
}
_PROVIDER_ALIASES = {
    "gen_ai.system": "gen_ai.system",
    "gen_ai.provider.name": "gen_ai.system",
    "llm.provider": "gen_ai.system",
}
_TOOL_ALIASES = {
    "gen_ai.tool.name": "gen_ai.tool.name",
    "tool_name": "gen_ai.tool.name",
    "tool.name": "gen_ai.tool.name",
}
_AGENT_ALIASES = {
    "gen_ai.agent.name": "gen_ai.agent.name",
    "agent.name": "gen_ai.agent.name",
    "agent_name": "gen_ai.agent.name",
}
_INPUT_TOKEN_ALIASES = {"gen_ai.usage.input_tokens", "llm.usage.prompt_tokens", "llm.token_usage.input_tokens"}
_OUTPUT_TOKEN_ALIASES = {"gen_ai.usage.output_tokens", "llm.usage.completion_tokens", "llm.token_usage.output_tokens"}


def _make_attr(key: str, value: str) -> Dict[str, Any]:
    return {"key": key, "value": {"stringValue": str(value)}}


def _make_attr_int(key: str, value: int) -> Dict[str, Any]:
    return {"key": key, "value": {"intValue": str(int(value))}}


def _span_id_to_hex(raw: Any) -> str:
    """Cloud Trace span ids are uint64 decimal strings; OTLP wants 16 hex chars."""
    try:
        return format(int(raw), "016x")[:16]
    except (TypeError, ValueError):
        s = str(raw or "")
        return s.zfill(16)[:16] if s else ""


def _rfc3339_to_unix_nano(ts: Optional[str]) -> Optional[str]:
    """Parse an RFC3339 timestamp (optionally with fractional seconds / 'Z') to
    a Unix-nanoseconds string, preserving sub-microsecond precision."""
    if not ts:
        return None
    try:
        s = ts.strip()
        frac_ns = 0
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        # Split fractional seconds to keep full ns precision (fromisoformat caps at us).
        if "." in s:
            head, rest = s.split(".", 1)
            digits = ""
            tail = ""
            for i, ch in enumerate(rest):
                if ch.isdigit():
                    digits += ch
                else:
                    tail = rest[i:]
                    break
            frac_ns = int((digits + "000000000")[:9])
            s = head + tail
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        whole = int(dt.timestamp())
        return str(whole * 1_000_000_000 + frac_ns)
    except Exception:
        return None


def _kind_to_otlp(kind: Optional[str]) -> int:
    return {
        "RPC_SERVER": 2,       # SPAN_KIND_SERVER
        "RPC_CLIENT": 3,       # SPAN_KIND_CLIENT
        "SPAN_KIND_UNSPECIFIED": 1,
    }.get(kind or "", 1)       # default SPAN_KIND_INTERNAL


def _is_infra_span(labels: Dict[str, Any]) -> bool:
    """Google-managed Agent Gateway *infrastructure* spans carry no agent identity
    and are redundant with the agent's own tool-call spans, so surfacing them in
    the explorer only produces unattributed "Unknown" traces. Skip them: the
    gateway's Envoy HTTP load-balancer egress hops tag ``/component: HTTP load
    balancer``, the downstream Cloud Run MCP tool servers tag ``/component:
    AppServer``, and Model Armor guardrail spans tag ``service.name: modelarmor``.
    A span that carries an agent identity is always kept."""
    if labels.get("gen_ai.agent.name") or labels.get("agent.name"):
        return False
    return (
        labels.get("/component") in ("HTTP load balancer", "AppServer")
        or labels.get("service.name") == "modelarmor"
    )


def cloud_trace_to_otlp(trace: Dict[str, Any], project_id: str) -> Optional[Dict[str, Any]]:
    """Convert one Cloud Trace v1 trace (dict from the REST list_traces response)
    into an OTLP ``{"resourceSpans": [...]}`` payload. Returns None if the trace
    has no usable spans."""
    trace_id = trace.get("traceId") or ""
    raw_spans = trace.get("spans") or []
    if not trace_id or not raw_spans:
        return None

    otlp_spans: List[Dict[str, Any]] = []
    agent_name: Optional[str] = None
    model_name: Optional[str] = None

    for sp in raw_spans:
        start_ns = _rfc3339_to_unix_nano(sp.get("startTime"))
        end_ns = _rfc3339_to_unix_nano(sp.get("endTime"))
        # Accumulate by key so a normalized alias replaces (never duplicates) the
        # raw label it derives from.
        attr_by_key: Dict[str, Dict[str, Any]] = {}
        labels = sp.get("labels") or {}
        if _is_infra_span(labels):
            continue
        for k, v in labels.items():
            if k in _INPUT_TOKEN_ALIASES:
                try:
                    attr_by_key["gen_ai.usage.input_tokens"] = _make_attr_int(
                        "gen_ai.usage.input_tokens", int(v)
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            elif k in _OUTPUT_TOKEN_ALIASES:
                try:
                    attr_by_key["gen_ai.usage.output_tokens"] = _make_attr_int(
                        "gen_ai.usage.output_tokens", int(v)
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            # Keep the raw label, and set the normalized alias key (may equal k).
            attr_by_key.setdefault(k, _make_attr(k, v))
            if k in _MODEL_ALIASES:
                model_name = model_name or str(v)
                attr_by_key[_MODEL_ALIASES[k]] = _make_attr(_MODEL_ALIASES[k], v)
            elif k in _PROVIDER_ALIASES:
                attr_by_key[_PROVIDER_ALIASES[k]] = _make_attr(_PROVIDER_ALIASES[k], v)
            elif k in _TOOL_ALIASES:
                attr_by_key[_TOOL_ALIASES[k]] = _make_attr(_TOOL_ALIASES[k], v)
            elif k in _AGENT_ALIASES:
                agent_name = agent_name or str(v)
                attr_by_key[_AGENT_ALIASES[k]] = _make_attr(_AGENT_ALIASES[k], v)
        attrs: List[Dict[str, Any]] = list(attr_by_key.values())

        span: Dict[str, Any] = {
            "traceId": trace_id,
            "spanId": _span_id_to_hex(sp.get("spanId")),
            "name": sp.get("name") or "vertex.span",
            "kind": _kind_to_otlp(sp.get("kind")),
            "attributes": attrs,
            "status": {},
        }
        parent = sp.get("parentSpanId")
        if parent and str(parent) != "0":
            span["parentSpanId"] = _span_id_to_hex(parent)
        if start_ns:
            span["startTimeUnixNano"] = start_ns
        if end_ns:
            span["endTimeUnixNano"] = end_ns
        otlp_spans.append(span)

    if not otlp_spans:
        return None

    resource_attrs = [
        _make_attr("service.name", agent_name or "vertex"),
        _make_attr("gen_ai.system", "vertex_ai"),
        _make_attr("gen_ai.provider.name", "vertex_ai"),
        _make_attr("cloud.provider", "gcp"),
        _make_attr("gcp.project_id", project_id),
    ]
    if agent_name:
        resource_attrs.append(_make_attr("gen_ai.agent.name", agent_name))
    if model_name:
        resource_attrs.append(_make_attr("gen_ai.request.model", model_name))

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [
                    {
                        "scope": {"name": "open_cite.google_cloud"},
                        "spans": otlp_spans,
                    }
                ],
            }
        ]
    }


def merge_otlp(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge several single-trace OTLP payloads into one batched payload."""
    resource_spans: List[Dict[str, Any]] = []
    for p in payloads:
        if p:
            resource_spans.extend(p.get("resourceSpans", []))
    return {"resourceSpans": resource_spans}
