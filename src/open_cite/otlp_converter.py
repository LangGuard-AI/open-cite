"""
OTLP Converter — Convert MLflow traces and Genie messages to OTLP JSON format.

Produces standard OpenTelemetry Protocol (OTLP) JSON payloads with GenAI
semantic conventions, suitable for forwarding to any OTLP-compatible backend.
"""

import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def _normalize_trace_id(raw_id: str) -> str:
    """Normalize an MLflow trace ID to a valid OTLP trace ID (32 hex chars).

    MLflow request_ids have format "tr-{32hex}" — strip prefix.
    If the result is shorter than 32 chars, left-pad with zeros.
    """
    tid = raw_id[3:] if raw_id.startswith("tr-") else raw_id
    return tid.zfill(32)[:32]


def _normalize_span_id(raw_id: str) -> str:
    """Normalize an MLflow span ID to a valid OTLP span ID (16 hex chars).

    If the result is shorter than 16 chars, left-pad with zeros.
    """
    return raw_id.zfill(16)[:16]


def _make_attr(key: str, value: str) -> Dict[str, Any]:
    """Build an OTLP string attribute dict."""
    return {"key": key, "value": {"stringValue": str(value)}}


def _make_attr_int(key: str, value: int) -> Dict[str, Any]:
    """Build an OTLP integer attribute dict."""
    return {"key": key, "value": {"intValue": str(int(value))}}


# MLflow attribute key -> GenAI semantic convention key
_MODEL_ATTR_MAP = {
    "llm.model": "gen_ai.request.model",
    "ai.model.name": "gen_ai.request.model",
    "mlflow.chat.model": "gen_ai.request.model",
    "model": "gen_ai.request.model",
}

_PROVIDER_ATTR_MAP = {
    "ai.model.provider": "gen_ai.system",
}

_TOOL_ATTR_MAP = {
    "tool_name": "gen_ai.tool.name",
    "mlflow.tool.name": "gen_ai.tool.name",
}

_TOKEN_INPUT_KEYS = {"llm.token_usage.input_tokens"}
_TOKEN_OUTPUT_KEYS = {"llm.token_usage.output_tokens"}

# MLflow span types that carry well-known semantics
_SKIP_PREFIXES = frozenset(_MODEL_ATTR_MAP) | frozenset(_PROVIDER_ATTR_MAP) | frozenset(_TOOL_ATTR_MAP) | _TOKEN_INPUT_KEYS | _TOKEN_OUTPUT_KEYS


def mlflow_trace_to_otlp(
    trace,
    experiment_name: str,
    service_name: str = "databricks-mlflow",
) -> Dict[str, Any]:
    """
    Convert an MLflow Trace object to an OTLP JSON payload.

    Args:
        trace: mlflow.entities.trace.Trace object with .data.spans and .info
        experiment_name: Name of the parent MLflow experiment
        service_name: Value for resource service.name attribute

    Returns:
        OTLP JSON dict: {"resourceSpans": [...]}
    """
    # Preserve original MLflow request_id before normalizing to OTLP format
    mlflow_request_id = ""
    if hasattr(trace, "info") and hasattr(trace.info, "request_id"):
        mlflow_request_id = trace.info.request_id or ""

    resource_attrs = [
        _make_attr("service.name", service_name),
        _make_attr("service.namespace", "databricks"),
        _make_attr("mlflow.experiment.name", experiment_name),
    ]
    if mlflow_request_id:
        resource_attrs.append(_make_attr("mlflow.request_id", mlflow_request_id))

    otlp_spans: List[Dict[str, Any]] = []

    for span in trace.data.spans:
        attrs = span.attributes or {}
        otlp_attrs: List[Dict[str, Any]] = []

        # Map model attributes
        for mlflow_key, genai_key in _MODEL_ATTR_MAP.items():
            val = attrs.get(mlflow_key)
            if val:
                otlp_attrs.append(_make_attr(genai_key, val))
                break  # Only emit one gen_ai.request.model

        # Map provider
        for mlflow_key, genai_key in _PROVIDER_ATTR_MAP.items():
            val = attrs.get(mlflow_key)
            if val:
                otlp_attrs.append(_make_attr(genai_key, val))

        # Map tool name
        for mlflow_key, genai_key in _TOOL_ATTR_MAP.items():
            val = attrs.get(mlflow_key)
            if val:
                otlp_attrs.append(_make_attr(genai_key, val))
                break

        # Map token usage
        for key in _TOKEN_INPUT_KEYS:
            val = attrs.get(key)
            if val is not None:
                try:
                    otlp_attrs.append(_make_attr_int("gen_ai.usage.input_tokens", int(val)))
                except (ValueError, TypeError):
                    pass

        for key in _TOKEN_OUTPUT_KEYS:
            val = attrs.get(key)
            if val is not None:
                try:
                    otlp_attrs.append(_make_attr_int("gen_ai.usage.output_tokens", int(val)))
                except (ValueError, TypeError):
                    pass

        # Agent name from span type
        span_type = (span.span_type or "").upper()
        if span_type == "AGENT":
            otlp_attrs.append(_make_attr("gen_ai.agent.name", span.name or "unknown"))

        # Carry forward remaining MLflow attributes with mlflow. prefix
        for key, val in attrs.items():
            if key not in _SKIP_PREFIXES and val is not None:
                prefixed_key = key if key.startswith("mlflow.") else f"mlflow.{key}"
                if isinstance(val, (int, float)):
                    otlp_attrs.append(_make_attr_int(prefixed_key, int(val)))
                else:
                    otlp_attrs.append(_make_attr(prefixed_key, str(val)))

        # Build span dict — normalize IDs to OTLP standard lengths
        raw_trace_id = span.trace_id if hasattr(span, "trace_id") else ""
        raw_span_id = span.span_id if hasattr(span, "span_id") else ""

        otlp_span: Dict[str, Any] = {
            "traceId": _normalize_trace_id(raw_trace_id) if raw_trace_id else "",
            "spanId": _normalize_span_id(raw_span_id) if raw_span_id else "",
            "name": span.name or "",
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": str(span.start_time_ns) if hasattr(span, "start_time_ns") else "0",
            "endTimeUnixNano": str(span.end_time_ns) if hasattr(span, "end_time_ns") else "0",
            "attributes": otlp_attrs,
            "status": {},
        }

        if hasattr(span, "parent_id") and span.parent_id:
            otlp_span["parentSpanId"] = _normalize_span_id(span.parent_id)

        otlp_spans.append(otlp_span)

    return {
        "resourceSpans": [{
            "resource": {"attributes": resource_attrs},
            "scopeSpans": [{
                "scope": {"name": "open_cite.otlp_converter"},
                "spans": otlp_spans,
            }],
        }],
    }


def _deterministic_id(seed: str, length: int = 16) -> str:
    """Generate a deterministic hex ID from a seed string via MD5."""
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:length]


def genie_trace_to_otlp(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Genie trace dict (from DatabricksPlugin._process_genie_message)
    to a synthetic OTLP JSON payload.

    Creates a 2-span trace: root AGENT span + child LLM span.

    Args:
        trace_dict: Dict with keys like id, name, timestamp, agent_name,
                    model, provider, latency_ms, token_estimate, metadata, etc.

    Returns:
        OTLP JSON dict: {"resourceSpans": [...]}
    """
    trace_id_seed = trace_dict.get("id", "unknown")
    trace_id = _deterministic_id(f"trace:{trace_id_seed}", 32)
    root_span_id = _deterministic_id(f"agent:{trace_id_seed}", 16)
    llm_span_id = _deterministic_id(f"llm:{trace_id_seed}", 16)

    # Timestamps
    ts = trace_dict.get("timestamp")
    if isinstance(ts, datetime):
        start_ns = int(ts.timestamp() * 1_000_000_000)
    else:
        start_ns = int(time.time() * 1_000_000_000)

    latency_ms = trace_dict.get("latency_ms")
    if latency_ms is not None:
        end_ns = start_ns + int(latency_ms) * 1_000_000
    else:
        end_ns = start_ns

    metadata = trace_dict.get("metadata", {})
    space_name = metadata.get("genie_space_name", "")
    space_id = metadata.get("genie_space_id", "")
    conv_id = metadata.get("genie_conversation_id", "")

    # Resource attributes
    resource_attrs = [
        _make_attr("service.name", "databricks-genie"),
        _make_attr("service.namespace", "databricks"),
    ]
    if space_name:
        resource_attrs.append(_make_attr("genie.space_name", space_name))

    # Token estimates
    token_est = trace_dict.get("token_estimate", {})
    input_tokens = token_est.get("input_tokens", 0)
    output_tokens = token_est.get("output_tokens", 0)

    user_prompt = (trace_dict.get("input") or {}).get("prompt", "")

    # Root AGENT span
    agent_attrs = [
        _make_attr("gen_ai.agent.name", trace_dict.get("agent_name", "Genie")),
    ]
    if space_id:
        agent_attrs.append(_make_attr("genie.space_id", space_id))
    if conv_id:
        agent_attrs.append(_make_attr("genie.conversation_id", conv_id))
    if user_prompt:
        agent_attrs.append(_make_attr("gen_ai.prompt", user_prompt[:1000]))

    root_span = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": trace_dict.get("agent_name", "Genie"),
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": agent_attrs,
        "status": {},
    }

    # Child LLM span
    llm_attrs = [
        _make_attr("gen_ai.request.model", trace_dict.get("model", "databricks-genie")),
        _make_attr("gen_ai.system", trace_dict.get("provider", "databricks")),
    ]
    if input_tokens:
        llm_attrs.append(_make_attr_int("gen_ai.usage.input_tokens", input_tokens))
    if output_tokens:
        llm_attrs.append(_make_attr_int("gen_ai.usage.output_tokens", output_tokens))

    llm_span = {
        "traceId": trace_id,
        "spanId": llm_span_id,
        "parentSpanId": root_span_id,
        "name": "LLM",
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": llm_attrs,
        "status": {},
    }

    return {
        "resourceSpans": [{
            "resource": {"attributes": resource_attrs},
            "scopeSpans": [{
                "scope": {"name": "open_cite.otlp_converter"},
                "spans": [root_span, llm_span],
            }],
        }],
    }
