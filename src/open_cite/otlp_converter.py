"""
OTLP Converter — Convert MLflow traces and Genie messages to OTLP JSON format.

Produces standard OpenTelemetry Protocol (OTLP) JSON payloads with GenAI
semantic conventions, suitable for forwarding to any OTLP-compatible backend.
"""

import hashlib
import json
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

    # Extract user identity — priority: trace tags > experiment path
    # 1. trace.info.tags['mlflow.user'] — the actual user who ran the trace
    # 2. experiment_name /Users/<email>/... — the experiment owner
    user_id = None
    if hasattr(trace, "info"):
        tags = getattr(trace.info, "tags", None) or {}
        if isinstance(tags, dict):
            user_id = tags.get("mlflow.user")
        # Also extract notebook path from request_metadata
        req_meta = getattr(trace.info, "request_metadata", None) or {}
        if isinstance(req_meta, dict):
            notebook_path = req_meta.get("mlflow.databricks.notebookPath")
            if notebook_path:
                resource_attrs.append(_make_attr("mlflow.notebookPath", notebook_path))
    if not user_id and experiment_name and experiment_name.startswith("/Users/"):
        parts = experiment_name.split("/")
        if len(parts) >= 3 and parts[2]:
            user_id = parts[2]
    if user_id:
        resource_attrs.append(_make_attr("enduser.id", user_id))

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
    output_data = trace_dict.get("output") or {}

    # Root AGENT span
    agent_attrs = [
        _make_attr("gen_ai.agent.name", trace_dict.get("agent_name", "Genie")),
    ]
    if space_id:
        agent_attrs.append(_make_attr("genie.space_id", space_id))
    if conv_id:
        agent_attrs.append(_make_attr("genie.conversation_id", conv_id))

    # User ID (mapped to enduser.id for OTLP convention)
    user_id = trace_dict.get("user_id")
    if user_id:
        agent_attrs.append(_make_attr("enduser.id", user_id))

    # Session ID (conversation maps to session for trace grouping)
    session_id = trace_dict.get("session_id")
    if session_id:
        agent_attrs.append(_make_attr("session.id", session_id))

    # Input/output
    if user_prompt:
        agent_attrs.append(_make_attr("gen_ai.prompt", user_prompt))
    if output_data:
        agent_attrs.append(_make_attr("gen_ai.completion", json.dumps(output_data, default=str)))

    # Include the original Databricks API response as a JSON string attribute
    raw_response = trace_dict.get("raw_response")
    if raw_response:
        agent_attrs.append(_make_attr("databricks.raw_response", json.dumps(raw_response, default=str)))

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


def ai_gateway_usage_to_otlp(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Databricks AI Gateway usage table row to an OTLP JSON payload.

    Creates a 2-span trace:
      - Root span: the AI Gateway endpoint handling the request
      - Child span: the downstream LLM call (model, tokens, provider)

    The row dict keys should match the ``system.ai_gateway.usage`` schema
    columns (snake_case).

    Returns:
        OTLP JSON dict: ``{"resourceSpans": [...]}``
    """
    request_id = str(row.get("request_id", "unknown"))
    trace_id = _deterministic_id(f"gw-trace:{request_id}", 32)
    gateway_span_id = _deterministic_id(f"gw-root:{request_id}", 16)
    llm_span_id = _deterministic_id(f"gw-llm:{request_id}", 16)

    # Timestamps
    event_time = row.get("event_time")
    if isinstance(event_time, datetime):
        start_ns = int(event_time.timestamp() * 1_000_000_000)
    else:
        start_ns = int(time.time() * 1_000_000_000)

    latency_ms = row.get("latency_ms")
    if latency_ms is not None:
        end_ns = start_ns + int(latency_ms) * 1_000_000
    else:
        end_ns = start_ns

    endpoint_name = row.get("endpoint_name") or "ai-gateway"

    # --- Resource attributes ---
    resource_attrs = [
        _make_attr("service.name", f"databricks-ai-gateway:{endpoint_name}"),
        _make_attr("service.namespace", "databricks"),
        _make_attr("ai_gateway.endpoint.name", endpoint_name),
    ]
    endpoint_id = row.get("endpoint_id")
    if endpoint_id:
        resource_attrs.append(_make_attr("ai_gateway.endpoint.id", str(endpoint_id)))
    account_id = row.get("account_id")
    if account_id:
        resource_attrs.append(_make_attr("ai_gateway.account_id", str(account_id)))
    workspace_id = row.get("workspace_id")
    if workspace_id:
        resource_attrs.append(_make_attr("ai_gateway.workspace_id", str(workspace_id)))

    # --- Root gateway span ---
    gw_attrs: List[Dict[str, Any]] = [
        _make_attr("gen_ai.agent.name", endpoint_name),
        _make_attr("ai_gateway.request_id", request_id),
    ]

    # Requester identity
    requester = row.get("requester")
    if requester:
        gw_attrs.append(_make_attr("enduser.id", str(requester)))
        gw_attrs.append(_make_attr("user.id", str(requester)))
        # If it looks like an email, also set user.email
        if "@" in str(requester):
            gw_attrs.append(_make_attr("user.email", str(requester)))
    requester_type = row.get("requester_type")
    if requester_type:
        gw_attrs.append(_make_attr("user.type", str(requester_type)))

    # Network / HTTP context
    ip_address = row.get("ip_address")
    if ip_address:
        gw_attrs.append(_make_attr("net.peer.ip", str(ip_address)))
        gw_attrs.append(_make_attr("user.ip_address", str(ip_address)))
    url = row.get("url")
    if url:
        gw_attrs.append(_make_attr("http.url", str(url)))
    user_agent = row.get("user_agent")
    if user_agent:
        gw_attrs.append(_make_attr("http.user_agent", str(user_agent)))
    status_code = row.get("status_code")
    if status_code is not None:
        gw_attrs.append(_make_attr_int("http.status_code", int(status_code)))

    # API type (chat, completions, embeddings)
    api_type = row.get("api_type")
    if api_type:
        gw_attrs.append(_make_attr("gen_ai.operation.name", str(api_type)))
        gw_attrs.append(_make_attr("ai_gateway.api_type", str(api_type)))

    # Destination info
    destination_type = row.get("destination_type")
    if destination_type:
        gw_attrs.append(_make_attr("ai_gateway.destination_type", str(destination_type)))
    destination_name = row.get("destination_name")
    if destination_name:
        gw_attrs.append(_make_attr("ai_gateway.destination_name", str(destination_name)))

    # Timing
    if latency_ms is not None:
        gw_attrs.append(_make_attr_int("ai_gateway.latency_ms", int(latency_ms)))
    ttfb = row.get("time_to_first_byte_ms")
    if ttfb is not None:
        gw_attrs.append(_make_attr_int("ai_gateway.time_to_first_byte_ms", int(ttfb)))

    # Response content type
    resp_ct = row.get("response_content_type")
    if resp_ct:
        gw_attrs.append(_make_attr("http.response.content_type", str(resp_ct)))

    # Endpoint tags (MAP<STRING, STRING>)
    endpoint_tags = row.get("endpoint_tags")
    if endpoint_tags and isinstance(endpoint_tags, dict):
        for k, v in endpoint_tags.items():
            gw_attrs.append(_make_attr(f"ai_gateway.endpoint_tag.{k}", str(v)))

    # Request tags (MAP<STRING, STRING>)
    request_tags = row.get("request_tags")
    if request_tags and isinstance(request_tags, dict):
        for k, v in request_tags.items():
            gw_attrs.append(_make_attr(f"ai_gateway.request_tag.{k}", str(v)))

    # Routing information (STRUCT)
    routing = row.get("routing_information")
    if routing and isinstance(routing, dict):
        gw_attrs.append(_make_attr("ai_gateway.routing", json.dumps(routing, default=str)))

    # Endpoint metadata (STRUCT)
    endpoint_meta = row.get("endpoint_metadata")
    if endpoint_meta and isinstance(endpoint_meta, dict):
        gw_attrs.append(_make_attr("ai_gateway.endpoint_metadata", json.dumps(endpoint_meta, default=str)))

    root_span = {
        "traceId": trace_id,
        "spanId": gateway_span_id,
        "name": endpoint_name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": gw_attrs,
        "status": {},
    }

    # --- Child LLM span ---
    dest_model = row.get("destination_model") or "unknown"
    provider = row.get("destination_name") or "databricks"

    llm_attrs: List[Dict[str, Any]] = [
        _make_attr("gen_ai.request.model", str(dest_model)),
        _make_attr("gen_ai.system", str(provider)),
    ]

    input_tokens = row.get("input_tokens")
    if input_tokens is not None:
        try:
            llm_attrs.append(_make_attr_int("gen_ai.usage.input_tokens", int(input_tokens)))
        except (ValueError, TypeError):
            pass
    output_tokens = row.get("output_tokens")
    if output_tokens is not None:
        try:
            llm_attrs.append(_make_attr_int("gen_ai.usage.output_tokens", int(output_tokens)))
        except (ValueError, TypeError):
            pass
    total_tokens = row.get("total_tokens")
    if total_tokens is not None:
        try:
            llm_attrs.append(_make_attr_int("gen_ai.usage.total_tokens", int(total_tokens)))
        except (ValueError, TypeError):
            pass

    # Token details (STRUCT with cache_read, cache_creation, reasoning tokens)
    token_details = row.get("token_details")
    if token_details and isinstance(token_details, dict):
        for k, v in token_details.items():
            if v is not None:
                try:
                    llm_attrs.append(_make_attr_int(f"gen_ai.usage.{k}", int(v)))
                except (ValueError, TypeError):
                    llm_attrs.append(_make_attr(f"gen_ai.usage.{k}", str(v)))

    llm_span = {
        "traceId": trace_id,
        "spanId": llm_span_id,
        "parentSpanId": gateway_span_id,
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
