# Sending OpenTelemetry Traces to Open-CITE

Open-CITE accepts traces via the standard OTLP/HTTP protocol on `/v1/traces`. Several AI gateways and routers can export OpenTelemetry traces natively -- no custom instrumentation required.

## Prerequisites

Open-CITE must have an OpenTelemetry plugin instance running. The OTLP receiver listens on port **4318** by default (configurable).

**Accepted format:** OTLP/HTTP with JSON encoding (`application/json`).

Your Open-CITE endpoint URL will be:

| Scenario | Endpoint |
|----------|----------|
| Same machine | `http://localhost:4318/v1/traces` |
| Remote (LAN) | `http://<YOUR_IP>:4318/v1/traces` |
| Public (via ngrok / tunnel) | `https://<your-tunnel>.ngrok.io/v1/traces` |
| Headless API (embedded OTLP) | `http://<HOST>:<API_PORT>/v1/traces` |

For cloud-hosted gateways (Cloudflare, OpenRouter), your Open-CITE instance must be reachable from the internet. Use a tunnel (ngrok, Cloudflare Tunnel) or deploy Open-CITE on a host with a public IP.

---

## Cloudflare AI Gateway

Cloudflare AI Gateway can export OTLP traces for every AI request that passes through your gateway.

### Setup

1. Open the [Cloudflare dashboard](https://dash.cloudflare.com/) and navigate to **AI > AI Gateway > your gateway > Settings**.
2. Under **OpenTelemetry**, click **Add exporter**.
3. Configure the exporter:

   | Field | Value |
   |-------|-------|
   | **URL** | Your Open-CITE OTLP endpoint (e.g. `https://your-host:4318/v1/traces`) |
   | **Authorization** | Leave empty (Open-CITE has no auth by default) |
   | **Headers** | Leave empty, or add custom headers if needed |

4. Save the exporter.

All requests through the gateway will now produce OTLP trace spans sent to Open-CITE.

### Trace context propagation

To link AI Gateway spans to your application's existing traces, include these headers in your requests to the gateway:

| Header | Format | Description |
|--------|--------|-------------|
| `cf-aig-otel-trace-id` | 32-char hex string | Trace ID to use (instead of auto-generated) |
| `cf-aig-otel-parent-span-id` | 16-char hex string | Parent span ID to nest under |

If omitted, AI Gateway generates a new trace ID per request.

### Custom metadata

Add the `cf-aig-metadata` header (JSON object) to your gateway requests to attach custom span attributes. Keys prefixed with `gen_ai.` are reserved and will be ignored.

```bash
curl https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/openai/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H 'cf-aig-metadata: {"team": "platform", "env": "production"}' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Span attributes exported

Cloudflare AI Gateway exports these attributes following GenAI semantic conventions:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.request.model` | Model identifier |
| `gen_ai.model.provider` | Provider (e.g. `openai`) |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.usage.cost` | Estimated request cost |
| `gen_ai.prompt_json` | Encoded prompt/messages |
| `gen_ai.completion_json` | Encoded model response |

Open-CITE's detection logic reads `gen_ai.request.model` and `gen_ai.model.provider` to automatically discover models and tools.

### Limitations

- Only OTLP/JSON is supported (not protobuf).

---

## OpenRouter

OpenRouter's **Broadcast** feature sends OTLP traces for every API call, with zero code changes to your application.

### Setup

1. Go to [OpenRouter Settings > Observability](https://openrouter.ai/settings/observability).
2. Toggle **Enable Broadcast**.
3. Click the edit icon next to **OpenTelemetry Collector**.
4. Configure:

   | Field | Value |
   |-------|-------|
   | **Endpoint** | Your Open-CITE OTLP endpoint (e.g. `https://your-host:4318/v1/traces`) |
   | **Headers** | `{}` (empty JSON object -- Open-CITE has no auth by default) |

5. Click **Test Connection** to verify. The configuration only saves if the test passes.
6. Save.

All subsequent OpenRouter API calls will produce OTLP traces sent to Open-CITE.

### Sending trace metadata

You can enrich traces with custom metadata by including the `trace` field in your OpenRouter API requests:

```bash
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "user": "user_12345",
    "session_id": "session_abc",
    "trace": {
      "trace_id": "my_app_trace_001",
      "trace_name": "Chat Handler",
      "generation_name": "Generate Response",
      "environment": "production"
    }
  }'
```

### Metadata mapping

| Request field | OTLP destination | Description |
|---------------|------------------|-------------|
| `trace.trace_id` | Trace ID | Correlate multiple requests into one trace |
| `trace.trace_name` | Root span name | Name for the root span |
| `trace.span_name` | Span name | Label for intermediate spans |
| `trace.generation_name` | Span name | LLM operation identifier |
| `trace.parent_span_id` | Parent span ID | Connect to an existing trace |
| `user` | `user.id` attribute | User identifier |
| `session_id` | `session.id` attribute | Session identifier |
| Other `trace.*` fields | `trace.metadata.*` attributes | Custom span attributes |

Open-CITE reads `trace.metadata.openrouter.entity_id`, `trace.metadata.openrouter.api_key_name`, and `trace.metadata.openrouter.creator_user_id` for additional discovery context.

### Privacy mode

When Privacy Mode is enabled in OpenRouter settings, prompt and completion text are excluded from traces. Operational metrics (tokens, cost, timing, model info, custom metadata) are still sent.

### Protocol details

OpenRouter sends traces as **OTLP/HTTP with JSON encoding** to the `/v1/traces` path. This matches Open-CITE's accepted format exactly.

---

## Verifying traces are arriving

After configuring either integration, verify Open-CITE is receiving traces:

**GUI:** Open the Open-CITE dashboard. Discovered tools and models should appear within seconds.

**API:**
```bash
# Check plugin status
curl http://localhost:5000/api/instances

# List discovered assets
curl http://localhost:5000/api/assets
```

**Python:**
```python
status = client.verify_otel_connection()
print(f"Traces received: {status['traces_received']}")
print(f"Tools discovered: {status['tools_discovered']}")
```

## Troubleshooting

**No traces arriving:**
- Confirm Open-CITE is reachable from the internet (for cloud gateways). Test with `curl -X POST https://your-host:4318/v1/traces -H "Content-Type: application/json" -d '{"resourceSpans":[]}'` -- you should get `{"status": "success"}`.
- Check firewall rules allow inbound traffic on port 4318.
- For OpenRouter, ensure the **Test Connection** passed when saving.

**Traces arriving but no tools/models discovered:**
- Verify the traces contain `gen_ai.request.model` or similar model attributes. Check raw traces in the GUI or via `client.list_otel_traces()`.

**Tunnel setup (ngrok):**
```bash
ngrok http 4318
# Use the https URL (e.g. https://abc123.ngrok.io/v1/traces) as your endpoint
```
