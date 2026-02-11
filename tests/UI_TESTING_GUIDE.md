# OpenCITE UI Testing Guide

This guide will walk you through testing the OpenCITE web interface and seeing models, tools, and assets appear in real-time.

## Step 1: Access the Web Interface

1. **Make sure your Flask app is running** (you should see it in your terminal)
2. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```
   or
   ```
   http://localhost:5000
   ```

## Step 2: Configure Plugins

The UI has a sidebar with plugin configuration. Here's how to set up each plugin:

### OpenTelemetry Plugin (Recommended for Testing)

1. **Check the "OpenTelemetry" checkbox** in the sidebar
2. **No configuration needed** - it uses defaults (port 4318)
3. **Click "Start Discovery"** button
4. You should see:
   - ✅ Status indicator turn green
   - Progress messages showing the OTLP receiver started
   - Endpoint URLs displayed (localhost and network IP)

### Other Plugins (Optional)

- **MCP Plugin**: Check the box (no config needed) - works with OpenTelemetry
- **Databricks Plugin**: Requires `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars
- **Google Cloud Plugin**: Requires GCP project ID and credentials

## Step 3: Send Test Traces

Once OpenTelemetry is configured, you can send test traces to see models and tools appear.

### Option A: Use the Test Script (Easiest)

```bash
# Send a single test trace
python tests/fixtures/send_test_trace.py

# Send multiple traces with different models
python tests/fixtures/send_test_trace.py --tool "my-ai-app" --model "anthropic/claude-3-opus" --count 3

# Send traces with different tool names
python tests/fixtures/send_test_trace.py --tool "customer-service-bot" --model "openai/gpt-4" --count 5
python tests/fixtures/send_test_trace.py --tool "code-assistant" --model "google/gemini-pro" --count 2
```

### Option B: Use curl (Manual)

```bash
curl -X POST http://localhost:4318/v1/traces \
  -H "Content-Type: application/json" \
  -d '{
    "resourceSpans": [{
      "resource": {
        "attributes": [
          {"key": "service.name", "value": {"stringValue": "test-tool"}}
        ]
      },
      "scopeSpans": [{
        "scope": {"name": "openai"},
        "spans": [{
          "traceId": "abc123def456",
          "spanId": "123456",
          "name": "chat.completions",
          "attributes": [
            {"key": "gen_ai.request.model", "value": {"stringValue": "openai/gpt-4"}},
            {"key": "gen_ai.system", "value": {"stringValue": "openrouter"}}
          ]
        }]
      }]
    }]
  }'
```

### Option C: Use OpenRouter Broadcast (Real Production Data)

If you're using OpenRouter, you can enable automatic trace broadcasting:

1. Go to https://openrouter.ai/settings/broadcast
2. Add "OTel Collector" as a destination
3. Set the endpoint to: `http://localhost:4318/v1/traces` (or your network IP)
4. All your OpenRouter API calls will automatically appear in OpenCITE!

## Step 4: View Discovered Assets

After sending traces, the UI will automatically refresh every 3 seconds. You should see:

### In the Main Panel:

1. **Tools Tab**: Lists all discovered tools (services making AI API calls)
   - Tool name
   - Models used by that tool
   - Number of traces/calls

2. **Models Tab**: Lists all discovered AI models
   - Model name (e.g., "openai/gpt-4")
   - Which tools use this model
   - Usage statistics

3. **MCP Tab**: MCP servers, tools, and resources (if MCP plugin enabled)

4. **Data Assets Tab**: Databricks tables and MLflow experiments (if Databricks plugin enabled)

5. **GCP Tab**: Vertex AI models and endpoints (if Google Cloud plugin enabled)

### Status Indicators:

- **Green dot**: Discovery is running
- **Red dot**: Discovery stopped
- **Yellow dot**: Discovery in progress

## Step 5: Export Data

1. **Select plugins** to include in export (checkboxes)
2. **Click "Export to JSON"** button
3. **Download** the JSON file with all discovered assets

The export follows the OpenCITE schema and includes:
- All tools and models
- Usage statistics
- Timestamps
- Plugin metadata

## Step 6: Test Different Scenarios

### Test Multiple Tools

```bash
# Send traces from different tools
python tests/fixtures/send_test_trace.py --tool "customer-support" --model "openai/gpt-4" --count 10
python tests/fixtures/send_test_trace.py --tool "code-reviewer" --model "anthropic/claude-3-opus" --count 5
python tests/fixtures/send_test_trace.py --tool "document-analyzer" --model "google/gemini-pro" --count 3
```

Then check the UI - you should see:
- 3 different tools in the Tools tab
- 3 different models in the Models tab
- Usage counts for each

### Test Real-Time Updates

1. Configure OpenTelemetry plugin
2. Keep the UI open
3. Run `python tests/fixtures/send_test_trace.py` multiple times
4. Watch the UI auto-refresh and show new assets appearing

### Test Stop/Start

1. Click "Stop Discovery" button
2. Status should turn red
3. Click "Start Discovery" again
4. Status should turn green

## Troubleshooting

### No Assets Appearing?

1. **Check plugin status**: Make sure OpenTelemetry plugin shows "✓ Trace receiver ready"
2. **Check endpoint**: Verify the OTLP endpoint URL is correct
3. **Check Flask logs**: Look at terminal output for errors
4. **Wait a few seconds**: UI refreshes every 3 seconds

### Connection Errors?

- Make sure Flask app is running
- Check that port 4318 is not blocked by firewall
- Verify the endpoint URL matches what's shown in the UI status

### Want to Clear Data?

- Click "Stop Discovery" to reset
- Refresh the page to start fresh

## Quick Test Checklist

- [ ] Flask app is running
- [ ] UI accessible at http://localhost:5000
- [ ] OpenTelemetry plugin configured
- [ ] Status shows green (running)
- [ ] Sent test trace using `python send_test_trace.py`
- [ ] Tools appear in Tools tab
- [ ] Models appear in Models tab
- [ ] Export functionality works

## Next Steps

Once you've verified the UI works:

1. **Integrate with real applications**: Configure your AI tools to send OTLP traces
2. **Set up OpenRouter Broadcast**: For automatic trace collection
3. **Configure other plugins**: Add Databricks, Google Cloud, etc.
4. **Export and analyze**: Download JSON exports for further analysis

Happy testing! 🚀

