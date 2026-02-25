# AWS Plugins for OpenCITE

## Overview

OpenCITE includes three AWS plugins for discovering AI/ML assets:

- **AWS Bedrock** (`aws_bedrock`) — Foundation models, custom models, provisioned throughput, and model invocations
- **AWS SageMaker** (`aws_sagemaker`) — Endpoints, models, model packages, and training jobs
- **AWS Bedrock AgentCore** (`aws_agentcore`) — Agent runtimes, memory stores, and gateways

All plugins share authentication via `AWSClientMixin` and support the same credential methods.

## Authentication

Credentials are resolved in this order:

1. **Explicit credentials** — `access_key_id` + `secret_access_key` passed in config
2. **AWS profile** — `profile` name from `~/.aws/credentials`
3. **Environment variables** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
4. **Default chain** — IAM role (EC2/ECS/Lambda), AWS SSO, credential process

### Quick Setup

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"

# Option 2: AWS profile
export AWS_PROFILE="my-profile"

# Option 3: SSO login
aws sso login --profile my-sso-profile
```

### Role Assumption

Both plugins support assuming an IAM role via the `role_arn` config field. The plugin calls `sts:AssumeRole` using your base credentials, then uses the temporary credentials for all subsequent API calls.

## AWS Bedrock Plugin

### What It Discovers

| Asset Type | Description | Source |
|------------|-------------|--------|
| `model` | Foundation models (Claude, Llama, Titan, etc.) | Bedrock API |
| `custom_model` | Fine-tuned models | Bedrock API |
| `provisioned_throughput` | Provisioned throughput configs | Bedrock API |
| `invocation` | Model invocations | CloudTrail |

### Required IAM Permissions

```yaml
# Foundation model listing
bedrock:ListFoundationModels
bedrock:ListCustomModels
bedrock:ListProvisionedModelThroughputs

# Usage discovery (optional)
cloudtrail:LookupEvents
logs:FilterLogEvents      # If using CloudWatch invocation logs
```

### Configuration Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `region` | No | `us-east-1` | AWS region |
| `profile` | No | from env | AWS profile name |
| `access_key_id` | No | from env | AWS access key |
| `secret_access_key` | No | from env | AWS secret key |
| `role_arn` | No | — | IAM role to assume |

### Programmatic Usage

```python
from open_cite.plugins.registry import create_plugin_instance

plugin = create_plugin_instance("aws_bedrock", {
    "region": "us-east-1",
    "profile": "my-profile",
})

# Verify connection
status = plugin.verify_connection()
print(f"Models available: {status['foundation_models_available']}")

# List foundation models
models = plugin.list_assets("model")
for m in models:
    print(f"{m['provider']}/{m['name']} — streaming: {m['streaming_supported']}")

# List custom/fine-tuned models
custom = plugin.list_assets("custom_model")

# List provisioned throughput
throughput = plugin.list_assets("provisioned_throughput")

# Discover actual usage from CloudTrail (last 7 days)
invocations = plugin.list_assets("invocation")

# Aggregate usage by model
usage_by_model = plugin.get_usage_by_model(days=7)
for model_id, stats in usage_by_model.items():
    print(f"{model_id}: {stats['invocation_count']} calls, {stats['unique_user_count']} users")

# Aggregate usage by user/principal
usage_by_user = plugin.get_usage_by_user(days=7)
```

### Invocation Discovery

The Bedrock plugin discovers actual model usage through two sources:

**CloudTrail** — Captures `InvokeModel` and `InvokeModelWithResponseStream` events. Shows which models are being called, by whom, and from which IP addresses. Available by default (no extra setup).

**CloudWatch Logs** (optional) — If Bedrock model invocation logging is enabled, provides richer data including input/output token counts. Configure with the `cloudwatch_log_group` field.

## AWS SageMaker Plugin

### What It Discovers

| Asset Type | Description | Source |
|------------|-------------|--------|
| `endpoint` | Model serving endpoints with config details | SageMaker API |
| `model` | Registered model artifacts | SageMaker API |
| `model_package` | Model registry entries (versioned and unversioned) | SageMaker API |
| `training_job` | Recent training jobs (last 30 days) | SageMaker API |

### Required IAM Permissions

```yaml
# Core discovery
sagemaker:ListEndpoints
sagemaker:DescribeEndpoint
sagemaker:DescribeEndpointConfig
sagemaker:ListModels
sagemaker:DescribeModel
sagemaker:ListModelPackageGroups
sagemaker:ListModelPackages
sagemaker:ListTrainingJobs
sagemaker:DescribeTrainingJob

# Usage metrics (optional)
cloudwatch:GetMetricStatistics

# Account ID (optional)
sts:GetCallerIdentity
```

### Configuration Fields

Same as Bedrock — `region`, `profile`, `access_key_id`, `secret_access_key`, `role_arn`.

### Programmatic Usage

```python
from open_cite.plugins.registry import create_plugin_instance

plugin = create_plugin_instance("aws_sagemaker", {
    "region": "us-east-1",
})

# List endpoints
endpoints = plugin.list_assets("endpoint")
for ep in endpoints:
    print(f"{ep['name']}: {ep['status']} ({ep.get('instance_type', 'N/A')})")

# List models
models = plugin.list_assets("model")

# List model packages
packages = plugin.list_assets("model_package")

# List training jobs (last 30 days)
jobs = plugin.list_assets("training_job")

# Get invocation metrics for a specific endpoint
metrics = plugin.get_endpoint_invocation_metrics("my-endpoint", days=7)
print(f"Invocations: {metrics['invocations']}, Latency: {metrics['model_latency_avg_ms']}ms")

# Get metrics for all endpoints
all_metrics = plugin.get_all_endpoint_metrics(days=7)

# Get full usage summary
summary = plugin.get_usage_summary(days=7)
print(f"Active endpoints: {summary['summary']['active_endpoints']}")
print(f"Total invocations: {summary['summary']['total_invocations_last_n_days']}")
```

## AWS Bedrock AgentCore Plugin

### What It Discovers

| Asset Type | Description | Source |
|------------|-------------|--------|
| `agent_runtime` | Deployed agent runtimes | AgentCore API |
| `memory` | Memory stores attached to agents | AgentCore API |
| `gateway` | API gateways / MCP tool configurations | AgentCore API |

### Required IAM Permissions

```yaml
# Core discovery
bedrock-agentcore:ListAgentRuntimes
bedrock-agentcore:GetAgentRuntime
bedrock-agentcore:ListMemories
bedrock-agentcore:ListGateways
bedrock-agentcore:ListGatewayTargets

# Account ID (optional)
sts:GetCallerIdentity
```

### Configuration Fields

Same as Bedrock — `region`, `profile`, `access_key_id`, `secret_access_key`, `role_arn`.

### Programmatic Usage

```python
from open_cite.plugins.registry import create_plugin_instance

plugin = create_plugin_instance("aws_agentcore", {
    "region": "us-east-1",
})

# Verify connection
status = plugin.verify_connection()
print(status)

# List deployed agent runtimes
runtimes = plugin.list_assets("agent_runtime")
for rt in runtimes:
    print(f"{rt['name']}: {rt['status']} (region: {rt['region']})")

# List memory stores
memories = plugin.list_assets("memory")

# List gateways and their targets
gateways = plugin.list_assets("gateway")
for gw in gateways:
    print(f"{gw['name']}: {len(gw['targets'])} targets")
```

### OTel Trace Correlation

When the OpenTelemetry receiver is also running, the AgentCore plugin automatically correlates incoming traces with discovered runtimes. The OTel receiver is **auto-enabled** whenever AgentCore is enabled.

This means each runtime shows not just deployment info, but also **live telemetry**:

| Enriched Field | Description |
|---------------|-------------|
| `otel_correlated` | `true` if matching traces were found |
| `otel_agents` | Agents discovered via OTel matching this runtime |
| `models_used` | Models the agent called (from traces) |
| `tools_used` | Tools the agent invoked (from traces) |
| `token_usage` | Input/output token counts per model |
| `last_trace_seen` | Most recent activity timestamp |

To receive traces from your AgentCore agent, point its OTel exporter at OpenCITE:

```bash
# In your AgentCore agent's environment / runtime config:
OTEL_EXPORTER_OTLP_ENDPOINT=http://<opencite-host>:4318
```

### Testing: Verify OTel Receiver Captures AgentCore Traces

```python
import time
from open_cite.client import OpenCiteClient
from open_cite.plugins.registry import create_plugin_instance

client = OpenCiteClient()

# 1. Start the OTel receiver
otel = create_plugin_instance("opentelemetry", {"host": "0.0.0.0", "port": 4318})
client.register_plugin(otel)
otel.start()
print(f"OTLP receiver listening on http://0.0.0.0:4318/v1/traces")
print(f"Point your AgentCore agent here: OTEL_EXPORTER_OTLP_ENDPOINT=http://<this-host>:4318")

# 2. Start AgentCore discovery (auto-wired to OTel)
ac = create_plugin_instance("aws_agentcore", {"region": "us-east-1"})
client.register_plugin(ac)
ac.start()
print(f"AgentCore linked to OTel: {ac.get_config()['otel_linked']}")

# 3. Check if any traces have arrived
status = otel.verify_connection()
print(f"Traces received so far: {status.get('traces_received', 0)}")
print(f"Tools discovered: {status.get('tools_discovered', 0)}")

# 4. Invoke your AgentCore agent, wait for traces, then check again
print("\n--- Invoke your agent now, then press Enter ---")
input()

status = otel.verify_connection()
print(f"Traces received: {status.get('traces_received', 0)}")
print(f"Tools discovered: {status.get('tools_discovered', 0)}")

# 5. List discovered tools & agents from traces
tools = client.list_tools()
print(f"\nDiscovered tools ({len(tools)}):")
for t in tools:
    print(f"  {t['name']}")

agents = client.list_agents()
print(f"\nDiscovered agents ({len(agents)}):")
for a in agents:
    print(f"  {a['name']}")

# 6. List runtimes — now enriched with OTel data
runtimes = client.list_agentcore_runtimes()
print(f"\nAgent runtimes ({len(runtimes)}):")
for rt in runtimes:
    corr = rt.get('otel_correlated', False)
    print(f"  {rt['name']} — status: {rt['status']}, otel_correlated: {corr}")
    if corr:
        print(f"    models_used: {rt.get('models_used', [])}")
        print(f"    tools_used: {rt.get('tools_used', [])}")
        print(f"    token_usage: {rt.get('token_usage', {})}")
        print(f"    last_trace_seen: {rt.get('last_trace_seen')}")
```

### Environment Variable Quick Start

```bash
# Enable AgentCore discovery + OTel receiver (auto-enabled)
set OPENCITE_ENABLE_AGENTCORE=true
```

## GUI Usage

All three plugins appear in the GUI plugin list automatically. Add an instance, provide credentials (or leave blank to use the default credential chain), and start it. Discovered assets appear in the asset tabs.

## Troubleshooting

**`botocore.exceptions.NoCredentialsError`** — No credentials found. Set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or configure a profile.

**`botocore.exceptions.ClientError: AccessDeniedException`** — The IAM principal lacks required permissions. Check the permissions tables above.

**No invocations found (Bedrock)** — CloudTrail may not have events for the time range, or Bedrock hasn't been used in the target region. Try increasing the `days` parameter.

**Empty endpoints (SageMaker)** — No endpoints exist in the configured region. Try a different region.

**`UnknownServiceError: Unknown service: 'bedrock-agentcore'`** — Your boto3 version doesn't include the AgentCore service definition. Upgrade with `pip install --upgrade boto3 botocore`.

**Empty runtimes (AgentCore)** — No agent runtimes deployed in the configured region. Verify you deployed your agent in the same region you're querying.

## Related Documentation

- [OpenTelemetry Plugin](OPENTELEMETRY_PLUGIN.md)
- [Google Cloud Plugin](GOOGLE_CLOUD_PLUGIN.md)
- [Schema Documentation](../SCHEMA_DOCUMENTATION.md)
