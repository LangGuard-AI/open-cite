# AWS Plugins for OpenCITE

## Overview

OpenCITE includes two AWS plugins for discovering AI/ML assets:

- **AWS Bedrock** (`aws_bedrock`) — Foundation models, custom models, provisioned throughput, and model invocations
- **AWS SageMaker** (`aws_sagemaker`) — Endpoints, models, model packages, and training jobs

Both plugins share authentication via `AWSClientMixin` and support the same credential methods.

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

## GUI Usage

Both plugins appear in the GUI plugin list automatically. Add an instance, provide credentials (or leave blank to use the default credential chain), and start it. Discovered assets appear in the asset tabs.

## Troubleshooting

**`botocore.exceptions.NoCredentialsError`** — No credentials found. Set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or configure a profile.

**`botocore.exceptions.ClientError: AccessDeniedException`** — The IAM principal lacks required permissions. Check the permissions tables above.

**No invocations found (Bedrock)** — CloudTrail may not have events for the time range, or Bedrock hasn't been used in the target region. Try increasing the `days` parameter.

**Empty endpoints (SageMaker)** — No endpoints exist in the configured region. Try a different region.

## Related Documentation

- [OpenTelemetry Plugin](OPENTELEMETRY_PLUGIN.md)
- [Google Cloud Plugin](GOOGLE_CLOUD_PLUGIN.md)
- [Schema Documentation](../SCHEMA_DOCUMENTATION.md)
