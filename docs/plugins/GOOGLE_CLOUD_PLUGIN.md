# Google Cloud Plugin for OpenCITE

## Overview

The Google Cloud plugin automatically discovers AI tools and models deployed in Google Cloud Platform, with a focus on Vertex AI resources. It uses Google Cloud APIs to enumerate models, endpoints, deployments, and generative AI capabilities.

**Key Principle**: The plugin discovers AI infrastructure through GCP APIs, providing visibility into what models and services are deployed and available in your Google Cloud environment.

## Features

- ✅ **Vertex AI Models**: Discovers custom trained models in Model Registry
- ✅ **Vertex AI Endpoints**: Lists model serving endpoints
- ✅ **Model Deployments**: Tracks which models are deployed to which endpoints
- ✅ **Generative AI Models**: Lists available generative models (Gemini, PaLM, etc.)
- ✅ **MCP Server Discovery**: Discovers MCP servers running on Compute Engine instances via labels
- ✅ **Port Scanning**: Discover MCP servers by scanning open ports on Compute Engine instances
- ✅ **Resource Metadata**: Captures labels, descriptions, machine types, scaling config
- ✅ **JSON Export**: Full schema support for Google Cloud entities
- ✅ **Multi-region Support**: Works across all GCP regions

## Installation

### Prerequisites

```bash
# Install Google Cloud AI Platform SDK
pip install google-cloud-aiplatform

# For MCP server discovery on Compute Engine
pip install google-cloud-compute
```

### Authentication

The plugin uses Google Cloud Application Default Credentials (ADC). Choose the method that works best for your use case:

#### Method 1: User Account via gcloud (Easiest for Local Development)

**Best for:** Local testing, development, quick setup

```bash
# 1. Login (sets up Application Default Credentials)
gcloud auth application-default login

# 2. Set project
export GCP_PROJECT_ID="your-project-id"

# 3. Use the plugin - automatically uses your gcloud session
client = OpenCiteClient(
    enable_google_cloud=True,
    gcp_project_id="your-project-id"
)
```

**That's it!** No service account key needed for local development.

#### Method 2: Service Account Key (Recommended for Production/CI)

**Best for:** Production, CI/CD pipelines, automation, specific permissions

```bash
# 1. Create service account key (one-time setup)
gcloud iam service-accounts keys create key.json \
  --iam-account=your-sa@your-project.iam.gserviceaccount.com

# 2. Set environment variables
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="./key.json"

# 3. Use the plugin
client = OpenCiteClient(enable_google_cloud=True)
```

#### Method 3: Programmatic Credentials

**Best for:** Custom credential management, multi-project scenarios

```python
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/path/to/service-account-key.json'
)

client = OpenCiteClient(
    enable_google_cloud=True,
    gcp_credentials=credentials
)
```

#### How ADC Works

Google Cloud libraries check for credentials in this order:
1. **`GOOGLE_APPLICATION_CREDENTIALS`** environment variable (service account key)
2. **gcloud ADC** (from `gcloud auth application-default login`)
3. **Compute Engine/GKE metadata** (if running on GCP)

#### Quick Comparison

| Method | Setup Time | Best For | Key File Needed? |
|--------|------------|----------|------------------|
| **gcloud ADC** | 1 minute | Local development | ❌ No |
| **Service Account** | 5 minutes | CI/CD, production | ✅ Yes |
| **Programmatic** | 5 minutes | Custom scenarios | ✅ Yes |

### Required IAM Permissions

The service account or user must have these permissions:

```yaml
roles/aiplatform.user           # View Vertex AI resources
roles/aiplatform.viewer         # Read-only access to models/endpoints
```

Or create a custom role with these permissions:

```yaml
aiplatform.models.list
aiplatform.models.get
aiplatform.endpoints.list
aiplatform.endpoints.get
```

Grant permissions to your user account:
```bash
gcloud projects add-iam-policy-binding your-project \
  --member="user:your-email@gmail.com" \
  --role="roles/aiplatform.viewer"
```

## Quick Start

```python
from open_cite import OpenCiteClient

# Initialize with Google Cloud plugin
client = OpenCiteClient(
    enable_google_cloud=True,
    gcp_project_id="my-project-id",    # Optional, uses default if not specified
    gcp_location="us-central1"         # Optional, default: us-central1
)

# Verify connection
status = client.verify_gcp_connection()
print(f"Connected to project: {status['project_id']}")

# List Vertex AI models
models = client.list_gcp_models()
for model in models:
    print(f"Model: {model['name']} ({model['id']})")

# List endpoints
endpoints = client.list_gcp_endpoints()
for endpoint in endpoints:
    print(f"Endpoint: {endpoint['name']}")
    print(f"  Deployed models: {len(endpoint['deployed_models'])}")

# Export to JSON
client.export_to_json(
    include_google_cloud=True,
    filepath="gcp_discoveries.json"
)
```

## API Reference

### Client Initialization

```python
OpenCiteClient(
    enable_google_cloud=True,
    gcp_project_id="my-project-id",      # GCP project ID
    gcp_location="us-central1",          # GCP region
    gcp_credentials=None                 # Optional credentials object
)
```

**Parameters:**
- `enable_google_cloud` (bool): Enable the Google Cloud plugin
- `gcp_project_id` (str, optional): GCP project ID (defaults to ADC project)
- `gcp_location` (str): GCP region/location (default: "us-central1")
- `gcp_credentials` (optional): Google Cloud credentials object

### Client Methods

#### `list_gcp_models()`

List custom trained Vertex AI models.

```python
models = client.list_gcp_models()

# Returns:
# [
#   {
#     "id": "projects/123/locations/us-central1/models/456",
#     "name": "my-custom-model",
#     "resource_name": "projects/123/locations/us-central1/models/456",
#     "discovery_source": "google_cloud_api",
#     "type": "vertex_ai_model",
#     "created_time": "2024-01-15T10:30:00Z",
#     "updated_time": "2024-01-20T14:22:00Z",
#     "project_id": "my-project",
#     "location": "us-central1",
#     "version": "v1",
#     "description": "Custom sentiment analysis model",
#     "metadata": {
#       "labels": {"env": "production", "team": "ml"}
#     }
#   }
# ]
```

#### `list_gcp_endpoints()`

List Vertex AI model serving endpoints.

```python
endpoints = client.list_gcp_endpoints()

# Returns:
# [
#   {
#     "id": "projects/123/locations/us-central1/endpoints/789",
#     "name": "sentiment-endpoint",
#     "resource_name": "projects/123/locations/us-central1/endpoints/789",
#     "discovery_source": "google_cloud_api",
#     "type": "vertex_ai_endpoint",
#     "created_time": "2024-01-15T11:00:00Z",
#     "project_id": "my-project",
#     "location": "us-central1",
#     "deployed_models": [
#       {
#         "deployed_model_id": "123456",
#         "model": "projects/123/locations/us-central1/models/456",
#         "display_name": "sentiment-v1",
#         "machine_type": "n1-standard-4",
#         "min_replicas": 1,
#         "max_replicas": 5
#       }
#     ],
#     "metadata": {
#       "labels": {"env": "production"}
#     }
#   }
# ]
```

#### `list_gcp_deployments()`

List model deployments (models deployed to endpoints).

```python
deployments = client.list_gcp_deployments()

# Returns:
# [
#   {
#     "id": "projects/123/.../endpoints/789/123456",
#     "endpoint_id": "projects/123/locations/us-central1/endpoints/789",
#     "endpoint_name": "sentiment-endpoint",
#     "model_id": "projects/123/locations/us-central1/models/456",
#     "deployed_model_id": "123456",
#     "deployed_model_name": "sentiment-v1",
#     "discovery_source": "google_cloud_api",
#     "type": "vertex_ai_deployment",
#     "project_id": "my-project",
#     "location": "us-central1",
#     "machine_type": "n1-standard-4",
#     "min_replicas": 1,
#     "max_replicas": 5
#   }
# ]
```

#### `list_gcp_generative_models()`

List available generative AI models (Gemini, PaLM, etc.).

```python
generative_models = client.list_gcp_generative_models()

# Returns:
# [
#   {
#     "id": "gemini-pro",
#     "name": "Gemini Pro",
#     "discovery_source": "google_cloud_api",
#     "type": "generative_ai_model",
#     "provider": "google",
#     "model_family": "gemini",
#     "capabilities": ["text_generation", "chat", "code_generation"],
#     "project_id": "my-project",
#     "location": "us-central1"
#   },
#   {
#     "id": "gemini-1.5-pro",
#     "name": "Gemini 1.5 Pro",
#     "discovery_source": "google_cloud_api",
#     "type": "generative_ai_model",
#     "provider": "google",
#     "model_family": "gemini",
#     "capabilities": ["text_generation", "chat", "long_context", "multimodal"],
#     "project_id": "my-project",
#     "location": "us-central1"
#   }
# ]
```

**Available Generative Models:**
- Gemini Pro
- Gemini Pro Vision
- Gemini 1.5 Pro
- Gemini 1.5 Flash
- PaLM 2 Text Bison
- PaLM 2 Chat Bison
- PaLM 2 Code Chat Bison
- Text Embedding Gecko

#### `list_gcp_mcp_servers(**kwargs)`

Discover MCP servers running on Compute Engine instances.

**Requires:** Instances must be labeled with `mcp-server` label. See [GCP MCP Labeling Guide](GCP_MCP_LABELING_GUIDE.md) for details.

```python
# Discover all MCP servers
mcp_servers = client.list_gcp_mcp_servers()

# Discover only in specific zones
mcp_servers = client.list_gcp_mcp_servers(zones=["us-central1-a", "us-east1-b"])

# Returns:
# [
#   {
#     "id": "mcp-filesystem-filesystem",
#     "name": "filesystem",
#     "discovery_source": "gcp_compute_labels",
#     "instance_name": "mcp-filesystem",
#     "instance_id": "1234567890",
#     "zone": "us-central1-a",
#     "project_id": "my-project",
#     "status": "RUNNING",
#     "transport": "stdio",
#     "command": "npx",
#     "labels": {
#       "mcp-server": "filesystem",
#       "mcp-transport": "stdio",
#       "mcp-command": "npx"
#     },
#     "metadata": {
#       "machine_type": "e2-medium",
#       "created_time": "2024-01-15T10:00:00Z"
#     }
#   },
#   {
#     "id": "mcp-postgres-postgres",
#     "name": "postgres",
#     "discovery_source": "gcp_compute_labels",
#     "instance_name": "mcp-postgres",
#     "zone": "us-central1-a",
#     "status": "RUNNING",
#     "transport": "http",
#     "port": "3000",
#     "endpoint": "http://34.123.45.67:3000",
#     "external_ip": "34.123.45.67",
#     "internal_ip": "10.128.0.2"
#   }
# ]
```

**Required Instance Labels:**
- `mcp-server`: Server name (required)
- `mcp-transport`: Transport type (optional, default: stdio)
- `mcp-port`: Port for HTTP/SSE servers (optional, default: 3000)
- `mcp-command`: Start command (optional)

**Label Your Instances:**
```bash
# For stdio MCP server
gcloud compute instances add-labels my-instance \
    --labels=mcp-server=filesystem,mcp-transport=stdio \
    --zone=us-central1-a

# For HTTP MCP server
gcloud compute instances add-labels my-instance \
    --labels=mcp-server=postgres,mcp-transport=http,mcp-port=3000 \
    --zone=us-central1-a
```

See the complete [GCP MCP Labeling Guide](GCP_MCP_LABELING_GUIDE.md) for detailed instructions.

#### `scan_gcp_mcp_servers(zones=None, ports=None, min_ports_open=1)`

Discover MCP servers by scanning open ports on Compute Engine instances.

**Use Case:** Complement label-based discovery by finding servers that may not be properly labeled, or audit which instances have MCP-related ports open.

**Prerequisites:** Requires `gcloud` CLI to be installed and authenticated.

```python
# Scan all instances in all zones
servers = client.scan_gcp_mcp_servers()

# Scan specific zones only
servers = client.scan_gcp_mcp_servers(zones=["us-central1-a"])

# Scan specific ports
servers = client.scan_gcp_mcp_servers(ports=[3000, 8080, 9000])

# Require at least 2 open ports (more selective)
servers = client.scan_gcp_mcp_servers(min_ports_open=2)

# Returns:
# [
#   {
#     "id": "my-instance-us-central1-a",
#     "name": "filesystem",  # From label if available, or "unknown"
#     "discovery_source": "gcp_port_scan",
#     "instance_name": "my-instance",
#     "instance_id": "1234567890",
#     "zone": "us-central1-a",
#     "project_id": "my-project",
#     "status": "RUNNING",
#     "external_ip": "34.123.45.67",
#     "internal_ip": "10.128.0.2",
#     "open_ports": [
#       {"port": 3000, "state": "open", "service": "http-alt"},
#       {"port": 8080, "state": "open", "service": "http-proxy"}
#     ],
#     "scanned_ports": [3000, 3001, 3002, 8000, 8080, 8888, 5000, 9000],
#     "transport": "http",  # Inferred from open ports
#     "endpoint": "http://34.123.45.67:3000",
#     "inferred": true,  # True if no mcp-server label found
#     "labels": {}  # MCP labels if available
#   }
# ]
```

**Parameters:**
- `zones` (list, optional): List of zones to scan (e.g., `["us-central1-a"]`). If None, scans all zones.
- `ports` (list, optional): List of ports to scan (e.g., `[3000, 8080]`). If None, uses default MCP ports: `[3000, 3001, 3002, 8000, 8080, 8888, 5000, 9000]`
- `min_ports_open` (int): Minimum number of open ports to consider an instance as a potential MCP server (default: 1)

**Performance:**
- Scanning is network-bound and takes ~0.5 seconds per port per instance
- Ports are scanned in parallel for performance (max 10 concurrent)
- Limit zones or ports to reduce scan time
- Example: 10 instances × 8 ports = ~4 seconds total (with parallelization)

**Security:**
- Respects GCP firewall rules (closed ports appear as closed)
- Only TCP connection testing (does not send data to services)
- Requires network access to instance external IPs
- Only scans instances you have permission to list

**Combining with Label Discovery:**
```python
# Get comprehensive view
labeled = client.list_gcp_mcp_servers()
scanned = client.scan_gcp_mcp_servers()

# Merge results
all_servers = {}
for s in labeled:
    all_servers[s['instance_name']] = s
for s in scanned:
    if s['instance_name'] in all_servers:
        all_servers[s['instance_name']]['open_ports'] = s['open_ports']
    else:
        all_servers[s['instance_name']] = s

print(f"Total unique instances: {len(all_servers)}")
```

#### `verify_gcp_connection()`

Verify connection to Google Cloud.

```python
status = client.verify_gcp_connection()

# Returns:
# {
#     "success": True,
#     "project_id": "my-project",
#     "location": "us-central1",
#     "message": "Successfully connected to Google Cloud Vertex AI"
# }
```

#### `refresh_gcp_discovery()`

Refresh all cached Google Cloud discovery data.

```python
client.refresh_gcp_discovery()
```

### Plugin Direct Access

For advanced use cases, access the plugin directly:

```python
gcp_plugin = client.get_plugin("google_cloud")

# List models with kwargs
models = gcp_plugin.list_assets("model")

# List endpoints
endpoints = gcp_plugin.list_assets("endpoint")

# Refresh cache
gcp_plugin.refresh_discovery()
```

## Supported Asset Types

| Asset Type | Description | Client Method |
|------------|-------------|---------------|
| `model` | Custom trained Vertex AI models | `list_gcp_models()` |
| `endpoint` | Model serving endpoints | `list_gcp_endpoints()` |
| `deployment` | Model deployments to endpoints | `list_gcp_deployments()` |
| `generative_model` | Available generative AI models | `list_gcp_generative_models()` |
| `mcp_server` | MCP servers on Compute Engine | `list_gcp_mcp_servers()` |

## JSON Export Format

```json
{
  "version": "1.0.0",
  "metadata": {
    "generated_at": "2024-01-20T10:00:00Z",
    "plugins": [
      {"name": "google_cloud", "version": "1.0.0"}
    ]
  },
  "gcp_models": [
    {
      "model_id": "projects/123/locations/us-central1/models/456",
      "name": "my-custom-model",
      "resource_name": "projects/123/locations/us-central1/models/456",
      "discovery_source": "google_cloud_api",
      "type": "vertex_ai_model",
      "provider": "google",
      "project_id": "my-project",
      "location": "us-central1",
      "version": "v1",
      "created_time": "2024-01-15T10:30:00Z",
      "metadata": {
        "labels": {"env": "production"}
      }
    }
  ],
  "gcp_endpoints": [
    {
      "endpoint_id": "projects/123/locations/us-central1/endpoints/789",
      "name": "sentiment-endpoint",
      "discovery_source": "google_cloud_api",
      "type": "vertex_ai_endpoint",
      "deployed_models": [...]
    }
  ],
  "gcp_deployments": [...],
  "gcp_generative_models": [...]
}
```

## Use Cases

### 1. Model Inventory

Track all custom models deployed in your GCP environment:

```python
models = client.list_gcp_models()

print(f"Total models: {len(models)}")

# Group by environment
by_env = {}
for model in models:
    env = model.get("metadata", {}).get("labels", {}).get("env", "unknown")
    by_env.setdefault(env, []).append(model["name"])

for env, names in by_env.items():
    print(f"{env}: {', '.join(names)}")
```

### 2. Endpoint Monitoring

Monitor which models are deployed to endpoints:

```python
endpoints = client.list_gcp_endpoints()

for endpoint in endpoints:
    print(f"\n{endpoint['name']}:")
    for deployment in endpoint["deployed_models"]:
        print(f"  • {deployment['display_name']}")
        print(f"    Machine: {deployment.get('machine_type', 'N/A')}")
        print(f"    Replicas: {deployment.get('min_replicas')}-{deployment.get('max_replicas')}")
```

### 3. Cost Optimization

Identify resource-intensive deployments:

```python
deployments = client.list_gcp_deployments()

# Find deployments with large machine types
expensive = [
    d for d in deployments
    if "n1-standard-8" in d.get("machine_type", "") or
       "n1-standard-16" in d.get("machine_type", "")
]

print(f"High-cost deployments: {len(expensive)}")
for d in expensive:
    print(f"  {d['endpoint_name']}: {d['machine_type']}")
```

### 4. Generative AI Usage Tracking

Track which generative AI models are available:

```python
gen_models = client.list_gcp_generative_models()

# Group by family
by_family = {}
for model in gen_models:
    family = model["model_family"]
    by_family.setdefault(family, []).append(model["name"])

print("Available Generative AI Models:")
for family, models in by_family.items():
    print(f"  {family}: {', '.join(models)}")
```

### 5. MCP Server Discovery and Monitoring

Discover and monitor MCP servers running on Compute Engine:

```python
# Discover MCP servers
mcp_servers = client.list_gcp_mcp_servers()

print(f"Total MCP servers: {len(mcp_servers)}")

# Group by transport type
by_transport = {}
for server in mcp_servers:
    transport = server['transport']
    by_transport.setdefault(transport, []).append(server['name'])

print("\nBy Transport:")
for transport, names in by_transport.items():
    print(f"  {transport}: {', '.join(names)}")

# Check health
for server in mcp_servers:
    if server['status'] != 'RUNNING':
        print(f"WARNING: {server['name']} is {server['status']}")

# Combined view: Deployed vs. Active
from open_cite import OpenCiteClient

# Get deployed servers (from GCP labels)
client_gcp = OpenCiteClient(enable_google_cloud=True)
deployed_servers = client_gcp.list_gcp_mcp_servers()
deployed_names = {s['name'] for s in deployed_servers}

# Get active servers (from traces via OTel plugin)
client_traces = OpenCiteClient(enable_otel=True)
active_servers = client_traces.list_mcp_servers()
active_names = {s['name'] for s in active_servers}

# Find deployed but unused
unused = deployed_names - active_names
print(f"\nDeployed but unused: {unused}")
```

### 6. Port Scanning for MCP Server Discovery

Discover MCP servers by scanning for open ports on Compute Engine instances. This complements label-based discovery by finding servers that may not be properly labeled:

```python
# Scan all instances for open MCP-related ports
servers = client.scan_gcp_mcp_servers()

print(f"Found {len(servers)} instances with open ports")

for server in servers:
    print(f"\n{server['instance_name']}:")
    print(f"  IP: {server.get('external_ip', 'N/A')}")
    print(f"  Open ports: {server['open_ports']}")

    if server.get('endpoint'):
        print(f"  Endpoint: {server['endpoint']}")

    if server.get('inferred'):
        print(f"  Note: Server type inferred (no label found)")

# Scan specific zones only
central_servers = client.scan_gcp_mcp_servers(zones=["us-central1-a"])

# Scan specific ports
custom_ports = client.scan_gcp_mcp_servers(ports=[3000, 8080, 9000])

# Require multiple open ports (more selective)
multi_port = client.scan_gcp_mcp_servers(min_ports_open=2)

# Combine label-based and port-based discovery
labeled_servers = client.list_gcp_mcp_servers()
scanned_servers = client.scan_gcp_mcp_servers()

# Create comprehensive view
all_instances = {}
for server in labeled_servers:
    instance_id = f"{server['instance_name']}-{server['zone']}"
    all_instances[instance_id] = server
    all_instances[instance_id]['discovery_methods'] = ['labels']

for server in scanned_servers:
    instance_id = f"{server['instance_name']}-{server['zone']}"
    if instance_id in all_instances:
        all_instances[instance_id]['discovery_methods'].append('port_scan')
        all_instances[instance_id]['open_ports'] = server['open_ports']
    else:
        server['discovery_methods'] = ['port_scan']
        all_instances[instance_id] = server

print(f"\nTotal unique instances: {len(all_instances)}")
print(f"Discovered by labels only: {sum(1 for s in all_instances.values() if s['discovery_methods'] == ['labels'])}")
print(f"Discovered by port scan only: {sum(1 for s in all_instances.values() if s['discovery_methods'] == ['port_scan'])}")
print(f"Discovered by both methods: {sum(1 for s in all_instances.values() if len(s['discovery_methods']) > 1)}")
```

**Port Scanning Configuration:**

- **Default ports**: 3000, 3001, 3002, 8000, 8080, 8888, 5000, 9000
- **Timeout**: 0.5 seconds per port
- **Parallelization**: Scans multiple ports concurrently for performance
- **Prerequisites**: Requires `gcloud` CLI to be installed and authenticated

**Use Cases:**
- Find MCP servers that aren't properly labeled
- Audit which instances have MCP-related ports open
- Verify that labeled servers actually have ports open
- Discover legacy or undocumented MCP deployments

**Performance Considerations:**
- Scanning is network-bound and can take several seconds per instance
- Limit zones to reduce scan time: `zones=["us-central1-a"]`
- Reduce port list for faster scans: `ports=[3000, 8080]`
- Use `min_ports_open=2` to filter out false positives

**Security Considerations:**
- Port scanning requires network access to instance external IPs
- Respects GCP firewall rules (closed ports will appear as closed)
- Only scans instances you have permission to list
- Does not attempt to connect to services or send data

### 7. Multi-Region Discovery

Discover resources across multiple regions:

```python
regions = ["us-central1", "us-east1", "europe-west1", "asia-southeast1"]

all_models = []
for region in regions:
    client = OpenCiteClient(
        enable_google_cloud=True,
        gcp_project_id="my-project",
        gcp_location=region
    )
    models = client.list_gcp_models()
    all_models.extend(models)
    print(f"{region}: {len(models)} models")

print(f"\nTotal across all regions: {len(all_models)}")
```

## Multi-Project Discovery

Discover resources across multiple GCP projects:

```python
projects = ["project-dev", "project-staging", "project-prod"]

for project_id in projects:
    print(f"\n=== {project_id} ===")

    client = OpenCiteClient(
        enable_google_cloud=True,
        gcp_project_id=project_id
    )

    models = client.list_gcp_models()
    endpoints = client.list_gcp_endpoints()

    print(f"Models: {len(models)}")
    print(f"Endpoints: {len(endpoints)}")
```

## Troubleshooting

### Authentication Errors

**Error**: `google.auth.exceptions.DefaultCredentialsError`

**Solution**:
```bash
# Set up authentication
gcloud auth application-default login

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

### Permission Denied

**Error**: `403 Forbidden` or `Permission denied`

**Solution**: Ensure your service account has the required roles:
```bash
gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:my-sa@my-project.iam.gserviceaccount.com" \
    --role="roles/aiplatform.viewer"
```

### No Models Found

**Cause**: No custom models in the specified project/location.

**Solution**:
- Verify you're checking the correct project: `client.verify_gcp_connection()`
- Check different regions: Try `gcp_location="us-east1"` etc.
- Generative models are always available: Use `list_gcp_generative_models()`

### Plugin Not Registered

**Error**: `ValueError: Plugin 'google_cloud' not found`

**Cause**: Plugin not enabled during client initialization.

**Solution**:
```python
client = OpenCiteClient(enable_google_cloud=True)  # Enable the plugin
```

## Best Practices

1. **Use Service Accounts in Production**: Don't use user credentials in production
2. **Principle of Least Privilege**: Grant only necessary IAM permissions
3. **Cache Discovery Data**: Use `refresh_gcp_discovery()` to update periodically
4. **Handle Errors Gracefully**: Wrap calls in try/except for production code
5. **Multi-Region Awareness**: Discover across all regions where you deploy models
6. **Label Your Resources**: Use labels for better organization and filtering

## Limitations

- **API-based discovery only**: Discovers resources via GCP APIs (no trace analysis)
- **No usage metrics**: Does not capture prediction counts or latency (use Cloud Monitoring for that)
- **No cost data**: Does not calculate costs (use Cloud Billing for cost analysis)
- **Read-only**: Cannot create, modify, or delete resources
- **Custom models only**: Lists custom trained models, not all Model Garden models
- **No AutoML details**: Limited metadata for AutoML models

## Performance

- **Model listing**: ~1-2 seconds for 100 models
- **Endpoint listing**: ~1-2 seconds for 50 endpoints
- **Caching**: Discovery data is cached in memory
- **API quotas**: Subject to Vertex AI API quotas (typically generous)

## Integration with Other Plugins

### With OpenTelemetry Plugin

Combine GCP discovery with trace-based usage analysis:

```python
client = OpenCiteClient(
    enable_otel=True,              # Trace-based discovery
    enable_google_cloud=True       # GCP API discovery
)

# GCP shows what's deployed
gcp_models = client.list_gcp_models()
print(f"Deployed models: {len(gcp_models)}")

# OTEL shows what's being used
otel_tools = client.list_otel_tools()
print(f"Active tools: {len(otel_tools)}")
```

### With MCP Discovery

MCP discovery is built into the OpenTelemetry plugin. Combine GCP and OTel to discover both deployed infrastructure and active MCP usage:

```python
client = OpenCiteClient(
    enable_otel=True,
    enable_google_cloud=True
)

# GCP discovers MCP servers from Compute Engine labels
gcp_mcp = client.list_gcp_mcp_servers()

# OTel discovers MCP servers actually being used in traces
otel_mcp = client.list_mcp_servers()
```

## Related Documentation

- [OpenTelemetry Plugin](OPENTELEMETRY_PLUGIN.md)
- [MCP Discovery](MCP_PLUGIN.md)
- [Schema Documentation](../SCHEMA_DOCUMENTATION.md)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Google Cloud Python SDK](https://github.com/googleapis/python-aiplatform)

## Examples

See the [examples](examples/) directory:
- `examples/gcp_discovery_example.py` - Complete Google Cloud discovery example

## Support

For Google Cloud plugin questions:
- Verify GCP credentials are configured: `gcloud auth application-default login`
- Check IAM permissions for your service account
- Ensure the Vertex AI API is enabled in your project
- Try a different region if no models are found
