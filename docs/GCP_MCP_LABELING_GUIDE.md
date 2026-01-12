# GCP MCP Server Labeling Guide

## Overview

This guide explains how to label Google Cloud Compute Engine instances so that OpenCITE can discover MCP (Model Context Protocol) servers running on them.

## Quick Start

To make an MCP server discoverable, add labels to your Compute Engine instance:

```bash
# For a stdio MCP server
gcloud compute instances add-labels my-instance \
    --labels=mcp-server=filesystem,mcp-transport=stdio \
    --zone=us-central1-a

# For an HTTP MCP server
gcloud compute instances add-labels my-instance \
    --labels=mcp-server=postgres,mcp-transport=http,mcp-port=3000 \
    --zone=us-central1-a
```

## Required Labels

### `mcp-server` (Required)

The name or type of the MCP server running on the instance.

**Format:** Lowercase alphanumeric with hyphens
**Examples:**
- `filesystem`
- `postgres`
- `brave-search`
- `custom-tool`

```bash
--labels=mcp-server=filesystem
```

## Optional Labels

### `mcp-transport`

The transport protocol used by the MCP server.

**Values:**
- `stdio` (default) - Standard input/output
- `http` - HTTP protocol
- `sse` - Server-Sent Events

**Default:** `stdio` if not specified

```bash
--labels=mcp-transport=http
```

### `mcp-port`

The port number where the MCP server listens (for HTTP/SSE servers only).

**Format:** Port number (1-65535)
**Default:** `3000` if not specified

```bash
--labels=mcp-port=8080
```

### `mcp-command`

The command used to start the MCP server (for stdio servers).

**Format:** Command string (URL-encoded if necessary)
**Example:** `npx` or `python`

```bash
--labels=mcp-command=npx
```

### Custom Labels

Any label starting with `mcp-` will be captured as MCP server metadata. You can add custom labels for your own tracking:

```bash
--labels=mcp-version=1.0.0,mcp-owner=ml-team,mcp-environment=production
```

## Complete Examples

### Example 1: Filesystem MCP Server (stdio)

```bash
# Create instance with MCP labels
gcloud compute instances create mcp-filesystem \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --labels=mcp-server=filesystem,mcp-transport=stdio,mcp-command=npx

# Or add labels to existing instance
gcloud compute instances add-labels mcp-filesystem \
    --labels=mcp-server=filesystem,mcp-transport=stdio,mcp-command=npx \
    --zone=us-central1-a
```

**Discovery Result:**
```json
{
  "id": "mcp-filesystem-filesystem",
  "name": "filesystem",
  "discovery_source": "gcp_compute_labels",
  "instance_name": "mcp-filesystem",
  "transport": "stdio",
  "command": "npx",
  "zone": "us-central1-a"
}
```

### Example 2: HTTP MCP Server

```bash
gcloud compute instances create mcp-postgres \
    --zone=us-east1-b \
    --machine-type=n1-standard-2 \
    --labels=mcp-server=postgres,mcp-transport=http,mcp-port=3000
```

**Discovery Result:**
```json
{
  "id": "mcp-postgres-postgres",
  "name": "postgres",
  "discovery_source": "gcp_compute_labels",
  "instance_name": "mcp-postgres",
  "transport": "http",
  "port": "3000",
  "endpoint": "http://34.123.45.67:3000",
  "external_ip": "34.123.45.67",
  "internal_ip": "10.128.0.2",
  "zone": "us-east1-b"
}
```

### Example 3: Multiple MCP Servers with Custom Labels

```bash
# Production filesystem server
gcloud compute instances create mcp-fs-prod \
    --zone=us-central1-a \
    --labels=mcp-server=filesystem,\
mcp-transport=stdio,\
mcp-environment=production,\
mcp-owner=platform-team,\
mcp-version=2.1.0,\
env=production,\
team=platform

# Development HTTP server
gcloud compute instances create mcp-api-dev \
    --zone=us-central1-a \
    --labels=mcp-server=custom-api,\
mcp-transport=http,\
mcp-port=8080,\
mcp-environment=development,\
env=dev
```

## Discovery with OpenCITE

### Basic Discovery

```python
from open_cite import OpenCiteClient

# Initialize client
client = OpenCiteClient(
    enable_google_cloud=True,
    gcp_project_id="my-project-id"
)

# Discover MCP servers
mcp_servers = client.list_gcp_mcp_servers()

for server in mcp_servers:
    print(f"Found: {server['name']} on {server['instance_name']}")
    print(f"  Transport: {server['transport']}")
    if server.get('endpoint'):
        print(f"  Endpoint: {server['endpoint']}")
```

### Zone-Specific Discovery

```python
# Only scan specific zones
mcp_servers = client.list_gcp_mcp_servers(zones=["us-central1-a", "us-east1-b"])
```

### Filter by Environment

```python
# Get all MCP servers
all_servers = client.list_gcp_mcp_servers()

# Filter by custom label
prod_servers = [
    s for s in all_servers
    if s.get('labels', {}).get('mcp-environment') == 'production'
]

print(f"Production servers: {len(prod_servers)}")
```

## Terraform Example

### Single Instance

```hcl
resource "google_compute_instance" "mcp_filesystem" {
  name         = "mcp-filesystem"
  machine_type = "e2-medium"
  zone         = "us-central1-a"

  labels = {
    mcp-server    = "filesystem"
    mcp-transport = "stdio"
    mcp-command   = "npx"
    mcp-version   = "1-0-0"  # Terraform requires alphanumeric + hyphens
    env           = "production"
    team          = "ml"
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    # Install and start MCP server
    npm install -g @modelcontextprotocol/server-filesystem
    # Start server (example)
  EOF
}
```

### Instance Template for Multiple Servers

```hcl
resource "google_compute_instance_template" "mcp_server" {
  name_prefix  = "mcp-server-"
  machine_type = "e2-medium"

  labels = {
    mcp-server    = var.mcp_server_name
    mcp-transport = var.mcp_transport
    mcp-port      = var.mcp_port
    env           = var.environment
  }

  disk {
    source_image = "debian-cloud/debian-11"
  }

  network_interface {
    network = "default"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Create instances from template
resource "google_compute_instance_from_template" "mcp_filesystem" {
  name = "mcp-filesystem-prod"
  zone = "us-central1-a"

  source_instance_template = google_compute_instance_template.mcp_server.id

  labels = {
    mcp-server    = "filesystem"
    mcp-transport = "stdio"
    env           = "production"
  }
}
```

## Best Practices

### 1. Consistent Naming

Use consistent server names across your infrastructure:

```bash
# Good: Consistent naming
mcp-server=filesystem
mcp-server=postgres
mcp-server=brave-search

# Avoid: Inconsistent naming
mcp-server=FileSystem
mcp-server=POSTGRES
mcp-server=brave_search
```

### 2. Environment Tags

Always tag with environment for better filtering:

```bash
--labels=mcp-server=postgres,mcp-environment=production,env=prod
```

### 3. Version Tracking

Include version information for change tracking:

```bash
--labels=mcp-server=filesystem,mcp-version=2-1-0
```

Note: GCP labels don't allow dots, so use hyphens instead of `2.1.0`

### 4. Ownership

Tag servers with team/owner information:

```bash
--labels=mcp-server=custom-tool,mcp-owner=ml-team,team=ml
```

### 5. Documentation

Add description labels for complex setups:

```bash
--labels=mcp-server=custom-api,mcp-description=customer-facing-api
```

## IAM Permissions Required

The service account used by OpenCITE needs these permissions:

```yaml
# Minimum permissions
compute.instances.list
compute.instances.get
compute.zones.list

# Or use this role
roles/compute.viewer
```

### Grant Permissions

```bash
# Grant Compute Viewer role
gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:opencite@my-project.iam.gserviceaccount.com" \
    --role="roles/compute.viewer"
```

## Troubleshooting

### No MCP Servers Discovered

**Check 1: Verify labels are set**
```bash
gcloud compute instances describe my-instance --zone=us-central1-a --format="value(labels)"
```

**Check 2: Verify `mcp-server` label exists**
```bash
gcloud compute instances describe my-instance --zone=us-central1-a \
    --format="value(labels.mcp-server)"
```

**Check 3: Test discovery**
```python
servers = client.list_gcp_mcp_servers()
print(f"Found {len(servers)} servers")
```

### Servers Discovered but Missing Information

**Problem:** Endpoint is missing for HTTP server

**Solution:** Ensure the instance has an external IP or check internal IP:
```bash
gcloud compute instances describe my-instance --zone=us-central1-a \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

**Problem:** Transport type is wrong

**Solution:** Explicitly set `mcp-transport` label:
```bash
gcloud compute instances add-labels my-instance \
    --labels=mcp-transport=http \
    --zone=us-central1-a
```

### Permission Errors

**Error:** `Permission denied` when listing instances

**Solution:** Grant compute.viewer role:
```bash
gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:your-sa@project.iam.gserviceaccount.com" \
    --role="roles/compute.viewer"
```

## Label Constraints

GCP label constraints:
- Keys and values must be lowercase
- Keys must start with a letter
- Can contain letters, numbers, hyphens, and underscores
- Maximum 64 characters
- Maximum 64 labels per resource

**Valid:**
```bash
mcp-server=my-server-v2
mcp-transport=http
mcp-port=3000
```

**Invalid:**
```bash
MCP-Server=MyServer     # Uppercase not allowed
mcp-server=my_server!   # Special characters not allowed
mcp-port=3000.5         # Dots not allowed in values
```

## Integration Patterns

### Pattern 1: Label All MCP Instances at Creation

```bash
#!/bin/bash
# create-mcp-server.sh

INSTANCE_NAME=$1
MCP_SERVER_TYPE=$2
MCP_TRANSPORT=${3:-stdio}

gcloud compute instances create $INSTANCE_NAME \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --labels=mcp-server=$MCP_SERVER_TYPE,\
mcp-transport=$MCP_TRANSPORT,\
mcp-created-by=script,\
created=$(date +%Y-%m-%d | tr '-' '_')

# Usage:
# ./create-mcp-server.sh mcp-filesystem filesystem stdio
```

### Pattern 2: Automated Labeling via Startup Script

```bash
#!/bin/bash
# Instance startup script that self-labels

INSTANCE_NAME=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)

# Detect MCP server type from running processes or config
MCP_TYPE="detected-server-type"

# Add label
gcloud compute instances add-labels $INSTANCE_NAME \
    --labels=mcp-server=$MCP_TYPE,mcp-auto-detected=true \
    --zone=$ZONE
```

### Pattern 3: Discovery + Monitoring

```python
from open_cite import OpenCiteClient
import time

client = OpenCiteClient(enable_google_cloud=True)

# Continuous monitoring
while True:
    servers = client.list_gcp_mcp_servers()

    print(f"Active MCP servers: {len(servers)}")

    # Check health
    for server in servers:
        if server['status'] != 'RUNNING':
            print(f"WARNING: {server['name']} is {server['status']}")

    time.sleep(60)  # Check every minute
```

## Related Documentation

- [Google Cloud Plugin Documentation](GOOGLE_CLOUD_PLUGIN.md)
- [MCP Plugin Documentation](MCP_PLUGIN.md)
- [GCP Compute Labels Documentation](https://cloud.google.com/compute/docs/labeling-resources)
- [MCP Protocol Specification](https://modelcontextprotocol.io)

## Summary

**To make your MCP server discoverable:**

1. Add the `mcp-server` label with the server name
2. Optionally add `mcp-transport`, `mcp-port`, or other labels
3. Use OpenCITE to discover: `client.list_gcp_mcp_servers()`

**Minimum required label:**
```bash
gcloud compute instances add-labels my-instance \
    --labels=mcp-server=your-server-name \
    --zone=your-zone
```

That's it! OpenCITE will automatically discover and catalog your MCP servers.
