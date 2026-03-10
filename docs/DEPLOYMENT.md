# Deploying Open-CITE

Open-CITE runs as a headless REST API service in Docker or Kubernetes. The container exposes two ports:

- **8080** -- REST API (served by Gunicorn)
- **4318** -- OTLP/HTTP receiver (for trace ingestion)

## Docker

### Build

```bash
docker build -t opencite .
```

### Run

```bash
docker run -d \
  --name opencite \
  -p 8080:8080 \
  -p 4318:4318 \
  -v opencite-data:/data \
  opencite
```

The API is available at `http://localhost:8080` and the OTLP receiver at `http://localhost:4318/v1/traces`.

### Verify

```bash
# Health check
curl http://localhost:8080/healthz

# Readiness check
curl http://localhost:8080/readyz

# List plugin types
curl http://localhost:8080/api/v1/plugin-types
```

### Enable persistence

```bash
docker run -d \
  --name opencite \
  -p 8080:8080 \
  -p 4318:4318 \
  -v opencite-data:/data \
  -e OPENCITE_PERSISTENCE_ENABLED=true \
  opencite
```

SQLite is stored at `/data/opencite.db` inside the container. The `-v opencite-data:/data` mount ensures data survives container restarts.

### Enable plugins

Pass plugin credentials as environment variables:

```bash
docker run -d \
  --name opencite \
  -p 8080:8080 \
  -p 4318:4318 \
  -v opencite-data:/data \
  -e OPENCITE_PERSISTENCE_ENABLED=true \
  -e OPENCITE_ENABLE_DATABRICKS=true \
  -e DATABRICKS_HOST=https://dbc-xxx.cloud.databricks.com \
  -e DATABRICKS_TOKEN=dapi... \
  -e DATABRICKS_WAREHOUSE_ID=... \
  opencite
```

### Docker Compose

```yaml
services:
  opencite:
    build: .
    ports:
      - "8080:8080"
      - "4318:4318"
    volumes:
      - opencite-data:/data
    environment:
      OPENCITE_PERSISTENCE_ENABLED: "true"
      OPENCITE_ENABLE_OTEL: "true"
      OPENCITE_LOG_LEVEL: "INFO"
    restart: unless-stopped

volumes:
  opencite-data:
```

## Kubernetes

Production-ready manifests are provided in the `k8s/` directory.

### Quick deploy

```bash
# Apply all manifests
kubectl apply -k k8s/

# Verify
kubectl -n opencite get pods
kubectl -n opencite logs deployment/opencite-api
```

### What gets created

| Resource | Name | Description |
|----------|------|-------------|
| Namespace | `opencite` | Dedicated namespace |
| Deployment | `opencite-api` | Single-replica pod (Recreate strategy) |
| Service | `opencite-api` | ClusterIP with ports 80 (API) and 4318 (OTLP) |
| ConfigMap | `opencite-config` | Non-sensitive configuration |
| Secret | `opencite-secrets` | Plugin credentials (placeholder) |
| PVC | `opencite-data` | 1Gi persistent storage for SQLite |

### Configure

Edit `k8s/configmap.yaml` for non-sensitive settings:

```yaml
data:
  OPENCITE_ENABLE_OTEL: "true"
  OPENCITE_ENABLE_DATABRICKS: "true"
  OPENCITE_PERSISTENCE_ENABLED: "true"
  OPENCITE_LOG_LEVEL: "INFO"
```

Edit `k8s/secret.yaml` for credentials:

```yaml
stringData:
  DATABRICKS_HOST: "https://dbc-xxx.cloud.databricks.com"
  DATABRICKS_TOKEN: "dapi..."
  DATABRICKS_WAREHOUSE_ID: "..."
```

Re-apply after changes:

```bash
kubectl apply -k k8s/
kubectl -n opencite rollout restart deployment/opencite-api
```

### Use a custom image

Edit `k8s/kustomization.yaml`:

```yaml
images:
  - name: opencite
    newName: your-registry.com/opencite
    newTag: v1.0.0
```

### Expose OTLP externally

To receive traces from outside the cluster, uncomment the NodePort service in `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: opencite-otlp
  namespace: opencite
spec:
  type: NodePort
  selector:
    app.kubernetes.io/name: opencite
    app.kubernetes.io/component: api
  ports:
    - name: otlp
      port: 4318
      targetPort: otlp
      nodePort: 30318
```

Or use an Ingress / LoadBalancer for production traffic.

### Access the API from within the cluster

Other pods can reach Open-CITE at:

```
http://opencite-api.opencite.svc.cluster.local/api/v1/...
http://opencite-api.opencite.svc.cluster.local:4318/v1/traces
```

### Health probes

The deployment includes:

- **Liveness:** `GET /healthz` (30s interval) -- restarts the pod if the process is hung
- **Readiness:** `GET /readyz` (10s interval) -- removes the pod from service if the client isn't initialized

### Resource limits

Default requests/limits in `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

Adjust based on trace volume and number of plugins.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCITE_HOST` | `0.0.0.0` | API bind address |
| `OPENCITE_PORT` | `8080` | API port |
| `OPENCITE_OTLP_HOST` | `0.0.0.0` | OTLP receiver bind address |
| `OPENCITE_OTLP_PORT` | `4318` | OTLP receiver port |
| `OPENCITE_OTLP_EMBEDDED` | `false` | Serve OTLP on the API port instead of a separate server |
| `OPENCITE_ENABLE_OTEL` | `true` | Enable OpenTelemetry plugin |
| `OPENCITE_ENABLE_MCP` | `true` | Enable MCP discovery |
| `OPENCITE_ENABLE_DATABRICKS` | `false` | Enable Databricks plugin |
| `OPENCITE_ENABLE_GOOGLE_CLOUD` | `false` | Enable Google Cloud plugin |
| `OPENCITE_AUTO_START` | `true` | Auto-configure plugins on startup |
| `OPENCITE_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `OPENCITE_PERSISTENCE_ENABLED` | `false` | Enable SQLite persistence |
| `OPENCITE_DB_PATH` | `/data/opencite.db` | SQLite database path |

### Plugin credentials

| Variable | Plugin |
|----------|--------|
| `DATABRICKS_HOST` | Databricks |
| `DATABRICKS_TOKEN` | Databricks |
| `DATABRICKS_WAREHOUSE_ID` | Databricks |
| `GCP_PROJECT_ID` | Google Cloud |
| `GCP_LOCATION` | Google Cloud |
| `AWS_ACCESS_KEY_ID` | AWS (Bedrock/SageMaker) |
| `AWS_SECRET_ACCESS_KEY` | AWS (Bedrock/SageMaker) |
| `AWS_REGION` | AWS (Bedrock/SageMaker) |

## Architecture Notes

- **Single replica:** The default deployment uses one replica with SQLite persistence. SQLite requires exclusive access, so use the `Recreate` deployment strategy (not `RollingUpdate`).
- **Gunicorn:** 1 worker, 4 threads. A single worker process is required because plugins maintain in-memory state that must be shared across requests.
- **Non-root:** The container runs as user `opencite` (UID 1000) with `allowPrivilegeEscalation: false`.
- **Graceful shutdown:** 30-second termination grace period allows plugins to flush state and close connections.

## Related Documentation

- [REST API Reference](REST_API.md) -- API endpoints
- [Sending Traces to Open-CITE](SENDING_TRACES.md) -- Configure external trace sources
- [Development Guide](DEVELOPMENT.md) -- Local development
