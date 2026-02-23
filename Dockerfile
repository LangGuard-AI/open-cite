# OpenCITE API Service Dockerfile
# Multi-stage build for minimal image size

# Stage 1: Build
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy source code
COPY pyproject.toml README.md ./
COPY src/ src/

# Build wheel
RUN python -m build --wheel


# Stage 2: Runtime
FROM python:3.12-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash opencite

WORKDIR /app

# Install the wheel and gunicorn
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl gunicorn>=21.0.0 && \
    rm /tmp/*.whl

# Create data directory for persistence
RUN mkdir -p /data && chown opencite:opencite /data

# Switch to non-root user
USER opencite

# Environment variables with defaults
ENV OPENCITE_HOST=0.0.0.0 \
    OPENCITE_PORT=8080 \
    OPENCITE_OTLP_HOST=0.0.0.0 \
    OPENCITE_OTLP_PORT=4318 \
    OPENCITE_ENABLE_OTEL=true \
    OPENCITE_ENABLE_MCP=true \
    OPENCITE_ENABLE_DATABRICKS=false \
    OPENCITE_ENABLE_GOOGLE_CLOUD=false \
    OPENCITE_AUTO_START=true \
    OPENCITE_LOG_LEVEL=INFO \
    OPENCITE_PERSISTENCE_ENABLED=false \
    OPENCITE_DB_PATH=/data/opencite.db

# Volume for persistent data
VOLUME /data

# Expose ports
# 8080: API server
# 4318: OTLP HTTP receiver
EXPOSE 8080 4318

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

# Run with gunicorn
# - 1 worker (single process for in-memory state sharing)
# - 4 threads (concurrent request handling)
# - No --capture-output: all logging uses Python's logging module, so
#   stdout/stderr capture is unnecessary and causes duplicate log lines
#   when combined with --error-logfile -.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--worker-tmp-dir", "/dev/shm", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "open_cite.api.app:create_app()"]
