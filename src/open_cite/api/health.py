"""
Open-CITE API Health Check Endpoints.

Provides Kubernetes-compatible health probes:
- /healthz: Liveness probe (is the process alive?)
- /readyz: Readiness probe (is the service ready to accept traffic?)
"""

import logging
from typing import Callable, Optional
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)

# Dependency injection for checking client/status
_get_client: Optional[Callable] = None
_get_status: Optional[Callable] = None


def init_health(get_client: Callable, get_status: Callable):
    """
    Initialize health check dependencies.

    Args:
        get_client: Callable that returns the OpenCiteClient instance
        get_status: Callable that returns the discovery status dict
    """
    global _get_client, _get_status
    _get_client = get_client
    _get_status = get_status


@health_bp.route("/healthz")
def healthz():
    """
    Liveness probe endpoint.

    Returns 200 if the process is alive. This should always succeed
    unless the process is completely unresponsive.
    """
    return jsonify({"status": "ok"}), 200


@health_bp.route("/readyz")
def readyz():
    """
    Readiness probe endpoint.

    Returns 200 if the service is ready to accept traffic.
    Checks:
    - Client is initialized
    - OTLP receiver is running (if OpenTelemetry plugin is enabled)
    """
    checks = {
        "client_initialized": False,
        "otlp_receiver_running": False,
    }

    try:
        # Check if client is initialized
        if _get_client:
            client = _get_client()
            checks["client_initialized"] = client is not None

            # Check OTLP receiver if OpenTelemetry plugin is registered
            if client and "opentelemetry" in client.plugins:
                otel_plugin = client.plugins["opentelemetry"]
                is_running = (
                    otel_plugin.server_thread is not None
                    and otel_plugin.server_thread.is_alive()
                )
                checks["otlp_receiver_running"] = is_running
            else:
                # If OpenTelemetry is not enabled, consider this check passed
                checks["otlp_receiver_running"] = True

        # Determine overall readiness
        is_ready = checks["client_initialized"] and checks["otlp_receiver_running"]

        if is_ready:
            return jsonify({"status": "ready", "checks": checks}), 200
        else:
            return jsonify({"status": "not_ready", "checks": checks}), 503

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 503
