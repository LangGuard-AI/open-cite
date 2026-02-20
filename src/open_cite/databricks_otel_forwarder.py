"""
Databricks OTEL Forwarder.

Auto-detects Databricks workspace OTEL endpoints and mirrors incoming
OTLP traces and logs to them.  Separate from the webhook system because
Databricks needs SDK-managed OAuth auth (tokens that auto-refresh).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class DatabricksOtelForwarder:
    """Forward OTLP payloads to Databricks workspace OTEL endpoints.

    Probes ``/api/2.0/otel/v1/traces`` and ``/api/2.0/otel/v1/logs`` on
    the workspace host.  If available, incoming payloads are forwarded
    fire-and-forget via a thread pool.

    Auth is resolved per-request so OAuth token refresh is handled
    transparently by the Databricks SDK when running as a Databricks App.
    """

    _TRACES_PATH = "/api/2.0/otel/v1/traces"
    _LOGS_PATH = "/api/2.0/otel/v1/logs"

    # Retry backoffs in seconds (matches existing webhook pattern)
    _RETRY_BACKOFFS = [0.5, 1.0]

    def __init__(self, host: str, token: Optional[str] = None):
        # Normalise host: strip trailing slash, ensure https://
        host = host.rstrip("/")
        if not host.startswith("http"):
            host = f"https://{host}"
        self._host = host
        self._token = token

        self._traces_url = f"{self._host}{self._TRACES_PATH}"
        self._logs_url = f"{self._host}{self._LOGS_PATH}"

        self._traces_available = False
        self._logs_available = False
        self._enabled = True

        self._executor = ThreadPoolExecutor(max_workers=2)

        # Lazily resolved Databricks SDK config (if available)
        self._sdk_config = None
        self._sdk_checked = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_auth_headers(self) -> dict:
        """Return fresh auth headers for a Databricks API request.

        Primary path: Databricks SDK ``WorkspaceClient`` config authenticate
        (handles OAuth token refresh for Databricks Apps).
        Fallback: static ``Bearer <token>`` if an explicit token was provided.
        """
        # Try Databricks SDK first (auto-refreshing OAuth)
        if not self._sdk_checked:
            self._sdk_checked = True
            try:
                from databricks.sdk import WorkspaceClient
                w = WorkspaceClient()
                self._sdk_config = w.config
                logger.info("Databricks SDK auth available for OTEL forwarding")
            except Exception as exc:
                logger.debug("Databricks SDK not available for auth: %s", exc)

        if self._sdk_config is not None:
            try:
                headers = {}
                self._sdk_config.authenticate(headers)
                return headers
            except Exception as exc:
                logger.warning("Databricks SDK authenticate() failed: %s", exc)

        # Fallback to static token
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}

        return {}

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def probe_endpoints(self):
        """Probe Databricks OTEL endpoints to check availability.

        POSTs minimal empty payloads.  An endpoint is considered available
        if the response status is < 500 (200 = accepted, 400 = exists but
        rejected the empty payload — either way, the endpoint is there).
        """
        auth = self._get_auth_headers()
        if not auth:
            logger.warning(
                "Databricks OTEL forwarding: no auth credentials available "
                "(need Databricks SDK or DATABRICKS_TOKEN)"
            )
            return

        for label, url, flag_attr, payload in [
            ("traces", self._traces_url, "_traces_available", {"resourceSpans": []}),
            ("logs", self._logs_url, "_logs_available", {"resourceLogs": []}),
        ]:
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={**auth, "Content-Type": "application/json"},
                    timeout=10,
                )
                available = resp.status_code < 500
                setattr(self, flag_attr, available)
                logger.info(
                    "Databricks OTEL %s endpoint %s: HTTP %d %s",
                    label, url, resp.status_code,
                    "(available)" if available else "(unavailable)",
                )
            except Exception as exc:
                setattr(self, flag_attr, False)
                logger.info(
                    "Databricks OTEL %s endpoint %s: unreachable (%s)",
                    label, url, exc,
                )

        if self._traces_available or self._logs_available:
            logger.info(
                "Databricks OTEL auto-forwarding enabled "
                "(traces=%s, logs=%s)",
                self._traces_available, self._logs_available,
            )
        else:
            logger.info("Databricks OTEL auto-forwarding disabled — no endpoints available")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_traces(self, payload: dict):
        """Submit trace payload for async forwarding (fire-and-forget)."""
        if self._enabled and self._traces_available:
            self._executor.submit(self._send, self._traces_url, payload)

    def forward_logs(self, payload: dict):
        """Submit logs payload for async forwarding (fire-and-forget)."""
        if self._enabled and self._logs_available:
            self._executor.submit(self._send, self._logs_url, payload)

    def _send(self, url: str, payload: dict):
        """POST payload with fresh auth headers and retries."""
        headers = self._get_auth_headers()
        headers["Content-Type"] = "application/json"

        last_exc = None
        for attempt in range(1 + len(self._RETRY_BACKOFFS)):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=15)
                if resp.status_code < 400:
                    return
                logger.warning(
                    "Databricks OTEL forward to %s: HTTP %d (attempt %d)",
                    url, resp.status_code, attempt + 1,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Databricks OTEL forward to %s failed (attempt %d): %s",
                    url, attempt + 1, exc,
                )
            if attempt < len(self._RETRY_BACKOFFS):
                time.sleep(self._RETRY_BACKOFFS[attempt])

        if last_exc:
            logger.error("Databricks OTEL forward to %s gave up after %d attempts", url, len(self._RETRY_BACKOFFS) + 1)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Disable forwarding and shut down the thread pool."""
        self._enabled = False
        self._traces_available = False
        self._logs_available = False
        self._executor.shutdown(wait=False)
        logger.info("Databricks OTEL forwarder shut down")
