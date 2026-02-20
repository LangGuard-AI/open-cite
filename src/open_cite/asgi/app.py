"""
Composite ASGI application that routes between gRPC, SocketIO, and Flask.

Dispatches incoming requests:
  - content-type: application/grpc  → GrpcOtlpHandler
  - /socket.io/ (WebSocket or HTTP) → socketio.ASGIApp
  - Everything else (HTTP)          → WsgiToAsgi(Flask)
"""

import logging
from typing import Callable, Optional

from asgiref.wsgi import WsgiToAsgi

from .grpc_handler import GrpcOtlpHandler

logger = logging.getLogger(__name__)


def create_asgi_app(
    flask_app,
    sio_server=None,
    ingest_fn: Optional[Callable] = None,
    logs_ingest_fn: Optional[Callable] = None,
):
    """Create a composite ASGI application.

    Args:
        flask_app: The Flask WSGI application.
        sio_server: Optional python-socketio Server instance (None for API-only mode).
        ingest_fn: Callable(data, headers) for OTLP trace ingestion.
                   If None, gRPC OTLP is disabled.
        logs_ingest_fn: Optional callable(data, headers) for OTLP log ingestion.
                        Logs are converted to synthetic traces before processing.

    Returns:
        ASGI callable.
    """
    # Wrap Flask as ASGI
    wsgi_asgi = WsgiToAsgi(flask_app)

    # gRPC handler (only if ingest_fn provided)
    grpc_handler = GrpcOtlpHandler(ingest_fn, logs_ingest_fn=logs_ingest_fn) if ingest_fn else None

    # SocketIO ASGI wrapper (only if sio_server provided)
    sio_asgi = None
    if sio_server is not None:
        import socketio as sio_module
        # Wrap: SocketIO handles /socket.io/ requests, falls back to wsgi_asgi
        sio_asgi = sio_module.ASGIApp(sio_server, other_asgi_app=wsgi_asgi)

    async def asgi_dispatch(scope, receive, send):
        if scope["type"] == "lifespan":
            # Handle lifespan events (startup/shutdown)
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return

        if scope["type"] == "websocket":
            # WebSocket requests go to SocketIO if available
            if sio_asgi is not None:
                await sio_asgi(scope, receive, send)
            else:
                # No SocketIO — reject WebSocket
                await send({"type": "websocket.close", "code": 1000})
            return

        # HTTP requests
        if scope["type"] == "http":
            # Check for gRPC content-type
            headers = dict(scope.get("headers", []))
            content_type = headers.get(b"content-type", b"").decode("latin-1", errors="replace")
            path = scope.get("path", "")
            method = scope.get("method", "?")

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[ASGI] {method} {path} content-type={content_type!r} "
                    f"http_version={scope.get('http_version', '?')}"
                )

            if "application/grpc" in content_type and grpc_handler is not None:
                await grpc_handler(scope, receive, send)
                return

            # Check if this is a socket.io polling request
            if path.startswith("/socket.io") and sio_asgi is not None:
                await sio_asgi(scope, receive, send)
                return

            # Everything else goes to Flask (via ASGI wrapper)
            await wsgi_asgi(scope, receive, send)
            return

    return asgi_dispatch
