"""
ASGI handler for gRPC OTLP trace and log ingestion.

Handles gRPC-framed requests on the standard OTLP gRPC paths,
deserializes protobuf, converts to JSON-compatible dict, and
forwards to the injected ingest functions.
"""

import logging
import struct
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Standard gRPC service paths for OTLP export
GRPC_TRACE_PATH = "/opentelemetry.proto.collector.trace.v1.TraceService/Export"
GRPC_LOGS_PATH = "/opentelemetry.proto.collector.logs.v1.LogsService/Export"


class GrpcOtlpHandler:
    """ASGI application that handles gRPC OTLP trace and log export requests.

    Args:
        ingest_fn: Callable(data: dict, headers: dict) invoked with the
                   deserialized trace payload and forwarded headers.
        logs_ingest_fn: Optional callable(data: dict, headers: dict) invoked
                        with the deserialized logs payload and forwarded headers.
    """

    def __init__(self, ingest_fn: Callable, logs_ingest_fn: Optional[Callable] = None):
        self._ingest_fn = ingest_fn
        self._logs_ingest_fn = logs_ingest_fn

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await _send_grpc_error(send, 12, "Unimplemented")
            return

        path = scope.get("path", "")
        if path == GRPC_TRACE_PATH:
            await self._handle_traces(scope, receive, send)
        elif path == GRPC_LOGS_PATH:
            await self._handle_logs(scope, receive, send)
        else:
            await _send_grpc_error(send, 12, f"Unknown service path: {path}")

    async def _handle_traces(self, scope, receive, send):
        """Handle gRPC OTLP trace export requests."""
        body = await _read_grpc_body(receive)
        try:
            protobuf_data = _parse_grpc_frame(body)
            if protobuf_data is None:
                await _send_grpc_error(send, 3, "Invalid gRPC frame")
                return

            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
                ExportTraceServiceResponse,
            )
            from google.protobuf.json_format import MessageToDict

            request = ExportTraceServiceRequest()
            request.ParseFromString(protobuf_data)
            data = MessageToDict(request)

            headers = _extract_headers(scope)
            self._ingest_fn(data, headers)

            response = ExportTraceServiceResponse()
            await _send_grpc_success(send, response.SerializeToString())

        except ImportError:
            logger.error("opentelemetry-proto or protobuf not installed for gRPC support")
            await _send_grpc_error(send, 12, "gRPC protobuf support not available")
        except Exception as e:
            logger.error(f"gRPC OTLP trace handler error: {e}")
            await _send_grpc_error(send, 13, str(e))

    async def _handle_logs(self, scope, receive, send):
        """Handle gRPC OTLP logs export requests."""
        if self._logs_ingest_fn is None:
            await _send_grpc_error(send, 12, "Logs ingestion not configured")
            return

        body = await _read_grpc_body(receive)
        try:
            protobuf_data = _parse_grpc_frame(body)
            if protobuf_data is None:
                await _send_grpc_error(send, 3, "Invalid gRPC frame")
                return

            from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
                ExportLogsServiceRequest,
                ExportLogsServiceResponse,
            )
            from google.protobuf.json_format import MessageToDict

            request = ExportLogsServiceRequest()
            request.ParseFromString(protobuf_data)
            data = MessageToDict(request)

            headers = _extract_headers(scope)
            self._logs_ingest_fn(data, headers)

            response = ExportLogsServiceResponse()
            await _send_grpc_success(send, response.SerializeToString())

        except ImportError:
            logger.error("opentelemetry-proto or protobuf not installed for gRPC logs support")
            await _send_grpc_error(send, 12, "gRPC protobuf support not available")
        except Exception as e:
            logger.error(f"gRPC OTLP logs handler error: {e}")
            await _send_grpc_error(send, 13, str(e))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _read_grpc_body(receive) -> bytes:
    """Read the full request body from an ASGI receive channel."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    return body


def _parse_grpc_frame(body: bytes) -> Optional[bytes]:
    """Parse a gRPC frame and return the protobuf payload, or None on error."""
    if len(body) < 5:
        return None

    compressed = body[0]
    msg_length = struct.unpack(">I", body[1:5])[0]
    protobuf_data = body[5:5 + msg_length]

    if compressed:
        logger.warning("Compressed gRPC messages not supported")
        return None

    if len(protobuf_data) != msg_length:
        logger.warning("Incomplete gRPC frame")
        return None

    return protobuf_data


def _extract_headers(scope) -> dict:
    """Extract forwarded headers from an ASGI scope, skipping hop-by-hop headers."""
    _SKIP = {b"content-length", b"transfer-encoding", b"connection",
             b"keep-alive", b"te", b"trailers", b"upgrade", b"content-type"}
    headers = {}
    for name, value in scope.get("headers", []):
        if name in _SKIP:
            continue
        key = name.decode("latin-1")
        val = value.decode("latin-1")
        if key.lower() == "host":
            headers["OTEL-HOST"] = val
        else:
            headers[key] = val
    return headers


async def _send_grpc_success(send, response_bytes: bytes):
    """Send a successful gRPC response."""
    frame = struct.pack(">BI", 0, len(response_bytes)) + response_bytes
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"content-type", b"application/grpc"),
            (b"grpc-status", b"0"),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": frame,
        "more_body": False,
    })


async def _send_grpc_error(send, status_code: int, message: str):
    """Send a gRPC error response."""
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"content-type", b"application/grpc"),
            (b"grpc-status", str(status_code).encode()),
            (b"grpc-message", message.encode()),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    })
