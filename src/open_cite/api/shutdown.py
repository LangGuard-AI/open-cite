"""
OpenCITE API Graceful Shutdown Handler.

Handles SIGTERM and SIGINT signals for graceful shutdown in Kubernetes.
"""

import logging
import signal
import sys
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_shutdown_callback: Optional[Callable] = None
_original_sigterm: Optional[signal.Handlers] = None
_original_sigint: Optional[signal.Handlers] = None


def register_shutdown_handler(callback: Callable):
    """
    Register a callback to be called on SIGTERM or SIGINT.

    The callback should clean up resources like:
    - Stopping the OTLP receiver
    - Closing database connections
    - Flushing logs

    Args:
        callback: Function to call on shutdown (no arguments)
    """
    global _shutdown_callback, _original_sigterm, _original_sigint

    _shutdown_callback = callback

    def signal_handler(signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")

        if _shutdown_callback:
            try:
                _shutdown_callback()
                logger.info("Shutdown callback completed successfully")
            except Exception as e:
                logger.error(f"Error during shutdown callback: {e}")

        # Exit cleanly
        sys.exit(0)

    # Store original handlers
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    _original_sigint = signal.getsignal(signal.SIGINT)

    # Register our handler
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Registered graceful shutdown handlers for SIGTERM and SIGINT")


def unregister_shutdown_handler():
    """
    Restore original signal handlers.
    """
    global _shutdown_callback, _original_sigterm, _original_sigint

    if _original_sigterm is not None:
        signal.signal(signal.SIGTERM, _original_sigterm)
    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)

    _shutdown_callback = None
    _original_sigterm = None
    _original_sigint = None

    logger.info("Unregistered shutdown handlers")
