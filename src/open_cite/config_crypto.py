"""
Symmetric encryption for sensitive plugin configuration fields.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the ``cryptography`` library.
The master key is read from the ``OPENCITE_CONFIG_KEY`` environment variable.
When the variable is unset, encryption is **skipped** and a warning is logged
on first use — configs are stored in plaintext as before.

Generate a key once with::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Then export it::

    export OPENCITE_CONFIG_KEY="<base64-key>"
"""

import base64
import json
import logging
import os
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Config field names that hold secrets.  Kept in sync with plugin CONFIG_FIELDS
# entries whose ``type`` is ``"password"`` or that are known credential fields.
SENSITIVE_FIELDS: Set[str] = {
    "token",
    "api_key",
    "password",
    "secret_access_key",
    "session_token",
    "access_token",
    "service_account_key",
    "client_secret",
    "access_key_id",
}

_ENCRYPTED_PREFIX = "enc:"

_fernet: Optional[Any] = None
_warned = False


def _get_fernet():
    """Return a Fernet instance, or None when no key is configured."""
    global _fernet, _warned
    if _fernet is not None:
        return _fernet

    key = os.environ.get("OPENCITE_CONFIG_KEY")
    if not key:
        if not _warned:
            logger.warning(
                "OPENCITE_CONFIG_KEY not set — plugin credentials will be stored in plaintext. "
                "Generate a key with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
            _warned = True
        return None

    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
        return _fernet
    except Exception as exc:
        logger.error("Failed to initialise Fernet with OPENCITE_CONFIG_KEY: %s", exc)
        return None


def encrypt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *config* with sensitive fields encrypted."""
    f = _get_fernet()
    if f is None:
        return config

    out = dict(config)
    for key in SENSITIVE_FIELDS:
        val = out.get(key)
        if val and isinstance(val, str) and not val.startswith(_ENCRYPTED_PREFIX):
            encrypted = f.encrypt(val.encode("utf-8")).decode("ascii")
            out[key] = _ENCRYPTED_PREFIX + encrypted
    return out


def decrypt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *config* with sensitive fields decrypted."""
    f = _get_fernet()
    if f is None:
        return config

    out = dict(config)
    for key in SENSITIVE_FIELDS:
        val = out.get(key)
        if val and isinstance(val, str) and val.startswith(_ENCRYPTED_PREFIX):
            try:
                token = val[len(_ENCRYPTED_PREFIX):]
                out[key] = f.decrypt(token.encode("ascii")).decode("utf-8")
            except Exception as exc:
                logger.error("Failed to decrypt config field '%s': %s", key, exc)
    return out
