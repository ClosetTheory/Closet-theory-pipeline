"""Presentation-grade password hashing and session tokens — stdlib-only, no new
dependencies. Explicitly out of scope (per product decision): OAuth, MFA, rate limiting,
refresh-token rotation, email verification. Password hashing itself is still done properly
(salted PBKDF2) — that's baseline hygiene, not "extra" security.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional
from app.config import settings

PBKDF2_ITERATIONS = 260_000
SALT_BYTES = 16


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_str, salt_hex, digest_hex = password_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations_str))
    return hmac.compare_digest(actual, expected)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_session_token(user_id: str, expires_in_days: Optional[int] = None) -> str:
    expires_in_days = expires_in_days if expires_in_days is not None else settings.AUTH_TOKEN_EXPIRE_DAYS
    payload = {"sub": user_id, "exp": int(time.time()) + expires_in_days * 86400}
    payload_b64 = _b64encode(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(settings.AUTH_SECRET_KEY.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64encode(signature)}"


def verify_session_token(token: str) -> Optional[str]:
    """Returns the user_id if the token is validly signed and not expired, else None."""
    try:
        payload_b64, signature_b64 = token.split(".")
    except ValueError:
        return None

    expected_signature = hmac.new(
        settings.AUTH_SECRET_KEY.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
    ).digest()
    try:
        actual_signature = _b64decode(signature_b64)
    except Exception:
        return None
    if not hmac.compare_digest(actual_signature, expected_signature):
        return None

    try:
        payload = json.loads(_b64decode(payload_b64))
    except Exception:
        return None

    if payload.get("exp", 0) < time.time():
        return None
    return payload.get("sub")
