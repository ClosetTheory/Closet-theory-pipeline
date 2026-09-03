"""Idempotency hash generation and evaluation."""

import hashlib
import json
from typing import Any, Dict, Optional


def compute_stage_input_hash(input_data: Any) -> str:
    """Produces deterministic SHA256 hex digest for any structured input or bytes."""
    if isinstance(input_data, bytes):
        return hashlib.sha256(input_data).hexdigest()
    if isinstance(input_data, str):
        return hashlib.sha256(input_data.encode("utf-8")).hexdigest()
    if isinstance(input_data, (dict, list)):
        serialized = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return hashlib.sha256(str(input_data).encode("utf-8")).hexdigest()


def compute_idempotency_key(
    garment_id: str,
    stage: str,
    input_hash: str,
    version: str,
) -> str:
    """Computes unique composite hash: garment_id + stage + input_hash + algorithm/model_version."""
    raw = f"{garment_id}:{stage}:{input_hash}:{version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
