"""Unit tests for pipeline idempotency and hashing."""

from app.pipeline.idempotency import compute_idempotency_key, compute_stage_input_hash


def test_input_hash_determinism():
    data_1 = {"category": "shirt", "color": "white"}
    data_2 = {"color": "white", "category": "shirt"}  # different key order

    hash_1 = compute_stage_input_hash(data_1)
    hash_2 = compute_stage_input_hash(data_2)

    assert hash_1 == hash_2
    assert len(hash_1) == 64


def test_input_hash_different_data():
    hash_a = compute_stage_input_hash(b"image-bytes-a")
    hash_b = compute_stage_input_hash(b"image-bytes-b")

    assert hash_a != hash_b


def test_composite_idempotency_key():
    key_1 = compute_idempotency_key("garm_1", "STAGE_01", "hash123", "v1")
    key_2 = compute_idempotency_key("garm_1", "STAGE_01", "hash123", "v1")
    key_3 = compute_idempotency_key("garm_1", "STAGE_01", "hash123", "v2")

    assert key_1 == key_2
    assert key_1 != key_3
