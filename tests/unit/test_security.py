"""
Unit tests for cryptographic API key generation and timing-safe hash verification.
"""
import pytest
from src.core.security import (
    generate_raw_api_key,
    hash_api_key,
    verify_api_key_hash,
    extract_key_prefix,
)


def test_generate_raw_api_key_format():
    """Verify raw API key starts with prefix and has sufficient cryptographic entropy."""
    raw_key = generate_raw_api_key(prefix="sv_live_")
    assert raw_key.startswith("sv_live_")
    # Prefix (8 chars) + 64 hex characters (32 random bytes) = 72 chars
    assert len(raw_key) == 72
    # Ensure successive calls produce distinct keys
    assert raw_key != generate_raw_api_key(prefix="sv_live_")


def test_hash_api_key_deterministic():
    """Verify hashing is deterministic for identical key and salt."""
    key = "sv_live_abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    salt = "super-secret-cryptographic-salt-32-chars-minimum"

    hash1 = hash_api_key(key, salt)
    hash2 = hash_api_key(key, salt)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest length


def test_verify_api_key_hash_timing_safe():
    """Verify constant-time verification succeeds on match and fails on mismatch."""
    key = "sv_live_1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
    salt = "production-test-salt-32-chars-long-minimum"
    correct_hash = hash_api_key(key, salt)

    assert verify_api_key_hash(key, salt, correct_hash) is True
    # Tampered key
    assert verify_api_key_hash(key + "bad", salt, correct_hash) is False
    # Tampered salt
    assert verify_api_key_hash(key, salt + "bad", correct_hash) is False
    # Tampered hash
    assert verify_api_key_hash(key, salt, "0" * 64) is False


def test_extract_key_prefix():
    """Verify prefix extraction extracts the first 16 characters."""
    key = "sv_live_1234567890abcdef1234567890"
    prefix = extract_key_prefix(key)
    assert prefix == "sv_live_12345678"
    assert len(prefix) == 16
