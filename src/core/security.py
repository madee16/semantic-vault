"""
Cryptographic security module for API key generation, prefix extraction, and constant-time verification.
"""
import hashlib
import hmac
import secrets


def generate_raw_api_key(prefix: str = "sv_live_") -> str:
    """
    Generate a cryptographically secure random API key.
    Format: <prefix><64-hex-chars> (32 random bytes)
    Example: sv_live_7a8b9c0d... (72 characters total)
    """
    random_bytes = secrets.token_bytes(32)
    hex_token = random_bytes.hex()
    return f"{prefix}{hex_token}"


def extract_key_prefix(raw_key: str, length: int = 16) -> str:
    """Extract public prefix from raw key for database lookup."""
    return raw_key[:length]


def hash_api_key(raw_key: str, salt: str) -> str:
    """
    Compute salted SHA-256 hash of the plain API key.
    Raw keys are never stored in plaintext in the database.
    """
    salted_payload = f"{raw_key}:{salt}".encode("utf-8")
    return hashlib.sha256(salted_payload).hexdigest()


def verify_api_key_hash(raw_key: str, salt: str, expected_hash: str) -> bool:
    """
    Verify plain key against stored hash using constant-time comparison
    to prevent timing side-channel attacks.
    """
    computed_hash = hash_api_key(raw_key, salt)
    return hmac.compare_digest(computed_hash, expected_hash)
