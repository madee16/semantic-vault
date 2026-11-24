"""
Unit tests for the SlidingWindowRateLimiter middleware.
"""
import pytest
from fastapi import HTTPException
from src.api.middleware.rate_limiter import SlidingWindowRateLimiter


def test_rate_limiter_allows_under_limit():
    """Verify requests under the rate limit are permitted."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=5)
    key_id = "test-key-1"

    for _ in range(5):
        limiter.check_rate_limit(key_id)  # Should not raise


def test_rate_limiter_blocks_over_limit():
    """Verify exceeding requests_per_minute raises HTTPException 429 with Retry-After header."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=3)
    key_id = "test-key-2"

    for _ in range(3):
        limiter.check_rate_limit(key_id)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(key_id)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers
    assert int(exc_info.value.headers["Retry-After"]) >= 1
