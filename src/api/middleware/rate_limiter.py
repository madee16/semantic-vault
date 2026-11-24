"""
In-memory sliding window rate limiter for per-key rate limiting.
"""
import time
from collections import defaultdict
from typing import Dict, List, Optional
from fastapi import HTTPException, status


class SlidingWindowRateLimiter:
    """
    Sliding window in-memory rate limiter per key.
    Enforces maximum requests within a rolling time window (default 60 seconds).
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._history: Dict[str, List[float]] = defaultdict(list)

    def check_rate_limit(self, key_id: str) -> None:
        """
        Record request timestamp for key_id and enforce rate limit.
        Raises HTTPException(429) if threshold exceeded.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Prune timestamps older than window
        timestamps = self._history[key_id]
        valid_timestamps = [t for t in timestamps if t > window_start]

        if len(valid_timestamps) >= self.requests_per_minute:
            retry_after = int(self.window_seconds - (now - valid_timestamps[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit of {self.requests_per_minute} requests per minute exceeded.",
                headers={"Retry-After": str(max(1, retry_after))},
            )

        valid_timestamps.append(now)
        self._history[key_id] = valid_timestamps

    def reset(self, key_id: Optional[str] = None) -> None:
        """Reset rate limiter state (useful for tests)."""
        if key_id:
            self._history.pop(key_id, None)
        else:
            self._history.clear()


rate_limiter = SlidingWindowRateLimiter()
