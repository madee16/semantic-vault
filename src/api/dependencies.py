"""
FastAPI dependency injection for API key authentication, permission scopes, and rate limiting.
"""
from typing import Callable, Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import logger
from src.api.middleware.rate_limiter import rate_limiter
from src.db.session import get_db
from src.models.api_key import APIKey
from src.services.key_service import KeyAuthError, KeyService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(required_scope: Optional[str] = None) -> Callable:
    """
    Factory dependency creating a validator for X-API-Key with scope enforcement.
    Applies constant-time verification, rate-limiting, and atomic quota tracking.
    """
    async def dependency(
        raw_key: Optional[str] = Security(api_key_header),
        db: AsyncSession = Depends(get_db),
    ) -> APIKey:
        if not raw_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing 'X-API-Key' authentication header.",
            )

        try:
            api_key = await KeyService.authenticate_key(
                db=db,
                raw_key=raw_key,
                required_scope=required_scope,
            )
        except KeyAuthError as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=e.message,
            )

        # 1. Enforce rolling rate limit
        rate_limiter.check_rate_limit(api_key.id)

        # 2. Record atomic quota usage in SQL
        quota_ok = await KeyService.record_atomic_usage(db, api_key.id)
        if not quota_ok:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Monthly request quota exhausted for this API key.",
            )

        return api_key

    return dependency
