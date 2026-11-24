"""
API Key service managing issuance, verification, scopes, and atomic quota enforcement.
"""
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.security import (
    extract_key_prefix,
    generate_raw_api_key,
    hash_api_key,
    verify_api_key_hash,
)
from src.models.api_key import APIKey
from src.models.base import utc_now


class KeyAuthError(Exception):
    """Exception raised when API key authentication or authorization fails."""
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class KeyService:
    """Service handling API key lifecycle, verification, and usage quotas."""

    @staticmethod
    async def create_api_key(
        db: AsyncSession,
        developer_id: str,
        name: str = "Default Key",
        scopes: Optional[List[str]] = None,
        monthly_quota: int = 10000,
        expires_in_days: Optional[int] = None,
        salt: Optional[str] = None,
    ) -> Tuple[str, APIKey]:
        """
        Generate a new cryptographically secure API key and store its salted hash.
        Returns the plaintext key (displayed only once to the developer) and the persisted model.
        """
        effective_salt = salt or get_settings().API_KEY_SALT
        effective_scopes = scopes or ["documents:read", "documents:write", "search:read"]

        raw_key = generate_raw_api_key(prefix="sv_live_")
        key_prefix = extract_key_prefix(raw_key)
        key_hash = hash_api_key(raw_key, effective_salt)

        expires_at = None
        if expires_in_days is not None:
            expires_at = utc_now() + timedelta(days=expires_in_days)

        api_key = APIKey(
            developer_id=developer_id,
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            scopes=effective_scopes,
            monthly_quota=monthly_quota,
            requests_used=0,
            expires_at=expires_at,
            is_active=True,
        )
        db.add(api_key)
        await db.commit()
        await db.refresh(api_key)
        return raw_key, api_key

    @staticmethod
    async def authenticate_key(
        db: AsyncSession,
        raw_key: str,
        required_scope: Optional[str] = None,
        salt: Optional[str] = None,
    ) -> APIKey:
        """
        Authenticate an API key in constant time and verify scopes, expiration, and quota.
        Raises KeyAuthError with appropriate status code if verification fails.
        """
        if not raw_key or not raw_key.startswith("sv_live_"):
            raise KeyAuthError("Invalid API key format.", status_code=401)

        effective_salt = salt or get_settings().API_KEY_SALT
        prefix = extract_key_prefix(raw_key)

        # Lookup by prefix first
        stmt = select(APIKey).where(APIKey.key_prefix == prefix)
        result = await db.execute(stmt)
        candidates = result.scalars().all()

        matching_key: Optional[APIKey] = None
        for candidate in candidates:
            if verify_api_key_hash(raw_key, effective_salt, candidate.key_hash):
                matching_key = candidate
                break

        if not matching_key:
            raise KeyAuthError("Invalid or unrecognized API key.", status_code=401)

        # Verification checks
        if matching_key.revoked_at is not None or not matching_key.is_active:
            raise KeyAuthError("API key has been revoked.", status_code=401)

        if matching_key.expires_at is not None:
            expires_at = matching_key.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if utc_now() > expires_at:
                raise KeyAuthError("API key has expired.", status_code=401)

        current_period = utc_now().strftime("%Y-%m")
        if matching_key.current_period == current_period and matching_key.requests_used >= matching_key.monthly_quota:
            raise KeyAuthError("Monthly quota exceeded for this API key.", status_code=429)

        if required_scope and not matching_key.has_scope(required_scope):
            raise KeyAuthError(
                f"Insufficient scope. Endpoint requires '{required_scope}' permission.",
                status_code=403,
            )

        return matching_key

    _key_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @classmethod
    async def record_atomic_usage(cls, db: AsyncSession, api_key_id: str) -> bool:
        """
        Atomically increment API key usage count in SQL if below monthly quota.
        Automatically resets requests_used when the calendar month rolls over (YYYY-MM).
        Guarantees strict quota enforcement against concurrent race conditions.
        Returns True if increment succeeded, False if quota is exhausted.
        """
        current_period = utc_now().strftime("%Y-%m")
        lock = cls._key_locks[api_key_id]
        async with lock:
            # 1. Check if the month has rolled over and reset atomically
            reset_stmt = (
                update(APIKey)
                .where(
                    APIKey.id == api_key_id,
                    APIKey.current_period != current_period,
                    APIKey.is_active == True,
                )
                .values(requests_used=1, current_period=current_period)
            )
            reset_res = await db.execute(reset_stmt)
            if reset_res.rowcount > 0:
                await db.commit()
                return True

            # 2. Increment within current monthly period if below quota
            stmt = (
                update(APIKey)
                .where(
                    APIKey.id == api_key_id,
                    APIKey.current_period == current_period,
                    APIKey.requests_used < APIKey.monthly_quota,
                    APIKey.is_active == True,
                )
                .values(requests_used=APIKey.requests_used + 1)
            )
            res = await db.execute(stmt)
            await db.commit()
            return res.rowcount > 0

    @staticmethod
    async def revoke_api_key(db: AsyncSession, developer_id: str, key_id: str) -> bool:
        """Revoke an active API key immediately."""
        stmt = (
            update(APIKey)
            .where(
                APIKey.id == key_id,
                APIKey.developer_id == developer_id,
                APIKey.is_active == True,
            )
            .values(
                is_active=False,
                revoked_at=utc_now(),
            )
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def list_keys_for_developer(db: AsyncSession, developer_id: str) -> List[APIKey]:
        """Retrieve all API keys belonging to a developer."""
        stmt = select(APIKey).where(APIKey.developer_id == developer_id).order_by(APIKey.created_at.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())
