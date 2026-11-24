"""
Unit tests for KeyService: key lifecycle, verification, scopes, and atomic quota enforcement.
"""
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.models.base import Base
from src.models.developer import Developer
from src.services.key_service import KeyService, KeyAuthError


@pytest.fixture
async def async_session():
    """In-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def test_developer(async_session: AsyncSession) -> Developer:
    """Fixture creating a test developer."""
    dev = Developer(email="security@example.com", name="Security Lead")
    async_session.add(dev)
    await async_session.commit()
    await async_session.refresh(dev)
    return dev


@pytest.mark.asyncio
async def test_create_and_verify_api_key(async_session: AsyncSession, test_developer: Developer):
    """Test standard API key creation and subsequent verification."""
    salt = "test-salt-secret-at-least-32-chars-long"
    raw_key, api_key_model = await KeyService.create_api_key(
        db=async_session,
        developer_id=test_developer.id,
        name="Production Ingestion Key",
        scopes=["documents:read", "documents:write"],
        monthly_quota=1000,
        salt=salt,
    )

    assert raw_key.startswith("sv_live_")
    assert api_key_model.id is not None
    assert api_key_model.requests_used == 0

    # Verify the key using the raw key
    verified_key = await KeyService.authenticate_key(
        db=async_session,
        raw_key=raw_key,
        required_scope="documents:write",
        salt=salt,
    )
    assert verified_key.id == api_key_model.id
    assert verified_key.developer_id == test_developer.id


@pytest.mark.asyncio
async def test_insufficient_scope_rejected(async_session: AsyncSession, test_developer: Developer):
    """Test that a key without the required scope is rejected."""
    salt = "test-salt-secret-at-least-32-chars-long"
    raw_key, _ = await KeyService.create_api_key(
        db=async_session,
        developer_id=test_developer.id,
        name="Read Only Key",
        scopes=["documents:read"],
        salt=salt,
    )

    with pytest.raises(KeyAuthError, match="Insufficient scope"):
        await KeyService.authenticate_key(
            db=async_session,
            raw_key=raw_key,
            required_scope="documents:write",
            salt=salt,
        )


@pytest.mark.asyncio
async def test_expired_key_rejected(async_session: AsyncSession, test_developer: Developer):
    """Test that an expired key is rejected."""
    salt = "test-salt-secret-at-least-32-chars-long"
    raw_key, api_key = await KeyService.create_api_key(
        db=async_session,
        developer_id=test_developer.id,
        name="Expired Key",
        scopes=["documents:read"],
        expires_in_days=-1,  # Expired yesterday
        salt=salt,
    )

    with pytest.raises(KeyAuthError, match="API key has expired"):
        await KeyService.authenticate_key(
            db=async_session,
            raw_key=raw_key,
            salt=salt,
        )


@pytest.mark.asyncio
async def test_revoked_key_rejected(async_session: AsyncSession, test_developer: Developer):
    """Test that a revoked key cannot authenticate."""
    salt = "test-salt-secret-at-least-32-chars-long"
    raw_key, api_key = await KeyService.create_api_key(
        db=async_session,
        developer_id=test_developer.id,
        name="To Revoke Key",
        scopes=["documents:read"],
        salt=salt,
    )

    # Revoke key
    revoked = await KeyService.revoke_api_key(
        db=async_session,
        developer_id=test_developer.id,
        key_id=api_key.id,
    )
    assert revoked is True

    with pytest.raises(KeyAuthError, match="API key has been revoked"):
        await KeyService.authenticate_key(
            db=async_session,
            raw_key=raw_key,
            salt=salt,
        )


@pytest.mark.asyncio
async def test_atomic_quota_enforcement(async_session: AsyncSession, test_developer: Developer):
    """Test atomic quota tracking and rejection when quota is exhausted."""
    salt = "test-salt-secret-at-least-32-chars-long"
    raw_key, api_key = await KeyService.create_api_key(
        db=async_session,
        developer_id=test_developer.id,
        name="Small Quota Key",
        scopes=["search:read"],
        monthly_quota=2,  # Only 2 requests allowed
        salt=salt,
    )

    # 1st request -> allowed
    success1 = await KeyService.record_atomic_usage(async_session, api_key.id)
    assert success1 is True

    # 2nd request -> allowed
    success2 = await KeyService.record_atomic_usage(async_session, api_key.id)
    assert success2 is True

    # 3rd request -> blocked by atomic quota
    success3 = await KeyService.record_atomic_usage(async_session, api_key.id)
    assert success3 is False

    # Authenticating should now fail due to quota exhaustion
    with pytest.raises(KeyAuthError, match="Monthly quota exceeded"):
        await KeyService.authenticate_key(
            db=async_session,
            raw_key=raw_key,
            salt=salt,
        )


@pytest.mark.asyncio
async def test_monthly_quota_period_rollover(async_session: AsyncSession, test_developer: Developer):
    """Test that requests_used resets to 1 and updates current_period when month rolls over."""
    from sqlalchemy import update
    from src.models.api_key import APIKey
    salt = "test-salt-secret-at-least-32-chars-long"
    raw_key, api_key = await KeyService.create_api_key(
        db=async_session,
        developer_id=test_developer.id,
        name="Rollover Key",
        scopes=["search:read"],
        monthly_quota=2,
        salt=salt,
    )

    # Simulate key was exhausted in a previous month (e.g. 2026-01)
    await async_session.execute(
        update(APIKey)
        .where(APIKey.id == api_key.id)
        .values(current_period="2026-01", requests_used=2)
    )
    await async_session.commit()

    # Now record usage in current month -> should detect rollover, reset counter, and allow!
    rollover_ok = await KeyService.record_atomic_usage(async_session, api_key.id)
    assert rollover_ok is True

    # Check key state
    updated_key = await async_session.get(APIKey, api_key.id)
    assert updated_key.requests_used == 1
    assert updated_key.current_period != "2026-01"
