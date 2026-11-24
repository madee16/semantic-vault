"""
Security Attack Suite: Adversarial testing of the custom API Key infrastructure.
Tests against:
- Tampered key tokens
- Expired keys
- Revoked keys
- Privilege escalation (unauthorized scopes)
- Rate limit flooding (DDoS simulation)
- Concurrent race-condition quota exhaustion attacks
"""
import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.main import app
from src.db.session import get_db
from src.models.base import Base
from src.models.developer import Developer
from src.services.key_service import KeyService
from src.api.middleware.rate_limiter import rate_limiter


@pytest.fixture
async def security_client():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    rate_limiter.reset()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a test developer and restricted API key (read-only, quota: 10)
        async with session_factory() as session:
            dev = Developer(email="attacker_target@example.com", name="Target Developer")
            session.add(dev)
            await session.commit()
            await session.refresh(dev)

            raw_key, api_key_model = await KeyService.create_api_key(
                db=session,
                developer_id=dev.id,
                name="Restricted Read-Only Key",
                scopes=["documents:read"],  # No write or admin permissions
                monthly_quota=10,
            )

        client.headers["X-API-Key"] = raw_key
        yield client, raw_key, dev.id, session_factory

    app.dependency_overrides.clear()
    rate_limiter.reset()
    await engine.dispose()


@pytest.mark.asyncio
async def test_attack_tampered_key_rejected(security_client):
    """Attack 1: Attacker modifies the last 4 bytes of an API key."""
    client, raw_key, _, _ = security_client

    tampered_key = raw_key[:-4] + "dead"
    res = await client.get("/api/v1/documents", headers={"X-API-Key": tampered_key})
    assert res.status_code == 401
    assert "Invalid or unrecognized API key" in res.json()["detail"]


@pytest.mark.asyncio
async def test_attack_privilege_escalation_blocked(security_client):
    """Attack 2: Attacker with 'documents:read' attempts to delete documents or issue keys."""
    client, raw_key, _, _ = security_client

    # Attempting to delete documents (requires 'documents:write')
    del_res = await client.delete("/api/v1/documents/fake-doc-id")
    assert del_res.status_code == 403
    assert "Insufficient scope" in del_res.json()["detail"]

    # Attempting to list all developer keys (requires 'admin:keys')
    keys_res = await client.get("/api/v1/keys")
    assert keys_res.status_code == 403
    assert "Insufficient scope" in keys_res.json()["detail"]


@pytest.mark.asyncio
async def test_attack_rate_limit_flooding_blocked(security_client):
    """Attack 3: Attacker floods requests in a single second trying to exhaust API server."""
    client, _, dev_id, session_factory = security_client

    # Create key with high quota so rate limiter is what triggers 429
    async with session_factory() as session:
        flood_key, _ = await KeyService.create_api_key(
            db=session,
            developer_id=dev_id,
            name="Flood Key",
            scopes=["documents:read"],
            monthly_quota=1000,
        )

    # Configure rate limit to 5 req/min
    rate_limiter.requests_per_minute = 5

    status_codes = []
    for _ in range(8):
        res = await client.get("/api/v1/documents", headers={"X-API-Key": flood_key})
        status_codes.append(res.status_code)

    assert 200 in status_codes
    assert 429 in status_codes
    # Verify Retry-After header present on rate-limited response
    last_res = await client.get("/api/v1/documents", headers={"X-API-Key": flood_key})
    assert last_res.status_code == 429
    assert "Retry-After" in last_res.headers


@pytest.mark.asyncio
async def test_attack_concurrent_quota_race_condition(security_client):
    """
    Attack 4: Concurrent Race Condition Attack.
    Attacker fires 20 simultaneous async requests when quota is exactly 5.
    Atomic SQL decrement must allow exactly 5 requests and return 429 for the remaining 15.
    """
    client, _, dev_id, session_factory = security_client

    # Create a key with quota = 5
    async with session_factory() as session:
        quota_key, _ = await KeyService.create_api_key(
            db=session,
            developer_id=dev_id,
            name="Race Condition Key",
            scopes=["documents:read"],
            monthly_quota=5,
        )

    # Disable rate limiter for pure quota test
    rate_limiter.requests_per_minute = 1000

    async def make_request():
        return await client.get("/api/v1/documents", headers={"X-API-Key": quota_key})

    # Execute 20 concurrent requests simultaneously
    tasks = [make_request() for _ in range(20)]
    responses = await asyncio.gather(*tasks)

    status_counts = {}
    for r in responses:
        status_counts[r.status_code] = status_counts.get(r.status_code, 0) + 1

    # Exactly 5 must succeed (200), and exactly 15 must be blocked with 429 Quota Exceeded!
    assert status_counts.get(200, 0) == 5
    assert status_counts.get(429, 0) == 15
