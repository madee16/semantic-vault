"""
Integration tests for API Key endpoints (/api/v1/keys).
"""
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.main import app
from src.db.session import get_db
from src.models.base import Base


@pytest.fixture
async def test_app():
    """Setup test FastAPI app with in-memory SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_health_endpoint(test_app: AsyncClient):
    """Test health check returns HTTP 200 and healthy status."""
    response = await test_app.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data


@pytest.mark.asyncio
async def test_create_developer_and_key(test_app: AsyncClient):
    """Test registering a developer and issuing a secure API key."""
    payload = {
        "email": "lead_engineer@example.com",
        "name": "Lead Engineer",
        "key_name": "Primary Ingestion Key",
        "scopes": ["documents:read", "documents:write", "search:read", "admin:keys"],
        "monthly_quota": 5000,
    }

    response = await test_app.post("/api/v1/keys", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert "raw_key" in data
    assert data["raw_key"].startswith("sv_live_")
    assert data["key_prefix"].startswith("sv_live_")
    assert data["monthly_quota"] == 5000
    assert "admin:keys" in data["scopes"]


@pytest.mark.asyncio
async def test_list_and_revoke_keys(test_app: AsyncClient):
    """Test listing developer keys and revoking an active key."""
    # 1. Create key
    create_res = await test_app.post(
        "/api/v1/keys",
        json={
            "email": "admin@example.com",
            "name": "Admin",
            "key_name": "Admin Key",
            "scopes": ["admin:all"],
            "monthly_quota": 1000,
        },
    )
    key_data = create_res.json()
    raw_key = key_data["raw_key"]
    key_id = key_data["key_id"]

    # 2. List keys using authentication header
    list_res = await test_app.get("/api/v1/keys", headers={"X-API-Key": raw_key})
    assert list_res.status_code == 200
    keys = list_res.json()
    assert len(keys) >= 1
    assert any(k["id"] == key_id for k in keys)

    # 3. Revoke the key
    del_res = await test_app.delete(f"/api/v1/keys/{key_id}", headers={"X-API-Key": raw_key})
    assert del_res.status_code == 200

    # 4. Attempting to use the revoked key must now return 401
    fail_res = await test_app.get("/api/v1/keys", headers={"X-API-Key": raw_key})
    assert fail_res.status_code == 401
