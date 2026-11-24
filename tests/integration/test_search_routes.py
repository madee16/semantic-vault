"""
Integration tests for Search API endpoint (/api/v1/search).
"""
import io
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.main import app
from src.db.session import get_db
from src.models.base import Base
from src.schemas.enrichment import QueryRewriteResult
from src.services.ollama_service import OllamaProvider


@pytest.fixture
async def test_search_client():
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
        # Issue an API key for testing
        key_res = await client.post(
            "/api/v1/keys",
            json={
                "email": "search_tester@example.com",
                "name": "Search Tester",
                "scopes": ["documents:read", "documents:write", "search:read"],
                "monthly_quota": 500,
            },
        )
        raw_key = key_res.json()["raw_key"]
        client.headers["X-API-Key"] = raw_key

        # Ingest a sample document to search over
        with patch.object(OllamaProvider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            files = {"file": ("manual.txt", io.BytesIO(b"PostgreSQL pgvector allows fast vector similarity."), "text/plain")}
            await client.post("/api/v1/documents", files=files)

        yield client

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_search_endpoint_success(test_search_client: AsyncClient):
    """Test semantic search endpoint returns matching results and metrics."""
    with patch.object(OllamaProvider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[0.1] * 768]

        search_payload = {
            "query": "vector similarity postgres",
            "top_k": 3,
            "threshold": 0.0,
        }
        res = await test_search_client.post("/api/v1/search", json=search_payload)
        assert res.status_code == 200
        data = res.json()

        assert data["query"] == "vector similarity postgres"
        assert data["total_results"] >= 1
        assert "latency_ms" in data
        assert len(data["results"]) >= 1
        assert "similarity_score" in data["results"][0]


@pytest.mark.asyncio
async def test_search_with_query_rewriting(test_search_client: AsyncClient):
    """Test search endpoint with rewrite_query=True invokes Ollama rewriting."""
    rewrite_obj = QueryRewriteResult(
        original_query="postgres",
        rewritten_query="PostgreSQL relational database with pgvector extension",
        was_rewritten=True,
    )

    with patch.object(OllamaProvider, "rewrite_query", new_callable=AsyncMock) as mock_rewrite, \
         patch.object(OllamaProvider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_rewrite.return_value = rewrite_obj
        mock_embed.return_value = [[0.1] * 768]

        search_payload = {
            "query": "postgres",
            "rewrite_query": True,
            "top_k": 3,
        }
        res = await test_search_client.post("/api/v1/search", json=search_payload)
        assert res.status_code == 200
        data = res.json()

        assert data["was_rewritten"] is True
        assert data["rewritten_query"] == "PostgreSQL relational database with pgvector extension"


@pytest.mark.asyncio
async def test_search_without_api_key_rejected(test_search_client: AsyncClient):
    """Test search endpoint without API key header returns HTTP 401 Unauthorized."""
    test_search_client.headers.pop("X-API-Key", None)
    res = await test_search_client.post("/api/v1/search", json={"query": "test"})
    assert res.status_code == 401
