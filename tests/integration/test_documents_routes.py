"""
Integration tests for Document ingestion and management endpoints (/api/v1/documents).
"""
import io
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.main import app
from src.db.session import get_db
from src.models.base import Base
from src.services.ollama_service import OllamaProvider


@pytest.fixture
async def test_client():
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
                "email": "doc_author@example.com",
                "name": "Doc Author",
                "scopes": ["documents:read", "documents:write", "search:read"],
                "monthly_quota": 500,
            },
        )
        raw_key = key_res.json()["raw_key"]
        client.headers["X-API-Key"] = raw_key
        yield client

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_document_lifecycle(test_client: AsyncClient):
    """Test full document upload, retrieval, and deletion lifecycle."""
    mock_embeddings = [[0.01] * 768, [0.02] * 768]

    with patch.object(OllamaProvider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = mock_embeddings

        # 1. Upload a text document
        file_content = b"SemanticVault document upload integration test content. Chunk 1 and Chunk 2."
        files = {"file": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}
        data = {
            "chunk_size": "50",
            "chunk_overlap": "10",
            "metadata": '{"department": "AI", "project": "SemanticVault"}',
        }

        upload_res = await test_client.post("/api/v1/documents", files=files, data=data)
        assert upload_res.status_code == 201
        uploaded_doc = upload_res.json()
        doc_id = uploaded_doc["id"]
        assert uploaded_doc["filename"] == "test_doc.txt"
        assert uploaded_doc["metadata_json"]["department"] == "AI"
        assert uploaded_doc["total_chunks"] >= 1

        # 2. List documents
        list_res = await test_client.get("/api/v1/documents")
        assert list_res.status_code == 200
        docs = list_res.json()
        assert len(docs) == 1
        assert docs[0]["id"] == doc_id

        # 3. Retrieve document detail
        detail_res = await test_client.get(f"/api/v1/documents/{doc_id}")
        assert detail_res.status_code == 200
        detail = detail_res.json()
        assert "chunks" in detail
        assert len(detail["chunks"]) >= 1

        # 4. Delete document
        del_res = await test_client.delete(f"/api/v1/documents/{doc_id}")
        assert del_res.status_code == 200

        # 5. Confirm document no longer exists
        get_res = await test_client.get(f"/api/v1/documents/{doc_id}")
        assert get_res.status_code == 404
