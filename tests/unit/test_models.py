"""
Unit tests for SemanticVault SQLAlchemy models and embedding compatibility schema.
"""
import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from src.models.base import Base
from src.models.developer import Developer
from src.models.api_key import APIKey
from src.models.embedding_index import EmbeddingIndex, IndexStatus
from src.models.document import Document, DocumentChunk, IngestionStatus


@pytest.fixture
async def async_session():
    """Create an in-memory SQLite database session for unit testing models."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_developer_and_api_key_creation(async_session: AsyncSession):
    """Test creating a developer and associated API key with scopes and quota."""
    developer = Developer(
        email="developer@example.com",
        name="Test Engineer",
    )
    async_session.add(developer)
    await async_session.commit()
    await async_session.refresh(developer)

    assert developer.id is not None
    assert developer.is_active is True

    api_key = APIKey(
        developer_id=developer.id,
        key_prefix="sv_live_abcd1234",
        key_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        scopes=["documents:read", "documents:write", "search:read"],
        monthly_quota=5000,
        requests_used=0,
    )
    async_session.add(api_key)
    await async_session.commit()
    await async_session.refresh(api_key)

    assert api_key.id is not None
    assert "documents:write" in api_key.scopes
    assert api_key.is_active is True
    assert api_key.revoked_at is None


@pytest.mark.asyncio
async def test_embedding_index_and_document_compatibility(async_session: AsyncSession):
    """Test creating an embedding index and binding a document and chunk to it."""
    # 1. Create developer
    dev = Developer(email="rag@example.com", name="RAG User")
    async_session.add(dev)
    await async_session.flush()

    # 2. Create explicit embedding index
    index = EmbeddingIndex(
        model_name="embeddinggemma",
        model_version="v1",
        dimension=768,
        distance_metric="cosine",
        status=IndexStatus.READY,
    )
    async_session.add(index)
    await async_session.flush()

    assert index.id is not None
    assert index.dimension == 768

    # 3. Create document associated with that specific index
    doc = Document(
        developer_id=dev.id,
        index_id=index.id,
        filename="system_design.md",
        file_type="md",
        content_hash="abc123hash",
        raw_content="# System Architecture",
        metadata_json={"author": "Staff AI Engineer"},
        enrichment_json={"summary": "Overview of RAG system", "tags": ["rag", "ai"]},
        status=IngestionStatus.INDEXED,
    )
    async_session.add(doc)
    await async_session.flush()

    # 4. Create document chunk
    chunk = DocumentChunk(
        document_id=doc.id,
        index_id=index.id,
        chunk_index=0,
        content="# System Architecture\nThis document describes...",
        embedding=[0.01] * 768,
        token_count=12,
        chunk_metadata={"section": "intro"},
    )
    async_session.add(chunk)
    await async_session.commit()

    # 5. Verify relationship queries
    await async_session.refresh(doc)
    assert doc.index.model_name == "embeddinggemma"
    assert len(doc.chunks) == 1
    assert doc.chunks[0].token_count == 12


@pytest.mark.asyncio
async def test_chunk_dimension_mismatch_validation(async_session: AsyncSession):
    """Verify that chunks with dimension mismatch compared to index raise an explicit ValueError."""
    dev = Developer(email="mismatch@example.com", name="Mismatch Tester")
    async_session.add(dev)
    await async_session.flush()

    index = EmbeddingIndex(
        model_name="embeddinggemma",
        model_version="v1",
        dimension=768,
        status=IndexStatus.READY,
    )
    async_session.add(index)
    await async_session.flush()

    doc = Document(
        developer_id=dev.id,
        index_id=index.id,
        filename="test.txt",
        file_type="txt",
        content_hash="hash1",
        raw_content="test content",
        status=IngestionStatus.INDEXED,
    )
    async_session.add(doc)
    await async_session.flush()

    # Try creating chunk with 384 dimensions when index requires 768
    with pytest.raises(ValueError, match="Embedding dimension 384 does not match index dimension 768"):
        DocumentChunk(
            document_id=doc.id,
            index_id=index.id,
            chunk_index=0,
            content="test",
            embedding=[0.1] * 384,  # Mismatched dimension!
            token_count=1,
            expected_dimension=index.dimension,
        )
