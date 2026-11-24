"""
Unit tests for DocumentService: document ingestion, index resolution, chunk persistence, and deduplication.
"""
from unittest.mock import AsyncMock, patch
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.models.base import Base
from src.models.developer import Developer
from src.models.embedding_index import EmbeddingIndex, IndexStatus
from src.schemas.enrichment import DocumentEnrichmentResult
from src.services.document_service import DocumentService
from src.services.ollama_service import OllamaProvider


@pytest.fixture
async def async_session():
    """In-memory SQLite async session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def test_developer(async_session: AsyncSession) -> Developer:
    dev = Developer(email="doc_tester@example.com", name="Doc Lead")
    async_session.add(dev)
    await async_session.commit()
    await async_session.refresh(dev)
    return dev


@pytest.mark.asyncio
async def test_ingest_document_success(async_session: AsyncSession, test_developer: Developer):
    """Test full document ingestion: parsing, chunking, embedding generation, and DB storage."""
    mock_provider = OllamaProvider(
        base_url="http://localhost:11434",
        embedding_model="embeddinggemma",
        expected_dim=768,
    )

    # Mock embeddings generation and enrichment
    mock_embeddings = [[0.01] * 768, [0.02] * 768]
    mock_enrichment = DocumentEnrichmentResult(
        summary="Architecture guide for semantic indexing.",
        tags=["search", "vector"],
        categories=["AI"],
        main_topics=["Vector Indexing"],
        keywords=["embedding", "similarity"],
        language="en",
    )

    with patch.object(mock_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch.object(mock_provider, "enrich_document", new_callable=AsyncMock) as mock_enrich:
        mock_embed.return_value = mock_embeddings
        mock_enrich.return_value = mock_enrichment

        sample_text = (
            "SemanticVault is an enterprise semantic search API. "
            "It indexes documents into pgvector embeddings and allows cosine retrieval. "
            "Developers can securely search chunks using hybrid reranking."
        ).encode("utf-8")

        doc = await DocumentService.ingest_document(
            db=async_session,
            developer_id=test_developer.id,
            file_bytes=sample_text,
            filename="architecture.txt",
            chunk_size=100,
            chunk_overlap=20,
            enrich=True,
            ollama_provider=mock_provider,
        )

        assert doc.id is not None
        assert doc.filename == "architecture.txt"
        assert doc.file_type == "txt"
        assert len(doc.chunks) >= 1
        assert doc.enrichment_json["summary"] == "Architecture guide for semantic indexing."
        assert len(doc.chunks[0].embedding) == 768


@pytest.mark.asyncio
async def test_ingest_duplicate_document(async_session: AsyncSession, test_developer: Developer):
    """Test that ingesting the exact same document for the same developer returns existing document."""
    mock_provider = OllamaProvider(expected_dim=768)

    with patch.object(mock_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch.object(mock_provider, "enrich_document", new_callable=AsyncMock) as mock_enrich:
        mock_embed.return_value = [[0.01] * 768]
        mock_enrich.return_value = DocumentEnrichmentResult(
            summary="A short note.", tags=[], categories=[], main_topics=[], keywords=[], language="en"
        )

        content = b"Exact identical text content for duplicate detection test."

        doc1 = await DocumentService.ingest_document(
            db=async_session,
            developer_id=test_developer.id,
            file_bytes=content,
            filename="doc1.txt",
            ollama_provider=mock_provider,
        )

        doc2 = await DocumentService.ingest_document(
            db=async_session,
            developer_id=test_developer.id,
            file_bytes=content,
            filename="doc1_duplicate.txt",
            ollama_provider=mock_provider,
        )

        assert doc1.id == doc2.id
        assert doc1.content_hash == doc2.content_hash
