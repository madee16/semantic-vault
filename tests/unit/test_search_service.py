"""
Unit tests for SearchService: vector similarity, metadata filtering, thresholding, hybrid search, and controlled reindexing.
"""
from unittest.mock import AsyncMock, patch
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from src.models.base import Base
from src.models.developer import Developer
from src.models.document import Document, DocumentChunk, IngestionStatus
from src.models.embedding_index import EmbeddingIndex, IndexStatus
from src.schemas.search import SearchRequest
from src.services.ollama_service import OllamaProvider
from src.services.search_service import SearchService


@pytest.fixture
async def async_session():
    """In-memory SQLite async database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def seed_data(async_session: AsyncSession):
    """Seed test developer, index, and 3 documents with known embeddings."""
    dev = Developer(email="searcher@example.com", name="Search Engineer")
    async_session.add(dev)
    await async_session.flush()

    index = EmbeddingIndex(
        model_name="embeddinggemma",
        model_version="v1",
        dimension=4,  # Use small 4-dim for precise vector math verification
        status=IndexStatus.READY,
    )
    async_session.add(index)
    await async_session.flush()

    # Document 1: Highly aligned with [1.0, 0.0, 0.0, 0.0]
    doc1 = Document(
        developer_id=dev.id,
        index_id=index.id,
        filename="vector_db.txt",
        file_type="txt",
        content_hash="hash_doc1",
        raw_content="PostgreSQL pgvector documentation.",
        metadata_json={"category": "database", "verified": True},
        status=IngestionStatus.INDEXED,
    )
    async_session.add(doc1)
    await async_session.flush()

    chunk1 = DocumentChunk(
        document_id=doc1.id,
        index_id=index.id,
        chunk_index=0,
        content="PostgreSQL pgvector provides vector indexing.",
        embedding=[1.0, 0.0, 0.0, 0.0],
        token_count=10,
        chunk_metadata={"section": "intro"},
    )
    async_session.add(chunk1)

    # Document 2: Orthogonal vector [0.0, 1.0, 0.0, 0.0]
    doc2 = Document(
        developer_id=dev.id,
        index_id=index.id,
        filename="cooking.txt",
        file_type="txt",
        content_hash="hash_doc2",
        raw_content="Pasta and tomato sauce recipe.",
        metadata_json={"category": "recipes", "verified": False},
        status=IngestionStatus.INDEXED,
    )
    async_session.add(doc2)
    await async_session.flush()

    chunk2 = DocumentChunk(
        document_id=doc2.id,
        index_id=index.id,
        chunk_index=0,
        content="Boil water for pasta sauce recipe.",
        embedding=[0.0, 1.0, 0.0, 0.0],
        token_count=10,
        chunk_metadata={"section": "cooking"},
    )
    async_session.add(chunk2)

    await async_session.commit()
    return dev, index


@pytest.mark.asyncio
async def test_semantic_search_ranking(async_session: AsyncSession, seed_data):
    """Test pure vector search ranks doc1 first when query vector matches [1, 0, 0, 0]."""
    dev, index = seed_data

    mock_provider = OllamaProvider(expected_dim=4)
    # Query vector identical to chunk 1
    with patch.object(mock_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[1.0, 0.0, 0.0, 0.0]]

        req = SearchRequest(query="pgvector postgres", top_k=5)
        response = await SearchService.search(
            db=async_session,
            developer_id=dev.id,
            request=req,
            ollama_provider=mock_provider,
        )

        assert response.total_results == 2
        # First result must be chunk 1 with similarity ~ 1.0
        assert response.results[0].document_filename == "vector_db.txt"
        assert response.results[0].similarity_score > 0.95
        # Second result is orthogonal (similarity ~ 0.0)
        assert response.results[1].document_filename == "cooking.txt"
        assert response.results[1].similarity_score < 0.1


@pytest.mark.asyncio
async def test_search_with_similarity_threshold(async_session: AsyncSession, seed_data):
    """Test that similarity threshold filters out low-scoring chunks."""
    dev, index = seed_data

    mock_provider = OllamaProvider(expected_dim=4)
    with patch.object(mock_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[1.0, 0.0, 0.0, 0.0]]

        req = SearchRequest(query="pgvector postgres", top_k=5, threshold=0.5)
        response = await SearchService.search(
            db=async_session,
            developer_id=dev.id,
            request=req,
            ollama_provider=mock_provider,
        )

        # Only doc1 exceeds 0.5 threshold
        assert response.total_results == 1
        assert response.results[0].document_filename == "vector_db.txt"


@pytest.mark.asyncio
async def test_search_with_metadata_filter(async_session: AsyncSession, seed_data):
    """Test metadata filtering limits search to matching documents."""
    dev, index = seed_data

    mock_provider = OllamaProvider(expected_dim=4)
    with patch.object(mock_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[1.0, 0.0, 0.0, 0.0]]

        req = SearchRequest(
            query="food or database",
            top_k=5,
            metadata_filter={"category": "recipes"},
        )
        response = await SearchService.search(
            db=async_session,
            developer_id=dev.id,
            request=req,
            ollama_provider=mock_provider,
        )

        assert response.total_results == 1
        assert response.results[0].document_filename == "cooking.txt"


@pytest.mark.asyncio
async def test_hybrid_search_rrf(async_session: AsyncSession, seed_data):
    """Test hybrid search combining vector retrieval with keyword ranking."""
    dev, index = seed_data

    mock_provider = OllamaProvider(expected_dim=4)
    with patch.object(mock_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        # Slightly favor chunk 2 (cooking) in both vector and keywords
        mock_embed.return_value = [[0.3, 0.7, 0.0, 0.0]]

        req = SearchRequest(
            query="recipe sauce",
            hybrid=True,
            alpha=0.4,
            top_k=2,
        )
        response = await SearchService.search(
            db=async_session,
            developer_id=dev.id,
            request=req,
            ollama_provider=mock_provider,
        )

        assert response.total_results == 2
        # "cooking.txt" contains keywords "recipe" and "sauce", so it should rank first in hybrid search
        assert response.results[0].document_filename == "cooking.txt"


@pytest.mark.asyncio
async def test_controlled_vault_reindexing(async_session: AsyncSession, seed_data):
    """Test reindexing vault with a new model and dimension without mixing vector spaces."""
    dev, old_index = seed_data

    # Reindex from 4-dim to new 768-dim model
    new_provider = OllamaProvider(embedding_model="nomic-embed-text", expected_dim=768)

    with patch.object(new_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[0.05] * 768]

        reindexed_count = await SearchService.reindex_vault(
            db=async_session,
            developer_id=dev.id,
            new_model_name="nomic-embed-text",
            new_dimension=768,
            new_model_version="v1",
            ollama_provider=new_provider,
        )

        assert reindexed_count == 2

        # Verify docs are now bound to the new 768-dim index
        stmt = (
            select(Document)
            .where(Document.developer_id == dev.id)
            .options(selectinload(Document.index), selectinload(Document.chunks))
        )
        res = await async_session.execute(stmt)
        docs = res.scalars().all()

        for doc in docs:
            assert doc.index.dimension == 768
            assert doc.index.model_name == "nomic-embed-text"
            assert len(doc.chunks[0].embedding) == 768
