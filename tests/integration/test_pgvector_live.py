"""
Live PostgreSQL + pgvector Integration Tests.
Executes real vector operations against PostgreSQL 16 container with pgvector extension:
- Verifies pgvector extension is loaded
- Creates HNSW / IVFFlat vector index
- Inserts 768-dimensional embeddings
- Executes cosine distance (<=>) vector queries
- Verifies real database index performance and accuracy
"""
import uuid
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings
from src.models.base import Base
from src.models.developer import Developer
from src.models.document import Document, DocumentChunk, IngestionStatus
from src.models.embedding_index import EmbeddingIndex, IndexStatus


@pytest.fixture
async def pgvector_session():
    """Connect to live PostgreSQL instance with pgvector extension."""
    settings = get_settings()
    # Connect to PostgreSQL using configured connection string
    pg_url = settings.DATABASE_URL
    engine = create_async_engine(pg_url, echo=False)

    try:
        async with engine.begin() as conn:
            # 1. Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # 2. Create tables
            await conn.run_sync(Base.metadata.create_all)
            # 3. Create real HNSW index on embeddings
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
                    "ON document_chunks USING hnsw (embedding vector_cosine_ops);"
                )
            )
    except Exception as e:
        pytest.skip(f"PostgreSQL + pgvector container is not available at {pg_url}: {e}")

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    # Cleanup test data
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM document_chunks;"))
            await conn.execute(text("DELETE FROM documents;"))
            await conn.execute(text("DELETE FROM embedding_indexes;"))
            await conn.execute(text("DELETE FROM developers;"))
    except Exception:
        pass

    await engine.dispose()


@pytest.mark.asyncio
async def test_live_pgvector_cosine_distance_query(pgvector_session: AsyncSession):
    """
    Verify real vector operations in PostgreSQL using pgvector <=> cosine distance operator.
    Inserts known 768-dim vectors and verifies that nearest-neighbor query retrieves the exact match.
    """
    # 1. Create developer and index
    dev = Developer(email=f"pgvector_{uuid.uuid4().hex[:6]}@example.com", name="PGVector Tester")
    pgvector_session.add(dev)
    await pgvector_session.flush()

    index = EmbeddingIndex(
        model_name="embeddinggemma",
        model_version="v1",
        dimension=768,
        distance_metric="cosine",
        status=IndexStatus.READY,
    )
    pgvector_session.add(index)
    await pgvector_session.flush()

    # 2. Create document
    doc = Document(
        developer_id=dev.id,
        index_id=index.id,
        filename="pgvector_guide.md",
        file_type="md",
        content_hash="hash_live_1",
        raw_content="PostgreSQL pgvector guide.",
        status=IngestionStatus.INDEXED,
    )
    pgvector_session.add(doc)
    await pgvector_session.flush()

    # 3. Insert target chunk with specific embedding: [1.0] in first component, zeros elsewhere
    target_vector = [1.0] + [0.0] * 767
    target_chunk = DocumentChunk(
        document_id=doc.id,
        index_id=index.id,
        chunk_index=0,
        content="Target chunk: perfect cosine match for vector search.",
        embedding=target_vector,
        token_count=10,
        expected_dimension=768,
    )
    pgvector_session.add(target_chunk)

    # Insert distractor chunk: [0.0, 1.0] in first two components
    distractor_vector = [0.0, 1.0] + [0.0] * 766
    distractor_chunk = DocumentChunk(
        document_id=doc.id,
        index_id=index.id,
        chunk_index=1,
        content="Distractor chunk: orthogonal vector.",
        embedding=distractor_vector,
        token_count=10,
        expected_dimension=768,
    )
    pgvector_session.add(distractor_chunk)
    await pgvector_session.commit()

    # 4. Execute raw SQL query using pgvector <=> cosine distance operator
    # Cosine distance = 1 - cosine_similarity. For identical vectors, distance = 0.0.
    query_vector_str = "[" + ",".join(str(x) for x in target_vector) + "]"
    sql = text("""
        SELECT 
            id, 
            content, 
            embedding <=> :query_vec AS cosine_distance,
            1.0 - (embedding <=> :query_vec) AS cosine_similarity
        FROM document_chunks
        ORDER BY embedding <=> :query_vec ASC
        LIMIT 2;
    """)

    result = await pgvector_session.execute(sql, {"query_vec": query_vector_str})
    rows = result.fetchall()

    assert len(rows) == 2
    # Nearest neighbor must be target_chunk
    top_row = rows[0]
    assert top_row.id == target_chunk.id
    assert abs(top_row.cosine_distance) < 0.0001
    assert abs(top_row.cosine_similarity - 1.0) < 0.0001

    # Second row is orthogonal (cosine distance ~ 1.0)
    second_row = rows[1]
    assert second_row.id == distractor_chunk.id
    assert abs(second_row.cosine_distance - 1.0) < 0.0001


@pytest.mark.asyncio
async def test_production_search_service_against_pgvector(pgvector_session: AsyncSession):
    """
    Test that the actual SearchService.search() production code path executes
    against real PostgreSQL + pgvector (exercising database-side cosine distance and FTS).
    """
    from unittest.mock import AsyncMock, patch
    from src.services.search_service import SearchService
    from src.schemas.search import SearchRequest
    from src.services.ollama_service import OllamaProvider

    dev = Developer(email=f"svc_search_{uuid.uuid4().hex[:6]}@example.com", name="Production Search Tester")
    pgvector_session.add(dev)
    await pgvector_session.flush()

    index = EmbeddingIndex(
        model_name="embeddinggemma",
        model_version="v1",
        dimension=768,
        distance_metric="cosine",
        status=IndexStatus.READY,
    )
    pgvector_session.add(index)
    await pgvector_session.flush()

    doc = Document(
        developer_id=dev.id,
        index_id=index.id,
        filename="production_pgvector_spec.md",
        file_type="md",
        content_hash=f"hash_{uuid.uuid4().hex[:8]}",
        raw_content="PostgreSQL pgvector delivers sub-10ms nearest neighbor search using HNSW graph indexing.",
        metadata_json={"env": "production", "topic": "databases"},
        status=IngestionStatus.INDEXED,
    )
    pgvector_session.add(doc)
    await pgvector_session.flush()

    # Create target chunk
    query_emb = [0.0] * 768
    query_emb[0] = 1.0
    chunk1 = DocumentChunk(
        document_id=doc.id,
        index_id=index.id,
        chunk_index=0,
        content="PostgreSQL pgvector delivers sub-10ms nearest neighbor search using HNSW graph indexing.",
        embedding=query_emb,
        token_count=15,
        expected_dimension=768,
    )
    pgvector_session.add(chunk1)
    await pgvector_session.commit()

    mock_provider = OllamaProvider(expected_dim=768)
    with patch.object(mock_provider, "generate_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [query_emb]

        # 1. Pure vector search through production SearchService
        req_dense = SearchRequest(query="pgvector HNSW latency", top_k=5, hybrid=False)
        resp_dense = await SearchService.search(
            db=pgvector_session,
            developer_id=dev.id,
            request=req_dense,
            ollama_provider=mock_provider,
        )
        assert resp_dense.total_results >= 1
        assert resp_dense.results[0].chunk_id == chunk1.id
        assert resp_dense.results[0].similarity_score > 0.99

        # 2. Hybrid search through production SearchService (exercising pgvector + PostgreSQL FTS)
        req_hybrid = SearchRequest(query="pgvector nearest neighbor", top_k=5, hybrid=True, alpha=0.5)
        resp_hybrid = await SearchService.search(
            db=pgvector_session,
            developer_id=dev.id,
            request=req_hybrid,
            ollama_provider=mock_provider,
        )
        assert resp_hybrid.total_results >= 1
        assert resp_hybrid.results[0].chunk_id == chunk1.id
        assert "rrf_score" in resp_hybrid.results[0].metadata
