"""
Unit tests for OllamaProvider: embeddings, query rewriting, enrichment, and fallback handling.
"""
from unittest.mock import AsyncMock, patch
import pytest
import httpx

from src.schemas.enrichment import DocumentEnrichmentResult, QueryRewriteResult
from src.services.ollama_service import OllamaProvider, EmbeddingDimensionError


@pytest.fixture
def ollama_provider():
    """Create an OllamaProvider instance for testing."""
    return OllamaProvider(
        base_url="http://localhost:11434",
        embedding_model="embeddinggemma",
        expected_dim=768,
        chat_model="llama3.2:1b",
    )


def test_dimension_validation(ollama_provider: OllamaProvider):
    """Verify that model dimension checks succeed for 768 and fail for wrong dimensions."""
    assert ollama_provider.validate_dimension(768) is True
    assert ollama_provider.validate_dimension(384) is False

    with pytest.raises(EmbeddingDimensionError, match="dimension mismatch"):
        ollama_provider.assert_dimension(384)


@pytest.mark.asyncio
async def test_generate_embeddings_success(ollama_provider: OllamaProvider):
    """Verify OllamaProvider calls /api/embed and returns verified vector embeddings."""
    mock_embeddings = [[0.05] * 768, [-0.02] * 768]
    mock_response = httpx.Response(
        status_code=200,
        json={"embeddings": mock_embeddings},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        vectors = await ollama_provider.generate_embeddings(["Hello world", "Semantic search test"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 768
        assert len(vectors[1]) == 768


@pytest.mark.asyncio
async def test_generate_embeddings_dimension_mismatch(ollama_provider: OllamaProvider):
    """Verify that if Ollama returns embeddings with unexpected dimension, it is rejected."""
    mock_embeddings = [[0.05] * 384]  # Unexpected 384 instead of 768!
    mock_response = httpx.Response(
        status_code=200,
        json={"embeddings": mock_embeddings},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(EmbeddingDimensionError, match="returned 384 dimensions, but expected 768"):
            await ollama_provider.generate_embeddings(["Test mismatch"])


@pytest.mark.asyncio
async def test_rewrite_query_success(ollama_provider: OllamaProvider):
    """Verify query rewriting expands vague query and preserves original."""
    raw_query = "refund policy"
    expanded_text = "What is the return and refund policy timeframe, requirements, and conditions?"

    mock_response = httpx.Response(
        status_code=200,
        json={"message": {"content": expanded_text}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        res = await ollama_provider.rewrite_query(raw_query)
        assert isinstance(res, QueryRewriteResult)
        assert res.original_query == raw_query
        assert res.rewritten_query == expanded_text
        assert res.was_rewritten is True


@pytest.mark.asyncio
async def test_rewrite_query_fallback_on_network_failure(ollama_provider: OllamaProvider):
    """Verify that when Ollama is unreachable, query rewriting falls back gracefully to raw query."""
    raw_query = "authentication middleware"

    with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("Connection refused")):
        res = await ollama_provider.rewrite_query(raw_query)
        assert isinstance(res, QueryRewriteResult)
        assert res.original_query == raw_query
        assert res.rewritten_query == raw_query
        assert res.was_rewritten is False


@pytest.mark.asyncio
async def test_enrich_document_success(ollama_provider: OllamaProvider):
    """Verify document enrichment parses structured JSON into DocumentEnrichmentResult."""
    sample_text = "PostgreSQL pgvector enables fast HNSW and IVFFlat vector indexing for RAG systems."
    mock_json = """
    {
        "summary": "Guide to pgvector and vector indexing.",
        "tags": ["postgres", "pgvector", "indexing"],
        "categories": ["Database", "AI Engineering"],
        "main_topics": ["Vector Search", "HNSW"],
        "keywords": ["pgvector", "HNSW", "PostgreSQL"],
        "language": "en"
    }
    """
    mock_response = httpx.Response(
        status_code=200,
        json={"message": {"content": mock_json}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        enrichment = await ollama_provider.enrich_document(sample_text)
        assert isinstance(enrichment, DocumentEnrichmentResult)
        assert "pgvector" in enrichment.tags
        assert enrichment.language == "en"
        assert len(enrichment.categories) == 2


@pytest.mark.asyncio
async def test_enrich_document_fallback_on_invalid_json(ollama_provider: OllamaProvider):
    """Verify that invalid/non-JSON LLM responses fall back to basic heuristic enrichment without crashing."""
    sample_text = "Simple text document content."
    mock_response = httpx.Response(
        status_code=200,
        json={"message": {"content": "Sorry, I cannot format this in JSON."}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        enrichment = await ollama_provider.enrich_document(sample_text)
        assert isinstance(enrichment, DocumentEnrichmentResult)
        assert enrichment.summary != ""
        assert enrichment.language == "en"
