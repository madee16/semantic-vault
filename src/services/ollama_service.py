"""
Ollama Local AI Provider layer.
Handles local embedding generation, intelligent query rewriting, and structured document enrichment.
Includes dimension validation and graceful fallback handling when models or services are offline.
"""
import json
import re
from typing import Any, Dict, List, Optional
import httpx

from src.core.config import get_settings
from src.core.logging import logger
from src.schemas.enrichment import DocumentEnrichmentResult, QueryRewriteResult


class EmbeddingDimensionError(Exception):
    """Raised when generated embeddings do not match the expected vector index dimension."""
    pass


class OllamaProvider:
    """
    Client for interacting with local Ollama instance.
    Interfaces with /api/embed for embeddings and /api/chat for generative tasks.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        expected_dim: Optional[int] = None,
        chat_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.expected_dim = expected_dim or settings.EMBEDDING_DIM
        self.chat_model = chat_model or settings.CHAT_MODEL
        self.timeout = timeout_seconds or settings.OLLAMA_TIMEOUT_SECONDS

    def validate_dimension(self, dim: int) -> bool:
        """Return True if dimension matches expected index dimension."""
        return dim == self.expected_dim

    def assert_dimension(self, dim: int) -> None:
        """Raise EmbeddingDimensionError if dimension does not match expected dimension."""
        if not self.validate_dimension(dim):
            raise EmbeddingDimensionError(
                f"Embedding dimension mismatch: provided {dim}, but index requires {self.expected_dim}."
            )

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of text chunks using Ollama's /api/embed endpoint.
        Strictly enforces that returned vector dimensions match the configured index dimension.
        """
        if not texts:
            return []

        payload = {
            "model": self.embedding_model,
            "input": texts,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.warning(
                f"Ollama /api/embed request failed ({e}). Attempting local fallback if configured..."
            )
            return await self._generate_fallback_embeddings(texts)

        raw_embeddings = data.get("embeddings", [])
        if not raw_embeddings:
            raise RuntimeError(f"Ollama returned empty embeddings response for model '{self.embedding_model}'.")

        # Dimension validation check on all returned vectors
        for i, vector in enumerate(raw_embeddings):
            dim = len(vector)
            if dim != self.expected_dim:
                raise EmbeddingDimensionError(
                    f"Model '{self.embedding_model}' returned {dim} dimensions, but expected {self.expected_dim}."
                )

        return raw_embeddings

    async def _generate_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fallback embedding generator using local SentenceTransformers.
        Used when the Ollama daemon is offline or during offline testing.
        """
        try:
            from sentence_transformers import SentenceTransformer
            # Use small fast model for fallback
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts, convert_to_numpy=True).tolist()
            # If the fallback dimension differs from expected_dim, raise dimension error
            for vec in embeddings:
                if len(vec) != self.expected_dim:
                    raise EmbeddingDimensionError(
                        f"Fallback model returned {len(vec)} dimensions, but index requires {self.expected_dim}."
                    )
            return embeddings
        except Exception as fallback_err:
            logger.error(f"Fallback embedding generation failed: {fallback_err}")
            raise RuntimeError(f"Failed to generate embeddings: {fallback_err}")

    async def rewrite_query(self, query: str) -> QueryRewriteResult:
        """
        Use local chat LLM to rewrite and expand vague user queries for semantic vector search.
        Gracefully preserves original query on failure.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                was_rewritten=False,
            )

        prompt = (
            "You are a search query optimizer. Given the user's raw search query, rewrite it into a "
            "clear, specific, and semantically rich query that will retrieve the most relevant documentation. "
            "Do not answer the query. Return ONLY the rewritten query text, with no preamble or explanation.\n\n"
            f"User Query: {cleaned_query}"
        )

        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()

                if content and len(content) > 3 and not content.startswith("{"):
                    # Remove any surrounding quotes
                    content = content.strip('"\'')
                    return QueryRewriteResult(
                        original_query=cleaned_query,
                        rewritten_query=content,
                        was_rewritten=True,
                    )
        except Exception as e:
            logger.warning(f"Query rewriting via Ollama failed ({e}). Falling back to original query.")

        return QueryRewriteResult(
            original_query=cleaned_query,
            rewritten_query=cleaned_query,
            was_rewritten=False,
        )

    async def enrich_document(self, text: str) -> DocumentEnrichmentResult:
        """
        Extract structured metadata from text content using Ollama with JSON output enforcement.
        Falls back to safe heuristics if Ollama is unreachable or outputs malformed data.
        """
        # Truncate text sample for enrichment to keep prompt latency fast
        sample_text = text[:3000].strip()

        system_instruction = (
            "You are a technical document analyzer. Analyze the provided text and output a JSON object "
            "with the following exact keys:\n"
            "- summary: A concise 2-3 sentence overview.\n"
            "- tags: Array of 3-6 relevant lowercase tags.\n"
            "- categories: Array of 1-3 high-level categories.\n"
            "- main_topics: Array of 2-5 technical topics.\n"
            "- keywords: Array of 5-8 specific keywords.\n"
            "- language: ISO 639-1 two-letter language code (e.g. 'en').\n"
            "Return valid JSON only."
        )

        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": sample_text},
            ],
            "format": "json",
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                raw_content = data.get("message", {}).get("content", "").strip()

                # Extract JSON if enclosed in markdown backticks
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if json_match:
                    parsed_dict = json.loads(json_match.group(0))
                    return DocumentEnrichmentResult.model_validate(parsed_dict)
        except Exception as e:
            logger.warning(f"Document enrichment via Ollama failed ({e}). Using basic heuristic fallback.")

        # Heuristic fallback
        words = re.findall(r"\b[A-Za-z]{4,}\b", sample_text)
        unique_words = list(dict.fromkeys(w.lower() for w in words))[:6]
        first_sentence = sample_text.split(". ")[0] if sample_text else "Document content."

        return DocumentEnrichmentResult(
            summary=first_sentence[:200] + ".",
            tags=unique_words[:4],
            categories=["General"],
            main_topics=["Documentation"],
            keywords=unique_words,
            language="en",
        )
