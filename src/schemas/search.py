"""
Pydantic schemas for semantic and hybrid search queries, ranking filters, and search responses.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request specification."""
    query: str = Field(..., min_length=1, description="Natural language search query.")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum number of relevant chunks to return.")
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold (0.0 to 1.0) for returned chunks.",
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key-value filters matching document or chunk metadata.",
    )
    hybrid: bool = Field(
        default=False,
        description="Combine semantic vector retrieval with keyword search.",
    )
    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight of semantic vector score (alpha) versus keyword score (1 - alpha) in hybrid search.",
    )
    rewrite_query: bool = Field(
        default=False,
        description="Use local LLM to expand/clarify vague queries before embedding.",
    )


class SearchResultItem(BaseModel):
    """An individual matched document chunk with relevance score and metadata."""
    chunk_id: str
    document_id: str
    document_filename: str
    chunk_index: int
    content: str
    similarity_score: float = Field(
        ...,
        description="Relevance score normalized between 0.0 (unrelated) and 1.0 (identical).",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Structured response returned by the search engine."""
    query: str
    original_query: str
    rewritten_query: Optional[str] = None
    was_rewritten: bool = False
    total_results: int
    latency_ms: float
    results: List[SearchResultItem]
