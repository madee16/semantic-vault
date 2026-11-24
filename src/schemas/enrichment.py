"""
Pydantic schemas for structured document enrichment and query rewriting outputs.
"""
from typing import List
from pydantic import BaseModel, Field


class DocumentEnrichmentResult(BaseModel):
    """
    Validated structured enrichment metadata generated from document content.
    Enforces clean JSON schemas for LLM extraction.
    """
    summary: str = Field(
        ...,
        description="Concise 2-4 sentence summary of the document's core content.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Descriptive tags for categorization and search filtering.",
    )
    categories: List[str] = Field(
        default_factory=list,
        description="High-level domain categories (e.g. Engineering, Finance, Legal, AI).",
    )
    main_topics: List[str] = Field(
        default_factory=list,
        description="Key subject matters and technical themes explored in the text.",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Prominent technical keywords and domain terms ranked by relevance.",
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code of the text (e.g., en, es, fr, de).",
    )


class QueryRewriteResult(BaseModel):
    """
    Result of intelligent query expansion/reformulation.
    Preserves original query for hybrid retrieval fallback.
    """
    original_query: str = Field(
        ...,
        description="The raw, unedited user query.",
    )
    rewritten_query: str = Field(
        ...,
        description="The clarified, context-rich reformulation optimized for semantic search.",
    )
    was_rewritten: bool = Field(
        default=False,
        description="Whether a successful rewriting was applied or fallback was used.",
    )
