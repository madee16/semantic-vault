"""
Semantic and Hybrid Search service with pgvector integration, cosine similarity,
metadata filtering, Reciprocal Rank Fusion (RRF), and controlled vault re-indexing.
"""
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import logger
from src.models.document import Document, DocumentChunk
from src.models.embedding_index import EmbeddingIndex, IndexStatus
from src.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from src.services.chunking_service import ChunkingService
from src.services.ollama_service import OllamaProvider


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute normalized cosine similarity between two float vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw_similarity = dot_product / (norm_a * norm_b)
    # Clamp between 0.0 and 1.0 (treating negative correlation as 0.0 for text search relevance)
    return max(0.0, min(1.0, (raw_similarity + 1.0) / 2.0 if raw_similarity < 0 else raw_similarity))


class SearchService:
    """Service executing semantic vector search, hybrid retrieval, and index migrations."""

    @classmethod
    async def search(
        cls,
        db: AsyncSession,
        developer_id: str,
        request: SearchRequest,
        ollama_provider: Optional[OllamaProvider] = None,
    ) -> SearchResponse:
        """
        Execute semantic or hybrid vector retrieval for a developer's indexed documents.
        """
        start_time = time.perf_counter()
        provider = ollama_provider or OllamaProvider()

        original_query = request.query.strip()
        effective_query = original_query
        was_rewritten = False
        rewritten_text: Optional[str] = None

        # 1. Optional Intelligent Query Rewriting via Ollama LLM
        if request.rewrite_query:
            logger.info(f"Executing query rewriting for: '{original_query}'")
            rewrite_result = await provider.rewrite_query(original_query)
            if rewrite_result.was_rewritten:
                effective_query = rewrite_result.rewritten_query
                rewritten_text = rewrite_result.rewritten_query
                was_rewritten = True
                logger.info(f"Query rewritten to: '{effective_query}'")

        # 2. Generate Query Vector Embedding
        query_vectors = await provider.generate_embeddings([effective_query])
        if not query_vectors:
            raise RuntimeError("Failed to generate query embedding vector.")
        query_vector = query_vectors[0]

        # 3. Retrieve Candidate Document Chunks for Developer
        # (Select chunks belonging to indexed documents owned by this developer)
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.developer_id == developer_id)
            .options(selectinload(DocumentChunk.document))
        )

        result = await db.execute(stmt)
        rows = result.all()

        if not rows:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                query=effective_query,
                original_query=original_query,
                rewritten_query=rewritten_text,
                was_rewritten=was_rewritten,
                total_results=0,
                latency_ms=round(latency_ms, 2),
                results=[],
            )

        # 4. Filter by Metadata if Specified
        filtered_candidates: List[Tuple[DocumentChunk, Document]] = []
        for chunk, doc in rows:
            if cls._matches_metadata(doc, chunk, request.metadata_filter):
                filtered_candidates.append((chunk, doc))

        # 5. Score Candidates
        if request.hybrid:
            ranked_items = cls._rank_hybrid(
                candidates=filtered_candidates,
                query_vector=query_vector,
                raw_query=original_query,
                alpha=request.alpha,
                threshold=request.threshold,
                top_k=request.top_k,
            )
        else:
            ranked_items = cls._rank_semantic(
                candidates=filtered_candidates,
                query_vector=query_vector,
                threshold=request.threshold,
                top_k=request.top_k,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResponse(
            query=effective_query,
            original_query=original_query,
            rewritten_query=rewritten_text,
            was_rewritten=was_rewritten,
            total_results=len(ranked_items),
            latency_ms=round(latency_ms, 2),
            results=ranked_items,
        )

    @staticmethod
    def _matches_metadata(
        doc: Document,
        chunk: DocumentChunk,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if document or chunk matches all criteria in metadata_filter."""
        if not metadata_filter:
            return True

        combined_metadata = {**doc.metadata_json, **chunk.chunk_metadata}
        for key, expected_value in metadata_filter.items():
            actual_value = combined_metadata.get(key)
            if actual_value != expected_value:
                return False
        return True

    @staticmethod
    def _rank_semantic(
        candidates: List[Tuple[DocumentChunk, Document]],
        query_vector: List[float],
        threshold: float,
        top_k: int,
    ) -> List[SearchResultItem]:
        """Rank chunks by pure cosine vector similarity."""
        scored: List[Tuple[float, DocumentChunk, Document]] = []

        for chunk, doc in candidates:
            score = compute_cosine_similarity(query_vector, chunk.embedding)
            if score >= threshold:
                scored.append((score, chunk, doc))

        # Sort descending by similarity score
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResultItem] = []
        for score, chunk, doc in scored[:top_k]:
            results.append(
                SearchResultItem(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    document_filename=doc.filename,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    similarity_score=round(score, 4),
                    metadata={**doc.metadata_json, **chunk.chunk_metadata},
                )
            )
        return results

    @staticmethod
    def _rank_hybrid(
        candidates: List[Tuple[DocumentChunk, Document]],
        query_vector: List[float],
        raw_query: str,
        alpha: float,
        threshold: float,
        top_k: int,
    ) -> List[SearchResultItem]:
        """
        Rank chunks using Reciprocal Rank Fusion (RRF) combining dense vector similarity
        with sparse keyword frequency matching.
        """
        if not candidates:
            return []

        # 1. Dense vector ranking
        vector_scores = [
            (compute_cosine_similarity(query_vector, chunk.embedding), chunk, doc)
            for chunk, doc in candidates
        ]
        vector_scores.sort(key=lambda x: x[0], reverse=True)
        vector_ranks = {chunk.id: rank for rank, (_, chunk, _) in enumerate(vector_scores, start=1)}

        # 2. Sparse keyword ranking (Term frequency match over words in raw_query)
        query_tokens = [w.lower() for w in raw_query.split() if len(w) > 2]
        keyword_scores: List[Tuple[float, DocumentChunk, Document]] = []
        for chunk, doc in candidates:
            content_lower = chunk.content.lower()
            matches = sum(content_lower.count(token) for token in query_tokens)
            keyword_scores.append((float(matches), chunk, doc))

        keyword_scores.sort(key=lambda x: x[0], reverse=True)
        keyword_ranks = {chunk.id: rank for rank, (_, chunk, _) in enumerate(keyword_scores, start=1)}

        # 3. Reciprocal Rank Fusion (k = 60 constant)
        rrf_constant = 60.0
        combined: List[Tuple[float, float, DocumentChunk, Document]] = []

        for chunk, doc in candidates:
            v_rank = vector_ranks[chunk.id]
            k_rank = keyword_ranks[chunk.id]
            v_sim = next(s for s, c, _ in vector_scores if c.id == chunk.id)

            rrf_score = (alpha / (rrf_constant + v_rank)) + ((1.0 - alpha) / (rrf_constant + k_rank))

            # Apply threshold against dense vector similarity
            if v_sim >= threshold:
                combined.append((rrf_score, v_sim, chunk, doc))

        combined.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResultItem] = []
        for rrf_score, v_sim, chunk, doc in combined[:top_k]:
            results.append(
                SearchResultItem(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    document_filename=doc.filename,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    similarity_score=round(v_sim, 4),
                    metadata={
                        **doc.metadata_json,
                        **chunk.chunk_metadata,
                        "rrf_score": round(rrf_score, 6),
                    },
                )
            )
        return results

    @classmethod
    async def reindex_vault(
        cls,
        db: AsyncSession,
        developer_id: str,
        new_model_name: str,
        new_dimension: int,
        new_model_version: str = "v1",
        ollama_provider: Optional[OllamaProvider] = None,
    ) -> int:
        """
        Controlled reindexing routine for migrating a developer's documents to a new embedding model.
        Never mixes models in an index: creates new index, regenerates embeddings, updates references.
        Returns number of re-indexed documents.
        """
        provider = ollama_provider or OllamaProvider(
            embedding_model=new_model_name,
            expected_dim=new_dimension,
        )

        # 1. Resolve or create target index
        stmt = select(EmbeddingIndex).where(
            EmbeddingIndex.model_name == new_model_name,
            EmbeddingIndex.model_version == new_model_version,
            EmbeddingIndex.dimension == new_dimension,
        )
        res = await db.execute(stmt)
        target_index = res.scalar_one_or_none()

        if not target_index:
            target_index = EmbeddingIndex(
                model_name=new_model_name,
                model_version=new_model_version,
                dimension=new_dimension,
                distance_metric="cosine",
                status=IndexStatus.READY,
            )
            db.add(target_index)
            await db.flush()

        # 2. Retrieve all documents belonging to developer
        doc_stmt = (
            select(Document)
            .where(Document.developer_id == developer_id)
            .options(selectinload(Document.chunks))
        )
        doc_res = await db.execute(doc_stmt)
        documents = doc_res.scalars().all()

        reindexed_count = 0
        for doc in documents:
            if doc.index_id == target_index.id:
                continue  # Already up to date

            # Re-chunk raw content
            text_chunks = ChunkingService.chunk_text(doc.raw_content)
            chunk_texts = [c.content for c in text_chunks]
            new_embeddings = await provider.generate_embeddings(chunk_texts)

            # Delete old chunks
            doc.chunks.clear()
            doc.index_id = target_index.id

            # Add new chunks
            for chunk_meta, emb in zip(text_chunks, new_embeddings):
                doc.chunks.append(
                    DocumentChunk(
                        document_id=doc.id,
                        index_id=target_index.id,
                        chunk_index=chunk_meta.chunk_index,
                        content=chunk_meta.content,
                        embedding=emb,
                        token_count=chunk_meta.token_count,
                        chunk_metadata=chunk_meta.metadata,
                        expected_dimension=target_index.dimension,
                    )
                )
            reindexed_count += 1

        await db.commit()
        return reindexed_count
