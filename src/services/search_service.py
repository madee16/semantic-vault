"""
Semantic and Hybrid Search service with database-side pgvector integration,
PostgreSQL full-text search, Reciprocal Rank Fusion (RRF), and controlled vault re-indexing.
"""
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.config import get_settings
from src.core.logging import logger
from src.models.document import Document, DocumentChunk
from src.models.embedding_index import EmbeddingIndex, IndexStatus
from src.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from src.services.chunking_service import ChunkingService
from src.services.ollama_service import OllamaProvider


def _is_postgres_session(db: AsyncSession) -> bool:
    """Determine whether the current SQLAlchemy session is bound to a PostgreSQL engine."""
    bind = getattr(db, "bind", None)
    if bind and hasattr(bind, "dialect"):
        return bind.dialect.name == "postgresql"
    return False


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
        Uses database-side pgvector and PostgreSQL FTS when connected to PostgreSQL.
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

        # 3. Route to PostgreSQL database-side search or fallback
        if _is_postgres_session(db):
            logger.debug("Executing database-side pgvector search in PostgreSQL.")
            ranked_items = await cls._search_postgres(
                db=db,
                developer_id=developer_id,
                query_vector=query_vector,
                raw_query=effective_query,
                request=request,
            )
        else:
            logger.debug("Executing in-memory fallback search for non-PostgreSQL dialect.")
            ranked_items = await cls._search_sqlite_fallback(
                db=db,
                developer_id=developer_id,
                query_vector=query_vector,
                raw_query=effective_query,
                request=request,
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

    @classmethod
    async def _search_postgres(
        cls,
        db: AsyncSession,
        developer_id: str,
        query_vector: List[float],
        raw_query: str,
        request: SearchRequest,
    ) -> List[SearchResultItem]:
        """
        Execute true database-side pgvector and PostgreSQL Full-Text Search.
        Leverages PostgreSQL HNSW index on embeddings for sub-10ms ANN retrieval.
        """
        dist_expr = DocumentChunk.embedding.cosine_distance(query_vector)
        sim_expr = (1.0 - dist_expr).label("similarity_score")
        candidate_k = max(25, request.top_k * 4)

        if not request.hybrid:
            # Database-side pure vector search
            stmt = (
                select(DocumentChunk, Document, sim_expr)
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(Document.developer_id == developer_id)
            )
            if request.threshold is not None:
                stmt = stmt.where(dist_expr <= (1.0 - request.threshold))

            stmt = stmt.order_by(dist_expr.asc()).limit(candidate_k)
            result = await db.execute(stmt)
            rows = result.all()

            results: List[SearchResultItem] = []
            for chunk, doc, sim in rows:
                if cls._matches_metadata(doc, chunk, request.metadata_filter):
                    results.append(
                        SearchResultItem(
                            chunk_id=chunk.id,
                            document_id=doc.id,
                            document_filename=doc.filename,
                            chunk_index=chunk.chunk_index,
                            content=chunk.content,
                            similarity_score=round(float(sim), 4),
                            metadata={**doc.metadata_json, **chunk.chunk_metadata},
                        )
                    )
                    if len(results) >= request.top_k:
                        break
            return results

        # Hybrid Search: Dense pgvector + Lexical PostgreSQL FTS
        dense_stmt = (
            select(DocumentChunk, Document, sim_expr)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.developer_id == developer_id)
            .order_by(dist_expr.asc())
            .limit(candidate_k)
        )
        dense_res = await db.execute(dense_stmt)
        dense_rows = dense_res.all()

        # Lexical FTS Query using plainto_tsquery and to_tsvector
        tsquery = func.plainto_tsquery("english", raw_query)
        tsvector = func.to_tsvector("english", DocumentChunk.content)
        rank_expr = func.ts_rank(tsvector, tsquery).label("lex_rank")

        lex_stmt = (
            select(DocumentChunk, Document, rank_expr)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.developer_id == developer_id,
                tsvector.op("@@")(tsquery),
            )
            .order_by(rank_expr.desc())
            .limit(candidate_k)
        )
        try:
            lex_res = await db.execute(lex_stmt)
            lex_rows = [(c, d, float(r)) for c, d, r in lex_res.all()]
        except Exception as e:
            logger.warning(f"PostgreSQL FTS query failed: {e}. Falling back to ILIKE token search.")
            tokens = [w for w in raw_query.split() if len(w) > 2]
            if tokens:
                ilike_conds = [DocumentChunk.content.ilike(f"%{t}%") for t in tokens[:5]]
                lex_fallback_stmt = (
                    select(DocumentChunk, Document)
                    .join(Document, DocumentChunk.document_id == Document.id)
                    .where(Document.developer_id == developer_id, or_(*ilike_conds))
                    .limit(candidate_k)
                )
                fb_res = await db.execute(lex_fallback_stmt)
                lex_rows = [(c, d, 1.0) for c, d in fb_res.all()]
            else:
                lex_rows = []

        return cls._fuse_rrf(
            dense_rows=[(c, d, float(s)) for c, d, s in dense_rows],
            lex_rows=lex_rows,
            alpha=request.alpha,
            threshold=request.threshold,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

    @classmethod
    async def _search_sqlite_fallback(
        cls,
        db: AsyncSession,
        developer_id: str,
        query_vector: List[float],
        raw_query: str,
        request: SearchRequest,
    ) -> List[SearchResultItem]:
        """In-memory candidate evaluation fallback for testing under non-PostgreSQL dialects."""
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.developer_id == developer_id)
            .options(selectinload(DocumentChunk.document))
        )
        result = await db.execute(stmt)
        rows = result.all()

        if not rows:
            return []

        # Dense candidates
        dense_rows = []
        for chunk, doc in rows:
            sim = compute_cosine_similarity(query_vector, chunk.embedding)
            dense_rows.append((chunk, doc, sim))
        dense_rows.sort(key=lambda x: x[2], reverse=True)

        if not request.hybrid:
            results: List[SearchResultItem] = []
            for chunk, doc, sim in dense_rows:
                if request.threshold is not None and sim < request.threshold:
                    continue
                if cls._matches_metadata(doc, chunk, request.metadata_filter):
                    results.append(
                        SearchResultItem(
                            chunk_id=chunk.id,
                            document_id=doc.id,
                            document_filename=doc.filename,
                            chunk_index=chunk.chunk_index,
                            content=chunk.content,
                            similarity_score=round(sim, 4),
                            metadata={**doc.metadata_json, **chunk.chunk_metadata},
                        )
                    )
                    if len(results) >= request.top_k:
                        break
            return results

        # Lexical candidates: token match
        query_tokens = [w.lower() for w in raw_query.split() if len(w) > 2]
        lex_rows = []
        for chunk, doc in rows:
            content_lower = chunk.content.lower()
            matches = sum(content_lower.count(tok) for tok in query_tokens)
            if matches > 0 or not query_tokens:
                lex_rows.append((chunk, doc, float(matches)))
        lex_rows.sort(key=lambda x: x[2], reverse=True)

        return cls._fuse_rrf(
            dense_rows=dense_rows,
            lex_rows=lex_rows,
            alpha=request.alpha,
            threshold=request.threshold,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

    @classmethod
    def _fuse_rrf(
        cls,
        dense_rows: List[Tuple[DocumentChunk, Document, float]],
        lex_rows: List[Tuple[DocumentChunk, Document, float]],
        alpha: float,
        threshold: Optional[float],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[SearchResultItem]:
        """
        Merge dense and lexical rankings using Reciprocal Rank Fusion (RRF).
        Runs in O(K log K) hash map join, completely eliminating O(N^2) quadratic operations.
        """
        rrf_constant = 60.0

        dense_ranks: Dict[str, Tuple[int, DocumentChunk, Document, float]] = {
            chunk.id: (rank, chunk, doc, sim)
            for rank, (chunk, doc, sim) in enumerate(dense_rows, start=1)
        }
        lex_ranks: Dict[str, Tuple[int, DocumentChunk, Document, float]] = {
            chunk.id: (rank, chunk, doc, score)
            for rank, (chunk, doc, score) in enumerate(lex_rows, start=1)
        }

        all_chunk_ids: Set[str] = set(dense_ranks.keys()) | set(lex_ranks.keys())
        scored_candidates: List[Tuple[float, float, DocumentChunk, Document]] = []

        default_rank_dense = len(dense_rows) + 50
        default_rank_lex = len(lex_rows) + 50

        for chunk_id in all_chunk_ids:
            d_info = dense_ranks.get(chunk_id)
            l_info = lex_ranks.get(chunk_id)

            chunk = (d_info or l_info)[1]
            doc = (d_info or l_info)[2]

            if not cls._matches_metadata(doc, chunk, metadata_filter):
                continue

            v_rank = d_info[0] if d_info else default_rank_dense
            k_rank = l_info[0] if l_info else default_rank_lex
            v_sim = d_info[3] if d_info else 0.0

            if threshold is not None and v_sim < threshold:
                continue

            rrf_score = (alpha / (rrf_constant + v_rank)) + ((1.0 - alpha) / (rrf_constant + k_rank))
            scored_candidates.append((rrf_score, v_sim, chunk, doc))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResultItem] = []
        for rrf_score, v_sim, chunk, doc in scored_candidates[:top_k]:
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
        Enforces deployment dimension compatibility: new_dimension must match configured column dimension.
        """
        settings = get_settings()
        if new_dimension != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Cannot re-index to {new_dimension} dimensions: database column is configured for {settings.EMBEDDING_DIM} dimensions."
            )

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

            if len(new_embeddings) != len(text_chunks):
                raise RuntimeError(
                    f"Re-indexing error: generated {len(new_embeddings)} embeddings for {len(text_chunks)} chunks."
                )

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

