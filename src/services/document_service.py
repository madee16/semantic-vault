"""
Document Service coordinating file parsing, chunking, enrichment, vector generation, and database indexing.
"""
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.config import get_settings
from src.core.logging import logger
from src.models.document import Document, DocumentChunk, IngestionStatus
from src.models.embedding_index import EmbeddingIndex, IndexStatus
from src.services.chunking_service import ChunkingService
from src.services.document_parser import DocumentParser
from src.services.ollama_service import OllamaProvider


class DocumentService:
    """Service handling document lifecycle from upload to vector indexing."""

    @classmethod
    async def get_or_create_index(
        cls,
        db: AsyncSession,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> EmbeddingIndex:
        """
        Retrieve active embedding index matching model specifications or create one if absent.
        Guarantees that index constraints remain strictly enforced.
        """
        settings = get_settings()
        target_model = model_name or settings.EMBEDDING_MODEL
        target_version = model_version or settings.EMBEDDING_MODEL_VERSION
        target_dim = dimension or settings.EMBEDDING_DIM

        stmt = select(EmbeddingIndex).where(
            EmbeddingIndex.model_name == target_model,
            EmbeddingIndex.model_version == target_version,
            EmbeddingIndex.dimension == target_dim,
        )
        result = await db.execute(stmt)
        index = result.scalar_one_or_none()

        if not index:
            index = EmbeddingIndex(
                model_name=target_model,
                model_version=target_version,
                dimension=target_dim,
                distance_metric="cosine",
                status=IndexStatus.READY,
            )
            db.add(index)
            await db.commit()
            await db.refresh(index)

        return index

    @classmethod
    async def ingest_document(
        cls,
        db: AsyncSession,
        developer_id: str,
        file_bytes: bytes,
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enrich: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        ollama_provider: Optional[OllamaProvider] = None,
    ) -> Document:
        """
        Ingest, parse, chunk, enrich, embed, and store a document.
        Detects duplicates via content hash for idempotent re-uploads.
        """
        settings = get_settings()
        c_size = chunk_size if chunk_size is not None else settings.DEFAULT_CHUNK_SIZE
        c_overlap = chunk_overlap if chunk_overlap is not None else settings.DEFAULT_CHUNK_OVERLAP
        provider = ollama_provider or OllamaProvider()

        # 1. Parse and normalize text
        normalized_text, file_type, content_hash = DocumentParser.parse(file_bytes, filename)

        # 2. Check for duplicate content for this developer
        existing_stmt = (
            select(Document)
            .where(
                Document.developer_id == developer_id,
                Document.content_hash == content_hash,
            )
            .options(selectinload(Document.chunks))
        )
        existing_result = await db.execute(existing_stmt)
        existing_doc = existing_result.scalar_one_or_none()
        if existing_doc:
            logger.info(f"Duplicate document detected ({content_hash}). Returning existing record.")
            return existing_doc

        # 3. Resolve active embedding index
        index = await cls.get_or_create_index(
            db=db,
            model_name=provider.embedding_model,
            dimension=provider.expected_dim,
        )

        # 4. Optional AI Enrichment via Ollama
        enrichment_dict: Dict[str, Any] = {}
        if enrich:
            logger.info(f"Enriching document '{filename}' via Ollama...")
            enrichment_result = await provider.enrich_document(normalized_text)
            enrichment_dict = enrichment_result.model_dump()

        # 5. Segment text into sliding window chunks
        text_chunks = ChunkingService.chunk_text(
            text=normalized_text,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
        )

        # 6. Generate embeddings for all chunks in batch
        chunk_texts = [c.content for c in text_chunks]
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks using {provider.embedding_model}...")
        embeddings = await provider.generate_embeddings(chunk_texts)

        # Validate that embedding generator returned exact vector count matching chunk count
        if len(embeddings) != len(text_chunks):
            raise RuntimeError(
                f"Embedding generation count mismatch: received {len(embeddings)} vectors for {len(text_chunks)} chunks."
            )

        # 7. Persist Document and DocumentChunks in database with concurrency-safe duplicate guard
        doc = Document(
            developer_id=developer_id,
            index_id=index.id,
            filename=filename,
            file_type=file_type,
            content_hash=content_hash,
            raw_content=normalized_text,
            metadata_json=metadata or {},
            enrichment_json=enrichment_dict,
            status=IngestionStatus.INDEXED,
        )
        db.add(doc)

        try:
            await db.flush()

            for chunk_meta, emb in zip(text_chunks, embeddings):
                chunk_row = DocumentChunk(
                    document_id=doc.id,
                    index_id=index.id,
                    chunk_index=chunk_meta.chunk_index,
                    content=chunk_meta.content,
                    embedding=emb,
                    token_count=chunk_meta.token_count,
                    chunk_metadata=chunk_meta.metadata,
                    expected_dimension=index.dimension,
                )
                db.add(chunk_row)

            await db.commit()
            await db.refresh(doc)
            return doc
        except IntegrityError:
            # Handle concurrent race condition when identical document is uploaded simultaneously
            await db.rollback()
            logger.info(f"Concurrent duplicate insertion caught for {content_hash}. Retrieving existing record.")
            dup_stmt = (
                select(Document)
                .where(
                    Document.developer_id == developer_id,
                    Document.content_hash == content_hash,
                )
                .options(selectinload(Document.chunks))
            )
            dup_res = await db.execute(dup_stmt)
            existing = dup_res.scalar_one_or_none()
            if existing:
                return existing
            raise

    @classmethod
    async def get_document(
        cls,
        db: AsyncSession,
        developer_id: str,
        document_id: str,
    ) -> Optional[Document]:
        """Retrieve a specific document with its chunks by ID."""
        stmt = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.developer_id == developer_id,
            )
            .options(selectinload(Document.chunks))
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @classmethod
    async def list_documents(
        cls,
        db: AsyncSession,
        developer_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Document]:
        """List documents owned by a developer with chunk count."""
        stmt = (
            select(Document)
            .where(Document.developer_id == developer_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
            .options(selectinload(Document.chunks))
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @classmethod
    async def delete_document(
        cls,
        db: AsyncSession,
        developer_id: str,
        document_id: str,
    ) -> bool:
        """Delete document and cascade-delete its chunks and vector embeddings."""
        doc = await cls.get_document(db, developer_id, document_id)
        if not doc:
            return False
        await db.delete(doc)
        await db.commit()
        return True
