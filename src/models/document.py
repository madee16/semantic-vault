"""
Document and DocumentChunk ORM models for document management and vector storage.
"""
import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, VectorField, utc_now

if TYPE_CHECKING:
    from src.models.developer import Developer
    from src.models.embedding_index import EmbeddingIndex


class IngestionStatus(str, enum.Enum):
    """Document processing and ingestion status."""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(Base):
    """Uploaded and parsed document root."""
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("developer_id", "content_hash", name="uq_dev_content_hash"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    developer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("developers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("embedding_indexes.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        doc="Strictly binds this document's chunks to a verified embedding index",
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    file_type: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        doc="Normalized file extension: pdf, txt, or md",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="SHA-256 hash of original normalized text content for deduplication",
    )
    raw_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Full extracted and normalized text",
    )
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="Arbitrary developer-supplied metadata tags for filtering",
    )
    enrichment_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="LLM-enriched metadata: summary, tags, categories, keywords, language",
    )
    status: Mapped[IngestionStatus] = mapped_column(
        Enum(IngestionStatus),
        default=IngestionStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Relationships
    developer: Mapped["Developer"] = relationship("Developer", back_populates="documents")
    index: Mapped["EmbeddingIndex"] = relationship("EmbeddingIndex", back_populates="documents")
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
        lazy="selectin",
    )


class DocumentChunk(Base):
    """Individual chunk of document content with its vector embedding."""
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("embedding_indexes.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    embedding: Mapped[List[float]] = mapped_column(
        VectorField(dimension=768),
        nullable=False,
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    index: Mapped["EmbeddingIndex"] = relationship("EmbeddingIndex", back_populates="chunks")

    def __init__(self, *args: Any, expected_dimension: Optional[int] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if expected_dimension is not None and self.embedding is not None:
            if len(self.embedding) != expected_dimension:
                raise ValueError(
                    f"Embedding dimension {len(self.embedding)} does not match index dimension {expected_dimension}"
                )
