"""
Embedding Index ORM model for strict vector model versioning and dimension compatibility tracking.
Ensures incompatible models and dimensions can NEVER be mixed in the same vector collection.
"""
import enum
import uuid
from datetime import datetime
from typing import List, TYPE_CHECKING
from sqlalchemy import DateTime, Enum, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, utc_now

if TYPE_CHECKING:
    from src.models.document import Document, DocumentChunk


class IndexStatus(str, enum.Enum):
    """Lifecycle status of a vector index."""
    READY = "ready"
    REINDEXING = "reindexing"
    DEPRECATED = "deprecated"


class DistanceMetric(str, enum.Enum):
    """Supported distance metrics for vector similarity search."""
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class EmbeddingIndex(Base):
    """
    Registry for active vector embedding configurations.
    Enforces strict dimension and model consistency across documents and chunks.
    """
    __tablename__ = "embedding_indexes"
    __table_args__ = (
        UniqueConstraint("model_name", "model_version", "dimension", name="uq_model_version_dim"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Name of the embedding model (e.g., embeddinggemma, nomic-embed-text)",
    )
    model_version: Mapped[str] = mapped_column(
        String(20),
        default="v1",
        nullable=False,
        doc="Semantic or tag version of the embedding weights",
    )
    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Vector embedding dimension (e.g., 768, 384)",
    )
    distance_metric: Mapped[str] = mapped_column(
        String(20),
        default="cosine",
        nullable=False,
    )
    status: Mapped[IndexStatus] = mapped_column(
        Enum(IndexStatus),
        default=IndexStatus.READY,
        nullable=False,
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
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="index",
        lazy="selectin",
    )
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="index",
        lazy="selectin",
    )
