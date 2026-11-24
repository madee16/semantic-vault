"""
Developer ORM model representing users/tenants possessing API keys and document vaults.
"""
import uuid
from typing import List, TYPE_CHECKING
from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, utc_now

if TYPE_CHECKING:
    from src.models.api_key import APIKey
    from src.models.document import Document
    from src.models.usage_log import APIUsageLog


class Developer(Base):
    """Developer account model."""
    __tablename__ = "developers"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at: Mapped[utc_now] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey",
        back_populates="developer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="developer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    usage_logs: Mapped[List["APIUsageLog"]] = relationship(
        "APIUsageLog",
        back_populates="developer",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
