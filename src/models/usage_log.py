"""
API Usage Log model for auditing developer requests, latency, and endpoint metrics.
"""
import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, utc_now

if TYPE_CHECKING:
    from src.models.developer import Developer
    from src.models.api_key import APIKey


class APIUsageLog(Base):
    """Audit log entry recorded for every API key request."""
    __tablename__ = "api_usage_logs"

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
    api_key_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    endpoint: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    method: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )
    status_code: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    response_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    # Relationships
    developer: Mapped["Developer"] = relationship("Developer", back_populates="usage_logs")
    api_key: Mapped[Optional["APIKey"]] = relationship("APIKey", back_populates="usage_logs")
