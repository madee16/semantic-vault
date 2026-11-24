"""
API Key ORM model for client authentication, scope authorization, and quota tracking.
"""
import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, utc_now

if TYPE_CHECKING:
    from src.models.developer import Developer
    from src.models.usage_log import APIUsageLog


class APIKey(Base):
    """Developer API Key with cryptographically hashed secrets, scopes, and quotas."""
    __tablename__ = "api_keys"

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
    name: Mapped[str] = mapped_column(
        String(100),
        default="Default API Key",
        nullable=False,
    )
    key_prefix: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        index=True,
        doc="Public prefix for key identification (e.g., sv_live_a1b2c3d4)",
    )
    key_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        doc="Salted SHA-256 hash of the plain secret key. Plain text is NEVER stored.",
    )
    scopes: Mapped[List[str]] = mapped_column(
        JSON,
        default=lambda: ["documents:read", "documents:write", "search:read"],
        nullable=False,
    )
    monthly_quota: Mapped[int] = mapped_column(
        Integer,
        default=10000,
        nullable=False,
    )
    requests_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    current_period: Mapped[str] = mapped_column(
        String(7),
        default=lambda: utc_now().strftime("%Y-%m"),
        nullable=False,
        doc="Active monthly quota period in YYYY-MM format. Resets usage automatically on month rollover.",
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    developer: Mapped["Developer"] = relationship("Developer", back_populates="api_keys")
    usage_logs: Mapped[List["APIUsageLog"]] = relationship(
        "APIUsageLog",
        back_populates="api_key",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def is_valid(self) -> bool:
        """Check if key is active, not revoked, not expired, and within quota."""
        if not self.is_active or self.revoked_at is not None:
            return False
        if self.expires_at is not None and utc_now() > self.expires_at:
            return False
        if self.requests_used >= self.monthly_quota:
            return False
        return True

    def has_scope(self, required_scope: str) -> bool:
        """Verify if key contains the required scope or admin privilege."""
        if "admin:all" in self.scopes:
            return True
        return required_scope in self.scopes
