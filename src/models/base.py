"""
SQLAlchemy Declarative Base and custom hybrid types for SemanticVault.
Provides cross-database compatibility (PostgreSQL pgvector in production, JSON-serialized vectors in SQLite tests).
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Any, List
from sqlalchemy import DateTime, TypeDecorator, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class VectorField(TypeDecorator):
    """
    Custom vector column type decorator.
    Uses pgvector's Vector(dim) in PostgreSQL dialects,
    and falls back to JSON-serialized floats in SQLite.
    """
    impl = Text
    cache_ok = True
    comparator_factory = Vector.comparator_factory

    def __init__(self, dimension: int = 768, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.dimension = dimension

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dimension))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        # SQLite storage: serialize list of floats to JSON
        if isinstance(value, (list, tuple)):
            return json.dumps([float(x) for x in value])
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        # SQLite result: deserialize JSON to list of floats
        if isinstance(value, str):
            return json.loads(value)
        return list(value)


def utc_now() -> datetime:
    """Generate timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)
