"""
Database session management and connection pooling for SemanticVault.
Provides asynchronous database sessions and initializes pgvector extension on PostgreSQL.
"""
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import get_settings
from src.core.logging import logger
from src.models.base import Base

settings = get_settings()

# Engine creation with connection pooling settings
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    future=True,
    pool_pre_ping=True,
)

# Asynchronous session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding database session with automatic cleanup."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db(engine_instance: AsyncEngine = engine) -> None:
    """Initialize database tables and enable pgvector extension if PostgreSQL."""
    async with engine_instance.begin() as conn:
        dialect_name = engine_instance.dialect.name
        if dialect_name == "postgresql":
            logger.info("Enabling pgvector extension on PostgreSQL database...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        logger.info("Creating database tables if not present...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialization completed.")
