"""
Unit tests for SemanticVault configuration and environment settings.
"""
import pytest
from src.core.config import Settings, get_settings


def test_default_settings():
    """Verify default settings provide correct types, models, and dimensions."""
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/semanticvault",
        API_KEY_SALT="test-salt-secret-at-least-32-chars-long",
    )
    assert settings.PROJECT_NAME == "SemanticVault"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.EMBEDDING_MODEL == "embeddinggemma"
    assert settings.EMBEDDING_DIM == 768
    assert settings.DEFAULT_MONTHLY_QUOTA == 10000
    assert settings.RATE_LIMIT_PER_MINUTE == 60
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"


def test_embedding_model_dimension_validation():
    """Verify that dimension mismatches for known models raise a ValueError."""
    with pytest.raises(ValueError, match="Dimension mismatch"):
        Settings(
            DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/semanticvault",
            API_KEY_SALT="test-salt-secret-at-least-32-chars-long",
            EMBEDDING_MODEL="embeddinggemma",
            EMBEDDING_DIM=384,  # embeddinggemma must be 768
        )


def test_api_key_salt_minimum_length():
    """Verify that an insecure/short salt is rejected."""
    with pytest.raises(ValueError, match="API_KEY_SALT must be at least 32 characters"):
        Settings(
            DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/semanticvault",
            API_KEY_SALT="too-short",
        )
