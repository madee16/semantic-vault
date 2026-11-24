"""
Core configuration settings for SemanticVault.
Uses Pydantic BaseSettings with environment variable resolution and validation.
"""
from functools import lru_cache
from typing import ClassVar, Dict
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration and environment variable bindings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Service Information
    PROJECT_NAME: str = "SemanticVault"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgrespassword@localhost:5432/semanticvault",
        description="Async PostgreSQL connection string with pgvector extension",
    )
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # API Security & Quotas
    API_KEY_SALT: str = Field(
        default="semanticvault-default-dev-salt-replace-in-production-min-32-chars",
        description="Cryptographic salt used for API key SHA-256 hashing",
    )
    DEFAULT_MONTHLY_QUOTA: int = 10000
    RATE_LIMIT_PER_MINUTE: int = 60

    # Local AI / Ollama Inference Provider
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # Embedding Model & Compatibility Constraints
    EMBEDDING_MODEL: str = "embeddinggemma"
    EMBEDDING_MODEL_VERSION: str = "v1"
    EMBEDDING_DIM: int = 768

    # Generative LLM for Query Rewriting & Document Enrichment
    CHAT_MODEL: str = "llama3.2:1b"

    # Ingestion & Chunking
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    # Known embedding model dimension specifications
    KNOWN_MODEL_DIMENSIONS: ClassVar[Dict[str, int]] = {
        "embeddinggemma": 768,
        "nomic-embed-text": 768,
        "all-minilm-l6-v2": 384,
        "bge-small-en-v1.5": 384,
        "bge-base-en-v1.5": 768,
        "text-embedding-3-small": 1536,
    }

    @field_validator("API_KEY_SALT")
    @classmethod
    def validate_api_key_salt(cls, v: str) -> str:
        """Enforce minimum cryptographic key length for salt."""
        if len(v) < 32:
            raise ValueError("API_KEY_SALT must be at least 32 characters long for cryptographic security.")
        return v

    @model_validator(mode="after")
    def validate_embedding_dimensions(self) -> "Settings":
        """Verify that configured dimension strictly matches known model specifications."""
        model_name = self.EMBEDDING_MODEL.lower().strip()
        expected_dim = self.KNOWN_MODEL_DIMENSIONS.get(model_name)
        if expected_dim is not None and self.EMBEDDING_DIM != expected_dim:
            raise ValueError(
                f"Dimension mismatch for model '{self.EMBEDDING_MODEL}': "
                f"configured EMBEDDING_DIM={self.EMBEDDING_DIM}, but model requires {expected_dim}."
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
