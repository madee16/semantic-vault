"""
Pydantic schemas for Document operations, responses, and chunk representations.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentUploadMetadata(BaseModel):
    """Optional developer-supplied metadata when uploading documents."""
    title: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunkResponse(BaseModel):
    """Schema representing an individual indexed chunk of a document."""
    id: str
    chunk_index: int
    content: str
    token_count: int
    chunk_metadata: Dict[str, Any]
    created_at: datetime


class DocumentResponse(BaseModel):
    """Schema representing an uploaded and indexed document summary."""
    id: str
    filename: str
    file_type: str
    status: str
    content_hash: str
    total_chunks: int = 0
    metadata_json: Dict[str, Any]
    enrichment_json: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    """Detailed response including all document chunks and text content."""
    raw_content: str
    chunks: List[DocumentChunkResponse] = Field(default_factory=list)
