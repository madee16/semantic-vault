"""
Document ingestion and management routes.
"""
import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import require_api_key
from src.db.session import get_db
from src.models.api_key import APIKey
from src.schemas.document import DocumentChunkResponse, DocumentDetailResponse, DocumentResponse
from src.services.document_parser import EmptyDocumentError, UnsupportedFileTypeError
from src.services.document_service import DocumentService
from src.services.ollama_service import OllamaProvider

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Upload & Ingest Document")
async def upload_document(
    file: UploadFile = File(..., description="Document file: PDF, Markdown, or TXT."),
    metadata: Optional[str] = Form(default=None, description="Optional JSON string of custom metadata tags."),
    chunk_size: Optional[int] = Form(default=None, description="Chunk size in characters (default: 500)."),
    chunk_overlap: Optional[int] = Form(default=None, description="Overlap in characters (default: 50)."),
    enrich: bool = Form(default=False, description="Use local LLM to generate summary, tags, and topics."),
    current_key: APIKey = Depends(require_api_key(required_scope="documents:write")),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """
    Ingest a document: parse text (PDF/MD/TXT), chunk into overlapping windows,
    generate vector embeddings via Ollama, and store in PostgreSQL vector vault.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    parsed_metadata: Dict[str, Any] = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid JSON string provided in 'metadata' field.",
            )

    try:
        doc = await DocumentService.ingest_document(
            db=db,
            developer_id=current_key.developer_id,
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_document.txt",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enrich=enrich,
            metadata=parsed_metadata,
        )
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EmptyDocumentError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {e}")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        status=doc.status.value,
        content_hash=doc.content_hash,
        total_chunks=len(doc.chunks),
        metadata_json=doc.metadata_json,
        enrichment_json=doc.enrichment_json,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("", response_model=List[DocumentResponse], summary="List Ingested Documents")
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    current_key: APIKey = Depends(require_api_key(required_scope="documents:read")),
    db: AsyncSession = Depends(get_db),
) -> List[DocumentResponse]:
    """List documents belonging to the authenticated developer."""
    docs = await DocumentService.list_documents(db, current_key.developer_id, limit=limit, offset=offset)
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            file_type=d.file_type,
            status=d.status.value,
            content_hash=d.content_hash,
            total_chunks=len(d.chunks),
            metadata_json=d.metadata_json,
            enrichment_json=d.enrichment_json,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentDetailResponse, summary="Get Document Details & Chunks")
async def get_document_detail(
    document_id: str,
    current_key: APIKey = Depends(require_api_key(required_scope="documents:read")),
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Retrieve full details of a specific document, including raw content and chunk segments."""
    doc = await DocumentService.get_document(db, current_key.developer_id, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    chunk_responses = [
        DocumentChunkResponse(
            id=c.id,
            chunk_index=c.chunk_index,
            content=c.content,
            token_count=c.token_count,
            chunk_metadata=c.chunk_metadata,
            created_at=c.created_at,
        )
        for c in doc.chunks
    ]

    return DocumentDetailResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        status=doc.status.value,
        content_hash=doc.content_hash,
        total_chunks=len(doc.chunks),
        metadata_json=doc.metadata_json,
        enrichment_json=doc.enrichment_json,
        raw_content=doc.raw_content,
        chunks=chunk_responses,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/{document_id}", summary="Delete Document & Vector Embeddings")
async def delete_document(
    document_id: str,
    current_key: APIKey = Depends(require_api_key(required_scope="documents:write")),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Delete a document and cascade-delete all its chunks and vector embeddings."""
    deleted = await DocumentService.delete_document(db, current_key.developer_id, document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return {"message": "Document and associated vector embeddings deleted successfully.", "document_id": document_id}
