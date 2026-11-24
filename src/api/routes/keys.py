"""
API Key management routes: developer registration, key generation, key listing, and revocation.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import require_api_key
from src.db.session import get_db
from src.models.api_key import APIKey
from src.models.developer import Developer
from src.services.key_service import KeyService

router = APIRouter(prefix="/keys", tags=["API Keys"])


class CreateKeyRequest(BaseModel):
    """Payload for registering a developer and issuing a new API key."""
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$", description="Developer account email address.")
    name: str = Field(..., min_length=2, max_length=100, description="Developer or company name.")
    key_name: str = Field(default="Default API Key", max_length=100)
    scopes: List[str] = Field(
        default=["documents:read", "documents:write", "search:read"],
        description="Allowed permission scopes: documents:read, documents:write, search:read, admin:keys, admin:all.",
    )
    monthly_quota: int = Field(default=10000, ge=1, le=1000000)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class KeyCreatedResponse(BaseModel):
    """Response returned upon successful key generation. Displays raw_key exactly once."""
    key_id: str
    developer_id: str
    raw_key: str = Field(..., description="The unhashed API key secret. Save this safely; it cannot be retrieved again.")
    key_prefix: str
    name: str
    scopes: List[str]
    monthly_quota: int
    expires_at: Optional[datetime]
    created_at: datetime
    security_warning: str = "Store this raw_key securely. It is hashed before storage and cannot be recovered."


class KeyInfoResponse(BaseModel):
    """Public representation of an existing API key (without plaintext secret)."""
    id: str
    key_prefix: str
    name: str
    scopes: List[str]
    monthly_quota: int
    requests_used: int
    is_active: bool
    expires_at: Optional[datetime]
    revoked_at: Optional[datetime]
    created_at: datetime


@router.post("", response_model=KeyCreatedResponse, status_code=status.HTTP_201_CREATED, summary="Issue New Developer API Key")
async def create_api_key(
    payload: CreateKeyRequest,
    db: AsyncSession = Depends(get_db),
) -> KeyCreatedResponse:
    """
    Register a developer account (or resolve existing by email) and issue a cryptographically random API key.
    The secret is hashed with SHA-256 and salt before storage.
    """
    # 1. Resolve or create Developer
    stmt = select(Developer).where(Developer.email == payload.email)
    res = await db.execute(stmt)
    developer = res.scalar_one_or_none()

    if not developer:
        developer = Developer(email=payload.email, name=payload.name)
        db.add(developer)
        await db.commit()
        await db.refresh(developer)

    # 2. Issue secure API Key
    raw_key, api_key_model = await KeyService.create_api_key(
        db=db,
        developer_id=developer.id,
        name=payload.key_name,
        scopes=payload.scopes,
        monthly_quota=payload.monthly_quota,
        expires_in_days=payload.expires_in_days,
    )

    return KeyCreatedResponse(
        key_id=api_key_model.id,
        developer_id=developer.id,
        raw_key=raw_key,
        key_prefix=api_key_model.key_prefix,
        name=api_key_model.name,
        scopes=api_key_model.scopes,
        monthly_quota=api_key_model.monthly_quota,
        expires_at=api_key_model.expires_at,
        created_at=api_key_model.created_at,
    )


@router.get("", response_model=List[KeyInfoResponse], summary="List Developer API Keys")
async def list_api_keys(
    current_key: APIKey = Depends(require_api_key(required_scope="admin:keys")),
    db: AsyncSession = Depends(get_db),
) -> List[KeyInfoResponse]:
    """List all API keys belonging to the authenticated developer."""
    keys = await KeyService.list_keys_for_developer(db, current_key.developer_id)
    return [
        KeyInfoResponse(
            id=k.id,
            key_prefix=k.key_prefix,
            name=k.name,
            scopes=k.scopes,
            monthly_quota=k.monthly_quota,
            requests_used=k.requests_used,
            is_active=k.is_active,
            expires_at=k.expires_at,
            revoked_at=k.revoked_at,
            created_at=k.created_at,
        )
        for k in keys
    ]


@router.delete("/{key_id}", summary="Revoke API Key")
async def revoke_api_key(
    key_id: str,
    current_key: APIKey = Depends(require_api_key(required_scope="admin:keys")),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Immediately deactivate and revoke an API key."""
    revoked = await KeyService.revoke_api_key(
        db=db,
        developer_id=current_key.developer_id,
        key_id=key_id,
    )
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or already revoked.",
        )
    return {"message": "API key successfully revoked.", "key_id": key_id}
