"""
Health check route verifying API status, database connectivity, and Ollama inference availability.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.db.session import get_db

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", summary="System Health & Readiness Check")
async def health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Perform deep health check:
    1. Ping PostgreSQL database
    2. Probe local Ollama service endpoint
    """
    settings = get_settings()
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1;"))
    except Exception as e:
        db_status = f"unreachable: {e}"

    ollama_status = "connected"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if res.status_code != 200:
                ollama_status = f"degraded (HTTP {res.status_code})"
    except Exception:
        ollama_status = "offline"

    overall_status = "healthy" if db_status == "connected" and ollama_status == "connected" else "degraded"

    return {
        "status": overall_status,
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
        "components": {
            "database": db_status,
            "ollama_inference": ollama_status,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dim": settings.EMBEDDING_DIM,
            "chat_model": settings.CHAT_MODEL,
        },
    }
