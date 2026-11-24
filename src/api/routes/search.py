"""
Semantic and Hybrid search endpoint routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import require_api_key
from src.db.session import get_db
from src.models.api_key import APIKey
from src.schemas.search import SearchRequest, SearchResponse
from src.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse, summary="Execute Semantic or Hybrid Vector Search")
async def execute_search(
    request: SearchRequest,
    current_key: APIKey = Depends(require_api_key(required_scope="search:read")),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Search indexed documents using natural language queries:
    - Pure Vector Search: Dense semantic similarity via cosine distance.
    - Hybrid Search: Combines dense vector similarity with sparse keyword matching using Reciprocal Rank Fusion (RRF).
    - Intelligent Query Rewriting: Optional clarification of vague queries using local LLM inference.
    - Filtering: Filters by document/chunk metadata and minimum similarity score threshold.
    """
    try:
        response = await SearchService.search(
            db=db,
            developer_id=current_key.developer_id,
            request=request,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search execution failed: {e}",
        )
