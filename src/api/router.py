"""
Centralized API router aggregating all version 1 endpoints.
"""
from fastapi import APIRouter

from src.api.routes import documents, health, keys, search

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(keys.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
