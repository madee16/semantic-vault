"""
SemanticVault FastAPI Application Entrypoint.
Production-ready Semantic Search as a Service platform.
"""
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.router import api_router
from src.core.config import get_settings
from src.core.logging import logger
from src.db.session import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info(f"Starting {settings.PROJECT_NAME} in [{settings.ENVIRONMENT}] mode...")
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database init warning (may already be initialized or running in test): {e}")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


app = FastAPI(
    title="SemanticVault API",
    description=(
        "Semantic Search as a Service backend with custom developer API key management, "
        "PostgreSQL pgvector storage, and local Ollama AI integration."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    """Measure request latency and record audit logs."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({round(process_time, 2)}ms)")
    return response


# Include Versioned API Endpoints
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """Service landing page."""
    return {
        "service": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": "/api/v1/health",
        "version": "1.0.0",
    }
