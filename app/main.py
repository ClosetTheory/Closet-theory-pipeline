"""FastAPI application main entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.config import settings
from app.database import init_db
from app.observability import logger, metrics
from app.worker.queue import start_in_memory_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info(f"Starting Image Ingestion Pipeline v{settings.PIPELINE_VERSION} ({settings.ENV})...")
    # Initialize database tables and pgvector extension
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Start in-memory worker if configured
    if settings.USE_IN_MEMORY_QUEUE:
        start_in_memory_worker()
        logger.info("Asynchronous in-memory background worker initialized.")

    yield

    logger.info("Shutting down Image Ingestion Pipeline...")


app = FastAPI(
    title="Image Ingestion Pipeline",
    description="Backend image-ingestion and garment-intelligence pipeline producing CanonicalGarment representations.",
    version=settings.PIPELINE_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return {
        "status": "HEALTHY",
        "pipeline_version": settings.PIPELINE_VERSION,
        "environment": settings.ENV,
    }


@app.get("/metrics", status_code=status.HTTP_200_OK, tags=["Observability"])
async def get_metrics():
    """Returns in-memory latency and counter metrics."""
    return metrics.get_snapshot()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
