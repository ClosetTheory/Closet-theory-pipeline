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


from pathlib import Path
from fastapi.responses import HTMLResponse, PlainTextResponse

STATIC_DIR = Path(__file__).parent / "static"
STATIC_INDEX = STATIC_DIR / "index.html"
STATIC_CATALOGUE = STATIC_DIR / "catalogue.html"
STATIC_GARMENT = STATIC_DIR / "garment.html"
STATIC_STYLING = STATIC_DIR / "styling.html"
STATIC_LOGIN = STATIC_DIR / "login.html"
STATIC_PROFILE = STATIC_DIR / "profile.html"
STATIC_REVIEW = STATIC_DIR / "review.html"
STATIC_PIPELINES = STATIC_DIR / "pipelines.html"
STATIC_ADMIN = STATIC_DIR / "admin.html"
STATIC_STYLIST = STATIC_DIR / "stylist.html"
# The engineering reference is authored as markdown and rendered client-side, so the
# page and docs/PIPELINES.md can never drift apart.
PIPELINES_DOC = Path(__file__).parent.parent / "docs" / "PIPELINES.md"


def _serve_static(path: Path, label: str) -> HTMLResponse:
    if path.exists():
        return HTMLResponse(content=path.read_text(encoding="utf-8"))
    return HTMLResponse(f"<h1>{label} not found</h1>", status_code=404)


@app.get("/login", response_class=HTMLResponse, tags=["Auth"])
async def get_login_page():
    """Sign in / register page."""
    return _serve_static(STATIC_LOGIN, "Login")


@app.get("/profile", response_class=HTMLResponse, tags=["Auth"])
async def get_profile_page():
    """Learned styling preference profile: boldness + per-attribute affinities."""
    return _serve_static(STATIC_PROFILE, "Profile")


@app.get("/", response_class=HTMLResponse, tags=["Catalogue"])
async def get_catalogue():
    """Wardrobe catalogue homepage: browse ingested garments and get styling recommendations."""
    return _serve_static(STATIC_CATALOGUE, "Catalogue")


@app.get("/garment", response_class=HTMLResponse, tags=["Catalogue"])
async def get_garment_page():
    """Single garment detail page."""
    return _serve_static(STATIC_GARMENT, "Garment page")


@app.get("/styling", response_class=HTMLResponse, tags=["Styling"])
async def get_styling_page():
    """Live, step-by-step Styling Pipeline detail view (all 10 stages)."""
    return _serve_static(STATIC_STYLING, "Styling pipeline")


@app.get("/visualizer", response_class=HTMLResponse, tags=["Visualizer"])
async def get_visualizer():
    """Visual interactive educational dashboard for explaining the Image Ingestion Pipeline."""
    return _serve_static(STATIC_INDEX, "Visualizer")


@app.get("/review", response_class=HTMLResponse, tags=["Styling"])
async def get_review_page():
    """Internal stylist QA panel: like/dislike + comment on this account's own generated
    outfits. Not part of the user-facing app — for company stylists reviewing dressing sense
    and overall aesthetics."""
    return _serve_static(STATIC_REVIEW, "Stylist review queue")


@app.get("/pipeline-info", response_class=HTMLResponse, tags=["Visualizer"])
async def get_pipeline_info_page():
    """Stage-by-stage engineering reference for both pipelines, with live Mermaid diagrams."""
    return _serve_static(STATIC_PIPELINES, "Pipeline info")


@app.get("/pipeline-info/source", response_class=PlainTextResponse, tags=["Visualizer"])
async def get_pipeline_info_source():
    """The raw markdown behind /pipeline-info, so the page renders the doc itself rather than
    a hand-copied duplicate of it."""
    if PIPELINES_DOC.exists():
        return PlainTextResponse(PIPELINES_DOC.read_text(encoding="utf-8"))
    return PlainTextResponse("Pipeline documentation not found.", status_code=404)


@app.get("/admin", response_class=HTMLResponse, tags=["Evaluation Characters"])
async def get_admin_page():
    """Character roster, stylist assignment and role management. The page itself is public HTML;
    every endpoint behind it requires the admin role, and the page shows a notice rather than
    content to anyone without it."""
    return _serve_static(STATIC_ADMIN, "Admin panel")


@app.get("/stylist", response_class=HTMLResponse, tags=["Evaluation Characters"])
async def get_stylist_page():
    """A stylist's assigned characters, and the scoring form for outfits generated for them."""
    return _serve_static(STATIC_STYLIST, "Stylist panel")


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
