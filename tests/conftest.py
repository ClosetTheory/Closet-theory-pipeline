"""Pytest test fixtures and configuration."""

import io
from typing import AsyncGenerator
from PIL import Image
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.dependencies import get_db_session, get_storage
from app.config import settings
from app.database import get_db
from app.main import app
from app.models.base import Base
from app.storage.local_storage import LocalStorageClient

# Test SQLite async in-memory database
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def _reset_worker_queue_singletons():
    """app.worker.queue keeps its in-memory queue/worker-task as module-level singletons
    (by design, for lightweight standalone/local execution — see that module's docstring).
    pytest-asyncio gives each test its own event loop, so a singleton bound to one test's
    loop by an earlier enqueue_garment_pipeline() call would break (or hang) a later test
    that tries to reuse it. Reset them before every test so each gets a fresh queue/task
    bound to its own loop."""
    import app.worker.queue as queue_module

    if queue_module._worker_task is not None and not queue_module._worker_task.done():
        queue_module._worker_task.cancel()
    queue_module._worker_task = None
    queue_module._in_memory_queue = None
    yield


@pytest.fixture
def test_storage(tmp_path):
    """Provides an isolated LocalStorageClient rooted in a temp directory."""
    return LocalStorageClient(base_dir=str(tmp_path / "storage"), bucket_name="test-wardrobe")


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides a fresh database session with tables initialized."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session
        await session.rollback()

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def sample_catalog_image_bytes() -> bytes:
    """Creates a synthetic square catalog garment image (800x800)."""
    img = Image.new("RGB", (800, 800), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def sample_full_body_image_bytes() -> bytes:
    """Creates a synthetic full body portrait image (600x1200, ratio 2.0)."""
    img = Image.new("RGB", (600, 1200), color=(200, 220, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def valid_attributes_dict():
    """Returns a valid garment attributes dictionary."""
    return {
        "category": "shirt",
        "subcategory": "oxford_shirt",
        "colour": ["white", "light_blue"],
        "pattern": "solid",
        "material": "cotton",
        "fit": "regular",
        "silhouette": "straight",
        "sleeve_length": "long",
        "occasion": ["smart_casual", "work"],
        "season": ["summer", "spring"],
        "layering_role": "base",
        "warmth": 0.25,
        "versatility": 0.85,
        "confidence": 0.95,
    }


@pytest.fixture
async def client(db_session, test_storage) -> AsyncGenerator[AsyncClient, None]:
    """Provides an AsyncClient for testing FastAPI endpoints."""
    async def override_get_db():
        yield db_session

    def override_get_storage():
        return test_storage

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_storage] = override_get_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict:
    """Registers a fresh test user and returns an Authorization header for it."""
    res = await client.post(
        "/api/v1/auth/register",
        json={"email": "test_user@example.com", "password": "testpassword123", "display_name": "Test User"},
    )
    token = res.json()["token"]
    return {"Authorization": f"Bearer {token}"}
