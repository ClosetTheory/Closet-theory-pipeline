"""Database connection and session management."""

from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.config import settings
from app.models.base import Base

DEMO_USER_ID = "tenant_1"
DEMO_USER_EMAIL = "demo@closettheory.local"
DEMO_USER_PASSWORD = "demo1234"

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG and settings.ENV == "development",
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db():
    """Initializes database schema and enables pgvector extension on PostgreSQL."""
    async with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            except Exception:
                pass
        await conn.run_sync(Base.metadata.create_all)

    await _ensure_demo_user()


async def _ensure_demo_user() -> None:
    """Seeds a demo account (id=tenant_1/member_1) so the pre-existing demo wardrobe ingested
    before the auth system existed stays reachable — every genuinely new registered user gets
    their own fresh tenant_id/member_id instead, with an empty wardrobe."""
    from app.auth.security import hash_password
    from app.models.user import User

    async with AsyncSessionLocal() as session:
        existing = await session.get(User, DEMO_USER_ID)
        if existing:
            return
        session.add(User(
            id=DEMO_USER_ID,
            email=DEMO_USER_EMAIL,
            display_name="Demo",
            password_hash=hash_password(DEMO_USER_PASSWORD),
            tenant_id=DEMO_USER_ID,
            member_id="member_1",
        ))
        await session.commit()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
