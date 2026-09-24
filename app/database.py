"""Database connection and session management."""

from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.config import settings
from app.models.base import Base
from app.observability import logger
from app.schema_sync import sync_missing_columns

DEMO_USER_ID = "tenant_1"
DEMO_USER_EMAIL = "admin@closettheory.co"
DEMO_USER_PASSWORD = "admin1234"

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
        # create_all never alters a table that already exists, so a model gaining a column
        # after its table shipped left production 500-ing twice (users on 09-11, outfit_votes
        # on 09-23). Add whatever is missing, additively only — see app/schema_sync.py.
        added = await conn.run_sync(sync_missing_columns, Base.metadata)
        if added:
            logger.warning(f"Schema sync added {len(added)} missing column(s): {', '.join(added)}")

    await _ensure_demo_user()
    await _ensure_admin_roles()


async def _ensure_admin_roles() -> None:
    """Grants the "admin" role to any existing account listed in settings.ADMIN_EMAILS.

    The bootstrap exists because the first admin cannot be created through the admin endpoints —
    those require already being one. Idempotent, no network calls, and it only ever grants to
    accounts that already exist, so a typo in the setting is inert rather than account-creating.

    Note what is deliberately NOT here: seeding the evaluation characters. init_db()'s caller
    wraps it in a try/except that only logs (app/main.py), so a partial seed of twenty characters
    plus twenty image generations would fail invisibly and retry on every restart. That belongs
    in a script run on purpose — see scripts/seed_personas.py.
    """
    emails = [e.strip().lower() for e in (settings.ADMIN_EMAILS or "").split(",") if e.strip()]
    if not emails:
        return

    from sqlalchemy import select
    from app.models.role import ROLE_ADMIN, UserRole
    from app.models.user import User

    async with AsyncSessionLocal() as session:
        users = (await session.execute(select(User).where(User.email.in_(emails)))).scalars().all()
        if not users:
            return
        existing = set(
            (await session.execute(
                select(UserRole.user_id).where(
                    UserRole.user_id.in_([u.id for u in users]), UserRole.role == ROLE_ADMIN
                )
            )).scalars().all()
        )
        granted = 0
        for user in users:
            if user.id in existing:
                continue
            session.add(UserRole(user_id=user.id, role=ROLE_ADMIN))
            granted += 1
        if granted:
            await session.commit()


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
