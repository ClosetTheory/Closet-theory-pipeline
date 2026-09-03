"""API dependencies for database and storage injection."""

from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.storage import get_storage_client, StorageClient


async def get_db_session(session: AsyncSession = Depends(get_db)) -> AsyncSession:
    return session


def get_storage() -> StorageClient:
    return get_storage_client()
