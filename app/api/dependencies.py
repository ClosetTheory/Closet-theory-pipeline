"""API dependencies for database, storage, and auth injection."""

from typing import AsyncGenerator
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.storage import get_storage_client, StorageClient


async def get_db_session(session: AsyncSession = Depends(get_db)) -> AsyncSession:
    return session


def get_storage() -> StorageClient:
    return get_storage_client()


async def get_current_user(
    authorization: str = Header(default=""),
    session: AsyncSession = Depends(get_db_session),
):
    from app.auth.security import verify_session_token
    from app.models.user import User

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or malformed Authorization header")

    user_id = verify_session_token(authorization.removeprefix("Bearer "))
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session token")

    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user
