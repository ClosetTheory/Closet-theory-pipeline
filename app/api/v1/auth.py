"""Auth endpoints: register, login, me. See app/auth/security.py for hashing/token details."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_user, get_db_session
from app.auth.security import create_session_token, hash_password, verify_password
from app.models.base import generate_uuid
from app.models.user import User
from app.schemas.auth import AuthResponse, LoginRequest, MeResponse, RegisterRequest

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, session: AsyncSession = Depends(get_db_session)):
    existing = (await session.execute(select(User).where(User.email == request.email))).scalars().first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")

    new_id = generate_uuid("user")
    user = User(
        id=new_id,
        email=request.email,
        display_name=request.display_name,
        password_hash=hash_password(request.password),
        tenant_id=new_id,
        member_id=new_id,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    return AuthResponse(token=create_session_token(user.id), user_id=user.id, display_name=user.display_name)


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, session: AsyncSession = Depends(get_db_session)):
    user = (await session.execute(select(User).where(User.email == request.email))).scalars().first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    return AuthResponse(token=create_session_token(user.id), user_id=user.id, display_name=user.display_name)


@router.get("/me", response_model=MeResponse)
async def me(current_user: User = Depends(get_current_user)):
    return MeResponse(user_id=current_user.id, email=current_user.email, display_name=current_user.display_name)
