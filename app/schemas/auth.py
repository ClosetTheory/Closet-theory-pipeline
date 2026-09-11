"""Auth request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

GENDER_VALUES = {"women", "men", "unisex"}


def _validate_email(value: str) -> str:
    # Minimal shape check only — no email-validator dependency (stdlib-only, presentation scope).
    value = value.strip().lower()
    if "@" not in value or " " in value or value.startswith("@") or value.endswith("@"):
        raise ValueError("Invalid email address")
    return value


def _validate_gender(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().lower()
    if value not in GENDER_VALUES:
        raise ValueError(f"gender must be one of {sorted(GENDER_VALUES)}")
    return value


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=255)
    gender: Optional[str] = Field(
        default=None,
        description="'women' | 'men' | 'unisex' — defaults the styling pipeline's wardrobe "
        "gender filter for this account when a request doesn't specify one itself.",
    )

    _validate_email = field_validator("email")(_validate_email)
    _validate_gender = field_validator("gender")(_validate_gender)


class UpdateProfileRequest(BaseModel):
    gender: Optional[str] = None

    _validate_gender = field_validator("gender")(_validate_gender)


class LoginRequest(BaseModel):
    email: str
    password: str

    _validate_email = field_validator("email")(_validate_email)


class AuthResponse(BaseModel):
    token: str
    user_id: str
    display_name: str


class MeResponse(BaseModel):
    user_id: str
    email: str
    display_name: str
    gender: Optional[str] = None
