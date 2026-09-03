"""Auth request/response schemas."""

from pydantic import BaseModel, Field, field_validator


def _validate_email(value: str) -> str:
    # Minimal shape check only — no email-validator dependency (stdlib-only, presentation scope).
    value = value.strip().lower()
    if "@" not in value or " " in value or value.startswith("@") or value.endswith("@"):
        raise ValueError("Invalid email address")
    return value


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=255)

    _validate_email = field_validator("email")(_validate_email)


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
