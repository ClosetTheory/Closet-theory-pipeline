"""User account model. tenant_id/member_id are real columns (not derived from id) so the
one seeded demo account can be pointed at the pre-existing tenant_1/member_1 demo wardrobe;
every other (real, registered) user gets tenant_id = member_id = their own user id."""

from typing import Optional
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, TimestampMixin, generate_uuid


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("user"))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Styling gender preference ("women" | "men" | "unisex"), matching Garment.gender's own
    # values (see app/schemas/attributes.py::GenderEnum) — lets Stage 1/4 of the styling
    # pipeline default a request's intent.gender to this member's own preference instead of
    # combining every gendered item in a mixed wardrobe into the same outfit shortlist (wastes
    # real image-generation cost on outfits the member never wanted in the first place).
    gender: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
