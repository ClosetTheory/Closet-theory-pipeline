"""Per-user roles, as a table rather than a column on `users`.

The obvious design would be `users.role`. It is not safe here. This project has no Alembic —
`init_db()` only calls `Base.metadata.create_all`, which creates missing TABLES but never adds a
column to an existing one. `get_current_user` does `session.get(User, user_id)`, which SELECTs
every mapped column, so a `users.role` present in the model but absent from the deployed database
breaks login for everyone — and with auth down there is no endpoint left to repair it through.
That exact failure already happened once on this project when `users.gender` was added.

A new table has no such failure mode: it appears on the next boot, and nothing that already runs
selects from it.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now

# Kept as plain strings rather than an Enum column so adding a role later is a code change, not
# a database migration — same reasoning as every other enum-ish column in this schema.
ROLE_ADMIN = "admin"
ROLE_STYLIST = "stylist"
KNOWN_ROLES = frozenset({ROLE_ADMIN, ROLE_STYLIST})


class UserRole(Base):
    """One row per (user, role). A user with no rows here is an ordinary member."""

    __tablename__ = "user_roles"
    __table_args__ = (UniqueConstraint("user_id", "role", name="uq_user_role"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: generate_uuid("urole"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "admin" | "stylist"
    granted_by_user_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
