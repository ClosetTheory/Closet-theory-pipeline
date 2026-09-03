"""GarmentEmbedding model with pgvector support and cosine similarity."""

from typing import List
from sqlalchemy import DateTime, ForeignKey, Integer, String, TypeDecorator, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, generate_uuid, utc_now

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None


class PortableVector(TypeDecorator):
    """Custom type that uses pgvector.Vector on PostgreSQL and JSON elsewhere (e.g. SQLite tests)."""

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and Vector is not None:
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            return value.tolist()
        return [float(x) for x in value]


class GarmentEmbedding(Base):
    """Normalized embedding vector representation for garment similarity retrieval."""

    __tablename__ = "garment_embeddings"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("emb"),
    )
    garment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("garments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[List[float]] = mapped_column(PortableVector(768), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, default=768, nullable=False)
    source_image_version: Mapped[str] = mapped_column(String(64), default="v1", nullable=False)

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
