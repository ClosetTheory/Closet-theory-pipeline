"""Garment model representing the canonical garment entity."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import DateTime, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base, TimestampMixin, generate_uuid, utc_now
from app.models.image_asset import ImageAsset


class Garment(Base, TimestampMixin):
    """Core domain entity holding the canonical garment representation."""

    __tablename__ = "garments"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_uuid("garm"),
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    member_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    source_image_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("image_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    garment_crop_refs: Mapped[List[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )

    # Which garment (of possibly several in the same source photo) this record represents,
    # e.g. "outerwear" / "top" / "footwear" — from Stage 2's detection call. Stage 3/4 use it
    # to tell the model which garment to isolate within the full photo, since garment_crop_refs
    # now always points at the same full source image rather than a physical per-garment crop.
    # None for catalog/single-garment images, where there's no ambiguity to resolve.
    detected_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Gendered styling association ("women" | "men" | "unisex"), from Stage 3's attribute
    # extraction. Mirrors category/subcategory's dedicated-column pattern for fast SQL filtering
    # in styling (see app/styling/filtering.py) rather than requiring a JSON path query.
    gender: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)

    category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    subcategory: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    garment_class: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    attributes_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    canonical_image_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("image_assets.id", ondelete="SET NULL"),
        nullable=True,
    )

    compatibility_features: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="RECEIVED",
        nullable=False,
        index=True,
    )
    quality_status: Mapped[str] = mapped_column(
        String(32),
        default="PENDING",
        nullable=False,
    )

    provenance: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    pipeline_version: Mapped[str] = mapped_column(String(32), default="1.0.0", nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Relationships
    source_image: Mapped["ImageAsset"] = relationship(
        "ImageAsset",
        foreign_keys=[source_image_id],
        lazy="joined",
    )
    canonical_image: Mapped[Optional["ImageAsset"]] = relationship(
        "ImageAsset",
        foreign_keys=[canonical_image_id],
        lazy="joined",
    )
