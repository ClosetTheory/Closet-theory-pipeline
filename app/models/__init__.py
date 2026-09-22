"""Database models package.

Everything that defines a table must be imported here. `create_all` only creates tables it can
see on `Base.metadata`, and a model that nothing imports is silently skipped — which is how
`User`, `StyleProfile` and `StylistReview` came to exist only because an API module happened to
import them at startup. They are exported below so that stops being luck.
"""

from app.models.base import Base, TimestampMixin, generate_uuid, utc_now
from app.models.image_asset import ImageAsset
from app.models.garment import Garment
from app.models.pipeline_stage import PipelineStageRun
from app.models.embedding import GarmentEmbedding, PortableVector
from app.models.compatibility import CompatibilityResult
from app.models.styling import StylingRequest, Outfit, OutfitGarment, StylistReview
from app.models.styling_run import StylingRunProgress, RUN_STATUSES
from app.models.ootd import OOTDSubscription, OutfitOfTheDay
from app.models.style_profile import StyleProfile
from app.models.user import User
from app.models.role import UserRole, ROLE_ADMIN, ROLE_STYLIST, KNOWN_ROLES
from app.models.persona import Persona, PersonaAssignment, PersonaGarmentUpload
from app.models.persona_review import PersonaOutfitReview, REVIEW_DIMENSIONS

__all__ = [
    "Base",
    "TimestampMixin",
    "generate_uuid",
    "utc_now",
    "ImageAsset",
    "Garment",
    "PipelineStageRun",
    "GarmentEmbedding",
    "PortableVector",
    "CompatibilityResult",
    "StylingRequest",
    "Outfit",
    "OutfitGarment",
    "StylistReview",
    "StylingRunProgress",
    "RUN_STATUSES",
    "OOTDSubscription",
    "OutfitOfTheDay",
    "StyleProfile",
    "User",
    "UserRole",
    "ROLE_ADMIN",
    "ROLE_STYLIST",
    "KNOWN_ROLES",
    "Persona",
    "PersonaAssignment",
    "PersonaGarmentUpload",
    "PersonaOutfitReview",
    "REVIEW_DIMENSIONS",
]
