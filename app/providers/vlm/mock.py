"""Mock VLM Provider for visual compatibility judgments."""

from typing import Any, Dict, Optional, Tuple
from app.config import settings
from app.providers.base import BaseVLMProvider


class MockVLMProvider(BaseVLMProvider):
    """Simulates multimodal VLM visual compatibility evaluation."""

    def __init__(
        self,
        model_name: str = settings.VLM_MODEL_NAME,
        model_version: str = settings.VLM_MODEL_VERSION,
        forced_decision: Optional[str] = None,
        forced_score: float = 0.85,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.forced_decision = forced_decision
        self.forced_score = forced_score

    async def evaluate_visual_compatibility(
        self,
        image_a_bytes: Optional[bytes],
        image_b_bytes: Optional[bytes],
        attrs_a: Dict[str, Any],
        attrs_b: Dict[str, Any],
    ) -> Tuple[str, float, str]:
        if self.forced_decision:
            return (
                self.forced_decision,
                self.forced_score,
                f"VLM judgment ({self.model_name}): Forced test decision {self.forced_decision}.",
            )

        # Semantic aesthetic judgment based on colors and materials
        color_a = ", ".join(attrs_a.get("colour", ["neutral"]))
        color_b = ", ".join(attrs_b.get("colour", ["neutral"]))
        mat_a = attrs_a.get("material", "cotton")
        mat_b = attrs_b.get("material", "cotton")

        return (
            "COMPATIBLE",
            0.87,
            f"VLM judgment ({self.model_name}): Textural interplay between {mat_a} and {mat_b} "
            f"paired with {color_a} and {color_b} provides tasteful visual balance.",
        )
