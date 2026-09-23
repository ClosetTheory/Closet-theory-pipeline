"""Mock feedback extractor: the deterministic keyword heuristic, no model call."""

from typing import Sequence
from app.providers.base import BaseFeedbackExtractorProvider
from app.rules.feedback import FeedbackExtraction, FeedbackGarment, heuristic_extract


class MockFeedbackExtractorProvider(BaseFeedbackExtractorProvider):
    def __init__(self, model_name: str = "mock-feedback-extractor", model_version: str = "v1"):
        self.model_name = model_name
        self.model_version = model_version

    async def extract(self, comment: str, vote: str, garments: Sequence[FeedbackGarment]) -> FeedbackExtraction:
        return heuristic_extract(comment, vote, garments)
