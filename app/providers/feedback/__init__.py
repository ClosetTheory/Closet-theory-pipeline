"""Feedback extractor provider factory: free-text review comments -> FeedbackExtraction.

Selected by settings.STYLING_FEEDBACK_PROVIDER ("openrouter" | "mock"). The mock is the keyword
heuristic in app.rules.feedback, which the OpenRouter implementation also falls back to when the
model call fails or returns nothing usable — a comment can only ever sharpen a vote, never lose it.
"""

from app.config import settings
from app.providers.base import BaseFeedbackExtractorProvider
from app.providers.feedback.mock import MockFeedbackExtractorProvider


def get_feedback_extractor_provider() -> BaseFeedbackExtractorProvider:
    provider_name = settings.STYLING_FEEDBACK_PROVIDER.lower()
    if provider_name == "openrouter" and settings.OPENROUTER_API_KEY:
        from app.providers.feedback.openrouter import OpenRouterFeedbackExtractorProvider

        return OpenRouterFeedbackExtractorProvider(
            api_key=settings.OPENROUTER_API_KEY,
            model_name=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockFeedbackExtractorProvider()
