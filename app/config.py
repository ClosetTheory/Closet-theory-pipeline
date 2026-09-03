"""Application configuration and settings."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General
    ENV: str = "development"
    DEBUG: bool = True
    PIPELINE_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/wardrobe"

    # Storage
    STORAGE_BACKEND: str = "local"  # "local" or "s3"
    LOCAL_STORAGE_DIR: str = "data/storage"
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = "minioadmin"
    S3_SECRET_ACCESS_KEY: Optional[str] = "minioadmin"
    S3_BUCKET_NAME: str = "wardrobe-assets"
    S3_REGION: str = "us-east-1"

    # Redis / Async Worker Queue
    REDIS_URL: str = "redis://localhost:6379/0"
    USE_IN_MEMORY_QUEUE: bool = True
    WORKER_CONCURRENCY: int = 5  # number of pipeline jobs processed in parallel per worker process

    # Image Upload Security Limits
    MAX_IMAGE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_IMAGE_PIXELS: int = 89_478_485  # Pillow default decompression bomb ceiling
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]

    # Stage 1: Classifier (MobileNetV3)
    CLASSIFIER_PROVIDER: str = "mock"  # "mobilenet" | "mock"
    CLASSIFIER_MODEL_NAME: str = "MobileNetV3"
    CLASSIFIER_MODEL_VERSION: str = "v1"
    CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.70

    # OpenRouter API Configuration (GPT Models)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "openai/gpt-4o"
    OPENROUTER_GENAI_MODEL: str = "openai/dall-e-3"
    OPENROUTER_IMAGE_MODEL: str = "openai/gpt-image-2"

    # NVIDIA NIM API Configuration
    NVIDIA_API_KEY: Optional[str] = None
    NVIDIA_VLM_MODEL: str = "meta/llama-3.2-11b-vision-instruct"
    NVIDIA_VLM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_GENAI_MODEL: str = "black-forest-labs/flux.1-schnell"
    NVIDIA_GENAI_BASE_URL: str = "https://ai.api.nvidia.com/v1/genai"

    # Stage 2: Detection & Garment Crop (OpenCV / RetinaFace + SAM)
    DETECTION_PROVIDER: str = "opencv"  # "opencv" | "retina_sam" | "mock"
    DETECTION_MODEL_NAME: str = "OpenCV-Haar+SAM"
    DETECTION_MODEL_VERSION: str = "v1"

    # Stage 3: Image -> Garment Attributes (OpenRouter GPT / NVIDIA NIM / Gemini / Claude / MODA_NER)
    ATTRIBUTE_PROVIDER: str = "openrouter"  # "openrouter" | "nvidia_nim" | "gemini" | "claude" | "moda_ner" | "mock"
    ATTRIBUTE_MODEL_NAME: str = "openai/gpt-4o"
    ATTRIBUTE_MODEL_VERSION: str = "v1"
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Stage 4: Image Digitisation (FLUX.2)
    DIGITISATION_PROVIDER: str = "flux"  # "flux" | "gpt" | "mock"
    DIGITISATION_MODEL_NAME: str = "black-forest-labs/flux.2-pro"
    DIGITISATION_MODEL_VERSION: str = "v1"
    DIGITISATION_PROMPT_VERSION: str = "prompt_v1"
    DIGITISATION_MAX_RETRIES: int = 3
    DIGITISATION_QUALITY_THRESHOLD: float = 0.75

    # Stage 5: Image Embedding (MODA SigLIP Distilled)
    EMBEDDING_PROVIDER: str = "mock"  # "siglip" | "mock"
    EMBEDDING_MODEL_NAME: str = "MODA SigLIP Distilled"
    EMBEDDING_MODEL_VERSION: str = "v1"
    EMBEDDING_DIMENSION: int = 768
    HF_API_KEY: Optional[str] = None
    HF_EMBEDDING_MODEL: str = "google/siglip-base-patch16-224"
    HF_INFERENCE_BASE_URL: str = "https://api-inference.huggingface.co/models"

    # Stage 6: Category Bundling (Lookup Table)
    TAXONOMY_VERSION: str = "taxonomy_v1"

    # Stage 7: Layering Compatibility Rules
    LAYERING_RULE_VERSION: str = "layering_v1"

    # Stage 8: Structural Compatibility Rules
    STRUCTURAL_RULE_VERSION: str = "structural_v1"

    # Stage 9: Visual Compatibility (Rules + VLM Fallback)
    VISUAL_RULE_VERSION: str = "visual_v1"
    VLM_PROVIDER: str = "mock"  # "gemini" | "claude" | "mock"
    VLM_MODEL_NAME: str = "gemini-flash"
    VLM_MODEL_VERSION: str = "v1"

    # --- Styling Pipeline (Outfit Recommendation) ---
    STYLING_NORMALIZER_PROVIDER: str = "openrouter"  # "openrouter" | "mock"
    STYLING_VALIDATOR_PROVIDER: str = "openrouter"  # "openrouter" | "mock"
    STYLING_VISUAL_VALIDATOR_PROVIDER: str = "openrouter"  # "openrouter" | "mock"
    STYLING_OUTFIT_IMAGE_PROVIDER: str = "gpt"  # "gpt" | "mock"
    STYLING_SCORER_VERSION: str = "styling_scorer_v1"
    STYLING_MAX_CANDIDATES_PER_ROLE: int = 6
    STYLING_MAX_OUTFITS_RETURNED: int = 3
    STYLING_IMAGE_MAX_RETRIES: int = 2


settings = Settings()
