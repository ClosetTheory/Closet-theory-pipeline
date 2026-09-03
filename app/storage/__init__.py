"""Storage package exports."""

from app.config import settings
from app.storage.base import StorageClient
from app.storage.local_storage import LocalStorageClient
from app.storage.s3_storage import S3StorageClient

_storage_client: StorageClient | None = None


def get_storage_client() -> StorageClient:
    """Singleton getter for the configured storage client."""
    global _storage_client
    if _storage_client is None:
        if settings.STORAGE_BACKEND.lower() == "s3":
            _storage_client = S3StorageClient(
                bucket_name=settings.S3_BUCKET_NAME,
                endpoint_url=settings.S3_ENDPOINT_URL,
                access_key_id=settings.S3_ACCESS_KEY_ID,
                secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION,
            )
        else:
            _storage_client = LocalStorageClient(
                base_dir=settings.LOCAL_STORAGE_DIR,
                bucket_name=settings.S3_BUCKET_NAME,
            )
    return _storage_client
