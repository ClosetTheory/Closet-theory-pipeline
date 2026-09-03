"""Local filesystem implementation of StorageClient for development and tests."""

import os
from pathlib import Path
from app.storage.base import StorageClient


class LocalStorageClient(StorageClient):
    """Stores binary objects on the local filesystem."""

    def __init__(self, base_dir: str = "data/storage", bucket_name: str = "wardrobe"):
        self.base_dir = Path(base_dir)
        self.bucket_name = bucket_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        clean_key = key.lstrip("/\\").replace("object://", "")
        if clean_key.startswith(f"{self.bucket_name}/"):
            clean_key = clean_key[len(self.bucket_name) + 1:]
        return self.base_dir / clean_key

    async def put_object(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        dest_path = self._resolve_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        clean_key = key.lstrip("/\\")
        if not clean_key.startswith(f"object://{self.bucket_name}/"):
            return f"object://{self.bucket_name}/{clean_key}"
        return clean_key

    async def get_object(self, key: str) -> bytes:
        dest_path = self._resolve_path(key)
        if not dest_path.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        return dest_path.read_bytes()

    async def delete_object(self, key: str) -> bool:
        dest_path = self._resolve_path(key)
        if dest_path.exists():
            dest_path.unlink()
            return True
        return False

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        dest_path = self._resolve_path(key)
        return f"file:///{dest_path.resolve().as_posix()}"

    async def exists(self, key: str) -> bool:
        return self._resolve_path(key).exists()
