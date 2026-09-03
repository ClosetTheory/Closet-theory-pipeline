"""Abstract base class for object storage backends."""

from abc import ABC, abstractmethod
from typing import Optional


class StorageClient(ABC):
    """Abstract interface separating byte storage from metadata/relational persistence."""

    @abstractmethod
    async def put_object(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        """Store bytes at key. Returns the canonical object URI (e.g. object://bucket/key)."""
        pass

    @abstractmethod
    async def get_object(self, key: str) -> bytes:
        """Retrieve raw bytes for key."""
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """Delete object at key."""
        pass

    @abstractmethod
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a secure pre-signed read URL for external clients."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if an object exists at key."""
        pass
