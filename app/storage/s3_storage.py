"""AWS S3 / MinIO implementation of StorageClient."""

import asyncio
from typing import Optional
import boto3
from botocore.exceptions import ClientError
from app.storage.base import StorageClient


class S3StorageClient(StorageClient):
    """S3-compatible object storage client (AWS S3 or MinIO)."""

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
    ):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            except Exception:
                pass

    def _clean_key(self, key: str) -> str:
        clean = key.replace(f"object://{self.bucket_name}/", "")
        clean = clean.lstrip("/")
        return clean

    async def put_object(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        clean_key = self._clean_key(key)

        def _sync_put():
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=clean_key,
                Body=data,
                ContentType=content_type,
            )

        await asyncio.to_thread(_sync_put)
        return f"object://{self.bucket_name}/{clean_key}"

    async def get_object(self, key: str) -> bytes:
        clean_key = self._clean_key(key)

        def _sync_get():
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=clean_key)
            return response["Body"].read()

        return await asyncio.to_thread(_sync_get)

    async def delete_object(self, key: str) -> bool:
        clean_key = self._clean_key(key)

        def _sync_delete():
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=clean_key)
            return True

        return await asyncio.to_thread(_sync_delete)

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        clean_key = self._clean_key(key)

        def _sync_presign():
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": clean_key},
                ExpiresIn=expires_in,
            )

        return await asyncio.to_thread(_sync_presign)

    async def exists(self, key: str) -> bool:
        clean_key = self._clean_key(key)

        def _sync_exists():
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=clean_key)
                return True
            except ClientError:
                return False

        return await asyncio.to_thread(_sync_exists)
