"""Cloudflare R2 storage client (S3-compatible).

Handles all file storage for the Maskki platform:
- Employee photos and face data
- Session audio and rendered video
- Voice enrollment samples

Uses boto3 with S3-compatible endpoint to connect to Cloudflare R2.
Public URL: https://media.maskki.com/{key}
"""

from __future__ import annotations

import time
from typing import Optional

from src.config import Settings
from src.utils.exceptions import SmartTalkerError
from src.utils.logger import setup_logger

logger = setup_logger("services.r2")


class R2Error(SmartTalkerError):
    """Error from Cloudflare R2 storage."""
    pass


class R2Storage:
    """Cloudflare R2 storage client using S3-compatible API.

    Manages media files for employees, sessions, and voice data.
    All uploads return a public URL via the configured R2 public domain.
    """

    def __init__(self, config: Settings) -> None:
        self._bucket = config.r2_bucket
        self._public_url = config.r2_public_url.rstrip("/")
        self._account_id = config.r2_account_id
        self._client = None

        # Store credentials for lazy client init (boto3 may not be installed in tests)
        self._endpoint_url = f"https://{config.r2_account_id}.r2.cloudflarestorage.com"
        self._access_key_id = config.r2_access_key_id
        self._secret_access_key = config.r2_secret_access_key

        logger.info(
            "R2Storage initialized",
            extra={
                "bucket": self._bucket,
                "public_url": bool(self._public_url),
            },
        )

    def _get_client(self):
        """Lazy-init the boto3 S3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
            except ImportError:
                raise R2Error(
                    message="boto3 not installed",
                    detail="Install with: pip install boto3",
                )

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                config=BotoConfig(
                    region_name="auto",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
        return self._client

    def _public_key_url(self, key: str) -> str:
        """Build public URL for an R2 object key."""
        return f"{self._public_url}/{key}"

    # ── Employee Media ─────────────────────────────────────────────────────

    def upload_employee_photo(
        self, employee_id: str, photo_bytes: bytes, content_type: str = "image/jpeg"
    ) -> str:
        """Upload an employee photo to R2.

        Args:
            employee_id: Employee identifier.
            photo_bytes: Raw photo bytes.
            content_type: MIME type (default image/jpeg).

        Returns:
            Public URL of the uploaded photo.
        """
        key = f"employees/{employee_id}/photo.jpg"
        self._put_object(key, photo_bytes, content_type)
        url = self._public_key_url(key)

        logger.info(
            "Employee photo uploaded",
            extra={"employee_id": employee_id, "size_bytes": len(photo_bytes)},
        )
        return url

    def upload_face_data(self, employee_id: str, face_data: bytes) -> str:
        """Upload preprocessed face data (LivePortrait output) to R2.

        Args:
            employee_id: Employee identifier.
            face_data: Serialized face data bytes.

        Returns:
            Public URL of the uploaded face data.
        """
        key = f"employees/{employee_id}/face_data.bin"
        self._put_object(key, face_data, "application/octet-stream")
        url = self._public_key_url(key)

        logger.info(
            "Face data uploaded",
            extra={"employee_id": employee_id, "size_bytes": len(face_data)},
        )
        return url

    def upload_voice_sample(self, employee_id: str, audio_bytes: bytes) -> str:
        """Upload a voice enrollment sample to R2.

        Args:
            employee_id: Employee identifier.
            audio_bytes: Raw audio bytes (WAV or PCM).

        Returns:
            Public URL of the uploaded voice sample.
        """
        key = f"employees/{employee_id}/voice_sample.wav"
        self._put_object(key, audio_bytes, "audio/wav")
        url = self._public_key_url(key)

        logger.info(
            "Voice sample uploaded",
            extra={"employee_id": employee_id, "size_bytes": len(audio_bytes)},
        )
        return url

    # ── Session Media ──────────────────────────────────────────────────────

    def upload_audio(self, session_id: str, audio_bytes: bytes) -> str:
        """Upload session audio to R2.

        Args:
            session_id: Session identifier.
            audio_bytes: Raw PCM audio bytes.

        Returns:
            Public URL of the uploaded audio.
        """
        timestamp = int(time.time() * 1000)
        key = f"sessions/{session_id}/audio_{timestamp}.pcm"
        self._put_object(key, audio_bytes, "audio/pcm")
        url = self._public_key_url(key)

        logger.info(
            "Session audio uploaded",
            extra={"session_id": session_id, "size_bytes": len(audio_bytes)},
        )
        return url

    def upload_video(self, session_id: str, video_bytes: bytes) -> str:
        """Upload rendered video to R2.

        Args:
            session_id: Session identifier.
            video_bytes: Rendered video bytes (MP4).

        Returns:
            Public URL of the uploaded video.
        """
        timestamp = int(time.time() * 1000)
        key = f"sessions/{session_id}/video_{timestamp}.mp4"
        self._put_object(key, video_bytes, "video/mp4")
        url = self._public_key_url(key)

        logger.info(
            "Session video uploaded",
            extra={"session_id": session_id, "size_bytes": len(video_bytes)},
        )
        return url

    # ── Deletion ───────────────────────────────────────────────────────────

    def delete_employee_media(self, employee_id: str) -> int:
        """Delete all media for an employee.

        Removes everything under employees/{employee_id}/.

        Args:
            employee_id: Employee identifier.

        Returns:
            Number of objects deleted.
        """
        prefix = f"employees/{employee_id}/"
        count = self._delete_prefix(prefix)

        logger.info(
            "Employee media deleted",
            extra={"employee_id": employee_id, "objects_deleted": count},
        )
        return count

    def delete_session_media(self, session_id: str) -> int:
        """Delete all media for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of objects deleted.
        """
        prefix = f"sessions/{session_id}/"
        count = self._delete_prefix(prefix)

        logger.info(
            "Session media deleted",
            extra={"session_id": session_id, "objects_deleted": count},
        )
        return count

    def delete_customer_media(
        self, customer_id: str, employee_ids: list[str]
    ) -> int:
        """Batch delete all media for a customer's employees.

        Args:
            customer_id: Customer identifier (for logging).
            employee_ids: List of employee IDs whose media to delete.

        Returns:
            Total number of objects deleted.
        """
        total = 0
        for emp_id in employee_ids:
            total += self.delete_employee_media(emp_id)

        logger.info(
            "Customer media deleted",
            extra={
                "customer_id": customer_id,
                "employee_count": len(employee_ids),
                "objects_deleted": total,
            },
        )
        return total

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _put_object(
        self, key: str, body: bytes, content_type: str
    ) -> None:
        """Upload bytes to R2.

        Args:
            key: S3 object key.
            body: File content.
            content_type: MIME type.
        """
        client = self._get_client()
        try:
            client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        except Exception as exc:
            raise R2Error(
                message=f"R2 upload failed: {key}",
                detail=str(exc),
                original_exception=exc,
            ) from exc

    def _delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix.

        Args:
            prefix: S3 key prefix.

        Returns:
            Number of objects deleted.
        """
        client = self._get_client()
        deleted = 0

        try:
            # List objects with prefix
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)

            for page in pages:
                objects = page.get("Contents", [])
                if not objects:
                    continue

                # Batch delete (max 1000 per request)
                delete_keys = [{"Key": obj["Key"]} for obj in objects]
                client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": delete_keys, "Quiet": True},
                )
                deleted += len(delete_keys)

        except Exception as exc:
            raise R2Error(
                message=f"R2 delete failed for prefix: {prefix}",
                detail=str(exc),
                original_exception=exc,
            ) from exc

        return deleted
