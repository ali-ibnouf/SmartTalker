"""Local file storage manager.

Handles output file cleanup, path resolution,
and storage statistics for generated audio/video files.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import Settings
from src.utils.logger import setup_logger

logger = setup_logger("integrations.storage")


@dataclass
class StorageStats:
    """Storage usage statistics.

    Attributes:
        total_files: Total number of stored files.
        total_size_mb: Total storage used in megabytes.
        oldest_file_age_hours: Age of oldest file in hours.
    """

    total_files: int = 0
    total_size_mb: float = 0.0
    oldest_file_age_hours: float = 0.0


class StorageManager:
    """Local file storage manager for generated outputs.

    Manages file lifecycle, automatic cleanup of old files,
    and storage statistics.

    Args:
        config: Application settings.
    """

    def __init__(self, config: Settings) -> None:
        """Initialize the storage manager.

        Args:
            config: Application settings instance.
        """
        self._config = config
        self._base_dir = config.storage_base_dir
        self._max_age_hours = config.storage_max_file_age_hours

        # Ensure all output subdirectories exist
        for subdir in ["tts", "video", "upscale", "uploads", "whatsapp_media"]:
            (self._base_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "StorageManager initialized",
            extra={"base_dir": str(self._base_dir), "max_age_hours": self._max_age_hours},
        )

    def get_stats(self) -> StorageStats:
        """Calculate storage usage statistics.

        Returns:
            StorageStats with file count, size, and oldest file age.
        """
        total_files = 0
        total_bytes = 0
        oldest_mtime = time.time()

        for file_path in self._base_dir.rglob("*"):
            if file_path.is_file():
                total_files += 1
                total_bytes += file_path.stat().st_size
                mtime = file_path.stat().st_mtime
                if mtime < oldest_mtime:
                    oldest_mtime = mtime

        oldest_age_hours = (time.time() - oldest_mtime) / 3600 if total_files > 0 else 0.0

        return StorageStats(
            total_files=total_files,
            total_size_mb=round(total_bytes / (1024 * 1024), 2),
            oldest_file_age_hours=round(oldest_age_hours, 1),
        )

    def cleanup_old_files(self) -> int:
        """Remove files older than the configured max age.

        Returns:
            Number of files deleted.
        """
        cutoff_time = time.time() - (self._max_age_hours * 3600)
        deleted_count = 0

        for file_path in self._base_dir.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError as exc:
                    logger.warning(f"Failed to delete {file_path}: {exc}")

        if deleted_count > 0:
            logger.info(
                "Storage cleanup complete",
                extra={"deleted": deleted_count, "max_age_hours": self._max_age_hours},
            )

        return deleted_count

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path within the storage directory.

        Args:
            relative_path: Path relative to the storage base directory.

        Returns:
            Absolute Path within the storage directory.
        """
        resolved = (self._base_dir / relative_path).resolve()

        # Security: ensure the resolved path is within the base directory
        if not str(resolved).startswith(str(self._base_dir.resolve())):
            raise ValueError(f"Path traversal attempt: {relative_path}")

        return resolved

    def get_file_url(self, file_path: Path, base_url: str = "") -> str:
        """Convert a storage path to a serveable URL.

        Args:
            file_path: Absolute path to the file.
            base_url: Server base URL.

        Returns:
            URL string.
        """
        try:
            relative = file_path.relative_to(self._base_dir)
            return f"{base_url.rstrip('/')}/files/{relative.name}"
        except ValueError:
            return f"/files/{file_path.name}"

    def clear_all(self) -> None:
        """Remove all stored files (use with caution)."""
        for subdir in self._base_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.iterdir():
                    if f.is_file():
                        f.unlink()
        logger.info("All stored files cleared")
