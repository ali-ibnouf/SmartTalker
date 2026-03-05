"""RunPod Serverless client for GPU rendering tasks.

Submits jobs to RunPod endpoints for MuseTalk lip-sync rendering
and LivePortrait face preprocessing. Replaces the old GPU RenderNode.

Endpoints:
- musetalk-render: MuseTalk lip-sync (RTX 4090, ~$0.00076/sec)
- face-preprocess: LivePortrait face extraction (RTX 4090)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from src.config import Settings
from src.utils.exceptions import SmartTalkerError
from src.utils.logger import setup_logger, log_with_latency

logger = setup_logger("services.runpod")

# RunPod RTX 4090 serverless cost: ~$0.00076/sec
COST_PER_SEC = 0.00076

# Polling interval for job status
POLL_INTERVAL = 0.5  # seconds
DEFAULT_TIMEOUT = 120  # seconds


class RunPodError(SmartTalkerError):
    """Error from RunPod Serverless."""
    pass


@dataclass
class RenderResult:
    """Result of a RunPod rendering job."""
    video_url: str
    execution_time_ms: int = 0
    cost_usd: float = 0.0
    job_id: str = ""


@dataclass
class PreprocessResult:
    """Result of a RunPod face preprocessing job."""
    face_data_url: str
    execution_time_ms: int = 0
    cost_usd: float = 0.0
    job_id: str = ""


class RunPodServerless:
    """Client for RunPod Serverless GPU endpoints.

    Submits async jobs and polls for results. Used for:
    - MuseTalk lip-sync rendering (audio + face data -> video)
    - LivePortrait face preprocessing (photo -> face data)
    """

    def __init__(self, config: Settings) -> None:
        self._api_key = config.runpod_api_key
        self._endpoint_musetalk = config.runpod_endpoint_musetalk
        self._endpoint_preprocess = config.runpod_endpoint_preprocess
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "RunPodServerless initialized",
            extra={
                "musetalk_endpoint": bool(self._endpoint_musetalk),
                "preprocess_endpoint": bool(self._endpoint_preprocess),
            },
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def preprocess_face(
        self, photo_url: str, employee_id: str
    ) -> PreprocessResult:
        """Submit a face preprocessing job to RunPod.

        Downloads photo, extracts face landmarks via LivePortrait,
        uploads face_data to R2.

        Args:
            photo_url: R2 URL to the employee's photo.
            employee_id: Employee identifier for output naming.

        Returns:
            PreprocessResult with face_data_url and cost.
        """
        if not self._endpoint_preprocess:
            raise RunPodError(message="RunPod preprocess endpoint not configured")

        body = {
            "input": {
                "task": "preprocess_face",
                "photo_url": photo_url,
                "employee_id": employee_id,
            }
        }

        result = await self._run_job(self._endpoint_preprocess, body)

        face_data_url = result.get("face_data_url", "")
        if not face_data_url:
            raise RunPodError(
                message="RunPod preprocess returned no face_data_url",
                detail=str(result),
            )

        execution_ms = result.get("execution_time_ms", 0)
        cost = (execution_ms / 1000.0) * COST_PER_SEC

        return PreprocessResult(
            face_data_url=face_data_url,
            execution_time_ms=execution_ms,
            cost_usd=cost,
            job_id=result.get("job_id", ""),
        )

    async def render_lipsync(
        self,
        audio_url: str,
        face_data_url: str,
        employee_id: str,
        session_id: str,
    ) -> RenderResult:
        """Submit a MuseTalk lip-sync rendering job to RunPod.

        Takes audio + preprocessed face data, renders lip-synced video.

        Args:
            audio_url: R2 URL to the audio file.
            face_data_url: R2 URL to the preprocessed face data.
            employee_id: Employee identifier.
            session_id: Session identifier for output naming.

        Returns:
            RenderResult with video_url and cost.
        """
        if not self._endpoint_musetalk:
            raise RunPodError(message="RunPod MuseTalk endpoint not configured")

        body = {
            "input": {
                "task": "render_lipsync",
                "audio_url": audio_url,
                "face_data_url": face_data_url,
                "employee_id": employee_id,
                "session_id": session_id,
            }
        }

        result = await self._run_job(self._endpoint_musetalk, body)

        video_url = result.get("video_url", "")
        if not video_url:
            raise RunPodError(
                message="RunPod render returned no video_url",
                detail=str(result),
            )

        execution_ms = result.get("execution_time_ms", 0)
        cost = (execution_ms / 1000.0) * COST_PER_SEC

        return RenderResult(
            video_url=video_url,
            execution_time_ms=execution_ms,
            cost_usd=cost,
            job_id=result.get("job_id", ""),
        )

    async def _run_job(
        self,
        endpoint: str,
        body: dict[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Submit a job to RunPod and poll for completion.

        Args:
            endpoint: RunPod endpoint URL (e.g. https://api.runpod.ai/v2/{id}).
            body: Job input payload.
            timeout: Maximum wait time in seconds.

        Returns:
            Job output dict from RunPod.
        """
        client = await self._get_client()
        start = time.perf_counter()

        # Submit job
        try:
            submit_resp = await client.post(f"{endpoint}/run", json=body)
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()
        except httpx.HTTPStatusError as exc:
            raise RunPodError(
                message=f"RunPod job submission failed: {exc.response.status_code}",
                detail=exc.response.text,
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise RunPodError(
                message="RunPod job submission failed",
                detail=str(exc),
                original_exception=exc,
            ) from exc

        job_id = submit_data.get("id", "")
        if not job_id:
            raise RunPodError(
                message="RunPod returned no job ID",
                detail=str(submit_data),
            )

        logger.info("RunPod job submitted", extra={"job_id": job_id, "endpoint": endpoint})

        # Poll for completion
        status_url = f"{endpoint}/status/{job_id}"
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                raise RunPodError(
                    message=f"RunPod job timed out after {timeout}s",
                    detail=f"job_id={job_id}",
                )

            await asyncio.sleep(POLL_INTERVAL)

            try:
                status_resp = await client.get(status_url)
                status_resp.raise_for_status()
                status_data = status_resp.json()
            except Exception:
                continue  # Retry on transient errors

            status = status_data.get("status", "")

            if status == "COMPLETED":
                output = status_data.get("output", {})
                execution_ms = int((time.perf_counter() - start) * 1000)
                output["execution_time_ms"] = execution_ms
                output["job_id"] = job_id

                log_with_latency(
                    logger, "RunPod job completed", execution_ms,
                    extra={"job_id": job_id},
                )
                return output

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                raise RunPodError(
                    message=f"RunPod job failed: {error}",
                    detail=f"job_id={job_id}",
                )

            elif status in ("CANCELLED", "TIMED_OUT"):
                raise RunPodError(
                    message=f"RunPod job {status.lower()}",
                    detail=f"job_id={job_id}",
                )

            # IN_QUEUE, IN_PROGRESS — keep polling

    @staticmethod
    def calculate_cost(execution_time_ms: int) -> float:
        """Calculate cost for a RunPod job based on execution time."""
        return (execution_time_ms / 1000.0) * COST_PER_SEC

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("RunPod client closed")
