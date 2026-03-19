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

        # Rolling buffers for monitoring
        from collections import deque
        self._recent_render_times: deque[float] = deque(maxlen=50)  # seconds (all tasks)
        self._recent_preprocess_times: deque[float] = deque(maxlen=50)  # preprocess_face only
        self._recent_lipsync_times: deque[float] = deque(maxlen=50)  # render_lipsync only
        self._recent_r2_latencies: deque[float] = deque(maxlen=50)  # R2 upload seconds
        self._recent_timeouts: deque[float] = deque(maxlen=20)  # timeout timestamps
        self._recent_failures: deque[float] = deque(maxlen=20)  # failure timestamps

        # Consecutive failure tracking (for agent escalation)
        self._consecutive_failures: int = 0
        self.video_disabled: bool = False  # Set by agent auto-fix after 3+ failures

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

        result = await self._run_job(self._endpoint_preprocess, body, task_type="preprocess_face")

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

        result = await self._run_job(self._endpoint_musetalk, body, task_type="render_lipsync")

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
        task_type: str = "",
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Submit a job to RunPod and poll for completion.

        On FAILED status, retries up to max_retries times before raising.
        Tracks consecutive failures for agent escalation monitoring.

        Args:
            endpoint: RunPod endpoint URL (e.g. https://api.runpod.ai/v2/{id}).
            body: Job input payload.
            timeout: Maximum wait time in seconds.
            task_type: Task identifier for per-type metrics tracking.
            max_retries: Max retry attempts on job FAILED status.

        Returns:
            Job output dict from RunPod.
        """
        last_error: Optional[RunPodError] = None

        for attempt in range(1 + max_retries):
            try:
                result = await self._run_job_once(
                    endpoint, body, timeout, task_type,
                )
                # Success — reset consecutive failure counter
                self._consecutive_failures = 0
                return result
            except RunPodError as exc:
                last_error = exc
                is_job_failure = "job failed" in str(exc.message).lower()
                if is_job_failure and attempt < max_retries:
                    logger.warning(
                        f"RunPod job failed (attempt {attempt + 1}/{1 + max_retries}), retrying",
                        extra={"task_type": task_type, "error": str(exc.message)},
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))  # 1s, 2s backoff
                    continue
                # Not retriable or exhausted retries
                self._consecutive_failures += 1
                raise

        # Should not reach here, but just in case
        self._consecutive_failures += 1
        raise last_error  # type: ignore[misc]

    async def _run_job_once(
        self,
        endpoint: str,
        body: dict[str, Any],
        timeout: float,
        task_type: str,
    ) -> dict[str, Any]:
        """Submit and poll a single job attempt."""
        client = await self._get_client()
        start = time.perf_counter()

        # Submit job
        try:
            submit_resp = await client.post(f"{endpoint}/run", json=body)
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()
        except httpx.HTTPStatusError as exc:
            self._recent_failures.append(time.time())
            raise RunPodError(
                message=f"RunPod job submission failed: {exc.response.status_code}",
                detail=exc.response.text,
                original_exception=exc,
            ) from exc
        except Exception as exc:
            self._recent_failures.append(time.time())
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
        poll_failures = 0
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                self._recent_timeouts.append(time.time())
                raise RunPodError(
                    message=f"RunPod job timed out after {timeout}s",
                    detail=f"job_id={job_id}",
                )

            await asyncio.sleep(POLL_INTERVAL)

            try:
                status_resp = await client.get(status_url, timeout=10.0)
                status_resp.raise_for_status()
                status_data = status_resp.json()
            except Exception as poll_exc:
                poll_failures += 1
                if poll_failures >= 10:
                    raise RunPodError(
                        message=f"RunPod polling failed {poll_failures} consecutive times",
                        detail=f"job_id={job_id}, last_error={poll_exc}",
                    )
                logger.debug(f"RunPod poll transient error (attempt {poll_failures}): {poll_exc}")
                continue

            poll_failures = 0  # Reset on successful poll
            status = status_data.get("status", "")

            if status == "COMPLETED":
                output = status_data.get("output", {})
                wall_clock_ms = int((time.perf_counter() - start) * 1000)
                # Prefer RunPod-reported GPU execution time over wall-clock time
                gpu_exec_ms = status_data.get("executionTime")
                if gpu_exec_ms is not None:
                    execution_ms = int(gpu_exec_ms)
                else:
                    execution_ms = wall_clock_ms
                execution_s = execution_ms / 1000.0
                output.setdefault("execution_time_ms", execution_ms)
                output["job_id"] = job_id

                # Record metrics per task type
                self._recent_render_times.append(execution_s)
                if task_type == "preprocess_face":
                    self._recent_preprocess_times.append(execution_s)
                elif task_type == "render_lipsync":
                    self._recent_lipsync_times.append(execution_s)

                # Record R2 upload latency if worker reported it
                r2_upload_ms = output.get("r2_upload_ms")
                if r2_upload_ms is not None:
                    self._recent_r2_latencies.append(r2_upload_ms / 1000.0)

                log_with_latency(
                    logger, "RunPod job completed", execution_ms,
                    extra={"job_id": job_id, "task_type": task_type},
                )
                return output

            elif status == "FAILED":
                self._recent_failures.append(time.time())
                error = status_data.get("error", "Unknown error")
                raise RunPodError(
                    message=f"RunPod job failed: {error}",
                    detail=f"job_id={job_id}",
                )

            elif status in ("CANCELLED", "TIMED_OUT"):
                if status == "TIMED_OUT":
                    self._recent_timeouts.append(time.time())
                else:
                    self._recent_failures.append(time.time())
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
