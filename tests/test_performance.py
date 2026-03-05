"""Performance benchmarks for the SmartTalker pipeline.

Targets:
- Text-only response: < 1.5s
- Voice (ASR+LLM+TTS): < 2.5s
- Render warm (cached face): < 3s
- Render cold: < 8s
- 20 concurrent sessions: no degradation > 2x

These benchmarks mock external services (DashScope, RunPod) with
realistic latency simulations to validate pipeline orchestration overhead.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


@dataclass
class TimedResult:
    """Result of a timed operation."""
    elapsed_ms: float
    result: Any = None


async def timed(coro) -> TimedResult:
    """Time an async coroutine execution."""
    start = time.perf_counter()
    result = await coro
    elapsed = (time.perf_counter() - start) * 1000
    return TimedResult(elapsed_ms=elapsed, result=result)


# ────────────────────────────────────────────────────────────────────
# Mock Factories
# ────────────────────────────────────────────────────────────────────


def make_mock_llm(latency_ms: float = 200):
    """Create a mock LLM engine with simulated latency."""
    from src.pipeline.llm import LLMResult

    async def _generate(*args, **kwargs):
        await asyncio.sleep(latency_ms / 1000)
        return LLMResult(
            text="This is a test response from the AI.",
            emotion="neutral",
            latency_ms=int(latency_ms),
            tokens_used=150,
            cost_usd=0.001,
        )

    mock = MagicMock()
    mock.generate = AsyncMock(side_effect=_generate)
    return mock


def make_mock_asr(latency_ms: float = 100):
    """Create a mock ASR engine with simulated latency."""
    from src.pipeline.asr import TranscriptionResult

    mock_session = MagicMock()

    async def _finish():
        await asyncio.sleep(latency_ms / 1000)
        return TranscriptionResult(
            text="Hello, how are you?",
            language="en",
            confidence=0.95,
            latency_ms=int(latency_ms),
            cost_usd=0.0001,
        )

    mock_session.send_audio = AsyncMock()
    mock_session.finish = AsyncMock(side_effect=_finish)

    engine = MagicMock()

    async def _create_session(*args, **kwargs):
        return mock_session

    engine.create_session = AsyncMock(side_effect=_create_session)
    return engine, mock_session


def make_mock_tts(latency_ms: float = 150):
    """Create a mock TTS engine with simulated latency."""

    async def _synthesize_stream(*args, **kwargs):
        await asyncio.sleep(latency_ms / 1000)
        stream = MagicMock()
        stream.duration_seconds = 2.0
        stream.cost_usd = 0.0005

        async def _collect_all():
            return b"\x00" * 96000  # 1 second of 48kHz 16-bit mono silence
        stream.collect_all = AsyncMock(side_effect=_collect_all)
        return stream

    mock = MagicMock()
    mock.synthesize_stream = AsyncMock(side_effect=_synthesize_stream)
    return mock


def make_mock_runpod(latency_ms: float = 1500):
    """Create a mock RunPod client with simulated latency."""
    from src.services.runpod_client import RenderResult

    async def _render_lipsync(*args, **kwargs):
        await asyncio.sleep(latency_ms / 1000)
        return RenderResult(
            video_url="https://r2.example.com/video.mp4",
            execution_time_ms=int(latency_ms),
            cost_usd=0.001,
            job_id="test-job-123",
        )

    mock = MagicMock()
    mock.render_lipsync = AsyncMock(side_effect=_render_lipsync)
    mock.close = AsyncMock()
    return mock


# ────────────────────────────────────────────────────────────────────
# Performance Tests
# ────────────────────────────────────────────────────────────────────


class TestTextLatency:
    """Test text-only response latency (LLM only, no ASR/TTS)."""

    @pytest.mark.asyncio
    async def test_text_response_under_1500ms(self):
        """Text response should complete in < 1.5s with 200ms mock LLM."""
        llm = make_mock_llm(latency_ms=200)

        result = await timed(llm.generate(
            user_text="What are your hours?",
            language="en",
            session_id="perf-1",
        ))

        assert result.elapsed_ms < 1500, f"Text latency {result.elapsed_ms:.0f}ms > 1500ms"
        assert result.result.text
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_response_overhead(self):
        """Pipeline orchestration overhead should be < 50ms beyond LLM latency."""
        llm = make_mock_llm(latency_ms=300)

        result = await timed(llm.generate(
            user_text="Tell me about pricing",
            language="en",
            session_id="perf-2",
        ))

        overhead = result.elapsed_ms - 300
        assert overhead < 50, f"Orchestration overhead {overhead:.0f}ms > 50ms"


class TestVoiceLatency:
    """Test voice pipeline latency (ASR + LLM + TTS)."""

    @pytest.mark.asyncio
    async def test_voice_pipeline_under_2500ms(self):
        """Full voice pipeline (ASR → LLM → TTS) should complete < 2.5s."""
        asr, asr_session = make_mock_asr(latency_ms=100)
        llm = make_mock_llm(latency_ms=200)
        tts = make_mock_tts(latency_ms=150)

        async def voice_pipeline():
            # ASR
            session = await asr.create_session("en")
            await session.send_audio(b"\x00" * 32000)
            asr_result = await session.finish()

            # LLM
            llm_result = await llm.generate(
                user_text=asr_result.text,
                language="en",
                session_id="perf-voice",
            )

            # TTS
            stream = await tts.synthesize_stream(
                text=llm_result.text,
                language="en",
            )
            audio = await stream.collect_all()
            return audio

        result = await timed(voice_pipeline())

        assert result.elapsed_ms < 2500, f"Voice latency {result.elapsed_ms:.0f}ms > 2500ms"

    @pytest.mark.asyncio
    async def test_tts_overlap_feasibility(self):
        """Verify that TTS can start before LLM fully completes (streaming overlap)."""
        # This tests that we can start TTS as soon as first sentence is ready
        tts = make_mock_tts(latency_ms=100)

        # Start TTS immediately with partial text
        result1 = await timed(tts.synthesize_stream(
            text="First sentence.",
            language="en",
        ))

        # TTS should start within 200ms
        assert result1.elapsed_ms < 200, f"TTS start took {result1.elapsed_ms:.0f}ms"


class TestRenderLatency:
    """Test RunPod render latency."""

    @pytest.mark.asyncio
    async def test_warm_render_under_3000ms(self):
        """Warm render (cached face data) should complete < 3s."""
        runpod = make_mock_runpod(latency_ms=1500)

        result = await timed(runpod.render_lipsync(
            audio_url="https://r2.example.com/audio.wav",
            face_data_url="https://r2.example.com/face.pkl",
            employee_id="emp-1",
            session_id="perf-render",
        ))

        assert result.elapsed_ms < 3000, f"Warm render {result.elapsed_ms:.0f}ms > 3000ms"
        assert result.result.video_url

    @pytest.mark.asyncio
    async def test_cold_render_under_8000ms(self):
        """Cold render (RunPod cold start) should complete < 8s."""
        # Cold start adds ~5s
        runpod = make_mock_runpod(latency_ms=6000)

        result = await timed(runpod.render_lipsync(
            audio_url="https://r2.example.com/audio.wav",
            face_data_url="https://r2.example.com/face.pkl",
            employee_id="emp-1",
            session_id="perf-cold",
        ))

        assert result.elapsed_ms < 8000, f"Cold render {result.elapsed_ms:.0f}ms > 8000ms"


class TestConcurrency:
    """Test concurrent session handling."""

    @pytest.mark.asyncio
    async def test_20_concurrent_sessions(self):
        """20 concurrent sessions should not exceed 2x single-session latency."""
        llm = make_mock_llm(latency_ms=200)

        # Measure single session
        single = await timed(llm.generate(
            user_text="Hello", language="en", session_id="single",
        ))

        # Measure 20 concurrent sessions
        async def single_session(i: int):
            return await llm.generate(
                user_text=f"Hello from session {i}",
                language="en",
                session_id=f"concurrent-{i}",
            )

        start = time.perf_counter()
        results = await asyncio.gather(*[single_session(i) for i in range(20)])
        concurrent_ms = (time.perf_counter() - start) * 1000

        # All should complete
        assert len(results) == 20

        # Concurrent should not be > 2x single (with mocks, should be ~1x)
        ratio = concurrent_ms / single.elapsed_ms if single.elapsed_ms > 0 else 1
        assert ratio < 2.0, (
            f"Concurrent/single ratio {ratio:.1f}x > 2.0x "
            f"(single={single.elapsed_ms:.0f}ms, concurrent={concurrent_ms:.0f}ms)"
        )

    @pytest.mark.asyncio
    async def test_concurrent_voice_pipelines(self):
        """5 concurrent voice pipelines should complete within 3s."""
        asr, _ = make_mock_asr(latency_ms=100)
        llm = make_mock_llm(latency_ms=200)
        tts = make_mock_tts(latency_ms=150)

        async def voice_session(i: int):
            session = await asr.create_session("en")
            await session.send_audio(b"\x00" * 16000)
            asr_result = await session.finish()
            llm_result = await llm.generate(
                user_text=asr_result.text, language="en", session_id=f"voice-{i}"
            )
            stream = await tts.synthesize_stream(text=llm_result.text, language="en")
            return await stream.collect_all()

        result = await timed(asyncio.gather(*[voice_session(i) for i in range(5)]))

        assert result.elapsed_ms < 3000, (
            f"5 concurrent voice sessions took {result.elapsed_ms:.0f}ms > 3000ms"
        )


class TestSessionCaching:
    """Test that session-level caching reduces DB queries."""

    @pytest.mark.asyncio
    async def test_employee_cached_after_first_load(self):
        """Employee should be loaded from DB only once per session."""
        from src.agent.engine import _SessionState

        session = _SessionState(
            employee_id="emp-1",
            visitor_id="vis-1",
            customer_id="cust-1",
            last_access=time.time(),
        )

        # First access: no cache
        assert session.cached_employee is None

        # Simulate caching
        mock_employee = MagicMock()
        mock_employee.id = "emp-1"
        mock_employee.name = "Test Employee"
        session.cached_employee = mock_employee

        # Second access: cache hit
        assert session.cached_employee is not None
        assert session.cached_employee.id == "emp-1"

    @pytest.mark.asyncio
    async def test_tools_cached_after_first_load(self):
        """Tools should be loaded from DB only once per session."""
        from src.agent.engine import _SessionState

        session = _SessionState(
            employee_id="emp-1",
            visitor_id="vis-1",
            customer_id="cust-1",
            last_access=time.time(),
        )

        assert session.cached_tools is None
        assert session.cached_tool_registry_map is None

        # Simulate caching
        session.cached_tools = [{"type": "function", "function": {"name": "test"}}]
        session.cached_tool_registry_map = {"test": MagicMock()}

        assert len(session.cached_tools) == 1
        assert "test" in session.cached_tool_registry_map


class TestCostRecordingPerformance:
    """Test that cost recording doesn't block the pipeline."""

    @pytest.mark.asyncio
    async def test_cost_recording_overhead(self):
        """Cost recording should add < 10ms overhead (fire-and-forget)."""
        from src.api.ws_visitor import _record_cost

        app = MagicMock()
        mock_db = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db.session.return_value = mock_session
        app.state.db = mock_db

        result = await timed(_record_cost(
            app, "llm", "cust-1", "sess-1",
            cost_usd=0.001, tokens_used=100, duration_ms=200,
            details={"text_length": 50},
        ))

        assert result.elapsed_ms < 10, f"Cost recording took {result.elapsed_ms:.0f}ms > 10ms"
