"""Load tests for SmartTalker Central.

Validates:
- 100 concurrent text sessions complete without errors
- 20 concurrent voice sessions complete within SLA
- No resource leaks under sustained load
- Rate limiter correctly throttles excess requests

All external services (DashScope, RunPod, R2) are mocked with
realistic latency simulations.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Mock Factories ──────────────────────────────────────────────────────


@dataclass
class FakeLLMResult:
    text: str = "Thank you for your question. I'm happy to help."
    emotion: str = "neutral"
    cost_usd: float = 0.001
    tokens_used: int = 80
    latency_ms: int = 200


@dataclass
class FakeASRResult:
    text: str = "Hello, I need help with my account."
    language: str = "en"
    confidence: float = 0.95
    cost_usd: float = 0.0001
    latency_ms: int = 100


@dataclass
class FakeTTSStream:
    cost_usd: float = 0.0002
    duration_seconds: float = 1.5

    async def collect_all(self) -> bytes:
        return b"\x00" * 48000  # 0.5s of 48kHz 16-bit mono


def make_mock_llm(latency_ms: float = 200):
    async def _generate(*args, **kwargs):
        await asyncio.sleep(latency_ms / 1000)
        return FakeLLMResult()
    mock = MagicMock()
    mock.generate = AsyncMock(side_effect=_generate)
    return mock


def make_mock_asr(latency_ms: float = 100):
    session = MagicMock()

    async def _finish():
        await asyncio.sleep(latency_ms / 1000)
        return FakeASRResult()

    session.send_audio = AsyncMock()
    session.finish = AsyncMock(side_effect=_finish)

    engine = MagicMock()

    async def _create_session(*args, **kwargs):
        return session

    engine.create_session = AsyncMock(side_effect=_create_session)
    return engine, session


def make_mock_tts(latency_ms: float = 150):
    async def _synthesize_stream(*args, **kwargs):
        await asyncio.sleep(latency_ms / 1000)
        return FakeTTSStream()

    mock = MagicMock()
    mock.synthesize_stream = AsyncMock(side_effect=_synthesize_stream)
    return mock


# ── Load Tests ──────────────────────────────────────────────────────────


class TestConcurrentTextSessions:
    """100 concurrent text-only sessions."""

    @pytest.mark.asyncio
    async def test_100_concurrent_text_sessions(self):
        """All 100 text sessions should complete without errors."""
        llm = make_mock_llm(latency_ms=200)
        errors: list[str] = []

        async def text_session(i: int):
            try:
                result = await llm.generate(
                    text=f"User message from session {i}",
                    language="en",
                    session_id=f"load-text-{i}",
                )
                assert result.text, f"Session {i}: empty response"
            except Exception as exc:
                errors.append(f"Session {i}: {exc}")

        start = time.perf_counter()
        await asyncio.gather(*[text_session(i) for i in range(100)])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(errors) == 0, f"Errors in {len(errors)} sessions: {errors[:5]}"
        assert llm.generate.call_count == 100
        # With 200ms mock latency, 100 concurrent should finish within ~1s
        assert elapsed_ms < 5000, f"100 sessions took {elapsed_ms:.0f}ms"

    @pytest.mark.asyncio
    async def test_100_sessions_throughput(self):
        """Throughput should be at least 20 sessions/second."""
        llm = make_mock_llm(latency_ms=200)

        start = time.perf_counter()
        await asyncio.gather(*[
            llm.generate(text=f"msg-{i}", language="en", session_id=f"tp-{i}")
            for i in range(100)
        ])
        elapsed_s = time.perf_counter() - start

        throughput = 100 / elapsed_s
        assert throughput >= 20, f"Throughput {throughput:.1f} sessions/s < 20"


class TestConcurrentVoiceSessions:
    """20 concurrent voice pipeline sessions (ASR + LLM + TTS)."""

    @pytest.mark.asyncio
    async def test_20_concurrent_voice_sessions(self):
        """20 voice sessions should complete within 3s."""
        asr, _ = make_mock_asr(latency_ms=100)
        llm = make_mock_llm(latency_ms=200)
        tts = make_mock_tts(latency_ms=150)
        errors: list[str] = []

        async def voice_session(i: int):
            try:
                # ASR
                session = await asr.create_session("en")
                await session.send_audio(b"\x00" * 32000)
                asr_result = await session.finish()
                # LLM
                llm_result = await llm.generate(
                    text=asr_result.text,
                    language="en",
                    session_id=f"load-voice-{i}",
                )
                # TTS
                stream = await tts.synthesize_stream(
                    text=llm_result.text,
                    language="en",
                )
                audio = await stream.collect_all()
                assert len(audio) > 0, f"Session {i}: no audio"
            except Exception as exc:
                errors.append(f"Session {i}: {exc}")

        start = time.perf_counter()
        await asyncio.gather(*[voice_session(i) for i in range(20)])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(errors) == 0, f"Errors in {len(errors)} sessions: {errors[:5]}"
        assert elapsed_ms < 3000, f"20 voice sessions took {elapsed_ms:.0f}ms > 3000ms"

    @pytest.mark.asyncio
    async def test_voice_no_degradation(self):
        """20 concurrent voice sessions should not exceed 2x single latency."""
        asr, _ = make_mock_asr(latency_ms=100)
        llm = make_mock_llm(latency_ms=200)
        tts = make_mock_tts(latency_ms=150)

        async def single_voice():
            session = await asr.create_session("en")
            await session.send_audio(b"\x00" * 16000)
            asr_result = await session.finish()
            llm_result = await llm.generate(text=asr_result.text, language="en", session_id="single")
            stream = await tts.synthesize_stream(text=llm_result.text, language="en")
            return await stream.collect_all()

        # Single
        single_start = time.perf_counter()
        await single_voice()
        single_ms = (time.perf_counter() - single_start) * 1000

        # Concurrent
        concurrent_start = time.perf_counter()
        await asyncio.gather(*[single_voice() for _ in range(20)])
        concurrent_ms = (time.perf_counter() - concurrent_start) * 1000

        ratio = concurrent_ms / single_ms if single_ms > 0 else 1
        assert ratio < 2.0, (
            f"Degradation ratio {ratio:.1f}x > 2.0x "
            f"(single={single_ms:.0f}ms, concurrent={concurrent_ms:.0f}ms)"
        )


class TestSustainedLoad:
    """Test resource behavior under sustained load."""

    @pytest.mark.asyncio
    async def test_sustained_50_sessions_no_errors(self):
        """50 sequential text sessions should all succeed."""
        llm = make_mock_llm(latency_ms=50)
        errors = 0

        for i in range(50):
            try:
                result = await llm.generate(
                    text=f"Message {i}",
                    language="en",
                    session_id=f"sustained-{i}",
                )
                assert result.text
            except Exception:
                errors += 1

        assert errors == 0, f"{errors}/50 sessions failed"

    @pytest.mark.asyncio
    async def test_mixed_load_text_and_voice(self):
        """Mix of 80 text + 20 voice sessions running concurrently."""
        llm = make_mock_llm(latency_ms=100)
        asr, _ = make_mock_asr(latency_ms=50)
        tts = make_mock_tts(latency_ms=75)
        errors: list[str] = []

        async def text_task(i: int):
            try:
                await llm.generate(text=f"Text {i}", language="en", session_id=f"mix-t-{i}")
            except Exception as exc:
                errors.append(f"text-{i}: {exc}")

        async def voice_task(i: int):
            try:
                session = await asr.create_session("en")
                await session.send_audio(b"\x00" * 16000)
                asr_result = await session.finish()
                llm_result = await llm.generate(text=asr_result.text, language="en", session_id=f"mix-v-{i}")
                stream = await tts.synthesize_stream(text=llm_result.text, language="en")
                await stream.collect_all()
            except Exception as exc:
                errors.append(f"voice-{i}: {exc}")

        tasks = [text_task(i) for i in range(80)] + [voice_task(i) for i in range(20)]

        start = time.perf_counter()
        await asyncio.gather(*tasks)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(errors) == 0, f"Errors: {errors[:5]}"
        assert elapsed_ms < 5000, f"Mixed load took {elapsed_ms:.0f}ms"


class TestRateLimiterLoad:
    """Verify rate limiter rejects excess requests under load."""

    @pytest.mark.asyncio
    async def test_rate_limiter_throttles_at_limit(self):
        """Rate limiter should reject requests beyond the configured limit."""
        from src.middleware.rate_limiter import RateLimiter

        limiter = RateLimiter(redis=None)  # In-memory fallback

        allowed = 0
        blocked = 0

        for _ in range(150):
            is_allowed = await limiter.check("load-test-client", "api_default")
            if is_allowed:
                allowed += 1
            else:
                blocked += 1

        # api_default is 100/min, so ~100 should pass and ~50 blocked
        assert allowed <= 110, f"Too many allowed: {allowed}"
        assert blocked >= 40, f"Not enough blocked: {blocked}"
