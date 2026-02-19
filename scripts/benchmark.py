"""Benchmark script for SmartTalker pipeline latency.

Measures per-layer and end-to-end latency for each pipeline
component. Run with: python scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_section(name: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")


def format_latency(values: list[float]) -> str:
    """Format latency statistics."""
    if not values:
        return "N/A"
    avg = statistics.mean(values)
    p50 = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 3 else max(values)
    return f"avg={avg:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms"


async def benchmark_llm(iterations: int = 5) -> Optional[list[float]]:
    """Benchmark LLM generation latency."""
    benchmark_section("LLM (Qwen 2.5 via Ollama)")

    try:
        from src.config import get_settings
        from src.pipeline.llm import LLMEngine

        config = get_settings()
        engine = LLMEngine(config)

        prompts = [
            "Hello, how are you?",
            "What is the capital of Oman?",
            "Tell me about bus travel in the Middle East.",
            "Translate 'Good morning' to Arabic.",
            "What are popular destinations in MENA?",
        ]

        latencies: list[float] = []

        for i in range(iterations):
            prompt = prompts[i % len(prompts)]
            start = time.perf_counter()
            try:
                result = await engine.generate(user_text=prompt, language="en")
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
                print(f"  [{i+1}/{iterations}] {elapsed:.0f}ms — {result.text[:60]}...")
            except Exception as exc:
                print(f"  [{i+1}/{iterations}] FAILED: {exc}")

        await engine.close()

        if latencies:
            print(f"\n  Results: {format_latency(latencies)}")
            return latencies

    except Exception as exc:
        print(f"  SKIPPED: {exc}")

    return None


def benchmark_emotion(iterations: int = 10) -> Optional[list[float]]:
    """Benchmark emotion detection latency."""
    benchmark_section("Emotion Detection")

    try:
        from src.config import get_settings
        from src.pipeline.emotions import EmotionEngine

        config = get_settings()
        engine = EmotionEngine(config)
        engine.load()

        test_texts = [
            "\u0634\u0643\u0631\u0627 \u062c\u0632\u064a\u0644\u0627",
            "I'm so sorry about that",
            "This is amazing!",
            "\u0623\u0646\u0627 \u063a\u0627\u0636\u0628 \u062c\u062f\u0627",
            "What time is the next bus?",
        ]

        latencies: list[float] = []

        for i in range(iterations):
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            result = engine.detect_from_text(text)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            print(f"  [{i+1}/{iterations}] {elapsed:.1f}ms — {result.primary_emotion} ({result.confidence:.2f})")

        engine.unload()

        if latencies:
            print(f"\n  Results: {format_latency(latencies)}")
            return latencies

    except Exception as exc:
        print(f"  SKIPPED: {exc}")

    return None


async def benchmark_pipeline(iterations: int = 3) -> Optional[list[float]]:
    """Benchmark full text pipeline (LLM + TTS)."""
    benchmark_section("Full Pipeline (LLM + TTS)")

    try:
        from src.config import get_settings
        from src.pipeline.orchestrator import SmartTalkerPipeline

        config = get_settings()
        pipeline = SmartTalkerPipeline(config)
        pipeline.load_all()

        prompts = [
            "Hello, I'd like to book a bus ticket.",
            "When is the next bus to Muscat?",
            "How much does a ticket cost?",
        ]

        latencies: list[float] = []

        for i in range(iterations):
            prompt = prompts[i % len(prompts)]
            start = time.perf_counter()
            try:
                result = await pipeline.process_text(
                    text=prompt,
                    language="en",
                    enable_video=False,
                )
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
                print(f"  [{i+1}/{iterations}] {elapsed:.0f}ms — breakdown: {result.breakdown}")
            except Exception as exc:
                print(f"  [{i+1}/{iterations}] FAILED: {exc}")

        await pipeline.unload_all()

        if latencies:
            print(f"\n  Results: {format_latency(latencies)}")
            return latencies

    except Exception as exc:
        print(f"  SKIPPED: {exc}")

    return None


async def main() -> None:
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print(" SmartTalker Pipeline Benchmark")
    print(f" Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results: dict[str, Optional[list[float]]] = {}

    results["emotion"] = benchmark_emotion()
    results["llm"] = await benchmark_llm()
    results["pipeline"] = await benchmark_pipeline()

    # ── Summary ──────────────────────────────────────────────────────────
    benchmark_section("Summary")
    for name, latencies in results.items():
        if latencies:
            print(f"  {name:12s}: {format_latency(latencies)}")
        else:
            print(f"  {name:12s}: SKIPPED")

    print(f"\n{'=' * 60}")
    print(" Benchmark complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
