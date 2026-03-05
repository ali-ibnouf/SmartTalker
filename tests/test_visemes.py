"""Tests for VisemeExtractor."""

from __future__ import annotations

import pytest

from src.pipeline.visemes import LipParams, PHONEME_DURATION, VISEME_TO_VRM, VISEME_WEIGHTS, VisemeExtractor, VisemeFrame


class TestVisemeExtractFromText:
    """Tests for extract_from_text."""

    def test_empty_text_returns_silent(self):
        result = VisemeExtractor.extract_from_text("")
        assert result.viseme_sequence == ["SILENT"]
        assert result.weights == [0.0]

    def test_whitespace_only_returns_silent(self):
        result = VisemeExtractor.extract_from_text("   ")
        assert result.viseme_sequence == ["SILENT"]

    def test_english_word(self):
        result = VisemeExtractor.extract_from_text("hello")
        assert isinstance(result, LipParams)
        assert len(result.viseme_sequence) > 0
        assert "SILENT" not in result.viseme_sequence or result.viseme_sequence == ["SILENT"]

    def test_arabic_word(self):
        result = VisemeExtractor.extract_from_text("\u0645\u0631\u062d\u0628\u0627")
        assert len(result.viseme_sequence) > 0
        # meem -> M, ra -> R, ha -> AH, ba -> M, alef -> AA
        assert "M" in result.viseme_sequence

    def test_consecutive_duplicates_removed(self):
        """Consecutive identical visemes should be deduplicated."""
        result = VisemeExtractor.extract_from_text("mm")
        # Both 'm' map to M, so only one M should appear
        assert result.viseme_sequence.count("M") == 1

    def test_punctuation_stripped(self):
        result = VisemeExtractor.extract_from_text("hello!")
        # Should produce same as "hello"
        result_clean = VisemeExtractor.extract_from_text("hello")
        assert result.viseme_sequence == result_clean.viseme_sequence

    def test_weights_match_sequence_length(self):
        result = VisemeExtractor.extract_from_text("test word")
        assert len(result.viseme_sequence) == len(result.weights)

    def test_to_dict(self):
        result = VisemeExtractor.extract_from_text("hi")
        d = result.to_dict()
        assert "viseme_sequence" in d
        assert "weights" in d


class TestVisemeExtractTimed:
    """Tests for extract_timed."""

    def test_timed_returns_frames(self):
        frames = VisemeExtractor.extract_timed("hello", 500)
        assert len(frames) > 0
        assert all(isinstance(f, VisemeFrame) for f in frames)

    def test_timed_covers_duration(self):
        frames = VisemeExtractor.extract_timed("hello world", 1000)
        if frames:
            assert frames[0].start_ms == 0
            assert frames[-1].end_ms == 1000

    def test_timed_empty_text(self):
        frames = VisemeExtractor.extract_timed("", 500)
        # Empty text -> SILENT -> 1 frame
        assert len(frames) == 1
        assert frames[0].viseme == "SILENT"

    def test_timed_ordering(self):
        frames = VisemeExtractor.extract_timed("testing", 700)
        for i in range(1, len(frames)):
            assert frames[i].start_ms >= frames[i - 1].start_ms


class TestVisemeToVRM:
    """Tests for VISEME_TO_VRM mapping."""

    def test_all_viseme_weights_have_vrm_mapping(self):
        """Every viseme in VISEME_WEIGHTS should have a VRM mapping."""
        for viseme_name in VISEME_WEIGHTS:
            assert viseme_name in VISEME_TO_VRM, f"Missing VRM mapping for {viseme_name}"

    def test_vrm_names_are_valid(self):
        """VRM mapped names should be valid VRM expression presets."""
        valid_vrm = {"neutral", "aa", "ih", "ou", "ee", "oh"}
        for viseme, vrm_name in VISEME_TO_VRM.items():
            assert vrm_name in valid_vrm, f"Invalid VRM name '{vrm_name}' for viseme '{viseme}'"

    def test_vowel_visemes_map_to_vowel_vrm(self):
        """Open-mouth visemes should map to open VRM shapes."""
        assert VISEME_TO_VRM["AH"] == "aa"
        assert VISEME_TO_VRM["AA"] == "aa"
        assert VISEME_TO_VRM["EE"] == "ee"
        assert VISEME_TO_VRM["OH"] == "oh"
        assert VISEME_TO_VRM["OO"] == "ou"


class TestExtractFromWordTimings:
    """Tests for extract_from_word_timings."""

    def test_basic_word_timings(self):
        """Extracts visemes from word timing data."""
        timings = [
            {"word": "hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.3, "end": 0.6},
        ]
        frames = VisemeExtractor.extract_from_word_timings(timings, 0, 600)
        assert len(frames) > 0
        assert all(isinstance(f, VisemeFrame) for f in frames)
        # Visemes should be VRM names (aa, ee, oh, etc.)
        valid_vrm = {"neutral", "aa", "ih", "ou", "ee", "oh"}
        for f in frames:
            assert f.viseme in valid_vrm

    def test_words_outside_window_skipped(self):
        """Words entirely outside the chunk window are ignored."""
        timings = [
            {"word": "first", "start": 0.0, "end": 0.2},
            {"word": "second", "start": 0.5, "end": 0.8},
        ]
        # Only request chunk 0-200ms — should only get "first"
        frames = VisemeExtractor.extract_from_word_timings(timings, 0, 200)
        # Should have frames only from "first"
        assert len(frames) > 0
        assert all(f.end_ms <= 200 for f in frames)

    def test_empty_timings_returns_neutral(self):
        """No word timings → single neutral frame."""
        frames = VisemeExtractor.extract_from_word_timings([], 0, 200)
        assert len(frames) == 1
        assert frames[0].viseme == "neutral"
        assert frames[0].weight == 0.0

    def test_timing_clipped_to_chunk(self):
        """Words spanning chunk boundary are clipped."""
        timings = [
            {"word": "hello", "start": 0.1, "end": 0.4},
        ]
        # Chunk is 200-400ms — word starts at 100ms, ends at 400ms
        frames = VisemeExtractor.extract_from_word_timings(timings, 200, 400)
        assert len(frames) > 0
        # All frames should be relative to chunk start (0 to 200ms)
        for f in frames:
            assert f.start_ms >= 0
            assert f.end_ms <= 200

    def test_frame_ordering(self):
        """Frames should be in chronological order."""
        timings = [
            {"word": "one", "start": 0.0, "end": 0.2},
            {"word": "two", "start": 0.2, "end": 0.4},
            {"word": "three", "start": 0.4, "end": 0.6},
        ]
        frames = VisemeExtractor.extract_from_word_timings(timings, 0, 600)
        for i in range(1, len(frames)):
            assert frames[i].start_ms >= frames[i - 1].start_ms


class TestPhonemeDurationWeighting:
    """Tests for phoneme-aware duration weighting."""

    def test_phoneme_duration_table_covers_all_visemes(self):
        """Every viseme in VISEME_WEIGHTS should have a duration factor."""
        for v in VISEME_WEIGHTS:
            assert v in PHONEME_DURATION, f"Missing PHONEME_DURATION for {v}"

    def test_vowels_longer_than_consonants(self):
        """Vowel phonemes should have longer duration than stop consonants."""
        assert PHONEME_DURATION["AH"] > PHONEME_DURATION["M"]
        assert PHONEME_DURATION["AA"] > PHONEME_DURATION["K"]
        assert PHONEME_DURATION["OH"] > PHONEME_DURATION["N"]

    def test_weighted_durations_not_uniform(self):
        """Phoneme-weighted viseme durations should differ from even distribution."""
        timings = [{"word": "hello", "start": 0.0, "end": 0.5}]
        frames = VisemeExtractor.extract_from_word_timings(timings, 0, 500)
        durations = [f.end_ms - f.start_ms for f in frames]
        # With weighted distribution, not all durations should be equal
        if len(durations) > 1:
            assert len(set(durations)) > 1, "Durations should not all be equal"

    def test_consonant_visemes_map_to_nonsilent(self):
        """Consonant visemes M and N should map to non-neutral VRM shapes."""
        assert VISEME_TO_VRM["M"] != "neutral"
        assert VISEME_TO_VRM["N"] != "neutral"


class TestArabicVisemes:
    """Tests for Arabic text handling."""

    def test_arabic_with_diacritics(self):
        """Arabic text with harakat should produce valid visemes."""
        # marhaba with diacritics: م َ ر ْ ح َ ب ا
        text = "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u0627"
        result = VisemeExtractor.extract_from_text(text)
        assert len(result.viseme_sequence) > 0

    def test_arabic_vowel_marks_produce_visemes(self):
        """Arabic vowel diacritics (fatha/damma/kasra) should produce visemes."""
        # fatha → AH, damma → OO, kasra → EE
        result = VisemeExtractor.extract_from_text("\u064e\u064f\u0650")
        assert len(result.viseme_sequence) > 0
