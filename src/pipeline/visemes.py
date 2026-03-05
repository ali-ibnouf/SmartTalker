"""Viseme extraction for lip animation hints.

Maps text characters/phonemes to viseme sequences with timing weights.
These are supplementary animation hints for the browser-side VRM avatar —
the frontend uses these alongside audio data for lip-sync animation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Viseme labels and their default mouth-openness weights (0.0 = closed, 1.0 = wide open)
VISEME_WEIGHTS: dict[str, float] = {
    "SILENT": 0.0,
    "M": 0.1,     # lips pressed (m, b, p)
    "F": 0.3,     # lower lip to upper teeth (f, v)
    "TH": 0.35,   # tongue between teeth (th)
    "S": 0.25,    # teeth together, air (s, z, ts)
    "SH": 0.3,    # wider teeth (sh, ch, j, zh)
    "N": 0.3,     # tongue to ridge (n, d, t, l)
    "R": 0.35,    # tongue back (r)
    "K": 0.4,     # back of tongue (k, g, ng)
    "EH": 0.5,    # mid open (e, short a)
    "AH": 0.7,    # wide open (a, ah)
    "AA": 0.8,    # widest (aa, open a)
    "EE": 0.4,    # spread lips (ee, i)
    "OH": 0.6,    # rounded lips (o)
    "OO": 0.5,    # tight round (oo, u, w)
}

# Character-to-viseme mapping (simplified — covers Latin + Arabic)
_CHAR_VISEME: dict[str, str] = {
    # English consonants
    "m": "M", "b": "M", "p": "M",
    "f": "F", "v": "F",
    "s": "S", "z": "S", "c": "S",
    "n": "N", "d": "N", "t": "N", "l": "N",
    "r": "R",
    "k": "K", "g": "K", "q": "K", "x": "K",
    "j": "SH", "y": "EE",
    "w": "OO", "h": "AH",
    # English vowels
    "a": "AH", "e": "EH", "i": "EE", "o": "OH", "u": "OO",
    # Arabic approximate mappings
    "\u0627": "AA",   # alef
    "\u0628": "M",    # ba
    "\u062a": "N",    # ta
    "\u062b": "TH",   # tha
    "\u062c": "SH",   # jeem
    "\u062d": "AH",   # ha
    "\u062e": "K",    # kha
    "\u062f": "N",    # dal
    "\u0630": "TH",   # dhal
    "\u0631": "R",    # ra
    "\u0632": "S",    # zay
    "\u0633": "S",    # seen
    "\u0634": "SH",   # sheen
    "\u0635": "S",    # sad
    "\u0636": "N",    # dad
    "\u0637": "N",    # ta (emphatic)
    "\u0638": "TH",   # dha (emphatic)
    "\u0639": "AH",   # ain
    "\u063a": "K",    # ghain
    "\u0641": "F",    # fa
    "\u0642": "K",    # qaf
    "\u0643": "K",    # kaf
    "\u0644": "N",    # lam
    "\u0645": "M",    # meem
    "\u0646": "N",    # noon
    "\u0647": "AH",   # ha
    "\u0648": "OO",   # waw
    "\u064a": "EE",   # ya
    # Arabic vowel marks
    "\u064e": "AH",   # fatha
    "\u064f": "OO",   # damma
    "\u0650": "EE",   # kasra
}

# Map internal viseme names → VRM expression preset names (5 vowel shapes)
VISEME_TO_VRM: dict[str, str] = {
    "SILENT": "neutral",
    "M": "ou",          # lips pressed together — slight lip rounding
    "F": "ih",          # lower lip to teeth — narrow opening
    "TH": "ee",         # tongue between teeth — narrow mouth
    "S": "ee",          # teeth together — narrow
    "SH": "ee",         # wider teeth — still narrow mouth
    "N": "ih",          # tongue to ridge — slight opening
    "R": "oh",          # tongue back — slightly rounded
    "K": "oh",          # back of tongue — slightly open
    "EH": "ee",         # mid open — spread
    "AH": "aa",         # wide open
    "AA": "aa",         # widest open
    "EE": "ee",         # spread lips
    "OH": "oh",         # rounded lips
    "OO": "ou",         # tight round
}

# Relative phoneme duration factors (consonants shorter, vowels longer).
# Values are relative to an average phoneme; used by extract_from_word_timings
# for weighted duration distribution within a word.
PHONEME_DURATION: dict[str, float] = {
    # Stop consonants — very brief
    "M": 0.4, "K": 0.35, "N": 0.4,
    # Fricatives — moderate
    "F": 0.45, "TH": 0.45, "S": 0.5, "SH": 0.55, "R": 0.55,
    # Vowels — longest
    "EH": 0.9, "AH": 1.0, "AA": 1.1, "EE": 0.85, "OH": 1.0, "OO": 0.9,
    # Silence
    "SILENT": 0.2,
}


@dataclass
class VisemeFrame:
    """Single viseme with timing."""

    viseme: str
    weight: float
    start_ms: int = 0
    end_ms: int = 0


@dataclass
class LipParams:
    """Lip animation parameters for a single audio chunk."""

    viseme_sequence: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "viseme_sequence": self.viseme_sequence,
            "weights": self.weights,
        }


class VisemeExtractor:
    """Extract viseme sequences from text for lip animation hints.

    These are supplementary hints for the browser-side VRM avatar.
    The frontend combines these with audio data for smooth lip-sync.
    """

    @staticmethod
    def extract_from_text(text: str, duration_ms: int = 200) -> LipParams:
        """Map text to a viseme sequence with weights.

        Args:
            text: Text segment corresponding to an audio chunk.
            duration_ms: Duration of the audio chunk in milliseconds.

        Returns:
            LipParams with viseme sequence and weights.
        """
        if not text or not text.strip():
            return LipParams(viseme_sequence=["SILENT"], weights=[0.0])

        # Strip whitespace and punctuation for viseme extraction
        clean = re.sub(r"[^\w\s]", "", text.strip().lower())
        if not clean:
            return LipParams(viseme_sequence=["SILENT"], weights=[0.0])

        visemes: list[str] = []
        weights: list[float] = []

        for char in clean:
            if char in (" ", "\t", "\n"):
                # Brief pause between words
                if not visemes or visemes[-1] != "SILENT":
                    visemes.append("SILENT")
                    weights.append(0.0)
                continue

            viseme = _CHAR_VISEME.get(char, "AH")
            weight = VISEME_WEIGHTS.get(viseme, 0.5)

            # Avoid consecutive duplicates
            if visemes and visemes[-1] == viseme:
                continue

            visemes.append(viseme)
            weights.append(weight)

        if not visemes:
            return LipParams(viseme_sequence=["SILENT"], weights=[0.0])

        return LipParams(viseme_sequence=visemes, weights=weights)

    @staticmethod
    def extract_timed(text: str, duration_ms: int) -> list[VisemeFrame]:
        """Extract visemes with per-frame timing.

        Distributes visemes evenly across the audio chunk duration.

        Args:
            text: Text for this audio chunk.
            duration_ms: Total duration in milliseconds.

        Returns:
            List of VisemeFrame with start/end timing.
        """
        params = VisemeExtractor.extract_from_text(text, duration_ms)
        n = len(params.viseme_sequence)
        if n == 0:
            return []

        frame_ms = duration_ms / n
        frames: list[VisemeFrame] = []

        for i, (vis, wt) in enumerate(zip(params.viseme_sequence, params.weights)):
            frames.append(VisemeFrame(
                viseme=vis,
                weight=wt,
                start_ms=int(i * frame_ms),
                end_ms=int((i + 1) * frame_ms),
            ))

        return frames

    @staticmethod
    def extract_from_word_timings(
        word_timings: list[dict],
        chunk_start_ms: int,
        chunk_end_ms: int,
    ) -> list[VisemeFrame]:
        """Extract visemes from TTS word-level timings within a time window.

        Uses real word boundaries from TTS and phoneme duration weighting
        for accurate viseme timing (consonants shorter, vowels longer).

        Args:
            word_timings: List of {"word": str, "start": float, "end": float} (seconds).
            chunk_start_ms: Start of the audio chunk in milliseconds.
            chunk_end_ms: End of the audio chunk in milliseconds.

        Returns:
            List of VisemeFrame with start/end timing relative to chunk start.
        """
        chunk_start_s = chunk_start_ms / 1000.0
        chunk_end_s = chunk_end_ms / 1000.0
        frames: list[VisemeFrame] = []

        for wt in word_timings:
            w_start = wt.get("start", 0.0)
            w_end = wt.get("end", 0.0)
            word = wt.get("word", "")

            # Skip words outside this chunk's time window
            if w_end <= chunk_start_s or w_start >= chunk_end_s:
                continue

            # Clip to chunk boundaries
            effective_start = max(w_start, chunk_start_s)
            effective_end = min(w_end, chunk_end_s)
            word_dur_ms = int((effective_end - effective_start) * 1000)
            if word_dur_ms <= 0:
                continue

            # Extract visemes for this word
            params = VisemeExtractor.extract_from_text(word)
            n = len(params.viseme_sequence)
            if n == 0:
                continue

            # Weighted duration distribution based on phoneme type
            raw_durations = [PHONEME_DURATION.get(v, 0.6) for v in params.viseme_sequence]
            total_weight = sum(raw_durations)
            if total_weight <= 0:
                total_weight = n  # fallback to even

            base_ms = int((effective_start - chunk_start_s) * 1000)
            cursor = 0.0

            for i, (vis, wt_val) in enumerate(zip(params.viseme_sequence, params.weights)):
                vrm_name = VISEME_TO_VRM.get(vis, "neutral")
                vis_dur = word_dur_ms * raw_durations[i] / total_weight
                start = base_ms + int(cursor)
                cursor += vis_dur
                end = base_ms + int(cursor)
                frames.append(VisemeFrame(
                    viseme=vrm_name,
                    weight=wt_val,
                    start_ms=start,
                    end_ms=end,
                ))

        if not frames:
            frames.append(VisemeFrame(viseme="neutral", weight=0.0, start_ms=0, end_ms=chunk_end_ms - chunk_start_ms))

        return frames
