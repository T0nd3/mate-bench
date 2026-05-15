"""Piper TTS synthesis helpers."""
from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass


@dataclass
class TtsResult:
    """Result of a single Piper TTS synthesis call."""

    text: str
    audio_duration_s: float
    processing_time_s: float

    @property
    def rtf(self) -> float:
        """Real-Time Factor: processing_time / audio_duration (lower is better)."""
        return self.processing_time_s / max(self.audio_duration_s, 1e-9)

    @property
    def chars_per_second(self) -> float:
        """Characters synthesised per second of processing time."""
        return len(self.text) / max(self.processing_time_s, 1e-9)


def _wav_duration(wav_bytes: bytes) -> float:
    """Parse a WAV byte string and return its duration in seconds."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes() / wf.getframerate()


def synthesize(voice: object, text: str) -> TtsResult:
    """Synthesize text with a PiperVoice and return timing stats.

    Args:
        voice: A piper.PiperVoice instance.
        text: The text to synthesize.

    Returns:
        TtsResult with audio_duration_s and processing_time_s.
    """
    buf = io.BytesIO()
    t0 = time.perf_counter()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize(text, wav_file)  # type: ignore[union-attr]
    elapsed = time.perf_counter() - t0

    wav_bytes = buf.getvalue()
    audio_duration_s = _wav_duration(wav_bytes) if wav_bytes else 0.0

    return TtsResult(text=text, audio_duration_s=audio_duration_s, processing_time_s=elapsed)
