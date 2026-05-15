"""Kokoro synthesis helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TtsResult:
    """Result of a single TTS synthesis call."""

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


def synthesize(pipeline: object, text: str, voice: str, sample_rate: int = 24000) -> TtsResult:
    """Synthesize text with a Kokoro pipeline and return timing stats.

    Args:
        pipeline: A KPipeline instance from the kokoro package.
        text: The text to synthesize.
        voice: Kokoro voice name, e.g. "af_heart".
        sample_rate: Audio sample rate in Hz (Kokoro default: 24000).

    Returns:
        TtsResult with audio_duration_s and processing_time_s.
    """
    import numpy as np

    t0 = time.perf_counter()
    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        audio_chunks.append(audio)
    elapsed = time.perf_counter() - t0

    if audio_chunks:
        combined = np.concatenate(audio_chunks)
        audio_duration_s = len(combined) / sample_rate
    else:
        audio_duration_s = 0.0

    return TtsResult(text=text, audio_duration_s=audio_duration_s, processing_time_s=elapsed)
