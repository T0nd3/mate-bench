from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscribeResult:
    text: str
    audio_duration_s: float
    processing_time_s: float

    @property
    def rtf(self) -> float:
        return self.processing_time_s / self.audio_duration_s if self.audio_duration_s > 0 else 0.0


def transcribe(model_obj: object, audio_path: Path, language: str = "en") -> TranscribeResult:
    """Run inference via a loaded WhisperModel; return text + timing."""
    t0 = time.perf_counter()
    segments, info = model_obj.transcribe(  # type: ignore[attr-defined]
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=False,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    processing_time = time.perf_counter() - t0
    return TranscribeResult(
        text=text,
        audio_duration_s=info.duration,
        processing_time_s=processing_time,
    )
