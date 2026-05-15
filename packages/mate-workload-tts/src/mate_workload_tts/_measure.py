"""Measurement loop for the TTS workload."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Protocol


class _TtsResult(Protocol):
    audio_duration_s: float
    processing_time_s: float


class _TtsEngine(Protocol):
    def synthesize(self, text: str, model: str, voice: str) -> _TtsResult: ...


@dataclass
class _RunStats:
    rtf: float
    chars_per_second: float
    total_audio_seconds: float


def _aggregate_run(results: list[_TtsResult], texts: list[str]) -> _RunStats:
    """Compute RTF, chars/s and total audio from a single run's results."""
    total_processing = sum(r.processing_time_s for r in results)
    total_audio = sum(r.audio_duration_s for r in results)
    total_chars = sum(len(t) for t in texts)
    rtf = total_processing / max(total_audio, 1e-9)
    chars_per_second = total_chars / max(total_processing, 1e-9)
    return _RunStats(rtf=rtf, chars_per_second=chars_per_second, total_audio_seconds=total_audio)


def measure(
    engine: _TtsEngine,
    model: str,
    voice: str,
    sentences: list[dict[str, Any]],
    runs: int,
    warmup_runs: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Run the TTS benchmark loop; return (median, std_dev, throttling_detected)."""
    texts = [s["text"] for s in sentences]
    all_stats: list[_RunStats] = []

    for i in range(warmup_runs + runs):
        run_results: list[_TtsResult] = []
        for text in texts:
            result = engine.synthesize(text=text, model=model, voice=voice)
            run_results.append(result)
        if i >= warmup_runs:
            all_stats.append(_aggregate_run(run_results, texts))

    rtf_values = [s.rtf for s in all_stats]
    cps_values = [s.chars_per_second for s in all_stats]
    audio_values = [s.total_audio_seconds for s in all_stats]

    median_rtf = statistics.median(rtf_values) if rtf_values else 0.0
    median_cps = statistics.median(cps_values) if cps_values else 0.0
    median_audio = statistics.median(audio_values) if audio_values else 0.0
    std_rtf = statistics.stdev(rtf_values) if len(rtf_values) > 1 else 0.0
    std_cps = statistics.stdev(cps_values) if len(cps_values) > 1 else 0.0

    cv = std_rtf / median_rtf if median_rtf > 0 else 0.0
    throttling_detected = cv > 0.15

    median_stats: dict[str, Any] = {
        "rtf": median_rtf,
        "chars_per_second": median_cps,
        "total_audio_seconds": median_audio,
    }
    std_dev_stats: dict[str, Any] = {
        "rtf": std_rtf,
        "chars_per_second": std_cps,
    }
    return median_stats, std_dev_stats, throttling_detected
