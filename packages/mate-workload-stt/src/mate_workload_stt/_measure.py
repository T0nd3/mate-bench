from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class _TranscribeResult(Protocol):
    text: str
    audio_duration_s: float
    processing_time_s: float


class _SttEngine(Protocol):
    def transcribe(self, audio_path: Path, model: str) -> _TranscribeResult: ...


@dataclass
class _RunStats:
    total_audio_s: float
    total_processing_s: float
    rtf: float
    wer: float


def _word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """Word Error Rate over a corpus (lower is better)."""
    total_errors = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        r = ref.upper().split()
        h = hyp.upper().split()
        total_errors += _word_edit_distance(r, h)
        total_words += len(r)
    return total_errors / max(total_words, 1)


def _aggregate_run(
    results: list[_TranscribeResult],
    references: list[str],
) -> _RunStats:
    total_audio = sum(r.audio_duration_s for r in results)
    total_proc = sum(r.processing_time_s for r in results)
    rtf = total_proc / total_audio if total_audio > 0 else 0.0
    wer = corpus_wer(references, [r.text for r in results])
    return _RunStats(
        total_audio_s=total_audio,
        total_processing_s=total_proc,
        rtf=rtf,
        wer=wer,
    )


def measure(
    engine: _SttEngine,
    model: str,
    clips: list[dict[str, Any]],
    clip_paths: dict[str, Path],
    runs: int,
    warmup_runs: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Run the STT benchmark loop; return (median, std_dev, throttling_detected)."""
    references = [c["reference"] for c in clips]
    all_stats: list[_RunStats] = []

    for i in range(warmup_runs + runs):
        run_results: list[_TranscribeResult] = []
        for clip in clips:
            path = clip_paths[clip["id"]]
            result = engine.transcribe(path, model)
            run_results.append(result)

        if i >= warmup_runs:
            all_stats.append(_aggregate_run(run_results, references))

    rtf_values = [s.rtf for s in all_stats]
    wer_values = [s.wer for s in all_stats]

    median_rtf = statistics.median(rtf_values) if rtf_values else 0.0
    median_wer = statistics.median(wer_values) if wer_values else 0.0
    std_rtf = statistics.stdev(rtf_values) if len(rtf_values) > 1 else 0.0
    std_wer = statistics.stdev(wer_values) if len(wer_values) > 1 else 0.0

    cv = std_rtf / median_rtf if median_rtf > 0 else 0.0
    throttling_detected = cv > 0.15

    median_stats = {
        "rtf": median_rtf,
        "wer": median_wer,
        "total_audio_seconds": all_stats[0].total_audio_s if all_stats else 0.0,
    }
    std_dev_stats = {"rtf": std_rtf, "wer": std_wer}

    return median_stats, std_dev_stats, throttling_detected
