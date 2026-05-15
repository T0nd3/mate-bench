"""Text-to-speech workload plugin for mate-bench.

Measures TTS synthesis speed (RTF, chars/s) using a bundled CC0
sentence corpus. Supports both closed mode (kokoro reference) and
open mode (user-specified model/voice).
"""
from __future__ import annotations

import json
from typing import Any, ClassVar

from mate_bench.plugin import (
    EnginePlugin,
    Measurement,
    Mode,
    PluginManifest,
    ProfileConfig,
    TestSetSpec,
)

from ._measure import measure
from ._profiles import BUNDLED_TEST_SETS, PROFILES, TEST_SETS

__all__ = ["TtsWorkload"]


class TtsWorkload:
    """TTS workload: measures synthesis speed with a fixed sentence corpus."""

    name = "tts"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    profiles: ClassVar[dict[str, ProfileConfig]] = PROFILES
    test_sets: ClassVar[dict[str, TestSetSpec]] = TEST_SETS

    def __init__(self) -> None:
        self._open_model: str | None = None
        self._open_voice: str | None = None

    def estimate_download(self, profile: str) -> int:
        return PROFILES[profile].download_size_bytes

    def estimate_vram(self, profile: str) -> int:
        return int(PROFILES[profile].vram_required_gb * 1024**3)

    def estimate_runtime(self, profile: str) -> int:
        return PROFILES[profile].estimated_runtime_seconds

    def required_models(self, profile: str) -> list[str]:
        cfg = PROFILES[profile].reference_engine_config
        return [cfg["model"]]

    def _load_sentences(self, profile: str) -> list[dict[str, Any]]:
        test_set_id = PROFILES[profile].test_set_id
        path = BUNDLED_TEST_SETS[test_set_id]
        with open(str(path)) as f:
            data = json.load(f)
        return data["sentences"]

    def setup_closed(self, profile: str) -> None:
        pass  # test sets are bundled, nothing to fetch

    def setup_open(self, profile: str, user_inputs: dict[str, Any]) -> None:
        model = user_inputs.get("model", "").strip()
        if not model:
            raise ValueError("user_inputs must contain a non-empty 'model' key")
        self._open_model = model
        self._open_voice = user_inputs.get("voice", "")

    def run(
        self,
        profile: str,
        mode: Mode,
        engine: EnginePlugin,
        runs: int,
        warmup_runs: int,
    ) -> Measurement:
        """Run the TTS benchmark and return a Measurement."""
        if mode == Mode.OPEN:
            if not self._open_model:
                raise RuntimeError("call setup_open() before run() in open mode")
            model = self._open_model
            voice = self._open_voice or ""
        else:
            cfg = PROFILES[profile].reference_engine_config
            model = cfg["model"]
            voice = cfg.get("voice", "")

        sentences = self._load_sentences(profile)
        median_stats, std_dev_stats, throttling_detected = measure(
            engine,  # type: ignore[arg-type]
            model,
            voice,
            sentences,
            runs,
            warmup_runs,
        )
        return Measurement(
            runs=runs,
            warmup_runs=warmup_runs,
            median=median_stats,
            std_dev=std_dev_stats,
            vram_peak_gb=0.0,
            throttling_detected=throttling_detected,
        )

    def cleanup(self, profile: str) -> None:
        pass  # nothing cached
