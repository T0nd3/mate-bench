"""Image generation workload plugin for mate-bench (open mode only).

Measures txt2img throughput (images/s, steps/s) across a fixed prompt
suite.  Closed mode is intentionally unsupported — see ImageGenWorkload.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

__all__ = ["ImageGenWorkload"]


class ImageGenWorkload:
    """Image generation workload — open mode only.

    Closed mode is intentionally unsupported: image model weights are not
    standardised across installations, making apples-to-apples comparison
    impossible without a canonical model reference.  Users run open mode and
    specify the model they have locally.
    """

    name = "imagegen"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    profiles: dict[str, ProfileConfig] = PROFILES
    test_sets: dict[str, TestSetSpec] = TEST_SETS

    def estimate_download(self, profile: str) -> int:
        return PROFILES[profile].download_size_bytes

    def estimate_vram(self, profile: str) -> int:
        return int(PROFILES[profile].vram_required_gb * 1024**3)

    def estimate_runtime(self, profile: str) -> int:
        return PROFILES[profile].estimated_runtime_seconds

    def required_models(self, profile: str) -> list[str]:
        return []  # user-supplied in open mode

    def setup_closed(self, profile: str) -> None:
        raise NotImplementedError(
            "imagegen workload supports open mode only — "
            "run with --mode open and --model <your-checkpoint>"
        )

    def setup_open(self, profile: str, user_inputs: dict[str, Any]) -> None:
        pass  # test sets are bundled, nothing to fetch

    def _load_tasks(self, profile: str) -> tuple[list[dict], int, int, int, float, str]:
        test_set_id = PROFILES[profile].test_set_id
        path = BUNDLED_TEST_SETS[test_set_id]
        with path.open() as f:
            data = json.load(f)
        return (
            data["tasks"],
            data["width"],
            data["height"],
            data["steps"],
            data["cfg"],
            data["sampler"],
        )

    def run(
        self,
        profile: str,
        mode: Mode,
        engine: EnginePlugin,
        runs: int,
        warmup_runs: int,
        model: str = "",
        seed: int = 42,
    ) -> Measurement:
        if mode != Mode.OPEN:
            raise NotImplementedError(
                "imagegen workload supports open mode only — "
                "run with --mode open and --model <your-checkpoint>"
            )
        if not model:
            raise ValueError("model must be specified for imagegen open-mode runs")

        tasks, width, height, steps, cfg, sampler = self._load_tasks(profile)

        median_stats, std_dev_stats, throttling_detected = measure(
            engine,  # type: ignore[arg-type]
            model,
            tasks,
            width,
            height,
            steps,
            cfg,
            sampler,
            runs,
            warmup_runs,
            seed=seed,
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
