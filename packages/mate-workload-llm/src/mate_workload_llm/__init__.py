from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from mate_bench._utils import sha256_file
from mate_bench.paths import TEST_SETS_DIR
from mate_bench.plugin import (
    EnginePlugin,
    Measurement,
    Mode,
    PluginManifest,
    ProfileConfig,
    TestSetSpec,
)

from ._measure import measure
from ._profiles import BUNDLED_TEST_SETS, FULL_PROFILE_MODELS, PROFILES, TEST_SETS


class LlmWorkload:
    name = "llm"
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
        if profile == "full":
            return [p["model"] for p in FULL_PROFILE_MODELS]
        return [PROFILES[profile].reference_engine_config["model"]]

    def _copy_test_set(self, test_set_id: str) -> None:
        src = BUNDLED_TEST_SETS[test_set_id]
        dst_dir = TEST_SETS_DIR / "llm"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    def setup_closed(self, profile: str) -> None:
        if profile == "full":
            for pair in FULL_PROFILE_MODELS:
                self._copy_test_set(pair["test_set_id"])
        else:
            self._copy_test_set(PROFILES[profile].test_set_id)

    def setup_open(self, profile: str, user_inputs: dict[str, Any]) -> None:
        raise NotImplementedError("open mode not yet supported for llm workload")

    def _load_prompts_by_id(self, test_set_id: str) -> list[dict[str, Any]]:
        cached = TEST_SETS_DIR / "llm" / BUNDLED_TEST_SETS[test_set_id].name
        if not cached.exists():
            self._copy_test_set(test_set_id)
        with cached.open() as f:
            data = json.load(f)
        return data["prompts"]

    def _load_prompts(self, profile: str) -> list[dict[str, Any]]:
        return self._load_prompts_by_id(PROFILES[profile].test_set_id)

    def _model_for_profile(self, profile: str) -> str:
        return PROFILES[profile].reference_engine_config["model"]

    def run(
        self,
        profile: str,
        mode: Mode,
        engine: EnginePlugin,
        runs: int,
        warmup_runs: int,
    ) -> Measurement:
        if mode != Mode.CLOSED:
            raise NotImplementedError("open mode not yet supported for llm workload")

        if profile == "full":
            return self._run_full(engine, runs, warmup_runs)

        prompts = self._load_prompts(profile)
        model = self._model_for_profile(profile)

        median_stats, std_dev_stats, throttling_detected = measure(
            engine,  # type: ignore[arg-type]
            model,
            prompts,
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

    def _run_full(self, engine: EnginePlugin, runs: int, warmup_runs: int) -> Measurement:
        per_model_median: dict[str, Any] = {}
        per_model_std_dev: dict[str, Any] = {}
        any_throttling = False

        for pair in FULL_PROFILE_MODELS:
            model = pair["model"]
            prompts = self._load_prompts_by_id(pair["test_set_id"])
            median, std_dev, throttling = measure(
                engine,  # type: ignore[arg-type]
                model,
                prompts,
                runs,
                warmup_runs,
            )
            per_model_median[model] = median
            per_model_std_dev[model] = std_dev
            any_throttling = any_throttling or throttling

        return Measurement(
            runs=runs,
            warmup_runs=warmup_runs,
            median=per_model_median,
            std_dev=per_model_std_dev,
            vram_peak_gb=0.0,
            throttling_detected=any_throttling,
        )

    def get_cached_test_set_path(self, profile: str) -> Path:
        test_set_id = PROFILES[profile].test_set_id
        return TEST_SETS_DIR / "llm" / BUNDLED_TEST_SETS[test_set_id].name

    def test_set_hash(self, profile: str) -> str:
        path = self.get_cached_test_set_path(profile)
        if not path.exists():
            self.setup_closed(profile)
        return sha256_file(path)

    def cleanup(self, profile: str) -> None:
        ids_to_clean: set[str]
        if profile == "full":
            ids_to_clean = {p["test_set_id"] for p in FULL_PROFILE_MODELS}
        else:
            ids_to_clean = {PROFILES[profile].test_set_id}

        for test_set_id in ids_to_clean:
            cached = TEST_SETS_DIR / "llm" / BUNDLED_TEST_SETS[test_set_id].name
            if cached.exists():
                cached.unlink()
