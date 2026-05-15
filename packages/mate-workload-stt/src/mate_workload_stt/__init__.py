"""Speech-to-text workload plugin for mate-bench.

Benchmarks STT transcription speed (RTF) and accuracy (WER) using
LibriSpeech test-clean audio clips with a faster-whisper backend.
"""
from __future__ import annotations

import json
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

from ._download import fetch_clips, fetch_manifest
from ._measure import measure
from ._profiles import PROFILES, TEST_SETS

__all__ = ["SttWorkload"]

_STT_DIR = TEST_SETS_DIR / "stt"


class SttWorkload:
    name = "stt"
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
        return [PROFILES[profile].reference_engine_config["model"]]

    def _manifest_path(self, test_set_id: str) -> Path:
        spec = TEST_SETS[test_set_id]
        name = spec.url.rsplit("/", 1)[-1]
        return _STT_DIR / "manifests" / name

    def setup_closed(self, profile: str) -> None:
        spec = TEST_SETS[PROFILES[profile].test_set_id]
        cached_manifest = fetch_manifest(spec.url, spec.sha256, _STT_DIR)
        fetch_clips(cached_manifest, _STT_DIR / "clips")

    def setup_open(self, profile: str, user_inputs: dict[str, Any]) -> None:
        raise NotImplementedError("open mode not supported for stt workload")

    def _load_clips_and_paths(self, profile: str) -> tuple[list[dict], dict[str, Path]]:
        test_set_id = PROFILES[profile].test_set_id
        spec = TEST_SETS[test_set_id]
        cached_manifest = fetch_manifest(spec.url, spec.sha256, _STT_DIR)
        clip_paths = fetch_clips(cached_manifest, _STT_DIR / "clips")
        return cached_manifest["clips"], clip_paths

    def run(
        self,
        profile: str,
        mode: Mode,
        engine: EnginePlugin,
        runs: int,
        warmup_runs: int,
    ) -> Measurement:
        if mode != Mode.CLOSED:
            raise NotImplementedError("open mode not supported for stt workload")

        cfg = PROFILES[profile].reference_engine_config
        model = cfg["model"]
        clips, clip_paths = self._load_clips_and_paths(profile)

        median_stats, std_dev_stats, throttling_detected = measure(
            engine,  # type: ignore[arg-type]
            model,
            clips,
            clip_paths,
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

    def test_set_hash(self, profile: str) -> str:
        path = self._manifest_path(PROFILES[profile].test_set_id)
        if not path.exists():
            self.setup_closed(profile)
        return sha256_file(path)

    def cleanup(self, profile: str) -> None:
        test_set_id = PROFILES[profile].test_set_id
        spec = TEST_SETS[test_set_id]
        name = spec.url.rsplit("/", 1)[-1]
        manifest_path = _STT_DIR / "manifests" / name
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)
            for clip in manifest.get("clips", []):
                fname = clip["url"].rsplit("/", 1)[-1]
                p = _STT_DIR / "clips" / fname
                if p.exists():
                    p.unlink()
            manifest_path.unlink()
