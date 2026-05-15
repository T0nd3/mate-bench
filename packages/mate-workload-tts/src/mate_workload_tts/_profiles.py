"""Profile definitions for the TTS workload."""
from __future__ import annotations

from importlib.resources import files

from mate_bench.plugin import ProfileConfig, TestSetSpec

# Bundled test sets — no CDN needed, just text
_DATA = files("mate_workload_tts") / "data" / "test-sets"

BUNDLED_TEST_SETS: dict[str, object] = {
    "tts-sentences-quick-v1":    _DATA / "tts-sentences-quick-v1.json",
    "tts-sentences-standard-v1": _DATA / "tts-sentences-standard-v1.json",
}

TEST_SETS: dict[str, TestSetSpec] = {
    "tts-sentences-quick-v1": TestSetSpec(
        id="tts-sentences-quick-v1",
        url="bundled",
        sha256="bundled",
        size_bytes=0,
        license="CC0-1.0",
        source="Custom CC0 sentence corpus",
    ),
    "tts-sentences-standard-v1": TestSetSpec(
        id="tts-sentences-standard-v1",
        url="bundled",
        sha256="bundled",
        size_bytes=0,
        license="CC0-1.0",
        source="Custom CC0 sentence corpus",
    ),
}

PROFILES: dict[str, ProfileConfig] = {
    "quick": ProfileConfig(
        name="quick",
        description="5 sentences (~60 words) with kokoro-v1.0/af_heart (~2 min, ~1.5 GB VRAM)",
        test_set_id="tts-sentences-quick-v1",
        reference_engine_config={"engine": "kokoro", "model": "kokoro-v1.0", "voice": "af_heart"},
        vram_required_gb=1.5,
        download_size_bytes=350_000_000,  # kokoro-v1.0 ~350 MB
        estimated_runtime_seconds=120,
    ),
    "standard": ProfileConfig(
        name="standard",
        description="20 sentences (~240 words) with kokoro-v1.0/af_heart (~6 min, ~1.5 GB VRAM)",
        test_set_id="tts-sentences-standard-v1",
        reference_engine_config={"engine": "kokoro", "model": "kokoro-v1.0", "voice": "af_heart"},
        vram_required_gb=1.5,
        download_size_bytes=350_000_000,
        estimated_runtime_seconds=360,
    ),
}
