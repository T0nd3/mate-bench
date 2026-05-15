from __future__ import annotations

from pathlib import Path

from mate_bench.plugin import ProfileConfig, TestSetSpec

_DATA_DIR = Path(__file__).parent / "data" / "test-sets"

BUNDLED_TEST_SETS: dict[str, Path] = {
    "imagegen-512-v1": _DATA_DIR / "imagegen-512-v1.json",
    "imagegen-1024-v1": _DATA_DIR / "imagegen-1024-v1.json",
}

# No CDN URLs — test sets are bundled (prompts are plain text, no media).
# sha256 values are informational only; integrity is checked against the bundled file.
TEST_SETS: dict[str, TestSetSpec] = {
    "imagegen-512-v1": TestSetSpec(
        id="imagegen-512-v1",
        url="bundled",
        sha256="sha256:bundled",
        size_bytes=780,
        license="CC0-1.0",
        source="mate-bench project (original)",
    ),
    "imagegen-1024-v1": TestSetSpec(
        id="imagegen-1024-v1",
        url="bundled",
        sha256="sha256:bundled",
        size_bytes=520,
        license="CC0-1.0",
        source="mate-bench project (original)",
    ),
}

# Open-mode profiles: user specifies the model, we supply the prompt set + resolution.
PROFILES: dict[str, ProfileConfig] = {
    "quick-512": ProfileConfig(
        name="quick-512",
        description="5 prompts at 512×512, 20 steps — open mode, specify your own model",
        test_set_id="imagegen-512-v1",
        reference_engine_config={"engine": "comfyui"},
        vram_required_gb=4.0,
        download_size_bytes=0,
        estimated_runtime_seconds=120,
    ),
    "standard-1024": ProfileConfig(
        name="standard-1024",
        description="3 prompts at 1024×1024, 20 steps — open mode, specify your own model",
        test_set_id="imagegen-1024-v1",
        reference_engine_config={"engine": "comfyui"},
        vram_required_gb=8.0,
        download_size_bytes=0,
        estimated_runtime_seconds=300,
    ),
}
