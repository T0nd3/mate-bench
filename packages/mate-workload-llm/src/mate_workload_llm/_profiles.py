from __future__ import annotations

from pathlib import Path

from mate_bench.plugin import ProfileConfig, TestSetSpec

_DATA_DIR = Path(__file__).parent / "data" / "test-sets"

BUNDLED_TEST_SETS: dict[str, Path] = {
    "llm-short-v1": _DATA_DIR / "llm-short-v1.json",
    "llm-medium-v1": _DATA_DIR / "llm-medium-v1.json",
}

CDN_BASE = "https://pub-f27eb09940c14a8dac6ae7fe10e789f3.r2.dev"

TEST_SETS: dict[str, TestSetSpec] = {
    "llm-short-v1": TestSetSpec(
        id="llm-short-v1",
        url=f"{CDN_BASE}/llm-short-v1.json",
        sha256="sha256:68aecfa32e38afc4e817a254b7e30dc5e732efa9985b133e115cbd15251e03ed",
        size_bytes=1362,
        license="CC0-1.0",
        source="mate-bench project (original)",
    ),
    "llm-medium-v1": TestSetSpec(
        id="llm-medium-v1",
        url=f"{CDN_BASE}/llm-medium-v1.json",
        sha256="sha256:5a267661575170831c80d658c62b17851f13722c4a793c522c00c6a1693a1c37",
        size_bytes=3739,
        license="CC0-1.0",
        source="mate-bench project (original)",
    ),
}

FULL_PROFILE_MODELS: list[dict[str, str]] = [
    {"model": "llama3.2:latest", "test_set_id": "llm-short-v1"},
    {"model": "llama3.1:8b", "test_set_id": "llm-medium-v1"},
]

PROFILES: dict[str, ProfileConfig] = {
    "quick": ProfileConfig(
        name="quick",
        description="5 short prompts with llama3.2:latest (~2-4 min, ~3 GB VRAM)",
        test_set_id="llm-short-v1",
        reference_engine_config={"engine": "ollama", "model": "llama3.2:latest"},
        vram_required_gb=3.0,
        download_size_bytes=2_019_393_189,
        estimated_runtime_seconds=240,
    ),
    "standard": ProfileConfig(
        name="standard",
        description="3 medium prompts with llama3.1:8b (~5-10 min, ~6 GB VRAM)",
        test_set_id="llm-medium-v1",
        reference_engine_config={"engine": "ollama", "model": "llama3.1:8b"},
        vram_required_gb=5.5,
        download_size_bytes=4_661_211_136,
        estimated_runtime_seconds=600,
    ),
    "full": ProfileConfig(
        name="full",
        description="Both test sets with llama3.2:latest and llama3.1:8b (~15-25 min, ~6 GB VRAM)",
        test_set_id="llm-medium-v1",
        reference_engine_config={"engine": "ollama", "model": "llama3.1:8b"},
        vram_required_gb=5.5,
        download_size_bytes=6_680_604_325,
        estimated_runtime_seconds=1200,
    ),
}
