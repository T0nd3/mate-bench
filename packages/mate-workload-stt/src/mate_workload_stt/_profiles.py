from __future__ import annotations

from mate_bench.plugin import ProfileConfig, TestSetSpec

# Public R2 CDN
CDN_BASE = "https://pub-f27eb09940c14a8dac6ae7fe10e789f3.r2.dev"

# SHA256 values are filled in by scripts/prepare_stt_testset.py after upload.
TEST_SETS: dict[str, TestSetSpec] = {
    "stt-librispeech-quick-v1": TestSetSpec(
        id="stt-librispeech-quick-v1",
        url=f"{CDN_BASE}/stt/stt-librispeech-quick-v1.json",
        sha256="sha256:420a49113d77e4e72a1740eeb9fff9b1918def6821c994b1d2c7e59990b8adc3",
        size_bytes=754300,
        license="CC BY 4.0",
        source="LibriSpeech test-clean (Panayotov et al., 2015) — openslr.org/12",
    ),
    "stt-librispeech-standard-v1": TestSetSpec(
        id="stt-librispeech-standard-v1",
        url=f"{CDN_BASE}/stt/stt-librispeech-standard-v1.json",
        sha256="sha256:66a200da99544095c68b90fff5287ed4460b61c854ef29a29524373973a2915b",
        size_bytes=3184083,
        license="CC BY 4.0",
        source="LibriSpeech test-clean (Panayotov et al., 2015) — openslr.org/12",
    ),
}

PROFILES: dict[str, ProfileConfig] = {
    "quick": ProfileConfig(
        name="quick",
        description="5 LibriSpeech clips (~50 s audio) with whisper-large-v3 (~3 min, ~3 GB VRAM)",
        test_set_id="stt-librispeech-quick-v1",
        reference_engine_config={"engine": "faster-whisper", "model": "large-v3"},
        vram_required_gb=3.0,
        download_size_bytes=1_500_000_000,  # large-v3 ~1.5 GB
        estimated_runtime_seconds=180,
    ),
    "standard": ProfileConfig(
        name="standard",
        description="20 LibriSpeech clips (~200 s audio) with whisper-large-v3 (~8 min, ~3 GB VRAM)",
        test_set_id="stt-librispeech-standard-v1",
        reference_engine_config={"engine": "faster-whisper", "model": "large-v3"},
        vram_required_gb=3.0,
        download_size_bytes=1_500_000_000,
        estimated_runtime_seconds=480,
    ),
}
