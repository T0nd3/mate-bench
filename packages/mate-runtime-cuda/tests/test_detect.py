from __future__ import annotations

from unittest.mock import patch

import pytest

from mate_runtime_cuda._detect import _query_smi, is_cuda_available, query_gpu
from mate_runtime_cuda._names import chip_from_name, normalize

from .fixtures import (
    SMI_A100,
    SMI_EMPTY,
    SMI_LIST_EMPTY,
    SMI_LIST_RTX4090,
    SMI_RTX3080,
    SMI_RTX4090,
    SMI_UNKNOWN,
)

# ── is_cuda_available ─────────────────────────────────────────────────────────


class TestIsCudaAvailable:
    def test_available_when_smi_returns_output(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_LIST_RTX4090):
            assert is_cuda_available() is True

    def test_unavailable_when_smi_empty(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_LIST_EMPTY):
            assert is_cuda_available() is False


# ── _query_smi ────────────────────────────────────────────────────────────────


class TestQuerySmi:
    def test_parses_rtx4090_row(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_RTX4090):
            rows = _query_smi(["name", "memory.total", "driver_version", "cuda_version"])
        assert len(rows) == 1
        assert "RTX 4090" in rows[0][0]

    def test_empty_returns_empty_list(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_EMPTY):
            assert _query_smi(["name"]) == []


# ── query_gpu ─────────────────────────────────────────────────────────────────


class TestQueryGpu:
    def test_rtx4090(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_RTX4090):
            info = query_gpu(0)
        assert info.chip == "AD102"
        assert info.name == "RTX 4090"
        assert info.name_raw == "NVIDIA GeForce RTX 4090"
        assert info.name_known is True
        assert info.vram_gb == 24.0
        assert info.cuda_version == "CUDA 12.4"
        assert info.driver_version == "550.54.15"

    def test_rtx3080(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_RTX3080):
            info = query_gpu(0)
        assert info.chip == "GA102"
        assert info.name == "RTX 3080"
        assert info.vram_gb == 10.0

    def test_a100(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_A100):
            info = query_gpu(0)
        assert info.chip == "GA100"
        assert info.vram_gb == 80.0

    def test_unknown_gpu_not_known(self):
        with patch("mate_runtime_cuda._detect._run", return_value=SMI_UNKNOWN):
            info = query_gpu(0)
        assert info.name_known is False
        assert info.name == "SomeUnknownGPU"

    def test_no_gpu_raises(self):
        with (
            patch("mate_runtime_cuda._detect._run", return_value=SMI_EMPTY),
            pytest.raises(RuntimeError, match="No NVIDIA GPU found"),
        ):
            query_gpu(0)

    def test_index_out_of_range_raises(self):
        with (
            patch("mate_runtime_cuda._detect._run", return_value=SMI_RTX4090),
            pytest.raises(IndexError),
        ):
            query_gpu(1)


# ── normalize & chip_from_name ────────────────────────────────────────────────


class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expected_name",
        [
            ("NVIDIA GeForce RTX 4090", "RTX 4090"),
            ("NVIDIA GeForce RTX 3080", "RTX 3080"),
            ("NVIDIA A100-SXM4-80GB", "A100-SXM4-80GB"),
            ("NVIDIA GeForce GTX 1080 Ti", "GTX 1080 Ti"),
            ("NVIDIA GeForce RTX 2080 Ti", "RTX 2080 Ti"),
        ],
    )
    def test_strips_prefix(self, raw, expected_name):
        name, _ = normalize(raw)
        assert name == expected_name

    @pytest.mark.parametrize(
        "raw, expected_chip",
        [
            ("NVIDIA GeForce RTX 4090", "AD102"),
            ("NVIDIA GeForce RTX 3090", "GA102"),
            ("NVIDIA GeForce RTX 3080", "GA102"),
            ("NVIDIA GeForce RTX 3070", "GA104"),
            ("NVIDIA GeForce RTX 2080 Ti", "TU102"),
            ("NVIDIA GeForce GTX 1080 Ti", "GP102"),
            ("NVIDIA H100", "GH100"),
            ("NVIDIA A100-SXM4-80GB", "GA100"),
        ],
    )
    def test_known_chips(self, raw, expected_chip):
        name, _ = normalize(raw)
        chip, known = chip_from_name(name)
        assert chip == expected_chip
        assert known is True

    def test_unknown_chip(self):
        _, known = normalize("NVIDIA Futuristic GPU XYZ")
        assert known is False
