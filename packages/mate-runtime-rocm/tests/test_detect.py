from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mate_runtime_rocm._detect import (
    _driver_version,
    _parse_rocminfo,
    _rocm_version,
    _vram_gb,
    query_gpu,
)
from mate_runtime_rocm._names import normalize

from .fixtures import (
    ROCMINFO_NO_GPU,
    ROCMINFO_RX6700XT,
    ROCMINFO_RX7900XTX,
    ROCMINFO_UNKNOWN_CHIP,
    ROCM_SMI_DRIVER,
    ROCM_SMI_VRAM_6700XT,
    ROCM_SMI_VRAM_7900XTX,
    ROCM_VERSION_FILE,
)


# ── _parse_rocminfo ───────────────────────────────────────────────────────────

class TestParseRocminfo:
    def test_rx7900xtx(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCMINFO_RX7900XTX):
            agents = _parse_rocminfo()
        gpu_agents = [a for a in agents if a.get("chip", "").startswith("gfx")]
        assert len(gpu_agents) == 1
        assert gpu_agents[0]["chip"] == "gfx1100"
        assert gpu_agents[0]["marketing_name"] == "AMD Radeon RX 7900 XTX"

    def test_rx6700xt(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCMINFO_RX6700XT):
            agents = _parse_rocminfo()
        gpu_agents = [a for a in agents if a.get("chip", "").startswith("gfx")]
        assert len(gpu_agents) == 1
        assert gpu_agents[0]["chip"] == "gfx1031"
        assert gpu_agents[0]["marketing_name"] == "AMD Radeon RX 6700 XT"

    def test_no_gpu_returns_empty(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCMINFO_NO_GPU):
            agents = _parse_rocminfo()
        gpu_agents = [a for a in agents if a.get("chip", "").startswith("gfx")]
        assert gpu_agents == []

    def test_rocminfo_unavailable_returns_empty(self):
        with patch("mate_runtime_rocm._detect._run", return_value=""):
            assert _parse_rocminfo() == []


# ── _vram_gb ─────────────────────────────────────────────────────────────────

class TestVramGb:
    def test_rx7900xtx_is_24gb(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCM_SMI_VRAM_7900XTX):
            vram = _vram_gb(0)
        assert vram == 24.0

    def test_rx6700xt_is_12gb(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCM_SMI_VRAM_6700XT):
            vram = _vram_gb(0)
        assert vram == 12.0

    def test_unavailable_returns_zero(self):
        with patch("mate_runtime_rocm._detect._run", return_value=""):
            assert _vram_gb(0) == 0.0


# ── _rocm_version ─────────────────────────────────────────────────────────────

class TestRocmVersion:
    def test_reads_version_file(self, tmp_path):
        version_file = tmp_path / "version"
        version_file.write_text(ROCM_VERSION_FILE)
        with patch("mate_runtime_rocm._detect.Path") as mock_path:
            # Make the first path check succeed
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = ROCM_VERSION_FILE
            version = _rocm_version()
        assert version == "rocm-6.3.1"

    def test_falls_back_to_rocm_smi(self):
        with patch("mate_runtime_rocm._detect.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with patch("mate_runtime_rocm._detect._run", return_value="ROCM version: 6.2.0\n"):
                version = _rocm_version()
        assert version == "rocm-6.2.0"

    def test_unknown_fallback(self):
        with patch("mate_runtime_rocm._detect.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with patch("mate_runtime_rocm._detect._run", return_value=""):
                version = _rocm_version()
        assert version == "rocm-unknown"


# ── _driver_version ───────────────────────────────────────────────────────────

class TestDriverVersion:
    def test_parses_kernel_line(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCM_SMI_DRIVER):
            assert _driver_version() == "6.7.0"

    def test_unknown_on_empty(self):
        with patch("mate_runtime_rocm._detect._run", return_value=""):
            assert _driver_version() == "unknown"


# ── query_gpu ─────────────────────────────────────────────────────────────────

class TestQueryGpu:
    def _mock_run(self, rocminfo_out: str, vram_out: str, driver_out: str):
        def side_effect(cmd, **kwargs):
            joined = " ".join(cmd)
            if "rocminfo" in joined:
                return rocminfo_out
            if "showmeminfo" in joined:
                return vram_out
            if "showdriverversion" in joined:
                return driver_out
            return ""
        return side_effect

    def test_rx7900xtx_full(self):
        with patch("mate_runtime_rocm._detect._run",
                   side_effect=self._mock_run(ROCMINFO_RX7900XTX, ROCM_SMI_VRAM_7900XTX, ROCM_SMI_DRIVER)), \
             patch("mate_runtime_rocm._detect.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            info = query_gpu(0)

        assert info.chip == "gfx1100"
        assert info.name == "RX 7900 XTX"
        assert info.name_raw == "AMD Radeon RX 7900 XTX"
        assert info.name_known is True
        assert info.vram_gb == 24.0
        assert info.driver_version == "6.7.0"

    def test_unknown_chip_flagged(self):
        with patch("mate_runtime_rocm._detect._run",
                   side_effect=self._mock_run(ROCMINFO_UNKNOWN_CHIP, "", "")), \
             patch("mate_runtime_rocm._detect.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            info = query_gpu(0)

        assert info.chip == "gfx9999"
        assert info.name_known is False
        assert info.name == "gfx9999"

    def test_no_gpu_raises(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCMINFO_NO_GPU), \
             patch("mate_runtime_rocm._detect.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="No AMD GPU found"):
                query_gpu(0)

    def test_index_out_of_range_raises(self):
        with patch("mate_runtime_rocm._detect._run", return_value=ROCMINFO_RX7900XTX), \
             patch("mate_runtime_rocm._detect.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(IndexError):
                query_gpu(1)


# ── normalize ─────────────────────────────────────────────────────────────────

class TestNormalize:
    @pytest.mark.parametrize("raw, expected", [
        ("AMD Radeon RX 7900 XTX",  "RX 7900 XTX"),
        ("AMD Radeon RX 7900 XT",   "RX 7900 XT"),
        ("AMD Radeon RX 6700 XT",   "RX 6700 XT"),
        ("AMD Radeon VII",           "Radeon VII"),
        ("AMD Radeon 780M",          "Radeon 780M"),
        ("AMD Instinct MI300X",      "Instinct MI300X"),
        ("AMD Radeon Pro W7900",     "Pro W7900"),
    ])
    def test_known_patterns(self, raw, expected):
        name, known = normalize(raw)
        assert name == expected
        assert known is True

    def test_empty_string(self):
        name, known = normalize("")
        assert known is False
