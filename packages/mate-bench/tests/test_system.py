from __future__ import annotations

from unittest.mock import MagicMock

from mate_bench._system import collect_system_info


def _runtime(available: bool, gpu_info: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.is_available.return_value = available
    if gpu_info is not None:
        r.gpu_info.return_value = gpu_info
    return r


GPU_INFO = {
    "gpu_vendor": "amd",
    "gpu_name": "RX 7900 XTX",
    "gpu_chip": "gfx1100",
    "vram_gb": 24.0,
    "runtime": "ROCm 6.1.0",
    "driver": "6.7.0",
}


class TestCollectSystemInfo:
    def test_uses_first_available_runtime(self):
        runtimes = {"rocm": _runtime(True, GPU_INFO)}
        info = collect_system_info(runtimes)
        assert info.gpu_vendor == "amd"
        assert info.gpu_name == "RX 7900 XTX"
        assert info.vram_gb == 24.0

    def test_skips_unavailable_runtime(self):
        runtimes = {
            "rocm": _runtime(False),
            "cuda": _runtime(True, {**GPU_INFO, "gpu_vendor": "nvidia", "gpu_name": "RTX 4090"}),
        }
        info = collect_system_info(runtimes)
        assert info.gpu_vendor == "nvidia"

    def test_falls_back_to_cpu_when_no_runtime(self):
        info = collect_system_info({})
        assert info.gpu_vendor == "cpu"
        assert info.vram_gb == 0.0

    def test_falls_back_when_runtime_raises(self):
        bad = MagicMock()
        bad.is_available.side_effect = RuntimeError("boom")
        info = collect_system_info({"bad": bad})
        assert info.gpu_vendor == "cpu"

    def test_falls_back_when_gpu_info_raises(self):
        bad = _runtime(True)
        bad.gpu_info.side_effect = RuntimeError("no gpu")
        info = collect_system_info({"bad": bad})
        assert info.gpu_vendor == "cpu"

    def test_os_and_kernel_always_populated(self):
        info = collect_system_info({})
        assert len(info.os) > 0
        assert len(info.kernel) > 0
