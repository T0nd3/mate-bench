from __future__ import annotations

from typing import Any

from mate_bench.plugin import PluginManifest

from ._detect import GpuInfo, is_rocm_available, query_gpu


class RocmRuntime:
    name = "rocm"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)

    def is_available(self) -> bool:
        return is_rocm_available()

    def gpu_info(self, index: int = 0) -> dict[str, Any]:
        info: GpuInfo = query_gpu(index)
        return {
            "gpu_vendor": "amd",
            "gpu_name": info.name,
            "gpu_chip": info.chip,
            "vram_gb": info.vram_gb,
            "runtime": info.rocm_version,
            "driver": info.driver_version,
            # Extra fields for transparency — not part of SystemInfo but useful for debugging
            "_gpu_name_raw": info.name_raw,
            "_gpu_name_known": info.name_known,
        }
