from __future__ import annotations

import platform
from typing import Any

from .schema import SystemInfo


def collect_system_info(runtimes: dict[str, Any]) -> SystemInfo:
    for runtime in runtimes.values():
        try:
            if not runtime.is_available():
                continue
            info = runtime.gpu_info()
            return SystemInfo(
                gpu_vendor=info["gpu_vendor"],
                gpu_name=info.get("gpu_name", "unknown"),
                gpu_chip=info.get("gpu_chip", "unknown"),
                vram_gb=float(info.get("vram_gb") or 0.0),
                runtime=str(info.get("runtime", "unknown")),
                driver=str(info.get("driver", "unknown")),
                os=platform.system(),
                kernel=platform.release(),
            )
        except Exception:
            continue

    return SystemInfo(
        gpu_vendor="cpu",
        gpu_name="unknown",
        gpu_chip="unknown",
        vram_gb=0.0,
        runtime="cpu",
        driver="unknown",
        os=platform.system(),
        kernel=platform.release(),
    )
