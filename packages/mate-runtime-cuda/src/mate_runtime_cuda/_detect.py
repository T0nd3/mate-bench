from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from ._names import chip_from_name, normalize


@dataclass
class GpuInfo:
    chip: str  # die codename, e.g. "AD102"
    name: str  # normalized, e.g. "RTX 4090"
    name_raw: str  # as reported by nvidia-smi, e.g. "NVIDIA GeForce RTX 4090"
    name_known: bool  # False → unknown model, stored raw + flagged
    vram_gb: float
    cuda_version: str  # e.g. "CUDA 12.4"
    driver_version: str  # e.g. "550.54.15"


def is_cuda_available() -> bool:
    """CUDA is available if nvidia-smi lists at least one GPU."""
    return bool(_run(["nvidia-smi", "-L"]))


def query_gpu(index: int = 0) -> GpuInfo:
    """Query GPU info for the given device index via nvidia-smi."""
    rows = _query_smi(
        ["name", "memory.total", "driver_version", "cuda_version"],
        units=False,
    )
    if not rows:
        raise RuntimeError("No NVIDIA GPU found. Are the drivers installed?")
    if index >= len(rows):
        raise IndexError(f"GPU index {index} out of range ({len(rows)} GPUs found)")

    row = rows[index]
    raw_name = row[0].strip()
    vram_mib = _parse_float(row[1])
    driver = row[2].strip()
    cuda = row[3].strip()

    name, name_known = normalize(raw_name)
    chip, chip_known = chip_from_name(name)
    if not chip_known:
        chip = name  # fall back to display name as chip key

    return GpuInfo(
        chip=chip,
        name=name,
        name_raw=raw_name,
        name_known=name_known and chip_known,
        vram_gb=round(vram_mib / 1024, 1),
        cuda_version=f"CUDA {cuda}" if cuda else "CUDA unknown",
        driver_version=driver or "unknown",
    )


# ── internal helpers ──────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _query_smi(fields: list[str], units: bool = False) -> list[list[str]]:
    """Run nvidia-smi --query-gpu and return parsed rows."""
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader" + ("" if units else ",nounits"),
    ]
    out = _run(cmd)
    if not out.strip():
        return []
    return [line.split(",") for line in out.strip().splitlines() if line.strip()]


def _parse_float(s: str) -> float:
    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else 0.0
