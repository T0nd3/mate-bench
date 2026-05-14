from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._names import CHIP_NAMES, normalize


@dataclass
class GpuInfo:
    chip: str           # e.g. "gfx1100"
    name: str           # normalized, e.g. "RX 7900 XTX"
    name_raw: str       # as reported by rocminfo, e.g. "AMD Radeon RX 7900 XTX"
    name_known: bool    # False → unknown chip, stored raw + flagged for review
    vram_gb: float
    rocm_version: str   # e.g. "rocm-6.3.1"
    driver_version: str # e.g. "6.7.0"


def is_rocm_available() -> bool:
    """ROCm is available if the kernel driver device node exists."""
    return Path("/dev/kfd").exists()


def query_gpu(index: int = 0) -> GpuInfo:
    """Query GPU info for the given device index.

    Uses rocminfo for chip/name, rocm-smi for VRAM and driver version,
    and /opt/rocm/.info/version for the ROCm stack version.
    """
    agents = _parse_rocminfo()
    gpu_agents = [a for a in agents if a.get("chip", "").startswith("gfx")]
    if not gpu_agents:
        raise RuntimeError("No AMD GPU found via rocminfo. Is ROCm installed?")
    if index >= len(gpu_agents):
        raise IndexError(f"GPU index {index} out of range ({len(gpu_agents)} GPUs found)")

    agent = gpu_agents[index]
    chip = agent["chip"]
    raw_name = agent.get("marketing_name", "")

    if raw_name:
        name, known = normalize(raw_name)
    elif chip in CHIP_NAMES:
        name = CHIP_NAMES[chip]
        raw_name = name
        known = True
    else:
        name = chip
        raw_name = chip
        known = False

    return GpuInfo(
        chip=chip,
        name=name,
        name_raw=raw_name,
        name_known=known,
        vram_gb=_vram_gb(index),
        rocm_version=_rocm_version(),
        driver_version=_driver_version(),
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _parse_rocminfo() -> list[dict]:
    """Parse rocminfo output into a list of agent dicts."""
    out = _run(["rocminfo"])
    if not out:
        return []

    agents: list[dict] = []
    current: dict = {}
    in_agent = False

    for line in out.splitlines():
        stripped = line.strip()

        if re.match(r"^Agent \d+\s*$", stripped):
            if in_agent:
                agents.append(current)
            current = {}
            in_agent = True
            continue

        if not in_agent:
            continue

        if stripped.startswith("Name:") and "gfx" in stripped:
            m = re.search(r"(gfx\w+)", stripped)
            if m:
                current["chip"] = m.group(1)

        elif stripped.startswith("Marketing Name:"):
            current["marketing_name"] = stripped.split(":", 1)[1].strip()

    if in_agent:
        agents.append(current)

    return agents


def _vram_gb(index: int = 0) -> float:
    """Query total VRAM in GB via rocm-smi."""
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    # Output format: "GPU[N]          : vram Total Memory (B): 25753247744"
    pattern = rf"GPU\[{index}\]\s*:.*Total Memory.*?:\s*(\d+)"
    m = re.search(pattern, out, re.IGNORECASE)
    if m:
        return round(int(m.group(1)) / 1024**3, 1)
    return 0.0


def _rocm_version() -> str:
    # Prefer the version file shipped with the ROCm stack
    for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-utils")):
        if path.exists():
            raw = path.read_text().strip().split("\n")[0]
            version = raw.split("-")[0]   # "6.3.1-45" → "6.3.1"
            return f"rocm-{version}"

    # Fall back to rocm-smi --version
    out = _run(["rocm-smi", "--version"])
    m = re.search(r"(\d+\.\d+[\.\d]*)", out)
    if m:
        return f"rocm-{m.group(1)}"

    return "rocm-unknown"


def _driver_version() -> str:
    out = _run(["rocm-smi", "--showdriverversion"])
    # Match "kernel   6.7.0" or "Driver version: 6.7.0" — avoid matching separator lines
    m = re.search(r"(?:^kernel|driver\s+version\s*:)\s+(\S+)", out, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1)
    return "unknown"
