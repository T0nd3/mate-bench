from __future__ import annotations

import re

# gfx chip code → canonical GPU name
# Used as fallback when rocminfo provides no marketing name.
# When rocminfo does provide a marketing name, _normalize() is used instead.
CHIP_NAMES: dict[str, str] = {
    # RDNA 3 (Navi31/32/33)
    "gfx1100": "RX 7900 XTX",   # Navi31 — also RX 7900 XT, differentiated by marketing name
    "gfx1101": "RX 7800 XT",    # Navi32 — also RX 7700 XT
    "gfx1102": "RX 7600",       # Navi33 — also RX 7600 XT
    "gfx1103": "Radeon 780M",   # Phoenix iGPU
    "gfx1150": "RX 9070 XT",    # Navi48 (RDNA 4)
    "gfx1151": "RX 9070",       # Navi44 (RDNA 4)
    # RDNA 2 (Navi21/22/23/24)
    "gfx1030": "RX 6900 XT",    # Navi21 — also RX 6800 XT, RX 6800
    "gfx1031": "RX 6700 XT",    # Navi22 — also RX 6700
    "gfx1032": "RX 6600 XT",    # Navi23 — also RX 6600
    "gfx1034": "RX 6500 XT",    # Navi24 — also RX 6400
    "gfx1035": "Radeon 680M",   # iGPU
    "gfx1036": "Radeon 660M",   # iGPU
    # RDNA 1 (Navi10/12/14)
    "gfx1010": "RX 5700 XT",    # Navi10 — also RX 5700, RX 5600 XT
    "gfx1011": "RX 5600",       # Navi12 OEM
    "gfx1012": "RX 5500 XT",    # Navi14
    # Vega / CDNA
    "gfx906":  "Radeon VII",    # Vega20 — also MI50, MI60
    "gfx908":  "MI100",
    "gfx90a":  "MI200",         # MI210, MI250, MI250X
    "gfx940":  "MI300A",
    "gfx941":  "MI300X",
    "gfx942":  "MI300X",
}


def normalize(raw: str) -> tuple[str, bool]:
    """Return (normalized_name, is_known_pattern).

    Strips vendor/family prefixes so "AMD Radeon RX 7900 XTX" → "RX 7900 XTX".
    Returns is_known_pattern=False for unrecognized strings.
    """
    if not raw:
        return raw, False
    # "AMD Radeon RX ..." → "RX ..."
    if re.match(r"AMD Radeon RX\s", raw):
        return raw[len("AMD Radeon "):], True
    # "AMD Radeon Pro ..." → "Pro ..."  (workstation cards)
    if re.match(r"AMD Radeon Pro\s", raw):
        return raw[len("AMD Radeon "):], True
    # "AMD Radeon VII / 680M / ..." → "Radeon ..."
    if raw.startswith("AMD Radeon "):
        return raw[len("AMD "):], True
    # "AMD Instinct MI..." → "Instinct MI..."
    if raw.startswith("AMD "):
        return raw[len("AMD "):], True
    return raw, True
