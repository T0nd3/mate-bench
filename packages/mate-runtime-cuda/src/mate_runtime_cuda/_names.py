from __future__ import annotations

import re

# (substring to match in normalized name, chip codename)
# Ordered from most specific to least specific.
_GPU_CHIP_MAP: list[tuple[str, str]] = [
    # Blackwell — RTX 50xx
    ("RTX 5090",        "GB202"),
    ("RTX 5080",        "GB203"),
    ("RTX 5070 Ti",     "GB203"),
    ("RTX 5070",        "GB205"),
    ("RTX 5060 Ti",     "GB206"),
    ("RTX 5060",        "GB206"),
    # Ada Lovelace — RTX 40xx + professional
    ("RTX 6000 Ada",    "AD102"),
    ("RTX 4090",        "AD102"),
    ("RTX 4080 Super",  "AD103"),
    ("RTX 4080",        "AD103"),
    ("RTX 4070 Ti Super", "AD103"),
    ("RTX 4070 Ti",     "AD104"),
    ("RTX 4070 Super",  "AD104"),
    ("RTX 4070",        "AD104"),
    ("RTX 4500 Ada",    "AD104"),
    ("RTX 4000 Ada",    "AD104"),
    ("RTX 4060 Ti",     "AD106"),
    ("RTX 4060",        "AD107"),
    # Hopper
    ("H200",            "GH200"),
    ("H100",            "GH100"),
    # Ampere — data centre
    ("A100",            "GA100"),
    ("A800",            "GA100"),
    ("RTX A6000",       "GA102"),
    ("RTX A5000",       "GA102"),
    ("RTX A4000",       "GA104"),
    ("RTX A3000",       "GA104"),
    ("RTX A2000",       "GA106"),
    # Ampere — RTX 30xx
    ("RTX 3090 Ti",     "GA102"),
    ("RTX 3090",        "GA102"),
    ("RTX 3080 Ti",     "GA102"),
    ("RTX 3080",        "GA102"),
    ("RTX 3070 Ti",     "GA104"),
    ("RTX 3070",        "GA104"),
    ("RTX 3060 Ti",     "GA104"),
    ("RTX 3060",        "GA106"),
    ("RTX 3050",        "GA107"),
    # Volta
    ("Titan V",         "GV100"),
    ("V100",            "GV100"),
    # Turing — RTX 20xx
    ("RTX 2080 Ti",     "TU102"),
    ("RTX 2080 Super",  "TU104"),
    ("RTX 2080",        "TU104"),
    ("RTX 2070 Super",  "TU104"),
    ("RTX 2070",        "TU106"),
    ("RTX 2060 Super",  "TU106"),
    ("RTX 2060",        "TU106"),
    # Turing — GTX 16xx
    ("GTX 1660 Ti",     "TU116"),
    ("GTX 1660 Super",  "TU116"),
    ("GTX 1660",        "TU116"),
    ("GTX 1650 Super",  "TU116"),
    ("GTX 1650",        "TU117"),
    # Pascal — GTX 10xx
    ("Titan Xp",        "GP102"),
    ("GTX 1080 Ti",     "GP102"),
    ("GTX 1080",        "GP104"),
    ("GTX 1070 Ti",     "GP104"),
    ("GTX 1070",        "GP104"),
    ("GTX 1060",        "GP106"),
    ("GTX 1050 Ti",     "GP107"),
    ("GTX 1050",        "GP107"),
]

_STRIP_PREFIX = re.compile(
    r"^(?:NVIDIA\s+)?(?:GeForce\s+|Tesla\s+|Quadro\s+)?", re.IGNORECASE
)


def normalize(raw: str) -> tuple[str, bool]:
    """Return (display_name, is_known_chip).

    Strips vendor/product-line prefixes and looks up the die codename.
    """
    display = _STRIP_PREFIX.sub("", raw).strip()
    chip, known = chip_from_name(display)
    return display, known


def chip_from_name(normalized: str) -> tuple[str, bool]:
    """Return (chip_codename, is_known) for a normalized GPU name."""
    for substring, chip in _GPU_CHIP_MAP:
        if substring.lower() in normalized.lower():
            return chip, True
    return normalized, False
