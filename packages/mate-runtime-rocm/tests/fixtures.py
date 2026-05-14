# Sample outputs from real ROCm 6.x systems — used in tests to avoid needing real hardware.

ROCMINFO_RX7900XTX = """\
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
System Timestamp Freq.:  1000.000000MHz

==========
Agent 1
==========
  Name:                    cpu-0
  Uuid:                    CPU-XX
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
  Vendor Name:             CPU

==========
Agent 2
==========
  Name:                    gfx1100
  Uuid:                    GPU-7f3a1b2c3d4e5f60
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 FULL_PROFILE

==========
Agent 3
==========
"""

ROCMINFO_RX6700XT = """\
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.1

==========
Agent 1
==========
  Name:                    cpu-0
  Uuid:                    CPU-XX
  Marketing Name:          AMD Ryzen 5 5600X 6-Core Processor
  Vendor Name:             CPU

==========
Agent 2
==========
  Name:                    gfx1031
  Uuid:                    GPU-aabbccddeeff0011
  Marketing Name:          AMD Radeon RX 6700 XT
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH

==========
Agent 3
==========
"""

ROCMINFO_UNKNOWN_CHIP = """\
ROCk module is loaded

==========
Agent 1
==========
  Name:                    gfx9999
  Uuid:                    GPU-unknown
  Marketing Name:
  Vendor Name:             AMD

==========
Agent 2
==========
"""

ROCMINFO_NO_GPU = """\
ROCk module is loaded
=====================
HSA System Attributes
=====================

==========
Agent 1
==========
  Name:                    cpu-0
  Uuid:                    CPU-XX
  Marketing Name:          Intel Core i9
  Vendor Name:             CPU

==========
Agent 2
==========
"""

ROCM_SMI_VRAM_7900XTX = """\
======================= ROCm System Management Interface =======================
=================================== Memory ====================================
GPU[0]          : vram Total Memory (B): 25753247744
GPU[0]          : vram Total Used Memory (B): 524288000
================================================================================
============================= End of ROCm SMI Log ==============================
"""

ROCM_SMI_VRAM_6700XT = """\
======================= ROCm System Management Interface =======================
=================================== Memory ====================================
GPU[0]          : vram Total Memory (B): 12868804608
GPU[0]          : vram Total Used Memory (B): 0
================================================================================
============================= End of ROCm SMI Log ==============================
"""

ROCM_SMI_DRIVER = """\
======================= ROCm System Management Interface =======================
============================== Driver version =================================
kernel                    6.7.0
================================================================================
============================= End of ROCm SMI Log ==============================
"""

ROCM_VERSION_FILE = "6.3.1-45\n"
