# mate-runtime-rocm

AMD ROCm runtime plugin for [mate-bench](https://github.com/T0nd3/mate-bench).

Detects GPU name, chip, VRAM, ROCm version and driver from `rocminfo` and `rocm-smi`.

```bash
pip install mate-bench mate-runtime-rocm
mate list-runtimes
```

## Requirements

- Linux with ROCm 6.x installed
- AMD GPU with supported chip (RDNA1/2/3, CDNA, Vega)
