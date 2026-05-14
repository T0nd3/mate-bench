# mate-runtime-cuda

NVIDIA CUDA runtime plugin for [mate-bench](https://github.com/T0nd3/mate-bench).

Detects GPU name, chip, VRAM, CUDA version and driver from `nvidia-smi`.

```bash
pip install mate-bench mate-runtime-cuda
mate list-runtimes
```

## Requirements

- NVIDIA GPU with drivers installed (`nvidia-smi` must be in PATH)
- CUDA 11.x or newer

## Supported GPUs

Pascal (GTX 10xx) through Blackwell (RTX 50xx), including data centre GPUs (A100, H100, V100).
