# mate-engine-faster-whisper

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) engine plugin for
[mate-bench](https://github.com/T0nd3/mate-bench).

Wraps `WhisperModel` with automatic device selection (CUDA / CPU) and model caching
so repeated calls within a benchmark run avoid re-loading weights.

## Usage

Registered automatically as `faster-whisper` when installed alongside mate-bench:

```bash
pip install mate-engine-faster-whisper
mate-bench run stt --profile quick --engine faster-whisper
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `device` | `auto` | `cuda`, `cpu`, or `auto` (detected via PyTorch) |
| `compute_type` | `auto` | `float16` on CUDA, `int8` on CPU |
| `language` | `en` | BCP-47 language code passed to Whisper |

## Requirements

- CUDA toolkit for GPU inference
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) >= 1.0
