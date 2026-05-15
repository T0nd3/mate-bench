# mate-engine-kokoro

[Kokoro](https://github.com/hexgrad/kokoro) TTS engine plugin for
[mate-bench](https://github.com/T0nd3/mate-bench).

Kokoro is a lightweight, high-quality TTS model (~82M parameters, ~350 MB).
Supports GPU inference via PyTorch.

## Usage

```bash
pip install mate-engine-kokoro
mate-bench run tts --profile quick
```

## Voices

Default closed-mode voice: `af_heart` (American English, female).

Available voices include: `af_heart`, `af_bella`, `af_sarah`, `am_adam`,
`am_michael`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`.

## Requirements

- [kokoro](https://github.com/hexgrad/kokoro) >= 0.9
- PyTorch (for GPU inference)
