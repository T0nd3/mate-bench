# mate-engine-piper

[Piper](https://github.com/rhasspy/piper) TTS engine plugin for
[mate-bench](https://github.com/T0nd3/mate-bench).

Piper is an extremely fast, local neural TTS system widely used in
Home Assistant and offline voice applications.

## Usage

```bash
pip install mate-engine-piper
mate-bench run tts --profile quick --mode open --model "/path/to/en_US-lessac-medium.onnx"
```

## Model setup

Download a voice model from the [Piper releases](https://github.com/rhasspy/piper/releases):

```bash
# Example: en_US lessac medium quality
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Then run:
```bash
mate-bench run tts --profile quick --mode open --model "en_US-lessac-medium.onnx"
```

## Requirements

- [piper-tts](https://github.com/rhasspy/piper) >= 1.2
- ONNX Runtime (CPU inference, no GPU required)
