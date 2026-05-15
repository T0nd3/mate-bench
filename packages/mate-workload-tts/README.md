# mate-workload-tts

Text-to-speech workload plugin for [mate-bench](https://github.com/T0nd3/mate-bench).

Measures TTS synthesis speed using a bundled CC0 sentence corpus.

## Metrics

| Metric | Description |
|--------|-------------|
| `rtf` | Real-Time Factor — `processing_time / audio_duration` (lower is better) |
| `chars_per_second` | Characters synthesised per second (higher is better) |
| `total_audio_seconds` | Total audio generated per run |

## Profiles

| Profile | Sentences | Approx. words | Reference model |
|---------|-----------|---------------|-----------------|
| `quick` | 5 | ~60 | kokoro-v1.0 / af_heart |
| `standard` | 20 | ~240 | kokoro-v1.0 / af_heart |

## Usage

```bash
# Closed mode (reference: kokoro-v1.0 / af_heart)
mate-bench run tts --profile quick

# Open mode (your model/voice)
mate-bench run tts --profile quick --mode open --model "en_US-lessac-medium"
```
