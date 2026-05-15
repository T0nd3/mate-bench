# mate-workload-stt

Speech-to-text workload plugin for [mate-bench](https://github.com/T0nd3/mate-bench).

Benchmarks STT transcription speed and accuracy using LibriSpeech test-clean audio clips
with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Metrics

| Metric | Description |
|--------|-------------|
| `rtf` | Real-Time Factor — `processing_time / audio_duration` (lower is better) |
| `wer` | Word Error Rate — edit distance / total reference words (lower is better) |
| `total_audio_seconds` | Total audio processed per run |

## Profiles

| Profile | Clips | Audio | Model |
|---------|-------|-------|-------|
| `quick` | 5 | ~50 s | whisper-large-v3 |
| `standard` | 20 | ~200 s | whisper-large-v3 |

## Usage

```bash
mate-bench run stt --profile quick
mate-bench run stt --profile standard
```

## Test data

[LibriSpeech test-clean](https://openslr.org/12) (Panayotov et al., 2015).
Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
