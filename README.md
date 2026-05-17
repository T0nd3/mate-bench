# mate-bench

**MATE — Model AI Throughput Evaluator**

Reproducible AI inference benchmarks for local hardware. Measure tokens/sec, speech RTF, image throughput and TTS speed — then submit results to the community leaderboard.

```bash
pip install mate-bench mate-engine-ollama mate-workload-llm
mate run llm --profile quick
mate submit
```

---

## What it measures

| Workload | Metric | Engine |
|---|---|---|
| `llm` | tokens / second | Ollama |
| `stt` | real-time factor (RTF), word error rate | faster-whisper |
| `imagegen` | images / second, steps / second | ComfyUI |
| `tts` | real-time factor (RTF), chars / second | Kokoro, Piper |

Results include GPU info, model identity (with digest hash), and an integrity signature — making them comparable across machines.

---

## Requirements

- Python 3.11+
- Engine running locally (e.g. [Ollama](https://ollama.com) for LLM)

---

## Installation

Install the core CLI plus the workload and engine plugins you need:

```bash
# LLM benchmarks (Ollama)
pip install mate-bench mate-engine-ollama mate-workload-llm

# Speech-to-text benchmarks (faster-whisper)
pip install mate-bench mate-engine-faster-whisper mate-workload-stt

# Image generation benchmarks (ComfyUI)
pip install mate-bench mate-engine-comfyui mate-workload-imagegen

# Text-to-speech benchmarks (Kokoro)
pip install mate-bench mate-engine-kokoro mate-workload-tts
```

GPU runtime detection (optional):
```bash
pip install mate-runtime-rocm   # AMD ROCm
pip install mate-runtime-cuda   # NVIDIA CUDA
```

---

## Quick start

```bash
# Check what's installed and detected
mate status

# Dry run — shows plan without executing
mate run llm --profile quick --dry-run

# Run the quick LLM profile (~2–4 min)
mate run llm --profile quick

# Submit result to the leaderboard
mate submit
```

---

## Profiles

### LLM

| Profile | Model | VRAM | ~Time |
|---|---|---|---|
| `quick` | llama3.2:latest (3B) | 3 GB | 2–4 min |
| `standard` | llama3.1:8b (8B) | 5.5 GB | 5–10 min |
| `full` | both | 5.5 GB | 15–25 min |

### STT

| Profile | Test set | ~Time |
|---|---|---|
| `quick` | 5 LibriSpeech clips | 1–2 min |
| `standard` | 20 LibriSpeech clips | 5–8 min |

### TTS

| Profile | Test set | Engine |
|---|---|---|
| `quick` | 5 sentences | Kokoro (reference) |
| `standard` | 20 sentences | Kokoro (reference) |

---

## Result format

Every benchmark produces a YAML file with:

- Hardware info (GPU vendor, name, chip, VRAM, driver)
- Model identity (name, source, digest hash)
- Measurement stats (median + std dev, throttling flag)
- Integrity hash (SHA-256 of all fields — detects accidental edits)

Example snippet:
```yaml
workload: llm
profile: quick
model:
  name: llama3.2:latest
  source: ollama
  file_hash: sha256:a80c4f...
measurement:
  runs: 5
  median:
    tokens_per_second: 204.8
  throttling_detected: false
```

---

## Plugin architecture

mate-bench is a thin orchestration core. Everything else is a plugin:

| Group | Entry point | Example |
|---|---|---|
| Workload | `mate_bench.workload` | `mate-workload-llm` |
| Engine | `mate_bench.engine` | `mate-engine-ollama` |
| Runtime | `mate_bench.runtime` | `mate-runtime-rocm` |

Plugins are discovered via Python entry points — install any `mate-engine-*`, `mate-runtime-*`, or `mate-workload-*` package and it appears automatically in `mate status`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to write your own plugin.

---

## CLI reference

```
mate run <workload>        Run benchmark
  --profile quick|standard|full
  --runs N                 Measurement runs (default: 5)
  --warmup N               Warmup runs (default: 1)
  --engine NAME            Override default engine
  --mode closed|open       Benchmark mode (default: closed)
  --model NAME             Model name for open mode
  --local                  Only use already-pulled models
  --dry-run                Show plan without executing
  --output FILE            Save result to specific path

mate submit [FILE]         Submit result to leaderboard
  --print                  Print YAML to stdout
  --discord                Show YAML for Discord submission

mate cleanup [WORKLOAD]    Remove cached test sets
mate config                Set default profile
mate status                Show installed plugins
mate list-engines          Show engines and pulled models
mate list-runtimes         Show runtime / GPU info
mate list-workloads        Show installed workloads
mate list-test-sets        Show cached test sets
```

---

## Leaderboard

Results submitted via `mate submit` are aggregated at:

**https://t0nd3.github.io/mate-bench-leaderboard/**

---

## Roadmap

- [x] LLM workload — tokens/sec via Ollama
- [x] STT workload — RTF + WER via faster-whisper
- [x] Image generation workload — throughput via ComfyUI
- [x] TTS workload — RTF + chars/s via Kokoro / Piper
- [x] NVIDIA CUDA runtime
- [x] AMD ROCm runtime
- [x] Public leaderboard
- [x] PyPI release
- [ ] HuggingFace engine (`mate-engine-hf`)
- [ ] VRAM peak polling during benchmark
- [ ] Leaderboard TTS section

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
