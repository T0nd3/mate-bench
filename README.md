# mate-bench

**MATE — Model AI Throughput Evaluator**

Reproducible LLM inference benchmarks for local hardware. Run a standardized prompt suite, get tokens/sec, submit to the community leaderboard.

```
mate run llm --profile quick
mate submit
```

---

## What it measures

mate-bench runs a fixed set of prompts through a local LLM engine and records **median tokens/second** across multiple runs. Results include GPU info, model identity (with digest hash), and an integrity signature — making them comparable across machines.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- One of the supported models pulled (`ollama pull llama3.2:latest`)

For GPU runtime detection (optional):
- AMD: ROCm 6.x on Linux

---

## Installation

> mate-bench is not yet on PyPI. Install from source using [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/T0nd3/mate-bench
cd mate-bench
uv sync
```

This installs the `mate` CLI and all bundled plugins (Ollama engine, ROCm runtime, LLM workload).

**Future PyPI install (coming soon):**
```bash
pip install mate-bench mate-engine-ollama mate-workload-llm
# AMD GPU:
pip install mate-runtime-rocm
```

---

## Quick start

```bash
# Check what's installed and detected
mate status

# See available engines and pulled models
mate list-engines

# Dry run — shows plan without executing
mate run llm --profile quick --dry-run

# Run the quick profile (llama3.2:latest, ~3 min)
mate run llm --profile quick

# Submit result to the leaderboard
mate submit
```

---

## Profiles

| Profile | Model | Test set | VRAM | ~Time |
|---|---|---|---|---|
| `quick` | llama3.2:latest (3B) | 5 short prompts | 3 GB | 2–4 min |
| `standard` | llama3.1:8b (8B) | 3 medium prompts | 5.5 GB | 5–10 min |
| `full` | both | both test sets | 5.5 GB | 15–25 min |

```bash
mate run llm --profile standard --runs 5
```

Results are saved to your local results directory and can be submitted with `mate submit`.

---

## Result format

Every benchmark produces a YAML file with:

- Hardware info (GPU vendor, name, chip, VRAM, driver)
- Model identity (name, source, digest hash)
- Measurement stats (median tokens/sec, std dev, throttling flag)
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

Results submitted via `mate submit` are stored and aggregated at:

> Leaderboard coming soon

---

## Roadmap

- [ ] NVIDIA CUDA runtime (`mate-runtime-cuda`)
- [ ] Image generation workload (`mate-workload-image`)
- [ ] Speech-to-text workload (`mate-workload-speech`)
- [ ] PyPI release
- [ ] Public leaderboard website
- [ ] HuggingFace engine (`mate-engine-hf`)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
