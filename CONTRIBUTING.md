# Contributing to mate-bench

## Development setup

```bash
git clone https://github.com/T0nd3/mate-bench
cd mate-bench
uv sync
```

This installs all packages in editable mode via the uv workspace.

### Run the test suite

```bash
# All packages
uv run pytest packages/

# Single package
uv run --package mate-bench pytest packages/mate-bench/tests/

# Skip integration tests (no Ollama required)
uv run pytest packages/ -m "not integration"

# Integration tests (requires Ollama running)
uv run pytest packages/ -m integration
```

---

## Project structure

```
packages/
  mate-bench/            Core CLI + plugin interfaces
  mate-engine-ollama/    Ollama engine plugin
  mate-runtime-rocm/     AMD ROCm runtime plugin
  mate-workload-llm/     LLM throughput workload plugin
worker/                  Cloudflare Worker (submission API)
schemas/                 JSON Schema for result validation
```

---

## Writing a plugin

All plugins follow the same pattern: a Python package with an entry point in the correct group.

### Engine plugin (`mate_bench.engine`)

Connects to an inference server and runs generation requests.

```python
# pyproject.toml
[project.entry-points."mate_bench.engine"]
my-engine = "mate_engine_myengine:MyEngine"
```

```python
from mate_bench.plugin import EnginePlugin, PluginManifest
from mate_bench.schema import ModelInfo

class MyEngine:
    name = "my-engine"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    supported_runtimes = ["cuda", "rocm", "cpu"]

    def is_available(self) -> bool: ...
    def version(self) -> str: ...
    def list_models(self) -> list[dict]: ...
    def model_info(self, name: str) -> ModelInfo: ...
    def generate(self, model: str, prompt: str, options=None) -> GenerateResult: ...
    def pull(self, model: str, on_progress=None) -> None: ...
```

`generate()` must return an object with at least:
- `generated_tokens: int`
- `eval_duration_ns: int`
- `tokens_per_second: float`

### Runtime plugin (`mate_bench.runtime`)

Detects GPU hardware and reports its capabilities.

```python
# pyproject.toml
[project.entry-points."mate_bench.runtime"]
cuda = "mate_runtime_cuda:CudaRuntime"
```

```python
class CudaRuntime:
    name = "cuda"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)

    def is_available(self) -> bool: ...

    def gpu_info(self) -> dict:
        return {
            "gpu_vendor": "nvidia",   # amd | nvidia | intel | cpu
            "gpu_name": "RTX 4090",
            "gpu_chip": "AD102",
            "vram_gb": 24.0,
            "runtime": "CUDA 12.4",
            "driver": "550.54",
            # optional debug fields (prefixed with _):
            "_gpu_name_known": True,
        }
```

### Workload plugin (`mate_bench.workload`)

Defines a benchmark task and runs it against an engine.

```python
# pyproject.toml
[project.entry-points."mate_bench.workload"]
image = "mate_workload_image:ImageWorkload"
```

```python
from mate_bench.plugin import (
    WorkloadPlugin, PluginManifest, ProfileConfig,
    TestSetSpec, Measurement, Mode, EnginePlugin,
)

class ImageWorkload:
    name = "image"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    profiles: dict[str, ProfileConfig] = { ... }
    test_sets: dict[str, TestSetSpec] = { ... }

    def estimate_download(self, profile: str) -> int: ...
    def estimate_vram(self, profile: str) -> int: ...
    def estimate_runtime(self, profile: str) -> int: ...
    def setup_closed(self, profile: str) -> None: ...
    def setup_open(self, profile: str, user_inputs: dict) -> None: ...
    def run(self, profile, mode, engine, runs, warmup_runs) -> Measurement: ...
    def cleanup(self, profile: str) -> None: ...
```

---

## Adding test sets

Test sets for closed-mode benchmarks are JSON files with this structure:

```json
{
  "id": "my-workload-v1",
  "version": 1,
  "license": "CC0-1.0",
  "source": "...",
  "description": "...",
  "prompts": [
    { "id": "p01", "text": "...", "max_tokens": 200 }
  ]
}
```

Files should be:
- Licensed CC0 (public domain) so results can be shared freely
- Bundled in the workload package under `data/test-sets/`
- Uploaded to the CDN and SHA256-pinned in `TestSetSpec`

---

## Result schema

The result format is defined in [`schemas/result-v1.json`](schemas/result-v1.json).  
Python validation: `BenchmarkResult.validate()` in `mate_bench.schema`.  
Worker validation: `worker/src/validate.ts`.

When bumping the schema version, update both validators and add a migration to `worker/schema/`.

---

## Code style

- Python: no formatter enforced, follow existing style (type hints, dataclasses, no comments unless non-obvious)
- TypeScript: standard strict mode, no formatter enforced
- Tests: pytest for Python, no JS tests yet
- Commit messages: imperative mood, no Co-authored-by trailers
