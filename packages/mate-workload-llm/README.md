# mate-workload-llm

LLM inference throughput workload for [mate-bench](https://github.com/T0nd3/mate-bench).

Measures tokens/second across standardized prompt suites using a fixed set of open-licensed prompts.

```bash
pip install mate-bench mate-engine-ollama mate-workload-llm
mate run llm --profile quick
```

## Profiles

| Profile | Model | VRAM |
|---|---|---|
| `quick` | llama3.2:latest | 3 GB |
| `standard` | llama3.1:8b | 5.5 GB |
| `full` | both | 5.5 GB |

## Test sets

Hosted on Cloudflare R2, SHA256-pinned per release:

| ID | Prompts | License |
|---|---|---|
| `llm-short-v1` | 5 short (200 tok max) | CC0-1.0 |
| `llm-medium-v1` | 3 medium (500 tok max) | CC0-1.0 |
