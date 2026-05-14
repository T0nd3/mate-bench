# mate-engine-ollama

[Ollama](https://ollama.com) engine plugin for [mate-bench](https://github.com/T0nd3/mate-bench).

```bash
pip install mate-bench mate-engine-ollama mate-workload-llm
ollama pull llama3.2:latest
mate run llm --profile quick
```

## Requirements

- Ollama running locally (`ollama serve`)
- At least one model pulled

## Supported runtimes

Works with any runtime: AMD ROCm, NVIDIA CUDA, Apple Metal, CPU.
