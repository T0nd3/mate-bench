# mate-bench

Core CLI and plugin interfaces for **MATE — Model AI Throughput Evaluator**.

Install this package together with at least one engine plugin and one workload plugin:

```bash
pip install mate-bench mate-engine-ollama mate-workload-llm
```

See the [main README](https://github.com/T0nd3/mate-bench) for full documentation.

## Usage

```bash
mate run llm --profile quick
mate submit
mate status
```
