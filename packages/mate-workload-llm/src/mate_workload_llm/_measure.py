from __future__ import annotations

import statistics
from typing import Any, Protocol


class _GenerateResult(Protocol):
    generated_tokens: int
    eval_duration_ns: int
    tokens_per_second: float


class _Engine(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> _GenerateResult: ...


def _aggregate_run(results: list[_GenerateResult]) -> dict[str, float]:
    total_tokens = sum(r.generated_tokens for r in results)
    total_eval_ns = sum(r.eval_duration_ns for r in results)
    tps = total_tokens / (total_eval_ns / 1e9) if total_eval_ns > 0 else 0.0
    return {"tokens_per_second": tps, "total_tokens": total_tokens}


def measure(
    engine: _Engine,
    model: str,
    prompts: list[dict[str, Any]],
    runs: int,
    warmup_runs: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Run the benchmark loop and return (median, std_dev, throttling_detected).

    Each prompt dict must have "text" and "max_tokens" keys.
    """
    all_agg: list[dict[str, float]] = []

    for i in range(warmup_runs + runs):
        run_results: list[_GenerateResult] = []
        for prompt in prompts:
            result = engine.generate(
                model,
                prompt["text"],
                options={"num_predict": prompt["max_tokens"]},
            )
            run_results.append(result)

        if i >= warmup_runs:
            all_agg.append(_aggregate_run(run_results))

    tps_values = [a["tokens_per_second"] for a in all_agg]

    median_tps = statistics.median(tps_values) if tps_values else 0.0
    std_dev_tps = statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0

    cv = std_dev_tps / median_tps if median_tps > 0 else 0.0
    throttling_detected = cv > 0.15

    median_stats = {"tokens_per_second": median_tps}
    std_dev_stats = {"tokens_per_second": std_dev_tps}

    return median_stats, std_dev_stats, throttling_detected
