from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Protocol


class _ImageResult(Protocol):
    generation_time_s: float


class _ImageEngine(Protocol):
    def txt2img(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        width: int,
        height: int,
        cfg: float,
        sampler: str,
        model: str,
        seed: int,
    ) -> _ImageResult: ...


@dataclass
class _RunStats:
    images_per_second: float
    steps_per_second: float
    time_per_image_s: float


def _aggregate_run(results: list[_ImageResult], steps: int) -> _RunStats:
    """Compute images/s, steps/s and time/image from a single run's results."""
    total_time = sum(r.generation_time_s for r in results)
    n = len(results)
    time_per_image = total_time / n if n > 0 else 0.0
    images_per_second = n / total_time if total_time > 0 else 0.0
    steps_per_second = steps / time_per_image if time_per_image > 0 else 0.0
    return _RunStats(
        images_per_second=images_per_second,
        steps_per_second=steps_per_second,
        time_per_image_s=time_per_image,
    )


def measure(
    engine: _ImageEngine,
    model: str,
    tasks: list[dict[str, Any]],
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    runs: int,
    warmup_runs: int,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Run the image-gen benchmark loop; return (median, std_dev, throttling_detected)."""
    all_stats: list[_RunStats] = []

    for i in range(warmup_runs + runs):
        run_results: list[_ImageResult] = []
        for task in tasks:
            result = engine.txt2img(
                prompt=task["prompt"],
                negative_prompt=task.get("negative_prompt", ""),
                steps=steps,
                width=width,
                height=height,
                cfg=cfg,
                sampler=sampler,
                model=model,
                seed=seed,
            )
            run_results.append(result)

        if i >= warmup_runs:
            all_stats.append(_aggregate_run(run_results, steps))

    ips_values = [s.images_per_second for s in all_stats]
    sps_values = [s.steps_per_second for s in all_stats]
    tpi_values = [s.time_per_image_s for s in all_stats]

    median_ips = statistics.median(ips_values) if ips_values else 0.0
    median_sps = statistics.median(sps_values) if sps_values else 0.0
    median_tpi = statistics.median(tpi_values) if tpi_values else 0.0
    std_ips = statistics.stdev(ips_values) if len(ips_values) > 1 else 0.0
    std_sps = statistics.stdev(sps_values) if len(sps_values) > 1 else 0.0
    std_tpi = statistics.stdev(tpi_values) if len(tpi_values) > 1 else 0.0

    cv = std_ips / median_ips if median_ips > 0 else 0.0
    throttling_detected = cv > 0.15

    median_stats: dict[str, Any] = {
        "images_per_second": median_ips,
        "steps_per_second": median_sps,
        "time_per_image_s": median_tpi,
        "resolution": f"{width}x{height}",
        "steps": steps,
    }
    std_dev_stats: dict[str, Any] = {
        "images_per_second": std_ips,
        "steps_per_second": std_sps,
        "time_per_image_s": std_tpi,
    }

    return median_stats, std_dev_stats, throttling_detected
