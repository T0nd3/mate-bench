from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mate_workload_imagegen._measure import measure

TASKS = [
    {"id": "t001", "prompt": "a red apple", "negative_prompt": "blurry"},
    {"id": "t002", "prompt": "a blue car", "negative_prompt": "blurry"},
]


def _make_engine(generation_time: float) -> MagicMock:
    result = MagicMock()
    result.generation_time_s = generation_time
    engine = MagicMock()
    engine.txt2img.return_value = result
    return engine


class TestMeasure:
    def test_images_per_second(self):
        engine = _make_engine(2.0)
        median, _, _ = measure(
            engine,
            "model.safetensors",
            TASKS,
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler_ancestral",
            runs=1,
            warmup_runs=0,
        )
        # 2 tasks × 2.0s each → total 4.0s, 2 images → 0.5 img/s
        assert median["images_per_second"] == pytest.approx(0.5)

    def test_steps_per_second(self):
        engine = _make_engine(2.0)
        median, _, _ = measure(
            engine,
            "model.safetensors",
            TASKS,
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler_ancestral",
            runs=1,
            warmup_runs=0,
        )
        # avg time_per_image = 2.0s, steps=20 → 10.0 steps/s
        assert median["steps_per_second"] == pytest.approx(10.0)

    def test_resolution_in_output(self):
        engine = _make_engine(1.0)
        median, _, _ = measure(
            engine,
            "m",
            TASKS,
            width=1024,
            height=1024,
            steps=20,
            cfg=7.0,
            sampler="euler_ancestral",
            runs=1,
            warmup_runs=0,
        )
        assert median["resolution"] == "1024x1024"

    def test_warmup_excluded(self):
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.generation_time_s = 1.0
            return r

        engine = MagicMock()
        engine.txt2img.side_effect = side_effect
        measure(
            engine,
            "m",
            TASKS,
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler_ancestral",
            runs=2,
            warmup_runs=1,
        )
        # 1 warmup + 2 runs, each with 2 tasks → 6 calls
        assert call_count == 6

    def test_no_throttling_stable(self):
        engine = _make_engine(1.0)
        _, _, throttling = measure(
            engine,
            "m",
            TASKS,
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler_ancestral",
            runs=3,
            warmup_runs=0,
        )
        assert throttling is False
