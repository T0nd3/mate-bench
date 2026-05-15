from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mate_workload_llm._measure import measure


def _make_result(generated_tokens: int, eval_duration_ns: int) -> MagicMock:
    r = MagicMock()
    r.generated_tokens = generated_tokens
    r.eval_duration_ns = eval_duration_ns
    r.tokens_per_second = (
        generated_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
    )
    return r


PROMPTS = [{"text": "hello", "max_tokens": 10}]


def _engine_with_results(results: list) -> MagicMock:
    engine = MagicMock()
    engine.generate.side_effect = results
    return engine


class TestMeasureStats:
    def test_single_run_median_equals_value(self):
        engine = _engine_with_results(
            [
                _make_result(100, 1_000_000_000),  # 100 t/s
            ]
        )
        median, std_dev, _throttling = measure(engine, "m", PROMPTS, runs=1, warmup_runs=0)
        assert median["tokens_per_second"] == pytest.approx(100.0)
        assert std_dev["tokens_per_second"] == pytest.approx(0.0)

    def test_warmup_runs_not_counted(self):
        # warmup returns 200 t/s, measurement returns 100 t/s — median should be 100
        engine = _engine_with_results(
            [
                _make_result(200, 1_000_000_000),  # warmup
                _make_result(100, 1_000_000_000),  # run 1
            ]
        )
        median, _, _ = measure(engine, "m", PROMPTS, runs=1, warmup_runs=1)
        assert median["tokens_per_second"] == pytest.approx(100.0)
        assert engine.generate.call_count == 2

    def test_median_of_three_runs(self):
        engine = _engine_with_results(
            [
                _make_result(80, 1_000_000_000),  # 80 t/s
                _make_result(100, 1_000_000_000),  # 100 t/s
                _make_result(120, 1_000_000_000),  # 120 t/s
            ]
        )
        median, _, _ = measure(engine, "m", PROMPTS, runs=3, warmup_runs=0)
        assert median["tokens_per_second"] == pytest.approx(100.0)

    def test_std_dev_correct(self):
        import statistics as _stats

        values = [80.0, 100.0, 120.0]
        engine = _engine_with_results(
            [
                _make_result(80, 1_000_000_000),
                _make_result(100, 1_000_000_000),
                _make_result(120, 1_000_000_000),
            ]
        )
        _, std_dev, _ = measure(engine, "m", PROMPTS, runs=3, warmup_runs=0)
        assert std_dev["tokens_per_second"] == pytest.approx(_stats.stdev(values))

    def test_no_throttling_when_cv_below_threshold(self):
        # CV = std/mean; values 98, 100, 102 → std≈2, mean=100, CV=2% → no throttling
        engine = _engine_with_results(
            [
                _make_result(98, 1_000_000_000),
                _make_result(100, 1_000_000_000),
                _make_result(102, 1_000_000_000),
            ]
        )
        _, _, throttling = measure(engine, "m", PROMPTS, runs=3, warmup_runs=0)
        assert throttling is False

    def test_throttling_detected_when_cv_above_threshold(self):
        # 100 t/s then 50 t/s → CV = std/mean = 35% → throttling
        engine = _engine_with_results(
            [
                _make_result(100, 1_000_000_000),
                _make_result(50, 1_000_000_000),
            ]
        )
        _, _, throttling = measure(engine, "m", PROMPTS, runs=2, warmup_runs=0)
        assert throttling is True

    def test_zero_duration_does_not_raise(self):
        engine = _engine_with_results(
            [
                _make_result(0, 0),
            ]
        )
        median, _std_dev, throttling = measure(engine, "m", PROMPTS, runs=1, warmup_runs=0)
        assert median["tokens_per_second"] == pytest.approx(0.0)
        assert throttling is False

    def test_multiple_prompts_aggregated_per_run(self):
        # two prompts, each 50 tokens in 0.5s → 100 t/s combined
        two_prompts = [
            {"text": "p1", "max_tokens": 5},
            {"text": "p2", "max_tokens": 5},
        ]
        engine = _engine_with_results(
            [
                _make_result(50, 500_000_000),  # prompt 1
                _make_result(50, 500_000_000),  # prompt 2
            ]
        )
        median, _, _ = measure(engine, "m", two_prompts, runs=1, warmup_runs=0)
        # 100 tokens / 1s = 100 t/s
        assert median["tokens_per_second"] == pytest.approx(100.0)

    def test_generate_called_with_num_predict_option(self):
        engine = _engine_with_results([_make_result(10, 100_000_000)])
        measure(engine, "mymodel", PROMPTS, runs=1, warmup_runs=0)
        engine.generate.assert_called_once_with(
            "mymodel",
            "hello",
            options={"num_predict": 10},
        )
