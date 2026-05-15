from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mate_workload_llm import LlmWorkload
from mate_bench.plugin import Mode


class TestLlmWorkloadMeta:
    def test_profiles_defined(self):
        w = LlmWorkload()
        assert "quick" in w.profiles
        assert "standard" in w.profiles
        assert "full" in w.profiles

    def test_estimate_download_quick(self):
        w = LlmWorkload()
        assert w.estimate_download("quick") > 0

    def test_estimate_vram_quick_under_4gb(self):
        w = LlmWorkload()
        # quick profile targets llama3.2:latest, ~3 GB
        assert w.estimate_vram("quick") < 4 * 1024**3

    def test_estimate_runtime_positive(self):
        w = LlmWorkload()
        assert w.estimate_runtime("standard") > 0


class TestSetupClosed:
    def test_copies_bundled_json_to_cache(self, tmp_path):
        w = LlmWorkload()
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            w.setup_closed("quick")
        assert (tmp_path / "llm" / "llm-short-v1.json").exists()

    def test_idempotent_second_call(self, tmp_path):
        w = LlmWorkload()
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            w.setup_closed("quick")
            w.setup_closed("quick")
        assert (tmp_path / "llm" / "llm-short-v1.json").exists()


class TestCleanup:
    def test_removes_cached_file(self, tmp_path):
        w = LlmWorkload()
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            w.setup_closed("quick")
            assert (tmp_path / "llm" / "llm-short-v1.json").exists()
            w.cleanup("quick")
            assert not (tmp_path / "llm" / "llm-short-v1.json").exists()

    def test_cleanup_noop_when_file_absent(self, tmp_path):
        w = LlmWorkload()
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            w.cleanup("quick")  # should not raise


class TestRun:
    def _make_engine(self, tps=100.0):
        engine = MagicMock()
        result = MagicMock()
        result.generated_tokens = 100
        result.eval_duration_ns = 1_000_000_000
        result.tokens_per_second = tps
        engine.generate.return_value = result
        return engine

    def test_run_returns_measurement(self, tmp_path):
        w = LlmWorkload()
        engine = self._make_engine()
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            measurement = w.run("quick", Mode.CLOSED, engine, runs=2, warmup_runs=1)
        assert measurement.runs == 2
        assert measurement.warmup_runs == 1
        assert "tokens_per_second" in measurement.median
        assert "tokens_per_second" in measurement.std_dev

    def test_run_open_mode_without_setup_raises(self, tmp_path):
        w = LlmWorkload()
        engine = self._make_engine()
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="setup_open"):
                w.run("quick", Mode.OPEN, engine, runs=1, warmup_runs=0)

    def test_run_open_mode_uses_user_model(self, tmp_path):
        w = LlmWorkload()
        engine = self._make_engine()
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            w.setup_open("quick", {"model": "deepseek-r1:14b"})
            measurement = w.run("quick", Mode.OPEN, engine, runs=1, warmup_runs=0)
        assert "tokens_per_second" in measurement.median
        # engine was called with the user-specified model
        call_args = engine.generate.call_args_list
        assert all(args[0][0] == "deepseek-r1:14b" for args in call_args)

    def test_warmup_calls_not_in_stats(self, tmp_path):
        # warmup returns 200 t/s, measurement returns 50 t/s
        w = LlmWorkload()
        results = []
        # quick has 5 prompts; 1 warmup run + 1 measurement run = 10 generate calls
        for _ in range(5):
            r = MagicMock()
            r.generated_tokens = 200
            r.eval_duration_ns = 1_000_000_000  # 200 t/s
            results.append(r)
        for _ in range(5):
            r = MagicMock()
            r.generated_tokens = 50
            r.eval_duration_ns = 1_000_000_000  # 50 t/s
            results.append(r)
        engine = MagicMock()
        engine.generate.side_effect = results
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            measurement = w.run("quick", Mode.CLOSED, engine, runs=1, warmup_runs=1)
        # only the 50 t/s run should be in the median
        assert measurement.median["tokens_per_second"] == pytest.approx(50.0)


@pytest.mark.integration
class TestLlmWorkloadIntegration:
    """Requires a running Ollama server with llama3.2:latest pulled."""

    def test_quick_profile_runs(self, tmp_path):
        from mate_engine_ollama import OllamaEngine
        w = LlmWorkload()
        engine = OllamaEngine()
        assert engine.is_available(), "Ollama must be running for integration tests"
        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            measurement = w.run("quick", Mode.CLOSED, engine, runs=2, warmup_runs=1)
        assert measurement.runs == 2
        assert measurement.median["tokens_per_second"] > 0
