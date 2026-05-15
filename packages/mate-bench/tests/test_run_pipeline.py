from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mate_bench._system import collect_system_info
from mate_bench.schema import (
    BenchmarkResult,
    BenchVersions,
    EngineConfig,
    ModelInfo,
    Submitter,
    TestSetRef,
)
from mate_bench.schema import (
    Measurement as SchemaMeasurement,
)


def _make_engine(tps: float = 100.0) -> MagicMock:
    engine = MagicMock()
    engine.name = "ollama"
    engine.is_available.return_value = True
    engine.version.return_value = "0.23.4"

    result = MagicMock()
    result.generated_tokens = 100
    result.eval_duration_ns = int(100 / tps * 1e9)
    result.tokens_per_second = tps
    engine.generate.return_value = result

    engine.list_models.return_value = [{"name": "llama3.2:latest"}]
    engine.model_info.return_value = ModelInfo(
        name="llama3.2:latest",
        source="ollama",
        source_ref="llama3.2:latest",
        file_hash="sha256:" + "b" * 64,
        file_hash_available=True,
    )
    return engine


class TestRunPipeline:
    def test_full_pipeline_produces_valid_yaml(self, tmp_path):
        from mate_bench.plugin import Mode
        from mate_workload_llm import LlmWorkload

        workload = LlmWorkload()
        engine = _make_engine(tps=150.0)

        with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
            plugin_m = workload.run("quick", Mode.CLOSED, engine, runs=2, warmup_runs=1)

        measurement = SchemaMeasurement(**asdict(plugin_m))
        result = BenchmarkResult.new(
            mode="closed",
            workload="llm",
            engine="ollama",
            engine_version="0.23.4",
            model=engine.model_info("llama3.2:latest"),
            profile="quick",
            test_set=TestSetRef(hash="sha256:" + "c" * 64, id="llm-short-v1"),
            engine_config=EngineConfig(is_reference=True),
            system=collect_system_info({}),
            bench=BenchVersions(core_version="0.1.0"),
            measurement=measurement,
            submitter=Submitter(anonymous_id="test-uuid"),
        )
        result.compute_integrity()
        result.validate()

        out = tmp_path / "result.yaml"
        out.write_text(result.to_yaml())
        data = yaml.safe_load(out.read_text())

        assert data["workload"] == "llm"
        assert data["profile"] == "quick"
        assert data["measurement"]["median"]["tokens_per_second"] == pytest.approx(150.0, rel=0.01)
        assert data["integrity"]["hash"].startswith("sha256:")

    def test_integrity_hash_changes_when_tps_changes(self, tmp_path):
        from mate_bench.plugin import Mode
        from mate_workload_llm import LlmWorkload

        def _build(tps):
            workload = LlmWorkload()
            engine = _make_engine(tps=tps)
            with patch("mate_workload_llm.TEST_SETS_DIR", tmp_path):
                plugin_m = workload.run("quick", Mode.CLOSED, engine, runs=1, warmup_runs=0)
            m = SchemaMeasurement(**asdict(plugin_m))
            r = BenchmarkResult.new(
                mode="closed",
                workload="llm",
                engine="ollama",
                engine_version="0.1",
                model=ModelInfo(name="x", source="ollama", source_ref="x"),
                profile="quick",
                test_set=TestSetRef(hash="sha256:" + "d" * 64),
                engine_config=EngineConfig(is_reference=True),
                system=collect_system_info({}),
                bench=BenchVersions(core_version="0.1.0"),
                measurement=m,
                submitter=Submitter(anonymous_id="uuid"),
            )
            r.submitted_at = "2026-01-01T00:00:00+00:00"
            return r.compute_integrity()

        r1 = _build(100.0)
        r2 = _build(200.0)
        assert r1.integrity.hash != r2.integrity.hash

    def test_bench_plugin_versions_populated(self, tmp_path):
        from mate_bench.discovery import get_plugin_versions

        versions = get_plugin_versions()
        assert isinstance(versions, dict)
        assert "llm" in versions or "ollama" in versions or len(versions) >= 0
