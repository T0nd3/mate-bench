from __future__ import annotations

import re

import pytest

from mate_bench.schema import (
    BenchmarkResult,
    BenchVersions,
    EngineConfig,
    Measurement,
    ModelInfo,
    Submitter,
    SystemInfo,
    TestSetRef,
    _load_json_schema,
)

_FAKE_HASH = "sha256:" + "a" * 64


def _make_result(**overrides) -> BenchmarkResult:
    defaults = {
        "mode": "closed",
        "workload": "llm",
        "engine": "ollama",
        "engine_version": "0.23.4",
        "model": ModelInfo(name="test", source="ollama", source_ref="test:latest"),
        "profile": "quick",
        "test_set": TestSetRef(hash=_FAKE_HASH, id="llm-short-v1"),
        "engine_config": EngineConfig(is_reference=True),
        "system": SystemInfo(
            gpu_vendor="cpu",
            gpu_name="unknown",
            gpu_chip="unknown",
            vram_gb=0.0,
            runtime="cpu",
            driver="unknown",
            os="Linux",
            kernel="6.7.0",
        ),
        "bench": BenchVersions(core_version="0.1.0"),
        "measurement": Measurement(
            runs=2,
            warmup_runs=1,
            median={"tokens_per_second": 100.0},
            std_dev={"tokens_per_second": 0.5},
            vram_peak_gb=0.0,
            throttling_detected=False,
        ),
        "submitter": Submitter(anonymous_id="test-uuid"),
    }
    defaults.update(overrides)
    return BenchmarkResult.new(**defaults)


class TestLoadJsonSchema:
    def test_schema_file_found(self):
        schema = _load_json_schema(1)
        assert schema["title"] == "mate-bench Result Schema v1"

    def test_schema_has_required_fields(self):
        schema = _load_json_schema(1)
        assert "schema_version" in schema["properties"]
        assert "measurement" in schema["properties"]
        assert "integrity" in schema["properties"]


class TestComputeIntegrity:
    def test_adds_integrity_field(self):
        result = _make_result()
        assert result.integrity is None
        result.compute_integrity()
        assert result.integrity is not None

    def test_hash_has_correct_format(self):
        result = _make_result().compute_integrity()
        assert re.match(r"^sha256:[a-f0-9]{64}$", result.integrity.hash)

    def test_same_result_same_hash(self):
        r1 = _make_result()
        r2 = _make_result()
        r1.submitted_at = r2.submitted_at = "2026-01-01T00:00:00+00:00"
        r1.compute_integrity()
        r2.compute_integrity()
        assert r1.integrity.hash == r2.integrity.hash

    def test_different_measurement_different_hash(self):
        r1 = _make_result()
        r2 = _make_result(
            measurement=Measurement(
                runs=3,
                warmup_runs=1,
                median={"tokens_per_second": 200.0},
                std_dev={"tokens_per_second": 1.0},
                vram_peak_gb=0.0,
                throttling_detected=False,
            )
        )
        r1.submitted_at = r2.submitted_at = "2026-01-01T00:00:00+00:00"
        r1.compute_integrity()
        r2.compute_integrity()
        assert r1.integrity.hash != r2.integrity.hash


class TestValidate:
    def test_valid_result_passes(self):
        result = _make_result().compute_integrity()
        result.validate()

    def test_invalid_mode_fails(self):
        import jsonschema

        result = _make_result()
        result.mode = "invalid"
        result.compute_integrity()
        with pytest.raises(jsonschema.ValidationError):
            result.validate()

    def test_invalid_gpu_vendor_fails(self):
        import jsonschema

        result = _make_result(
            system=SystemInfo(
                gpu_vendor="unknown-vendor",
                gpu_name="X",
                gpu_chip="X",
                vram_gb=0.0,
                runtime="cpu",
                driver="x",
                os="Linux",
                kernel="6.7.0",
            )
        )
        result.compute_integrity()
        with pytest.raises(jsonschema.ValidationError):
            result.validate()


class TestToYaml:
    def test_yaml_contains_schema_version(self):
        import yaml

        result = _make_result().compute_integrity()
        data = yaml.safe_load(result.to_yaml())
        assert data["schema_version"] == 1

    def test_engine_config_settings_flattened(self):
        import yaml

        result = _make_result(
            engine_config=EngineConfig(is_reference=True, settings={"model": "llama3.2:latest"})
        )
        result.compute_integrity()
        data = yaml.safe_load(result.to_yaml())
        assert data["engine_config"]["model"] == "llama3.2:latest"
        assert "settings" not in data["engine_config"]

    def test_integrity_hash_in_yaml(self):
        import yaml

        result = _make_result().compute_integrity()
        data = yaml.safe_load(result.to_yaml())
        assert "integrity" in data
        assert data["integrity"]["hash"].startswith("sha256:")
