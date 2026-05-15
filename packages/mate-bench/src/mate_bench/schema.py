from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import jsonschema
import yaml

SCHEMA_VERSION = 1
_SCHEMA_DIR = Path(__file__).parents[4] / "schemas"


def _load_json_schema(version: int) -> dict:
    path = _SCHEMA_DIR / f"result-v{version}.json"
    with path.open() as f:
        return json.load(f)


@dataclass
class ModelInfo:
    name: str
    source: str  # ollama | huggingface | local
    source_ref: str
    file_hash: str | None = None
    file_hash_available: bool = False


@dataclass
class TestSetRef:
    hash: str
    id: str | None = None
    duration_seconds: float | None = None
    custom: bool = False
    files: list[str] = field(default_factory=list)


@dataclass
class EngineConfig:
    is_reference: bool
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemInfo:
    gpu_vendor: str
    gpu_name: str
    gpu_chip: str
    vram_gb: float
    runtime: str
    driver: str
    os: str
    kernel: str


@dataclass
class BenchVersions:
    core_version: str
    plugin_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class Measurement:
    runs: int
    warmup_runs: int
    median: dict[str, Any]
    std_dev: dict[str, Any]
    vram_peak_gb: float
    throttling_detected: bool


@dataclass
class Submitter:
    anonymous_id: str


@dataclass
class Integrity:
    hash: str


@dataclass
class BenchmarkResult:
    schema_version: int
    submitted_at: str
    mode: Literal["closed", "open"]
    workload: str
    engine: str
    engine_version: str
    model: ModelInfo
    profile: str
    test_set: TestSetRef
    engine_config: EngineConfig
    system: SystemInfo
    bench: BenchVersions
    measurement: Measurement
    submitter: Submitter
    integrity: Integrity | None = None

    @classmethod
    def new(
        cls,
        *,
        mode: Literal["closed", "open"],
        workload: str,
        engine: str,
        engine_version: str,
        model: ModelInfo,
        profile: str,
        test_set: TestSetRef,
        engine_config: EngineConfig,
        system: SystemInfo,
        bench: BenchVersions,
        measurement: Measurement,
        submitter: Submitter,
    ) -> BenchmarkResult:
        return cls(
            schema_version=SCHEMA_VERSION,
            submitted_at=datetime.now(UTC).isoformat(),
            mode=mode,
            workload=workload,
            engine=engine,
            engine_version=engine_version,
            model=model,
            profile=profile,
            test_set=test_set,
            engine_config=engine_config,
            system=system,
            bench=bench,
            measurement=measurement,
            submitter=submitter,
        )

    def _to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Flatten engine_config.settings into the engine_config dict
        settings = d["engine_config"].pop("settings", {})
        d["engine_config"].update(settings)
        d.pop("integrity", None)
        return d

    def compute_integrity(self) -> BenchmarkResult:
        serialized = json.dumps(self._to_dict(), sort_keys=True, default=str)
        h = hashlib.sha256(serialized.encode()).hexdigest()
        self.integrity = Integrity(hash=f"sha256:{h}")
        return self

    def validate(self) -> None:
        schema = _load_json_schema(self.schema_version)
        d = self._to_dict()
        if self.integrity:
            d["integrity"] = asdict(self.integrity)
        jsonschema.validate(d, schema)

    def to_yaml(self) -> str:
        d = self._to_dict()
        if self.integrity:
            d["integrity"] = asdict(self.integrity)
        return yaml.dump(d, sort_keys=False, allow_unicode=True)
