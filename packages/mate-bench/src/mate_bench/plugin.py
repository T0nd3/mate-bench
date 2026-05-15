from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Mode(StrEnum):
    CLOSED = "closed"
    OPEN = "open"


@dataclass
class PluginManifest:
    requires_mate_bench: str  # PEP 440 version spec, e.g. ">=0.1,<0.2"
    api_version: int


@dataclass
class TestSetSpec:
    id: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    source: str
    fallback_urls: list[str] = field(default_factory=list)


@dataclass
class ProfileConfig:
    name: str
    description: str
    test_set_id: str
    reference_engine_config: dict[str, Any]
    vram_required_gb: float
    download_size_bytes: int
    estimated_runtime_seconds: int


@dataclass
class Measurement:
    runs: int
    warmup_runs: int
    median: dict[str, Any]
    std_dev: dict[str, Any]
    vram_peak_gb: float
    throttling_detected: bool


@runtime_checkable
class WorkloadPlugin(Protocol):
    name: str
    manifest: PluginManifest
    profiles: dict[str, ProfileConfig]
    test_sets: dict[str, TestSetSpec]

    def estimate_download(self, profile: str) -> int: ...
    def estimate_vram(self, profile: str) -> int: ...
    def estimate_runtime(self, profile: str) -> int: ...
    def setup_closed(self, profile: str) -> None: ...
    def setup_open(self, profile: str, user_inputs: dict[str, Any]) -> None: ...
    def run(
        self,
        profile: str,
        mode: Mode,
        engine: EnginePlugin,
        runs: int,
        warmup_runs: int,
    ) -> Measurement: ...
    def cleanup(self, profile: str) -> None: ...


@runtime_checkable
class EnginePlugin(Protocol):
    name: str
    manifest: PluginManifest
    supported_runtimes: list[str]

    def is_available(self) -> bool: ...
    def version(self) -> str: ...


@runtime_checkable
class RuntimePlugin(Protocol):
    name: str
    manifest: PluginManifest

    def is_available(self) -> bool: ...
    def gpu_info(self) -> dict[str, Any]: ...
