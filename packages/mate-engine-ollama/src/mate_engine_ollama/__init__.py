from __future__ import annotations

import os
from typing import Any, ClassVar

from mate_bench.plugin import PluginManifest
from mate_bench.schema import ModelInfo

from ._client import GenerateResult, OllamaClient
from ._models import model_info_from_ollama

# Env vars that affect Ollama inference performance — captured at benchmark time
_PERF_ENV_VARS = (
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_NUM_GPU",
    "OLLAMA_KEEP_ALIVE",
)


class OllamaEngine:
    name = "ollama"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    supported_runtimes: ClassVar[list[str]] = ["rocm", "cuda", "cpu"]

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._client = OllamaClient(base_url)

    def is_available(self) -> bool:
        return self._client.is_alive()

    def version(self) -> str:
        return self._client.version()

    def list_models(self) -> list[dict[str, Any]]:
        return self._client.list_models()

    def model_info(self, name: str) -> ModelInfo:
        models = self._client.list_models()
        return model_info_from_ollama(name, models)

    def generate(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> GenerateResult:
        return self._client.generate(model, prompt, options=options, timeout=timeout)

    def pull(self, model: str, on_progress=None) -> None:
        self._client.pull_model(model, on_progress=on_progress)

    def engine_config_snapshot(self) -> dict[str, str]:
        """Return Ollama performance env vars that were set at benchmark time.

        Included in the result YAML under engine_config.settings._env so that
        differences in Flash Attention, KV-cache type etc. are visible when
        comparing results across machines.
        """
        return {k: os.environ[k] for k in _PERF_ENV_VARS if k in os.environ}
