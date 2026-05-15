from __future__ import annotations

from typing import Any, ClassVar

from mate_bench.plugin import PluginManifest
from mate_bench.schema import ModelInfo

from ._client import GenerateResult, OllamaClient
from ._models import model_info_from_ollama


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
