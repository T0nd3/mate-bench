"""faster-whisper engine plugin for mate-bench.

Wraps WhisperModel with automatic CUDA/CPU device selection and
in-process model caching to avoid redundant weight loads across runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from mate_bench.plugin import PluginManifest

from ._transcribe import TranscribeResult, transcribe

__all__ = ["FasterWhisperEngine", "TranscribeResult"]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


class FasterWhisperEngine:
    name = "faster-whisper"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    supported_runtimes: ClassVar[list[str]] = ["cuda", "rocm", "cpu"]

    def __init__(
        self,
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
    ) -> None:
        self._device_arg = device
        self._compute_type_arg = compute_type
        self._language = language
        self._model_cache: dict[str, object] = {}

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("faster-whisper")
        except Exception:
            return "unknown"

    def _load_model(self, model_name: str) -> object:
        if model_name not in self._model_cache:
            from faster_whisper import WhisperModel

            device = _resolve_device(self._device_arg)
            compute_type = (
                _default_compute_type(device)
                if self._compute_type_arg == "auto"
                else self._compute_type_arg
            )
            self._model_cache[model_name] = WhisperModel(
                model_name, device=device, compute_type=compute_type
            )
        return self._model_cache[model_name]

    def transcribe(self, audio_path: Path, model: str) -> TranscribeResult:
        model_obj = self._load_model(model)
        return transcribe(model_obj, audio_path, language=self._language)
