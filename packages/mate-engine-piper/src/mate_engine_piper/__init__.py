"""Piper TTS engine plugin for mate-bench.

Wraps PiperVoice with model caching so repeated synthesis calls within
a benchmark run avoid reloading the ONNX model from disk.
"""
from __future__ import annotations

from typing import ClassVar

from mate_bench.plugin import PluginManifest

from ._synthesize import TtsResult, synthesize

__all__ = ["PiperEngine", "TtsResult"]


class PiperEngine:
    """mate-bench engine plugin for Piper TTS."""

    name = "piper"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    supported_runtimes: ClassVar[list[str]] = ["cpu"]

    def __init__(self, use_cuda: bool = False) -> None:
        """Initialise the Piper engine.

        Args:
            use_cuda: Enable CUDA inference if available (requires onnxruntime-gpu).
        """
        self._use_cuda = use_cuda
        self._voice_cache: dict[str, object] = {}

    def _load_voice(self, model: str) -> object:
        """Load or return a cached PiperVoice for the given model path."""
        if model not in self._voice_cache:
            from piper import PiperVoice

            self._voice_cache[model] = PiperVoice.load(model, use_cuda=self._use_cuda)
        return self._voice_cache[model]

    def is_available(self) -> bool:
        """Return True if the piper-tts package is importable."""
        try:
            import piper  # noqa: F401

            return True
        except ImportError:
            return False

    def version(self) -> str:
        """Return the installed piper-tts package version."""
        try:
            from importlib.metadata import version

            return version("piper-tts")
        except Exception:
            return "unknown"

    def synthesize(self, text: str, model: str, voice: str = "") -> TtsResult:
        """Synthesize text with Piper and return timing information.

        Args:
            text: Input text to synthesize.
            model: Path to the .onnx model file, e.g. "en_US-lessac-medium.onnx".
            voice: Unused for Piper (voice is determined by the model file).
        """
        piper_voice = self._load_voice(model)
        return synthesize(piper_voice, text)
