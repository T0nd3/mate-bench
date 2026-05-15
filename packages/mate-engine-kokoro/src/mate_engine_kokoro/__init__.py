"""Kokoro TTS engine plugin for mate-bench.

Wraps KPipeline with automatic device selection and pipeline caching
to avoid reloading model weights between benchmark runs.
"""
from __future__ import annotations

from typing import ClassVar

from mate_bench.plugin import PluginManifest

from ._synthesize import TtsResult, synthesize

__all__ = ["KokoroEngine", "TtsResult"]

_SAMPLE_RATE = 24_000  # Kokoro outputs 24 kHz audio


class KokoroEngine:
    """mate-bench engine plugin for Kokoro TTS."""

    name = "kokoro"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    supported_runtimes: ClassVar[list[str]] = ["cuda", "cpu"]

    def __init__(self, device: str = "auto", lang_code: str = "a") -> None:
        """Initialise the Kokoro engine.

        Args:
            device: "cuda", "cpu", or "auto" (detects via PyTorch).
            lang_code: Kokoro language code ("a" = American English, "b" = British English).
        """
        self._device_arg = device
        self._lang_code = lang_code
        self._pipeline_cache: dict[str, object] = {}

    def _resolve_device(self) -> str:
        """Return the actual device string, resolving 'auto' via PyTorch."""
        if self._device_arg != "auto":
            return self._device_arg
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _get_pipeline(self, model: str) -> object:
        """Load or return cached KPipeline for the given model."""
        if model not in self._pipeline_cache:
            from kokoro import KPipeline

            device = self._resolve_device()
            self._pipeline_cache[model] = KPipeline(
                lang_code=self._lang_code, repo_id=model, device=device
            )
        return self._pipeline_cache[model]

    def is_available(self) -> bool:
        """Return True if the kokoro package is importable."""
        try:
            import kokoro  # noqa: F401

            return True
        except ImportError:
            return False

    def version(self) -> str:
        """Return the installed kokoro package version."""
        try:
            from importlib.metadata import version

            return version("kokoro")
        except Exception:
            return "unknown"

    def synthesize(self, text: str, model: str, voice: str = "af_heart") -> TtsResult:
        """Synthesize text and return a TtsResult with timing information.

        Args:
            text: Input text to synthesize.
            model: Kokoro model repo ID, e.g. "kokoro-v1.0" or "hexgrad/Kokoro-82M".
            voice: Voice name, e.g. "af_heart", "af_bella", "bm_george".
        """
        pipeline = self._get_pipeline(model)
        return synthesize(pipeline, text, voice=voice, sample_rate=_SAMPLE_RATE)
