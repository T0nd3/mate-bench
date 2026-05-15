"""ComfyUI engine plugin for mate-bench.

Connects to a running ComfyUI server via its REST API and submits
txt2img workflows, measuring wall-clock generation time per image.
"""
from __future__ import annotations

from mate_bench.plugin import PluginManifest

from ._api import ComfyUiClient, ImageResult
from ._workflow import txt2img_workflow

__all__ = ["ComfyUiEngine", "ImageResult"]


class ComfyUiEngine:
    name = "comfyui"
    manifest = PluginManifest(requires_mate_bench=">=0.1,<0.2", api_version=1)
    supported_runtimes = ["cuda", "rocm", "cpu", "mps"]

    def __init__(self, host: str = "127.0.0.1", port: int = 8188, timeout: float = 600.0) -> None:
        self._client = ComfyUiClient(host=host, port=port, timeout=timeout)

    def is_available(self) -> bool:
        """Return True if a ComfyUI server is reachable at the configured host/port."""
        try:
            self._client.system_stats()
            return True
        except Exception:
            return False

    def version(self) -> str:
        """Return the ComfyUI version string reported by the server, or 'unknown'."""
        try:
            stats = self._client.system_stats()
            return stats.get("system", {}).get("comfyui_version", "unknown")
        except Exception:
            return "unknown"

    def txt2img(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        width: int,
        height: int,
        cfg: float,
        sampler: str,
        model: str,
        seed: int = 42,
    ) -> ImageResult:
        """Submit a txt2img workflow to ComfyUI and return timing + output filenames."""
        workflow = txt2img_workflow(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            width=width,
            height=height,
            cfg=cfg,
            sampler=sampler,
            seed=seed,
        )
        return self._client.run_workflow(workflow)
