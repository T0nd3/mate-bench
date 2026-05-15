from __future__ import annotations

import time
from dataclasses import dataclass

import requests


@dataclass
class ImageResult:
    generation_time_s: float
    filenames: list[str]


class ComfyUiClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188, timeout: float = 600.0) -> None:
        self._base = f"http://{host}:{port}"
        self._timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(f"{self._base}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        resp = requests.get(f"{self._base}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def queue_prompt(self, workflow: dict) -> str:
        data = self._post("/prompt", {"prompt": workflow})
        return data["prompt_id"]

    def wait_for_completion(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            history = self._get(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                if entry.get("status", {}).get("completed"):
                    return entry
                status_str = entry.get("status", {}).get("status_str", "")
                if status_str == "error":
                    raise RuntimeError(f"ComfyUI reported error for prompt {prompt_id}")
            time.sleep(0.5)
        raise TimeoutError(f"ComfyUI did not complete prompt {prompt_id} within {self._timeout}s")

    def run_workflow(self, workflow: dict) -> ImageResult:
        t0 = time.perf_counter()
        prompt_id = self.queue_prompt(workflow)
        entry = self.wait_for_completion(prompt_id)
        elapsed = time.perf_counter() - t0

        filenames: list[str] = []
        for node_output in entry.get("outputs", {}).values():
            for img in node_output.get("images", []):
                filenames.append(img["filename"])

        return ImageResult(generation_time_s=elapsed, filenames=filenames)

    def system_stats(self) -> dict:
        return self._get("/system_stats")
