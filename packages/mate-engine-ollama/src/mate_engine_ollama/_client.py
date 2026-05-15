from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class GenerateResult:
    model: str
    prompt_tokens: int
    generated_tokens: int
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int

    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration_ns == 0:
            return 0.0
        return self.generated_tokens / (self.eval_duration_ns / 1e9)

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.prompt_eval_duration_ns == 0:
            return 0.0
        return self.prompt_tokens / (self.prompt_eval_duration_ns / 1e9)

    @property
    def load_duration_ms(self) -> float:
        return self.load_duration_ns / 1e6


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self._base}{path}")
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, body: dict, timeout: float | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=timeout or self._timeout) as client:
            resp = client.post(f"{self._base}{path}", json=body)
            resp.raise_for_status()
            return resp.json()

    def is_alive(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                client.get(f"{self._base}/api/version")
            return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    def version(self) -> str:
        data = self._get("/api/version")
        return data.get("version", "unknown")

    def list_models(self) -> list[dict[str, Any]]:
        data = self._get("/api/tags")
        return data.get("models", [])

    def show_model(self, name: str) -> dict[str, Any]:
        return self._post("/api/show", {"name": name})

    def pull_model(
        self,
        name: str,
        on_progress: Callable[[str, float | None], None] | None = None,
    ) -> None:
        """Pull a model, calling on_progress(status, percent) as it downloads."""
        with (
            httpx.Client(timeout=None) as client,
            client.stream("POST", f"{self._base}/api/pull", json={"name": name}) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                import json

                chunk = json.loads(line)
                status = chunk.get("status", "")
                total = chunk.get("total")
                completed = chunk.get("completed")
                percent = (completed / total * 100) if total and completed else None
                if on_progress:
                    on_progress(status, percent)

    def generate(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> GenerateResult:
        """Run a single generation and return timing metrics.

        Uses non-streaming mode — Ollama includes full metrics in the final response.
        """
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            body["options"] = options

        data = self._post("/api/generate", body, timeout=timeout)

        return GenerateResult(
            model=model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            generated_tokens=data.get("eval_count", 0),
            total_duration_ns=data.get("total_duration", 0),
            load_duration_ns=data.get("load_duration", 0),
            prompt_eval_duration_ns=data.get("prompt_eval_duration", 0),
            eval_duration_ns=data.get("eval_duration", 0),
        )
