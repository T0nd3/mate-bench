from __future__ import annotations

from mate_bench.schema import ModelInfo


def model_info_from_ollama(name: str, models: list[dict]) -> ModelInfo:
    """Build a ModelInfo from an Ollama model name and the /api/tags response."""
    digest = None
    for m in models:
        if m.get("name") == name or m.get("model") == name:
            digest = m.get("digest")
            break

    return ModelInfo(
        name=name,
        source="ollama",
        source_ref=name,
        file_hash=f"sha256:{digest}" if digest else None,
        file_hash_available=digest is not None,
    )
