from __future__ import annotations

from rich.console import Console

console = Console()


class PreflightError(Exception):
    pass


def check_engine_available(engine) -> None:
    if not engine.is_available():
        raise PreflightError(
            f"Engine '{engine.name}' is not reachable. "
            "Is the server running? (e.g. `ollama serve`)"
        )


def check_model_pulled(engine, model_name: str, *, local_only: bool = False) -> bool:
    """Return True if model is already available locally."""
    if not hasattr(engine, "list_models"):
        return True
    models = engine.list_models()
    names = {m.get("name", "") for m in models}
    if model_name in names:
        return True
    if local_only:
        raise PreflightError(
            f"Model '{model_name}' is not pulled and --local is set. "
            f"Run: ollama pull {model_name}"
        )
    return False


def check_vram(vram_available_gb: float, vram_required_gb: float, profile: str) -> None:
    if vram_available_gb > 0 and vram_available_gb < vram_required_gb:
        console.print(
            f"[yellow]Warning: profile '{profile}' requires ~{vram_required_gb:.1f} GB VRAM "
            f"but only {vram_available_gb:.1f} GB detected. "
            "Benchmark may be slow or fail.[/yellow]"
        )
