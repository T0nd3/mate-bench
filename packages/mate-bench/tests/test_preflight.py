from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mate_bench._preflight import PreflightError, check_engine_available, check_model_pulled, check_vram


def _engine(available: bool, models: list[str] | None = None) -> MagicMock:
    e = MagicMock()
    e.name = "mock-engine"
    e.is_available.return_value = available
    if models is not None:
        e.list_models.return_value = [{"name": m} for m in models]
    return e


class TestCheckEngineAvailable:
    def test_raises_when_unavailable(self):
        with pytest.raises(PreflightError, match="not reachable"):
            check_engine_available(_engine(available=False))

    def test_passes_when_available(self):
        check_engine_available(_engine(available=True))


class TestCheckModelPulled:
    def test_returns_true_when_model_found(self):
        engine = _engine(True, models=["llama3.2:latest", "qwen2.5:7b"])
        assert check_model_pulled(engine, "llama3.2:latest") is True

    def test_returns_false_when_not_found(self):
        engine = _engine(True, models=["qwen2.5:7b"])
        assert check_model_pulled(engine, "llama3.2:latest") is False

    def test_raises_with_local_only_when_not_found(self):
        engine = _engine(True, models=["qwen2.5:7b"])
        with pytest.raises(PreflightError, match="--local"):
            check_model_pulled(engine, "llama3.2:latest", local_only=True)

    def test_returns_true_when_engine_has_no_list_models(self):
        engine = MagicMock(spec=[])
        assert check_model_pulled(engine, "any-model") is True


class TestCheckVram:
    def test_no_warning_when_sufficient(self, capsys):
        check_vram(vram_available_gb=8.0, vram_required_gb=5.5, profile="standard")

    def test_no_warning_when_vram_unknown(self, capsys):
        check_vram(vram_available_gb=0.0, vram_required_gb=5.5, profile="standard")

    def test_warning_when_insufficient(self, capsys):
        from io import StringIO
        from unittest.mock import patch
        from rich.console import Console

        output = StringIO()
        mock_console = Console(file=output, highlight=False)
        with patch("mate_bench._preflight.console", mock_console):
            check_vram(vram_available_gb=3.0, vram_required_gb=5.5, profile="standard")
        assert "Warning" in output.getvalue() or "standard" in output.getvalue()
