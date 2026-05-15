from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mate_engine_faster_whisper import FasterWhisperEngine
from mate_engine_faster_whisper._transcribe import TranscribeResult, transcribe

# ── TranscribeResult ──────────────────────────────────────────────────────────


class TestTranscribeResult:
    def test_rtf_computed(self):
        r = TranscribeResult(text="hello", audio_duration_s=10.0, processing_time_s=1.0)
        assert r.rtf == pytest.approx(0.1)

    def test_rtf_zero_audio(self):
        r = TranscribeResult(text="", audio_duration_s=0.0, processing_time_s=0.5)
        assert r.rtf == 0.0


# ── transcribe() ──────────────────────────────────────────────────────────────


def _mock_model(text: str = "hello world", duration: float = 5.0) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    info = MagicMock()
    info.duration = duration
    model = MagicMock()
    model.transcribe.return_value = ([seg], info)
    return model


class TestTranscribeFn:
    def test_returns_correct_text(self):
        model = _mock_model(text=" hello world ")
        result = transcribe(model, Path("/audio.flac"))
        assert result.text == "hello world"

    def test_audio_duration_from_info(self):
        model = _mock_model(duration=12.5)
        result = transcribe(model, Path("/audio.flac"))
        assert result.audio_duration_s == pytest.approx(12.5)

    def test_processing_time_is_positive(self):
        model = _mock_model()
        result = transcribe(model, Path("/audio.flac"))
        assert result.processing_time_s >= 0.0

    def test_multi_segment_joined(self):
        seg1 = MagicMock()
        seg1.text = " Hello"
        seg2 = MagicMock()
        seg2.text = " world"
        info = MagicMock()
        info.duration = 3.0
        model = MagicMock()
        model.transcribe.return_value = ([seg1, seg2], info)
        result = transcribe(model, Path("/audio.flac"))
        assert result.text == "Hello world"


# ── FasterWhisperEngine ───────────────────────────────────────────────────────


class TestFasterWhisperEngine:
    def test_is_available_true_when_importable(self):
        engine = FasterWhisperEngine()
        with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
            assert engine.is_available() is True

    def test_model_cached_after_first_call(self):
        engine = FasterWhisperEngine(device="cpu", compute_type="int8")
        mock_cls = MagicMock()
        mock_model = _mock_model()
        mock_cls.return_value = mock_model

        with (
            patch("faster_whisper.WhisperModel", mock_cls),
            patch("mate_engine_faster_whisper._transcribe.transcribe", return_value=MagicMock()),
        ):
            engine._load_model("large-v3")
            engine._load_model("large-v3")

        mock_cls.assert_called_once()

    def test_version_returns_string(self):
        engine = FasterWhisperEngine()
        v = engine.version()
        assert isinstance(v, str)
