from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mate_engine_kokoro import KokoroEngine, TtsResult
from mate_engine_kokoro._synthesize import synthesize


class TestTtsResult:
    def test_rtf_computed(self):
        r = TtsResult(text="hello", audio_duration_s=2.0, processing_time_s=1.0)
        assert r.rtf == pytest.approx(0.5)

    def test_rtf_zero_audio_no_division_error(self):
        r = TtsResult(text="hello", audio_duration_s=0.0, processing_time_s=1.0)
        assert r.rtf > 0

    def test_chars_per_second(self):
        r = TtsResult(text="hello", audio_duration_s=2.0, processing_time_s=2.0)
        assert r.chars_per_second == pytest.approx(2.5)  # 5 chars / 2.0s


class TestSynthesizeFn:
    def _make_pipeline(self, audio_length: int = 24000) -> MagicMock:
        """Return a mock KPipeline that yields one chunk of given length."""
        import numpy as np

        audio = np.zeros(audio_length, dtype=np.float32)
        pipeline = MagicMock()
        pipeline.return_value = [("g", "p", audio)]
        return pipeline

    def test_returns_tts_result(self):
        pipeline = self._make_pipeline(audio_length=24000)  # 1.0s at 24kHz
        result = synthesize(pipeline, "Hello world", voice="af_heart")
        assert isinstance(result, TtsResult)
        assert result.audio_duration_s == pytest.approx(1.0)
        assert result.processing_time_s > 0

    def test_empty_output_no_error(self):
        pipeline = MagicMock()
        pipeline.return_value = []
        result = synthesize(pipeline, "Hello", voice="af_heart")
        assert result.audio_duration_s == 0.0

    def test_multi_chunk_concatenated(self):
        import numpy as np

        chunk = np.zeros(12000, dtype=np.float32)  # 0.5s each
        pipeline = MagicMock()
        pipeline.return_value = [("g1", "p1", chunk), ("g2", "p2", chunk)]
        result = synthesize(pipeline, "Hello world", voice="af_heart")
        assert result.audio_duration_s == pytest.approx(1.0)


class TestKokoroEngine:
    def test_is_available_true_when_kokoro_installed(self):
        engine = KokoroEngine()
        with patch("mate_engine_kokoro.KokoroEngine.is_available", return_value=True):
            assert engine.is_available() is True

    def test_is_available_false_when_missing(self):
        engine = KokoroEngine()
        with patch("builtins.__import__", side_effect=ImportError):
            assert engine.is_available() is False

    def test_version_returns_string(self):
        engine = KokoroEngine()
        v = engine.version()
        assert isinstance(v, str)

    def test_pipeline_cached_after_first_call(self):
        engine = KokoroEngine(device="cpu")
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value = MagicMock()
        with (
            patch("mate_engine_kokoro._synthesize.synthesize", return_value=MagicMock()),
            patch("kokoro.KPipeline", mock_pipeline_cls),
        ):
            engine._get_pipeline("kokoro-v1.0")
            engine._get_pipeline("kokoro-v1.0")
        mock_pipeline_cls.assert_called_once()
