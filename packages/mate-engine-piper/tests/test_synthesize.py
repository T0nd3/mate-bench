from __future__ import annotations

import io
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest

from mate_engine_piper import PiperEngine, TtsResult
from mate_engine_piper._synthesize import _wav_duration, synthesize


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 22050) -> bytes:
    """Create a minimal valid WAV byte string of the given duration."""
    n_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


class TestTtsResult:
    def test_rtf_computed(self):
        r = TtsResult(text="hello", audio_duration_s=2.0, processing_time_s=0.5)
        assert r.rtf == pytest.approx(0.25)

    def test_rtf_zero_audio_no_division_error(self):
        r = TtsResult(text="hello", audio_duration_s=0.0, processing_time_s=1.0)
        assert r.rtf > 0

    def test_chars_per_second(self):
        r = TtsResult(text="hello", audio_duration_s=1.0, processing_time_s=1.0)
        assert r.chars_per_second == pytest.approx(5.0)


class TestWavDuration:
    def test_correct_duration(self):
        wav_bytes = _make_wav_bytes(duration_s=2.5, sample_rate=22050)
        assert _wav_duration(wav_bytes) == pytest.approx(2.5, abs=0.01)


class TestSynthesizeFn:
    def _make_voice(self, duration_s: float = 1.0) -> MagicMock:
        wav_bytes = _make_wav_bytes(duration_s)

        def fake_synthesize(text: str, wav_file: wave.Wave_write) -> None:
            with wave.open(io.BytesIO(wav_bytes)) as src:
                wav_file.setnchannels(src.getnchannels())
                wav_file.setsampwidth(src.getsampwidth())
                wav_file.setframerate(src.getframerate())
                wav_file.writeframes(src.readframes(src.getnframes()))

        voice = MagicMock()
        voice.synthesize = fake_synthesize
        return voice

    def test_returns_tts_result(self):
        voice = self._make_voice(duration_s=1.5)
        result = synthesize(voice, "Hello world")
        assert isinstance(result, TtsResult)
        assert result.audio_duration_s == pytest.approx(1.5, abs=0.01)
        assert result.processing_time_s >= 0


class TestPiperEngine:
    def test_is_available_false_when_missing(self):
        engine = PiperEngine()
        with patch("builtins.__import__", side_effect=ImportError):
            assert engine.is_available() is False

    def test_version_returns_string(self):
        engine = PiperEngine()
        v = engine.version()
        assert isinstance(v, str)

    def test_voice_cached_after_first_call(self):
        engine = PiperEngine()
        mock_voice_cls = MagicMock()
        mock_voice_cls.load.return_value = MagicMock()
        with (
            patch("piper.PiperVoice", mock_voice_cls),
            patch("mate_engine_piper._synthesize.synthesize", return_value=MagicMock()),
        ):
            engine._load_voice("en_US-lessac-medium.onnx")
            engine._load_voice("en_US-lessac-medium.onnx")
        mock_voice_cls.load.assert_called_once()
