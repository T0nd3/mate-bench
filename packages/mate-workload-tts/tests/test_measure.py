from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mate_workload_tts._measure import measure

SENTENCES = [
    {"id": "t01", "text": "Hello world.", "words": 2},
    {"id": "t02", "text": "The quick brown fox.", "words": 4},
]


def _make_engine(processing_time: float = 1.0, audio_duration: float = 2.0) -> MagicMock:
    engine = MagicMock()
    result = MagicMock()
    result.processing_time_s = processing_time
    result.audio_duration_s = audio_duration
    engine.synthesize.return_value = result
    return engine


class TestMeasure:
    def test_rtf_is_correct(self):
        engine = _make_engine(processing_time=1.0, audio_duration=2.0)
        median, _std_dev, _throttling = measure(engine, "m", "v", SENTENCES, runs=1, warmup_runs=0)
        # total_processing=2.0s (2 sentences), total_audio=4.0s → RTF=0.5
        assert median["rtf"] == pytest.approx(0.5)

    def test_chars_per_second(self):
        engine = _make_engine(processing_time=1.0, audio_duration=2.0)
        median, _std_dev, _throttling = measure(engine, "m", "v", SENTENCES, runs=1, warmup_runs=0)
        total_chars = len("Hello world.") + len("The quick brown fox.")
        expected_cps = total_chars / 2.0  # 2 sentences × 1.0s each
        assert median["chars_per_second"] == pytest.approx(expected_cps)

    def test_total_audio_seconds(self):
        engine = _make_engine(processing_time=1.0, audio_duration=3.0)
        median, _std_dev, _throttling = measure(engine, "m", "v", SENTENCES, runs=1, warmup_runs=0)
        assert median["total_audio_seconds"] == pytest.approx(6.0)  # 2 sentences × 3.0s

    def test_warmup_excluded_from_stats(self):
        results = iter(
            [MagicMock(processing_time_s=10.0, audio_duration_s=2.0)] * 2  # warmup
            + [MagicMock(processing_time_s=1.0, audio_duration_s=2.0)] * 2  # measurement
        )
        engine = MagicMock()
        engine.synthesize.side_effect = results
        median, _std_dev, _throttling = measure(engine, "m", "v", SENTENCES, runs=1, warmup_runs=1)
        assert median["rtf"] == pytest.approx(0.5)

    def test_throttling_detected_on_high_variance(self):
        results = iter(
            [MagicMock(processing_time_s=1.0, audio_duration_s=2.0)] * 2
            + [MagicMock(processing_time_s=10.0, audio_duration_s=2.0)] * 2
        )
        engine = MagicMock()
        engine.synthesize.side_effect = results
        _median, _std_dev, throttling = measure(engine, "m", "v", SENTENCES, runs=2, warmup_runs=0)
        assert throttling is True

    def test_multiple_runs_median(self):
        results = iter(
            [MagicMock(processing_time_s=1.0, audio_duration_s=2.0)] * 2  # run 1: RTF 0.5
            + [MagicMock(processing_time_s=3.0, audio_duration_s=2.0)] * 2  # run 2: RTF 1.5
        )
        engine = MagicMock()
        engine.synthesize.side_effect = results
        median, _std_dev, _throttling = measure(engine, "m", "v", SENTENCES, runs=2, warmup_runs=0)
        assert median["rtf"] == pytest.approx(1.0)  # median of [0.5, 1.5]
