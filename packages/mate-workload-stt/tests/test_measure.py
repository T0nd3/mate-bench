from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mate_workload_stt._measure import corpus_wer, measure

from .fixtures import MOCK_CLIP_PATHS, MOCK_CLIPS


# ── corpus_wer ────────────────────────────────────────────────────────────────

class TestCorpusWer:
    def test_perfect_match(self):
        assert corpus_wer(["HELLO WORLD"], ["HELLO WORLD"]) == 0.0

    def test_one_substitution_of_two(self):
        result = corpus_wer(["HELLO WORLD"], ["HELLO EARTH"])
        assert result == pytest.approx(0.5)

    def test_full_deletion(self):
        result = corpus_wer(["ONE TWO THREE"], [""])
        assert result == pytest.approx(1.0)

    def test_empty_reference(self):
        assert corpus_wer([""], [""]) == 0.0

    def test_case_insensitive(self):
        assert corpus_wer(["hello world"], ["HELLO WORLD"]) == 0.0

    def test_multi_clip_corpus(self):
        refs = ["ONE TWO", "THREE FOUR"]
        hyps = ["ONE TWO", "THREE FIVE"]
        # 0 errors + 1 error over 4 total words = 0.25
        assert corpus_wer(refs, hyps) == pytest.approx(0.25)


# ── measure ───────────────────────────────────────────────────────────────────

def _make_engine(processing_time: float, text: str) -> MagicMock:
    result = MagicMock()
    result.text = text
    result.audio_duration_s = 10.0
    result.processing_time_s = processing_time
    engine = MagicMock()
    engine.transcribe.return_value = result
    return engine


class TestMeasure:
    def test_rtf_is_correct(self):
        engine = _make_engine(processing_time=1.0, text="HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS")
        median, std_dev, throttling = measure(engine, "large-v3", MOCK_CLIPS, MOCK_CLIP_PATHS, runs=1, warmup_runs=0)
        # processing_time=1.0, audio_duration=10.0 per clip → RTF = 0.1
        assert median["rtf"] == pytest.approx(0.1)

    def test_warmup_excluded(self):
        results = iter([
            _make_engine(5.0, "HELLO").transcribe(Path("/x"), "m").processing_time_s,
        ])
        call_count = 0
        engine = MagicMock()

        def side_effect(path, model):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.text = "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS"
            r.audio_duration_s = 10.0
            r.processing_time_s = 1.0
            return r

        engine.transcribe.side_effect = side_effect
        measure(engine, "large-v3", MOCK_CLIPS, MOCK_CLIP_PATHS, runs=1, warmup_runs=2)
        # 2 warmup + 1 run, each with 2 clips → 6 total calls
        assert call_count == 6

    def test_no_throttling_stable_run(self):
        engine = _make_engine(1.0, "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS")
        _, _, throttling = measure(engine, "large-v3", MOCK_CLIPS, MOCK_CLIP_PATHS, runs=3, warmup_runs=0)
        assert throttling is False
