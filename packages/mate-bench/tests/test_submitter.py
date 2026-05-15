from __future__ import annotations

import uuid
from unittest.mock import patch

from mate_bench._submitter import get_anonymous_id


class TestGetAnonymousId:
    def test_creates_file_on_first_call(self, tmp_path):
        id_file = tmp_path / "submitter_id.txt"
        with patch("mate_bench._submitter._ID_FILE", id_file):
            result = get_anonymous_id()
        assert id_file.exists()
        assert id_file.read_text().strip() == result

    def test_returns_same_id_on_second_call(self, tmp_path):
        id_file = tmp_path / "submitter_id.txt"
        with patch("mate_bench._submitter._ID_FILE", id_file):
            first = get_anonymous_id()
            second = get_anonymous_id()
        assert first == second

    def test_id_is_valid_uuid(self, tmp_path):
        id_file = tmp_path / "submitter_id.txt"
        with patch("mate_bench._submitter._ID_FILE", id_file):
            result = get_anonymous_id()
        try:
            uuid.UUID(result)
        except ValueError as exc:
            raise AssertionError(f"Not a valid UUID: {result!r}") from exc

    def test_uses_existing_id_if_file_present(self, tmp_path):
        id_file = tmp_path / "submitter_id.txt"
        id_file.write_text("fixed-id-value")
        with patch("mate_bench._submitter._ID_FILE", id_file):
            result = get_anonymous_id()
        assert result == "fixed-id-value"
