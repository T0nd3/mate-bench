from __future__ import annotations

from pathlib import Path

from mate_bench._utils import sha256_file


class TestSha256File:
    def test_known_content(self, tmp_path):
        import hashlib
        content = b"hello mate-bench"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert sha256_file(f) == expected

    def test_empty_file(self, tmp_path):
        import hashlib
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = "sha256:" + hashlib.sha256(b"").hexdigest()
        assert sha256_file(f) == expected

    def test_result_starts_with_sha256(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("x")
        assert sha256_file(f).startswith("sha256:")

    def test_result_has_64_hex_chars(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("x")
        _, digest = sha256_file(f).split(":", 1)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
