from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dst.open("wb") as f:  # noqa: S310
        while chunk := resp.read(65536):
            f.write(chunk)


def fetch_manifest(url: str, expected_sha256: str, cache_dir: Path) -> dict:
    """Download and cache the test-set manifest JSON; verify hash."""
    name = url.rsplit("/", 1)[-1]
    dst = cache_dir / "manifests" / name
    if not dst.exists():
        _download(url, dst)
    if expected_sha256 != "sha256:PENDING":
        actual = _sha256_file(dst)
        if actual != expected_sha256:
            dst.unlink()
            raise ValueError(f"Manifest hash mismatch: expected {expected_sha256}, got {actual}")
    with dst.open() as f:
        return json.load(f)


def fetch_clips(manifest: dict, clips_dir: Path) -> dict[str, Path]:
    """Download all clips referenced in the manifest; return {clip_id: local_path}."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for clip in manifest["clips"]:
        clip_id = clip["id"]
        fname = clip["url"].rsplit("/", 1)[-1]
        dst = clips_dir / fname
        if not dst.exists():
            _download(clip["url"], dst)
        actual = _sha256_file(dst)
        if actual != clip["sha256"]:
            dst.unlink()
            raise ValueError(f"Clip {clip_id} hash mismatch: expected {clip['sha256']}, got {actual}")
        result[clip_id] = dst
    return result
