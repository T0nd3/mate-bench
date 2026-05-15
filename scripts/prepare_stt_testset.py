#!/usr/bin/env python3
"""
Prepare LibriSpeech test-clean audio subsets for mate-workload-stt.

Downloads LibriSpeech test-clean, selects N clips by duration, uploads them
to the Cloudflare R2 bucket, and prints the JSON manifests + the snippet to
paste into packages/mate-workload-stt/src/mate_workload_stt/_profiles.py.

Requirements:
    pip install boto3

Environment variables:
    CLOUDFLARE_ACCOUNT_ID    Your Cloudflare account ID
    R2_ACCESS_KEY_ID         R2 API token key ID
    R2_SECRET_ACCESS_KEY     R2 API token secret
    R2_BUCKET_NAME           R2 bucket name (default: mate-bench)

Usage:
    uv run python scripts/prepare_stt_testset.py
    uv run python scripts/prepare_stt_testset.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
CDN_BASE = "https://pub-f27eb09940c14a8dac6ae7fe10e789f3.r2.dev"
R2_BUCKET = os.environ.get("R2_BUCKET_NAME", "mate-bench")
R2_PREFIX = "stt"

# Clip duration bounds for selection
MIN_DURATION_S = 5.0
MAX_DURATION_S = 15.0

QUICK_COUNT = 5
STANDARD_COUNT = 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def download_librispeech(cache_dir: Path) -> Path:
    archive = cache_dir / "test-clean.tar.gz"
    if archive.exists():
        print(f"  Using cached archive: {archive}")
        return archive
    print(f"  Downloading LibriSpeech test-clean (~346 MB)...")
    with urllib.request.urlopen(LIBRISPEECH_URL) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with archive.open("wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)
    print()
    return archive


def extract_and_select(archive: Path, extract_dir: Path) -> list[dict]:
    """Extract archive, parse transcripts, return clip metadata sorted by duration."""
    print("  Extracting archive...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_dir)

    clips: list[dict] = []

    # LibriSpeech structure: LibriSpeech/test-clean/<speaker>/<chapter>/
    base = extract_dir / "LibriSpeech" / "test-clean"
    for trans_file in sorted(base.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        transcripts: dict[str, str] = {}
        with trans_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                clip_id, _, text = line.partition(" ")
                transcripts[clip_id] = text

        for clip_id, reference in transcripts.items():
            flac_path = chapter_dir / f"{clip_id}.flac"
            if not flac_path.exists():
                continue
            size = flac_path.stat().st_size
            # Estimate duration: FLAC at ~64 kbps ≈ 8000 bytes/s
            est_duration_s = size / 8000
            if MIN_DURATION_S <= est_duration_s <= MAX_DURATION_S:
                clips.append({
                    "id": clip_id,
                    "path": flac_path,
                    "est_duration_s": est_duration_s,
                    "reference": reference,
                })

    # Sort by estimated duration for diversity, take a spread
    clips.sort(key=lambda c: c["est_duration_s"])
    return clips


def _r2_client():
    import boto3
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_clip(s3, clip_path: Path, clip_id: str, dry_run: bool) -> str:
    key = f"{R2_PREFIX}/clips/{clip_path.name}"
    url = f"{CDN_BASE}/{key}"
    if dry_run:
        print(f"  [dry-run] Would upload {clip_path.name} → {url}")
        return url
    print(f"  Uploading {clip_path.name}...", end=" ", flush=True)
    s3.upload_file(str(clip_path), R2_BUCKET, key, ExtraArgs={"ContentType": "audio/flac"})
    print("done")
    return url


def upload_manifest(s3, manifest: dict, name: str, dry_run: bool) -> tuple[str, str]:
    content = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    key = f"{R2_PREFIX}/{name}"
    url = f"{CDN_BASE}/{key}"
    sha256 = f"sha256:{hashlib.sha256(content).hexdigest()}"
    if dry_run:
        print(f"  [dry-run] Would upload manifest → {url}  ({sha256})")
        return url, sha256
    print(f"  Uploading manifest {name}...", end=" ", flush=True)
    import io
    s3.upload_fileobj(
        io.BytesIO(content),
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": "application/json"},
    )
    print("done")
    return url, sha256


def select_diverse(clips: list[dict], n: int) -> list[dict]:
    """Pick n clips spread evenly across the duration-sorted list."""
    if len(clips) <= n:
        return clips
    step = len(clips) / n
    return [clips[int(i * step)] for i in range(n)]


def build_manifest(clips: list[dict], s3, dry_run: bool) -> list[dict]:
    result = []
    for clip in clips:
        clip_path = clip["path"]
        sha = sha256_file(clip_path)
        # Compute accurate duration from file size for FLAC
        url = upload_clip(s3, clip_path, clip["id"], dry_run)
        result.append({
            "id": clip["id"],
            "url": url,
            "sha256": sha,
            "duration_s": round(clip["est_duration_s"], 2),
            "reference": clip["reference"],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Skip uploads, print what would happen")
    args = parser.parse_args()

    if not args.dry_run:
        for var in ("CLOUDFLARE_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
            if not os.environ.get(var):
                print(f"ERROR: {var} not set. Use --dry-run or set all R2 credentials.", file=sys.stderr)
                sys.exit(1)

    s3 = None if args.dry_run else _r2_client()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print("=== Step 1: Download LibriSpeech ===")
        archive = download_librispeech(tmp_path)

        print("=== Step 2: Extract + select clips ===")
        all_clips = extract_and_select(archive, tmp_path)
        print(f"  Found {len(all_clips)} clips in {MIN_DURATION_S}–{MAX_DURATION_S}s range")

        if len(all_clips) < STANDARD_COUNT:
            print(f"WARNING: only {len(all_clips)} clips found, need {STANDARD_COUNT}", file=sys.stderr)

        quick_clips = select_diverse(all_clips, QUICK_COUNT)
        standard_clips = select_diverse(all_clips, STANDARD_COUNT)

        print("=== Step 3: Upload clips + manifests ===")
        quick_clip_data = build_manifest(quick_clips, s3, args.dry_run)
        standard_clip_data = build_manifest(standard_clips, s3, args.dry_run)

        quick_manifest = {
            "version": "stt-librispeech-quick-v1",
            "license": "CC BY 4.0",
            "source": "LibriSpeech test-clean (Panayotov et al., 2015) — openslr.org/12",
            "clips": quick_clip_data,
        }
        standard_manifest = {
            "version": "stt-librispeech-standard-v1",
            "license": "CC BY 4.0",
            "source": "LibriSpeech test-clean (Panayotov et al., 2015) — openslr.org/12",
            "clips": standard_clip_data,
        }

        quick_url, quick_sha = upload_manifest(s3, quick_manifest, "stt-librispeech-quick-v1.json", args.dry_run)
        std_url, std_sha = upload_manifest(s3, standard_manifest, "stt-librispeech-standard-v1.json", args.dry_run)

    print()
    print("=== Step 4: Paste into _profiles.py ===")
    print()
    print("# Replace the sha256 / size_bytes / url values in TEST_SETS:")
    print(f'    "stt-librispeech-quick-v1": TestSetSpec(')
    print(f'        id="stt-librispeech-quick-v1",')
    print(f'        url="{quick_url}",')
    print(f'        sha256="{quick_sha}",')
    quick_size = sum(c.get("size_bytes", 0) for c in quick_clip_data)
    print(f'        size_bytes={quick_size},')
    print(f'        license="CC BY 4.0",')
    print(f'        source="LibriSpeech test-clean (Panayotov et al., 2015)",')
    print(f'    ),')
    print()
    print(f'    "stt-librispeech-standard-v1": TestSetSpec(')
    print(f'        id="stt-librispeech-standard-v1",')
    print(f'        url="{std_url}",')
    print(f'        sha256="{std_sha}",')
    std_size = sum(c.get("size_bytes", 0) for c in standard_clip_data)
    print(f'        size_bytes={std_size},')
    print(f'        license="CC BY 4.0",')
    print(f'        source="LibriSpeech test-clean (Panayotov et al., 2015)",')
    print(f'    ),')


if __name__ == "__main__":
    main()
