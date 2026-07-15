#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_window(raw_path: Path, output_path: Path, start: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(raw_path),
            "-t",
            "5",
            "-ar",
            "32000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(output_path),
        ],
        check=True,
        timeout=90,
    )


def build(root: Path, work_dir: Path) -> None:
    files = sorted(root.glob("*.wav"))
    if not files:
        raise ValueError(f"no WAV files found in {root}")
    manifest_dir = work_dir / "manifests" / "private-local-frog-challenge"
    window_dir = work_dir / "windows" / "private-local-frog-challenge"
    recordings: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for file_index, raw_path in enumerate(files, start=1):
        raw_sha = sha256_file(raw_path)
        opaque_id = raw_sha[:24]
        recording_id = f"private-local-frog:{opaque_id}"
        recordings.append(
            {
                "schema_version": 1,
                "source_id": "private-local-frog-stationary-mic",
                "recording_id": recording_id,
                "private_source_path": str(raw_path),
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": raw_sha,
                "source_label": "frog",
                "label_authority": "reviewed_private_archive",
                "rights_lane": "private_diagnostic_only",
                "publication": "never_publish_path_or_audio",
            }
        )
        for window_index, start in enumerate((0, 5, 10)):
            output_path = window_dir / f"{opaque_id}_{window_index}.wav"
            extract_window(raw_path, output_path, start)
            windows.append(
                {
                    "schema_version": 1,
                    "window_id": f"private-local-frog:{opaque_id}:{window_index}",
                    "recording_id": recording_id,
                    "group_id": recording_id,
                    "partition": "test",
                    "labels": [],
                    "expected_chicken_vocalization": False,
                    "source_label": "frog",
                    "label_authority": "reviewed_private_archive",
                    "rights_lane": "private_diagnostic_only",
                    "sample_weight": 1.0,
                    "start_ms": start * 1000,
                    "end_ms": (start + 5) * 1000,
                    "preprocessing_recipe": "ffmpeg-fixed0-5-10s-32khz-mono-s16-v1",
                    "window_path": str(output_path.relative_to(work_dir)),
                    "window_sha256": sha256_file(output_path),
                }
            )
        if file_index % 50 == 0:
            print(f"processed {file_index}/{len(files)} private frog files", flush=True)
    write_jsonl(manifest_dir / "recordings.private.jsonl", recordings)
    write_jsonl(manifest_dir / "windows.private.jsonl", windows)
    summary = {
        "schema_version": 1,
        "source_id": "private-local-frog-stationary-mic",
        "files": len(recordings),
        "windows": len(windows),
        "recordings_manifest_sha256": sha256_file(manifest_dir / "recordings.private.jsonl"),
        "windows_manifest_sha256": sha256_file(manifest_dir / "windows.private.jsonl"),
        "privacy": "private paths and audio excluded from Git and Hugging Face",
        "role": "locked local negative challenge; never threshold selection",
    }
    (manifest_dir / "summary.private.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private stationary-mic frog challenge")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.work_dir.resolve())


if __name__ == "__main__":
    main()
