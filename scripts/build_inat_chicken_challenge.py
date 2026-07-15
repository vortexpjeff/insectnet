#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "inaturalist-domestic-chicken-2026-07-15"
TAXON_ID = 505478
PERMISSIVE = {"cc0": "CC0-1.0", "cc-by": "CC-BY-4.0"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return float(result.stdout.strip())


def make_center_window(raw_path: Path, output_path: Path, duration: float) -> float:
    start = max(0.0, (duration - 5.0) / 2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
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
    return start


def build(work_dir: Path) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "PineHollowResearch/1.0"
    observations: list[dict[str, Any]] = []
    total = None
    page = 1
    while total is None or len(observations) < total:
        response = session.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_id": TAXON_ID,
                "sounds": "true",
                "per_page": 200,
                "page": page,
                "order_by": "id",
                "order": "asc",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        total = int(payload["total_results"])
        observations.extend(payload["results"])
        page += 1

    selected: list[dict[str, Any]] = []
    exclusion_counts = Counter()
    for observation in observations:
        for sound in observation.get("sounds", []):
            normalized_license = PERMISSIVE.get(sound.get("license_code"))
            if normalized_license is None:
                exclusion_counts[f"license:{sound.get('license_code')}"] += 1
                continue
            selected.append(
                {
                    "observation_id": int(observation["id"]),
                    "observer_id": int(observation["user"]["id"]),
                    "sound_id": int(sound["id"]),
                    "file_url": sound["file_url"],
                    "file_content_type": sound.get("file_content_type"),
                    "license_code": normalized_license,
                    "attribution": sound.get("attribution"),
                    "taxon_id": int(observation["taxon"]["id"]),
                    "taxon_name": observation["taxon"]["name"],
                    "quality_grade": observation.get("quality_grade"),
                    "observation_uri": observation["uri"],
                }
            )
    selected.sort(key=lambda row: (row["observation_id"], row["sound_id"]))
    if len(selected) != 42:
        raise ValueError(f"expected frozen permissive count 42, found {len(selected)}")

    raw_dir = work_dir / "raw" / "inat-chicken-challenge"
    window_dir = work_dir / "windows" / "inat-chicken-challenge"
    manifest_dir = work_dir / "manifests" / "inat-chicken-challenge"
    recordings: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        extension = ".m4a" if "mp4" in str(row["file_content_type"]) else ".wav"
        raw_path = raw_dir / f"{row['sound_id']}{extension}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            response = session.get(row["file_url"], timeout=120)
            response.raise_for_status()
            raw_path.write_bytes(response.content)
        duration = duration_seconds(raw_path)
        output_path = window_dir / f"{row['sound_id']}.wav"
        start = make_center_window(raw_path, output_path, duration)
        recording_id = f"{SOURCE_ID}:sound:{row['sound_id']}"
        group_id = f"{SOURCE_ID}:observer:{row['observer_id']}"
        recordings.append(
            {
                "schema_version": 1,
                "source_id": SOURCE_ID,
                "recording_id": recording_id,
                "upstream_observation_id": row["observation_id"],
                "upstream_sound_id": row["sound_id"],
                "source_url": row["observation_uri"],
                "media_url": row["file_url"],
                "taxon_id": row["taxon_id"],
                "taxon_name": row["taxon_name"],
                "quality_grade": row["quality_grade"],
                "license_spdx": row["license_code"],
                "attribution": row["attribution"],
                "group_id": group_id,
                "role": "locked_external_challenge",
                "raw_path": str(raw_path.relative_to(work_dir)),
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": sha256_file(raw_path),
                "duration_seconds": duration,
            }
        )
        windows.append(
            {
                "schema_version": 1,
                "window_id": f"{SOURCE_ID}:{row['sound_id']}",
                "recording_id": recording_id,
                "group_id": group_id,
                "partition": "test",
                "labels": ["chicken_vocalization_present"],
                "label_authority": "source_taxon_weak_label",
                "rights_lane": "core_releasable",
                "sample_weight": 1.0,
                "role": "locked_external_challenge_not_training",
                "start_ms": round(start * 1000),
                "end_ms": round((start + 5.0) * 1000),
                "preprocessing_recipe": "ffmpeg-centered5s-32khz-mono-s16-v1",
                "window_path": str(output_path.relative_to(work_dir)),
                "window_sha256": sha256_file(output_path),
            }
        )
        print(f"processed {index}/{len(selected)} iNaturalist sounds", flush=True)

    write_jsonl(manifest_dir / "recordings.jsonl", recordings)
    write_jsonl(manifest_dir / "windows.jsonl", windows)
    write_jsonl(
        manifest_dir / "EXCLUSIONS.jsonl",
        [{"reason": reason, "count": count} for reason, count in sorted(exclusion_counts.items())],
    )
    retrieved_at = subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip()
    write_json(
        manifest_dir / "source.json",
        {
            "schema_version": 1,
            "source_id": SOURCE_ID,
            "title": "iNaturalist Domestic Chicken permissive-audio challenge",
            "api_url": "https://api.inaturalist.org/v1/observations",
            "taxon_id": TAXON_ID,
            "query": "taxon_id=505478&sounds=true&order_by=id&order=asc",
            "retrieved_at": retrieved_at,
            "total_observations_with_sound": len(observations),
            "selected_sounds": len(selected),
            "selected_observers": len({row["observer_id"] for row in selected}),
            "license_policy": "sound object license CC0 or CC BY only",
            "location_policy": "location and coordinate fields discarded before manifesting",
            "role": "locked_external_challenge_not_training_or_threshold_selection",
        },
    )
    summary = {
        "source_id": SOURCE_ID,
        "recordings": len(recordings),
        "windows": len(windows),
        "observers": len({row["group_id"] for row in windows}),
        "licenses": dict(Counter(row["license_spdx"] for row in recordings)),
        "excluded_license_counts": dict(exclusion_counts),
        "recordings_manifest_sha256": sha256_file(manifest_dir / "recordings.jsonl"),
        "windows_manifest_sha256": sha256_file(manifest_dir / "windows.jsonl"),
        "exclusions_manifest_sha256": sha256_file(manifest_dir / "EXCLUSIONS.jsonl"),
    }
    write_json(manifest_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a locked iNaturalist chicken challenge")
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.work_dir.resolve())


if __name__ == "__main__":
    main()
