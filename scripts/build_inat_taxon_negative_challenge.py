#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

PERMISSIVE = {"cc0": "CC0-1.0", "cc-by": "CC-BY-4.0"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def download(session: requests.Session, url: str, path: Path) -> None:
    last_error = ""
    for attempt in range(5):
        try:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            if not response.content:
                raise ValueError("empty response")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return
        except Exception as error:
            last_error = str(error)
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def build(
    work_dir: Path,
    *,
    taxon_id: int,
    source_id: str,
    title: str,
    max_sounds: int,
    min_sounds: int,
) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "PineHollowResearch/1.0"
    observations: list[dict[str, Any]] = []
    total = None
    page = 1
    while total is None or len(observations) < total:
        response = session.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_id": taxon_id,
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

    candidates: list[dict[str, Any]] = []
    exclusions = Counter()
    for observation in observations:
        for sound in observation.get("sounds", []):
            license_spdx = PERMISSIVE.get(sound.get("license_code"))
            if license_spdx is None:
                exclusions[f"license:{sound.get('license_code')}"] += 1
                continue
            candidates.append(
                {
                    "observation_id": int(observation["id"]),
                    "observer_id": int(observation["user"]["id"]),
                    "sound_id": int(sound["id"]),
                    "file_url": sound["file_url"],
                    "file_content_type": sound.get("file_content_type"),
                    "license_spdx": license_spdx,
                    "attribution": sound.get("attribution"),
                    "taxon_id": int(observation["taxon"]["id"]),
                    "taxon_name": observation["taxon"]["name"],
                    "quality_grade": observation.get("quality_grade"),
                    "observation_uri": observation["uri"],
                }
            )
    candidates.sort(key=lambda row: (row["observation_id"], row["sound_id"]))
    selected: list[dict[str, Any]] = []
    seen_observers: set[int] = set()
    for row in candidates:
        if row["observer_id"] in seen_observers:
            continue
        selected.append(row)
        seen_observers.add(row["observer_id"])
        if len(selected) == max_sounds:
            break
    if len(selected) < min_sounds:
        raise ValueError(f"needed at least {min_sounds} permissive observers, found {len(selected)}")

    raw_dir = work_dir / "raw" / source_id
    window_dir = work_dir / "windows" / source_id
    manifest_dir = work_dir / "manifests" / source_id
    recordings: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        content_type = str(row["file_content_type"] or "")
        extension = ".m4a" if "mp4" in content_type else ".wav"
        raw_path = raw_dir / f"{row['sound_id']}{extension}"
        if not raw_path.exists():
            download(session, str(row["file_url"]), raw_path)
        duration = duration_seconds(raw_path)
        output_path = window_dir / f"{row['sound_id']}.wav"
        start = make_center_window(raw_path, output_path, duration)
        recording_id = f"{source_id}:sound:{row['sound_id']}"
        group_id = f"{source_id}:observer:{row['observer_id']}"
        recordings.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "recording_id": recording_id,
                "upstream_observation_id": row["observation_id"],
                "upstream_sound_id": row["sound_id"],
                "source_url": row["observation_uri"],
                "media_url": row["file_url"],
                "taxon_id": row["taxon_id"],
                "taxon_name": row["taxon_name"],
                "quality_grade": row["quality_grade"],
                "license_spdx": row["license_spdx"],
                "attribution": row["attribution"],
                "group_id": group_id,
                "role": "locked_external_insect_negative_challenge",
                "raw_path": str(raw_path.relative_to(work_dir)),
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": sha256_file(raw_path),
                "duration_seconds": duration,
            }
        )
        windows.append(
            {
                "schema_version": 1,
                "window_id": f"{source_id}:{row['sound_id']}",
                "recording_id": recording_id,
                "group_id": group_id,
                "partition": "test",
                "labels": [],
                "label_authority": "source_taxon_negative_for_insect_targets",
                "rights_lane": "core_releasable",
                "sample_weight": 1.0,
                "role": "locked_external_challenge_not_training_or_threshold_selection",
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
        [{"reason": reason, "count": count} for reason, count in sorted(exclusions.items())],
    )
    retrieved_at = subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip()
    source = {
        "schema_version": 1,
        "source_id": source_id,
        "title": title,
        "api_url": "https://api.inaturalist.org/v1/observations",
        "taxon_id": taxon_id,
        "query": f"taxon_id={taxon_id}&sounds=true&order_by=id&order=asc",
        "retrieved_at": retrieved_at,
        "total_observations_with_sound": len(observations),
        "permissive_candidates": len(candidates),
        "selected_sounds": len(selected),
        "selected_observers": len(seen_observers),
        "selection_policy": "earliest permissive sound per observer, ordered by observation and sound ID",
        "license_policy": "sound object license CC0 or CC BY only",
        "location_policy": "location and coordinate fields discarded before manifesting",
        "role": "locked_external_challenge_not_training_or_threshold_selection",
    }
    write_json(manifest_dir / "source.json", source)
    summary = {
        "schema_version": 1,
        "source_id": source_id,
        "recordings": len(recordings),
        "windows": len(windows),
        "observers": len(seen_observers),
        "licenses": dict(Counter(row["license_spdx"] for row in recordings)),
        "recordings_manifest_sha256": sha256_file(manifest_dir / "recordings.jsonl"),
        "windows_manifest_sha256": sha256_file(manifest_dir / "windows.jsonl"),
        "exclusions_manifest_sha256": sha256_file(manifest_dir / "EXCLUSIONS.jsonl"),
    }
    write_json(manifest_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a locked iNaturalist taxon negative challenge")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--taxon-id", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--max-sounds", type=int, default=80)
    parser.add_argument("--min-sounds", type=int, default=40)
    args = parser.parse_args()
    build(
        args.work_dir.resolve(),
        taxon_id=args.taxon_id,
        source_id=args.source_id,
        title=args.title,
        max_sounds=args.max_sounds,
        min_sounds=args.min_sounds,
    )


if __name__ == "__main__":
    main()
