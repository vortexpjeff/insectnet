#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "insectset459-v1.1-hf-11b209d"
PERMISSIVE = {
    "//creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "//creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
}


def normalize_license(value: str) -> str | None:
    normalized = (
        (value or "").strip().replace("https://", "//").replace("http://", "//").rstrip("/")
        + "/"
    )
    return PERMISSIVE.get(normalized)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def stable_partition(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def make_window(raw_path: Path, window_path: Path) -> tuple[float, float]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(raw_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    duration = float(probe.stdout.strip())
    start = max(0.0, (duration - 5.0) / 2.0)
    window_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.6f}", "-i", str(raw_path), "-t", "5", "-ar", "32000", "-ac", "1", "-sample_fmt", "s16", str(window_path)],
        check=True,
        timeout=120,
    )
    return duration, start


def download(row: dict[str, str]) -> tuple[dict[str, str], bytes, str]:
    headers = {"User-Agent": "BioacousticsModelResearch/1.0"}
    last_error = ""
    for attempt in range(5):
        try:
            response = requests.get(row["file"], headers=headers, timeout=180)
            response.raise_for_status()
            if not response.content:
                raise ValueError("empty response")
            return row, response.content, response.url
        except Exception as error:
            last_error = str(error)
            time.sleep(2**attempt)
    raise RuntimeError(f"{row['file_name']}: {last_error}")


def build(work_dir: Path, workers: int) -> None:
    manifest_dir = work_dir / "manifests" / "insectnet"
    annotation_path = manifest_dir / "InsectSet459_Train_Val_Test_Annotation.csv"
    recordings_path = manifest_dir / "recordings.jsonl"
    windows_path = manifest_dir / "windows.jsonl"
    downloads_path = manifest_dir / "downloads.jsonl"
    exclusions_path = manifest_dir / "EXCLUSIONS.jsonl"
    raw_dir = work_dir / "raw" / "insectset459"
    window_dir = work_dir / "windows" / "insectset459"
    taxonomy = {
        row["species_name"]: row for row in load_jsonl(manifest_dir / "gbif_taxonomy.jsonl")
    }
    annotations = list(csv.DictReader(annotation_path.open(encoding="utf-8-sig")))
    eligible = [
        row
        for row in annotations
        if row["subset"] in {"Train", "Validation"}
        and normalize_license(row["license"]) is not None
    ]
    chosen_by_recording: dict[str, dict[str, str]] = {}
    for row in sorted(eligible, key=lambda item: item["file_name"]):
        recording_id = f"{SOURCE_ID}:{row['observation'] or row['file_name']}"
        chosen_by_recording.setdefault(recording_id, row)
    existing = load_jsonl(recordings_path)
    seen = {str(row["recording_id"]) for row in existing}
    missing = [
        (recording_id, row)
        for recording_id, row in sorted(chosen_by_recording.items())
        if recording_id not in seen
    ]
    print(f"direct-source missing observations: {len(missing)}", flush=True)
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [(recording_id, executor.submit(download, row)) for recording_id, row in missing]
        for index, (recording_id, future) in enumerate(futures, start=1):
            try:
                row, audio_bytes, final_url = future.result()
                license_id = normalize_license(row["license"])
                if license_id is None:
                    raise ValueError("license changed after selection")
                audio_sha = sha256_bytes(audio_bytes)
                extension = Path(row["file_name"]).suffix.lower() or ".audio"
                raw_name = f"{audio_sha[:16]}_{safe_component(row['file_name'])}"
                if not raw_name.endswith(extension):
                    raw_name += extension
                raw_path = raw_dir / raw_name
                raw_path.write_bytes(audio_bytes)
                window_id = f"{SOURCE_ID}:{audio_sha[:24]}:center5s"
                window_path = window_dir / f"{audio_sha[:24]}.wav"
                duration, start = make_window(raw_path, window_path)
                contributor = (row["contributor"] or "unknown").strip()
                group_id = f"insectset459:contributor:{contributor}"
                partition = stable_partition(group_id)
                taxon = taxonomy.get(row["species_name"], {})
                labels = ["insect_present", "cicada"] if row["group"] == "Cicadidae" else ["insect_present", "orthoptera"]
                append_jsonl(
                    recordings_path,
                    {
                        "schema_version": 1,
                        "source_id": SOURCE_ID,
                        "recording_id": recording_id,
                        "upstream_file_name": row["file_name"],
                        "upstream_observation": row["observation"],
                        "source_url": row["file"],
                        "source_subset": row["subset"],
                        "raw_label": row["species_name"],
                        "taxon_order": taxon.get("order"),
                        "taxon_suborder": taxon.get("suborder"),
                        "taxon_family": taxon.get("family"),
                        "gbif_usage_key": taxon.get("usage_key"),
                        "contributor": contributor,
                        "license_uri": row["license"],
                        "license_spdx": license_id,
                        "rights_lane": "core_releasable",
                        "group_id": group_id,
                        "partition": partition,
                        "raw_path": str(raw_path.relative_to(work_dir)),
                        "raw_bytes": len(audio_bytes),
                        "raw_sha256": audio_sha,
                        "duration_seconds": duration,
                        "review_authority": "source_weak_label",
                        "repository_revision": "11b209dd8754ee45128c258e5337b69f2acaafbd",
                        "source_shard": None,
                        "source_shard_sha256": None,
                        "acquisition_method": "direct_original_url_from_frozen_annotation",
                    },
                )
                append_jsonl(
                    windows_path,
                    {
                        "schema_version": 1,
                        "window_id": window_id,
                        "recording_id": recording_id,
                        "group_id": group_id,
                        "partition": partition,
                        "labels": labels,
                        "raw_species_label": row["species_name"],
                        "label_authority": "source_weak_label",
                        "uncertainty": "weak_file_label_center_crop",
                        "rights_lane": "core_releasable",
                        "sample_weight": 1.0,
                        "start_ms": round(start * 1000),
                        "end_ms": round((start + 5.0) * 1000),
                        "preprocessing_recipe": "ffmpeg-center5s-32khz-mono-s16-v1",
                        "window_path": str(window_path.relative_to(work_dir)),
                        "window_sha256": sha256_file(window_path),
                    },
                )
                append_jsonl(
                    downloads_path,
                    {
                        "status": "completed",
                        "access_method": "direct_original_url_from_frozen_annotation",
                        "source_url": row["file"],
                        "final_url": final_url,
                        "upstream_file_name": row["file_name"],
                        "bytes": len(audio_bytes),
                        "sha256": audio_sha,
                        "repository_revision": "11b209dd8754ee45128c258e5337b69f2acaafbd",
                    },
                )
                seen.add(recording_id)
            except Exception as error:
                failures.append({"recording_id": recording_id, "error": str(error)})
            if index % 25 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)} direct observations failures={len(failures)}", flush=True)
    for failure in failures:
        append_jsonl(exclusions_path, {"source_id": SOURCE_ID, "reason": "direct_source_download_failed", **failure})
    expected = set(chosen_by_recording)
    missing_after = sorted(expected - seen)
    all_recordings = load_jsonl(recordings_path)
    all_windows = load_jsonl(windows_path)
    if len(all_recordings) != len(all_windows):
        raise ValueError("recording/window manifests differ after direct completion")
    if missing_after:
        raise RuntimeError(f"{len(missing_after)} expected observations remain missing")
    summary = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "repository_revision": "11b209dd8754ee45128c258e5337b69f2acaafbd",
        "unique_expected_observations": len(expected),
        "recordings": len(all_recordings),
        "windows": len(all_windows),
        "partitions": dict(Counter(str(row["partition"]) for row in all_windows)),
        "labels": dict(Counter(label for row in all_windows for label in row["labels"])),
        "licenses": dict(Counter(str(row["license_spdx"]) for row in all_recordings)),
        "recordings_manifest_sha256": sha256_file(recordings_path),
        "windows_manifest_sha256": sha256_file(windows_path),
        "downloads_manifest_sha256": sha256_file(downloads_path),
        "exclusions_manifest_sha256": sha256_file(exclusions_path),
        "completion_method": "HF Parquet permissive rows plus direct original URLs from frozen annotation manifest",
        "direct_failures": len(failures),
        "complete_permissive_train_validation_observation_set": True,
    }
    write_json(manifest_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete InsectSet from frozen original URLs")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.work_dir.resolve(), args.workers)


if __name__ == "__main__":
    main()
