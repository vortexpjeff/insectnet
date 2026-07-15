#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_ID = "ross308-data3437-v1"
LICENSE = "CC-BY-4.0"
PUBLISHED_MD5 = "cadf1efd32e42b9c35007077ef57c3bc"


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bird_partition(bird_id: str) -> str:
    match = re.search(r"Pollo(\d+)", bird_id)
    if not match:
        raise ValueError(f"cannot parse bird number from {bird_id}")
    number = int(match.group(1))
    if number <= 4:
        return "train"
    if number == 5:
        return "validation"
    return "test"


def extract_window(raw_path: Path, output_path: Path, start_seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, start_seconds):.6f}",
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


def build(work_dir: Path, archive_path: Path, readme_path: Path) -> None:
    archive_md5 = hash_file(archive_path, "md5")
    if archive_md5 != PUBLISHED_MD5:
        raise ValueError(f"Ross archive MD5 mismatch: {archive_md5}")
    archive_sha = hash_file(archive_path)
    raw_root = work_dir / "raw" / "ross308"
    if not (raw_root / "segments" / "segments_definitiu.csv").exists():
        temp_root = work_dir / "cache" / "ross308-extract"
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_root)
        candidates = list(temp_root.glob("*/segments/segments_definitiu.csv"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one Ross metadata CSV, found {len(candidates)}")
        source_root = candidates[0].parent.parent
        shutil.rmtree(raw_root, ignore_errors=True)
        shutil.move(str(source_root), raw_root)
        shutil.rmtree(temp_root, ignore_errors=True)

    csv_path = raw_root / "segments" / "segments_definitiu.csv"
    annotations = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    if len(annotations) != 1_420:
        raise ValueError(f"expected 1420 Ross annotations, found {len(annotations)}")

    retrieved_at = subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip()
    manifest_dir = work_dir / "manifests" / "ross308-chickennet"
    source = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "title": "Vocalization Audio Dataset of Ross-308 Broiler Chickens",
        "version_doi": "10.34810/DATA3437",
        "source_url": "https://doi.org/10.34810/DATA3437",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_md5": archive_md5,
        "archive_sha256": archive_sha,
        "published_md5": PUBLISHED_MD5,
        "readme": str(readme_path),
        "readme_sha256": hash_file(readme_path),
        "metadata_sha256": hash_file(csv_path),
        "license_spdx": LICENSE,
        "rights_lane": "core_releasable",
        "retrieved_at": retrieved_at,
        "target_policy": "audible vocalization only; health and lighting retained as provenance",
        "event_policy": "clean annotations greedily merged into non-overlapping 5-second parent contexts",
        "split_policy": "birds 1-4 train, bird 5 validation, bird 6 test; both health groups kept together by number",
        "ambient_policy": "Fons1 validation, Fons2 test, sampled every 30 seconds",
    }
    write_json(manifest_dir / "source.json", source)

    by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for annotation in annotations:
        by_parent[annotation["audio_original"]].append(annotation)

    recording_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for parent, parent_annotations in sorted(by_parent.items()):
        raw_path = raw_root / "individual_recordings" / f"{parent}.wav"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        bird_ids = {row["pollastre"] for row in parent_annotations}
        if len(bird_ids) != 1:
            raise ValueError(f"parent {parent} maps to multiple bird IDs: {bird_ids}")
        bird_id = next(iter(bird_ids))
        partition = bird_partition(bird_id)
        group_id = f"{SOURCE_ID}:bird:{bird_id}"
        recording_id = f"{SOURCE_ID}:individual:{parent}"
        recording_rows.append(
            {
                "schema_version": 1,
                "source_id": SOURCE_ID,
                "recording_id": recording_id,
                "upstream_file_name": raw_path.name,
                "bird_id": bird_id,
                "health_condition_source": parent_annotations[0]["salut"],
                "lighting_condition_source": parent_annotations[0]["llum_foscor"],
                "license_spdx": LICENSE,
                "rights_lane": "core_releasable",
                "group_id": group_id,
                "partition": partition,
                "raw_path": str(raw_path.relative_to(work_dir)),
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": hash_file(raw_path),
                "duration_seconds": duration_seconds(raw_path),
                "review_authority": "manual_temporal_annotation",
            }
        )

        selected_centers: list[tuple[float, str]] = []
        for annotation in sorted(parent_annotations, key=lambda row: float(row["inici_s"])):
            center = (float(annotation["inici_s"]) + float(annotation["final_s"])) / 2.0
            if annotation["tipus"] != "Soroll":
                decision_rows.append(
                    {
                        **annotation,
                        "decision": "excluded_noisy_vocalization",
                        "reason": "Soroll Brut: clipping/contact artifact per bundled README",
                    }
                )
                exclusions.append(
                    {
                        "source_id": SOURCE_ID,
                        "upstream_id": annotation["segment_id"],
                        "recording_id": recording_id,
                        "reason": "noisy_vocalization_soroll_brut",
                    }
                )
                continue
            previous = next(
                ((chosen_center, window_id) for chosen_center, window_id in reversed(selected_centers) if center - chosen_center < 5.0),
                None,
            )
            if previous is not None:
                decision_rows.append(
                    {
                        **annotation,
                        "decision": "represented_in_existing_context_window",
                        "selected_window_id": previous[1],
                    }
                )
                continue
            start = max(0.0, center - 2.5)
            window_id = f"{SOURCE_ID}:{parent}:event-{annotation['segment_id']}"
            output_path = work_dir / "windows" / "ross308-chickennet" / f"{annotation['segment_id']}.wav"
            extract_window(raw_path, output_path, start)
            selected_centers.append((center, window_id))
            decision_rows.append(
                {
                    **annotation,
                    "decision": "selected_context_window_anchor",
                    "selected_window_id": window_id,
                }
            )
            window_rows.append(
                {
                    "schema_version": 1,
                    "window_id": window_id,
                    "recording_id": recording_id,
                    "group_id": group_id,
                    "partition": partition,
                    "labels": ["chicken_vocalization_present", "chicken_other_vocalization"],
                    "raw_quality_label": annotation["tipus"],
                    "health_condition_source": annotation["salut"],
                    "lighting_condition_source": annotation["llum_foscor"],
                    "label_authority": "manual_temporal_annotation",
                    "rights_lane": "core_releasable",
                    "sample_weight": 1.0,
                    "start_ms": round(start * 1000),
                    "end_ms": round((start + 5.0) * 1000),
                    "anchor_segment_id": annotation["segment_id"],
                    "preprocessing_recipe": "ffmpeg-event-centered5s-32khz-mono-s16-v1",
                    "window_path": str(output_path.relative_to(work_dir)),
                    "window_sha256": hash_file(output_path),
                }
            )

    for ambient_name, partition in (("Fons1.wav", "validation"), ("Fons2.wav", "test")):
        raw_path = raw_root / "corral_and_ambient_recordings" / ambient_name
        recording_id = f"{SOURCE_ID}:ambient:{Path(ambient_name).stem}"
        group_id = recording_id
        duration = duration_seconds(raw_path)
        recording_rows.append(
            {
                "schema_version": 1,
                "source_id": SOURCE_ID,
                "recording_id": recording_id,
                "upstream_file_name": ambient_name,
                "raw_label": "animal_free_ambient_ventilation",
                "license_spdx": LICENSE,
                "rights_lane": "core_releasable",
                "group_id": group_id,
                "partition": partition,
                "raw_path": str(raw_path.relative_to(work_dir)),
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": hash_file(raw_path),
                "duration_seconds": duration,
                "review_authority": "bundled_readme_environment_label",
            }
        )
        for index, start in enumerate(range(0, max(0, int(duration) - 5), 30)):
            window_id = f"{SOURCE_ID}:ambient:{Path(ambient_name).stem}:{index:04d}"
            output_path = work_dir / "windows" / "ross308-chickennet" / f"ambient_{Path(ambient_name).stem}_{index:04d}.wav"
            extract_window(raw_path, output_path, float(start))
            window_rows.append(
                {
                    "schema_version": 1,
                    "window_id": window_id,
                    "recording_id": recording_id,
                    "group_id": group_id,
                    "partition": partition,
                    "labels": [],
                    "raw_quality_label": "animal_free_ambient_ventilation",
                    "label_authority": "bundled_readme_environment_label",
                    "rights_lane": "core_releasable",
                    "sample_weight": 1.0,
                    "start_ms": start * 1000,
                    "end_ms": (start + 5) * 1000,
                    "preprocessing_recipe": "ffmpeg-every30s-first5s-32khz-mono-s16-v1",
                    "window_path": str(output_path.relative_to(work_dir)),
                    "window_sha256": hash_file(output_path),
                }
            )

    for corral in sorted((raw_root / "corral_and_ambient_recordings").glob("Poll*.wav")):
        exclusions.append(
            {
                "source_id": SOURCE_ID,
                "upstream_id": corral.name,
                "reason": "diagnostic_only_corral_recording_without_temporal_vocalization_labels",
            }
        )

    write_jsonl(manifest_dir / "recordings.jsonl", recording_rows)
    write_jsonl(manifest_dir / "windows.jsonl", window_rows)
    write_jsonl(manifest_dir / "annotation_decisions.jsonl", decision_rows)
    write_jsonl(manifest_dir / "EXCLUSIONS.jsonl", exclusions)
    label_counts = Counter(label for row in window_rows for label in row["labels"])
    partition_counts = Counter(row["partition"] for row in window_rows)
    summary = {
        "source_id": SOURCE_ID,
        "annotations": len(annotations),
        "clean_annotations": sum(row["tipus"] == "Soroll" for row in annotations),
        "noisy_annotations": sum(row["tipus"] != "Soroll" for row in annotations),
        "recordings": len(recording_rows),
        "windows": len(window_rows),
        "labels": dict(label_counts),
        "partitions": dict(partition_counts),
        "exclusions": len(exclusions),
        "recordings_manifest_sha256": hash_file(manifest_dir / "recordings.jsonl"),
        "windows_manifest_sha256": hash_file(manifest_dir / "windows.jsonl"),
        "annotation_decisions_sha256": hash_file(manifest_dir / "annotation_decisions.jsonl"),
        "exclusions_manifest_sha256": hash_file(manifest_dir / "EXCLUSIONS.jsonl"),
    }
    write_json(manifest_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the exact Ross-308 ChickenNet corpus")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()
    build(args.work_dir.resolve(), args.archive.resolve(), args.readme.resolve())


if __name__ == "__main__":
    main()
