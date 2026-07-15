#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "esc50-master"
LICENSE = "CC-BY-NC-3.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def partition_for_source(source_id: str) -> str:
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def make_window(raw_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
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


def build(work_dir: Path, archive_path: Path) -> None:
    archive_sha = sha256_file(archive_path)
    extract_root = work_dir / "raw" / "esc50"
    if not (extract_root / "meta" / "esc50.csv").exists():
        temp_root = work_dir / "cache" / "esc50-extract"
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_root)
        candidates = list(temp_root.glob("*/meta/esc50.csv"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one ESC-50 metadata file, found {len(candidates)}")
        source_root = candidates[0].parent.parent
        shutil.rmtree(extract_root, ignore_errors=True)
        shutil.move(str(source_root), extract_root)
        shutil.rmtree(temp_root, ignore_errors=True)

    metadata_path = extract_root / "meta" / "esc50.csv"
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    if len(rows) != 2_000:
        raise ValueError(f"expected 2000 ESC-50 rows, found {len(rows)}")

    official_fold_leakage = Counter()
    folds_by_source: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        folds_by_source[row["src_file"]].add(int(row["fold"]))
    for folds in folds_by_source.values():
        if len(folds) > 1:
            official_fold_leakage["sources_crossing_folds"] += 1

    github = requests.get(
        "https://api.github.com/repos/karolpiczak/ESC-50/commits/master", timeout=60
    )
    github.raise_for_status()
    revision = github.json()["sha"]
    retrieved_at = subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip()
    common_source = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "title": "ESC-50",
        "source_url": "https://github.com/karolpiczak/ESC-50",
        "repository_revision_observed": revision,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "metadata_sha256": sha256_file(metadata_path),
        "license_spdx": LICENSE,
        "rights_lane": "research_noncommercial",
        "retrieved_at": retrieved_at,
        "split_policy": "deterministic SHA-256 split by original Freesound src_file: 70/15/15",
        "group_policy": "original Freesound src_file; overrides official folds because source IDs cross them",
        "official_fold_audit": dict(official_fold_leakage),
    }

    model_rows: dict[str, list[dict[str, Any]]] = {
        "esc50-insectnet": [],
        "esc50-chickennet": [],
    }
    model_recordings: dict[str, list[dict[str, Any]]] = {
        "esc50-insectnet": [],
        "esc50-chickennet": [],
    }
    exclusions: dict[str, list[dict[str, Any]]] = {
        "esc50-insectnet": [],
        "esc50-chickennet": [],
    }
    for index, row in enumerate(rows, start=1):
        filename = row["filename"]
        category = row["category"]
        fold = int(row["fold"])
        partition = partition_for_source(row["src_file"])
        raw_path = extract_root / "audio" / filename
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw_sha = sha256_file(raw_path)
        recording_id = f"{SOURCE_ID}:{filename}"
        group_id = f"{SOURCE_ID}:src_file:{row['src_file']}"
        output_path = work_dir / "windows" / "esc50" / f"{Path(filename).stem}.wav"
        make_window(raw_path, output_path)
        window_sha = sha256_file(output_path)
        recording = {
            "schema_version": 1,
            "source_id": SOURCE_ID,
            "recording_id": recording_id,
            "upstream_file_name": filename,
            "raw_label": category,
            "target": int(row["target"]),
            "fold": fold,
            "src_file": row["src_file"],
            "take": row["take"],
            "license_spdx": LICENSE,
            "rights_lane": "research_noncommercial",
            "group_id": group_id,
            "partition": partition,
            "raw_path": str(raw_path.relative_to(work_dir)),
            "raw_bytes": raw_path.stat().st_size,
            "raw_sha256": raw_sha,
            "review_authority": "dataset_category_label",
        }

        if category == "insects":
            exclusions["esc50-insectnet"].append(
                {
                    "source_id": SOURCE_ID,
                    "recording_id": recording_id,
                    "reason": "target_overlap_insects_category",
                    "raw_label": category,
                }
            )
        else:
            model_recordings["esc50-insectnet"].append(recording)
            model_rows["esc50-insectnet"].append(
                {
                    "schema_version": 1,
                    "window_id": f"esc50-insectnet:{Path(filename).stem}",
                    "recording_id": recording_id,
                    "group_id": group_id,
                    "partition": partition,
                    "labels": [],
                    "raw_category": category,
                    "label_authority": "dataset_category_label",
                    "rights_lane": "research_noncommercial",
                    "sample_weight": 0.25,
                    "start_ms": 0,
                    "end_ms": 5000,
                    "preprocessing_recipe": "ffmpeg-first5s-32khz-mono-s16-v1",
                    "window_path": str(output_path.relative_to(work_dir)),
                    "window_sha256": window_sha,
                }
            )

        labels: list[str] = []
        if category == "rooster":
            labels = ["chicken_vocalization_present", "chicken_crow"]
        elif category == "hen":
            labels = ["chicken_vocalization_present", "chicken_other_vocalization"]
        model_recordings["esc50-chickennet"].append(recording)
        model_rows["esc50-chickennet"].append(
            {
                "schema_version": 1,
                "window_id": f"esc50-chickennet:{Path(filename).stem}",
                "recording_id": recording_id,
                "group_id": group_id,
                "partition": partition,
                "labels": labels,
                "raw_category": category,
                "label_authority": "dataset_category_label",
                "rights_lane": "research_noncommercial",
                "sample_weight": 1.0 if labels else 0.25,
                "start_ms": 0,
                "end_ms": 5000,
                "preprocessing_recipe": "ffmpeg-first5s-32khz-mono-s16-v1",
                "window_path": str(output_path.relative_to(work_dir)),
                "window_sha256": window_sha,
            }
        )
        if index % 100 == 0:
            print(f"processed {index}/{len(rows)} ESC-50 clips", flush=True)

    for model_key in model_rows:
        manifest_dir = work_dir / "manifests" / model_key
        write_json(manifest_dir / "source.json", {**common_source, "model_role": model_key})
        write_jsonl(manifest_dir / "recordings.jsonl", model_recordings[model_key])
        write_jsonl(manifest_dir / "windows.jsonl", model_rows[model_key])
        write_jsonl(manifest_dir / "EXCLUSIONS.jsonl", exclusions[model_key])
        label_counts = Counter(label for row in model_rows[model_key] for label in row["labels"])
        category_counts = Counter(row["raw_category"] for row in model_rows[model_key])
        partition_counts = Counter(row["partition"] for row in model_rows[model_key])
        summary = {
            "source_id": SOURCE_ID,
            "model_role": model_key,
            "recordings": len(model_recordings[model_key]),
            "windows": len(model_rows[model_key]),
            "exclusions": len(exclusions[model_key]),
            "labels": dict(label_counts),
            "categories": dict(category_counts),
            "partitions": dict(partition_counts),
            "recordings_manifest_sha256": sha256_file(manifest_dir / "recordings.jsonl"),
            "windows_manifest_sha256": sha256_file(manifest_dir / "windows.jsonl"),
            "exclusions_manifest_sha256": sha256_file(manifest_dir / "EXCLUSIONS.jsonl"),
        }
        write_json(manifest_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact ESC-50 model corpora")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    build(args.work_dir.resolve(), args.archive.resolve())


if __name__ == "__main__":
    main()
