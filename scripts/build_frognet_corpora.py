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

from insectnet.frognet_corpus import (
    FROG_CLASS,
    build_esc50_frog_rows,
    build_local_frog_rows,
    build_weak_negative_rows,
    partition_for_group,
    select_permissive_inat_sounds,
    source_balanced_weights,
)

SOURCE_ID = "inaturalist-regional-anura-2026-07-22"
ANURA_TAXON_ID = 20979
REGIONAL_PLACE_ID = 45


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_derived_manifests(work_dir: Path) -> dict[str, object]:
    local_source = work_dir / "manifests/private-local-frog-challenge"
    local_rows = load_jsonl(local_source / "windows.private.jsonl")
    local_recordings = load_jsonl(local_source / "recordings.private.jsonl")
    screen = json.loads((local_source / "perch_label_screen.json").read_text(encoding="utf-8"))
    uncertain_windows = {
        f"{row['recording_id']}:{row['window_index']}"
        for row in screen["rows"]
        if not row["anuran_in_top5"]
    }
    local_output = build_local_frog_rows(
        local_rows,
        local_recordings,
        uncertain_window_ids=uncertain_windows,
    )

    esc_source = work_dir / "manifests/esc50-chickennet/windows.jsonl"
    esc_output = build_esc50_frog_rows(load_jsonl(esc_source))

    ross_source = work_dir / "manifests/ross308-chickennet/windows.jsonl"
    ross_output = []
    for source in load_jsonl(ross_source):
        row = dict(source)
        row["labels"] = []
        row.pop("unknown_labels", None)
        row["label_authority"] = "controlled_poultry_recording_frog_absent"
        row["sample_weight"] = 0.25
        ross_output.append(row)

    insectset_output = build_weak_negative_rows(
        load_jsonl(work_dir / "manifests/insectnet/windows.jsonl"),
        label_authority="source_taxon_weak_frog_absence",
        sample_weight=0.25,
    )
    oxfordshire_output = build_weak_negative_rows(
        load_jsonl(work_dir / "manifests/oxfordshire-pam-full-train/windows.jsonl"),
        label_authority="site_annotation_weak_frog_absence",
        sample_weight=0.25,
    )

    targets = {
        "frognet-local": local_output,
        "frognet-esc50": esc_output,
        "frognet-ross308-negatives": ross_output,
        "frognet-insectset-negatives": insectset_output,
        "frognet-oxfordshire-negatives": oxfordshire_output,
    }
    summary: dict[str, object] = {}
    for name, rows in targets.items():
        directory = work_dir / "manifests" / name
        path = directory / "windows.jsonl"
        write_jsonl(path, rows)
        source_counts = Counter(
            FROG_CLASS if FROG_CLASS in row["labels"] else (
                "unknown" if FROG_CLASS in row.get("unknown_labels", []) else "absent"
            )
            for row in rows
        )
        item = {
            "windows": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "partitions": dict(Counter(row["partition"] for row in rows)),
            "frog_states": dict(source_counts),
            "windows_manifest_sha256": sha256_file(path),
        }
        write_json(directory / "summary.json", item)
        summary[name] = item
    return summary


def fetch_observations(session: requests.Session) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    page = 1
    total: int | None = None
    while total is None or len(observations) < total:
        response = session.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_id": ANURA_TAXON_ID,
                "place_id": REGIONAL_PLACE_ID,
                "quality_grade": "research",
                "has[]": "sounds",
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
    return observations


def build_inaturalist(work_dir: Path) -> dict[str, object]:
    session = requests.Session()
    session.headers["User-Agent"] = "BioacousticsModelResearch/1.0"
    observations = fetch_observations(session)
    selected, exclusion_counts = select_permissive_inat_sounds(observations)
    if len(selected) < 30 or len({row["observer_id"] for row in selected}) < 10:
        raise ValueError("insufficient permissive regional frog diversity")

    group_rows = [
        {**row, "group_id": f"{SOURCE_ID}:observer:{row['observer_id']}"}
        for row in selected
    ]
    weights = source_balanced_weights(group_rows)
    raw_dir = work_dir / "raw" / SOURCE_ID
    window_dir = work_dir / "windows" / SOURCE_ID
    manifest_dir = work_dir / "manifests" / SOURCE_ID
    recordings: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for index, (source, weight) in enumerate(zip(group_rows, weights, strict=True), start=1):
        content_type = str(source.get("file_content_type") or "")
        extension = ".m4a" if "mp4" in content_type else ".wav"
        raw_path = raw_dir / f"{source['sound_id']}{extension}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            response = session.get(str(source["file_url"]), timeout=120)
            response.raise_for_status()
            temporary = raw_path.with_suffix(raw_path.suffix + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(raw_path)
        duration = duration_seconds(raw_path)
        output_path = window_dir / f"{source['sound_id']}.wav"
        start = make_center_window(raw_path, output_path, duration)
        recording_id = f"{SOURCE_ID}:sound:{source['sound_id']}"
        group_id = str(source["group_id"])
        partition = partition_for_group(group_id)
        recordings.append(
            {
                "schema_version": 1,
                "source_id": SOURCE_ID,
                "recording_id": recording_id,
                "upstream_observation_id": source["observation_id"],
                "upstream_sound_id": source["sound_id"],
                "source_url": source["observation_uri"],
                "media_url": source["file_url"],
                "taxon_id": source["taxon_id"],
                "taxon_name": source["taxon_name"],
                "taxon_common_name": source["taxon_common_name"],
                "quality_grade": source["quality_grade"],
                "license_spdx": source["license_spdx"],
                "attribution": source["attribution"],
                "group_id": group_id,
                "partition": partition,
                "raw_path": str(raw_path.relative_to(work_dir)),
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": sha256_file(raw_path),
                "duration_seconds": duration,
            }
        )
        windows.append(
            {
                "schema_version": 1,
                "window_id": f"{SOURCE_ID}:{source['sound_id']}",
                "recording_id": recording_id,
                "group_id": group_id,
                "partition": partition,
                "labels": [FROG_CLASS],
                "label_authority": "research_grade_source_taxon_weak_window_label",
                "rights_lane": "core_releasable",
                "sample_weight": weight,
                "start_ms": round(start * 1000),
                "end_ms": round((start + 5.0) * 1000),
                "preprocessing_recipe": "ffmpeg-centered5s-32khz-mono-s16-v1",
                "window_path": str(output_path.relative_to(work_dir)),
                "window_sha256": sha256_file(output_path),
            }
        )
        print(f"processed {index}/{len(group_rows)} regional frog sounds", flush=True)

    write_jsonl(manifest_dir / "recordings.jsonl", recordings)
    write_jsonl(manifest_dir / "windows.jsonl", windows)
    write_jsonl(
        manifest_dir / "EXCLUSIONS.jsonl",
        [{"reason": reason, "count": count} for reason, count in exclusion_counts.items()],
    )
    source = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "title": "iNaturalist research-grade regional Anura permissive audio",
        "api_url": "https://api.inaturalist.org/v1/observations",
        "taxon_id": ANURA_TAXON_ID,
        "place_id": REGIONAL_PLACE_ID,
        "query": "taxon_id=20979&place_id=45&quality_grade=research&has[]=sounds",
        "total_observations_with_sound": len(observations),
        "selected_sounds": len(selected),
        "selected_observers": len({row["observer_id"] for row in selected}),
        "license_policy": "sound object license CC0 or CC BY only",
        "location_policy": "coordinates and observation location fields discarded",
        "split_policy": "deterministic SHA-256 split by observer: 70/15/15",
    }
    write_json(manifest_dir / "source.json", source)
    summary = {
        **source,
        "partitions": dict(Counter(row["partition"] for row in windows)),
        "species": dict(Counter(row["taxon_common_name"] for row in recordings)),
        "licenses": dict(Counter(row["license_spdx"] for row in recordings)),
        "recordings_manifest_sha256": sha256_file(manifest_dir / "recordings.jsonl"),
        "windows_manifest_sha256": sha256_file(manifest_dir / "windows.jsonl"),
        "exclusions_manifest_sha256": sha256_file(manifest_dir / "EXCLUSIONS.jsonl"),
    }
    write_json(manifest_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provenance-safe FrogNet corpora")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--skip-inaturalist", action="store_true")
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    report: dict[str, object] = {"derived": build_derived_manifests(work_dir)}
    if not args.skip_inaturalist:
        report["inaturalist"] = build_inaturalist(work_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
