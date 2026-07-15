#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests

HF_REPO = "academic-datasets/InsectSet459"
ZENODO_RECORD = "18554693"
SOURCE_ID = "insectset459-v1.1-hf-11b209d"
PERMISSIVE_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def normalize_license(uri: str) -> str | None:
    normalized = uri.strip()
    if normalized.startswith("//"):
        normalized = "http:" + normalized
    return PERMISSIVE_LICENSES.get(normalized)


def stable_partition(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:120] or "recording"


def request_bytes(url: str, *, timeout: int = 300) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def resolve_annotation_csv(session: requests.Session) -> tuple[bytes, dict[str, Any]]:
    record = session.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60)
    record.raise_for_status()
    metadata = record.json()
    for file_info in metadata["files"]:
        if file_info["key"].endswith("Annotation.csv"):
            response = session.get(file_info["links"]["self"], timeout=120)
            response.raise_for_status()
            return response.content, {
                "filename": file_info["key"],
                "bytes": file_info["size"],
                "upstream_checksum": file_info["checksum"],
                "sha256": sha256_bytes(response.content),
            }
    raise RuntimeError("InsectSet459 annotation CSV was not found in the Zenodo record")


def load_annotation_map(content: bytes) -> dict[str, dict[str, str]]:
    text = content.decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    return {row["file_name"]: row for row in rows}


def resolve_gbif_taxonomy(
    session: requests.Session,
    species_names: list[str],
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    expected = set(species_names)
    if output_path.exists():
        cached = {
            row["species_name"]: row
            for row in (
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        if expected.issubset(cached):
            print(f"[taxonomy] reuse {len(cached)} cached GBIF records", flush=True)
            return cached
    taxonomy: dict[str, dict[str, Any]] = {}
    output_path.unlink(missing_ok=True)
    for index, species_name in enumerate(sorted(set(species_names)), start=1):
        match = session.get(
            "https://api.gbif.org/v1/species/match",
            params={"name": species_name.replace("_", " "), "strict": "true"},
            timeout=60,
        )
        match.raise_for_status()
        match_data = match.json()
        usage_key = match_data.get("usageKey")
        parents: list[dict[str, Any]] = []
        if usage_key:
            parent_response = session.get(
                f"https://api.gbif.org/v1/species/{usage_key}/parents", timeout=60
            )
            parent_response.raise_for_status()
            parents = parent_response.json()
        ranks = {
            str(parent.get("rank", "")).upper(): parent.get("scientificName")
            for parent in parents
        }
        record = {
            "species_name": species_name,
            "query": species_name.replace("_", " "),
            "usage_key": usage_key,
            "match_type": match_data.get("matchType"),
            "confidence": match_data.get("confidence"),
            "canonical_name": match_data.get("canonicalName"),
            "status": match_data.get("status"),
            "order": ranks.get("ORDER") or match_data.get("order"),
            "suborder": ranks.get("SUBORDER"),
            "family": ranks.get("FAMILY") or match_data.get("family"),
            "authority": "GBIF Species API v1",
            "match_url": "https://api.gbif.org/v1/species/match",
            "parents_url": (
                f"https://api.gbif.org/v1/species/{usage_key}/parents" if usage_key else None
            ),
        }
        taxonomy[species_name] = record
        append_jsonl(output_path, record)
        print(
            f"[taxonomy {index}/{len(set(species_names))}] {species_name}: "
            f"{record['suborder'] or record['family'] or 'unresolved'}",
            flush=True,
        )
    return taxonomy


def labels_for(annotation: dict[str, str], taxonomy: dict[str, Any]) -> list[str]:
    labels = ["insect_present"]
    group = annotation.get("group", "").strip().lower()
    if group == "cicadidae":
        labels.append("cicada")
    elif group == "orthoptera":
        labels.append("orthoptera")
    return labels


def make_window(raw_path: Path, output_path: Path) -> tuple[float, float]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(raw_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    duration = float(probe.stdout.strip())
    if duration <= 0:
        raise ValueError("non-positive audio duration")
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
    return duration, start


def build(
    work_dir: Path,
    *,
    keep_shards: bool = False,
    max_shards_per_split: int | None = None,
    resume_train_index: int | None = None,
    resume_processed_rows: int | None = None,
    shard_plan_path: Path | None = None,
) -> None:
    manifest_dir = work_dir / "manifests" / "insectnet"
    raw_dir = work_dir / "raw" / "insectset459"
    window_dir = work_dir / "windows" / "insectset459"
    cache_dir = work_dir / "cache" / "insectset459"
    for directory in (manifest_dir, raw_dir, window_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    recordings_path = manifest_dir / "recordings.jsonl"
    windows_path = manifest_dir / "windows.jsonl"
    downloads_path = manifest_dir / "downloads.jsonl"
    exclusions_path = manifest_dir / "EXCLUSIONS.jsonl"
    resume = resume_train_index is not None
    if not resume:
        for path in (recordings_path, windows_path, downloads_path, exclusions_path):
            path.unlink(missing_ok=True)
            path.touch()
    elif resume_processed_rows is None:
        raise ValueError("--resume-processed-rows is required with --resume-train-index")

    session = requests.Session()
    session.headers["User-Agent"] = "BioacousticsModelResearch/1.0"
    hf_info = session.get(f"https://huggingface.co/api/datasets/{HF_REPO}", timeout=60)
    hf_info.raise_for_status()
    hf_metadata = hf_info.json()
    revision = hf_metadata["sha"]

    annotation_content, annotation_download = resolve_annotation_csv(session)
    annotation_map = load_annotation_map(annotation_content)
    annotation_path = manifest_dir / annotation_download["filename"]
    annotation_path.write_bytes(annotation_content)
    taxonomy_path = manifest_dir / "gbif_taxonomy.jsonl"
    taxonomy_map = resolve_gbif_taxonomy(
        session,
        [
            row["species_name"]
            for row in annotation_map.values()
            if normalize_license(row.get("license", "")) is not None
        ],
        taxonomy_path,
    )

    write_json(
        manifest_dir / "source.json",
        {
            "schema_version": 1,
            "source_id": SOURCE_ID,
            "title": "InsectSet459 v1.1 permissive subset",
            "version_doi": "10.5281/zenodo.18554693",
            "repository": f"https://huggingface.co/datasets/{HF_REPO}",
            "repository_revision": revision,
            "retrieved_at": subprocess.check_output(
                ["date", "--iso-8601=seconds"], text=True
            ).strip(),
            "rights_lane": "core_releasable",
            "accepted_licenses": sorted(PERMISSIVE_LICENSES.values()),
            "annotation": annotation_download,
            "taxonomy_authority": "GBIF Species API v1",
            "taxonomy_manifest_sha256": sha256_file(taxonomy_path),
            "notes": [
                "HF conversion exposes train and validation only; original test archive is absent.",
                "Partitions are reassigned deterministically by contributor to prevent contributor leakage.",
                "One centered five-second weak-label window is derived per retained recording.",
                "The source supports Cicadidae versus Orthoptera; no unsupported suborder label is inferred.",
            ],
        },
    )

    retained = Counter()
    excluded = Counter()
    seen_recordings: set[str] = set()
    if resume:
        assert resume_train_index is not None
        assert resume_processed_rows is not None
        existing_recordings = load_jsonl(recordings_path)
        existing_windows = load_jsonl(windows_path)
        existing_exclusions = load_jsonl(exclusions_path)
        if len(existing_recordings) != len(existing_windows):
            raise ValueError("resume manifests have different recording/window counts")
        seen_recordings.update(str(row["recording_id"]) for row in existing_recordings)
        for row in existing_recordings:
            retained[str(row["partition"])] += 1
            retained[str(row["license_spdx"])] += 1
        for row in existing_windows:
            labels = set(row["labels"])
            retained["Cicadidae" if "cicada" in labels else "Orthoptera"] += 1
        for row in existing_exclusions:
            excluded[str(row["reason"])] += 1
        incompatible = resume_processed_rows - len(existing_recordings) - len(existing_exclusions)
        if incompatible < 0:
            raise ValueError("resume processed-row count is inconsistent with existing manifests")
        excluded["incompatible_license"] = incompatible
        completed_downloads = [
            row
            for row in load_jsonl(downloads_path)
            if row["api_split"] != "train"
            or int(str(row["shard"]).split("-")[1].split(".")[0]) < resume_train_index
        ]
        write_jsonl(downloads_path, completed_downloads)
        print(
            f"[resume] recordings={len(existing_recordings)} "
            f"processed_rows={resume_processed_rows} train_index={resume_train_index}",
            flush=True,
        )
    permissive_license_values = sorted(
        {
            row["license"]
            for row in annotation_map.values()
            if normalize_license(row.get("license", "")) is not None
        }
    )
    expected_by_api_split = {
        "train": {
            row["file_name"]
            for row in annotation_map.values()
            if row["subset"] == "Train"
            and normalize_license(row.get("license", "")) is not None
        },
        "validation": {
            row["file_name"]
            for row in annotation_map.values()
            if row["subset"] == "Validation"
            and normalize_license(row.get("license", "")) is not None
        },
    }
    acquired_file_names = {
        str(row["upstream_file_name"]) for row in load_jsonl(recordings_path)
    }
    shard_plan = None
    shard_plan_sha = None
    planned_by_split: dict[str, dict[int, dict[str, Any]]] = {}
    if shard_plan_path is not None:
        shard_plan = json.loads(shard_plan_path.read_text(encoding="utf-8"))
        shard_plan_sha = sha256_file(shard_plan_path)
        planned_by_split = {
            split: {int(row["index"]): row for row in rows}
            for split, rows in shard_plan["splits"].items()
        }
    splits = ("train", "validation")
    for api_split in splits:
        expected_file_names = expected_by_api_split[api_split]
        if expected_file_names <= acquired_file_names:
            print(
                f"[{api_split}] all {len(expected_file_names)} permissive files already acquired",
                flush=True,
            )
            continue
        split_response = session.get(
            f"https://huggingface.co/api/datasets/{HF_REPO}/parquet/default/{api_split}",
            timeout=60,
        )
        split_response.raise_for_status()
        shard_urls = split_response.json()
        if max_shards_per_split is not None:
            shard_urls = shard_urls[:max_shards_per_split]
        for shard_index, shard_url in enumerate(shard_urls, start=1):
            zero_index = shard_index - 1
            if api_split == "train" and resume_train_index is not None:
                if zero_index < resume_train_index:
                    continue
            planned = planned_by_split.get(api_split, {}).get(zero_index)
            if planned is not None and not planned["missing_permissive_files"]:
                permissive_rows = len(planned["permissive_files"])
                excluded["incompatible_license"] += int(planned["source_rows"]) - permissive_rows
                excluded["duplicate_or_already_acquired_permissive"] += permissive_rows
                append_jsonl(
                    downloads_path,
                    {
                        "api_split": api_split,
                        "shard": f"{api_split}-{zero_index:05d}.parquet",
                        "source_url": shard_url,
                        "repository_revision": revision,
                        "source_rows": int(planned["source_rows"]),
                        "filtered_permissive_rows": permissive_rows,
                        "status": "skipped_no_missing_permissive_files",
                        "access_method": "remote_parquet_range_metadata",
                        "shard_plan_sha256": shard_plan_sha,
                    },
                )
                continue
            shard_name = f"{api_split}-{shard_index - 1:05d}.parquet"
            shard_path = cache_dir / shard_name
            if shard_path.exists():
                try:
                    pq.ParquetFile(shard_path)
                except Exception:
                    print(f"[{api_split}] discard incomplete cache {shard_name}", flush=True)
                    shard_path.unlink()
            action = "reuse" if shard_path.exists() else "download"
            print(f"[{api_split} {shard_index}/{len(shard_urls)}] {action} {shard_name}", flush=True)
            if not shard_path.exists():
                with session.get(shard_url, stream=True, timeout=300) as response:
                    response.raise_for_status()
                    with shard_path.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
            shard_sha = sha256_file(shard_path)
            parquet_file = pq.ParquetFile(shard_path)
            source_rows = parquet_file.metadata.num_rows
            table = pq.read_table(
                shard_path,
                filters=[("license", "in", permissive_license_values)],
            )
            excluded["incompatible_license"] += source_rows - table.num_rows
            for row in table.to_pylist():
                license_id = normalize_license(row["license"] or "")
                if license_id is None:
                    raise ValueError("Parquet license filter admitted an incompatible row")
                file_name = row["file_name"]
                annotation = annotation_map.get(file_name)
                if annotation is None:
                    excluded["missing_annotation"] += 1
                    append_jsonl(
                        exclusions_path,
                        {
                            "source_id": SOURCE_ID,
                            "upstream_id": file_name,
                            "reason": "missing_annotation",
                        },
                    )
                    continue
                audio = row["audio"]
                audio_bytes = audio.get("bytes") if audio else None
                if not audio_bytes:
                    excluded["missing_audio_bytes"] += 1
                    append_jsonl(
                        exclusions_path,
                        {
                            "source_id": SOURCE_ID,
                            "upstream_id": file_name,
                            "reason": "missing_audio_bytes",
                        },
                    )
                    continue
                observation = row.get("observation") or file_name
                recording_id = f"{SOURCE_ID}:{observation}"
                if recording_id in seen_recordings:
                    excluded["duplicate_recording_id"] += 1
                    append_jsonl(
                        exclusions_path,
                        {
                            "source_id": SOURCE_ID,
                            "upstream_id": file_name,
                            "recording_id": recording_id,
                            "reason": "duplicate_recording_id",
                        },
                    )
                    continue
                seen_recordings.add(recording_id)
                acquired_file_names.add(file_name)

                audio_sha = sha256_bytes(audio_bytes)
                extension = Path(file_name).suffix.lower() or ".audio"
                raw_name = f"{audio_sha[:16]}_{safe_component(file_name)}"
                if not raw_name.endswith(extension):
                    raw_name += extension
                raw_path = raw_dir / raw_name
                raw_path.write_bytes(audio_bytes)
                window_id = f"{SOURCE_ID}:{audio_sha[:24]}:center5s"
                window_path = window_dir / f"{audio_sha[:24]}.wav"
                try:
                    duration, start = make_window(raw_path, window_path)
                except Exception as error:
                    excluded["window_derivation_failed"] += 1
                    append_jsonl(
                        exclusions_path,
                        {
                            "source_id": SOURCE_ID,
                            "upstream_id": file_name,
                            "recording_id": recording_id,
                            "reason": "window_derivation_failed",
                            "error": str(error),
                        },
                    )
                    raw_path.unlink(missing_ok=True)
                    continue

                contributor = (row.get("contributor") or "unknown").strip()
                group_id = f"insectset459:contributor:{contributor}"
                partition = stable_partition(group_id)
                taxonomy = taxonomy_map.get(row.get("species_name") or "", {})
                normalized_labels = labels_for(annotation, taxonomy)
                recording_row = {
                    "schema_version": 1,
                    "source_id": SOURCE_ID,
                    "recording_id": recording_id,
                    "upstream_file_name": file_name,
                    "upstream_observation": observation,
                    "source_url": row.get("file"),
                    "source_subset": row.get("subset"),
                    "raw_label": row.get("species_name"),
                    "taxon_order": taxonomy.get("order"),
                    "taxon_suborder": taxonomy.get("suborder"),
                    "taxon_family": taxonomy.get("family"),
                    "gbif_usage_key": taxonomy.get("usage_key"),
                    "contributor": contributor,
                    "license_uri": row.get("license"),
                    "license_spdx": license_id,
                    "rights_lane": "core_releasable",
                    "group_id": group_id,
                    "partition": partition,
                    "raw_path": str(raw_path.relative_to(work_dir)),
                    "raw_bytes": len(audio_bytes),
                    "raw_sha256": audio_sha,
                    "duration_seconds": duration,
                    "review_authority": "source_weak_label",
                    "repository_revision": revision,
                    "source_shard": shard_name,
                    "source_shard_sha256": shard_sha,
                }
                window_row = {
                    "schema_version": 1,
                    "window_id": window_id,
                    "recording_id": recording_id,
                    "group_id": group_id,
                    "partition": partition,
                    "labels": normalized_labels,
                    "raw_species_label": row.get("species_name"),
                    "label_authority": "source_weak_label",
                    "uncertainty": "weak_file_label_center_crop",
                    "rights_lane": "core_releasable",
                    "sample_weight": 1.0,
                    "start_ms": round(start * 1000),
                    "end_ms": round((start + 5.0) * 1000),
                    "preprocessing_recipe": "ffmpeg-center5s-32khz-mono-s16-v1",
                    "window_path": str(window_path.relative_to(work_dir)),
                    "window_sha256": sha256_file(window_path),
                }
                append_jsonl(recordings_path, recording_row)
                append_jsonl(windows_path, window_row)
                retained[partition] += 1
                retained[license_id] += 1
                retained[row.get("group") or "unknown"] += 1

            append_jsonl(
                downloads_path,
                {
                    "api_split": api_split,
                    "shard": shard_name,
                    "source_url": shard_url,
                    "bytes": shard_path.stat().st_size,
                    "sha256": shard_sha,
                    "repository_revision": revision,
                    "source_rows": source_rows,
                    "filtered_permissive_rows": table.num_rows,
                    "status": "completed",
                },
            )

            if not keep_shards:
                shard_path.unlink(missing_ok=True)
            print(
                f"  retained={sum(retained[p] for p in ('train','validation','test'))} "
                f"excluded={sum(excluded.values())}",
                flush=True,
            )
            if expected_file_names <= acquired_file_names:
                print(
                    f"[{api_split}] acquired all {len(expected_file_names)} permissive files; "
                    "skip remaining restricted-only shards",
                    flush=True,
                )
                break

    summary = {
        "source_id": SOURCE_ID,
        "repository_revision": revision,
        "retained": dict(retained),
        "excluded": dict(excluded),
        "recordings_manifest_sha256": sha256_file(recordings_path),
        "windows_manifest_sha256": sha256_file(windows_path),
        "downloads_manifest_sha256": sha256_file(downloads_path),
        "exclusions_manifest_sha256": sha256_file(exclusions_path),
        "max_shards_per_split": max_shards_per_split,
        "complete_source_scan": max_shards_per_split is None,
        "shard_plan": str(shard_plan_path) if shard_plan_path is not None else None,
        "shard_plan_sha256": shard_plan_sha,
    }
    write_json(manifest_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exact CC0/CC-BY InsectSet459 Perch-window corpus"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--max-shards-per-split", type=int)
    parser.add_argument("--resume-train-index", type=int)
    parser.add_argument("--resume-processed-rows", type=int)
    parser.add_argument("--shard-plan", type=Path)
    args = parser.parse_args()
    build(
        args.work_dir.resolve(),
        keep_shards=args.keep_shards,
        max_shards_per_split=args.max_shards_per_split,
        resume_train_index=args.resume_train_index,
        resume_processed_rows=args.resume_processed_rows,
        shard_plan_path=args.shard_plan.resolve() if args.shard_plan else None,
    )


if __name__ == "__main__":
    main()
