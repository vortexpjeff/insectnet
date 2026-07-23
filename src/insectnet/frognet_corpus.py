from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FROG_CLASS = "frog_present"
PERMISSIVE_SOUND_LICENSES = {"cc0": "CC0-1.0", "cc-by": "CC-BY-4.0"}


def partition_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def build_esc50_frog_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["labels"] = [FROG_CLASS] if str(row.get("raw_category")) == "frog" else []
        row.pop("unknown_labels", None)
        row["label_authority"] = "dataset_category_label"
        output.append(row)
    return output


def build_weak_negative_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_authority: str,
    sample_weight: float,
) -> list[dict[str, Any]]:
    if not 0.0 < sample_weight <= 1.0:
        raise ValueError("sample_weight must be in (0, 1]")
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["labels"] = []
        row.pop("unknown_labels", None)
        row["label_authority"] = label_authority
        row["sample_weight"] = sample_weight
        output.append(row)
    return output


def _recording_timestamp(recording: Mapping[str, Any]) -> datetime:
    source_path = Path(str(recording["private_source_path"]))
    return datetime.strptime(source_path.name[:19], "%Y-%m-%d_%H:%M:%S")


def _local_groups(
    recordings: Sequence[Mapping[str, Any]], *, session_gap_seconds: int
) -> dict[str, tuple[str, str]]:
    ordered = sorted(recordings, key=_recording_timestamp)
    result: dict[str, tuple[str, str]] = {}
    previous: datetime | None = None
    session_start: datetime | None = None
    for recording in ordered:
        timestamp = _recording_timestamp(recording)
        if previous is None or (timestamp - previous).total_seconds() > session_gap_seconds:
            session_start = timestamp
        assert session_start is not None
        group_id = f"private-local-frog:session:{session_start.strftime('%Y%m%dT%H%M%S')}"
        if timestamp.date().isoformat() == "2026-05-29":
            partition = "validation"
        elif timestamp.date().isoformat() == "2026-05-31":
            partition = "test"
        else:
            partition = "train"
        result[str(recording["recording_id"])] = (group_id, partition)
        previous = timestamp
    return result


def build_local_frog_rows(
    rows: Sequence[Mapping[str, Any]],
    recordings: Sequence[Mapping[str, Any]],
    *,
    uncertain_recordings: set[str] | None = None,
    uncertain_window_ids: set[str] | None = None,
    session_gap_seconds: int = 300,
) -> list[dict[str, Any]]:
    uncertain = uncertain_recordings or set()
    uncertain_windows = uncertain_window_ids or set()
    groups = _local_groups(recordings, session_gap_seconds=session_gap_seconds)
    missing = sorted({str(row["recording_id"]) for row in rows} - set(groups))
    if missing:
        raise ValueError(f"local frog windows reference missing recordings: {missing[:3]}")

    eligible_counts = Counter(
        groups[str(row["recording_id"])][0]
        for row in rows
        if str(row["recording_id"]) not in uncertain
        and str(row["window_id"]) not in uncertain_windows
    )
    if not eligible_counts:
        raise ValueError("local frog corpus contains no eligible positive windows")
    # Normalize inverse group-size weights back to a mean sample weight of 1.0.
    # This preserves the source's total evidence mass while giving every session
    # the same aggregate contribution.
    mean_inverse = len(eligible_counts) / sum(eligible_counts.values())

    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        recording_id = str(row["recording_id"])
        group_id, partition = groups[recording_id]
        row["group_id"] = group_id
        row["partition"] = partition
        row["label_authority"] = "reviewed_private_archive_with_perch_screen"
        if recording_id in uncertain or str(row["window_id"]) in uncertain_windows:
            row["labels"] = []
            row["unknown_labels"] = [FROG_CLASS]
            row["sample_weight"] = 1.0
        else:
            row["labels"] = [FROG_CLASS]
            row.pop("unknown_labels", None)
            # Every local calling session contributes equal total mass.
            row["sample_weight"] = (1.0 / eligible_counts[group_id]) / mean_inverse
        output.append(row)
    return output


def select_permissive_inat_sounds(
    observations: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_sound_ids: set[int] = set()
    for observation in observations:
        user = observation.get("user") or {}
        taxon = observation.get("taxon") or {}
        for sound in observation.get("sounds", []) or []:
            normalized_license = PERMISSIVE_SOUND_LICENSES.get(str(sound.get("license_code")))
            if normalized_license is None:
                exclusions[f"license:{sound.get('license_code')}"] += 1
                continue
            sound_id = int(sound["id"])
            if sound_id in seen_sound_ids:
                exclusions["duplicate_sound_id"] += 1
                continue
            seen_sound_ids.add(sound_id)
            selected.append(
                {
                    "observation_id": int(observation["id"]),
                    "observer_id": int(user["id"]),
                    "sound_id": sound_id,
                    "file_url": str(sound["file_url"]),
                    "file_content_type": sound.get("file_content_type"),
                    "license_spdx": normalized_license,
                    "attribution": sound.get("attribution"),
                    "taxon_id": int(taxon["id"]),
                    "taxon_name": str(taxon["name"]),
                    "taxon_common_name": taxon.get("preferred_common_name"),
                    "quality_grade": observation.get("quality_grade"),
                    "observation_uri": str(observation["uri"]),
                }
            )
    selected.sort(key=lambda row: (row["observation_id"], row["sound_id"]))
    return selected, dict(sorted(exclusions.items()))


def source_balanced_weights(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    group_counts = Counter(str(row["group_id"]) for row in rows)
    if not group_counts:
        return []
    raw = [1.0 / group_counts[str(row["group_id"])] for row in rows]
    mean = sum(raw) / len(raw)
    return [value / mean for value in raw]
