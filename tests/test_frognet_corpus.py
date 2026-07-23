from __future__ import annotations

from collections import Counter

from insectnet.frognet_corpus import (
    build_esc50_frog_rows,
    build_local_frog_rows,
    build_weak_negative_rows,
    partition_for_group,
    select_permissive_inat_sounds,
)


def test_partition_for_group_is_deterministic_and_complete() -> None:
    values = {partition_for_group(f"observer:{index}") for index in range(200)}
    assert values == {"train", "validation", "test"}
    assert partition_for_group("observer:17") == partition_for_group("observer:17")


def test_esc50_frog_rows_relabel_only_frog_category() -> None:
    rows = [
        {
            "window_id": "frog-window",
            "recording_id": "frog-recording",
            "group_id": "source:1",
            "partition": "train",
            "labels": ["old_label"],
            "raw_category": "frog",
            "rights_lane": "research_noncommercial",
            "sample_weight": 0.25,
        },
        {
            "window_id": "bird-window",
            "recording_id": "bird-recording",
            "group_id": "source:2",
            "partition": "test",
            "labels": ["old_label"],
            "raw_category": "chirping_birds",
            "rights_lane": "research_noncommercial",
            "sample_weight": 0.25,
        },
    ]
    output = build_esc50_frog_rows(rows)
    assert [row["window_id"] for row in output] == ["frog-window", "bird-window"]
    assert output[0]["labels"] == ["frog_present"]
    assert output[1]["labels"] == []
    assert all(row["label_authority"] == "dataset_category_label" for row in output)


def test_weak_negative_rows_clear_target_labels_and_downweight() -> None:
    source = [{
        "window_id": "insect-window",
        "recording_id": "recording",
        "group_id": "contributor:1",
        "partition": "train",
        "labels": ["insect_present", "orthoptera"],
        "unknown_labels": ["cicada"],
        "sample_weight": 1.0,
    }]
    output = build_weak_negative_rows(
        source,
        label_authority="source_taxon_weak_frog_absence",
        sample_weight=0.25,
    )
    assert output[0]["labels"] == []
    assert "unknown_labels" not in output[0]
    assert output[0]["sample_weight"] == 0.25
    assert output[0]["label_authority"] == "source_taxon_weak_frog_absence"


def test_local_rows_group_adjacent_recordings_and_hold_out_dates() -> None:
    recordings = [
        {
            "recording_id": "r1",
            "private_source_path": "/private/2026-05-29_19:00:00_frog.wav",
        },
        {
            "recording_id": "r2",
            "private_source_path": "/private/2026-05-30_20:00:00_frog.wav",
        },
        {
            "recording_id": "r3",
            "private_source_path": "/private/2026-05-30_20:00:15_frog.wav",
        },
        {
            "recording_id": "r4",
            "private_source_path": "/private/2026-05-30_20:10:00_frog.wav",
        },
        {
            "recording_id": "r5",
            "private_source_path": "/private/2026-05-31_07:00:00_frog.wav",
        },
    ]
    windows = []
    for recording in recordings:
        for index in range(3):
            windows.append(
                {
                    "window_id": f"{recording['recording_id']}:{index}",
                    "recording_id": recording["recording_id"],
                    "group_id": recording["recording_id"],
                    "partition": "test",
                    "labels": [],
                    "rights_lane": "private_diagnostic_only",
                    "sample_weight": 1.0,
                }
            )

    output = build_local_frog_rows(
        windows,
        recordings,
        uncertain_recordings={"r4"},
        uncertain_window_ids={"r2:1"},
    )
    by_recording = {row["recording_id"]: row for row in output}
    by_window = {row["window_id"]: row for row in output}
    assert by_recording["r1"]["partition"] == "validation"
    assert by_recording["r2"]["partition"] == "train"
    assert by_recording["r3"]["group_id"] == by_recording["r2"]["group_id"]
    assert by_recording["r4"]["group_id"] != by_recording["r2"]["group_id"]
    assert by_recording["r5"]["partition"] == "test"
    assert by_recording["r4"]["unknown_labels"] == ["frog_present"]
    assert by_recording["r4"]["labels"] == []
    assert by_recording["r2"]["labels"] == ["frog_present"]
    assert by_window["r2:1"]["unknown_labels"] == ["frog_present"]
    assert by_window["r2:1"]["labels"] == []

    group_weight = Counter()
    eligible = 0
    for row in output:
        if "frog_present" not in row.get("unknown_labels", []):
            group_weight[row["group_id"]] += row["sample_weight"]
            eligible += 1
    assert len(set(round(value, 12) for value in group_weight.values())) == 1
    assert sum(group_weight.values()) == eligible


def test_select_permissive_inat_sounds_filters_license_and_location() -> None:
    observations = [
        {
            "id": 10,
            "quality_grade": "research",
            "uri": "https://example.test/observations/10",
            "location": "must-not-survive",
            "geojson": {"coordinates": [-83.0, 35.0]},
            "user": {"id": 20},
            "taxon": {
                "id": 30,
                "name": "Dryophytes chrysoscelis",
                "preferred_common_name": "Cope's Gray Tree Frog",
            },
            "sounds": [
                {
                    "id": 40,
                    "license_code": "cc-by",
                    "file_url": "https://example.test/40.wav",
                    "file_content_type": "audio/wav",
                    "attribution": "Observer, CC BY",
                },
                {
                    "id": 41,
                    "license_code": "cc-by-nc",
                    "file_url": "https://example.test/41.wav",
                },
            ],
        }
    ]
    selected, exclusions = select_permissive_inat_sounds(observations)
    assert len(selected) == 1
    assert selected[0]["sound_id"] == 40
    assert selected[0]["license_spdx"] == "CC-BY-4.0"
    assert "location" not in selected[0]
    assert "geojson" not in selected[0]
    assert exclusions == {"license:cc-by-nc": 1}
