from __future__ import annotations

import numpy as np
import pytest

from insectnet.candidate import (
    active_labels,
    canonical_manifest_hash,
    fit_candidate,
    predict_candidate,
    validate_manifest,
)


def row(
    window_id: str,
    labels: list[str],
    group_id: str,
    partition: str,
    sample_weight: float = 1.0,
    unknown_labels: list[str] | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "window_id": window_id,
        "labels": labels,
        "group_id": group_id,
        "partition": partition,
        "sample_weight": sample_weight,
        "rights_lane": "core_releasable",
        "recording_id": f"recording:{group_id}",
    }
    if unknown_labels is not None:
        item["unknown_labels"] = unknown_labels
    return item


def test_manifest_hash_covers_exact_training_identity() -> None:
    rows = [
        row("w2", ["cicada", "insect_present"], "g2", "validation", 0.5),
        row("w1", ["insect_present"], "g1", "train"),
    ]

    original = canonical_manifest_hash(rows)
    assert original == "1f8ae9f30eaa1f415425f27e82dc5a1841b2c066df220fdbc5a93217e872a517"
    assert original == canonical_manifest_hash(list(reversed(rows)))

    changed = [dict(item) for item in rows]
    changed[0]["sample_weight"] = 0.75
    assert canonical_manifest_hash(changed) != original

    changed = [dict(item) for item in rows]
    changed[0]["labels"] = ["cricket_katydid", "insect_present"]
    assert canonical_manifest_hash(changed) != original

    changed = [dict(item) for item in rows]
    changed[0]["partition"] = "test"
    assert canonical_manifest_hash(changed) != original

    changed = [dict(item) for item in rows]
    changed[0]["unknown_labels"] = ["cricket_katydid"]
    assert canonical_manifest_hash(changed) != original


def test_manifest_validation_rejects_duplicate_windows_and_group_leakage() -> None:
    with pytest.raises(ValueError, match="duplicate window_id"):
        validate_manifest(
            [
                row("same", ["insect_present"], "g1", "train"),
                row("same", [], "g2", "validation"),
            ]
        )

    with pytest.raises(ValueError, match="group g1 crosses partitions"):
        validate_manifest(
            [
                row("w1", ["insect_present"], "g1", "train"),
                row("w2", [], "g1", "test"),
            ]
        )

    with pytest.raises(ValueError, match="both known-positive and unknown"):
        validate_manifest(
            [
                row(
                    "w1",
                    ["insect_present"],
                    "g1",
                    "train",
                    unknown_labels=["insect_present"],
                )
            ]
        )


def test_candidate_trains_independent_heads_and_preserves_abstention() -> None:
    rows = [
        row("t1", ["insect_present", "cicada"], "t1", "train"),
        row("t2", ["insect_present", "cicada"], "t2", "train"),
        row("t3", ["insect_present", "cricket_katydid"], "t3", "train"),
        row("t4", ["insect_present", "cricket_katydid"], "t4", "train"),
        row("t5", [], "t5", "train"),
        row("t6", [], "t6", "train"),
        row("v1", ["insect_present", "cicada"], "v1", "validation"),
        row("v2", ["insect_present", "cricket_katydid"], "v2", "validation"),
        row("v3", [], "v3", "validation"),
        row("v4", [], "v4", "validation"),
        row("e1", ["insect_present", "cicada"], "e1", "test"),
        row("e2", [], "e2", "test"),
    ]
    X = np.array(
        [
            [3.0, 3.0],
            [2.7, 3.2],
            [3.0, -3.0],
            [2.8, -2.6],
            [-3.0, 0.0],
            [-2.7, 0.2],
            [3.1, 3.1],
            [3.2, -3.1],
            [-3.1, 0.1],
            [-2.9, -0.2],
            [3.1, 2.9],
            [-3.2, 0.0],
        ],
        dtype=np.float32,
    )
    classes = ["insect_present", "cicada", "cricket_katydid"]

    package = fit_candidate(
        X,
        rows,
        classes=classes,
        label_hierarchy={
            "cicada": "insect_present",
            "cricket_katydid": "insect_present",
        },
        random_state=42,
    )
    predictions = predict_candidate(package, X)

    assert package["classes"] == classes
    assert package["dataset_hash"] == canonical_manifest_hash(rows)
    assert set(package["thresholds"]) == set(classes)
    assert package["label_hierarchy"] == {
        "cicada": "insect_present",
        "cricket_katydid": "insect_present",
    }
    assert set(package["metrics"]) == {"validation", "test"}
    assert predictions[0]["insect_present"] > predictions[5]["insect_present"]
    assert predictions[0]["cicada"] > predictions[2]["cicada"]
    assert predictions[2]["cricket_katydid"] > predictions[0]["cricket_katydid"]

    active = active_labels(package, predictions[8])
    assert active == []


def test_candidate_excludes_unknown_labels_per_head() -> None:
    rows = [
        row("t1", ["insect_present", "cicada"], "t1", "train"),
        row(
            "t2",
            ["insect_present", "cricket_katydid"],
            "t2",
            "train",
            unknown_labels=["cicada"],
        ),
        row("t3", [], "t3", "train"),
        row("v1", ["insect_present", "cicada"], "v1", "validation"),
        row(
            "v2",
            ["insect_present", "cricket_katydid"],
            "v2",
            "validation",
            unknown_labels=["cicada"],
        ),
        row("v3", [], "v3", "validation"),
        row("e1", ["insect_present", "cicada"], "e1", "test"),
        row(
            "e2",
            ["insect_present", "cricket_katydid"],
            "e2",
            "test",
            unknown_labels=["cicada"],
        ),
        row("e3", [], "e3", "test"),
    ]
    X = np.asarray(
        [
            [3.0, 3.0],
            [3.0, -3.0],
            [-3.0, 0.0],
            [3.2, 3.1],
            [3.1, -3.0],
            [-3.1, 0.0],
            [3.1, 3.0],
            [3.0, -3.1],
            [-3.2, 0.0],
        ],
        dtype=np.float32,
    )
    package = fit_candidate(
        X,
        rows,
        classes=["insect_present", "cicada"],
        label_hierarchy={"cicada": "insect_present"},
    )

    assert package["head_eligibility"]["insect_present"] == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert package["head_eligibility"]["cicada"] == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert (
        package["metrics"]["test"]["per_class"]["cicada"]["evaluated_samples"]
        == 2
    )


def test_active_labels_enforces_parent_hierarchy() -> None:
    package = {
        "classes": ["insect_present", "cicada", "orthoptera"],
        "thresholds": {"insect_present": 0.7, "cicada": 0.1, "orthoptera": 0.8},
        "label_hierarchy": {
            "cicada": "insect_present",
            "orthoptera": "insect_present",
        },
    }

    assert active_labels(
        package,
        {"insect_present": 0.6, "cicada": 0.9, "orthoptera": 0.95},
    ) == []
    assert active_labels(
        package,
        {"insect_present": 0.8, "cicada": 0.9, "orthoptera": 0.2},
    ) == ["insect_present", "cicada"]
