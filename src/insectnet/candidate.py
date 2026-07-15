from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

PARTITIONS = {"train", "validation", "test"}
MANIFEST_IDENTITY_FIELDS = (
    "window_id",
    "recording_id",
    "labels",
    "group_id",
    "partition",
    "sample_weight",
    "rights_lane",
)


def _canonical_row(row: Mapping[str, object]) -> dict[str, object]:
    missing = [key for key in MANIFEST_IDENTITY_FIELDS if key not in row]
    if missing:
        raise ValueError(f"manifest row missing fields: {', '.join(missing)}")
    return {
        "window_id": str(row["window_id"]),
        "recording_id": str(row["recording_id"]),
        "labels": sorted({str(label) for label in row["labels"]}),
        "group_id": str(row["group_id"]),
        "partition": str(row["partition"]),
        "sample_weight": float(row["sample_weight"]),
        "rights_lane": str(row["rights_lane"]),
    }


def validate_manifest(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("manifest is empty")

    window_ids: set[str] = set()
    group_partitions: dict[str, str] = {}
    for row in rows:
        item = _canonical_row(row)
        window_id = str(item["window_id"])
        if window_id in window_ids:
            raise ValueError(f"duplicate window_id: {window_id}")
        window_ids.add(window_id)

        partition = str(item["partition"])
        if partition not in PARTITIONS:
            raise ValueError(f"invalid partition: {partition}")
        if float(item["sample_weight"]) <= 0:
            raise ValueError(f"sample_weight must be positive for {window_id}")

        group_id = str(item["group_id"])
        previous = group_partitions.setdefault(group_id, partition)
        if previous != partition:
            raise ValueError(f"group {group_id} crosses partitions: {previous}, {partition}")


def canonical_manifest_hash(rows: Sequence[Mapping[str, object]]) -> str:
    validate_manifest(rows)
    canonical = sorted((_canonical_row(row) for row in rows), key=lambda item: item["window_id"])
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _indicator(rows: Sequence[Mapping[str, object]], class_name: str) -> np.ndarray:
    return np.asarray(
        [int(class_name in {str(label) for label in row["labels"]}) for row in rows],
        dtype=np.int8,
    )


def _partition_indices(rows: Sequence[Mapping[str, object]], partition: str) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(rows) if row["partition"] == partition], dtype=np.int64
    )


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = -1.0
    for threshold in np.arange(0.05, 0.951, 0.01):
        predicted = scores >= threshold
        precision, _, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="binary", zero_division=0
        )
        if f1 > best_f1 or (np.isclose(f1, best_f1) and precision > best_precision):
            best_f1 = float(f1)
            best_precision = float(precision)
            best_threshold = float(threshold)
    return round(best_threshold, 4)


def _partition_metrics(
    package: Mapping[str, Any],
    X_scaled: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    indices: np.ndarray,
) -> dict[str, object]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    ap_values: list[float] = []
    for class_name in package["classes"]:
        y_true = _indicator(rows, class_name)[indices]
        scores = package["heads"][class_name].predict_proba(X_scaled[indices])[:, 1]
        predicted = scores >= package["thresholds"][class_name]
        parent = package.get("label_hierarchy", {}).get(class_name)
        if parent is not None:
            parent_scores = package["heads"][parent].predict_proba(X_scaled[indices])[:, 1]
            predicted &= parent_scores >= package["thresholds"][parent]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="binary", zero_division=0
        )
        if len(np.unique(y_true)) == 2:
            average_precision = float(average_precision_score(y_true, scores))
            ap_values.append(average_precision)
        else:
            average_precision = float("nan")
        f1_values.append(float(f1))
        per_class[class_name] = {
            "support": int(y_true.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "average_precision": average_precision,
        }
    return {
        "samples": int(len(indices)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_average_precision": float(np.mean(ap_values)) if ap_values else float("nan"),
        "per_class": per_class,
    }


def fit_candidate(
    X: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    *,
    classes: Sequence[str],
    label_hierarchy: Mapping[str, str] | None = None,
    random_state: int = 42,
    logistic_c: float = 0.1,
) -> dict[str, object]:
    validate_manifest(rows)
    if len(X) != len(rows):
        raise ValueError(f"feature rows ({len(X)}) do not match manifest rows ({len(rows)})")
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("features must be a non-empty two-dimensional array")
    if not np.isfinite(X).all():
        raise ValueError("features contain non-finite values")
    ordered_classes = list(classes)
    if not ordered_classes or len(set(ordered_classes)) != len(ordered_classes):
        raise ValueError("classes must be a non-empty unique sequence")
    hierarchy = dict(label_hierarchy or {})
    for child, parent in hierarchy.items():
        if child not in ordered_classes or parent not in ordered_classes:
            raise ValueError(f"hierarchy edge {child} -> {parent} references an unknown class")
        if child == parent:
            raise ValueError(f"hierarchy edge {child} -> {parent} cannot be self-referential")

    train_indices = _partition_indices(rows, "train")
    validation_indices = _partition_indices(rows, "validation")
    test_indices = _partition_indices(rows, "test")
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError("manifest must contain train, validation, and test rows")

    scaler = StandardScaler()
    X_scaled = np.empty_like(np.asarray(X, dtype=np.float64))
    X_scaled[train_indices] = scaler.fit_transform(X[train_indices])
    X_scaled[validation_indices] = scaler.transform(X[validation_indices])
    X_scaled[test_indices] = scaler.transform(X[test_indices])
    sample_weights = np.asarray([float(row["sample_weight"]) for row in rows])

    heads: dict[str, LogisticRegression] = {}
    thresholds: dict[str, float] = {}
    for class_name in ordered_classes:
        y = _indicator(rows, class_name)
        if len(np.unique(y[train_indices])) != 2:
            raise ValueError(f"class {class_name} needs positive and negative training rows")
        if len(np.unique(y[validation_indices])) != 2:
            raise ValueError(f"class {class_name} needs positive and negative validation rows")
        head = LogisticRegression(
            C=logistic_c,
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_state,
            solver="lbfgs",
        )
        head.fit(
            X_scaled[train_indices],
            y[train_indices],
            sample_weight=sample_weights[train_indices],
        )
        validation_scores = head.predict_proba(X_scaled[validation_indices])[:, 1]
        heads[class_name] = head
        thresholds[class_name] = _best_threshold(y[validation_indices], validation_scores)

    package: dict[str, object] = {
        "schema_version": 1,
        "model_family": "perch2_independent_logistic_heads",
        "classes": ordered_classes,
        "label_hierarchy": hierarchy,
        "thresholds": thresholds,
        "scaler": scaler,
        "heads": heads,
        "feature_dimension": int(X.shape[1]),
        "dataset_hash": canonical_manifest_hash(rows),
        "random_state": random_state,
        "logistic_c": logistic_c,
    }
    package["metrics"] = {
        "validation": _partition_metrics(package, X_scaled, rows, validation_indices),
        "test": _partition_metrics(package, X_scaled, rows, test_indices),
    }
    return package


def predict_candidate(package: Mapping[str, Any], X: np.ndarray) -> list[dict[str, float]]:
    if X.ndim != 2 or X.shape[1] != package["feature_dimension"]:
        raise ValueError(
            f"expected feature shape (*, {package['feature_dimension']}), got {tuple(X.shape)}"
        )
    X_scaled = package["scaler"].transform(X)
    predictions: list[dict[str, float]] = []
    class_scores = {
        class_name: package["heads"][class_name].predict_proba(X_scaled)[:, 1]
        for class_name in package["classes"]
    }
    for index in range(len(X)):
        predictions.append(
            {class_name: float(class_scores[class_name][index]) for class_name in package["classes"]}
        )
    return predictions


def active_labels(package: Mapping[str, Any], scores: Mapping[str, float]) -> list[str]:
    active = {
        class_name
        for class_name in package["classes"]
        if float(scores[class_name]) >= float(package["thresholds"][class_name])
    }
    hierarchy = package.get("label_hierarchy", {})
    return [
        class_name
        for class_name in package["classes"]
        if class_name in active
        and (class_name not in hierarchy or str(hierarchy[class_name]) in active)
    ]
