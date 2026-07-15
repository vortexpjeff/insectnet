#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from insectnet.candidate import canonical_manifest_hash, fit_candidate, validate_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def git_state(repo_root: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=repo_root, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best = (float("-inf"), float("-inf"), 0.0)
    low = float(np.min(scores))
    high = float(np.max(scores))
    thresholds = np.linspace(low, high, 181) if high > low else np.asarray([low])
    for threshold in thresholds:
        predicted = scores >= threshold
        precision, _, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="binary", zero_division=0
        )
        candidate = (float(f1), float(precision), float(threshold))
        if candidate[:2] > best[:2]:
            best = candidate
    return best[2]


def svm_audit(
    X: np.ndarray,
    rows: list[dict[str, Any]],
    classes: list[str],
    random_state: int,
) -> dict[str, object]:
    partition = np.asarray([row["partition"] for row in rows])
    train = np.flatnonzero(partition == "train")
    validation = np.flatnonzero(partition == "validation")
    test = np.flatnonzero(partition == "test")
    scaler = StandardScaler().fit(X[train])
    transformed = scaler.transform(X)
    sample_weights = np.asarray([float(row["sample_weight"]) for row in rows])
    per_class: dict[str, object] = {}
    test_f1: list[float] = []
    test_ap: list[float] = []
    for class_name in classes:
        labels = np.asarray([int(class_name in row["labels"]) for row in rows])
        model = LinearSVC(
            C=0.1,
            class_weight="balanced",
            dual="auto",
            max_iter=5_000,
            random_state=random_state,
        )
        model.fit(
            transformed[train], labels[train], sample_weight=sample_weights[train]
        )
        threshold = best_threshold(labels[validation], model.decision_function(transformed[validation]))
        scores = model.decision_function(transformed[test])
        predicted = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[test], predicted, average="binary", zero_division=0
        )
        average_precision = (
            float(average_precision_score(labels[test], scores))
            if len(np.unique(labels[test])) == 2
            else float("nan")
        )
        test_f1.append(float(f1))
        if np.isfinite(average_precision):
            test_ap.append(average_precision)
        per_class[class_name] = {
            "threshold": threshold,
            "test_support": int(labels[test].sum()),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_average_precision": average_precision,
        }
    return {
        "model_family": "linear_svc_audit_only",
        "score_semantics": "uncalibrated decision score",
        "test_macro_f1": float(np.mean(test_f1)),
        "test_macro_average_precision": float(np.mean(test_ap)) if test_ap else float("nan"),
        "per_class": per_class,
    }


def train(
    *,
    model_name: str,
    version: str,
    classes: list[str],
    label_hierarchy: dict[str, str],
    datasets: list[tuple[str, Path, Path, Path]],
    output_dir: Path,
    repo_root: Path,
    random_state: int,
) -> None:
    all_rows: list[dict[str, Any]] = []
    feature_blocks: list[np.ndarray] = []
    dataset_records: list[dict[str, object]] = []
    for dataset_name, windows_path, embeddings_path, ids_path in datasets:
        rows = load_jsonl(windows_path)
        features = np.load(embeddings_path, allow_pickle=False)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        row_ids = [row["window_id"] for row in rows]
        if row_ids != ids:
            raise ValueError(f"{dataset_name}: window manifest order does not match embedding IDs")
        if features.shape != (len(rows), 1536):
            raise ValueError(f"{dataset_name}: unexpected feature shape {features.shape}")
        for row in rows:
            row["dataset_name"] = dataset_name
        all_rows.extend(rows)
        feature_blocks.append(np.asarray(features, dtype=np.float32))
        dataset_records.append(
            {
                "name": dataset_name,
                "windows_manifest": str(windows_path),
                "windows_manifest_sha256": sha256_file(windows_path),
                "embeddings": str(embeddings_path),
                "embeddings_sha256": sha256_file(embeddings_path),
                "window_ids": str(ids_path),
                "window_ids_sha256": sha256_file(ids_path),
                "samples": len(rows),
            }
        )

    validate_manifest(all_rows)
    features = np.concatenate(feature_blocks, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    package = fit_candidate(
        features,
        all_rows,
        classes=classes,
        label_hierarchy=label_hierarchy,
        random_state=random_state,
    )
    created_at = subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip()
    portable_dataset_records = [
        {
            "name": row["name"],
            "samples": row["samples"],
            "windows_manifest_sha256": row["windows_manifest_sha256"],
            "embeddings_sha256": row["embeddings_sha256"],
            "window_ids_sha256": row["window_ids_sha256"],
        }
        for row in dataset_records
    ]
    package.update(
        {
            "model_name": model_name,
            "version": version,
            "created_at": created_at,
            "feature_contract": {
                "backbone": "Google Perch 2 CPU",
                "input": "5 seconds, 32 kHz, mono, float32",
                "dimension": 1536,
            },
            "datasets": portable_dataset_records,
            "rights_lanes": sorted({str(row["rights_lane"]) for row in all_rows}),
            "partition_policy": "preassigned immutable source groups",
        }
    )

    model_path = output_dir / f"{model_name}-{version}.joblib"
    joblib.dump(package, model_path, compress=3)
    manifest_path = output_dir / "training_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in sorted(all_rows, key=lambda item: item["window_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    svm = svm_audit(features, all_rows, classes, random_state)
    label_counts = Counter(label for row in all_rows for label in row["labels"])
    partition_counts = Counter(str(row["partition"]) for row in all_rows)
    source_counts = Counter(str(row["dataset_name"]) for row in all_rows)
    rights_counts = Counter(str(row["rights_lane"]) for row in all_rows)
    report = {
        "schema_version": 1,
        "model_name": model_name,
        "version": version,
        "created_at": created_at,
        "artifact": model_path.name,
        "artifact_bytes": model_path.stat().st_size,
        "artifact_sha256": sha256_file(model_path),
        "dataset_hash": canonical_manifest_hash(all_rows),
        "training_manifest": manifest_path.name,
        "training_manifest_sha256": sha256_file(manifest_path),
        "samples": len(all_rows),
        "classes": classes,
        "label_hierarchy": package["label_hierarchy"],
        "thresholds": package["thresholds"],
        "metrics": package["metrics"],
        "linear_svm_audit": svm,
        "counts": {
            "partition": dict(partition_counts),
            "label": dict(label_counts),
            "source": dict(source_counts),
            "rights_lane": dict(rights_counts),
        },
        "datasets": dataset_records,
        "git": git_state(repo_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "random_state": random_state,
        "deployment_status": "research_candidate_not_deployed",
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for _, windows_path, _, _ in datasets:
        source_dir = windows_path.parent
        for filename in ("source.json", "summary.json", "EXCLUSIONS.jsonl"):
            source_file = source_dir / filename
            if source_file.exists():
                destination = output_dir / f"{source_dir.name}-{filename}"
                shutil.copy2(source_file, destination)
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_dataset(value: list[str]) -> tuple[str, Path, Path, Path]:
    name, windows, embeddings, ids = value
    return name, Path(windows).resolve(), Path(embeddings).resolve(), Path(ids).resolve()


def parse_label_hierarchy(values: list[str] | None) -> dict[str, str]:
    hierarchy: dict[str, str] = {}
    for value in values or []:
        child, separator, parent = value.partition("=")
        if not separator or not child or not parent:
            raise ValueError(f"invalid label hierarchy edge: {value!r}; expected CHILD=PARENT")
        hierarchy[child] = parent
    return hierarchy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a provenance-locked Perch candidate")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--class", dest="classes", action="append", required=True)
    parser.add_argument("--label-parent", action="append")
    parser.add_argument(
        "--dataset",
        nargs=4,
        action="append",
        metavar=("NAME", "WINDOWS_JSONL", "EMBEDDINGS_NPY", "WINDOW_IDS_JSON"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(
        model_name=args.model_name,
        version=args.version,
        classes=args.classes,
        label_hierarchy=parse_label_hierarchy(args.label_parent),
        datasets=[parse_dataset(item) for item in args.dataset],
        output_dir=args.output_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
