#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from insectnet.candidate import active_labels, predict_candidate  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a frozen Perch candidate on a challenge set")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--mode", choices=("negative", "weak-positive", "activation"), required=True)
    parser.add_argument("--primary-class")
    parser.add_argument("--interpretation", required=True)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    embeddings_path = args.embeddings.resolve()
    windows_path = args.windows.resolve()
    package = joblib.load(artifact)
    embeddings = np.load(embeddings_path, allow_pickle=False)
    rows = load_jsonl(windows_path)
    if embeddings.shape != (len(rows), int(package["feature_dimension"])):
        raise ValueError(f"feature/manifest mismatch: {embeddings.shape}, {len(rows)} rows")
    scores = predict_candidate(package, embeddings)
    raw: dict[str, dict[str, object]] = {}
    gated: dict[str, dict[str, object]] = {}
    for class_name in package["classes"]:
        values = np.asarray([row[class_name] for row in scores], dtype=np.float64)
        crossings = values >= float(package["thresholds"][class_name])
        gated_crossings = np.asarray(
            [class_name in active_labels(package, row) for row in scores], dtype=bool
        )
        quantiles = {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        }
        raw[class_name] = {
            "windows": int(crossings.sum()),
            "rate": float(crossings.mean()),
            "score_quantiles": quantiles,
        }
        gated[class_name] = {
            "windows": int(gated_crossings.sum()),
            "rate": float(gated_crossings.mean()),
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "challenge": args.challenge,
        "mode": args.mode,
        "samples": len(rows),
        "groups": len({str(row["group_id"]) for row in rows}),
        "challenge_windows_sha256": sha256_file(windows_path),
        "embeddings_sha256": sha256_file(embeddings_path),
        "thresholds_locked_before_challenge": package["thresholds"],
        "label_hierarchy": package.get("label_hierarchy", {}),
        "raw_head_crossings": raw,
        "hierarchy_gated_outputs": gated,
        "interpretation": args.interpretation,
    }
    if args.primary_class:
        report["primary_class"] = args.primary_class
        report["primary_gated_rate"] = gated[args.primary_class]["rate"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
