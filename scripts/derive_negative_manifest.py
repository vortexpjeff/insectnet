#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_partition(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive a target-absent manifest from an audited corpus")
    parser.add_argument("--source-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--repartition-by-group", action="store_true")
    args = parser.parse_args()
    source = args.source_windows.resolve()
    output_dir = args.output_dir.resolve()
    rows = load_jsonl(source)
    derived_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "windows.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            derived = dict(row)
            derived["source_labels"] = list(row["labels"])
            derived["labels"] = []
            if args.repartition_by_group:
                derived["partition"] = stable_partition(str(row["group_id"]))
            derived["derived_role"] = args.role
            derived["negative_label_reason"] = args.reason
            derived["parent_windows_manifest_sha256"] = sha256_file(source)
            derived_rows.append(derived)
            handle.write(json.dumps(derived, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "role": args.role,
        "reason": args.reason,
        "windows": len(rows),
        "partitions": dict(Counter(str(row["partition"]) for row in derived_rows)),
        "rights_lanes": dict(Counter(str(row["rights_lane"]) for row in derived_rows)),
        "source_windows_manifest": source.name,
        "source_windows_manifest_sha256": sha256_file(source),
        "windows_manifest_sha256": sha256_file(output_path),
        "partition_policy": (
            "deterministic SHA-256 by existing group_id: 70/15/15"
            if args.repartition_by_group
            else "preserve source partitions"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
