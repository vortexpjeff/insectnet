#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def subset(
    *,
    target_windows: Path,
    source_windows: Path,
    source_embeddings: Path,
    source_ids: Path,
    source_metadata: Path,
    output_dir: Path,
) -> None:
    rows = load_jsonl(target_windows)
    target_ids = [str(row["window_id"]) for row in rows]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("target windows contain duplicate IDs")
    all_ids = json.loads(source_ids.read_text(encoding="utf-8"))
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("source embedding IDs contain duplicates")
    embeddings = np.load(source_embeddings, allow_pickle=False)
    if embeddings.shape != (len(all_ids), 1536):
        raise ValueError(f"unexpected source embedding shape: {embeddings.shape}")
    source_rows = load_jsonl(source_windows)
    source_window_ids = [str(row["window_id"]) for row in source_rows]
    if source_window_ids != all_ids:
        raise ValueError("source windows do not match source embedding ID order")
    source_recording_ids = [str(row["recording_id"]) for row in source_rows]
    if len(source_recording_ids) != len(set(source_recording_ids)):
        raise ValueError("source windows contain duplicate recording IDs")
    index = {
        recording_id: row_index
        for row_index, recording_id in enumerate(source_recording_ids)
    }
    target_recording_ids = [str(row["recording_id"]) for row in rows]
    missing = [recording_id for recording_id in target_recording_ids if recording_id not in index]
    if missing:
        raise ValueError(f"target recording IDs missing from source embeddings: {missing[:5]}")
    selected = np.asarray(
        [embeddings[index[recording_id]] for recording_id in target_recording_ids],
        dtype=np.float32,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "window_ids.json"
    manifest_path = output_dir / "embeddings.jsonl"
    np.save(embeddings_path, selected, allow_pickle=False)
    ids_path.write_text(json.dumps(target_ids, indent=2) + "\n", encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row_index, (window_id, recording_id, embedding) in enumerate(
            zip(target_ids, target_recording_ids, selected, strict=True)
        ):
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "window_id": window_id,
                        "row_index": row_index,
                        "dimension": 1536,
                        "dtype": "float32",
                        "embedding_sha256": hashlib.sha256(
                            embedding.astype("<f4", copy=False).tobytes()
                        ).hexdigest(),
                        "parent_recording_id": recording_id,
                        "parent_row_index": index[recording_id],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    parent_metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1,
        "operation": "deterministic_embedding_subset",
        "samples": len(target_ids),
        "feature_dimension": 1536,
        "target_windows": str(target_windows),
        "target_windows_sha256": sha256_file(target_windows),
        "source_embeddings": str(source_embeddings),
        "source_embeddings_sha256": sha256_file(source_embeddings),
        "source_ids": str(source_ids),
        "source_ids_sha256": sha256_file(source_ids),
        "source_windows": str(source_windows),
        "source_windows_sha256": sha256_file(source_windows),
        "source_metadata": str(source_metadata),
        "source_metadata_sha256": sha256_file(source_metadata),
        "parent_model_tree_sha256": parent_metadata["model_tree_sha256"],
        "embeddings_npy_sha256": sha256_file(embeddings_path),
        "window_ids_sha256": sha256_file(ids_path),
        "embeddings_manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an exact subset of Perch embeddings")
    parser.add_argument("--target-windows", type=Path, required=True)
    parser.add_argument("--source-embeddings", type=Path, required=True)
    parser.add_argument("--source-windows", type=Path, required=True)
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    subset(
        target_windows=args.target_windows.resolve(),
        source_windows=args.source_windows.resolve(),
        source_embeddings=args.source_embeddings.resolve(),
        source_ids=args.source_ids.resolve(),
        source_metadata=args.source_metadata.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
