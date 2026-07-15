#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, list[dict[str, object]]]:
    entries: list[dict[str, object]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": str(file_path.relative_to(path)),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), entries


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def decode_window(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "32000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if len(samples) < 160_000:
        samples = np.pad(samples, (0, 160_000 - len(samples)))
    elif len(samples) > 160_000:
        samples = samples[:160_000]
    if len(samples) != 160_000 or not np.isfinite(samples).all():
        raise ValueError(f"invalid decoded audio for {path}")
    return samples.astype(np.float32, copy=False)


def extract(
    *,
    work_dir: Path,
    windows_manifest: Path,
    output_dir: Path,
    model_dir: Path,
    batch_size: int,
) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    rows = load_jsonl(windows_manifest)
    if not rows:
        raise ValueError("windows manifest is empty")
    window_ids = [str(row["window_id"]) for row in rows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("windows manifest contains duplicate window IDs")

    model_tree_sha256, model_files = sha256_tree(model_dir)
    model = tf.saved_model.load(str(model_dir))
    infer = model.signatures["serving_default"]

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.empty((len(rows), 1536), dtype=np.float32)
    embedding_rows: list[dict[str, object]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        audio = np.stack(
            [decode_window(work_dir / str(row["window_path"])) for row in chunk]
        )
        output = infer(inputs=tf.constant(audio, dtype=tf.float32))
        batch_embeddings = np.asarray(output["embedding"].numpy(), dtype=np.float32)
        if batch_embeddings.shape != (len(chunk), 1536):
            raise ValueError(f"unexpected Perch embedding shape: {batch_embeddings.shape}")
        embeddings[start : start + len(chunk)] = batch_embeddings
        for offset, (row, embedding) in enumerate(zip(chunk, batch_embeddings, strict=True)):
            raw = embedding.astype("<f4", copy=False).tobytes()
            embedding_rows.append(
                {
                    "schema_version": 1,
                    "window_id": row["window_id"],
                    "row_index": start + offset,
                    "model_id": "google-perch-v2-cpu-1",
                    "model_tree_sha256": model_tree_sha256,
                    "extraction_recipe": "perch2-32khz-5s-float32-batch-v1",
                    "dimension": 1536,
                    "dtype": "float32",
                    "embedding_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        print(f"embedded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    np.save(output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    (output_dir / "window_ids.json").write_text(
        json.dumps(window_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "embeddings.jsonl").open("w", encoding="utf-8") as handle:
        for row in embedding_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "schema_version": 1,
        "windows_manifest": str(windows_manifest),
        "windows_manifest_sha256": sha256_file(windows_manifest),
        "model_dir": str(model_dir),
        "model_tree_sha256": model_tree_sha256,
        "model_files": model_files,
        "samples": len(rows),
        "feature_dimension": 1536,
        "embeddings_npy_sha256": sha256_file(output_dir / "embeddings.npy"),
        "window_ids_sha256": sha256_file(output_dir / "window_ids.json"),
        "embeddings_manifest_sha256": sha256_file(output_dir / "embeddings.jsonl"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Perch 2 embeddings")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--windows-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    extract(
        work_dir=args.work_dir.resolve(),
        windows_manifest=args.windows_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        model_dir=args.model_dir.resolve(),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
