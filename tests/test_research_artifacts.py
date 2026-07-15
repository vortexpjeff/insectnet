from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ARTIFACTS = {
    "chickennet-research-0.1.0-perch2": (
        "a5b83b648b19d2837fe775161cf35fce22f2a717e630c08253f2b9c6d2fe58d0",
        ("inat_challenge_report.json", "private_local_frog_challenge_report.json"),
    ),
    "insectnet-research-0.2.0-perch2": (
        "27bf603a6dec2df2789b3bf9241f5e035ccdea5909c4ecf252623ff9304afe32",
        ("inat_dog_negative_challenge_report.json", "private_local_frog_activation_report.json"),
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def string_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [item for child in value.values() for item in string_values(child)]
    if isinstance(value, (list, tuple, set)):
        return [item for child in value for item in string_values(child)]
    return [value] if isinstance(value, str) else []


def test_research_artifacts_match_public_metadata_and_contain_no_private_paths() -> None:
    for slug, (expected_sha256, challenge_files) in RESEARCH_ARTIFACTS.items():
        directory = ROOT / "models" / slug
        artifact = directory / f"{slug}.joblib"
        summary = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
        package = joblib.load(artifact)

        assert sha256_file(artifact) == expected_sha256
        assert summary["artifact_sha256"] == expected_sha256
        assert summary["dataset_hash"] == package["dataset_hash"]
        assert summary["thresholds"] == package["thresholds"]
        assert summary["label_hierarchy"] == package["label_hierarchy"]
        assert package["feature_dimension"] == 1536

        private_paths = [
            value
            for value in string_values(package)
            if value.startswith(("/home/", "/mnt/")) or "C:\\Users\\" in value
        ]
        assert private_paths == []

        for filename in challenge_files:
            report = json.loads((directory / filename).read_text(encoding="utf-8"))
            assert report["artifact_sha256"] == expected_sha256

        inventory = (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        for line in inventory:
            expected_file_hash, relative_path = line.split("  ", maxsplit=1)
            assert sha256_file(directory / relative_path) == expected_file_hash
