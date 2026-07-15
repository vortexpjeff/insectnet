from __future__ import annotations

import hashlib
import json
from pathlib import Path

from insectnet.release import load_manifest, verify_release

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "insectnet" / "data" / "classifier.joblib"
MANIFEST = ROOT / "src" / "insectnet" / "data" / "v0.1.0.manifest.json"
EXPECTED_SHA256 = "5e6ecfc68d78a2cf2e9e9e47da5cb58d696e8de354fd620cfcccc5db9da48702"
EXPECTED_CLASSES = [
    "background",
    "bee",
    "cicada_drone",
    "cricket_katydid",
    "frog",
    "grasshopper",
]


def test_exact_v01_artifact_is_preserved() -> None:
    assert MODEL.stat().st_size == 474_892
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_v01_is_the_only_canonical_packaged_model_artifact() -> None:
    artifacts = sorted((ROOT / "src" / "insectnet" / "data").glob("*.joblib"))
    assert artifacts == [MODEL]


def test_research_artifacts_are_separately_versioned() -> None:
    artifacts = sorted((ROOT / "models").glob("*/*.joblib"))
    assert [path.relative_to(ROOT).as_posix() for path in artifacts] == [
        "models/chickennet-research-0.1.0-perch2/chickennet-research-0.1.0-perch2.joblib",
        "models/insectnet-research-0.2.0-perch2/insectnet-research-0.2.0-perch2.joblib",
    ]


def test_release_manifest_declares_exact_contract() -> None:
    data = load_manifest(MANIFEST)
    assert data["release_id"] == "insectnet-v0.1.0"
    assert data["status"] == "historical_research_reference"
    assert data["artifact"]["sha256"] == EXPECTED_SHA256
    assert data["artifact"]["bytes"] == 474_892
    assert data["model_contract"]["classes"] == EXPECTED_CLASSES
    assert data["model_contract"]["feature_space"] == "birdnet_v2.4_logits"
    assert data["model_contract"]["feature_dimension"] == 6522
    assert data["model_contract"]["window_seconds"] == 3.0
    assert data["model_contract"]["sample_rate_hz"] == 48000
    assert data["model_contract"]["serialization_runtime"] == {"scikit_learn": "1.8.0"}


def test_release_verification_passes() -> None:
    result = verify_release(MANIFEST, MANIFEST.parent)
    assert result.ok is True
    assert result.errors == ()
    assert result.actual_sha256 == EXPECTED_SHA256


def test_hash_failure_stops_before_deserialization(tmp_path: Path, monkeypatch) -> None:
    bad_model = tmp_path / "classifier.joblib"
    bad_model.write_bytes(MODEL.read_bytes() + b"tampered")
    manifest_data = load_manifest(MANIFEST)
    bad_manifest = tmp_path / "v0.1.0.manifest.json"
    bad_manifest.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    called = False

    def forbidden_load(_path: Path) -> None:
        nonlocal called
        called = True
        raise AssertionError("untrusted joblib must not be deserialized")

    monkeypatch.setattr("insectnet.release.load_model", forbidden_load)
    result = verify_release(bad_manifest, tmp_path)
    assert result.ok is False
    assert "artifact SHA-256 mismatch" in result.errors
    assert "artifact byte-size mismatch" in result.errors
    assert called is False


def test_manifest_is_stable_json() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert MANIFEST.read_text(encoding="utf-8") == rendered
