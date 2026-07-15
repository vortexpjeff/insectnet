from __future__ import annotations

import hashlib
import json
import tomllib
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "5e6ecfc68d78a2cf2e9e9e47da5cb58d696e8de354fd620cfcccc5db9da48702"


def test_installed_package_exposes_model_and_manifest() -> None:
    data = files("insectnet").joinpath("data")
    model = data.joinpath("classifier.joblib")
    manifest = data.joinpath("v0.1.0.manifest.json")
    assert model.is_file()
    assert manifest.is_file()
    assert hashlib.sha256(model.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifact"]["path"] == "classifier.joblib"
    assert payload["artifact"]["sha256"] == EXPECTED_SHA256


def test_setuptools_declares_release_package_data() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["insectnet"]
    assert package_data == ["data/*.joblib", "data/*.json"]
