from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ModelContractError, load_model


@dataclass(frozen=True)
class ReleaseVerification:
    ok: bool
    errors: tuple[str, ...]
    actual_sha256: str | None


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = Path(path)
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"release manifest not found: {manifest}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid release manifest JSON: {manifest}") from exc
    if not isinstance(data, dict):
        raise ValueError("release manifest must contain a JSON object")
    return data


def verify_release(
    manifest_path: str | Path,
    repository_root: str | Path,
) -> ReleaseVerification:
    root = Path(repository_root).resolve()
    errors: list[str] = []
    actual_sha256: str | None = None

    try:
        manifest = load_manifest(manifest_path)
        artifact_spec = manifest["artifact"]
        relative_path = Path(artifact_spec["path"])
    except (KeyError, TypeError, ValueError) as exc:
        return ReleaseVerification(False, (f"manifest contract error: {exc}",), None)

    artifact = (root / relative_path).resolve()
    if root not in artifact.parents:
        return ReleaseVerification(False, ("artifact path escapes repository root",), None)
    if not artifact.is_file():
        return ReleaseVerification(False, (f"artifact not found: {relative_path}",), None)

    payload = artifact.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != artifact_spec.get("sha256"):
        errors.append("artifact SHA-256 mismatch")
    if len(payload) != artifact_spec.get("bytes"):
        errors.append("artifact byte-size mismatch")
    if errors:
        return ReleaseVerification(False, tuple(errors), actual_sha256)

    try:
        model = load_model(artifact)
        contract = manifest["model_contract"]
        if list(model.classes) != contract.get("classes"):
            errors.append("artifact class order differs from manifest")
        if model.feature_dimension != contract.get("feature_dimension"):
            errors.append("artifact feature dimension differs from manifest")
    except (ModelContractError, KeyError, TypeError) as exc:
        errors.append(f"model contract error: {exc}")

    return ReleaseVerification(not errors, tuple(errors), actual_sha256)
