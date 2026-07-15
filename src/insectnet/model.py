from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelContractError(ValueError):
    """Raised when an artifact or feature vector violates the v0.1 contract."""


@dataclass(frozen=True)
class ModelBundle:
    scaler: Any
    classifier: Any
    classes: tuple[str, ...]
    feature_dimension: int


def load_model(path: str | Path) -> ModelBundle:
    """Load and validate a trusted InsectNet joblib artifact.

    Joblib uses Python pickle internally. Only load an artifact whose SHA-256
    matches the release manifest.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise ModelContractError(f"model artifact not found: {artifact}")

    payload = joblib.load(artifact)
    if not isinstance(payload, dict):
        raise ModelContractError("model artifact must contain a dictionary")

    required = {"scaler", "classifier", "classes"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ModelContractError(f"model artifact missing keys: {missing}")

    classes = tuple(payload["classes"])
    if not classes or any(not isinstance(name, str) or not name for name in classes):
        raise ModelContractError("classes must be a non-empty sequence of names")
    if len(classes) != len(set(classes)):
        raise ModelContractError("classes must be unique and ordered")

    scaler = payload["scaler"]
    classifier = payload["classifier"]
    feature_dimension = int(getattr(scaler, "n_features_in_", 0))
    if feature_dimension != 6522:
        raise ModelContractError(
            f"expected 6522 BirdNET logits, artifact declares {feature_dimension}"
        )

    estimators = getattr(classifier, "estimators_", None)
    if estimators is None or len(estimators) != len(classes):
        raise ModelContractError("classifier estimator count does not match class order")

    return ModelBundle(
        scaler=scaler,
        classifier=classifier,
        classes=classes,
        feature_dimension=feature_dimension,
    )


def score_logits(logits: np.ndarray, model: ModelBundle) -> dict[str, float]:
    """Score one finite 6,522-dimensional BirdNET v2.4 logit vector."""
    vector = np.asarray(logits, dtype=np.float64)
    if vector.shape != (model.feature_dimension,):
        raise ModelContractError(
            f"expected feature shape ({model.feature_dimension},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ModelContractError("feature vector contains NaN or infinity")

    transformed = model.scaler.transform(vector.reshape(1, -1))
    probabilities = np.asarray(model.classifier.predict_proba(transformed))[0]
    if probabilities.shape != (len(model.classes),):
        raise ModelContractError("classifier output does not match declared class order")
    if not np.isfinite(probabilities).all():
        raise ModelContractError("classifier output contains NaN or infinity")

    return {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(model.classes)
    }
