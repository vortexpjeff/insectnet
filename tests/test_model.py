from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from insectnet.model import ModelContractError, load_model, score_logits

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "insectnet" / "data" / "classifier.joblib"
EXPECTED_CLASSES = (
    "background",
    "bee",
    "cicada_drone",
    "cricket_katydid",
    "frog",
    "grasshopper",
)


def test_v01_model_loads_with_expected_shape_and_classes() -> None:
    model = load_model(MODEL)
    assert model.classes == EXPECTED_CLASSES
    assert model.feature_dimension == 6522
    assert len(model.classifier.estimators_) == len(EXPECTED_CLASSES)


def test_zero_logits_score_every_declared_class() -> None:
    model = load_model(MODEL)
    scores = score_logits(np.zeros(6522, dtype=np.float32), model)
    assert tuple(scores) == EXPECTED_CLASSES
    assert all(0.0 <= score <= 1.0 for score in scores.values())


@pytest.mark.parametrize("shape", [(6521,), (6523,), (1, 6522)])
def test_invalid_feature_shape_is_rejected(shape: tuple[int, ...]) -> None:
    model = load_model(MODEL)
    with pytest.raises(ModelContractError):
        score_logits(np.zeros(shape, dtype=np.float32), model)


def test_nonfinite_logits_are_rejected() -> None:
    model = load_model(MODEL)
    logits = np.zeros(6522, dtype=np.float32)
    logits[0] = np.nan
    with pytest.raises(ModelContractError):
        score_logits(logits, model)
