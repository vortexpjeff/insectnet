from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from insectnet.cli import main


def test_verify_command_reports_exact_release(capsys) -> None:
    exit_code = main(["verify"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["sha256"].startswith("5e6ecfc68d78")


def test_score_command_outputs_all_classes(tmp_path: Path, capsys) -> None:
    logits_path = tmp_path / "logits.npy"
    np.save(logits_path, np.zeros(6522, dtype=np.float32))
    exit_code = main(["score", str(logits_path)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["feature_dimension"] == 6522
    assert list(payload["scores"]) == [
        "background",
        "bee",
        "cicada_drone",
        "cricket_katydid",
        "frog",
        "grasshopper",
    ]
