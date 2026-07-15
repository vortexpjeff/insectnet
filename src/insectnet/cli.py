from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Sequence

import numpy as np

from .model import ModelContractError, load_model, score_logits
from .release import verify_release


def packaged_data_dir() -> Path:
    return Path(str(files("insectnet").joinpath("data")))


def build_parser() -> argparse.ArgumentParser:
    data_dir = packaged_data_dir()
    parser = argparse.ArgumentParser(
        prog="insectnet",
        description="Verify or score the preserved InsectNet v0.1 research artifact.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="Verify artifact bytes and model contract")
    verify.add_argument(
        "--manifest", type=Path, default=data_dir / "v0.1.0.manifest.json"
    )
    verify.add_argument("--root", type=Path, default=data_dir)

    score = commands.add_parser("score", help="Score one precomputed BirdNET logit vector")
    score.add_argument("logits", type=Path, help="NumPy .npy file with shape (6522,)")
    score.add_argument("--model", type=Path, default=data_dir / "classifier.joblib")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        result = verify_release(args.manifest, args.root)
        print(
            json.dumps(
                {
                    "errors": list(result.errors),
                    "ok": result.ok,
                    "sha256": result.actual_sha256,
                },
                indent=2,
            )
        )
        return 0 if result.ok else 1

    try:
        logits = np.load(args.logits, allow_pickle=False)
        model = load_model(args.model)
        scores = score_logits(logits, model)
    except (FileNotFoundError, OSError, ValueError, ModelContractError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1

    print(
        json.dumps(
            {
                "feature_dimension": model.feature_dimension,
                "scores": scores,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
