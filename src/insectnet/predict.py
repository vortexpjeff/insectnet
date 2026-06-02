"""Classify a single audio clip through InsectNet.

Usage:
    python -m insectnet.predict clip.wav --model models/6class.joblib
    python -m insectnet.predict clip.wav --model models/6class.joblib --show-logits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import birdnet


def load_classifier(joblib_path: str) -> dict:
    import joblib

    m = joblib.load(joblib_path)
    return {
        "scaler": m["scaler"],
        "classifier": m["classifier"],
        "classes": m["classes"],
    }


def classify_logits(logits, clf_data: dict) -> dict[str, float]:
    X = clf_data["scaler"].transform(logits.reshape(1, -1))
    probs = clf_data["classifier"].predict_proba(X)[0]
    return {cls: float(probs[i]) for i, cls in enumerate(clf_data["classes"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single audio clip through InsectNet")
    parser.add_argument("audio", help="Path to WAV/MP3/M4A audio file")
    parser.add_argument("--model", "-m", default="models/6class.joblib",
                        help="Classifier .joblib path")
    parser.add_argument("--birdnet-model", default=None,
                        help="BirdNET TFLite model (uses default if not specified)")
    parser.add_argument("--show-logits", action="store_true",
                        help="Save extracted logits as JSON")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (machine-readable)")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"ERROR: classifier not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Load classifier
    clf = load_classifier(str(args.model))

    # Load BirdNET
    bnet_path = args.birdnet_model or "/home/birdnetpi/BirdNET-Pi/model/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
    if not Path(bnet_path).exists():
        print(f"ERROR: BirdNET model not found: {bnet_path}", file=sys.stderr)
        sys.exit(1)

    interp = birdnet.load_tflite(bnet_path)

    # Process
    audio = birdnet.load_audio(str(args.audio))
    logits = birdnet.extract_logits(audio, interp)
    scores = classify_logits(logits, clf)

    # Output
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])

    if args.json:
        result = {
            "file": args.audio,
            "predicted_class": sorted_scores[0][0],
            "confidence": sorted_scores[0][1],
            "all_scores": scores,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"{'='*50}")
        print(f"  InsectNet — Single Clip Prediction")
        print(f"{'='*50}")
        print(f"  File:      {args.audio}")
        print(f"  Classes:   {len(clf['classes'])}")
        for cls, conf in sorted_scores:
            marker = "←" if cls == sorted_scores[0][0] else ""
            print(f"    {cls:20s}  {conf*100:5.1f}%  {marker}")
        print(f"{'='*50}")

    if args.show_logits:
        logits_path = Path(args.audio).with_name(Path(args.audio).stem + "_logits.json")
        logits_list = logits.tolist()
        with open(logits_path, "w") as f:
            json.dump({"logits": logits_list, "shape": list(logits.shape)}, f)
        print(f"  Logits saved to {logits_path}")


if __name__ == "__main__":
    main()
