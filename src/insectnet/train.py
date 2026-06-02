"""Train InsectNet classifiers on BirdNET logits.

Usage:
    python -m insectnet.train --logits-dir ./training_logits --labels train.json
    python -m insectnet.train --from-archive /mnt/c/.../pine-hollow-archive

This trains a one-vs-rest LogisticRegression head on frozen BirdNET 6,522-dim
logits. The same architecture as the production InsectNet classifiers.

Output: models/{name}.joblib with keys: scaler, classifier, classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder


def load_training_data(logits_dir: str, labels_path: str) -> tuple[np.ndarray, list]:
    """Load logit vectors and labels from disk.

    Labels format: JSON array of objects with 'file' and 'label' keys.
    'label' can be a string (single-label) or list of strings (multi-label).
    """
    logits_dir = Path(logits_dir)

    with open(labels_path) as f:
        entries = json.load(f)

    X_list = []
    y_list = []

    for entry in entries:
        logit_file = logits_dir / entry["file"]
        if not logit_file.exists():
            continue
        vec = np.load(logit_file)
        X_list.append(vec)
        y_list.append(entry["label"])

    X = np.array(X_list)
    return X, y_list


def train_multilabel(X: np.ndarray, y: list, classes: list[str] | None = None) -> dict:
    """Train a multi-label classifier.

    Args:
        X: (n_samples, 6522) logit matrix.
        y: List of lists, e.g. [["cicada_drone"], ["background"], ["cicada_drone", "frog"]].
        classes: Explicit class order. If None, sorted alphabetically.

    Returns:
        Dict with scaler, classifier, classes.
    """
    binarizer = MultiLabelBinarizer(classes=classes) if classes else MultiLabelBinarizer()
    y_bin = binarizer.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, solver="lbfgs")
    clf = OneVsRestClassifier(base)
    clf.fit(X_scaled, y_bin)

    return {
        "scaler": scaler,
        "classifier": clf,
        "classes": binarizer.classes_.tolist(),
    }


def train_multiclass(X: np.ndarray, y: list[str]) -> dict:
    """Train a single-label multi-class classifier.

    For multi-class problems where each clip has exactly one label.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, solver="lbfgs")
    clf.fit(X_scaled, y_enc)

    return {
        "scaler": scaler,
        "classifier": clf,
        "classes": le.classes_.tolist(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train InsectNet classifier on BirdNET logits")
    parser.add_argument("--logits-dir", required=True,
                        help="Directory with .npy logit files")
    parser.add_argument("--labels", required=True,
                        help="JSON file: [{\"file\": \"x.npy\", \"label\": \"class\"}]")
    parser.add_argument("--output", "-o", default="models/insectnet.joblib",
                        help="Output .joblib path")
    parser.add_argument("--classes", nargs="*",
                        help="Explicit class order (for multi-label)")
    parser.add_argument("--multi-label", action="store_true",
                        help="Labels are lists (multi-label mode)")
    parser.add_argument("--cv", action="store_true",
                        help="Run cross-validation and print scores")
    return parser


def main() -> None:
    import joblib

    parser = _build_parser()
    args = parser.parse_args()

    print(f"Loading training data from {args.logits_dir}...")
    X, y_raw = load_training_data(args.logits_dir, args.labels)
    print(f"  Loaded {len(X)} samples, {X.shape[1]} features each")

    if args.multi_label:
        # Labels are list of lists
        y = y_raw
        print(f"  Multi-label mode, classes: {set(c for sublist in y for c in sublist)}")
        model = train_multilabel(X, y, classes=args.classes)
    else:
        # Labels are strings
        y = y_raw
        print(f"  Multi-class mode, classes: {set(y)}")
        model = train_multiclass(X, y)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(output_path))
    print(f"  ✓ Model saved to {output_path}")
    print(f"  Classes: {model['classes']}")

    if args.cv:
        print(f"\n  Running 5-fold cross-validation...")
        y_enc = y if args.multi_label else LabelEncoder().fit_transform(y)
        from sklearn.metrics import classification_report
        # Simple CV accuracy
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cv_clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
        preds = cross_val_predict(cv_clf, X_scaled, y_enc, cv=5)
        print(f"\n  CV Results:")
        print(classification_report(y_enc, preds, target_names=model["classes"]))


if __name__ == "__main__":
    main()
