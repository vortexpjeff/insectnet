# Models

Trained classifiers for the InsectNet sidecar. The sidecar expects a
`classifier.joblib` alongside it at `~/insectnet_capture/classifier.joblib`.

## classifier.joblib

The production BirdNET-logit classifier deployed on the Pi. Field-validated
at Pine Hollow for cicada_drone, frog, and cricket_katydid classes.

| Field | Value |
|-------|-------|
| Backbone | BirdNET TFLite logits (6,522-dim) |
| Architecture | StandardScaler → OneVsRest(LogisticRegression(C=0.1, balanced)) |
| Classes | 6: background, bee, cicada_drone, cricket_katydid, frog, grasshopper |
| Training data | InsectSet459 + iNatSounds + ESC-50 + field negatives (~3,636 clips) |
| Field validated | Cicada (83%), frog (51-99%), cricket (99%+) at Pine Hollow |
| Known gaps | Bee (43 clips, no real field captures), grasshopper (183 clips) |
| Deployed | May 29, 2026 on BirdNET-Pi (192.168.1.223) |

## Model Format

Each model is a joblib dictionary with keys:

| Key | Type | Description |
|-----|------|-------------|
| `scaler` | StandardScaler | Fitted on BirdNET 6,522-dim logits |
| `classifier` | OneVsRestClassifier | LogisticRegression(C=0.1, balanced) per class |
| `classes` | list[str] | 6 class names in alphabetical order |

## Training

See `src/insectnet/train.py` for the training pipeline. To train a new
classifier from field captures:

```bash
python -m insectnet.train --logits-dir ./logits --labels labels.json --output models/v0.2.0.joblib
```

To deploy a new model to the Pi:

```bash
scp models/v0.2.0.joblib birdnetpi@192.168.1.223:~/insectnet_capture/classifier.joblib
```

## Perch-Backbone Models

The separate [pine-hollow-archive](https://github.com/vortexpjeff/pine-hollow-archive)
project produces Perch 2.0 embedding-based classifiers (1,536-dim) for offline
multi-taxa analysis. These are not compatible with the Pi sidecar and live in
the archive's `models/` directory.
