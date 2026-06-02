# Models

The InsectNet sidecar expects a trained classifier at `classifier.joblib`
alongside the capture script on the BirdNET-Pi. This repo does not ship
pre-trained models — they are built from field data and deployed separately.

## Model Format

The classifier is a joblib dictionary with keys:

| Key | Type | Description |
|-----|------|-------------|
| `scaler` | StandardScaler | Fitted on BirdNET 6,522-dim logits |
| `classifier` | OneVsRestClassifier | LogisticRegression(C=0.1, balanced) |
| `classes` | list[str] | Class names in order |

Training data is BirdNET TFLite logits (6,522-dim), not raw audio or Perch
embeddings.

## BirdNET-Backbone Models

The sidecar uses a two-stage pipeline: BirdNET TFLite → logits → sklearn head.
These models operate on BirdNET's logit space, same as the real-time inference
on the Pi.

To build a classifier: collect logits from the Pi, label captures, and train
via `python -m insectnet.train`.

To deploy: `scp models/your_model.joblib birdnetpi@192.168.1.223:~/insectnet_capture/classifier.joblib`

## Perch-Backbone Models

The separate [pine-hollow-archive](https://github.com/vortexpjeff/pine-hollow-archive)
project produces Perch 2.0 embedding-based classifiers (1,536-dim) for
offline multi-taxa analysis. These are not compatible with the Pi sidecar
and live in the archive's `models/` directory.
