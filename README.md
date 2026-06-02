# InsectNet

A BirdNET-Pi sidecar that captures and classifies insect sounds in real time.

Watch audio, classify insects, keep tagged clips, discard silence. Runs alongside
BirdNET-Pi without touching its files, services, or configuration.

## Overview

BirdNET-Pi monitors bird song 24/7. InsectNet taps into the same audio stream,
runs it through the same BirdNET TFLite model, and feeds the 6,522-dim logits
into a lightweight sklearn classifier trained for insect acoustics.

The classifier uses a **one-vs-rest LogisticRegression** on frozen BirdNET
logits — no GPU needed, runs on a Pi 4 at near-zero CPU overhead.

### Architecture

```
BirdNET-Pi                              InsectNet Sidecar
┌─────────────────┐                    ┌──────────────────────┐
│  arecord (15s)  │  close_write       │  inotifywait         │
│  → StreamData/  │ ─────────────────→ │  → copy WAV → temp   │
└─────────────────┘                    │  → librosa load      │
                                       │  → BirdNET TFLite    │
┌─────────────────┐                    │  → 6,522-dim logits  │
│  BirdNET-Lite   │                    │  → sklearn head      │
│  (port 5050)    │  (independent)     │  → per-class conf.   │
└─────────────────┘                    │  → keep/discard      │
                                       └──────────────────────┘
                                            │
                                     Keep if non-background
                                            ↓
                                   captures/{class}/{ts}_{cls}_{conf}.wav
                                   detections.jsonl (append)
```

### Classes

| Class | Description | Status |
|-------|-------------|--------|
| `background` | Silence, wind, traffic, birds, human noise | Production |
| `cicada_drone` | Sustained tonal buzzing (Neotibicen, Megatibicen) | Production |
| `cricket_katydid` | Pulsed chirps, trills, rasps | Production |
| `frog` | Amphibian vocalizations | Experimental |
| `grasshopper` | Acrididae stridulation | Data-limited |
| `bee` | Hymenoptera wing buzz | Data-limited |

Per-class confidence thresholds are documented in [docs/thresholds.md](docs/thresholds.md).

## Quick Start

### On a BirdNET-Pi

```bash
# 1. Install
pip install insectnet
# Or from source:
# git clone https://github.com/vortexpjeff/insectnet && cd insectnet
# pip install -e .

# 2. Deploy a trained classifier
scp models/6class.joblib birdnetpi@192.168.1.223:~/insectnet_capture/classifier.joblib

# 3. Run the sidecar
python -m insectnet.capture --threshold 0.3 --show
```

### On a Workstation (pull captures)

```bash
python -m insectnet.capture --pull
```

### Predict a Single Clip

```bash
python -m insectnet.predict field_recording.wav --model models/6class.joblib
```

## Training

InsectNet classifiers are trained on BirdNET's 6,522-dim logit space, not raw
audio. The training workflow:

1. Collect WAVs from BirdNET-Pi StreamData (or playback sessions)
2. Extract logit vectors: `librosa.load()` → TFLite inference → save `.npy`
3. Label clips per-class or multi-label
4. Train:

```bash
python -m insectnet.train \
  --logits-dir ./training_logits \
  --labels labels.json \
  --output models/my_model.joblib \
  --multi-label
```

The logit-based approach means training is fast (seconds on a laptop) and
requires no GPU. See [docs/architecture.md](docs/architecture.md) for details.

## Models

The sidecar uses BirdNET TFLite logit-based classifiers — sklearn heads
trained on BirdNET's 6,522-dim output space. These are deployed as a single
`classifier.joblib` alongside the capture script on the Pi.

To build and deploy a classifier, see [models/README.md](models/README.md).

Perch 2.0 embedding-based classifiers (used by the separate archive system)
are not compatible with the Pi sidecar and live in the
[pine-hollow-archive](https://github.com/vortexpjeff/pine-hollow-archive) repo.

## Validation

InsectNet has been field-validated at Pine Hollow, Tennessee (35.8565, -83.3744):

- **Natural cicada** captured at 83% confidence, confirmed by ear
- **Frog chorus** — 440+ captures in one evening, two species identified
- **Cricket/katydid** — confirmed captures at 99%+ confidence
- **Known false positives** documented: AC unit → cicada_drone, weed whacker → bee

Full validation log: [docs/validation.md](docs/validation.md)

## Why This Works

BirdNET v2.4 has **31 Orthoptera species** in its 6,522-class label set — field
crickets, tree crickets, conehead katydids, ground crickets, and meadow katydids.
Its logit space already encodes insect acoustic structure. The sklearn head is
just reading the signal BirdNET was already picking up but discarding.

**Cicadas are the gap.** There are zero cicada species in BirdNET's labels.
InsectNet's cicada class relies on general acoustic features in the BirdNET
embedding space. Field validation shows this works (confirmed natural captures),
but species-level identification requires cosine similarity against training
centroids rather than BirdNET species channels.

## Repository Structure

```
insectnet/
├── src/insectnet/          # Python package
│   ├── capture.py          #   Real-time sidecar (inotify → classify)
│   ├── birdnet.py          #   TFLite inference wrapper
│   ├── predict.py          #   Single-clip prediction CLI
│   └── train.py            #   Training pipeline
├── models/                 # Trained classifiers + provenance
├── scripts/                # Deployment and session helpers
├── docs/                   # Architecture, validation, thresholds
└── tests/                  # (coming)
```

## License

MIT

## Related

- [Pine Hollow Archive](https://github.com/vortexpjeff/pine-hollow-archive) —
  Broader bioacoustics data factory: pulls captures, runs Perch 2.0 embeddings,
  review app, retrain pipeline.
- [BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi) — The bird monitoring
  platform InsectNet rides on.
