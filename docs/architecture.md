# Architecture

How InsectNet integrates with BirdNET-Pi and why it's designed this way.

## BirdNET-Pi Model

BirdNET-Pi uses a **socket-based client-server architecture** for audio analysis:

```
arecord (15s WAV → StreamData/)
  └→ birdnet_analysis.sh (shell loop)
       └→ analyze.py (socket client on port 5050)
            └→ BirdNET-Lite server loads WAV, runs TFLite, returns CSV
                 └→ detection: WAV → Extracted/By_Date/{species}/
                 └→ no detection: WAV deleted
```

Key design patterns InsectNet mirrors:
- **Binary WAV lifecycle** — every WAV is processed once. Keep or delete, no middle state.
- **Detection-only persistence** — non-detections produce zero artifacts.
- **Shell-based orchestration** — each service is an independent systemd unit.

## InsectNet's Role

InsectNet is a **read-only sidecar**. It never touches BirdNET-Pi's files — it
reads StreamData/ via inotify and copies WAVs to its own directory before
BirdNET-Pi deletes them.

```
StreamData/ (new WAV)
  │
  ├──→ BirdNET-Lite (port 5050) → CSV → keep/delete
  │
  └──→ InsectNet inotify → copy WAV → librosa → TFLite → logits → sklearn → keep/delete
                                                                              │
                                                          captures/{class}/{ts}_{cls}_{conf}.wav
                                                          detections.jsonl (append)
```

## Why BirdNET Logits

InsectNet classifiers train on BirdNET's **6,522-dim logit space**, not raw
audio. This is possible because BirdNET v2.4 has 31 Orthoptera species in its
label set — field crickets, tree crickets, conehead katydids, ground crickets,
and meadow katydids. The logit space already encodes insect acoustic structure.

Cicadas are absent from BirdNET's labels, but their acoustic features still
produce distinguishable patterns in the logit space (confirmed by field
validation with cosine similarity against training centroids).

## Classifier Architecture

All production InsectNet classifiers use:

```
StandardScaler → OneVsRest(LogisticRegression(C=0.1, class_weight='balanced'))
```

- **StandardScaler** normalizes the 6,522-dim logit vectors
- **OneVsRest** trains one binary classifier per class (sigmoid output)
- **LogisticRegression** with L2 regularization (C=0.1), balanced class weights

This is the same architecture BirdNET uses internally without the softmax —
sigmoid-per-class allows multi-label predictions (one clip can be both
"cicada_drone" and "frog").

## Multi-Label Training

Training data format: clips are labeled with lists of active classes, not a
single category. A clip containing overlapping frog and cricket calls is
labeled `["frog", "cricket_katydid"]`.

`MultiLabelBinarizer` converts to an indicator matrix. Per-class
F1-optimized thresholds are swept 0.1-0.95 during evaluation. Each class gets
its own decision threshold.

## Background Training Data

Background clips come from two sources:
1. **BirdNET bird clips** — every labeled bird clip is confirmed non-insect
   audio from the same microphone and environment.
2. **Public datasets** (ESC-50 for environmental noise, iNatSounds for
   labeled insect audio).

## Two-Tier System

InsectNet operates at two levels:

| Layer | Runs On | Backbone | Purpose |
|-------|---------|----------|---------|
| **Sidecar** | BirdNET-Pi (Pi 4) | BirdNET TFLite logits | Real-time capture, keeps WAVs |
| **Archive** | Workstation | Perch 2.0 embeddings | Offline enrichment, multi-taxa discovery |

The sidecar is the edge capture system. The archive (separate repo) is the
analysis layer that pulls captures, embeds them with Perch 2.0, and enables
multi-taxa classification. They are complementary.

## BirdNET Species Coverage

BirdNET v2.4 has 6,522 species labels. Insect-relevant coverage:

| Group | In BirdNET? | Notes |
|-------|-------------|-------|
| 31 Orthoptera (crickets, katydids) | ✅ | Field crickets, tree crickets, coneheads, ground crickets, meadow katydids |
| 0 Cicada species | ❌ | Zero cicada labels — relies on general acoustic features |
| 0 Bee species | ❌ | Zero Hymenoptera labels |
| 0 Grasshopper species | ❌ | Though some Acrididae may trigger Orthoptera channels |

This means 31 logit channels carry insect-class information directly; the
other 6,491 channels may carry incidental insect structure.

## BirdNET-Pi Access

Default credentials:
- **Host:** 192.168.1.223
- **User:** birdnetpi / birdnetpi
- **Python:** `/home/birdnetpi/BirdNET-Pi/birdnet/bin/python3`
- **Model:** `/home/birdnetpi/BirdNET-Pi/model/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite`
- **StreamData:** `/home/birdnetpi/BirdSongs/StreamData/`

The sidecar expects the TFLite model at `DEFAULT_BIRDNET_MODEL` and StreamData
at `DEFAULT_STREAMDATA` (both configurable via CLI args).
