---
license: cc-by-nc-sa-4.0
library_name: scikit-learn
tags:
  - audio-classification
  - bioacoustics
  - perch
  - chicken
  - research
---

# ChickenNet Research 0.1.0 — Perch 2

A research-only three-head chicken-vocalization classifier fitted on frozen 1,536-dimensional Google Perch 2 embeddings.

**Status:** research candidate, not deployed.  
**Artifact SHA-256:** `a5b83b648b19d2837fe775161cf35fce22f2a717e630c08253f2b9c6d2fe58d0`  
**Dataset hash:** `974df55df9a3262944e32563e1111cf6f32cf52a128548a39aab5d69852bc3b0`

## Outputs

The fitted heads are independent. They are not a softmax and do not sum to one. The
serialized inference contract gates both subtype heads on
`chicken_vocalization_present`, so crow or other-vocalization labels are suppressed
when the broad head does not pass.

| Head | Locked threshold |
|---|---:|
| `chicken_vocalization_present` | 0.21 |
| `chicken_crow` | 0.20 |
| `chicken_other_vocalization` | 0.34 |

A window may activate the broad presence head plus one subtype head. Low scores across every head should be treated as abstention/background.

## Feature contract

- Backbone: Google Perch 2 CPU release
- Input to Perch: five seconds, 32 kHz, mono, float32
- Input to this artifact: one 1,536-dimensional Perch embedding
- Verified Perch model-tree SHA-256: `3fb2d54b3e34534f1130052b25737e54bbb5ebfd340ec040d4510772b64c81ff`
- The Perch weights are **not** included in this repository.

The official Perch code repository is Apache-2.0. The Kaggle model-weight license was not exposed by the metadata available during this build, so this card does not claim that the weights themselves are Apache-2.0.

## Training data

| Source | Windows | Role | Rights |
|---|---:|---|---|
| Ross-308 | 317 | chicken vocalization contexts and animal-free ambient recordings | CC BY 4.0 |
| ESC-50 | 2,000 | rooster, hen, and broad environmental negatives | CC BY-NC 3.0 / noncommercial research lane |

Total: 2,317 five-second windows.

Ross-308 was split by individual bird. Its health and lighting fields were retained only as provenance, never used as targets. Clean annotations were merged into non-overlapping five-second contexts from the parent recordings. Noisy/clipped/contact-artifact annotations were excluded and logged.

ESC-50 was split by original Freesound `src_file`, not its published fold alone. The audit found original source IDs crossing published folds; grouping by `src_file` prevents that leakage.

No source audio is redistributed here.

## Evaluation

### Internal grouped test

| Head | Positive support | AP | F1 |
|---|---:|---:|---:|
| chicken vocalization | 62 | 1.000 | 1.000 |
| crow | 4 | 1.000 | 1.000 |
| other vocalization | 58 | 1.000 | 1.000 |

These values are an internal fit check, not field-readiness evidence. Crow support is especially small, and source-domain shortcuts can remain despite grouped splits.

### Locked external iNaturalist challenge

The thresholds were frozen before this challenge. It contains 42 permissively licensed Domestic Chicken sounds from 30 observers: 31 CC BY and 11 CC0.

- Broad-head passes: 33/42
- Broad-head weak-positive hit rate: **78.6%**
- Median broad-head score: **0.858**

This is a taxon-level weak-positive challenge. A centered crop may not contain an audible chicken call. It has no negative examples, so precision and false-positive rate cannot be measured. Crow versus other-call metrics are not reported because call type is not annotated.

### Private stationary-microphone confound audit

A private reviewed frog archive supplied 1,308 local fixed-microphone windows. Chicken absence was not separately reviewed, so the following is an upper bound on candidate false positives, not a confirmed false-positive rate:

- broad-head candidate positives: 10/1,308 windows across 8/436 files;
- crow-head candidate positives: 9/1,308 windows across 7/436 files;
- other-vocalization candidate positives: 0/1,308 windows.

No private paths or audio are included in the public artifact.

## Use

```python
import joblib

package = joblib.load("chickennet-research-0.1.0-perch2.joblib")
embedding = ...  # shape (N, 1536), produced by the verified Perch 2 feature contract
scaled = package["scaler"].transform(embedding)

scores = {
    name: package["heads"][name].predict_proba(scaled)[:, 1]
    for name in package["classes"]
}
detections = {
    name: scores[name] >= package["thresholds"][name]
    for name in package["classes"]
}
```

Do not feed waveforms directly into this artifact.

## Limitations

- Not calibrated for local field deployment.
- No independently reviewed chicken-negative field set yet.
- Crow has only four positive test examples.
- Domestic Chicken taxonomy does not provide call-type labels.
- Ross-308 is a small controlled broiler corpus.
- ESC-50 introduces a noncommercial restriction and coarse source domains.
- The private local candidate positives still require listening review.
- Thresholds should not be changed on the published external challenge.

## Sources

- Ross-308: Díaz de Cerio et al., DOI `10.34810/DATA3437`
- ESC-50: Piczak, DOI `10.1145/2733373.2806390`
- Perch: Ghani et al., *Global birdsong embeddings enable superior transfer learning for bioacoustic classification*, Scientific Reports (2023)

Exact source summaries, manifest hashes, run metrics, and the public challenge report are included alongside the artifact.
