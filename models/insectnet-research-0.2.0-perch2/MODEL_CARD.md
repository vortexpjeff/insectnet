---
license: cc-by-nc-sa-4.0
library_name: scikit-learn
tags:
  - audio-classification
  - bioacoustics
  - perch
  - insects
  - research
---

# InsectNet Research 0.2.0 — Perch 2

A reproducible research classifier for broad insect presence, cicadas, and Orthoptera,
fitted on frozen 1,536-dimensional Google Perch 2 embeddings.

> **Status: trained research artifact, not field-ready and not deployed.** The locked
> independent dog challenge produced broad insect activations in 11/26 windows. Do not
> use this model for unattended ecological counts or alerts.

**Artifact SHA-256:** `27bf603a6dec2df2789b3bf9241f5e035ccdea5909c4ecf252623ff9304afe32`  
**Dataset hash:** `d59cde46a933b85c1cd46f944c6df5c9c6a7587efe7c354b5cbe22b2d0698240`

## Outputs

| Head | Locked threshold | Parent |
|---|---:|---|
| `insect_present` | 0.25 | — |
| `cicada` | 0.08 | `insect_present` |
| `orthoptera` | 0.74 | `insect_present` |

The fitted heads are independent probabilities, not a softmax. The serialized inference
contract suppresses subtype outputs unless `insect_present` also passes.

The source evidence supports Cicadidae versus broad Orthoptera. It does **not** support
inventing cricket/katydid versus grasshopper labels.

## Feature contract

- Backbone: Google Perch 2 CPU release
- Audio input to Perch: five seconds, 32 kHz, mono, float32
- Input to this artifact: one 1,536-dimensional Perch embedding
- Verified Perch model-tree SHA-256: `3fb2d54b3e34534f1130052b25737e54bbb5ebfd340ec040d4510772b64c81ff`
- Perch weights are not included

The official Perch code repository is Apache-2.0. The available Kaggle metadata did not
expose a separate model-weight license during this build, so this card does not extend
the code license to the weights by assumption.

## Training data

| Source | Windows | Role | Rights |
|---|---:|---|---|
| InsectSet459 | 2,096 | insect positives | 1,727 CC BY 4.0; 369 CC0 |
| ESC-50 | 1,960 | environmental negatives; insect class excluded | CC BY-NC 3.0 |
| Ross-308 | 317 | poultry and ventilation hard negatives | CC BY 4.0 |
| iNaturalist Domestic Chicken | 42 | observer-grouped hard negatives | CC BY 4.0 / CC0 |
| iNaturalist Domestic Cat | 26 | observer-grouped hard negatives | CC BY 4.0 / CC0 |
| **Total** | **4,441** | | mixed research lane |

InsectSet459 contains 2,096 unique permissive Train/Validation observations from 334
contributor groups: 1,009 cicada and 1,087 Orthoptera windows. Group IDs, not windows,
were assigned to train, validation, and test.

No source audio is redistributed here.

## Internal grouped evaluation

| Head | Test support | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| insect present | 236 | 0.987 | 0.947 | 0.979 | 0.963 |
| cicada | 102 | 0.959 | 0.856 | 0.931 | 0.892 |
| Orthoptera | 134 | 0.943 | 0.885 | 0.866 | 0.875 |

Test macro AP: **0.963**. Test macro F1: **0.910**.

These results show that the grouped source partitions are learnable. They do not establish
stationary-microphone field reliability.

## Locked independent dog challenge

The final artifact was scored once on 26 permissively licensed Domestic Dog sounds from
26 observers. These samples were not used for fitting or threshold selection.

| Output | Raw head crossings | Hierarchy-gated outputs |
|---|---:|---:|
| insect present | 11/26 | 11/26 |
| cicada | 7/26 | 5/26 |
| Orthoptera | 4/26 | 2/26 |

Outdoor taxon recordings may contain incidental insects, but a **42.3% broad activation
rate** is too high to treat this model as field-ready.

## Private stationary-microphone activation audit

A private reviewed frog archive supplied 1,308 five-second windows from 436 recordings.
It was reviewed for frog presence, not insect absence, so these are activation rates—not
false-positive rates:

| Output | Hierarchy-gated activations |
|---|---:|
| insect present | 1,207/1,308 — 92.3% |
| cicada | 68/1,308 — 5.2% |
| Orthoptera | 820/1,308 — 62.7% |

Summer frog recordings can contain real insects. Listening review is required before any
of these activations can be called errors. No private paths or audio are included here.

## Use

```python
import joblib
from insectnet.candidate import active_labels, predict_candidate

package = joblib.load("insectnet-research-0.2.0-perch2.joblib")
embedding = ...  # shape (N, 1536), from the verified Perch 2 feature contract
scores = predict_candidate(package, embedding)
labels = [active_labels(package, row) for row in scores]
```

Do not feed waveforms directly into this artifact.

## What is validated

- Exact data, window, embedding, and artifact hashes are recorded.
- Contributor/observer/source groups do not cross partitions.
- Only CC0, CC BY, and the explicitly noncommercial ESC-50 research lane were used.
- Broad/subtype hierarchy is serialized and tested.
- The model loads and scores finite Perch embeddings.
- Internal grouped metrics and independent challenge failures are both published.

## What is not validated

- local field precision or recall
- unattended field detections
- species-level insect identity
- cricket/katydid versus grasshopper separation
- robustness to frogs, ventilation, rain, distant birds, or recorder changes
- calibrated ecological abundance estimates

## Known failure mode and next research requirement

Frozen linear transfer on this corpus still carries source/domain shortcuts. Adding chicken,
cat, and poultry hard negatives improved known-source rejection but did not generalize to
the untouched dog challenge. The next serious model should use reviewed same-recorder hard
negatives, more diverse iNaturalist non-insect taxa, and likely supervised backbone
fine-tuning or a richer temporal head rather than another threshold-only adjustment.

## Sources

- InsectSet459: `academic-datasets/InsectSet459`, pinned revision documented in provenance
- ESC-50: Piczak, DOI `10.1145/2733373.2806390`
- Ross-308: Díaz de Cerio et al., DOI `10.34810/DATA3437`
- Perch: Ghani et al., *Global birdsong embeddings enable superior transfer learning for bioacoustic classification* (2023)

Exact source summaries, hashes, metrics, and challenge reports are included beside the
artifact.
