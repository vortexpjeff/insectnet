---
license: cc-by-nc-sa-4.0
library_name: sklearn
tags:
  - bioacoustics
  - audio-classification
  - birdnet
  - edge-ai
  - research
  - non-commercial
datasets:
  - academic-datasets/InsectSet459
  - ESC-50
---

# InsectNet v0.1.0

InsectNet v0.1.0 is a preserved research classifier for scoring insect- and amphibian-related acoustic classes from frozen BirdNET v2.4 output logits.

This repository preserves one canonical v0.1 release identity and also carries separately
versioned Perch 2 research candidates. It does **not** publish live sensor addresses,
deployment credentials, exact collection locations, private media, or an unattended
capture service.

## Release identity

```text
Release:  insectnet-v0.1.0
Artifact: src/insectnet/data/classifier.joblib
SHA-256: 5e6ecfc68d78a2cf2e9e9e47da5cb58d696e8de354fd620cfcccc5db9da48702
Bytes:    474,892
Status:   historical research reference
```

The artifact is byte-for-byte preserved from the original v0.1 field prototype.

## Model contract

```text
Audio window:       3.0 seconds
Sample rate:        48,000 Hz mono
Backbone:           BirdNET v2.4 FP16 TFLite
Feature space:      6,522 BirdNET output logits
Classifier:         StandardScaler → OneVsRest LogisticRegression
Serialization:      scikit-learn 1.8.0
Output semantics:   independent per-class probabilities
```

Class order:

1. `background`
2. `bee`
3. `cicada_drone`
4. `cricket_katydid`
5. `frog`
6. `grasshopper`

The model file does not embed thresholds, a version number, or its original training snapshot. Those omissions are part of the preserved v0.1 record rather than silently reconstructed metadata.

## What is included

- the exact v0.1 model artifact;
- a machine-readable release manifest;
- artifact and feature-contract verification;
- offline scoring for precomputed BirdNET logit vectors;
- tests that enforce the model checksum, class order, feature dimension, and public privacy boundary;
- documented provenance and limitations.

## What is not included

- live capture or sensor-watching code;
- deployment scripts;
- device addresses or credentials;
- exact collection locations;
- raw or private field audio;
- claims of production readiness.

The original live sidecar diverged from the public v0.1 source during field experiments. That runtime is not republished here until it can be recovered, tested, and released under a separate reviewed version.

## Verify the preserved artifact

From a source checkout:

```bash
uv run insectnet verify
```

Expected SHA-256:

```text
5e6ecfc68d78a2cf2e9e9e47da5cb58d696e8de354fd620cfcccc5db9da48702
```

## Score precomputed logits

InsectNet v0.1 expects one finite NumPy vector with shape `(6522,)` extracted from the declared BirdNET backbone:

```bash
uv run insectnet score logits.npy
```

The command returns one probability per declared class. These scores are model assertions for review, not confirmed biological observations.

> **Joblib safety:** joblib artifacts use Python pickle internally. Load only the artifact whose checksum matches the release manifest.

## Known limitations

- Later audits found high false-positive rates on some bird vocalizations.
- Bee and grasshopper had limited training coverage.
- The surviving metrics came from limited public-data evaluation and one private field site; they do not establish general production performance.
- Exact reproduction is blocked because the original per-record training snapshot is unavailable.
- The classifier relies on BirdNET logits and cannot score raw audio by itself.
- There is no validated automated-decision threshold policy in this release.

## Perch 2 research candidates and field probes

Three provenance-locked specialist lines were trained and audited in July 2026. None
replaces the preserved BirdNET-logit v0.1 artifact. The public research packages remain
distinct from the later strict JSON/NPZ bundles used by the private field listener.

| Candidate | Status | Key external result |
|---|---|---|
| [ChickenNet Research 0.1.0](models/chickennet-research-0.1.0-perch2/MODEL_CARD.md) | research candidate; not deployed | 33/42 broad-head hits on a locked iNaturalist chicken challenge; 10/1,308 candidate activations on a private local confound set |
| [InsectNet Research 0.2.0](models/insectnet-research-0.2.0-perch2/MODEL_CARD.md) | trained but not field-ready; not deployed | 11/26 broad activations on an untouched iNaturalist dog challenge |
| [FrogNet field probe v0.1.0](docs/FROGNET_FIELD_PROBE.md) | noncommercial research prerelease; strict bundle deployed privately | 1,253/1,308 local frog-window activations and 0/3,781 tested confound activations at threshold `0.95` |

All three lines consume 1,536-dimensional Google Perch 2 embeddings from five-second,
32 kHz mono windows. They include exact model/data hashes, grouped split reports,
hierarchy contracts, source summaries, and challenge reports. They do not include Perch
weights or source audio.

## Training strategy

[`docs/PERCH2_TRAINING_STRATEGY.md`](docs/PERCH2_TRAINING_STRATEGY.md) records the
provenance-first design used for the Perch 2 specialist heads. The current FrogNet
deployment and release contract is frozen in [`docs/FROGNET_FIELD_PROBE.md`](docs/FROGNET_FIELD_PROBE.md).

## Provenance

The surviving records identify these source families:

- **InsectSet459:** current dataset card states CC BY 4.0, with some source material CC0.
- **ESC-50:** CC BY-NC 3.0.
- **iNaturalist audio:** licenses vary per recording; the original per-record manifest is unavailable.
- **Private field negatives:** not redistributed.

See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) and the release manifest for the exact surviving claims and gaps.

## Privacy and security

The public release intentionally omits exact collection location, network topology, account names, credentials, private paths, and raw evidence. See [`docs/SECURITY_AND_PRIVACY.md`](docs/SECURITY_AND_PRIVACY.md).

## License

The repository and preserved release are distributed under CC BY-NC-SA 4.0, subject to the licenses and terms of the upstream backbone and source media. Source-media rights vary; users are responsible for reviewing those upstream terms for their use case.

This provenance statement is not legal advice.
