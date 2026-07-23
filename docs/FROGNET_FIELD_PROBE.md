# FrogNet field probe v0.1.0

**Status:** public noncommercial research prerelease; safe JSON/NPZ bundle deployed as a private review-only field probe on 2026-07-22.

## Purpose

FrogNet is a binary `frog_present` linear head over frozen Google Perch 2 embeddings. It detects broad audible frog or toad vocalization; it is not a species classifier and does not establish ecological occurrence without human review.

The field runtime processes each 15-second recording as three non-overlapping five-second mono 32 kHz windows. Perch produces one 1,536-dimensional embedding per window. InsectNet, ChickenNet, and FrogNet then score that same embedding independently; Perch is not run once per head.

## Deployed contract

| Field | Value |
|---|---|
| Bundle | `frognet-dev4-field-probe` |
| Bundle ID | `db54359b42526010a2e7782837d2ff8a5e7d98beeebba9214211a2ee83572fa8` |
| Output | `frog_present` |
| Field threshold | `0.95` |
| Backbone | frozen Perch 2, 1,536 dimensions |
| Serialization | JSON metadata + NPZ numeric arrays only |
| Release | [frognet-field-probe-v0.1.0](https://github.com/vortexpjeff/insectnet/releases/tag/frognet-field-probe-v0.1.0) |

The public tag points to source commit `55d42a8c96584e09504b9b61127b0dd9e069cdfc`. The deterministic release archive SHA-256 is `bc82746ca83b305591073d1aeba8a4598e9860953d9a451d087695ac927d29e1`.

## Evaluation boundary

At threshold `0.95`, the release smoke test produced:

- 1,253 / 1,308 activations on private local frog windows;
- 106 / 128 activations on a regional iNaturalist frog set;
- 0 / 3,781 activations across locked insect, chicken, cat, dog, SmartEars, and laying-hen controls.

The highest tested confound was a dog window at `0.92168`, below threshold. Faint, distant, and mixed local vocalizations remain the principal known miss condition. These are challenge-set observations, not general prevalence-adjusted precision or recall claims.

## Rights and privacy

The bundle contains no audio, coordinates, observation locations, uploader identities, private manifests, credentials, or arbitrary Python serialization. The regional iNaturalist tranche admitted only CC0 or CC BY sounds. The broader private research corpus also used ESC-50 material under CC BY-NC terms; therefore the release is a noncommercial research field probe and must not be represented as a fully permissive corpus or model.

## Related implementation

- Corpus builder: `scripts/build_frognet_corpora.py`
- Grouping and provenance helpers: `src/insectnet/frognet_corpus.py`
- Tests: `tests/test_frognet_corpus.py`
- Shared runtime is maintained in the private field-operations repository.
- Archive validation integration is maintained in the separate field-operations repository.
