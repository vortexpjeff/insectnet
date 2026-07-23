# Perch 2 training strategy for InsectNet, ChickenNet, and FrogNet

Status: original research plan with deployed FrogNet amendment
Prepared: 2026-07-15
Amended: 2026-07-22
Scope: future model candidates; this document does not alter or supersede the preserved InsectNet v0.1.0 artifact.

## Decision

Train independent specialist packages on a shared, frozen Perch 2 embedding backbone:

- **InsectNet:** independent heads for insect presence and broad audible insect groups.
- **ChickenNet:** independent heads for audible chicken vocalization, crow, and other chicken vocalization.
- **FrogNet:** a separate broad `frog_present` head. Frog audio remains overlap and hard-negative material for the other models, never an InsectNet class.

The broad FrogNet dev4 head was deployed on 2026-07-22 as a conservative private field
probe at threshold `0.95`. It shares each window's frozen embedding with InsectNet and
ChickenNet and does not change their bundles, thresholds, or event semantics. See
[`FROGNET_FIELD_PROBE.md`](FROGNET_FIELD_PROBE.md) for the frozen release boundary.

BirdNET remains a separate bird detector. It is not a required feature extractor for these candidates.

The first candidates are frozen-embedding linear probes, not fine-tuned foundation models. Compare independent logistic-regression heads against linear SVM heads. Escalate to a neural head or backbone fine-tuning only if grouped, cross-source evaluation demonstrates that the linear boundary is inadequate.

## Deployment domain

The initial local calibration domain is one stationary microphone. This is valuable for learning that microphone's fixed response, background bed, seasonal activity, and recurring confounders. It is not evidence of generalization to other microphones or locations.

Public focal recordings and public passive-acoustic soundscapes provide most of the source, device, habitat, and geography diversity. Additional local areas and devices are later evaluation domains, not prerequisites for the first stationary-microphone candidate.

## Output contract

### InsectNet

Expose independent probabilities for:

1. `insect_present`
2. `cicada`
3. `cricket_katydid`
4. `grasshopper`
5. `insect_buzz`

`insect_present` is the broad gate. If it passes threshold while no subtype passes, report `insect_unresolved`; do not force a species or group.

`insect_buzz` is broader than `bee`. Public data does not support treating every wing-buzz or near-field insect movement as a bee.

### ChickenNet

Expose independent probabilities for:

1. `chicken_vocalization_present`
2. `chicken_crow`
3. `chicken_other_vocalization`

Do not expose `chicken_present`: an audio model cannot infer a silent chicken. No output passing threshold means no audible chicken vocalization; it is not a separate biological class.

Do not train health, disease, stress, welfare, hunger, heat, protein-deficiency, or emotional-state labels. Current public datasets confound those labels with flock, cage, age, room, treatment, recorder, and calendar time.

### FrogNet

Expose one broad probability:

1. `frog_present`

Do not expose species identity from this head. The local corpus is concentrated in a small
number of dates and chorus sessions; species-level claims require a separate dataset and
grouped external evaluation. Ambiguous windows remain unknown rather than forced negative.

### Mixed soundscapes

Use independent sigmoid heads. Valid windows include:

- frog and insect together;
- bird and insect together;
- chicken and machinery together;
- multiple insect groups together;
- no target heads active.

Never encode frog, bird, rain, machinery, or silence as one forced `background` class.

## Data rights lanes

Every recording must be assigned one immutable use lane before embedding:

| Lane | Allowed material | Model use |
|---|---|---|
| `core_releasable` | CC0, CC BY, and user-owned private audio | Default candidate and public release work |
| `sharealike_review` | CC BY-SA material | Separate legal/release review before inclusion |
| `research_noncommercial` | CC BY-NC, CC BY-NC-SA, iNatSounds research-only material | Separate noncommercial experiment only |
| `diagnostic_only` | Small challenge sets, uncertain terms, pretraining-overlap benchmarks | Evaluation or error analysis only |
| `excluded` | No stated license, ND licenses when transformations are required, withdrawn media, irrecoverable provenance | Never train |

Never merge these lanes into one unnamed artifact. Train at least a `core_releasable` model. A research-expanded model may be compared, but its manifest and artifact name must declare the restricted lane.

## Exact provenance ledger

Raw audio remains immutable. Git tracks ledgers and run manifests, not large audio archives.

### 1. `sources.csv`

One row per frozen dataset version or local collection:

- `source_id`
- canonical title
- version DOI or release identifier
- concept DOI when present
- repository and terms URLs
- retrieval timestamp
- dataset-level license statement
- clip-level-license authority
- deposited archive names, byte sizes, and upstream checksums
- local downloaded archive SHA-256
- intended role: train, validation, test, diagnostic, or excluded
- rights lane
- notes describing version/count discrepancies

### 2. `recordings.parquet`

One row per original recording:

- stable `recording_id` as `source_id:upstream_id`
- source archive and member path
- source URL and retrieval date
- original byte SHA-256
- perceptual/audio fingerprint for transcode duplicate detection
- duration, sample rate, channels, codec, and bit depth
- raw label and normalized labels
- taxon ID and taxonomic authority where available
- individual, flock, corral, observer, recordist, or uploader ID
- parent session, site, date, and device group
- per-recording license URI and attribution text
- rights lane
- private/public status
- review status and exclusion reason

For private field material, publish only opaque site/session IDs. Exact coordinates, private filenames, and private paths stay outside the public repository.

### 3. `windows.parquet`

One row per derived model window:

- deterministic `window_id`
- parent `recording_id`
- start and end milliseconds
- preprocessing recipe ID
- derived-audio SHA-256
- all active labels
- label authority: strong annotation, human review, source weak label, or model proposal
- uncertainty and overlap flags
- split group
- assigned train/validation/test partition
- sample weight
- augmentation parent IDs and parameters, if derived
- rights lane inherited from every source used in the window

A weak file-level label does not automatically apply to every five-second crop. Crops with no audible target become uncertain until reviewed or bounded by strong annotations.

### 4. `embeddings.parquet`

- `window_id`
- Perch model identifier and artifact SHA-256
- signature name
- input sample rate and window length
- embedding dimension and dtype
- embedding-file path and SHA-256
- extraction code commit
- extraction timestamp

### 5. `runs/<run_id>/training_manifest.json`

Every training run records:

- exact sorted window IDs and labels, or a content-addressed locked manifest containing them
- SHA-256 of the complete selected rows, labels, split groups, partitions, and weights
- source-version and rights-lane summary
- excluded rows and reasons
- code commit and dirty-tree state
- Python and dependency versions
- Perch model SHA-256
- preprocessing configuration
- split algorithm and random seed
- classifier hyperparameters
- thresholds and how they were selected
- grouped metrics and per-source metrics
- model artifact byte size and SHA-256

The current legacy `dataset_hash` hashes only sorted labels. That is insufficient: two different recordings with the same labels produce the same hash. Future hashes must cover exact sample IDs, labels, groups, splits, and weights.

### Attribution outputs

Each frozen candidate produces:

- `ATTRIBUTION.csv`
- `EXCLUSIONS.csv`
- `DATA_CARD.md`
- the locked source, recording, window, embedding, and run manifests

## Cross-source duplicate control

Exact SHA-256 catches byte duplicates but misses transcodes and excerpts. Before splitting:

1. compare upstream observation, Freesound, xeno-canto, and iNaturalist IDs;
2. compare exact SHA-256;
3. compute an audio fingerprint for transcoded duplicates;
4. search high-similarity Perch embeddings for excerpts or near duplicates;
5. group suspected duplicates into one parent cluster;
6. keep every cluster in one partition.

Highest-risk overlaps:

- InsectSet459 with direct iNaturalist and xeno-canto additions;
- iNatSounds with direct iNaturalist additions;
- ESC-50 with FSD50K and other Freesound-derived material;
- original recordings with their extracted segments or processed variants.

## InsectNet public source strategy

### Core releasable positives

#### InsectSet459 v1.1

- Version DOI: <https://doi.org/10.5281/zenodo.18554693>
- The inspected annotation CSV has 26,297 rows; the data paper/deposit describes 26,298 files. Record this discrepancy rather than silently choosing one count.
- 459 Orthoptera and Cicadidae species; about 83.66 GB deposited.
- Clip-level license audit:
  - 16,213 CC BY-NC
  - 5,411 CC BY-SA
  - 1,983 CC BY-NC-SA variants
  - 2,161 CC BY
  - 449 CC0
- Default core pool: exactly 2,610 CC0/CC BY rows before quality and duplicate filtering.
- Core subset: 248 species, 1,333 Orthoptera and 1,277 cicadas.

Use the core subset for broad positive diversity. Preserve observation, contributor, source URL, species, and clip-level license. Do not treat the official split as the final field benchmark because 1,115 recordists occur in multiple splits and 533 occur in all three.

#### Oxfordshire Orthoptera PAM dataset

- Version DOI: <https://doi.org/10.5281/zenodo.20625750>
- CC BY 4.0.
- Ten site ZIPs; 112,921,460,139 deposited bytes.
- Nineteen usable AudioMoths at ten sites; 96 kHz, 15-second scheduled field recordings.
- Label vocabulary: 386 labels in 22 categories, including 158 Orthoptera labels plus birds, bats, mammals, geophony, and anthropophony.
- Frozen label-table counts:
  - train: 3,752 annotation rows over 552 files
  - validation: 486 rows over 67 files
  - test: 1,571 rows over 197 files
  - total: 5,809 rows over 816 labeled files
  - 2,666 rows have `orthoptera` as the primary label category
- Keep the three held-out test sites (`CON_2`, `REG_1`, `REST_3`) locked and unavailable to training or adaptation.

Use the labeled training files for field-event supervision and mixed-soundscape negatives. Use validation for model selection. Use the held-out sites only as external cross-site challenge data.

#### Targeted North American additions

Add only Appalachian-relevant gaps not already represented in InsectSet459. Candidate sources are direct iNaturalist and xeno-canto queries filtered to compatible per-recording licenses.

Freeze:

- query text and taxon IDs;
- API version;
- retrieval timestamp;
- observation/recording IDs;
- recordist/observer;
- date and coarse public region;
- per-file license and attribution;
- content checksum.

Do not use iNatSounds official evaluation as independent proof after training on other iNaturalist-derived material.

### Research-only insect expansion

#### ECOSoundSet

- Version DOI: <https://doi.org/10.5281/zenodo.18636037>
- CC BY-NC 4.0.
- Current version: 11,224 recordings, 193 orthopteran species and 24 cicada species; about 124.61 GB deposited.
- Strong time/frequency annotations and natural soundscapes make it valuable for event localization and overlap.

Keep it in the `research_noncommercial` lane unless the release policy explicitly accepts NC material.

#### iNatSounds 2024

- 232,237 recordings, about 1,200 hours and more than 5,500 species.
- Insect training split: 10,080 files across 745 species.
- Research/education use with no redistribution; individual media licenses still apply.

Use only in a separately named research model after overlap checks.

#### InsectSound1000

- DOI: <https://doi.org/10.5073/20231024-173119-0>
- 165,982 published 2.5-second samples from 12 species, 72 recording nights, four channels at 16 kHz.
- Anechoic-box greenhouse insects; severe field-domain mismatch and an 8 kHz Nyquist limit.

Optional representation experiment only. It must not dominate sampling and is not a deployment benchmark.

### InsectNet negatives and mixed scenes

Use broad public sources plus stationary-microphone audio:

- Oxfordshire annotations for birds, bats, geophony, cars, planes, wind, and microphone artifacts;
- license-filtered CC0/CC BY FSD50K clips for birds, frogs, mammals, weather, machinery, speech, and domestic noise;
- reviewed local frog, bird, machinery, rain, creek, human, and quiet windows;
- local trigger-free intervals sampled across hour, season, weather, and noise state.

The local frog reservoir contains strong positives but heavy dependence: 436 files, 23 five-minute trigger groups across three days, and 392 files from one two-hour chorus. Cap or session-weight it. Review mixed frog/insect and frog/bird windows explicitly.

Trigger-free does not mean target-free. Screen local blank intervals before assigning all insect heads zero.

## ChickenNet public source strategy

### Core releasable positives

#### Ross-308 Broiler Vocalization Audio Dataset

- DOI: <https://doi.org/10.34810/DATA3437>
- CC BY 4.0; 304,283,565-byte ZIP.
- Twelve individually identified four-week broilers.
- Twenty-four individual recordings, four flock recordings, two animal-free ambient recordings.
- 1,420 manually bounded vocalization segments.

Use all audibly valid segments but weight by individual and parent recording. Split by bird for isolated recordings and by complete corral/session for flock recordings. Never split raw recordings and their extracted segments across partitions. Ross supplies mostly `chicken_other_vocalization`, not robust adult crow coverage.

#### Poultry Vocalization Signal Dataset for Early Disease Detection

- DOI: <https://doi.org/10.17632/zp4nf2dxbh.1>
- CC BY 4.0; 1,092,692,300 published bytes.
- 346 WAV files: 139 healthy, 121 unhealthy, 86 noise.

Discard health as a model target. After audio review, merge chicken-bearing files into `chicken_other_vocalization`. Use `noise` only when no chicken vocalization is audible. If parent sessions cannot be reconstructed, keep the entire dataset in training and never use it as validation evidence.

#### SmartEars

- DOI: <https://doi.org/10.17632/dy6gtvt4mk.2>
- CC BY 4.0.
- 6,000 five-second farm clips: 2,000 healthy, 2,000 sick, 2,000 none.

Collapse healthy and sick to audible chicken vocalization. Audit `none`; it can contain farm-domain confounders and may still contain faint calls. Parent grouping is not publicly exposed, so SmartEars is training-only and source-weighted so it cannot dominate.

#### FSD50K

- DOI: <https://doi.org/10.5281/zenodo.4060432>
- 173 `Chicken_and_rooster` clips: 138 development and 35 evaluation.
- Mixed clip-level licenses.

Manually relabel every candidate as crow, other chicken vocalization, mixed/ambiguous, or false positive. Core use accepts only CC0/CC BY clips. Preserve uploader-disjoint grouping and treat FSD50K as adaptation material, not independent Perch evaluation.

#### Direct iNaturalist domestic-chicken audio

A 2026-07-15 snapshot found 485 sound files for explicit domestic chicken observations. Only 42 were clearly CC0 or CC BY before quality filtering; three more were CC BY-SA.

Freeze the exact API response and query at corpus creation. Group by observation and observer. Manually label crow versus other vocalization. Do not mix broader wild `Gallus gallus` results into domestic-chicken counts.

#### Additional crow material

Use license-compatible xeno-canto Red Junglefowl and direct public recordings only after listening, license filtering, and overlap checks. Preserve wild versus domestic provenance. Public crow diversity is still insufficient to claim robust stationary-microphone performance without local crow recordings.

### Research-only or diagnostic chicken sources

#### ESC-50

- CC BY-NC 3.0.
- Forty rooster clips from 33 source files and 40 hen clips from 28 source files.

Use as a research seed or diagnostic set, preserving official folds and `src_file`. The ESC-50 `crow` class is a corvid and is a high-value hard negative, not a rooster label.

#### PLOS TinyML poultry supplement

- DOI: <https://doi.org/10.1371/journal.pone.0316920>
- The paper describes 3,600 examples, but the downloadable supplement contains only 19 WAV files totaling 883.37 seconds plus processed variants.

Use only audibly reviewed original recordings. Keep processed variants with their parent and never count them as independent evidence.

#### Laying-hen experimental recordings

The inspected Zenodo record 10433023 contains approximately 2.49 GB under CC BY 4.0, including control and treatment recordings. Audible dog playback, umbrella events, cage, age, and treatment can become shortcuts. Use only for chicken-vocalization presence or external stress testing; never use the experiment's stress labels as ChickenNet targets.

### ChickenNet negatives

Use both matched and open-world negatives:

- Ross animal-free ventilation audio;
- audited Mendeley and SmartEars noise/none clips;
- Oxfordshire field birds, galliforms, frogs, mammals, vehicles, wind, and recorder artifacts;
- license-filtered FSD50K public negatives;
- local stationary-microphone birds, frogs, insects, people, dogs, machinery, rain, creek, and quiet intervals.

Emphasize corvid crows, pheasants, turkeys, doves, owls, geese, woodpeckers, and songbirds. The Perch poultry probe showed that raw Perch labels can confuse crows and woodpeckers with chicken vocalizations.

Do not let source identify label: every label must draw from multiple datasets where possible, and every major source should contribute both target and non-target windows.

## Sampling and weighting

Balance by independent source group, not by raw crop count.

### Initial InsectNet batch sampling

- 50–60% Appalachian-relevant and broad core InsectSet459 positives plus targeted North American additions.
- 20–25% strongly annotated Oxfordshire field events and mixed scenes.
- 10–15% broader insect acoustic diversity.
- 20–30% explicit hard negatives and mixed soundscapes.

These overlap as multilabel sampling goals and are not raw-corpus percentages. Cap contribution per recording, recordist, site/date, and source dataset.

### Initial ChickenNet batch sampling

- Make Ross the principal `chicken_other_vocalization` source, but weight birds and parent recordings approximately equally.
- Build `chicken_crow` from multiple reviewed public sources; do not duplicate or heavily augment a few source files until they appear numerous.
- Include at least 30% explicit negative/mixed windows in each training batch.
- Cap SmartEars and other sliced datasets by source so thousands of neighboring or provenance-poor clips cannot dominate.

## Split strategy

Never split derived windows randomly.

### InsectNet

- InsectSet459: group by observation; strengthen grouping by contributor, date, and site where available.
- Oxfordshire: retain official train/validation split and lock all three held-out test sites.
- Direct citizen science: group by observation, uploader/recordist, location, and date/session.
- Local stationary microphone: split by complete date/time blocks; keep a final date-grouped challenge set untouched.
- Keep all windows from one parent recording, session, duplicate cluster, or augmentation family together.

### ChickenNet

- Ross: group by individual bird; flock audio by entire corral/session.
- Mendeley/SmartEars: group by the longest recoverable original session; if unavailable, training-only.
- FSD50K: preserve uploader-disjoint partitions.
- ESC-50: preserve official fold and source-file groups.
- Citizen science: group by observation, observer/recordist, location, and date.
- Local stationary microphone: use complete date blocks for negative calibration. It cannot measure positive recall until independent local chicken vocalizations exist.

## Training experiment matrix

Run the same frozen Perch embeddings through these candidates:

1. Logistic heads, core-releasable data only.
2. Linear SVM heads, core-releasable data only.
3. Winning linear family plus reviewed local stationary-microphone hard negatives.
4. Separately named research-expanded candidate adding NC/research-only data.
5. Ablation without local frog/bird negatives.
6. Ablation with local frog as a separate auxiliary output versus hard negatives only.

No foundation-model fine-tuning in this phase.

## Metrics and promotion gates

Report window-level metrics for diagnosis, but promote using grouped event metrics:

- per-head AUPRC, precision, recall, and F1;
- false positive events per recording hour;
- recall at fixed false-positive budgets;
- Brier score and calibration plots;
- session/site/source macro averages;
- confusion rates for bird-to-frog, bird-to-chicken, frog-to-insect, machinery-to-insect, and corvid-to-chicken;
- metrics with the largest source/session removed;
- cross-source and held-out-site performance.

Merge adjacent positive windows into one event before FP/hour calculation.

Minimum shadow-mode promotion goals:

- at least 90% event precision on the locked stationary-microphone challenge set;
- at least 80% recall on clearly audible held-out target events;
- no more than 0.25 false grouped events per target-absent recording hour;
- no material degradation when any one source dataset is removed;
- thresholds chosen without access to locked test partitions;
- complete per-recording provenance for every training row.

ChickenNet cannot satisfy a local-positive promotion gate until independent local chicken vocalization sessions are collected. Until then it is a public-data base model with local negative calibration only.

## Order of work

1. Build and validate the ledger schemas and rights-lane rules.
2. Freeze source versions and download only metadata/manifests first.
3. Generate candidate recording rows and exclusions before large audio downloads.
4. Download the selected core files and verify checksums.
5. Detect cross-source duplicates before assigning splits.
6. Derive non-overlapping five-second windows and record every transform.
7. Human-review weak positives and ambiguous/mixed windows selected for training or evaluation.
8. Extract Perch embeddings once with a hashed model and preprocessing recipe.
9. Lock groups and partitions.
10. Train the core logistic and SVM candidates.
11. Run public cross-source and local stationary-microphone challenge evaluations.
12. Only then decide whether restricted datasets or more local areas add enough value to justify another candidate.

## Explicit non-goals

- Do not modify BirdNET or use BirdNET logits as required features.
- Do not bulk-ingest the frog folder as 436 independent positives.
- Do not use the current public archive embeddings as release provenance; their raw audio and per-recording rights metadata are incomplete.
- Do not use shuffled K-fold evaluation.
- Do not optimize thresholds on the locked test set.
- Do not infer biological health or welfare state from these corpora.
- Do not publish private paths, exact local coordinates, or private raw audio.
