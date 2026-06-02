# Confidence Thresholds

Per-class confidence guidance for interpreting InsectNet predictions.
**Thresholds are class-specific** — a universal cutoff produces both
false positives and false negatives.

## Current Guidance

| Class | Recommended Threshold | Status | Notes |
|-------|----------------------|--------|-------|
| cicada_drone | 0.50 | Confirmed | Natural capture at 83%, AC false positive at 92% — use RMS to disambiguate |
| frog | 0.50 | Confirmed | Validated at 51%; chorus peaks at 80-99%. RMS >0.004 increases confidence |
| cricket_katydid | 0.50 | Tentative | Summer chorus data needed for natural threshold |
| grasshopper | 0.80 | Data-limited | Only 183 training clips; most captures likely noise |
| bee | 0.80 | Data-limited | Only 43 training clips; night detections are false positives |

## How Thresholds Were Determined

### Cicada (83% natural confirmed)

The first natural cicada capture scored 83% with RMS 0.009. Playback tests
hit 99-100%. However, an AC window unit also scores 92% on this class with
background <2%. The 80%+ range includes both real cicadas and false positives.

**Disambiguation:** RMS. Real cicada at 83% had RMS 0.009. AC at 92% had
RMS 0.02+. A high-confidence cicada detection with high RMS may be mechanical.

### Frog (51% natural confirmed)

A frog detected at only 51% was confirmed real by the user. The frog chorus
ranged from 55% (early evening, quiet) to 99.97% (peak chorus, loud).

**Guidance:** Any capture above 50% frog with RMS >0.004 is worth review.
Loud captures (RMS >0.015) at 80-99% are highly likely real.

### Bee (no natural data)

The bee class has no known real detections. All captures to date are false
positives: weed whacker at 98%, night ambient noise at 50-70%. The true bee
threshold cannot be established without natural captures.

**Guidance:** Discard bee detections at night. Treat day detections below
80% as noise.

## Production vs Testing Threshold

| Context | Threshold | Rationale |
|---------|-----------|-----------|
| **Production capture** | 0.30 default | Favor recall — a few false positives are cheaper than missed detections |
| **Automated decision** | 0.80 minimum | Only act on high-confidence predictions without human review |
| **Research / scanning** | 0.30 | Cast a wide net; review all uncertain captures manually |

The 0.30 default is conservative for capture. It produces some uncertain
clips but the cost of missed insects is higher than the cost of reviewing
a few extra WAVs.

## RMS Noise Floor

The sidecar skips inference for WAVs below the RMS noise floor (default
0.002). This was calibrated from Pine Hollow ambient measurements and
corresponds to quiet background with no detectable acoustic activity.

Playback sessions may need a lower floor (0.001) for quiet phone speakers.
