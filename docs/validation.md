# Field Validation

InsectNet field validation results from Pine Hollow, Tennessee (35.8565, -83.3744).
All detections are human-confirmed unless marked as playback.

## Confirmed Natural Detections

| Date | Class | Conf. | Species | Method |
|------|-------|-------|---------|--------|
| May 30 | cicada_drone | 83% | Neotibicen/Megatibicen | User confirmed by ear |
| May 30 | frog | 51% | Cope's Gray Treefrog | User heard outside + WAV confirmed |
| May 30 | frog | 80-99.97% | Cope's Gray Treefrog + Eastern Narrow-mouthed Toad | User confirmed chorus, two species |
| May 31 | cricket_katydid | 99% | Field cricket | User confirmed by ear |

## Confirmed Playback Detections

| Date | Class | Conf. | Species | Notes |
|------|-------|-------|---------|-------|
| May 29 | cicada_drone | 100% | Neotibicen lyricen | Phone playback at mic |
| May 29 | frog | 64.5% | American Toad | Phone playback, BirdNET cross-validated |
| May 29 | frog | 99.2% | Gray Treefrog (Dryophytes spp.) | Phone playback |
| May 29 | cricket_katydid | 100% | Gryllus campestris | Phone playback; BirdNET misidentified as G. fultoni |

## Known False Positives

| Source | Class Triggered | Confidence | Notes |
|--------|----------------|------------|-------|
| AC window unit | cicada_drone | Up to 92.3% | Background <2% — very confident wrong answer |
| Weed whacker | bee | 98.1% | User confirmed |
| Night ambient noise | bee | 50-70% | Temporal filter needed — bees don't fly at night |

## Key Validations

### Frog Chorus (May 30)

The first sustained natural capture event. ~440 frog detections over 2.5 hours
(21:00-23:30). Two clear phases:

1. **Early evening (17:25-18:35):** Individual frogs at 55-65% confidence,
   RMS 0.003-0.008
2. **Chorus peak (21:00-23:00):** Sustained 80-99.97% confidence,
   RMS 0.01-0.08

Time-resolved BirdNET logit analysis identified two species calling
simultaneously: Cope's Gray Treefrog (dominant, throughout) and Eastern
Narrow-mouthed Toad (secondary, second half).

### First Natural Cicada (May 30, 06:59)

After ~14 hours running unattended through overnight rain, the sidecar
captured a cicada at 83% confidence (RMS 0.009). The user confirmed it
sounded like a genuine cicada. Cosine similarity against training centroids
matched Neotibicen/Megatibicen (0.986).

### First Low-Confidence Frog (May 30, 20:08)

A frog detected at only 51% confidence (RMS 0.004) was confirmed real —
the user heard the frog near the mic and confirmed the WAV. This invalidated
the hypothesis that real detections always clear 80% and established
class-dependent thresholds.

## Validation vs Testing

- **Testing** confirms the pipeline works: inotify fires, WAVs get processed,
  files end up in the right places. Confidence scores are not evidence.
- **Validation** requires human listening and explicit species identification.
  Only validated captures qualify as training data.

All detections listed above are validated.
