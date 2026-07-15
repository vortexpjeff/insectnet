# Data and model provenance

## Preserved claim

This repository preserves one v0.1 classifier artifact with an independently verified SHA-256. The artifact consumes 6,522-dimensional output logits from the declared BirdNET v2.4 FP16 TFLite backbone.

The exact original per-record training snapshot no longer survives. Therefore this release does not claim to be independently reproducible from raw source media.

## Surviving source-family record

| Source family | Surviving license information | Release treatment |
|---|---|---|
| InsectSet459 | Dataset card states CC BY 4.0; source material may also be CC0 | Named as a source family; media not redistributed |
| ESC-50 | CC BY-NC 3.0 | Named as a source family; media not redistributed |
| iNaturalist audio | License varies by individual recording | No blanket license asserted; media not redistributed |
| Private field negatives | Private research evidence | Not redistributed and location omitted |
| BirdNET backbone | Separate upstream model terms apply | Backbone is not redistributed here |

## Missing evidence

The following v0.1 records are unavailable:

- exact source-file list and per-record licenses;
- exact normalized window manifest;
- grouped train/evaluation split;
- threshold-selection report;
- complete field-review ledger linked to this exact artifact;
- original environment lockfile.

These gaps are recorded rather than reconstructed from later experimental branches.

## Release rule

A future model version must include:

1. artifact SHA-256 and byte size;
2. feature extractor identity and checksum;
3. ordered classes and independent thresholds;
4. source/window manifest digest;
5. per-record source and license provenance;
6. grouped split policy and seed;
7. scoped metrics and known false positives;
8. code commit and runtime source checksum;
9. deployment target and rollback artifact;
10. privacy-reviewed public documentation.

Model outputs remain assertions for review. They do not become confirmed observations without human validation.
