# Changelog

## v0.1.0-clean — 2026-07-14

- Preserved the original v0.1 classifier bytes and checksum.
- Established v0.1 as the sole public model identity.
- Added a machine-readable release manifest and checksum verifier that refuses to deserialize mismatched bytes.
- Packaged the exact model and manifest with the Python distribution.
- Added contract-checked scoring for precomputed BirdNET logits.
- Added model, release, CLI, and privacy tests.
- Removed live capture and deployment code pending a separately reviewed release.
- Removed exact location, network, credential, and private-path information.
- Corrected source-family license statements and documented missing per-record provenance.
- Reframed the artifact as a historical research reference rather than a production model.
