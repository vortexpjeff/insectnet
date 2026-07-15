# Security and privacy boundary

## Public release boundary

This repository must not contain:

- exact collection coordinates or street addresses;
- private site names or fine-grained regional identifiers;
- device network addresses or topology;
- account names, passwords, API keys, tokens, or credential helpers;
- private workstation or device paths;
- raw private audio;
- source filenames that reveal private deployment details;
- unattended deployment commands without a separately reviewed operations plan.

The test suite scans public text files for known historical location and access strings.

## Artifact safety

The model is serialized with joblib, which relies on Python pickle. A malicious artifact can execute code when loaded. Verify the model SHA-256 against the packaged `v0.1.0.manifest.json` before loading it. The verifier stops before deserialization if byte size or SHA-256 differs.

The release verifier checks:

- repository-relative artifact path;
- exact SHA-256;
- exact byte size;
- required joblib keys;
- ordered classes;
- 6,522-feature contract;
- classifier estimator count.

## Operational boundary

This v0.1 release does not provide live capture or deployment automation. Any future edge deployment must be released separately with:

- no embedded credentials;
- explicit destination supplied by the operator;
- immutable backup before replacement;
- checksum and model-load preflight;
- source syntax checks;
- single-instance control;
- bounded resource use;
- disk guard and retention policy;
- rollback procedure;
- post-change health verification;
- no changes to the upstream monitoring system without separate approval.

## Reporting

Public performance reports must generalize location and describe the evaluation set, review authority, thresholds, and limitations. Confidence values alone are not validation.
