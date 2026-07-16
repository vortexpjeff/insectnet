# Per-head unknown-label semantics

The Perch candidate manifest supports partially known multi-label examples through an optional `unknown_labels` list.

## Why this exists

In passive acoustic data, absence of a label does not always mean confirmed absence of the sound. A window may be confirmed as one class while another class is not reviewed. Treating every omitted class as a negative introduces false-negative supervision.

For each independent classifier head:

- `labels` contains known positives;
- `unknown_labels` contains classes that must be excluded for that head;
- every other eligible class is treated as negative under the current manifest contract.

A class may not appear in both lists on the same row.

## Training and evaluation

Eligibility is computed independently for each class and partition. Rows where the class appears in `unknown_labels` are excluded from that head’s training, threshold selection, and evaluation metrics. The package records eligible row counts per head for train, validation, and test partitions.

This changes the candidate package schema to version 2 and records:

```text
label_semantics = per_head_unknown_labels_v1
head_eligibility[class][partition] = evaluated row count
```

The canonical manifest hash includes `unknown_labels`, so changing review certainty changes dataset identity.

## Boundary

This mechanism represents uncertainty in annotation coverage. It does not convert an unknown into a positive, infer a subtype, calibrate model probabilities, or bypass the requirement for leakage-safe recording groups and partition validation.
