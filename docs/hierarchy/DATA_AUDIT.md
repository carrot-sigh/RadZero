# Hierarchy Data Audit

Audit date: 2026-09-28.

## Source data

- Chest ImaGenome 1.0.0 archive:
  `/media/user/Data/Chest ImaGenome Dataset/chest-imagenome-dataset-1.0.0.zip`
- MIMIC-CXR-JPG 2.1.0 images:
  `/home/user/gxq/Data/mimic-cxr-jpg-2.1.0/files`
- MIMIC-CXR-JPG 2.1.0 metadata and CheXpert labels:
  `/media/user/Data/mimic/MIMIC/mimic-cxr-jpg-2.1.0`
- Existing RadZero Parquet manifests:
  `/home/user/gxq/radzero-data/data/MIMIC-CXR`

The separately extracted Chest ImaGenome scene-graph directory is incomplete:
only 833 of 67,241 files inspected were non-empty. The nested
`silver_dataset/scene_graph.zip` inside the official outer archive is healthy
and is the canonical source for this project.

## Verified coverage

- Scene graphs: 243,310 valid JSON files, with no decode failures.
- Official Chest ImaGenome split coverage: 166,504 train, 23,951 valid, 47,389
  test, and 5,466 excluded from the official split tables.
- Every scene graph resolves to a JPG under the local MIMIC-CXR-JPG image root.
- MIMIC study split coverage: 222,750 train, 1,808 validate, and 3,269 test.
- V1 label contract: 36 anatomy regions, 41 regional findings, 14 global
  labels, and 310 compatible anatomy-finding pairs.

## Frozen policy

- Use only official bbox-detectable anatomy regions.
- For regional findings, explicit `yes` is positive and explicit `no` is
  negative. Missing or conflicting polarity is unknown and excluded from loss.
- Exclude `rib fracture` and `subcutaneous air` from the regional vocabulary
  because neither has explicit supervision for a detectable anatomy in train.
  The global `Fracture` label remains available.
- For global labels, `1.0` is positive, `0.0` is negative, and `-1.0` or blank
  is unknown. A blank `No Finding` is negative only when another global label is
  explicitly positive.
- Derive compatibility and class statistics from training data only.
- Use the official Chest ImaGenome split as the single hierarchy split for all
  three tasks. MIMIC-CXR split disagreements are audited but do not remove rows;
  global targets and class weights follow the Chest split to prevent internal
  train/validation/test leakage.
- Retain split-resolved studies without a CheXpert row for anatomy/finding
  supervision, with every global-label validity bit set to false.

The generated contract is versioned at
`exp/cxr_pt/configs/hierarchy_contract.json`; the manifest builder preserves its
label orders and the canonical split assignments.

## Generated manifest

The complete V1 JSONL manifest is stored outside Git at
`/home/user/gxq/radzero-data/data/MIMIC-CXR/hierarchy-v1.jsonl` (237,844 rows,
560 MB). It contains 166,504 train, 23,951 valid, and 47,389 test images. Ten
rows have no CheXpert label record and therefore use an all-false global
validity mask. The builder recorded 73,981 MIMIC split disagreements for audit;
the canonical Chest ImaGenome split remains unchanged.
