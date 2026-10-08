# RadGraph-free Semantic-chain Audit

This audit is read-only with respect to labels and models. It evaluates whether
the existing Chest ImaGenome and CheXpert annotations can support semantically
distinct L1 evidence, L2 finding, and L3 interpretation nodes. It does not train
a model, export training labels, merge source assertions, or create gold labels.

## Inputs

- the existing hierarchy JSONL manifest and its canonical Chest ImaGenome split;
- the source-preserving Chest ImaGenome atomic-annotation Parquet;
- the Chest ImaGenome comparison-relation Parquet and scene-graph archive;
- MIMIC-CXR-JPG metadata/split and CheXpert CSV files;
- the optional local MIMIC report archive, used only to verify locators;
- `exp/cxr_pt/configs/semantic_chain_mapping.v1.json`.

The mapping file is the only place where candidate semantics are defined. Rules
are marked `confirmed`, `proposed`, or `rejected`. The v1 candidates are all
proposed or rejected; none should be treated as validated training labels.

## Run

Use the `radzero` environment. A smoke run is explicitly marked as sampled:

```bash
conda run -n radzero python -m exp.cxr_pt.semantic_chain_audit \
  --manifest /path/to/hierarchy-v1.jsonl \
  --atomic-annotations /path/to/atomic_annotations.parquet \
  --comparison-relations /path/to/comparison_relations.parquet \
  --chexpert /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
  --metadata /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --mimic-split /path/to/mimic-cxr-2.0.0-split.csv.gz \
  --chest-imagenome-zip /path/to/chest-imagenome-dataset-1.0.0.zip \
  --scene-graph-zip /path/to/scene_graph.zip \
  --reports-zip /path/to/mimic-cxr-reports.zip \
  --image-root /path/to/mimic-cxr-jpg-2.1.0/files \
  --mapping exp/cxr_pt/configs/semantic_chain_mapping.v1.json \
  --output-dir /path/to/local/audit-output \
  --sample-per-split 200 --seed 20261008
```

Omit `--sample-per-split` for full train/validation statistics. Test rows are
counted only for structural split checks and never enter semantic statistics.
The report archive argument can be omitted; report fidelity review is then
recorded as unavailable without blocking the structured audit.

## Outputs

The output directory contains JSON provenance/configuration, CSV inventories
and contingency tables, and `audit_report.md`. `review_queue_local.csv` and
`historical_trace_local.csv` contain controlled identifiers and locators. Keep
them outside version control. No report text is written.

Run the synthetic tests with:

```bash
conda run -n radzero python -m unittest tests.test_semantic_chain_audit -v
```

Synthetic fixture results test software behavior only and are not data evidence.
