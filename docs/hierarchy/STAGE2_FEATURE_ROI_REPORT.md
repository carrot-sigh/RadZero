# Stage 2 Feature ROI Pilot

Validation date: 2026-09-30. The experiment used the `radzero` Conda
environment, the released `/home/user/gxq/radzero-model` checkpoint, and the
canonical hierarchy manifest. Generated JSON reports and PNG overlays are kept
outside Git under `/home/user/gxq/stage2-validation`.

## Implementation

The new `feature_roi` path is independent of the existing masked-attention path:

```text
37x37 frozen patch features
  -> continuous GT bbox coordinates
  -> torchvision ROIAlign (14x14, aligned=True)
  -> 128-channel bottleneck depthwise residual refiner
  -> local Finding VL matching
```

The model still returns `finding_attention`, `finding_evidence`,
`finding_logits`, and `pair_mask`. Stage 3 receives the same evidence contract
and was not modified. The default remains `masked_attention`.

New configuration fields:

```yaml
finding_roi_strategy: masked_attention
feature_roi_size: 14
feature_roi_context_margin: 0.0
finding_local_refiner_dim: 128
```

`full_refiner` is an explicit capacity-control condition. `full`,
`masked_attention`, `feature_roi`, and `full_refiner` are available from
`exp.cxr_pt.hierarchy_experiments stage2-pilot`. Feature ROI adds 216,875
trainable parameters, versus 43 in the original Finding Head and 5,315,345 in
the existing Stage 3 Aggregator+Fusion.

## Smoke test

On the fixed 16-image real training subset, GT Feature ROI trained for 200
updates with finite gradients:

| Metric | Initial | Final |
| --- | ---: | ---: |
| Finding loss | 0.571831 | 0.000374 |
| Positive-negative logit separation | 1.624391 | 16.065249 |
| Macro AUROC | 0.889450 | 1.000000 |
| Macro AUPRC | 0.834262 | 1.000000 |

This is an implementation smoke test, not a method comparison.

## Held-out pilot

All strategies used the same seed (`20260930`), 256 train images, 128 separate
validation images, 16-image cached batches, 120 updates, the same AdamW
learning rate (`1e-3`), validity masks, labels, and frozen RadZero checkpoint.
The validation subset was never optimized.

| Strategy | Valid loss | Macro AUROC | Macro AUPRC | Macro F1 | Trainable params | Peak VRAM (MB) | Step (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Image | 0.503896 | 0.753947 | 0.805636 | 0.748765 | 43 | 4283 | 0.00992 |
| Masked ROI | 0.551174 | 0.761089 | 0.795189 | 0.758083 | 43 | 4690 | 0.00929 |
| Feature ROI + Refiner | 0.895156 | **0.779609** | 0.781876 | 0.593896 | 216,875 | 4691 | 0.03767 |
| Full Image + Refiner | 0.889950 | 0.780468 | **0.820844** | 0.593102 | 216,875 | 4410 | 0.01855 |

The step time measures the cached feature/training step. Feature ROI is about
4x slower than the 43-parameter baselines and has similar peak memory in this
batch configuration. Validation inference throughput is recorded in each
`stage2_pilot.json` report.

The feature ROI shuffled-geometry sanity run used the same 256/128 split and
120 updates:

| Strategy | Valid loss | Macro AUROC | Macro AUPRC | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| GT Feature ROI | 0.895156 | 0.779609 | 0.781876 | 0.593896 |
| Shuffled Feature ROI | 0.978290 | 0.757563 | 0.771256 | 0.516743 |

GT geometry is better than shuffled geometry on all four summary metrics.

## Per-finding signal

Feature ROI improved AUROC over Full Image for several localized findings,
including lung opacity (`0.882` vs `0.854`), consolidation (`0.927` vs
`0.839`), pleural effusion (`0.920` vs `0.860`), pneumothorax (`0.927` vs
`0.894`), and pulmonary edema/hazy opacity (`0.862` vs `0.791`). It regressed
on airspace opacity (`0.385` vs `0.577`), mediastinal displacement (`0.754` vs
`0.829`), mediastinal widening (`0.797` vs `0.883`), and vascular congestion
(`0.653` vs `0.894`). Rare classes have high uncertainty; these are pilot
signals, not final claims.

## Visual validation

The pilot saved original CXR overlays with GT boxes, feature-grid coordinate
mapping, 14x14 local Finding attention, and labels for right upper lung, right
lower lung, left lower lung, and cardiac silhouette. They are under:

`/home/user/gxq/stage2-validation/pilot-feature-final/feature_roi/visualizations/`

The inspected overlays showed correct left/right placement, bbox scale, and no
invalid local attention values.

## Decision

1. Feature ROI is not a clear overall replacement: it has the best AUROC, but
   worse validation loss, AUPRC, and F1 than the 43-parameter baselines.
2. The capacity control is essential. Full Image + the same refiner has nearly
   the same AUROC and higher AUPRC than Feature ROI, so the pilot does not
   isolate a benefit uniquely attributable to anatomy-specific ROIAlign.
3. GT Feature ROI beats Shuffled Feature ROI, providing a positive spatial
   sanity signal rather than evidence of complete ROI irrelevance.
4. The extra compute and capacity are not justified for making Feature ROI the
   default Stage 2 path from this pilot. Keep it as an ablation and investigate
   class-balanced training, context margin, and validation calibration only in
   a subsequent controlled study.
