# Stage 2/3 Real-Data Experiments

Validation date: 2026-09-30. Generated reports, checkpoints, and compressed
arrays are stored outside Git under `/home/user/gxq/stage2-validation` and
`/home/user/gxq/stage3-validation`.

## Reproducible entry point

Run the experiments in the recorded `radzero` Conda environment:

```bash
/home/user/miniconda3/envs/radzero/bin/python \
  -m exp.cxr_pt.hierarchy_experiments stage2-tiny-overfit \
  --num-images 16 --steps 500 --roi-mode gt \
  --output-dir /home/user/gxq/stage2-validation/roi-ablation-v2/gt-16x500
```

Stage 3 consumes the saved Stage 2 Finding Head:

```bash
/home/user/miniconda3/envs/radzero/bin/python \
  -m exp.cxr_pt.hierarchy_experiments stage3-tiny-overfit \
  --num-images 16 --steps 200 \
  --finding-checkpoint \
  /home/user/gxq/stage2-validation/roi-ablation-v2/gt-16x500/finding_head.pt \
  --output-dir \
  /home/user/gxq/stage3-validation/tiny-overfit-interventions-16x200
```

Both commands use seed `20260930` by default. They select real training images
with mixed positive and negative labels, cache frozen RadZero features, and
audit the exact trainable parameter set.

## Stage 2 GT-ROI tiny-overfit

The Finding Head trained on 16 images and 297 valid Anatomy-Finding pairs (120
positive, 177 negative). Its trainable set was exactly 43 parameters: two
temperatures and 41 finding biases.

| Metric | Initial | Final |
| --- | ---: | ---: |
| Finding loss | 0.617682 | 0.382294 |
| Positive-negative mean logit separation | 1.577603 | 2.802656 |
| Pair-level macro AUROC | 0.901334 | 0.891222 |
| Pair-level macro AUPRC | 0.846490 | 0.830627 |

The loss and mean separation improved with finite gradients. Ranking metrics did
not improve on this tiny training set. This verifies that the current head can
optimize its limited parameters; it is not evidence of generalization.

## Preliminary ROI diagnostic

GT, full-image, and sample-shuffled masks used identical samples, initialization,
seed, and 500-step optimization settings.

| ROI | Initial loss | Final loss | Final AUROC | Final AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Full image | 0.422943 | 0.274195 | 0.903448 | 0.858807 |
| GT ROI | 0.617682 | 0.382294 | 0.891222 | 0.830627 |
| Shuffled ROI | 0.598153 | 0.403680 | 0.844609 | 0.771304 |

Full-image evidence outperformed GT ROI on these 16 training images, while
shuffling the ROI was worst. This is only a tiny-overfit diagnostic. The full
four-way ablation remains incomplete because a reusable multi-image Stage 1
checkpoint is not yet available. Predicted ROI mode intentionally refuses to
run without `--anatomy-checkpoint`.

## Stage 3 tiny-overfit

## Deployment-aligned predicted-ROI chain

The deployment-aligned chain was also smoke-tested as:

```text
CXR -> Stage 1 predicted Anatomy ROI -> Stage 2 masked attention Finding
    -> Stage 3 Global fusion
```

Stage 2 and Stage 3 consumed predicted masks from the saved Anatomy Head; no
GT anatomy masks or GT anatomy boxes were passed to either Finding or Global
forward path. Finding and Global labels were retained only as supervised
training targets and validation metrics, since removing those would no longer
be a supervised training test.

| Stage | Images | Steps | Initial loss | Final loss | Finite | GT anatomy used |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| Stage 2 Finding | 8 | 2 | 0.604674 | 0.589864 | yes | no |
| Stage 3 Global | 8 | 2 | 0.502589 | 0.127840 | yes | no |

The Stage 3 report also records `roi_mode=predicted` and
`uses_gt_anatomy_masks=false`. The generated inference/intervention artifact
was written to `/home/user/gxq/stage3-validation/deployment-smoke`.

With Stage 1/2 outputs frozen, the 5,315,345 Aggregator and Fusion parameters
trained for 200 steps on 16 images with mixed global-label patterns.

| Metric | Initial | Final |
| --- | ---: | ---: |
| Global loss | 0.508452 | 0.000319 |
| Macro AUROC | 0.437500 | 1.000000 |
| Macro AUPRC | 0.580139 | 1.000000 |
| Mean fusion gate | 0.500000 | 0.492187 |

The Anatomy and Finding Head state dictionaries were exactly unchanged. This
confirms the Stage 3 training path can learn; the result is expected memorization
and does not establish validation performance.

## Evidence interventions

All interventions keep Global Branch evidence fixed. Values below are measured
on the same 16 training images after Stage 3 tiny-overfit.

| Intervention | Loss | AUROC | Mean absolute logit delta |
| --- | ---: | ---: | ---: |
| Normal fusion | 0.000319 | 1.000000 | 0.000000 |
| Shuffle Finding evidence | 1.022225 | 0.776389 | 2.805195 |
| Zero Finding evidence | 0.318055 | 1.000000 | 3.149287 |
| Shuffle Finding logits | 0.000693 | 1.000000 | 0.293761 |
| Finding-only (`g=1`) | 0.028918 | 1.000000 | 0.636196 |
| Global-only (`g=0`) | 0.041830 | 1.000000 | 3.033733 |

The memorized model is sensitive to Finding evidence tokens, especially their
sample correspondence, but is much less sensitive to shuffled Finding logits.
This establishes intervention plumbing and a positive tiny-overfit signal. The
same comparisons must be repeated on held-out validation data before drawing a
method conclusion.

`stage3_interventions.npz` contains normal and intervened logits, fusion gates,
Aggregator attention, and null attention for downstream analysis.
