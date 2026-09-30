# Stage 1 Readiness

Validation date: 2026-09-29. Generated images and JSON results are kept outside
Git under `/home/user/gxq/stage1-validation`.

## Spatial validation

Four validation images were selected with seed `20260929` from 23,951 records.
Each had all 36 anatomy boxes. Contact sheets overlay the 37x37 rasterized masks
on the exact normalized-then-denormalized 518x518 processor input.

- `gt-overlays/valid-009523.png`
- `gt-overlays/valid-012538.png`
- `gt-overlays/valid-016966.png`
- `gt-overlays/valid-021810.png`

Visual inspection confirmed radiological left/right orientation, superior and
inferior location, and expected coverage for lungs, zones, hila, mediastinum,
cardiac silhouette, diaphragms, costophrenic angles, clavicles, and aorta. No
crop, flip, or resize-coordinate offset was observed.

## Formal startup and validation

The unmodified hierarchy config loaded `/home/user/gxq/radzero-model` and
numerically verified a released RadZero tensor. The initial stage was
`anatomy`; the optimizer contained exactly the 13,104 Anatomy Head parameters.
Vision, text, and alignment encoders had no trainable parameters and remained
in eval mode.

Standard `trainer.evaluate()` on the four real validation images reported:

| Metric | Value |
| --- | ---: |
| Anatomy Dice | 0.305221 |
| Anatomy IoU | 0.196737 |
| Anatomy pointing accuracy | 0.562500 |
| Anatomy loss | 1.950116 |

These values are a startup baseline from an untrained Anatomy Head, not model
quality estimates. Full results are in
`formal-startup-and-real-validation.json`.

## Checkpoint resume

An end-to-end Trainer test saved at step 2 and resumed in a fresh model and
Trainer through step 4. It verifies model weights, optimizer state and parameter
ownership, scheduler epoch, global step, hierarchy stage, and RNG restoration.

## Real tiny-overfit

One training image with all 36 annotations was optimized for 1,000 steps using
cached frozen-encoder features and the Anatomy Head only.

| Metric | Initial | Final |
| --- | ---: | ---: |
| Loss | 1.673972 | 0.013954 |
| Dice | 0.341485 | 0.999050 |
| IoU | 0.217844 | 0.998113 |
| Predicted positive fraction | 0.152321 | 0.084023 |

The final prediction matches the target boxes instead of expanding to a
full-image mask. See `tiny-overfit/gt.png`, `prediction-initial.png`,
`prediction-final.png`, and `metrics.json`.
