# Implementation TODO

## P0 - Contracts and data

- [ ] Freeze the anatomy vocabulary; use bbox-supervised regions for V1.
- [ ] Freeze the finding vocabulary and anatomy-finding compatibility matrix.
- [ ] Define the exact 14 global labels and uncertain-label policy.
- [ ] Add MIMIC-CXR CheXpert labels to the hierarchy manifests.
- [x] Verify released checkpoint compatibility (only the loss-module prefix differs).
- [ ] Add automatic `KeyPhraseAlignmentLoss` -> `RadZeroLoss` checkpoint migration.
- [ ] Record and lock the working Conda environment.

## P1 - Dataset pipeline

- [ ] Add a hierarchy dataset backed by the existing Parquet manifests.
- [ ] Return anatomy boxes, finding targets, global targets, and validity masks.
- [ ] Implement joint image/bbox transforms.
- [ ] Handle left/right label swaps during horizontal flips, or disable flips.
- [ ] Add dataset schema and split-integrity checks.

## P2 - Model

- [x] Refactor VL-CABS to return similarity maps, attention, evidence, and logits.
- [x] Add masked-softmax support with an empty-mask fallback.
- [x] Implement the anatomy ROI mask head.
- [x] Implement Anatomy-Masked VL-CABS for region-finding pairs.
- [x] Add finding-logit temperature and bias.
- [x] Implement finding evidence aggregation for each global label.
- [x] Implement the global VL-CABS branch and gated fusion.
- [ ] Preserve the original RadZero forward and zero-shot inference paths.

## P3 - Training

- [ ] Implement masked anatomy, finding, and global losses.
- [ ] Add class-imbalance handling without treating unknown labels as negatives.
- [ ] Train the anatomy head first with the encoders frozen.
- [ ] Train finding recognition with GT ROI masks.
- [ ] Introduce predicted masks with scheduled teacher forcing.
- [ ] Train Stage 3, then run low-learning-rate joint fine-tuning.

## P4 - Evaluation

- [ ] Add anatomy IoU, Dice, and pointing-game metrics.
- [ ] Add finding-level AUROC/AUPRC metrics.
- [ ] Add 14-label AUROC/AUPRC/F1 metrics.
- [ ] Compare GT, predicted, shuffled, and full-image ROI masks.
- [ ] Compare finding-only, global-only, and fused predictions.
- [ ] Add evidence zero-out and shuffle interventions.
- [ ] Save attention maps, finding scores, evidence usage, and gate values.

## P5 - Verification

- [ ] Test bbox transforms and left/right semantics.
- [x] Test masked softmax and empty ROIs for finite outputs.
- [ ] Test that unknown labels do not contribute to losses.
- [x] Test Stage 1 -> Stage 2 -> Stage 3 tensor contracts.
- [x] Test gradient flow from global loss to the ROI head.
- [ ] Run a tiny-overfit experiment.
- [x] Add a regression test for original RadZero logits and inference shapes.
