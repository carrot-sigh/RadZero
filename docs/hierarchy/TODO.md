# Implementation TODO

## P0 - Contracts and data

- [x] Freeze the anatomy vocabulary; use 36 bbox-supervised regions for V1.
- [x] Freeze the 41-label finding vocabulary and 36x41 compatibility matrix.
- [x] Define the exact 14 global labels and uncertain-label policy.
- [x] Add MIMIC-CXR CheXpert labels to the hierarchy manifest.
- [x] Compute finding/global positive-class weights from the training split only.
- [x] Compute anatomy positive-class weights after patch-mask rasterization.
- [x] Verify released checkpoint compatibility (only the loss-module prefix differs).
- [x] Add automatic `KeyPhraseAlignmentLoss` -> `RadZeroLoss` checkpoint migration.
- [x] Record and lock the working Conda environment.
- [x] Require and numerically verify the released RadZero checkpoint at startup.

## P1 - Dataset pipeline

- [x] Add a random-access hierarchy dataset backed by a generated JSONL manifest.
- [x] Return anatomy masks, finding targets, global targets, and validity masks.
- [x] Rasterize original-coordinate boxes consistently with direct image resize.
- [x] Disable horizontal flips for V1 to preserve left/right semantics.
- [x] Add dataset schema, split-integrity, and real-data coverage checks.

## P2 - Model

- [x] Refactor VL-CABS to return similarity maps, attention, evidence, and logits.
- [x] Add masked-softmax support with an empty-mask fallback.
- [x] Implement the anatomy ROI mask head.
- [x] Implement Anatomy-Masked VL-CABS for region-finding pairs.
- [x] Add finding-logit temperature and bias.
- [x] Implement finding evidence aggregation for each global label.
- [x] Implement the global VL-CABS branch and gated fusion.
- [x] Wire all hierarchy stages into one model wrapper and experiment config.
- [x] Preserve the original RadZero forward and zero-shot inference paths.
- [x] Share the Stage 1 anatomy similarity map between ROI and matching paths.
- [x] Reuse full-token original RadZero VL-CABS in the global branch.

## P3 - Training

- [x] Implement masked anatomy, finding, and global losses.
- [x] Add configurable class-imbalance handling after validity masking.
- [ ] Train the anatomy head first with the encoders frozen.
- [ ] Train finding recognition with GT ROI masks.
- [x] Add predicted-mask teacher-forcing scheduling and checkpoint state.
- [x] Rebuild optimizer and scheduler when switching hierarchy stages.
- [ ] Train Stage 3, then run low-learning-rate joint fine-tuning.

## P4 - Evaluation

- [x] Add anatomy IoU, Dice, and pointing-game metrics.
- [x] Integrate validity-aware anatomy metrics into Trainer validation.
- [x] Add validity-aware finding/global AUROC/AUPRC metrics.
- [x] Add validity-aware 14-label F1 metrics.
- [ ] Compare GT, predicted, shuffled, and full-image ROI masks.
- [ ] Compare finding-only, global-only, and fused predictions.
- [ ] Add evidence zero-out and shuffle interventions.
- [x] Add structured export for attention maps, finding scores, evidence usage, and gate values.

## P5 - Verification

- [x] Test bbox transforms and left/right semantics on four real 518x518 overlays.
- [x] Test masked softmax and empty ROIs for finite outputs.
- [x] Test that unknown labels do not contribute to losses.
- [x] Test Stage 1 -> Stage 2 -> Stage 3 tensor contracts.
- [x] Test gradient flow from global loss to the ROI head.
- [x] Run a deterministic synthetic tiny-overfit experiment.
- [x] Add a regression test for original RadZero logits and inference shapes.
- [x] Test that GT anatomy annotation presence cannot change inference outputs.
- [x] Test bounded temperatures under extreme log-temperature values.
- [x] Test full Trainer checkpoint resume (model, optimizer, scheduler, step, stage, RNG).
- [x] Test the formal hierarchy config loads RadZero and builds the Stage 1 optimizer.
- [x] Run a real single-image Stage 1 overfit with Dice/IoU and ROI visualizations.
- [x] Run a real 16-image Stage 2 GT-ROI tiny-overfit with finite-gradient audit.
- [x] Run a real 16-image Stage 3 tiny-overfit and verify frozen upstream state.
- [x] Run Stage 3 evidence, Finding-logit, and forced-gate interventions on the
  tiny-overfit set.
- [x] Run the controlled Stage 2 Feature ROI held-out pilot and capacity control.
- [x] Run predicted-ROI Stage 2 pilots using a held-out-trained Anatomy Head.
