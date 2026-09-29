# Hierarchy Execution Plan

This is the ordered implementation path. Complete and verify one stage before
starting the next one.

## 1. Masked class balancing

- [x] Support optional positive-class weights for anatomy, finding, and global
  BCE losses.
- [x] Apply weights only after validity and structural masks are resolved.
- [x] Test scalar and per-class weights with unknown labels.

## 2. Training-stage control

- [x] Define anatomy, finding-with-GT-ROI, global, and joint training phases.
- [x] Freeze and unfreeze modules deterministically for each phase.
- [x] Add scheduled GT-to-predicted ROI teacher forcing.
- [x] Save the phase and schedule state in checkpoints.

## 3. Label contracts and vocabularies

- [x] Freeze the 36 anatomy and 41 region-finding vocabularies.
- [x] Freeze the 36x41 anatomy-finding compatibility matrix.
- [x] Freeze the 14 global labels and uncertain-label policy.
- [x] Compute finding/global positive-class weights from the training split only.
- [ ] Compute anatomy positive-class weights after patch-mask rasterization.

The versioned source of truth is
`exp/cxr_pt/configs/hierarchy_contract.json`. Anatomy weights remain part of the
data-pipeline stage because they depend on the final patch-mask resolution.

## 4. Hierarchy data pipeline

- [x] Define and validate the hierarchy manifest schema.
- [x] Load anatomy boxes, region findings, global labels, and validity masks.
- [x] Implement direct-resize-consistent patch-mask rasterization.
- [x] Disable horizontal flips for V1 to preserve left/right semantics.
- [x] Build a collator with fixed prompt banks and compatibility masks.

## 5. Trainer integration

- [x] Route hierarchy batches through `forward_hierarchy`.
- [x] Log each component loss, stage index, and teacher-forcing ratio.
- [x] Run one-batch forward/backward gradient-flow tests.
- [ ] Run full Trainer checkpoint-resume tests.
- [ ] Run a deterministic tiny-overfit experiment.

## 6. Evaluation

- [ ] Add anatomy IoU, Dice, and pointing-game metrics.
- [ ] Add finding AUROC/AUPRC metrics with validity masks.
- [ ] Add global AUROC/AUPRC/F1 metrics with the frozen uncertain policy.
- [ ] Export attention maps, finding scores, gates, and evidence usage.

## 7. Ablations and final training

- [ ] Compare GT, predicted, shuffled, and full-image ROI masks.
- [ ] Compare finding-only, global-only, and fused branches.
- [ ] Add evidence zero-out and shuffle interventions.
- [ ] Run staged training followed by low-learning-rate joint fine-tuning.
