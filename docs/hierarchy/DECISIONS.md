# Architecture Decisions

Use this file as an append-only research decision log. Record the date, decision,
rationale, and affected experiments whenever a design choice is frozen.

## Current working decisions

- V1 excludes Anatomy Token Entanglement.
- V1 predicts differentiable patch-level anatomy masks directly; auxiliary bbox
  regression remains optional for a later ablation.
- Stage 2 uses masks before softmax.
- Soft masks act as log-space attention priors; zero-valued positions receive no
  attention.
- An empty ROI falls back to full-image attention by default, with a strict error
  mode available for validation.
- Unknown Chest ImaGenome findings are ignored rather than treated as negative.
- All hierarchy targets use separate boolean validity masks; source-specific
  unknown and uncertain values are normalized by the dataset layer.
- Stage 3 retains a global visual branch, so the model is an evidence-guided
  fusion model rather than a strict finding bottleneck.
- Fusion uses a sample-and-label-specific gate initialized to equal weighting.
- Training starts with GT anatomy masks and transitions to predicted masks.
- Training uses explicit anatomy, finding, global, and joint stages; encoders are
  frozen by default, including during joint hierarchy training.
- The original RadZero zero-shot interface remains available for regression and
  ablation experiments.
- The hierarchy model subclasses RadZero and exposes `forward_hierarchy`; it does
  not replace the original `forward` or `compute_logits` methods.
- V1 uses the 36 official bbox-detectable Chest ImaGenome anatomy regions.
- V1 uses 41 explicitly supervised region findings; `rib fracture` and
  `subcutaneous air` are excluded from the regional vocabulary.
- Region-finding compatibility is learned from non-conflicting explicit labels
  in the Chest ImaGenome training split only.
- Global targets use the 14 CheXpert labels. Values `-1.0` and blank are ignored;
  blank `No Finding` becomes negative only when another label is positive.
- The official Chest ImaGenome split is canonical for every hierarchy task;
  MIMIC split assignments are audit metadata only.
- Anatomy ROI prediction and anatomy matching branch from one shared cosine
  similarity map. The ROI head refines that map; it does not compute a second
  query-token similarity map.
- The Stage 3 global visual branch reuses original RadZero VL-CABS inputs:
  full aligned vision tokens including CLS, RadZero layer norm, and attention
  temperature.
- GT anatomy annotation availability controls only loss and metrics. All fixed
  anatomy queries remain available during train, validation, and test.

## Open decisions

- [x] Exact anatomy vocabulary and whether to include regions without bbox.
- [x] Exact global 14-label vocabulary.
- [x] CheXpert uncertain-label policy.
- [x] Direct patch-mask prediction versus auxiliary bbox regression.
- [x] Static versus sample-dependent label-specific fusion gate.
- [ ] Prompt templates and prompt ensembling policy.
- [x] Encoder freezing and unfreezing schedule.

## Decision log

| Date | Decision | Rationale | Experiments affected |
| --- | --- | --- | --- |
| 2026-09-28 | Use explicit validity masks for all hierarchy targets. | Keeps unknown-label handling independent of source encodings and prevents false negatives. | All hierarchy training stages |
| 2026-09-28 | Freeze the V1 labels and compatibility matrix from official training splits. | Prevents vocabulary drift and validation/test leakage. | Data, training, and evaluation |
| 2026-09-29 | Separate anatomy query availability from annotation availability. | Prevents GT bbox-presence leakage into validation and test predictions. | All hierarchy stages and evaluation |
| 2026-09-29 | Share the Stage 1 anatomy similarity map and reuse full-token RadZero for the Stage 3 global branch. | Aligns implementation with the frozen research design and preserves a faithful global reference path. | Stage 1 and Stage 3 |
