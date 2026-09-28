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
- Stage 3 retains a global visual branch, so the model is an evidence-guided
  fusion model rather than a strict finding bottleneck.
- Fusion uses a sample-and-label-specific gate initialized to equal weighting.
- Training starts with GT anatomy masks and transitions to predicted masks.
- The original RadZero zero-shot interface remains available for regression and
  ablation experiments.

## Open decisions

- [ ] Exact anatomy vocabulary and whether to include regions without bbox.
- [ ] Exact global 14-label vocabulary.
- [ ] CheXpert uncertain-label policy.
- [x] Direct patch-mask prediction versus auxiliary bbox regression.
- [x] Static versus sample-dependent label-specific fusion gate.
- [ ] Prompt templates and prompt ensembling policy.
- [ ] Encoder freezing and unfreezing schedule.

## Decision log

| Date | Decision | Rationale | Experiments affected |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |
