# Architecture Decisions

Use this file as an append-only research decision log. Record the date, decision,
rationale, and affected experiments whenever a design choice is frozen.

## Current working decisions

- V1 excludes Anatomy Token Entanglement.
- Stage 2 uses masks before softmax.
- Unknown Chest ImaGenome findings are ignored rather than treated as negative.
- Stage 3 retains a global visual branch, so the model is an evidence-guided
  fusion model rather than a strict finding bottleneck.
- Training starts with GT anatomy masks and transitions to predicted masks.
- The original RadZero zero-shot interface remains available for regression and
  ablation experiments.

## Open decisions

- [ ] Exact anatomy vocabulary and whether to include regions without bbox.
- [ ] Exact global 14-label vocabulary.
- [ ] CheXpert uncertain-label policy.
- [ ] Direct patch-mask prediction versus auxiliary bbox regression.
- [ ] Static versus sample-dependent label-specific fusion gate.
- [ ] Prompt templates and prompt ensembling policy.
- [ ] Encoder freezing and unfreezing schedule.

## Decision log

| Date | Decision | Rationale | Experiments affected |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |
