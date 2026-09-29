# Hierarchy Label Contract

The hierarchy loss API separates target values from label validity. Dataset code
must convert source-specific unknown or uncertain values into explicit boolean
validity masks before calling the model.

## Tensors

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `anatomy_targets` | `[B, A, L]` | Binary patch-level anatomy masks |
| `anatomy_valid_mask` | `[B, A]` or `[B, A, L]` | Supervised regions or patches |
| `finding_targets` | `[B, A, F]` | Binary region-finding labels |
| `finding_valid_mask` | `[B, A, F]` | Annotated region-finding pairs |
| `global_targets` | `[B, J]` | Binary global labels |
| `global_valid_mask` | `[B, J]` | Global labels included in loss |

Valid target values must be in `[0, 1]`. Values outside that range, including
`-1` and `NaN`, are allowed only where the corresponding validity mask is false.
They never contribute to loss or gradients.

Optional BCE `pos_weight` values may be scalar or per class (`A`, `F`, or `J`).
They are applied after validity masking. Final values must be estimated from the
frozen training split only; validation and test labels must not contribute.
Finding and global suggestions use Laplace-smoothed prevalence weights,
`(negative + 1) / (positive + 1)`. Anatomy weights are deferred until boxes are
rasterized at the model's patch resolution. Both finding and global statistics
use the official Chest ImaGenome training split; global rows are deduplicated by
MIMIC study.

The frozen V1 vocabularies, ordering, counts, policies, compatibility matrix,
and suggested weights live in
`exp/cxr_pt/configs/hierarchy_contract.json`. Region findings are valid only for
non-conflicting explicit Chest ImaGenome `yes`/`no` annotations. Missing and
conflicting annotations are unknown. Global CheXpert `1.0` and `0.0` are valid,
while `-1.0` and blank are unknown. A blank `No Finding` is a valid negative
only when another global label is explicitly positive for the study.

Model structure masks are applied in addition to annotation validity:

- `anatomy_query_mask` excludes unavailable anatomy queries.
- `visual_token_mask` excludes padded or invalid image patches.
- the anatomy-finding compatibility mask excludes invalid pairs.
- `global_query_mask` excludes unavailable global-label queries.

An all-invalid task or batch returns a finite, differentiable zero loss. Targets
may be omitted by stage; `return_loss=True` requires at least one target tensor.

`gt_anatomy_masks` is a model input for teacher forcing, not a loss target. It
must always contain finite values in `[0, 1]`; unavailable anatomy regions use a
zero mask and are excluded with `anatomy_query_mask`.
