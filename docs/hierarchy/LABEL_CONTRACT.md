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

Model structure masks are applied in addition to annotation validity:

- `anatomy_query_mask` excludes unavailable anatomy queries.
- `visual_token_mask` excludes padded or invalid image patches.
- the anatomy-finding compatibility mask excludes invalid pairs.
- `global_query_mask` excludes unavailable global-label queries.

An all-invalid task or batch returns a finite, differentiable zero loss. Targets
may be omitted by stage; `return_loss=True` requires at least one target tensor.
