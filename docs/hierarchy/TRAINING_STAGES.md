# Hierarchy Training Stages

The training controller owns parameter freezing and GT ROI scheduling. Its
`stage_index` and `step` are persistent model buffers and therefore survive
checkpoint save/load.

| Stage | Trainable modules | Expected supervision |
| --- | --- | --- |
| `anatomy` | `anatomy_head` | Anatomy patch masks |
| `finding` | `finding_head` | Region-finding targets with GT ROI masks |
| `global` | `finding_aggregator`, `global_branch`, `evidence_fusion` | Global labels |
| `joint` | All hierarchy modules | All available targets |

Vision, text, and alignment encoders remain frozen in every stage by default.
Set `joint_train_encoders: true` only for a deliberate joint fine-tuning run.

## Runtime API

- `model.configure_hierarchy_training(stage)` changes the stage and reapplies
  deterministic parameter ownership.
- `model.set_hierarchy_training_step(step)` records the optimizer global step.
- `model.get_teacher_forcing_ratio()` returns the current linear schedule value.
- `gt_anatomy_masks` enables scheduled GT ROI use during training.
- `anatomy_mask_override` remains reserved for explicit experiments and cannot
  be combined with teacher forcing.

The trainer must set the hierarchy step from its optimizer global step, not from
micro-batch count. After loading a training checkpoint, call
`configure_hierarchy_training()` without a stage argument to reapply the stage
stored in the checkpoint buffer.
