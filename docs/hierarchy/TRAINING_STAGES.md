# Hierarchy Training Stages

The training controller owns parameter freezing and GT-to-predicted ROI
scheduling. Its
`stage_index` and `step` are persistent model buffers and therefore survive
checkpoint save/load.

| Stage | Trainable modules | Expected supervision |
| --- | --- | --- |
| `anatomy` | `anatomy_head` | Anatomy patch masks |
| `finding` | `finding_head` | Region-finding targets with scheduled GT-to-predicted ROI masks |
| `global` | `finding_aggregator`, `global_branch`, `evidence_fusion` | Global labels |
| `joint` | All hierarchy modules | All available targets |

Vision, text, and alignment encoders remain frozen in every stage by default.
Set `joint_train_encoders: true` only for a deliberate joint fine-tuning run.

## Runtime API

- `model.configure_hierarchy_training(stage)` changes the stage and reapplies
  deterministic parameter ownership before optimizer creation.
- `trainer.configure_hierarchy_stage(stage, num_training_steps)` is required
  when changing stages in the same process; it rebuilds optimizer and scheduler.
- `model.set_hierarchy_training_step(step)` records the optimizer global step.
- `model.get_teacher_forcing_ratio()` returns the current linear schedule value.
- `gt_anatomy_masks` enables scheduled GT ROI teacher forcing during training;
  the deployment-aligned endpoint is the predicted Anatomy ROI.
- `anatomy_mask_override` remains reserved for explicit experiments and cannot
  be combined with teacher forcing.

The primary Stage 2 route is `Stage 1 predicted ROI -> masked_attention`.
`feature_roi` and `full_refiner` are explicit ablation strategies and are not
the default production path.

The deployment smoke contract is therefore:
`CXR -> predicted Anatomy ROI -> Finding -> Global`. GT anatomy masks/boxes
may be used for teacher forcing or ablations, but must not be required by the
Finding or Global inference paths.

The trainer must set the hierarchy step from its optimizer global step, not from
micro-batch count. After loading a training checkpoint, call
`configure_hierarchy_training()` without a stage argument to reapply the stage
stored in the checkpoint buffer.

Do not call the model-level stage setter after optimizer creation. Newly
unfrozen parameters are not automatically added to an existing optimizer.
Frozen vision, text, and alignment encoders remain in eval mode during training.

## Formal schedule

The hierarchy entry point requires an explicit `training_schedule` in
`hierarchy_config`. It runs `anatomy`, `finding`, `global`, and `joint` in that
order. Every phase gets a fresh Trainer, optimizer, scheduler, and output
directory named after the stage. With `load_best_model_at_end`, the selected
phase checkpoint remains in memory as the starting model for the next phase.

To resume a staged run, set both `experiment.resume_from_checkpoint` and
`experiment.resume_stage`. Earlier phases are skipped and later phases continue
normally after the resumed phase completes.
