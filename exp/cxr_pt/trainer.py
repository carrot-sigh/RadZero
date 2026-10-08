import json
import os
import random
from typing import Optional

import numpy as np
import torch
from transformers.trainer import unwrap_model
from transformers.training_args import ParallelMode

from common.trainer import BaseTrainer, get_last_checkpoint, logger
from exp.cxr_pt.hierarchy_metrics import (
    compute_hierarchy_trainer_metrics,
    export_hierarchy_output,
)

HIERARCHY_STAGE_ORDER = ("anatomy", "finding", "global", "joint")


def validate_hierarchy_schedule(schedule, allow_subset=False):
    """Validate an explicit, ordered Anatomy -> Finding -> Global -> Joint plan."""
    if not isinstance(schedule, list) or (not allow_subset and len(schedule) != len(HIERARCHY_STAGE_ORDER)):
        raise ValueError(
            "hierarchy training_schedule must define anatomy, finding, global, "
            "and joint phases."
        )
    stages = tuple(phase.get("stage") for phase in schedule)
    if allow_subset:
        if not stages or any(stage not in HIERARCHY_STAGE_ORDER for stage in stages):
            raise ValueError(f"Unknown hierarchy training stages: {stages}.")
        if tuple(sorted(stages, key=HIERARCHY_STAGE_ORDER.index)) != stages or len(set(stages)) != len(stages):
            raise ValueError(f"Hierarchy training stages must be ordered and unique, got {stages}.")
    elif stages != HIERARCHY_STAGE_ORDER:
        raise ValueError(
            f"hierarchy training_schedule must follow {HIERARCHY_STAGE_ORDER}, "
            f"got {stages}."
        )
    normalized = []
    for phase in schedule:
        epochs = phase.get("num_train_epochs")
        max_steps = phase.get("max_steps")
        if (
            not isinstance(epochs, (int, float))
            or isinstance(epochs, bool)
            or epochs <= 0
        ):
            raise ValueError(
                f"Hierarchy phase {phase['stage']!r} requires positive "
                "num_train_epochs."
            )
        if max_steps is not None and (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise ValueError(
                f"Hierarchy phase {phase['stage']!r} max_steps must be a positive integer."
            )
        normalized.append(
            {
                "stage": phase["stage"],
                "num_train_epochs": float(epochs),
                "max_steps": max_steps,
            }
        )
    return normalized


class CXRPreTrainer(BaseTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        cfg=None,
        **kwargs,
    ):
        if (
            compute_metrics is None
            and model is not None
            and hasattr(model, "forward_hierarchy")
        ):
            compute_metrics = self._compute_hierarchy_metrics
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            cfg=cfg,
            **kwargs,
        )

    def _compute_hierarchy_metrics(self, eval_prediction):
        controller = getattr(self.model, "training_controller", None)
        stage = controller.stage if controller is not None else "joint"
        return compute_hierarchy_trainer_metrics(eval_prediction, stage=stage)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        hierarchy_model = unwrap_model(model if model is not None else self.model)
        if hasattr(hierarchy_model, "configure_hierarchy_training"):
            modules = hierarchy_model.configure_hierarchy_training()
            logger.info(
                "Restored hierarchy stage %s; trainable modules: %s",
                hierarchy_model.training_controller.stage,
                ", ".join(modules),
            )

    def _load_rng_state(self, checkpoint):
        """Load trusted Trainer RNG checkpoints under PyTorch 2.6+ defaults."""
        if checkpoint is None:
            return
        if self.args.world_size > 1:
            rng_file = os.path.join(
                checkpoint, f"rng_state_{self.args.process_index}.pth"
            )
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            logger.info(
                "No RNG state found at %s; exact resume is not guaranteed.", rng_file
            )
            return
        try:
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        except TypeError:
            checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available() and "cuda" in checkpoint_rng_state:
            try:
                if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
                else:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
            except Exception as error:
                logger.info("Could not restore CUDA RNG state: %s", error)

    def configure_hierarchy_stage(self, stage, num_training_steps):
        """Switch stages between train runs and rebuild optimization state."""
        if getattr(self, "is_in_train", False):
            raise RuntimeError(
                "Hierarchy stages cannot change inside an active train loop."
            )
        if not hasattr(self.model, "configure_hierarchy_training"):
            raise ValueError("Trainer model does not support hierarchy stages.")
        modules = self.model.configure_hierarchy_training(stage)
        self.optimizer = None
        self.lr_scheduler = None
        self.create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        if hasattr(self, "callback_handler"):
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
        logger.info(
            "Hierarchy stage switched to %s; rebuilt optimizer/scheduler for: %s",
            stage,
            ", ".join(modules),
        )
        return modules

    def compute_loss(self, model, inputs, return_outputs=False):
        if "encoded_anatomy_prompts" in inputs:
            hierarchy_model = unwrap_model(model)
            if hasattr(hierarchy_model, "set_hierarchy_training_step"):
                hierarchy_model.set_hierarchy_training_step(int(self.state.global_step))
            hierarchy_inputs = dict(inputs)
            hierarchy_inputs.pop("sample_metadata", None)
            controller = getattr(hierarchy_model, "training_controller", None)
            stage = controller.stage if controller is not None else "joint"
            if stage == "anatomy":
                hierarchy_inputs["finding_targets"] = None
                hierarchy_inputs["global_targets"] = None
            elif stage == "finding":
                hierarchy_inputs["anatomy_targets"] = None
                hierarchy_inputs["global_targets"] = None
            elif stage == "global":
                hierarchy_inputs["anatomy_targets"] = None
                hierarchy_inputs["finding_targets"] = None
            hierarchy_output = model(
                return_loss=True,
                **hierarchy_inputs,
            )
            losses = dict(hierarchy_output.losses)
            if hasattr(hierarchy_model, "get_teacher_forcing_ratio"):
                losses["teacher_forcing_ratio"] = hierarchy_output.logits.new_tensor(
                    hierarchy_model.get_teacher_forcing_ratio()
                )
            controller = getattr(hierarchy_model, "training_controller", None)
            if controller is not None:
                losses["hierarchy_stage_index"] = hierarchy_output.logits.new_tensor(
                    float(controller.stage_index.item())
                )
            outputs = {"logits": hierarchy_output.logits, "losses": losses}
            loss = losses["loss"]
            return (loss, outputs) if return_outputs else loss

        outputs = model(**inputs)
        losses = outputs.pop("losses")
        loss = losses.pop("loss")

        outputs = dict(outputs)
        outputs["losses"] = losses

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "encoded_anatomy_prompts" not in inputs:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        inputs = self._prepare_inputs(inputs)
        inference_inputs = dict(inputs)
        inference_inputs.pop("sample_metadata", None)
        inference_inputs.pop("gt_anatomy_masks", None)
        controller = getattr(unwrap_model(model), "training_controller", None)
        stage = controller.stage if controller is not None else "joint"
        supervision = self._hierarchy_supervision(inputs)
        stage_tasks = {
            "anatomy": {"anatomy"},
            "finding": {"finding"},
            "global": {"global"},
            "joint": {"anatomy", "finding", "global"},
        }[stage]
        for task in ("anatomy", "finding", "global"):
            if task not in stage_tasks or not supervision[task]:
                inference_inputs[f"{task}_targets"] = None
        return_loss = any(supervision[task] for task in stage_tasks)

        with torch.no_grad(), self.compute_loss_context_manager():
            output = model(return_loss=return_loss, **inference_inputs)
        loss = None
        losses = {}
        if return_loss:
            loss = output.losses["loss"].mean().detach()
            losses = {
                key: value.mean().detach() for key, value in output.losses.items()
            }
        if prediction_loss_only:
            return loss, None, None, losses

        predictions = (
            output.anatomy.roi_masks.detach().float(),
            output.finding.finding_logits.detach().float(),
            output.fusion.logits.detach().float(),
        )
        if not any(supervision.values()):
            return loss, predictions, None, losses

        anatomy_valid = inputs.get(
            "anatomy_annotation_mask", inputs.get("anatomy_valid_mask")
        )
        labels = (
            (
                inputs["anatomy_targets"].detach()
                if supervision["anatomy"]
                else torch.zeros_like(output.anatomy.roi_masks)
            ),
            (
                inputs["finding_targets"].detach()
                if supervision["finding"]
                else torch.zeros_like(output.finding.finding_logits)
            ),
            (
                inputs["global_targets"].detach()
                if supervision["global"]
                else torch.zeros_like(output.fusion.logits)
            ),
            (
                anatomy_valid.detach()
                if supervision["anatomy"]
                else torch.zeros(
                    output.anatomy.roi_masks.shape[:2],
                    dtype=torch.bool,
                    device=output.anatomy.roi_masks.device,
                )
            ),
            (
                inputs["finding_valid_mask"].detach()
                if supervision["finding"]
                else torch.zeros_like(output.finding.finding_logits, dtype=torch.bool)
            ),
            (
                inputs["global_valid_mask"].detach()
                if supervision["global"]
                else torch.zeros_like(output.fusion.logits, dtype=torch.bool)
            ),
        )
        return loss, predictions, labels, losses

    @staticmethod
    def _hierarchy_supervision(inputs):
        anatomy_valid = inputs.get(
            "anatomy_annotation_mask", inputs.get("anatomy_valid_mask")
        )
        return {
            "anatomy": inputs.get("anatomy_targets") is not None
            and anatomy_valid is not None,
            "finding": inputs.get("finding_targets") is not None
            and inputs.get("finding_valid_mask") is not None,
            "global": inputs.get("global_targets") is not None
            and inputs.get("global_valid_mask") is not None,
        }

    def inference_step(self, model, inputs, ignore_keys, **kwargs):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            hierarchy_model = unwrap_model(model)
            if hasattr(hierarchy_model, "config"):
                ignore_keys = getattr(
                    hierarchy_model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []
        if "encoded_anatomy_prompts" not in inputs:
            return {}

        inference_inputs = dict(inputs)
        for key in (
            "gt_anatomy_masks",
            "anatomy_targets",
            "finding_targets",
            "global_targets",
            "anatomy_annotation_mask",
            "anatomy_valid_mask",
            "finding_valid_mask",
            "global_valid_mask",
        ):
            inference_inputs.pop(key, None)
        with torch.no_grad(), self.compute_loss_context_manager():
            output = model(return_loss=False, **inference_inputs)
        outputs = export_hierarchy_output(output)
        outputs.update(
            {
                "anatomy_roi_logits": output.anatomy.roi_mask_logits.detach(),
                "finding_evidence": output.finding.evidence_tokens.detach(),
                "finding_matching_scores": output.finding.matching_scores.detach(),
                "aggregated_finding_evidence": (
                    output.finding_aggregation.evidence_tokens.detach()
                ),
                "global_evidence": output.global_evidence.evidence_tokens.detach(),
            }
        )

        # remove outputs to ignore
        for k in ignore_keys:
            if k in outputs:
                outputs.pop(k)

        return outputs

    def save_inference_outputs(
        self,
        outputs,
        output_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        **kwargs,
    ):
        if output_dir is None:
            output_dir = os.path.join(self.args.output_dir, "inference")
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_name = checkpoint_name or f"checkpoint-{self.state.global_step}"
        arrays = {
            name: value
            for name, value in outputs.predictions.items()
            if value is not None
        }
        tensor_path = os.path.join(output_dir, f"{checkpoint_name}-hierarchy.npz")
        manifest_path = os.path.join(output_dir, f"{checkpoint_name}-hierarchy.json")
        np.savez_compressed(tensor_path, **arrays)
        manifest = {
            "checkpoint": checkpoint_name,
            "num_samples": outputs.num_samples,
            "tensor_file": os.path.basename(tensor_path),
            "outputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            "metrics": outputs.metrics,
        }
        with open(manifest_path, "w", encoding="utf-8") as destination:
            json.dump(manifest, destination, indent=2)
        return {
            "tensor_path": tensor_path,
            "manifest_path": manifest_path,
            "manifest": manifest,
        }

    def inference_load_from_checkpoint(self, resume_from_checkpoint=None):

        # skip resume_from_checkpoint after train (self._load_best_model())
        if self.args.load_best_model_at_end and (
            resume_from_checkpoint == self.state.best_model_checkpoint
        ):
            logger.info(
                "Since best model has already been loaded, skip resume from checkpoint"
            )
            pass

        else:
            if resume_from_checkpoint is False:
                resume_from_checkpoint = None

            # Load potential model checkpoint
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(
                        f"No valid checkpoint found in output directory ({self.args.output_dir})"
                    )

            if resume_from_checkpoint is not None:
                self._load_from_checkpoint(resume_from_checkpoint)

        return resume_from_checkpoint
