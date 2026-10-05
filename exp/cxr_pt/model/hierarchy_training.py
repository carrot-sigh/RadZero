import torch
import torch.nn as nn


class HierarchyTrainingController(nn.Module):
    STAGES = ("anatomy", "finding", "global", "joint")
    STAGE_MODULES = {
        "anatomy": ("anatomy_head",),
        "finding": ("finding_head",),
        "global": ("finding_aggregator", "global_branch", "evidence_fusion"),
        "joint": (
            "anatomy_head",
            "finding_head",
            "finding_aggregator",
            "global_branch",
            "evidence_fusion",
        ),
    }
    ENCODER_MODULES = ("vision_model", "text_model", "align_transformer")

    def __init__(
        self,
        initial_stage="anatomy",
        joint_train_encoders=False,
        teacher_forcing_start_ratio=1.0,
        teacher_forcing_end_ratio=0.0,
        teacher_forcing_start_step=0,
        teacher_forcing_end_step=10000,
    ):
        super().__init__()
        self._validate_stage(initial_stage)
        self._validate_ratio(teacher_forcing_start_ratio, "start ratio")
        self._validate_ratio(teacher_forcing_end_ratio, "end ratio")
        if teacher_forcing_start_step < 0:
            raise ValueError("teacher_forcing_start_step must be non-negative.")
        if teacher_forcing_end_step <= teacher_forcing_start_step:
            raise ValueError(
                "teacher_forcing_end_step must be greater than start_step."
            )

        self.joint_train_encoders = joint_train_encoders
        self.teacher_forcing_start_ratio = teacher_forcing_start_ratio
        self.teacher_forcing_end_ratio = teacher_forcing_end_ratio
        self.teacher_forcing_start_step = teacher_forcing_start_step
        self.teacher_forcing_end_step = teacher_forcing_end_step
        self.register_buffer(
            "stage_index",
            torch.tensor(self.STAGES.index(initial_stage), dtype=torch.long),
        )
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    @property
    def stage(self):
        return self.STAGES[int(self.stage_index.item())]

    def set_step(self, step):
        if not isinstance(step, int) or step < 0:
            raise ValueError("Hierarchy training step must be a non-negative integer.")
        self.step.fill_(step)

    def teacher_forcing_ratio(self, step=None):
        step = int(self.step.item()) if step is None else step
        if not isinstance(step, int) or step < 0:
            raise ValueError("Hierarchy training step must be a non-negative integer.")
        if step <= self.teacher_forcing_start_step:
            return self.teacher_forcing_start_ratio
        if step >= self.teacher_forcing_end_step:
            return self.teacher_forcing_end_ratio
        progress = (step - self.teacher_forcing_start_step) / (
            self.teacher_forcing_end_step - self.teacher_forcing_start_step
        )
        return self.teacher_forcing_start_ratio + progress * (
            self.teacher_forcing_end_ratio - self.teacher_forcing_start_ratio
        )

    def configure_model(self, model, stage=None):
        if stage is not None:
            self._validate_stage(stage)
            self.stage_index.fill_(self.STAGES.index(stage))
        stage = self.stage

        for parameter in model.parameters():
            parameter.requires_grad = False
        module_names = list(self.STAGE_MODULES[stage])
        if getattr(model, "global_only", False) and stage in {"global", "joint"}:
            module_names = [name for name in module_names if name != "finding_aggregator"]
        if stage == "joint" and self.joint_train_encoders:
            module_names.extend(self.ENCODER_MODULES)
        for module_name in module_names:
            module = getattr(model, module_name, None)
            if module is None:
                raise ValueError(
                    f"Model does not provide hierarchy module {module_name}."
                )
            for parameter in module.parameters():
                parameter.requires_grad = True
        return tuple(module_names)

    def select_anatomy_masks(
        self,
        predicted_masks,
        ground_truth_masks,
        training,
        ratio=None,
    ):
        if ground_truth_masks is None:
            if ratio is not None:
                raise ValueError(
                    "teacher_forcing_ratio requires ground-truth anatomy masks."
                )
            return predicted_masks
        if ground_truth_masks.shape != predicted_masks.shape:
            raise ValueError(
                "ground-truth anatomy masks must match predicted anatomy mask shape."
            )
        if not training:
            return predicted_masks

        ratio = self.teacher_forcing_ratio() if ratio is None else ratio
        self._validate_ratio(ratio, "teacher-forcing ratio")
        ground_truth_masks = ground_truth_masks.to(
            device=predicted_masks.device, dtype=predicted_masks.dtype
        )
        if not torch.isfinite(ground_truth_masks).all() or torch.any(
            (ground_truth_masks < 0) | (ground_truth_masks > 1)
        ):
            raise ValueError(
                "ground-truth anatomy masks must contain values in [0, 1]."
            )
        if ratio == 0:
            return predicted_masks
        if ratio == 1:
            return ground_truth_masks
        selection_shape = predicted_masks.shape[:2] + (1,)
        use_ground_truth = (
            torch.rand(selection_shape, device=predicted_masks.device) < ratio
        )
        return torch.where(use_ground_truth, ground_truth_masks, predicted_masks)

    @classmethod
    def _validate_stage(cls, stage):
        if stage not in cls.STAGES:
            raise ValueError(
                f"Unknown hierarchy training stage {stage!r}; expected {cls.STAGES}."
            )

    @staticmethod
    def _validate_ratio(value, name):
        if not 0 <= value <= 1:
            raise ValueError(f"Teacher-forcing {name} must be in [0, 1].")
