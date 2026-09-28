from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierarchyLossOutput:
    loss: torch.Tensor
    anatomy_loss: torch.Tensor = None
    anatomy_bce_loss: torch.Tensor = None
    anatomy_dice_loss: torch.Tensor = None
    finding_loss: torch.Tensor = None
    global_loss: torch.Tensor = None

    def as_dict(self):
        return {name: value for name, value in vars(self).items() if value is not None}


class HierarchyLoss(nn.Module):
    """Compute stage losses while excluding unknown or structurally invalid labels."""

    def __init__(
        self,
        anatomy_weight=1.0,
        finding_weight=1.0,
        global_weight=1.0,
        anatomy_bce_weight=1.0,
        anatomy_dice_weight=1.0,
        dice_smooth=1.0,
    ):
        super().__init__()
        weights = {
            "anatomy_weight": anatomy_weight,
            "finding_weight": finding_weight,
            "global_weight": global_weight,
            "anatomy_bce_weight": anatomy_bce_weight,
            "anatomy_dice_weight": anatomy_dice_weight,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if dice_smooth <= 0:
            raise ValueError("dice_smooth must be positive.")

        self.anatomy_weight = anatomy_weight
        self.finding_weight = finding_weight
        self.global_weight = global_weight
        self.anatomy_bce_weight = anatomy_bce_weight
        self.anatomy_dice_weight = anatomy_dice_weight
        self.dice_smooth = dice_smooth

    def forward(
        self,
        anatomy_logits,
        finding_logits,
        global_logits,
        anatomy_targets=None,
        finding_targets=None,
        global_targets=None,
        anatomy_valid_mask=None,
        finding_valid_mask=None,
        global_valid_mask=None,
        anatomy_query_mask=None,
        visual_token_mask=None,
        finding_pair_mask=None,
        global_query_mask=None,
    ):
        if all(
            targets is None
            for targets in (anatomy_targets, finding_targets, global_targets)
        ):
            raise ValueError("At least one hierarchy target is required for loss.")

        total_loss = None
        output = {}
        if anatomy_targets is not None:
            valid_mask = self._anatomy_valid_mask(
                anatomy_logits,
                anatomy_valid_mask,
                anatomy_query_mask,
                visual_token_mask,
            )
            anatomy_bce = self._masked_bce_with_logits(
                anatomy_logits, anatomy_targets, valid_mask, "anatomy"
            )
            anatomy_dice = self._masked_dice_loss(
                anatomy_logits, anatomy_targets, valid_mask, "anatomy"
            )
            anatomy_loss = (
                self.anatomy_bce_weight * anatomy_bce
                + self.anatomy_dice_weight * anatomy_dice
            )
            output.update(
                anatomy_loss=anatomy_loss,
                anatomy_bce_loss=anatomy_bce,
                anatomy_dice_loss=anatomy_dice,
            )
            total_loss = self.anatomy_weight * anatomy_loss

        if finding_targets is not None:
            valid_mask = self._classification_valid_mask(
                finding_logits,
                finding_valid_mask,
                finding_pair_mask,
                "finding",
            )
            finding_loss = self._masked_bce_with_logits(
                finding_logits, finding_targets, valid_mask, "finding"
            )
            output["finding_loss"] = finding_loss
            weighted = self.finding_weight * finding_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        if global_targets is not None:
            query_mask = self._query_mask(
                global_query_mask,
                global_logits.shape,
                "global_query_mask",
                global_logits.device,
            )
            valid_mask = self._classification_valid_mask(
                global_logits,
                global_valid_mask,
                query_mask,
                "global",
            )
            global_loss = self._masked_bce_with_logits(
                global_logits, global_targets, valid_mask, "global"
            )
            output["global_loss"] = global_loss
            weighted = self.global_weight * global_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        return HierarchyLossOutput(loss=total_loss, **output)

    def _anatomy_valid_mask(self, logits, annotation_mask, query_mask, token_mask):
        self._require_shape(logits, 3, "anatomy_logits")
        valid = self._anatomy_annotation_mask(
            annotation_mask, logits.shape, logits.device
        )
        if query_mask is not None:
            query_mask = self._query_mask(
                query_mask,
                logits.shape[:2],
                "anatomy_query_mask",
                logits.device,
            ).unsqueeze(-1)
            valid = valid & query_mask
        if token_mask is not None:
            token_mask = self._token_mask(
                token_mask, (logits.shape[0], logits.shape[2]), logits.device
            ).unsqueeze(1)
            valid = valid & token_mask
        return valid

    @staticmethod
    def _anatomy_annotation_mask(mask, shape, device):
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if tuple(mask.shape) == tuple(shape[:2]):
            return mask.to(device=device, dtype=torch.bool).unsqueeze(-1).expand(shape)
        if tuple(mask.shape) == tuple(shape):
            return mask.to(device=device, dtype=torch.bool)
        raise ValueError(
            "anatomy_valid_mask must have shape "
            f"{tuple(shape[:2])} or {tuple(shape)}."
        )

    def _classification_valid_mask(
        self, logits, annotation_mask, structural_mask, name
    ):
        valid = self._exact_mask(
            annotation_mask, logits.shape, f"{name}_valid_mask", logits.device
        )
        if structural_mask is not None:
            valid = valid & self._exact_mask(
                structural_mask,
                logits.shape,
                f"{name}_structural_mask",
                logits.device,
            )
        return valid

    @staticmethod
    def _masked_bce_with_logits(logits, targets, valid_mask, name):
        targets = HierarchyLoss._prepare_targets(targets, logits, valid_mask, name)
        if not valid_mask.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits[valid_mask].float(), targets[valid_mask].float()
        )

    def _masked_dice_loss(self, logits, targets, valid_mask, name):
        targets = self._prepare_targets(targets, logits, valid_mask, name)
        flat_valid = valid_mask.reshape(-1, valid_mask.size(-1))
        valid_rows = flat_valid.any(dim=-1)
        if not valid_rows.any():
            return logits.sum() * 0.0

        probabilities = torch.sigmoid(logits.float()).reshape(flat_valid.shape)
        flat_targets = targets.float().reshape(flat_valid.shape)
        probabilities = probabilities[valid_rows]
        flat_targets = flat_targets[valid_rows]
        flat_valid = flat_valid[valid_rows]
        probabilities = torch.where(flat_valid, probabilities, 0.0)
        flat_targets = torch.where(flat_valid, flat_targets, 0.0)
        intersection = (probabilities * flat_targets).sum(dim=-1)
        denominator = probabilities.sum(dim=-1) + flat_targets.sum(dim=-1)
        dice = (2.0 * intersection + self.dice_smooth) / (
            denominator + self.dice_smooth
        )
        return (1.0 - dice).mean()

    @staticmethod
    def _prepare_targets(targets, logits, valid_mask, name):
        if targets.shape != logits.shape:
            raise ValueError(
                f"{name}_targets must have shape {tuple(logits.shape)}, "
                f"got {tuple(targets.shape)}."
            )
        targets = targets.to(device=logits.device)
        valid_targets = targets[valid_mask]
        if valid_targets.numel() and (
            not torch.isfinite(valid_targets).all()
            or torch.any((valid_targets < 0) | (valid_targets > 1))
        ):
            raise ValueError(f"Valid {name} targets must be finite values in [0, 1].")
        return targets

    @staticmethod
    def _exact_mask(mask, shape, name, device):
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if tuple(mask.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}.")
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _query_mask(mask, shape, name, device):
        if mask is None:
            return None
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        try:
            return torch.broadcast_to(mask.to(device=device, dtype=torch.bool), shape)
        except RuntimeError as error:
            raise ValueError(f"{name} cannot broadcast to {tuple(shape)}.") from error

    @staticmethod
    def _token_mask(mask, shape, device):
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        try:
            return torch.broadcast_to(mask.to(device=device, dtype=torch.bool), shape)
        except RuntimeError as error:
            raise ValueError(
                f"visual_token_mask cannot broadcast to {tuple(shape)}."
            ) from error

    @staticmethod
    def _require_shape(tensor, dimensions, name):
        if tensor.dim() != dimensions:
            raise ValueError(f"{name} must have {dimensions} dimensions.")
