from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def anatomy_iou_dice(predictions, targets, valid_mask=None, threshold=0.5, smooth=1.0):
    predictions, targets = _check_anatomy_shapes(predictions, targets)
    predicted = predictions >= threshold
    target = targets >= 0.5
    valid = _expand_anatomy_valid(valid_mask, targets)
    intersection = (predicted & target & valid).sum(dim=(-1, -2)).float()
    predicted_area = (predicted & valid).sum(dim=(-1, -2)).float()
    target_area = (target & valid).sum(dim=(-1, -2)).float()
    union = predicted_area + target_area - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (predicted_area + target_area + smooth)
    has_valid = valid.any(dim=(-1, -2))
    return {"iou": _masked_mean(iou, has_valid), "dice": _masked_mean(dice, has_valid)}


def pointing_game(attention, targets, valid_mask=None):
    attention, targets = _check_anatomy_shapes(attention, targets)
    valid = _expand_anatomy_valid(valid_mask, targets)
    attention = attention.masked_fill(~valid, float("-inf"))
    peak = attention.flatten(-2).argmax(dim=-1)
    target = targets.flatten(-2) >= 0.5
    hit = target.gather(-1, peak.unsqueeze(-1)).squeeze(-1)
    has_valid = valid.any(dim=(-1, -2))
    return {"pointing_accuracy": _masked_mean(hit.float(), has_valid)}


def multilabel_metrics(scores, targets, valid_mask=None, threshold=0.5):
    scores, targets, valid_mask = _classification_arrays(scores, targets, valid_mask)
    output = OrderedDict()
    predictions = (scores >= threshold).astype(np.int64)
    output["f1"] = _macro_f1(scores, targets, valid_mask, predictions)
    output["auroc"] = _macro_curve_metric(roc_auc_score, scores, targets, valid_mask)
    output["auprc"] = _macro_curve_metric(
        average_precision_score, scores, targets, valid_mask
    )
    return output


def _macro_f1(scores, targets, valid, predictions):
    values = []
    for index in range(targets.shape[1]):
        mask = valid[:, index]
        if mask.sum() == 0:
            continue
        values.append(
            f1_score(targets[mask, index], predictions[mask, index], zero_division=0)
        )
    return float(np.mean(values)) if values else float("nan")


def _macro_curve_metric(metric, scores, targets, valid):
    values = []
    for index in range(targets.shape[1]):
        mask = valid[:, index]
        if mask.sum() == 0 or np.unique(targets[mask, index]).size < 2:
            continue
        values.append(metric(targets[mask, index], scores[mask, index]))
    return float(np.mean(values)) if values else float("nan")


def _classification_arrays(scores, targets, valid_mask):
    scores = _to_numpy(scores)
    targets = _to_numpy(targets)
    if scores.shape != targets.shape or scores.ndim != 2:
        raise ValueError(
            "classification scores and targets must both have shape [B, C]."
        )
    if valid_mask is None:
        valid = np.ones_like(targets, dtype=bool)
    else:
        valid = _to_numpy(valid_mask).astype(bool)
        if valid.shape != targets.shape:
            raise ValueError("classification valid_mask must match scores shape.")
    if not np.isfinite(scores).all():
        raise ValueError("classification scores must be finite.")
    return scores, targets.astype(np.int64), valid


def _check_anatomy_shapes(predictions, targets):
    if predictions.shape != targets.shape or predictions.ndim != 4:
        raise ValueError(
            "anatomy predictions and targets must both have shape [B, A, H, W]."
        )
    return predictions, targets


def _expand_anatomy_valid(valid_mask, targets):
    if valid_mask is None:
        return torch.ones_like(targets, dtype=torch.bool)
    if valid_mask.shape == targets.shape[:2]:
        return valid_mask.to(dtype=torch.bool)[..., None, None].expand_as(targets)
    if valid_mask.shape == targets.shape:
        return valid_mask.to(dtype=torch.bool)
    raise ValueError("anatomy valid_mask must have shape [B, A] or [B, A, H, W].")


def _masked_mean(values, valid):
    values = values[valid]
    return values.mean() if values.numel() else values.new_zeros(())


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate_hierarchy_output(
    output,
    anatomy_targets=None,
    finding_targets=None,
    global_targets=None,
    anatomy_valid_mask=None,
    finding_valid_mask=None,
    global_valid_mask=None,
    grid_shape=None,
):
    """Evaluate a HierarchicalModelOutput using its predictions and validity masks."""
    metrics = OrderedDict()
    if anatomy_targets is not None:
        if grid_shape is None:
            raise ValueError("grid_shape is required for flattened anatomy targets.")
        height, width = grid_shape
        predicted_masks = output.anatomy.roi_masks.reshape(
            output.anatomy.roi_masks.size(0),
            output.anatomy.roi_masks.size(1),
            height,
            width,
        )
        target_masks = anatomy_targets.reshape(
            anatomy_targets.size(0), anatomy_targets.size(1), height, width
        )
        metrics.update(
            {
                f"anatomy_{key}": value
                for key, value in anatomy_iou_dice(
                    predicted_masks, target_masks, anatomy_valid_mask
                ).items()
            }
        )
        metrics.update(
            {
                f"anatomy_{key}": value
                for key, value in pointing_game(
                    predicted_masks, target_masks, anatomy_valid_mask
                ).items()
            }
        )
    if finding_targets is not None:
        finding_scores = torch.sigmoid(output.finding.finding_logits).mean(dim=1)
        finding_targets_flat = finding_targets.any(dim=1).float()
        finding_valid = (
            finding_valid_mask.any(dim=1) if finding_valid_mask is not None else None
        )
        if finding_scores.shape != finding_targets_flat.shape:
            raise ValueError(
                "finding output and targets must agree after anatomy aggregation."
            )
        metrics.update(
            {
                f"finding_{key}": value
                for key, value in multilabel_metrics(
                    finding_scores, finding_targets_flat, finding_valid
                ).items()
            }
        )
    if global_targets is not None:
        global_scores = torch.sigmoid(output.fusion.logits)
        metrics.update(
            {
                f"global_{key}": value
                for key, value in multilabel_metrics(
                    global_scores, global_targets, global_valid_mask
                ).items()
            }
        )
    return metrics


def export_hierarchy_output(output):
    """Return detached tensors useful for attention/evidence export."""
    return {
        "anatomy_masks": output.anatomy.roi_masks.detach(),
        "anatomy_attention": output.anatomy.attention_weights.detach(),
        "finding_logits": output.finding.finding_logits.detach(),
        "finding_attention": output.finding.attention_weights.detach(),
        "finding_pair_mask": output.finding.pair_mask.detach(),
        "global_logits": output.fusion.logits.detach(),
        "global_gates": output.fusion.gates.detach(),
        "finding_evidence_attention": output.finding_aggregation.attention_weights.detach(),
        "finding_null_attention": output.finding_aggregation.null_attention.detach(),
        "fused_evidence": output.fusion.fused_evidence.detach(),
    }
