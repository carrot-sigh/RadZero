import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file

from common.utils import Config
from exp.cxr_pt.hierarchy_dataset import HierarchyJsonlDataset, hierarchy_collate_fn
from exp.cxr_pt.hierarchy_metrics import multilabel_metrics
from exp.cxr_pt.hierarchy_metrics import bootstrap_multilabel_metrics
from exp.cxr_pt.hierarchy_metrics import anatomy_iou_dice
from exp.cxr_pt.model import load_model, masks_to_feature_boxes


def _load_config(config_path, config_overrides):
    args = SimpleNamespace(
        cfg_path=config_path,
        add_cfg_list=config_overrides,
        user=None,
        name=None,
        train=False,
        inference=False,
        output_dir=None,
        compute_metric=False,
        no_report=True,
    )
    return Config(args).config


def _select_balanced_indices(dataset, count, seed, candidate_pool=512):
    candidates = []
    split_index = 0
    with dataset.manifest_path.open() as source:
        for line in source:
            row = json.loads(line)
            if row["split"] != dataset.split:
                continue
            values = {finding[2] for finding in row["findings"]}
            if values == {0, 1}:
                candidates.append(split_index)
                if len(candidates) >= candidate_pool:
                    break
            split_index += 1
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} samples contain both positive and negative "
            f"finding labels; requested {count}."
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def _select_anatomy_indices(dataset, count, seed, candidate_pool=1024):
    candidates = []
    for index in range(min(len(dataset), candidate_pool)):
        item = dataset[index]
        if int(item["anatomy_annotation_mask"].sum().item()) >= 20:
            candidates.append(index)
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} samples have at least 20 anatomy boxes; "
            f"requested {count}."
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def _select_global_indices(dataset, count, seed, candidate_pool=512):
    candidates = []
    split_index = 0
    with dataset.manifest_path.open() as source:
        for line in source:
            row = json.loads(line)
            if row["split"] != dataset.split:
                continue
            global_values = set(row.get("global", {}).values())
            finding_values = {finding[2] for finding in row["findings"]}
            if global_values == {0, 1} and finding_values == {0, 1}:
                candidates.append(split_index)
                if len(candidates) >= candidate_pool:
                    break
            split_index += 1
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} samples contain mixed global and finding "
            f"labels; requested {count}."
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def _canonical_split_indices(dataset, count, seed, indices_path=None):
    """Return deterministic local indices for one canonical dataset split."""
    if indices_path is not None:
        path = Path(indices_path)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            indices = payload.get("indices", payload)
            indices = [int(index) for index in indices]
            if len(indices) != count or len(set(indices)) != count:
                raise ValueError(f"{path} must contain exactly {count} unique indices.")
            if any(index < 0 or index >= len(dataset) for index in indices):
                raise ValueError(f"{path} contains an out-of-range dataset index.")
            return indices

    if count > len(dataset):
        raise ValueError(
            f"Requested {count} samples from {dataset.split}, but only "
            f"{len(dataset)} are available."
        )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[:count])
    if indices_path is not None:
        path = Path(indices_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "split": dataset.split,
                    "seed": seed,
                    "count": count,
                    "indices": indices,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return indices


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _bounded_temperature(log_temperature):
    return log_temperature.detach().float().clamp(-6.907755, 0.0).exp().item()


def _finding_statistics(logits, targets, valid, labels):
    valid = valid.bool()
    positive = valid & (targets == 1)
    negative = valid & (targets == 0)
    result = {
        "valid_pair_count": int(valid.sum().item()),
        "positive_pair_count": int(positive.sum().item()),
        "negative_pair_count": int(negative.sum().item()),
        "positive_logit_mean": (
            float(logits[positive].float().mean().item()) if positive.any() else None
        ),
        "negative_logit_mean": (
            float(logits[negative].float().mean().item()) if negative.any() else None
        ),
        "per_finding": [],
    }
    finding_count = logits.size(-1)
    pair_metrics = multilabel_metrics(
        torch.sigmoid(logits).float().reshape(-1, finding_count),
        targets.float().reshape(-1, finding_count),
        valid.reshape(-1, finding_count),
        include_per_class=True,
    )
    result.update(
        {
            "macro_f1": pair_metrics["f1"],
            "macro_auroc": pair_metrics["auroc"],
            "macro_auprc": pair_metrics["auprc"],
        }
    )
    if positive.any() and negative.any():
        result["logit_separation"] = (
            result["positive_logit_mean"] - result["negative_logit_mean"]
        )
    else:
        result["logit_separation"] = None
    for index, label in enumerate(labels):
        class_valid = valid[..., index]
        class_positive = positive[..., index]
        class_negative = negative[..., index]
        if not class_valid.any():
            continue
        result["per_finding"].append(
            {
                "index": index,
                "label": label,
                "valid": int(class_valid.sum().item()),
                "positive": int(class_positive.sum().item()),
                "negative": int(class_negative.sum().item()),
                "positive_logit_mean": (
                    float(logits[..., index][class_positive].float().mean().item())
                    if class_positive.any()
                    else None
                ),
                "negative_logit_mean": (
                    float(logits[..., index][class_negative].float().mean().item())
                    if class_negative.any()
                    else None
                ),
                "auroc": pair_metrics.get(f"auroc_class_{index}"),
                "auprc": pair_metrics.get(f"auprc_class_{index}"),
            }
        )
    return result


def _finding_forward(model, features):
    device_type = features["visual_tokens"].device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        if features["finding_roi_strategy"] == "feature_roi":
            local_tokens, valid_boxes, _ = model.feature_roi_extractor(
                features["visual_tokens"],
                features["anatomy_boxes"],
                features["anatomy_annotation_mask"],
            )
            batch_size, anatomy_count, token_count, hidden_dim = local_tokens.shape
            local_tokens = model.finding_local_refiner(
                local_tokens.reshape(-1, token_count, hidden_dim),
                model.feature_roi_extractor.output_size,
            ).reshape(batch_size, anatomy_count, token_count, hidden_dim)
            output = model.finding_head.forward_local(
                features["finding_queries"],
                local_tokens,
                anatomy_valid_mask=valid_boxes,
                compatibility_mask=features["compatibility_mask"],
            )
        elif features["finding_roi_strategy"] == "full_refiner":
            grid_size = math.isqrt(features["visual_tokens"].size(1))
            refined_tokens = model.finding_local_refiner(
                features["visual_tokens"], grid_size
            )
            full_masks = features["visual_tokens"].new_ones(
                features["visual_tokens"].size(0),
                features["anatomy_masks"].size(1),
                features["visual_tokens"].size(1),
            )
            output = model.finding_head(
                features["finding_queries"],
                refined_tokens,
                full_masks,
                anatomy_query_mask=features["anatomy_query_mask"],
                compatibility_mask=features["compatibility_mask"],
            )
        else:
            output = model.finding_head(
                features["finding_queries"],
                features["visual_tokens"],
                features["anatomy_masks"],
                anatomy_query_mask=features["anatomy_query_mask"],
                compatibility_mask=features["compatibility_mask"],
            )
        loss = model.hierarchy_loss(
            None,
            output.finding_logits,
            None,
            finding_targets=features["finding_targets"],
            finding_valid_mask=features["finding_valid_mask"],
            finding_pair_mask=output.pair_mask,
        ).finding_loss
    return output, loss


def _load_anatomy_head_checkpoint(model, checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("--anatomy-checkpoint is required for predicted ROI mode.")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    prefix = "anatomy_head."
    prefixed = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    anatomy_state = prefixed or state_dict
    model.anatomy_head.load_state_dict(anatomy_state, strict=True)


def _load_finding_head_checkpoint(model, checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("--finding-checkpoint is required for Stage 3 tiny-overfit.")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    prefix = "finding_head."
    prefixed = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    model.finding_head.load_state_dict(prefixed or state_dict, strict=True)


def _load_full_hierarchy_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Hierarchy checkpoint not found: {checkpoint_path}")
    state_dict = load_file(str(checkpoint_path))
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible full hierarchy checkpoint: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def _select_roi_masks(model, batch, visual_tokens, anatomy_queries, args):
    ground_truth = batch["gt_anatomy_masks"]
    if args.roi_mode == "gt":
        return ground_truth, None
    if args.roi_mode == "full":
        return torch.ones_like(ground_truth), None
    if args.roi_mode == "shuffled":
        permutation = torch.roll(
            torch.arange(ground_truth.size(0), device=ground_truth.device), shifts=1
        )
        return ground_truth[permutation], permutation.cpu().tolist()
    if args.roi_mode == "predicted":
        _load_anatomy_head_checkpoint(model, args.anatomy_checkpoint)
        with torch.no_grad():
            prediction = model.anatomy_head(
                anatomy_queries,
                visual_tokens,
                anatomy_query_mask=batch["anatomy_query_mask"],
            )
        return prediction.roi_masks, None
    raise ValueError(f"Unsupported ROI mode: {args.roi_mode}")


def _select_roi_boxes(batch, args):
    boxes = batch["anatomy_boxes"]
    if args.roi_mode == "gt":
        return boxes, None
    if args.roi_mode == "shuffled":
        permutation = torch.roll(
            torch.arange(boxes.size(0), device=boxes.device), shifts=1
        )
        return boxes[permutation], permutation.cpu().tolist()
    raise ValueError(
        "Feature ROI supports --roi-mode gt, shuffled, or predicted; "
        f"got {args.roi_mode!r}."
    )


def _prepare_stage2_features(model, batch, args, device):
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        vision = model.forward_vision_model(batch["pixel_values"])
        finding_queries = model._encode_prompt_bank(
            batch["encoded_finding_prompts"], model.num_findings, "finding"
        )
        anatomy_annotation_mask = batch["anatomy_annotation_mask"]
        if args.finding_roi_strategy == "feature_roi":
            if args.roi_mode == "predicted":
                anatomy_queries = model._encode_prompt_bank(
                    batch["encoded_anatomy_prompts"], model.num_anatomies, "anatomy"
                )
                _load_anatomy_head_checkpoint(model, args.anatomy_checkpoint)
                prediction = model.anatomy_head(
                    anatomy_queries,
                    vision["image_patch_tokens"],
                    anatomy_query_mask=batch["anatomy_query_mask"],
                )
                anatomy_masks = prediction.roi_masks
                anatomy_boxes = masks_to_feature_boxes(anatomy_masks)
                anatomy_annotation_mask = (
                    anatomy_boxes[..., 2] > anatomy_boxes[..., 0]
                ) & (anatomy_boxes[..., 3] > anatomy_boxes[..., 1])
                roi_permutation = None
            else:
                anatomy_masks = batch["gt_anatomy_masks"]
                anatomy_boxes, roi_permutation = _select_roi_boxes(batch, args)
        elif args.finding_roi_strategy in {"full", "full_refiner"}:
            anatomy_masks = torch.ones_like(batch["gt_anatomy_masks"])
            anatomy_boxes = batch["anatomy_boxes"]
            roi_permutation = None
        else:
            anatomy_queries = None
            if args.roi_mode == "predicted":
                anatomy_queries = model._encode_prompt_bank(
                    batch["encoded_anatomy_prompts"], model.num_anatomies, "anatomy"
                )
            anatomy_masks, roi_permutation = _select_roi_masks(
                model,
                batch,
                vision["image_patch_tokens"],
                anatomy_queries,
                args,
            )
            anatomy_boxes = batch["anatomy_boxes"]
    return {
        "finding_roi_strategy": args.finding_roi_strategy,
        "visual_tokens": vision["image_patch_tokens"].detach(),
        "finding_queries": finding_queries.detach(),
        "anatomy_masks": anatomy_masks.detach(),
        "anatomy_boxes": anatomy_boxes.detach(),
        "anatomy_annotation_mask": anatomy_annotation_mask,
        "anatomy_query_mask": batch["anatomy_query_mask"],
        "compatibility_mask": batch["compatibility_mask"],
        "finding_targets": batch["finding_targets"],
        "finding_valid_mask": batch["finding_valid_mask"],
        "roi_permutation": roi_permutation,
    }


def _move_feature_batch(features, device):
    return {
        key: _to_device(value, device)
        for key, value in features.items()
        if key != "roi_permutation"
    }


def _cache_stage2_features(model, dataset, indices, batch_size, models, args, device):
    cached = []
    for start in range(0, len(indices), batch_size):
        samples = [dataset[index] for index in indices[start : start + batch_size]]
        batch = hierarchy_collate_fn(
            samples,
            tokenizer=models["tokenizer"],
            image_processor=models["image_processor"],
            contract_path=dataset.contract_path,
        )
        batch = _to_device(batch, device)
        features = _prepare_stage2_features(model, batch, args, device)
        cached.append(
            {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in features.items()
            }
        )
    return cached


def _cache_anatomy_features(model, dataset, indices, batch_size, models, device):
    cached = []
    for start in range(0, len(indices), batch_size):
        samples = [dataset[index] for index in indices[start : start + batch_size]]
        batch = hierarchy_collate_fn(
            samples,
            tokenizer=models["tokenizer"],
            image_processor=models["image_processor"],
            contract_path=dataset.contract_path,
        )
        batch = _to_device(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            vision = model.forward_vision_model(batch["pixel_values"])
            anatomy_queries = model._encode_prompt_bank(
                batch["encoded_anatomy_prompts"], model.num_anatomies, "anatomy"
            )
        cached.append(
            {
                "visual_tokens": vision["image_patch_tokens"].detach().cpu(),
                "anatomy_queries": anatomy_queries.detach().cpu(),
                "anatomy_targets": batch["anatomy_targets"].detach().cpu(),
                "anatomy_valid_mask": batch["anatomy_annotation_mask"].detach().cpu(),
                "anatomy_query_mask": batch["anatomy_query_mask"].detach().cpu(),
            }
        )
    return cached


def run_stage1_pilot(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
    device = torch.device(args.device)
    config = _load_config(args.config, args.config_overrides)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model(config["model"], output_dir=str(output_dir))
    model = models["model"].to(device)
    model.configure_hierarchy_training("anatomy")
    hierarchy_config = config["dataset"]["hierarchy"]
    train_dataset = HierarchyJsonlDataset(
        hierarchy_config["manifest_path"],
        hierarchy_config["image_root"],
        hierarchy_config["contract_path"],
        split=hierarchy_config.get("train_split", "train"),
        patch_grid_size=hierarchy_config["patch_grid_size"],
    )
    valid_dataset = HierarchyJsonlDataset(
        hierarchy_config["manifest_path"],
        hierarchy_config["image_root"],
        hierarchy_config["contract_path"],
        split=hierarchy_config.get("eval_split", "valid"),
        patch_grid_size=hierarchy_config["patch_grid_size"],
    )
    if args.indices_dir:
        indices_dir = Path(args.indices_dir)
        train_indices = _canonical_split_indices(
            train_dataset,
            args.train_images,
            args.seed,
            indices_dir / "pilot_train_indices.json",
        )
        valid_indices = _canonical_split_indices(
            valid_dataset,
            args.valid_images,
            args.seed + 1,
            indices_dir / "pilot_valid_indices.json",
        )
    else:
        train_indices = _select_anatomy_indices(
            train_dataset, args.train_images, args.seed
        )
        valid_indices = _select_anatomy_indices(
            valid_dataset, args.valid_images, args.seed + 1
        )
    train_cached = _cache_anatomy_features(
        model, train_dataset, train_indices, args.batch_size, models, device
    )
    valid_cached = _cache_anatomy_features(
        model, valid_dataset, valid_indices, args.batch_size, models, device
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or not all(
        name.startswith("anatomy_head.") for name in trainable_names
    ):
        raise RuntimeError(
            f"Unexpected Stage 1 trainable parameters: {trainable_names}"
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def evaluate(cached):
        losses = []
        predictions = []
        targets = []
        valid_masks = []
        model.eval()
        with torch.no_grad():
            for item in cached:
                visual_tokens = item["visual_tokens"].to(device)
                anatomy_queries = item["anatomy_queries"].to(device)
                anatomy_targets = item["anatomy_targets"].to(device)
                anatomy_valid = item["anatomy_valid_mask"].to(device)
                query_mask = item["anatomy_query_mask"].to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model.anatomy_head(
                        anatomy_queries,
                        visual_tokens,
                        anatomy_query_mask=query_mask,
                    )
                    loss = model.hierarchy_loss(
                        output.roi_mask_logits,
                        None,
                        None,
                        anatomy_targets=anatomy_targets,
                        anatomy_valid_mask=anatomy_valid,
                        anatomy_query_mask=query_mask,
                    ).anatomy_loss
                losses.append(float(loss.float().item()))
                predictions.append(output.roi_masks.detach().cpu())
                targets.append(anatomy_targets.detach().cpu())
                valid_masks.append(anatomy_valid.detach().cpu())
        predictions = torch.cat(predictions).reshape(
            -1, model.num_anatomies, *train_dataset.patch_grid_size
        )
        targets = torch.cat(targets).reshape_as(predictions)
        valid_masks = torch.cat(valid_masks)
        metrics = anatomy_iou_dice(predictions, targets, valid_masks)
        return {
            "loss": float(sum(losses) / len(losses)),
            "dice": float(metrics["dice"].item()),
            "iou": float(metrics["iou"].item()),
        }

    initial = evaluate(valid_cached)
    history = []
    model.train()
    for step in range(args.steps):
        item = train_cached[step % len(train_cached)]
        visual_tokens = item["visual_tokens"].to(device)
        anatomy_queries = item["anatomy_queries"].to(device)
        anatomy_targets = item["anatomy_targets"].to(device)
        anatomy_valid = item["anatomy_valid_mask"].to(device)
        query_mask = item["anatomy_query_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model.anatomy_head(
                anatomy_queries,
                visual_tokens,
                anatomy_query_mask=query_mask,
            )
            loss = model.hierarchy_loss(
                output.roi_mask_logits,
                None,
                None,
                anatomy_targets=anatomy_targets,
                anatomy_valid_mask=anatomy_valid,
                anatomy_query_mask=query_mask,
            ).anatomy_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Stage 1 loss at step {step + 1}.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Non-finite Stage 1 gradient.")
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().float().item()),
                    "gradient_norm": float(gradient_norm.detach().float().item()),
                }
            )
    final = evaluate(valid_cached)
    report = {
        "experiment": "stage1_real_pilot",
        "seed": args.seed,
        "train_images": args.train_images,
        "valid_images": args.valid_images,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "trainable_parameters": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "initial_validation": initial,
        "final_validation": final,
        "history": history,
        "loss_decreased": final["loss"] < initial["loss"],
    }
    report_path = output_dir / "stage1_pilot.json"
    checkpoint_path = output_dir / "anatomy_head.pt"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(model.anatomy_head.state_dict(), checkpoint_path)
    print(json.dumps({"report": str(report_path), **final}, indent=2))
    return report


def run_stage2_tiny_overfit(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = _load_config(args.config, args.config_overrides)
    hierarchy_config = config["dataset"]["hierarchy"]
    model_hierarchy_config = config["model"]["model_config"]["hierarchy_config"]
    model_hierarchy_config["finding_roi_strategy"] = args.finding_roi_strategy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model(config["model"], output_dir=str(output_dir))
    model = models["model"].to(device)
    model.configure_hierarchy_training("finding")

    dataset = HierarchyJsonlDataset(
        hierarchy_config["manifest_path"],
        hierarchy_config["image_root"],
        hierarchy_config["contract_path"],
        split=hierarchy_config.get("train_split", "train"),
        patch_grid_size=hierarchy_config["patch_grid_size"],
    )
    indices = _select_balanced_indices(dataset, args.num_images, args.seed)
    samples = [dataset[index] for index in indices]
    batch = hierarchy_collate_fn(
        samples,
        tokenizer=models["tokenizer"],
        image_processor=models["image_processor"],
        contract_path=hierarchy_config["contract_path"],
    )
    batch = _to_device(batch, device)

    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        vision = model.forward_vision_model(batch["pixel_values"])
        finding_queries = model._encode_prompt_bank(
            batch["encoded_finding_prompts"], model.num_findings, "finding"
        )
        anatomy_queries = None
        if args.roi_mode == "predicted":
            anatomy_queries = model._encode_prompt_bank(
                batch["encoded_anatomy_prompts"], model.num_anatomies, "anatomy"
            )
        anatomy_annotation_mask = batch["anatomy_annotation_mask"]
        if args.finding_roi_strategy == "feature_roi":
            if args.roi_mode == "predicted":
                anatomy_queries = model._encode_prompt_bank(
                    batch["encoded_anatomy_prompts"], model.num_anatomies, "anatomy"
                )
                _load_anatomy_head_checkpoint(model, args.anatomy_checkpoint)
                prediction = model.anatomy_head(
                    anatomy_queries,
                    vision["image_patch_tokens"],
                    anatomy_query_mask=batch["anatomy_query_mask"],
                )
                anatomy_masks = prediction.roi_masks
                anatomy_boxes = masks_to_feature_boxes(anatomy_masks)
                anatomy_annotation_mask = (
                    anatomy_boxes[..., 2] > anatomy_boxes[..., 0]
                ) & (anatomy_boxes[..., 3] > anatomy_boxes[..., 1])
                roi_permutation = None
            else:
                anatomy_masks = batch["gt_anatomy_masks"]
                anatomy_boxes, roi_permutation = _select_roi_boxes(batch, args)
        else:
            anatomy_masks, roi_permutation = _select_roi_masks(
                model,
                batch,
                vision["image_patch_tokens"],
                anatomy_queries,
                args,
            )
            anatomy_boxes = batch["anatomy_boxes"]
    features = {
        "finding_roi_strategy": args.finding_roi_strategy,
        "visual_tokens": vision["image_patch_tokens"].detach(),
        "finding_queries": finding_queries.detach(),
        "anatomy_masks": anatomy_masks.detach(),
        "anatomy_boxes": anatomy_boxes.detach(),
        "anatomy_annotation_mask": anatomy_annotation_mask,
        "anatomy_query_mask": batch["anatomy_query_mask"],
        "compatibility_mask": batch["compatibility_mask"],
        "finding_targets": batch["finding_targets"],
        "finding_valid_mask": batch["finding_valid_mask"],
    }
    del vision, batch
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    allowed_trainable_prefixes = (
        ("finding_head.", "finding_local_refiner.")
        if args.finding_roi_strategy in {"feature_roi", "full_refiner"}
        else ("finding_head.",)
    )
    if not trainable_names or not all(
        name.startswith(allowed_trainable_prefixes) for name in trainable_names
    ):
        raise RuntimeError(
            f"Unexpected Stage 2 trainable parameters: {trainable_names}"
        )

    contract = dataset.contract
    labels = contract["finding"]["labels"]
    pair_valid = features["finding_valid_mask"] & features[
        "compatibility_mask"
    ].unsqueeze(0)
    pair_valid = pair_valid & features["anatomy_query_mask"].unsqueeze(-1)
    with torch.no_grad():
        initial_output, initial_loss = _finding_forward(model, features)
    initial = _finding_statistics(
        initial_output.finding_logits,
        features["finding_targets"],
        pair_valid,
        labels,
    )
    initial["loss"] = float(initial_loss.float().item())
    initial["attention_temperature"] = _bounded_temperature(
        model.finding_head.log_attention_temperature
    )
    initial["logit_temperature"] = _bounded_temperature(
        model.finding_head.log_logit_temperature
    )
    initial["finding_bias"] = model.finding_head.finding_bias.detach().float().tolist()

    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output, loss = _finding_forward(model, features)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Stage 2 loss at step {step}: {loss.item()}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}.")
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().float().item()),
                    "gradient_norm": float(gradient_norm.detach().float().item()),
                    "attention_temperature": _bounded_temperature(
                        model.finding_head.log_attention_temperature
                    ),
                    "logit_temperature": _bounded_temperature(
                        model.finding_head.log_logit_temperature
                    ),
                }
            )

    model.eval()
    with torch.no_grad():
        final_output, final_loss = _finding_forward(model, features)
    final = _finding_statistics(
        final_output.finding_logits,
        features["finding_targets"],
        pair_valid,
        labels,
    )
    final["loss"] = float(final_loss.float().item())
    final["attention_temperature"] = _bounded_temperature(
        model.finding_head.log_attention_temperature
    )
    final["logit_temperature"] = _bounded_temperature(
        model.finding_head.log_logit_temperature
    )
    final["finding_bias"] = model.finding_head.finding_bias.detach().float().tolist()
    report = {
        "experiment": "stage2_roi_tiny_overfit",
        "finding_roi_strategy": args.finding_roi_strategy,
        "roi_mode": args.roi_mode,
        "anatomy_checkpoint": args.anatomy_checkpoint,
        "roi_permutation": roi_permutation,
        "roi_positive_fraction": float(
            (features["anatomy_masks"] > 0.5).float().mean().item()
        ),
        "seed": args.seed,
        "num_images": args.num_images,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dataset_indices": indices,
        "trainable_parameters": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "initial": initial,
        "final": final,
        "history": history,
        "loss_decreased": final["loss"] < initial["loss"],
        "all_values_finite": all(
            math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"])
            for item in history
        ),
    }
    report_path = output_dir / "stage2_real_smoke.json"
    checkpoint_path = output_dir / "finding_head.pt"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        model.finding_head.state_dict(),
        checkpoint_path,
    )
    if args.finding_roi_strategy in {"feature_roi", "full_refiner"}:
        torch.save(
            model.finding_local_refiner.state_dict(),
            output_dir / "finding_local_refiner.pt",
        )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "roi_mode": args.roi_mode,
                "initial_loss": initial["loss"],
                "final_loss": final["loss"],
                "initial_separation": initial["logit_separation"],
                "final_separation": final["logit_separation"],
                "initial_macro_auroc": initial["macro_auroc"],
                "final_macro_auroc": final["macro_auroc"],
                "initial_macro_auprc": initial["macro_auprc"],
                "final_macro_auprc": final["macro_auprc"],
                "loss_decreased": report["loss_decreased"],
            },
            indent=2,
        )
    )
    return report


def _evaluate_stage2_cached(model, cached, dataset, device):
    labels = dataset.contract["finding"]["labels"]
    losses = []
    logits = []
    targets = []
    valid_masks = []
    model.eval()
    with torch.no_grad():
        for cpu_features in cached:
            features = _move_feature_batch(cpu_features, device)
            output, loss = _finding_forward(model, features)
            losses.append(float(loss.float().item()))
            logits.append(output.finding_logits.detach().cpu())
            targets.append(features["finding_targets"].detach().cpu())
            valid_masks.append(
                (features["finding_valid_mask"] & output.pair_mask).detach().cpu()
            )
    logits = torch.cat(logits, dim=0)
    targets = torch.cat(targets, dim=0)
    valid_masks = torch.cat(valid_masks, dim=0)
    statistics = _finding_statistics(logits, targets, valid_masks, labels)
    statistics["loss"] = float(sum(losses) / len(losses))
    return statistics


def _save_feature_roi_visualizations(
    model, cached_features, dataset, indices, output_dir, device
):
    """Save bbox/grid/local-attention overlays for a few validation images."""
    if not cached_features:
        return []
    contract = dataset.contract
    anatomy_labels = contract["anatomy"]["labels"]
    finding_labels = contract["finding"]["labels"]
    selected_anatomies = (
        "right upper lung zone",
        "right lower lung zone",
        "left lower lung zone",
        "cardiac silhouette",
    )
    finding_index = (
        finding_labels.index("lung opacity") if "lung opacity" in finding_labels else 0
    )
    features = _move_feature_batch(cached_features[0], device)
    model.eval()
    with torch.no_grad():
        output, _ = _finding_forward(model, features)
    attention = output.attention_weights.float().cpu()
    boxes = features["anatomy_boxes"].float().cpu()
    valid = features["anatomy_annotation_mask"].cpu()
    saved = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_index, dataset_index in enumerate(indices[: attention.size(0)]):
        item = dataset[dataset_index]
        image = item["image"].convert("RGB")
        width, height = image.size
        scale_x = width / dataset.patch_grid_size[1]
        scale_y = height / dataset.patch_grid_size[0]
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        for anatomy_name in selected_anatomies:
            anatomy_index = anatomy_labels.index(anatomy_name)
            if not valid[batch_index, anatomy_index]:
                continue
            x1, y1, x2, y2 = boxes[batch_index, anatomy_index].tolist()
            px1, py1, px2, py2 = (
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            )
            draw.rectangle((px1, py1, px2, py2), outline=(0, 255, 0, 255), width=8)
            local_attention = attention[batch_index, anatomy_index, finding_index]
            local_size = math.isqrt(local_attention.numel())
            local_attention = local_attention.reshape(local_size, local_size)
            local_attention = local_attention / local_attention.max().clamp_min(1e-6)
            for row in range(local_size):
                for col in range(local_size):
                    alpha = int(180 * local_attention[row, col].item())
                    cell_x1 = px1 + (px2 - px1) * col / local_size
                    cell_y1 = py1 + (py2 - py1) * row / local_size
                    cell_x2 = px1 + (px2 - px1) * (col + 1) / local_size
                    cell_y2 = py1 + (py2 - py1) * (row + 1) / local_size
                    draw.rectangle(
                        (cell_x1, cell_y1, cell_x2, cell_y2),
                        fill=(255, 40, 0, alpha),
                    )
            draw.text((px1 + 4, py1 + 4), anatomy_name, fill=(0, 255, 0, 255))
        image.thumbnail((1024, 1024))
        path = output_dir / f"valid-{dataset_index:06d}-lung-opacity.png"
        overlay.thumbnail((1024, 1024))
        overlay.save(path)
        saved.append(str(path))
    return saved


def run_stage2_pilot(args):
    strategies = args.strategies
    pilot_reports = []
    for strategy_index, strategy in enumerate(strategies):
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        device = torch.device(args.device)
        config = _load_config(args.config, args.config_overrides)
        config["model"]["model_config"]["hierarchy_config"][
            "finding_roi_strategy"
        ] = strategy
        config["model"]["model_config"]["hierarchy_config"][
            "feature_roi_context_margin"
        ] = args.context_margin
        output_dir = Path(args.output_dir) / strategy
        output_dir.mkdir(parents=True, exist_ok=True)
        models = load_model(config["model"], output_dir=str(output_dir))
        model = models["model"].to(device)
        model.configure_hierarchy_training("finding")
        hierarchy_config = config["dataset"]["hierarchy"]
        train_dataset = HierarchyJsonlDataset(
            hierarchy_config["manifest_path"],
            hierarchy_config["image_root"],
            hierarchy_config["contract_path"],
            split=hierarchy_config.get("train_split", "train"),
            patch_grid_size=hierarchy_config["patch_grid_size"],
        )
        valid_dataset = HierarchyJsonlDataset(
            hierarchy_config["manifest_path"],
            hierarchy_config["image_root"],
            hierarchy_config["contract_path"],
            split=hierarchy_config.get("eval_split", "valid"),
            patch_grid_size=hierarchy_config["patch_grid_size"],
        )
        if args.indices_dir:
            indices_dir = Path(args.indices_dir)
            train_indices = _canonical_split_indices(
                train_dataset,
                args.train_images,
                args.seed,
                indices_dir / "pilot_train_indices.json",
            )
            valid_indices = _canonical_split_indices(
                valid_dataset,
                args.valid_images,
                args.seed + 1,
                indices_dir / "pilot_valid_indices.json",
            )
        else:
            train_indices = _select_balanced_indices(
                train_dataset, args.train_images, args.seed
            )
            valid_indices = _select_balanced_indices(
                valid_dataset, args.valid_images, args.seed + 1
            )
        strategy_args = SimpleNamespace(
            finding_roi_strategy=strategy,
            roi_mode=args.roi_mode,
            anatomy_checkpoint=args.anatomy_checkpoint,
        )
        train_cached = _cache_stage2_features(
            model,
            train_dataset,
            train_indices,
            args.batch_size,
            models,
            strategy_args,
            device,
        )
        valid_cached = _cache_stage2_features(
            model,
            valid_dataset,
            valid_indices,
            args.batch_size,
            models,
            strategy_args,
            device,
        )
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        allowed = (
            ("finding_head.", "finding_local_refiner.")
            if strategy in {"feature_roi", "full_refiner"}
            else ("finding_head.",)
        )
        if not trainable_names or not all(
            name.startswith(allowed) for name in trainable_names
        ):
            raise RuntimeError(
                f"Unexpected pilot trainable parameters: {trainable_names}"
            )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_start = time.perf_counter()
        initial = _evaluate_stage2_cached(model, valid_cached, valid_dataset, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_validation_seconds = time.perf_counter() - validation_start
        step_times = []
        history = []
        model.train()
        for step in range(args.steps):
            cpu_features = train_cached[step % len(train_cached)]
            features = _move_feature_batch(cpu_features, device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            _, loss = _finding_forward(model, features)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite pilot loss for {strategy} at step {step + 1}."
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                args.max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite pilot gradient for {strategy}.")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - start_time)
            if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                history.append(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach().float().item()),
                        "gradient_norm": float(gradient_norm.detach().float().item()),
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_start = time.perf_counter()
        final = _evaluate_stage2_cached(model, valid_cached, valid_dataset, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        final_validation_seconds = time.perf_counter() - validation_start
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        )
        step_seconds = float(sum(step_times) / len(step_times))
        report = {
            "strategy": strategy,
            "seed": args.seed,
            "train_images": args.train_images,
            "valid_images": args.valid_images,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "context_margin": args.context_margin,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "trainable_parameters": trainable_names,
            "initial_validation": initial,
            "final_validation": final,
            "history": history,
            "step_seconds": step_seconds,
            "train_samples_per_second": args.batch_size / step_seconds,
            "validation_samples_per_second": args.valid_images
            / max(final_validation_seconds, 1e-9),
            "initial_validation_seconds": initial_validation_seconds,
            "final_validation_seconds": final_validation_seconds,
            "peak_gpu_memory_mb": peak_memory,
            "loss_decreased": final["loss"] < initial["loss"],
        }
        if strategy == "feature_roi" and args.roi_mode == "gt":
            report["visualizations"] = _save_feature_roi_visualizations(
                model,
                valid_cached,
                valid_dataset,
                valid_indices,
                output_dir / "visualizations",
                device,
            )
        report_path = output_dir / "stage2_pilot.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pilot_reports.append(report)
        del model, models, train_cached, valid_cached
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "stage2_pilot_summary.json"
    summary.write_text(json.dumps(pilot_reports, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "strategy": report["strategy"],
                    "validation_loss": report["final_validation"]["loss"],
                    "macro_auroc": report["final_validation"]["macro_auroc"],
                    "macro_auprc": report["final_validation"]["macro_auprc"],
                    "macro_f1": report["final_validation"]["macro_f1"],
                    "parameters": report["trainable_parameter_count"],
                    "peak_gpu_memory_mb": report["peak_gpu_memory_mb"],
                    "step_seconds": report["step_seconds"],
                }
                for report in pilot_reports
            ],
            indent=2,
        )
    )
    return pilot_reports


def _global_forward(model, features):
    device_type = features["global_queries"].device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        aggregation = model.finding_aggregator(
            features["global_queries"],
            features["finding_evidence"],
            finding_scores=features["finding_logits"],
            pair_mask=features["pair_mask"],
            global_query_mask=features["global_query_mask"],
        )
        fusion = model.evidence_fusion(
            aggregation.evidence_tokens,
            features["global_evidence"],
            features["global_queries"],
            global_query_mask=features["global_query_mask"],
        )
        loss = model.hierarchy_loss(
            None,
            None,
            fusion.logits,
            global_targets=features["global_targets"],
            global_valid_mask=features["global_valid_mask"],
            global_query_mask=features["global_query_mask"],
        ).global_loss
    return aggregation, fusion, loss


def _global_statistics(logits, targets, valid, gates, aggregation):
    valid = valid.bool()
    positive = valid & (targets == 1)
    negative = valid & (targets == 0)
    metrics = multilabel_metrics(torch.sigmoid(logits), targets, valid)
    return {
        "valid_label_count": int(valid.sum().item()),
        "positive_label_count": int(positive.sum().item()),
        "negative_label_count": int(negative.sum().item()),
        "positive_logit_mean": (
            float(logits[positive].float().mean().item()) if positive.any() else None
        ),
        "negative_logit_mean": (
            float(logits[negative].float().mean().item()) if negative.any() else None
        ),
        "macro_f1": metrics["f1"],
        "macro_auroc": metrics["auroc"],
        "macro_auprc": metrics["auprc"],
        "gate_mean": float(gates[valid].float().mean().item()),
        "gate_min": float(gates[valid].float().min().item()),
        "gate_max": float(gates[valid].float().max().item()),
        "null_attention_mean": float(
            aggregation.null_attention[valid].float().mean().item()
        ),
    }


def _clone_module_state(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _module_state_unchanged(module, snapshot):
    current = module.state_dict()
    return current.keys() == snapshot.keys() and all(
        torch.equal(current[name].detach().cpu(), value)
        for name, value in snapshot.items()
    )


def _module_state_max_abs_delta(module, snapshot):
    current = module.state_dict()
    deltas = [
        (current[name].detach().float().cpu() - value.float()).abs().max().item()
        for name, value in snapshot.items()
        if name in current
    ]
    return float(max(deltas, default=0.0))


def _forced_gate_logits(model, aggregation, features, gate_value):
    fusion = model.evidence_fusion
    device_type = features["global_queries"].device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        finding = fusion.finding_projection(aggregation.evidence_tokens)
        global_evidence = fusion.global_projection(features["global_evidence"])
        fused = gate_value * finding + (1.0 - gate_value) * global_evidence
        logits = (
            F.cosine_similarity(fused, features["global_queries"], dim=-1)
            / fusion.log_logit_temperature.clamp(math.log(1e-3), 0.0).exp()
        )
        logits = logits + fusion.label_bias.view(1, -1)
        return logits * features["global_query_mask"]


def _intervention_statistics(
    model, logits, normal_logits, features, bootstrap_samples=1000, bootstrap_seed=42
):
    valid = features["global_valid_mask"] & features["global_query_mask"]
    metrics = multilabel_metrics(
        torch.sigmoid(logits), features["global_targets"], valid
    )
    loss = model.hierarchy_loss(
        None,
        None,
        logits,
        global_targets=features["global_targets"],
        global_valid_mask=features["global_valid_mask"],
        global_query_mask=features["global_query_mask"],
    ).global_loss
    delta = (logits - normal_logits).abs()[valid].float()
    statistics = {
        "loss": float(loss.float().item()),
        "macro_auroc": metrics["auroc"],
        "macro_auprc": metrics["auprc"],
        "mean_absolute_logit_delta": float(delta.mean().item()),
        "max_absolute_logit_delta": float(delta.max().item()),
    }
    statistics["bootstrap"] = bootstrap_multilabel_metrics(
        torch.sigmoid(logits),
        features["global_targets"],
        valid,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return statistics


def _run_stage3_interventions(
    model,
    features,
    normal_aggregation,
    normal_fusion,
    bootstrap_samples=1000,
    bootstrap_seed=42,
):
    permutation = torch.roll(
        torch.arange(
            features["finding_evidence"].size(0),
            device=features["finding_evidence"].device,
        ),
        shifts=1,
    )
    variants = {}
    shuffled_evidence = dict(features)
    shuffled_evidence["finding_evidence"] = features["finding_evidence"][permutation]
    _, fusion, _ = _global_forward(model, shuffled_evidence)
    variants["shuffled_finding_evidence"] = fusion.logits

    zero_evidence = dict(features)
    zero_evidence["finding_evidence"] = torch.zeros_like(features["finding_evidence"])
    _, fusion, _ = _global_forward(model, zero_evidence)
    variants["zero_evidence_token"] = fusion.logits

    zero_logits = dict(features)
    zero_logits["finding_logits"] = torch.zeros_like(features["finding_logits"])
    _, fusion, _ = _global_forward(model, zero_logits)
    variants["zero_finding_logits"] = fusion.logits

    zero_full_state = dict(features)
    zero_full_state["finding_evidence"] = torch.zeros_like(features["finding_evidence"])
    zero_full_state["finding_logits"] = torch.zeros_like(features["finding_logits"])
    zero_full_state["pair_mask"] = torch.zeros_like(features["pair_mask"])
    _, fusion, _ = _global_forward(model, zero_full_state)
    variants["zero_full_finding_state"] = fusion.logits

    shuffled_state = dict(features)
    shuffled_state["finding_evidence"] = features["finding_evidence"][permutation]
    shuffled_state["finding_logits"] = features["finding_logits"][permutation]
    shuffled_state["pair_mask"] = features["pair_mask"][permutation]
    _, fusion, _ = _global_forward(model, shuffled_state)
    variants["shuffled_full_finding_state"] = fusion.logits
    variants["finding_only_gate_1"] = _forced_gate_logits(
        model, normal_aggregation, features, 1.0
    )
    variants["global_only_gate_0"] = _forced_gate_logits(
        model, normal_aggregation, features, 0.0
    )

    statistics = {
        name: _intervention_statistics(
            model,
            logits,
            normal_fusion.logits,
            features,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for name, logits in variants.items()
    }
    return permutation.cpu().tolist(), variants, statistics


def _stage3_roi_substitutions(model, features, inputs, bootstrap_samples, seed):
    """Evaluate ROI substitutions with the already-trained Stage 3 modules."""
    masks = inputs["gt_anatomy_masks"]
    variants = {
        "gt_roi_substitution": masks,
        "full_roi_substitution": torch.ones_like(masks),
    }
    permutation = torch.roll(torch.arange(masks.size(0), device=masks.device), shifts=1)
    variants["shuffled_roi"] = masks[permutation]
    statistics = {}
    for name, replacement in variants.items():
        with torch.no_grad(), torch.autocast(
            device_type=replacement.device.type,
            dtype=torch.bfloat16,
            enabled=replacement.device.type == "cuda",
        ):
            finding = model.finding_head(
                inputs["finding_queries"],
                inputs["visual_tokens"],
                replacement,
                anatomy_query_mask=inputs["anatomy_query_mask"],
                compatibility_mask=inputs["compatibility_mask"],
            )
        variant_features = dict(features)
        variant_features["finding_evidence"] = finding.evidence_tokens.detach()
        variant_features["finding_logits"] = finding.finding_logits.detach()
        variant_features["pair_mask"] = finding.pair_mask
        aggregation, fusion, _ = _global_forward(model, variant_features)
        statistics[name] = _intervention_statistics(
            model,
            fusion.logits,
            features["global_logits"],
            variant_features,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed,
        )
        statistics[name]["roi_permutation"] = (
            permutation.cpu().tolist() if name == "shuffled_roi" else None
        )
    return statistics


def run_stage3_tiny_overfit(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = _load_config(args.config, args.config_overrides)
    hierarchy_config = config["dataset"]["hierarchy"]
    model_hierarchy_config = config["model"]["model_config"]["hierarchy_config"]
    model_hierarchy_config["finding_roi_strategy"] = args.finding_roi_strategy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model(config["model"], output_dir=str(output_dir))
    model = models["model"].to(device)
    _load_finding_head_checkpoint(model, args.finding_checkpoint)
    if args.roi_mode == "predicted":
        if not args.anatomy_checkpoint:
            raise ValueError(
                "--anatomy-checkpoint is required for predicted Stage 3 ROI."
            )
        _load_anatomy_head_checkpoint(model, args.anatomy_checkpoint)
    model.configure_hierarchy_training("global")

    dataset = HierarchyJsonlDataset(
        hierarchy_config["manifest_path"],
        hierarchy_config["image_root"],
        hierarchy_config["contract_path"],
        split=hierarchy_config.get("train_split", "train"),
        patch_grid_size=hierarchy_config["patch_grid_size"],
    )
    if args.indices_dir:
        indices = _canonical_split_indices(
            dataset,
            args.num_images,
            args.seed,
            Path(args.indices_dir) / "pilot_train_indices.json",
        )
    else:
        indices = _select_global_indices(dataset, args.num_images, args.seed)
    batch = hierarchy_collate_fn(
        [dataset[index] for index in indices],
        tokenizer=models["tokenizer"],
        image_processor=models["image_processor"],
        contract_path=hierarchy_config["contract_path"],
    )
    batch = _to_device(batch, device)

    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        vision = model.forward_vision_model(batch["pixel_values"])
        finding_queries = model._encode_prompt_bank(
            batch["encoded_finding_prompts"], model.num_findings, "finding"
        )
        global_queries = model._encode_prompt_bank(
            batch["encoded_global_prompts"], model.num_global_labels, "global"
        )
        if args.finding_roi_strategy == "full":
            finding_masks = torch.ones_like(batch["gt_anatomy_masks"])
        elif args.roi_mode == "predicted":
            anatomy_queries = model._encode_prompt_bank(
                batch["encoded_anatomy_prompts"], model.num_anatomies, "anatomy"
            )
            anatomy_prediction = model.anatomy_head(
                anatomy_queries,
                vision["image_patch_tokens"],
                anatomy_query_mask=batch["anatomy_query_mask"],
            )
            finding_masks = anatomy_prediction.roi_masks
        else:
            finding_masks = batch["gt_anatomy_masks"]
        finding = model.finding_head(
            finding_queries,
            vision["image_patch_tokens"],
            finding_masks,
            anatomy_query_mask=batch["anatomy_query_mask"],
            compatibility_mask=batch["compatibility_mask"],
        )
        global_queries, global_visual_tokens = model._prepare_radzero_global_inputs(
            global_queries, vision["vision_tokens"]
        )
        global_evidence = model.global_branch(
            global_queries,
            global_visual_tokens,
            global_query_mask=batch["global_query_mask"],
            temperature=model._radzero_attention_temperature(),
        )
    substitution_inputs = {
        "visual_tokens": vision["image_patch_tokens"].detach(),
        "finding_queries": finding_queries.detach(),
        "anatomy_query_mask": batch["anatomy_query_mask"],
        "compatibility_mask": batch["compatibility_mask"],
        "gt_anatomy_masks": batch["gt_anatomy_masks"].detach(),
    }
    deployment_inputs = {
        "pixel_values": batch["pixel_values"].detach(),
        "encoded_anatomy_prompts": batch["encoded_anatomy_prompts"],
        "encoded_finding_prompts": batch["encoded_finding_prompts"],
        "encoded_global_prompts": batch["encoded_global_prompts"],
        "anatomy_query_mask": batch["anatomy_query_mask"],
        "global_query_mask": batch["global_query_mask"],
        "compatibility_mask": batch["compatibility_mask"],
    }
    features = {
        "global_queries": global_queries.detach(),
        "finding_evidence": finding.evidence_tokens.detach(),
        "finding_logits": finding.finding_logits.detach(),
        "pair_mask": finding.pair_mask,
        "global_evidence": global_evidence.evidence_tokens.detach(),
        "global_query_mask": batch["global_query_mask"],
        "global_targets": batch["global_targets"],
        "global_valid_mask": batch["global_valid_mask"],
    }
    if args.global_only:
        features["finding_evidence"] = torch.zeros_like(features["finding_evidence"])
        features["finding_logits"] = torch.zeros_like(features["finding_logits"])
        features["pair_mask"] = torch.zeros_like(features["pair_mask"])
    del vision, finding, global_evidence, global_visual_tokens, batch

    upstream_snapshots = {
        "anatomy_head": _clone_module_state(model.anatomy_head),
        "finding_head": _clone_module_state(model.finding_head),
    }
    frozen_module_snapshots = {
        name: _clone_module_state(getattr(model, name))
        for name in ("vision_model", "text_model", "align_transformer")
    }
    model.train()
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    allowed_prefixes = ("finding_aggregator.", "evidence_fusion.")
    if not trainable_names or not all(
        name.startswith(allowed_prefixes) for name in trainable_names
    ):
        raise RuntimeError(
            f"Unexpected Stage 3 trainable parameters: {trainable_names}"
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    with torch.no_grad():
        initial_aggregation, initial_fusion, initial_loss = _global_forward(
            model, features
        )
    initial = _global_statistics(
        initial_fusion.logits,
        features["global_targets"],
        features["global_valid_mask"] & features["global_query_mask"],
        initial_fusion.gates.squeeze(-1),
        initial_aggregation,
    )
    initial["loss"] = float(initial_loss.float().item())

    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = _global_forward(model, features)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Stage 3 loss at step {step}: {loss.item()}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}.")
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().float().item()),
                    "gradient_norm": float(gradient_norm.detach().float().item()),
                }
            )

    model.eval()
    with torch.no_grad():
        final_aggregation, final_fusion, final_loss = _global_forward(model, features)
    features["global_logits"] = final_fusion.logits.detach()
    no_gt_audit = None
    if not args.global_only:
        with torch.no_grad():
            deployment_output = model.forward_hierarchy(**deployment_inputs)
            deployment_repeat = model.forward_hierarchy(**deployment_inputs)
        deployment_logits = deployment_output.fusion.logits.float()
        repeat_logits = deployment_repeat.fusion.logits.float()
        no_gt_delta = (deployment_logits - repeat_logits).abs()
        no_gt_audit = {
            "passed": bool(
                torch.allclose(
                    deployment_logits,
                    repeat_logits,
                    rtol=1e-3,
                    atol=1e-3,
                )
            ),
            "max_abs_logit_delta": float(no_gt_delta.max().item()),
            "input_gt_fields": [],
            "uses_gt_anatomy_masks": False,
            "uses_gt_anatomy_boxes": False,
            "uses_targets": False,
        }
    final = _global_statistics(
        final_fusion.logits,
        features["global_targets"],
        features["global_valid_mask"] & features["global_query_mask"],
        final_fusion.gates.squeeze(-1),
        final_aggregation,
    )
    final["loss"] = float(final_loss.float().item())
    final["bootstrap"] = bootstrap_multilabel_metrics(
        torch.sigmoid(final_fusion.logits),
        features["global_targets"],
        features["global_valid_mask"] & features["global_query_mask"],
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    with torch.no_grad():
        intervention_permutation, intervention_logits, interventions = (
            _run_stage3_interventions(
                model,
                features,
                final_aggregation,
                final_fusion,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
        roi_substitutions = _stage3_roi_substitutions(
            model,
            features,
            substitution_inputs,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    upstream_unchanged = {
        name: _module_state_unchanged(getattr(model, name), snapshot)
        for name, snapshot in upstream_snapshots.items()
    }
    frozen_module_audit = {
        name: {
            "unchanged": _module_state_unchanged(getattr(model, name), snapshot),
            "max_abs_parameter_delta": _module_state_max_abs_delta(
                getattr(model, name), snapshot
            ),
        }
        for name, snapshot in frozen_module_snapshots.items()
    }
    report = {
        "experiment": "stage3_real_tiny_overfit",
        "seed": args.seed,
        "num_images": args.num_images,
        "steps": args.steps,
        "roi_mode": args.roi_mode,
        "finding_roi_strategy": args.finding_roi_strategy,
        "global_only": args.global_only,
        "anatomy_checkpoint": args.anatomy_checkpoint,
        "uses_gt_anatomy_masks": args.roi_mode == "gt",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "finding_checkpoint": args.finding_checkpoint,
        "dataset_indices": indices,
        "trainable_parameters": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "upstream_unchanged": upstream_unchanged,
        "frozen_module_audit": frozen_module_audit,
        "no_gt_audit": no_gt_audit,
        "initial": initial,
        "final": final,
        "history": history,
        "intervention_permutation": intervention_permutation,
        "interventions": interventions,
        "roi_substitutions": roi_substitutions,
        "loss_decreased": final["loss"] < initial["loss"],
        "all_values_finite": all(
            math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"])
            for item in history
        ),
    }
    report_path = output_dir / "stage3_real_smoke.json"
    checkpoint_path = output_dir / "stage3_heads.pt"
    arrays_path = output_dir / "stage3_interventions.npz"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "finding_aggregator": model.finding_aggregator.state_dict(),
            "evidence_fusion": model.evidence_fusion.state_dict(),
        },
        checkpoint_path,
    )
    np.savez_compressed(
        arrays_path,
        normal_logits=final_fusion.logits.detach().float().cpu().numpy(),
        fusion_gates=final_fusion.gates.detach().float().cpu().numpy(),
        aggregator_attention=final_aggregation.attention_weights.detach()
        .float()
        .cpu()
        .numpy(),
        aggregator_null_attention=final_aggregation.null_attention.detach()
        .float()
        .cpu()
        .numpy(),
        **{
            f"{name}_logits": logits.detach().float().cpu().numpy()
            for name, logits in intervention_logits.items()
        },
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "initial_loss": initial["loss"],
                "final_loss": final["loss"],
                "initial_macro_auroc": initial["macro_auroc"],
                "final_macro_auroc": final["macro_auroc"],
                "initial_macro_auprc": initial["macro_auprc"],
                "final_macro_auprc": final["macro_auprc"],
                "initial_gate_mean": initial["gate_mean"],
                "final_gate_mean": final["gate_mean"],
                "upstream_unchanged": upstream_unchanged,
                "interventions": interventions,
                "loss_decreased": report["loss_decreased"],
            },
            indent=2,
        )
    )
    return report


def run_pilot_interventions(args):
    """Evaluate formal C-checkpoint interventions over fixed validation data."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = _load_config(args.config, args.config_overrides)
    config["model"]["model_config"]["hierarchy_config"]["finding_roi_strategy"] = "masked_attention"
    config["model"]["model_config"]["hierarchy_config"]["global_only"] = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model(config["model"], output_dir=str(output_dir))
    model = models["model"]
    base_encoder_state = {
        name: _clone_module_state(getattr(model, name))
        for name in ("vision_model", "text_model", "align_transformer")
    }
    _load_full_hierarchy_checkpoint(model, args.checkpoint)
    frozen_audit = {
        name: {
            "unchanged_from_radzero": _module_state_unchanged(getattr(model, name), state),
            "max_abs_parameter_delta": _module_state_max_abs_delta(getattr(model, name), state),
        }
        for name, state in base_encoder_state.items()
    }
    model.to(device).eval()

    hierarchy = config["dataset"]["hierarchy"]
    dataset = HierarchyJsonlDataset(
        hierarchy["manifest_path"], hierarchy["image_root"], hierarchy["contract_path"],
        split=hierarchy.get("eval_split", "valid"), patch_grid_size=hierarchy["patch_grid_size"],
    )
    indices_path = Path(args.indices_dir) / "pilot_valid_indices.json"
    payload = json.loads(indices_path.read_text(encoding="utf-8"))
    indices = [int(index) for index in payload.get("indices", payload)]
    variant_names = (
        "normal", "gt_roi_substitution", "full_roi_substitution", "shuffled_roi",
        "zero_evidence_token", "zero_finding_logits", "zero_full_finding_state",
        "shuffled_full_finding_state", "finding_only", "global_only",
    )
    collected = {name: [] for name in variant_names}
    targets, valid_masks, gates = [], [], []
    roi_values = {"mean": [], "max": [], "area": [], "empty": [], "gt_area": []}
    no_gt_audit = None

    for start in range(0, len(indices), args.batch_size):
        samples = [dataset[index] for index in indices[start : start + args.batch_size]]
        batch = hierarchy_collate_fn(
            samples, tokenizer=models["tokenizer"], image_processor=models["image_processor"],
            contract_path=hierarchy["contract_path"],
        )
        batch = _to_device(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            vision = model.forward_vision_model(batch["pixel_values"])
            anatomy_queries = model._encode_prompt_bank(batch["encoded_anatomy_prompts"], model.num_anatomies, "anatomy")
            finding_queries = model._encode_prompt_bank(batch["encoded_finding_prompts"], model.num_findings, "finding")
            global_queries = model._encode_prompt_bank(batch["encoded_global_prompts"], model.num_global_labels, "global")
            anatomy = model.anatomy_head(
                anatomy_queries, vision["image_patch_tokens"], anatomy_query_mask=batch["anatomy_query_mask"]
            )
            finding = model.finding_head(
                finding_queries, vision["image_patch_tokens"], anatomy.roi_masks,
                anatomy_query_mask=batch["anatomy_query_mask"], compatibility_mask=batch["compatibility_mask"],
            )
            global_queries, full_tokens = model._prepare_radzero_global_inputs(global_queries, vision["vision_tokens"])
            global_evidence = model.global_branch(
                global_queries, full_tokens, global_query_mask=batch["global_query_mask"],
                temperature=model._radzero_attention_temperature(),
            )
            features = {
                "global_queries": global_queries,
                "finding_evidence": finding.evidence_tokens,
                "finding_logits": finding.finding_logits,
                "pair_mask": finding.pair_mask,
                "global_evidence": global_evidence.evidence_tokens,
                "global_query_mask": batch["global_query_mask"],
                "global_targets": batch["global_targets"],
                "global_valid_mask": batch["global_valid_mask"],
            }
            normal_aggregation, normal_fusion, _ = _global_forward(model, features)
            logits = {"normal": normal_fusion.logits}

            roi_masks = {
                "gt_roi_substitution": batch["gt_anatomy_masks"],
                "full_roi_substitution": torch.ones_like(batch["gt_anatomy_masks"]),
                "shuffled_roi": batch["gt_anatomy_masks"][torch.roll(torch.arange(len(samples), device=device), 1)],
            }
            for name, masks in roi_masks.items():
                replacement_finding = model.finding_head(
                    finding_queries, vision["image_patch_tokens"], masks,
                    anatomy_query_mask=batch["anatomy_query_mask"], compatibility_mask=batch["compatibility_mask"],
                )
                variant = dict(features)
                variant["finding_evidence"] = replacement_finding.evidence_tokens
                variant["finding_logits"] = replacement_finding.finding_logits
                variant["pair_mask"] = replacement_finding.pair_mask
                _, variant_fusion, _ = _global_forward(model, variant)
                logits[name] = variant_fusion.logits

            permutation = torch.roll(torch.arange(len(samples), device=device), 1)
            feature_variants = {}
            feature_variants["zero_evidence_token"] = {**features, "finding_evidence": torch.zeros_like(features["finding_evidence"])}
            feature_variants["zero_finding_logits"] = {**features, "finding_logits": torch.zeros_like(features["finding_logits"])}
            feature_variants["zero_full_finding_state"] = {
                **features, "finding_evidence": torch.zeros_like(features["finding_evidence"]),
                "finding_logits": torch.zeros_like(features["finding_logits"]), "pair_mask": torch.zeros_like(features["pair_mask"]),
            }
            feature_variants["shuffled_full_finding_state"] = {
                **features, "finding_evidence": features["finding_evidence"][permutation],
                "finding_logits": features["finding_logits"][permutation], "pair_mask": features["pair_mask"][permutation],
            }
            for name, variant in feature_variants.items():
                _, variant_fusion, _ = _global_forward(model, variant)
                logits[name] = variant_fusion.logits
            logits["finding_only"] = _forced_gate_logits(model, normal_aggregation, features, 1.0)
            logits["global_only"] = _forced_gate_logits(model, normal_aggregation, features, 0.0)

        for name in variant_names:
            collected[name].append(logits[name].float().cpu())
        targets.append(batch["global_targets"].float().cpu())
        valid_masks.append((batch["global_valid_mask"] & batch["global_query_mask"]).cpu())
        gates.append(normal_fusion.gates.squeeze(-1).float().cpu())
        masks = anatomy.roi_masks.float().cpu()
        gt_masks = batch["gt_anatomy_masks"].float().cpu()
        roi_values["mean"].append(masks.mean(dim=-1))
        roi_values["max"].append(masks.max(dim=-1).values)
        roi_values["area"].append((masks > 0.5).float().mean(dim=-1))
        roi_values["empty"].append((masks > 0.5).sum(dim=-1) == 0)
        roi_values["gt_area"].append((gt_masks > 0.5).float().mean(dim=-1))

        if no_gt_audit is None:
            inference_inputs = {
                key: batch[key] for key in (
                    "pixel_values", "encoded_anatomy_prompts", "encoded_finding_prompts",
                    "encoded_global_prompts", "anatomy_query_mask", "global_query_mask", "compatibility_mask"
                )
            }
            with torch.no_grad():
                first = model.forward_hierarchy(**inference_inputs).fusion.logits.float()
                perturbed = model.forward_hierarchy(
                    **inference_inputs,
                    gt_anatomy_masks=torch.rand_like(batch["gt_anatomy_masks"]),
                    anatomy_targets=torch.rand_like(batch["anatomy_targets"]),
                    finding_targets=torch.rand_like(batch["finding_targets"]),
                    global_targets=torch.rand_like(batch["global_targets"]),
                    return_loss=False,
                ).fusion.logits.float()
            delta = (first - perturbed).abs()
            no_gt_audit = {"passed": bool(torch.allclose(first, perturbed, rtol=1e-3, atol=1e-3)), "max_abs_logit_delta": float(delta.max().item())}

    targets = torch.cat(targets)
    valid = torch.cat(valid_masks).bool()
    normal = torch.cat(collected["normal"])
    reports = {}
    for offset, name in enumerate(variant_names):
        variant_logits = torch.cat(collected[name])
        metrics = multilabel_metrics(torch.sigmoid(variant_logits), targets, valid)
        reports[name] = {
            "macro_auroc": metrics["auroc"], "macro_auprc": metrics["auprc"], "macro_f1": metrics["f1"],
            "mean_absolute_logit_delta": float((variant_logits - normal).abs()[valid].mean().item()),
            "bootstrap": bootstrap_multilabel_metrics(
                torch.sigmoid(variant_logits), targets, valid, samples=args.bootstrap_samples, seed=args.bootstrap_seed + offset
            ),
        }
    gate_tensor = torch.cat(gates)
    roi = {key: torch.cat(values) for key, values in roi_values.items()}
    report = {
        "experiment": "formal_c_checkpoint_interventions", "checkpoint": args.checkpoint,
        "validation_samples": len(indices), "seed": args.seed, "shuffle_rule": "deterministic batch roll by one",
        "metrics": reports,
        "gate_diagnostics": {
            "mean": float(gate_tensor.mean().item()), "std": float(gate_tensor.std().item()),
            "mean_per_label": gate_tensor.mean(dim=0).tolist(), "std_per_label": gate_tensor.std(dim=0).tolist(),
        },
        "roi_diagnostics": {
            "mask_mean": float(roi["mean"].mean().item()), "mask_max": float(roi["max"].mean().item()),
            "predicted_area": float(roi["area"].mean().item()), "gt_area": float(roi["gt_area"].mean().item()),
            "empty_mask_rate": float(roi["empty"].float().mean().item()),
        },
        "no_gt_audit": no_gt_audit, "frozen_module_audit": frozen_audit,
    }
    report_path = output_dir / "formal_interventions.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": reports}, indent=2))
    return report


def run_pilot_a0(args):
    """Evaluate the released frozen RadZero on the fixed pilot validation split."""
    device = torch.device(args.device)
    config = _load_config(args.config, args.config_overrides)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model(config["model"], output_dir=str(output_dir))
    model = models["model"].to(device).eval()
    hierarchy = config["dataset"]["hierarchy"]
    dataset = HierarchyJsonlDataset(
        hierarchy["manifest_path"], hierarchy["image_root"], hierarchy["contract_path"],
        split=hierarchy.get("eval_split", "valid"), patch_grid_size=hierarchy["patch_grid_size"],
    )
    payload = json.loads((Path(args.indices_dir) / "pilot_valid_indices.json").read_text(encoding="utf-8"))
    indices = [int(index) for index in payload.get("indices", payload)]
    logits, targets, valid_masks = [], [], []
    for start in range(0, len(indices), args.batch_size):
        samples = [dataset[index] for index in indices[start : start + args.batch_size]]
        batch = hierarchy_collate_fn(
            samples, tokenizer=models["tokenizer"], image_processor=models["image_processor"],
            contract_path=hierarchy["contract_path"],
        )
        batch = _to_device(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model.compute_logits(
                pixel_values=batch["pixel_values"],
                encoded_key_phrases=[batch["encoded_global_prompts"]],
            )
        logits.append(output["logits"].float().cpu())
        targets.append(batch["global_targets"].float().cpu())
        valid_masks.append((batch["global_valid_mask"] & batch["global_query_mask"]).cpu())
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    valid = torch.cat(valid_masks).bool()
    metrics = multilabel_metrics(torch.sigmoid(logits), targets, valid, include_per_class=True)
    report = {
        "experiment": "A0_original_radzero", "validation_samples": len(indices),
        "test_accessed": False, "metrics": metrics,
        "bootstrap": bootstrap_multilabel_metrics(
            torch.sigmoid(logits), targets, valid, samples=args.bootstrap_samples, seed=args.bootstrap_seed
        ),
    }
    report_path = output_dir / "a0_original_radzero.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": metrics}, indent=2))
    return report


def _yaml_scalar(value):
    return json.dumps(str(value))


def run_pilot_orchestrator(args):
    """Create one reproducible A1/B/C pilot plan and optionally execute it.

    The generated variant files are ordinary Config YAML overlays, so the
    actual training remains in the canonical ``run.py`` entry point.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_overrides = list(args.config_overrides)
    config = _load_config(args.config, config_overrides)
    hierarchy = config["dataset"]["hierarchy"]
    train_dataset = HierarchyJsonlDataset(
        hierarchy["manifest_path"], hierarchy["image_root"], hierarchy["contract_path"],
        split=hierarchy.get("train_split", "train"), patch_grid_size=hierarchy["patch_grid_size"],
    )
    valid_dataset = HierarchyJsonlDataset(
        hierarchy["manifest_path"], hierarchy["image_root"], hierarchy["contract_path"],
        split=hierarchy.get("eval_split", "valid"), patch_grid_size=hierarchy["patch_grid_size"],
    )
    train_path = output_dir / "pilot_train_indices.json"
    valid_path = output_dir / "pilot_valid_indices.json"
    train_indices = _canonical_split_indices(train_dataset, args.train_images, args.seed, train_path)
    valid_indices = _canonical_split_indices(valid_dataset, args.valid_images, args.seed + 1, valid_path)
    variants = {
        "A1_global_only": {"global_only": True, "finding_roi_strategy": "full",
                           "schedule": [("global", args.max_steps)]},
        "B_full_image_hierarchy": {"global_only": False, "finding_roi_strategy": "full",
                                    "schedule": [(stage, args.max_steps) for stage in ("anatomy", "finding", "global", "joint")]},
        "C_predicted_roi": {"global_only": False, "finding_roi_strategy": "masked_attention",
                             "schedule": [(stage, args.max_steps) for stage in ("anatomy", "finding", "global", "joint")]},
    }
    manifest = {
        "seed": args.seed, "train_count": len(train_indices), "valid_count": len(valid_indices),
        "test_accessed": False, "train_indices": str(train_path), "valid_indices": str(valid_path),
        "variants": {},
    }
    for name, spec in variants.items():
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        overlay = variant_dir / "pilot_variant.yaml"
        phases = "\n".join(
            f"        - stage: {stage}\n          num_train_epochs: 1\n          max_steps: {steps}"
            for stage, steps in spec["schedule"]
        )
        overlay.write_text(
            "model:\n  model_config:\n"
            f"    pretrained_dir: {_yaml_scalar(args.pretrained_dir)}\n"
            "    hierarchy_config:\n"
            f"      global_only: {'true' if spec['global_only'] else 'false'}\n"
            f"      finding_roi_strategy: {spec['finding_roi_strategy']}\n"
            "      training:\n        teacher_forcing_start_ratio: 0.0\n"
            "        teacher_forcing_end_ratio: 0.0\n"
            "      training_schedule:\n" + phases + "\n"
            "dataset:\n  hierarchy:\n"
            f"    train_indices_path: {_yaml_scalar(train_path)}\n"
            f"    eval_indices_path: {_yaml_scalar(valid_path)}\n"
            "train:\n"
            f"  seed: {args.seed}\n"
            f"  per_device_train_batch_size: {args.batch_size}\n"
            f"  per_device_eval_batch_size: {args.batch_size}\n"
            f"  eval_steps: {args.eval_steps}\n"
            f"  save_steps: {args.eval_steps}\n"
            "  save_total_limit: 2\n"
            "  evaluation_strategy: steps\n  save_strategy: steps\n"
            "experiment:\n"
            f"  output_root_dir: {_yaml_scalar(variant_dir)}\n"
            "  project: hierarchy_pilot\n"
            f"  user: {_yaml_scalar(name)}\n"
            "  name: run\n"
            "",
            encoding="utf-8",
        )
        command = [
            sys.executable, "-m", "exp.cxr_pt.run", "--cfg-path", args.config,
            "--add_cfg_list", *config_overrides, str(overlay),
            "--train", "true", "--inference", "true", "--no_report",
        ]
        manifest["variants"][name] = {"overlay": str(overlay), "command": command}
        if args.execute:
            subprocess.run(command, cwd=str(Path.cwd()), check=True)
            reports = sorted(variant_dir.rglob("*-hierarchy.json"))
            if reports:
                manifest["variants"][name]["report"] = str(reports[-1])
    manifest_path = output_dir / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def run_pilot_summary(args):
    """Aggregate pilot artifacts into a machine-readable and Markdown report."""
    if getattr(args, "pilot_dir", None):
        return _run_complete_pilot_summary(args)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = []
    for name, spec in manifest.get("variants", {}).items():
        report_path = spec.get("report")
        report = json.loads(Path(report_path).read_text(encoding="utf-8")) if report_path and Path(report_path).is_file() else None
        rows.append({
            "method": name,
            "report": report_path,
            "available": report is not None,
            "metrics": report.get("metrics", {}) if report else {},
        })
    output = {"manifest": args.manifest, "test_accessed": manifest.get("test_accessed", False), "methods": rows}
    path = Path(args.output or Path(args.manifest).with_name("pilot_summary.json"))
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def _final_eval_metrics(run_root, stage):
    states = list(Path(run_root).glob(f"**/{stage}/checkpoint-*/trainer_state.json"))
    if not states:
        raise FileNotFoundError(f"No {stage} trainer state found below {run_root}.")
    state_path = max(
        states, key=lambda path: int(path.parent.name.removeprefix("checkpoint-"))
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    evaluations = [row for row in state["log_history"] if "eval_loss" in row]
    if not evaluations:
        raise ValueError(f"No final evaluation metrics found in {state_path}.")
    return state_path, evaluations[-1]


def _global_labels_for_indices(config, indices_path):
    hierarchy = config["dataset"]["hierarchy"]
    dataset = HierarchyJsonlDataset(
        hierarchy["manifest_path"], hierarchy["image_root"], hierarchy["contract_path"],
        split=hierarchy.get("eval_split", "valid"), patch_grid_size=hierarchy["patch_grid_size"],
    )
    payload = json.loads(Path(indices_path).read_text(encoding="utf-8"))
    indices = [int(index) for index in payload.get("indices", payload)]
    targets = torch.zeros(len(indices), len(dataset.global_index))
    valid = torch.zeros_like(targets, dtype=torch.bool)
    with dataset.manifest_path.open("rb") as source:
        for output_index, dataset_index in enumerate(indices):
            source.seek(dataset.offsets[dataset_index])
            row = json.loads(source.readline())
            for label, value in row.get("global", {}).items():
                label_index = dataset.global_index[label]
                targets[output_index, label_index] = value
                valid[output_index, label_index] = True
    return targets, valid


def _checkpoint_capacity(state_path, prefixes):
    checkpoint = state_path.parent
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    total = 0
    trainable = 0
    with safe_open(model_path, framework="pt") as source:
        for name in source.keys():
            count = math.prod(source.get_slice(name).get_shape())
            total += count
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                trainable += count
    optimizer_path = checkpoint / "optimizer.pt"
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "optimizer_state_bytes": optimizer_path.stat().st_size if optimizer_path.is_file() else None,
    }


def _comparison_interpretation(methods, interventions):
    metric = lambda method, name: methods[method]["global"][name]
    deltas = {
        "B_minus_A1": {
            "macro_auroc": metric("B", "macro_auroc") - metric("A1", "macro_auroc"),
            "macro_auprc": metric("B", "macro_auprc") - metric("A1", "macro_auprc"),
        },
        "C_minus_B": {
            "macro_auroc": metric("C", "macro_auroc") - metric("B", "macro_auroc"),
            "macro_auprc": metric("C", "macro_auprc") - metric("B", "macro_auprc"),
        },
        "C_minus_A1": {
            "macro_auroc": metric("C", "macro_auroc") - metric("A1", "macro_auroc"),
            "macro_auprc": metric("C", "macro_auprc") - metric("A1", "macro_auprc"),
        },
        "C_minus_A0": {
            "macro_auroc": metric("C", "macro_auroc") - metric("A0", "macro_auroc"),
            "macro_auprc": metric("C", "macro_auprc") - metric("A0", "macro_auprc"),
        },
    }
    normal = interventions["normal"]
    mechanism = {
        name: {
            "macro_auroc": normal["macro_auroc"] - values["macro_auroc"],
            "macro_auprc": normal["macro_auprc"] - values["macro_auprc"],
        }
        for name, values in interventions.items() if name != "normal"
    }
    findings = [
        "The complete Stage 1 -> Stage 2 -> Stage 3 -> joint schedule completed with finite held-out metrics.",
        "B and C did not improve the primary Global Macro AUPRC over A1 under this budget; their Global AUROC was higher.",
        "C and B were effectively tied, so this pilot does not show a material benefit from predicted anatomy conditioning.",
        "Shuffling the full grounded finding state caused a large degradation, providing evidence that Stage 3 uses Stage 2 state.",
        "C outperformed zero-shot A0 on AUROC and AUPRC, but that comparison also includes supervised training and does not isolate hierarchy value.",
    ]
    return deltas, mechanism, findings


def _render_complete_pilot_markdown(report):
    methods = report["methods"]
    lines = [
        "# Hierarchy 5k/1k Directional Pilot", "",
        "> Scope: 5,000 train / 1,000 validation, seed 42, 500 optimizer steps per scheduled phase. Test split was not used.", "",
        "## Primary Results", "",
        "| Method | Anatomy Dice | Finding AUROC | Finding AUPRC | Global AUROC | Global AUPRC | Global F1 | Trainable params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A0", "A1", "B", "C", "D"):
        row = methods[name]
        fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['name']} | {fmt(row['anatomy_dice'])} | {fmt(row['finding_auroc'])} | "
            f"{fmt(row['finding_auprc'])} | {fmt(row['global']['macro_auroc'])} | "
            f"{fmt(row['global']['macro_auprc'])} | {fmt(row['global']['macro_f1'])} | "
            f"{row['capacity']['trainable_parameters'] if row.get('capacity') else 'N/A'} |"
        )
    lines.extend(["", "## Same-checkpoint Interventions", "", "| Intervention | Global AUROC | Global AUPRC | F1 | Delta AUPRC vs normal |", "|---|---:|---:|---:|---:|"])
    normal_auprc = report["interventions"]["normal"]["macro_auprc"]
    for name, values in report["interventions"].items():
        lines.append(
            f"| {name} | {values['macro_auroc']:.4f} | {values['macro_auprc']:.4f} | "
            f"{values['macro_f1']:.4f} | {values['macro_auprc'] - normal_auprc:+.4f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {finding}" for finding in report["interpretation"])
    lines.extend([
        "", "These are directional findings under the current pilot budget, weak-label quality, and one training seed. They are not a final significance claim.",
        "", "## Audits", "",
        f"- No-GT inference: {'PASS' if report['audits']['no_gt']['passed'] else 'FAIL'} (max logit delta {report['audits']['no_gt']['max_abs_logit_delta']:.3g})",
        f"- Frozen RadZero encoders: {'PASS' if report['audits']['frozen_modules_passed'] else 'FAIL'}",
        f"- Predicted ROI area: {report['diagnostics']['roi']['predicted_area']:.4f}; GT ROI area: {report['diagnostics']['roi']['gt_area']:.4f}",
        f"- Mean fusion gate: {report['diagnostics']['gate']['mean']:.4f} (1 = finding branch, 0 = global branch)",
        "", "## Decision", "",
        "The infrastructure is ready for this 5k pilot and the run is complete. The current results support keeping predicted-ROI hierarchy as an experimental mainline, but do not support claiming it is better than the supervised global-only or M=1 baselines. Before scaling, improve or calibrate Stage 1 ROI masks and repeat with multiple seeds.", "",
    ])
    return "\n".join(lines)


def _run_complete_pilot_summary(args):
    pilot_dir = Path(args.pilot_dir)
    config = _load_config(args.config, args.config_overrides)
    targets, valid = _global_labels_for_indices(
        config, pilot_dir / "pilot_valid_indices.json"
    )
    definitions = {
        "A1": ("A1 Supervised Global-only", "A1_global_only", "global", ("global_branch", "evidence_fusion")),
        "B": ("B Full-image Hierarchy / M=1", "B_full_image_hierarchy", "joint", ("anatomy_head", "finding_head", "finding_aggregator", "global_branch", "evidence_fusion")),
        "C": ("C Predicted ROI Mainline", "C_predicted_roi", "joint", ("anatomy_head", "finding_head", "finding_aggregator", "global_branch", "evidence_fusion")),
    }
    methods = {}
    for key, (display_name, directory, stage, prefixes) in definitions.items():
        root = pilot_dir / directory
        state_path, evaluation = _final_eval_metrics(root, stage)
        npz_paths = list(root.glob("**/inference/hierarchy/*-hierarchy.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"No inference archive found below {root}.")
        with np.load(npz_paths[-1]) as archive:
            logits = torch.from_numpy(archive["global_logits"].copy())
        bootstrap = bootstrap_multilabel_metrics(
            torch.sigmoid(logits), targets, valid,
            samples=args.bootstrap_samples, seed=args.bootstrap_seed,
        )
        methods[key] = {
            "name": display_name,
            "checkpoint": str(state_path.parent),
            "anatomy_dice": evaluation.get("eval_anatomy_dice") if key != "A1" else None,
            "finding_auroc": evaluation.get("eval_finding_auroc") if key != "A1" else None,
            "finding_auprc": evaluation.get("eval_finding_auprc") if key != "A1" else None,
            "global": {
                "macro_auroc": evaluation["eval_global_auroc"],
                "macro_auprc": evaluation["eval_global_auprc"],
                "macro_f1": evaluation["eval_global_f1"],
                "loss": evaluation["eval_global_loss"],
                "bootstrap": bootstrap,
            },
            "capacity": _checkpoint_capacity(state_path, prefixes),
            "optimizer_steps": int(state_path.parent.name.removeprefix("checkpoint-")),
        }
    a0 = json.loads((pilot_dir / "A0_original_radzero" / "a0_original_radzero.json").read_text(encoding="utf-8"))
    methods["A0"] = {
        "name": "A0 Original RadZero", "checkpoint": None,
        "anatomy_dice": None, "finding_auroc": None, "finding_auprc": None,
        "global": {
            "macro_auroc": a0["metrics"]["auroc"], "macro_auprc": a0["metrics"]["auprc"],
            "macro_f1": a0["metrics"]["f1"], "loss": None, "bootstrap": a0["bootstrap"],
        },
        "capacity": None, "optimizer_steps": 0,
    }
    intervention_report = json.loads((pilot_dir / "interventions" / "formal_interventions.json").read_text(encoding="utf-8"))
    interventions = intervention_report["metrics"]
    gt_roi = interventions["gt_roi_substitution"]
    methods["D"] = {
        "name": "D C-checkpoint + GT ROI substitution", "checkpoint": intervention_report["checkpoint"],
        "anatomy_dice": None, "finding_auroc": None, "finding_auprc": None,
        "global": {"macro_auroc": gt_roi["macro_auroc"], "macro_auprc": gt_roi["macro_auprc"], "macro_f1": gt_roi["macro_f1"], "loss": None, "bootstrap": gt_roi["bootstrap"]},
        "capacity": methods["C"]["capacity"], "optimizer_steps": 0, "intervention": True,
    }
    deltas, mechanism_deltas, interpretation = _comparison_interpretation(methods, interventions)
    frozen = intervention_report["frozen_module_audit"]
    report = {
        "experiment": "hierarchy_5k_1k_directional_pilot", "primary_checkpoint_rule": "last-step",
        "train_samples": 5000, "validation_samples": 1000, "seed": 42,
        "test_accessed": False, "primary_metric": "Global Macro AUPRC",
        "methods": methods, "method_deltas": deltas,
        "interventions": interventions, "mechanism_deltas_from_normal": mechanism_deltas,
        "diagnostics": {"gate": intervention_report["gate_diagnostics"], "roi": intervention_report["roi_diagnostics"]},
        "audits": {
            "no_gt": intervention_report["no_gt_audit"], "frozen_modules": frozen,
            "frozen_modules_passed": all(value["unchanged_from_radzero"] for value in frozen.values()),
        },
        "interpretation": interpretation,
        "limitations": ["single training seed", "500 optimizer steps per scheduled phase", "weak-label supervision", "validation-only directional pilot"],
    }
    json_path = Path(args.output or pilot_dir / "pilot_summary.json")
    markdown_path = json_path.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_complete_pilot_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "method_deltas": deltas}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Hierarchy real-data experiments.")
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    indices_parser = subparsers.add_parser(
        "pilot-indices", help="Create or validate fixed canonical pilot indices."
    )
    indices_parser.add_argument("--config", default="exp/cxr_pt/config.yaml")
    indices_parser.add_argument(
        "--config-overrides",
        nargs="+",
        default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"],
    )
    indices_parser.add_argument(
        "--output-dir", default="/home/user/gxq/hierarchy-pilot"
    )
    indices_parser.add_argument("--train-images", type=int, default=5000)
    indices_parser.add_argument("--valid-images", type=int, default=1000)
    indices_parser.add_argument("--seed", type=int, default=42)
    orchestrator = subparsers.add_parser("pilot-run", help="Create/execute reproducible A1/B/C pilot plan.")
    orchestrator.add_argument("--config", default="exp/cxr_pt/config.yaml")
    orchestrator.add_argument("--config-overrides", nargs="+", default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"])
    orchestrator.add_argument("--output-dir", default="/home/user/gxq/hierarchy-pilot")
    orchestrator.add_argument("--train-images", type=int, default=5000)
    orchestrator.add_argument("--valid-images", type=int, default=1000)
    orchestrator.add_argument("--seed", type=int, default=42)
    orchestrator.add_argument("--max-steps", type=int, default=500)
    orchestrator.add_argument("--eval-steps", type=int, default=100)
    orchestrator.add_argument("--batch-size", type=int, default=8)
    orchestrator.add_argument("--pretrained-dir", default="/home/user/gxq/radzero-model")
    orchestrator.add_argument("--execute", action="store_true")
    summary_parser = subparsers.add_parser("pilot-summary", help="Generate a unified pilot report.")
    summary_source = summary_parser.add_mutually_exclusive_group(required=True)
    summary_source.add_argument("--manifest")
    summary_source.add_argument("--pilot-dir")
    summary_parser.add_argument("--output")
    summary_parser.add_argument("--config", default="exp/cxr_pt/config.yaml")
    summary_parser.add_argument("--config-overrides", nargs="+", default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"])
    summary_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    summary_parser.add_argument("--bootstrap-seed", type=int, default=42)
    intervention_parser = subparsers.add_parser(
        "pilot-interventions", help="Evaluate formal C-checkpoint interventions."
    )
    intervention_parser.add_argument("--config", default="exp/cxr_pt/config.yaml")
    intervention_parser.add_argument("--config-overrides", nargs="+", default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"])
    intervention_parser.add_argument("--checkpoint", required=True)
    intervention_parser.add_argument("--indices-dir", required=True)
    intervention_parser.add_argument("--output-dir", required=True)
    intervention_parser.add_argument("--device", default="cuda:0")
    intervention_parser.add_argument("--batch-size", type=int, default=8)
    intervention_parser.add_argument("--seed", type=int, default=42)
    intervention_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    intervention_parser.add_argument("--bootstrap-seed", type=int, default=42)
    a0_parser = subparsers.add_parser("pilot-a0", help="Evaluate original frozen RadZero.")
    a0_parser.add_argument("--config", default="exp/cxr_pt/config.yaml")
    a0_parser.add_argument("--config-overrides", nargs="+", default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"])
    a0_parser.add_argument("--indices-dir", required=True)
    a0_parser.add_argument("--output-dir", required=True)
    a0_parser.add_argument("--device", default="cuda:0")
    a0_parser.add_argument("--batch-size", type=int, default=8)
    a0_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    a0_parser.add_argument("--bootstrap-seed", type=int, default=42)
    stage2 = subparsers.add_parser("stage2-tiny-overfit")
    stage2.add_argument("--config", default="exp/cxr_pt/config.yaml")
    stage2.add_argument(
        "--config-overrides",
        nargs="+",
        default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"],
    )
    stage2.add_argument(
        "--output-dir", default="/home/user/gxq/stage2-validation/tiny-overfit"
    )
    stage2.add_argument("--device", default="cuda:0")
    stage2.add_argument("--num-images", type=int, default=16)
    stage2.add_argument("--steps", type=int, default=500)
    stage2.add_argument(
        "--roi-mode",
        choices=("gt", "full", "predicted", "shuffled"),
        default="gt",
    )
    stage2.add_argument("--anatomy-checkpoint")
    stage2.add_argument(
        "--finding-roi-strategy",
        choices=("masked_attention", "full", "feature_roi", "full_refiner"),
        default="masked_attention",
    )
    stage2.add_argument("--learning-rate", type=float, default=1e-2)
    stage2.add_argument("--weight-decay", type=float, default=0.0)
    stage2.add_argument("--max-grad-norm", type=float, default=10.0)
    stage2.add_argument("--log-every", type=int, default=10)
    stage2.add_argument("--seed", type=int, default=20260930)
    stage1 = subparsers.add_parser("stage1-pilot")
    stage1.add_argument("--config", default="exp/cxr_pt/config.yaml")
    stage1.add_argument(
        "--config-overrides",
        nargs="+",
        default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"],
    )
    stage1.add_argument(
        "--output-dir", default="/home/user/gxq/stage1-validation/pilot"
    )
    stage1.add_argument("--device", default="cuda:0")
    stage1.add_argument("--train-images", type=int, default=256)
    stage1.add_argument("--valid-images", type=int, default=128)
    stage1.add_argument("--batch-size", type=int, default=16)
    stage1.add_argument("--steps", type=int, default=200)
    stage1.add_argument("--learning-rate", type=float, default=1e-2)
    stage1.add_argument("--weight-decay", type=float, default=0.0)
    stage1.add_argument("--max-grad-norm", type=float, default=10.0)
    stage1.add_argument("--log-every", type=int, default=20)
    stage1.add_argument("--seed", type=int, default=20260930)
    stage1.add_argument("--indices-dir")
    pilot = subparsers.add_parser("stage2-pilot")
    pilot.add_argument("--config", default="exp/cxr_pt/config.yaml")
    pilot.add_argument(
        "--config-overrides",
        nargs="+",
        default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"],
    )
    pilot.add_argument("--output-dir", default="/home/user/gxq/stage2-validation/pilot")
    pilot.add_argument("--device", default="cuda:0")
    pilot.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "masked_attention", "feature_roi", "full_refiner"],
        choices=("full", "masked_attention", "feature_roi", "full_refiner"),
    )
    pilot.add_argument("--train-images", type=int, default=256)
    pilot.add_argument("--valid-images", type=int, default=128)
    pilot.add_argument("--batch-size", type=int, default=16)
    pilot.add_argument("--steps", type=int, default=120)
    pilot.add_argument("--learning-rate", type=float, default=1e-3)
    pilot.add_argument("--weight-decay", type=float, default=0.0)
    pilot.add_argument("--max-grad-norm", type=float, default=10.0)
    pilot.add_argument("--log-every", type=int, default=20)
    pilot.add_argument("--seed", type=int, default=20260930)
    pilot.add_argument("--indices-dir")
    pilot.add_argument("--context-margin", type=float, default=0.0)
    pilot.add_argument(
        "--roi-mode", choices=("gt", "shuffled", "predicted"), default="gt"
    )
    pilot.add_argument("--anatomy-checkpoint")
    stage3 = subparsers.add_parser("stage3-tiny-overfit")
    stage3.add_argument("--config", default="exp/cxr_pt/config.yaml")
    stage3.add_argument(
        "--config-overrides",
        nargs="+",
        default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"],
    )
    stage3.add_argument(
        "--output-dir", default="/home/user/gxq/stage3-validation/tiny-overfit"
    )
    stage3.add_argument("--device", default="cuda:0")
    stage3.add_argument("--num-images", type=int, default=16)
    stage3.add_argument("--steps", type=int, default=500)
    stage3.add_argument("--learning-rate", type=float, default=1e-3)
    stage3.add_argument("--weight-decay", type=float, default=0.0)
    stage3.add_argument("--max-grad-norm", type=float, default=10.0)
    stage3.add_argument("--log-every", type=int, default=10)
    stage3.add_argument("--seed", type=int, default=20260930)
    stage3.add_argument("--finding-checkpoint", required=True)
    stage3.add_argument("--indices-dir")
    stage3.add_argument(
        "--finding-roi-strategy",
        choices=("masked_attention", "full"),
        default="masked_attention",
    )
    stage3.add_argument("--global-only", action="store_true")
    stage3.add_argument("--bootstrap-samples", type=int, default=1000)
    stage3.add_argument("--bootstrap-seed", type=int, default=42)
    stage3.add_argument("--roi-mode", choices=("predicted", "gt"), default="predicted")
    stage3.add_argument("--anatomy-checkpoint")
    args = parser.parse_args()
    if args.experiment == "pilot-indices":
        config = _load_config(args.config, args.config_overrides)
        hierarchy_config = config["dataset"]["hierarchy"]
        train_dataset = HierarchyJsonlDataset(
            hierarchy_config["manifest_path"],
            hierarchy_config["image_root"],
            hierarchy_config["contract_path"],
            split=hierarchy_config.get("train_split", "train"),
            patch_grid_size=hierarchy_config["patch_grid_size"],
        )
        valid_dataset = HierarchyJsonlDataset(
            hierarchy_config["manifest_path"],
            hierarchy_config["image_root"],
            hierarchy_config["contract_path"],
            split=hierarchy_config.get("eval_split", "valid"),
            patch_grid_size=hierarchy_config["patch_grid_size"],
        )
        output_dir = Path(args.output_dir)
        train_indices = _canonical_split_indices(
            train_dataset,
            args.train_images,
            args.seed,
            output_dir / "pilot_train_indices.json",
        )
        valid_indices = _canonical_split_indices(
            valid_dataset,
            args.valid_images,
            args.seed + 1,
            output_dir / "pilot_valid_indices.json",
        )
        report = {
            "seed": args.seed,
            "train_split": train_dataset.split,
            "valid_split": valid_dataset.split,
            "train_count": len(train_indices),
            "valid_count": len(valid_indices),
            "test_accessed": False,
            "train_indices": str(output_dir / "pilot_train_indices.json"),
            "valid_indices": str(output_dir / "pilot_valid_indices.json"),
        }
        report_path = output_dir / "pilot_indices.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.experiment == "pilot-run":
        run_pilot_orchestrator(args)
    elif args.experiment == "pilot-summary":
        run_pilot_summary(args)
    elif args.experiment == "pilot-interventions":
        run_pilot_interventions(args)
    elif args.experiment == "pilot-a0":
        run_pilot_a0(args)
    elif args.experiment == "stage2-tiny-overfit":
        run_stage2_tiny_overfit(args)
    elif args.experiment == "stage1-pilot":
        run_stage1_pilot(args)
    elif args.experiment == "stage2-pilot":
        run_stage2_pilot(args)
    elif args.experiment == "stage3-tiny-overfit":
        run_stage3_tiny_overfit(args)


if __name__ == "__main__":
    main()
