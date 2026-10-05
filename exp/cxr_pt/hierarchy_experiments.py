import argparse
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageDraw

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
