import argparse
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from common.utils import Config
from exp.cxr_pt.hierarchy_dataset import HierarchyJsonlDataset, hierarchy_collate_fn
from exp.cxr_pt.hierarchy_metrics import multilabel_metrics
from exp.cxr_pt.model import load_model


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
        torch.sigmoid(logits).reshape(-1, finding_count),
        targets.reshape(-1, finding_count),
        valid.reshape(-1, finding_count),
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
            }
        )
    return result


def _finding_forward(model, features):
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


def run_stage2_tiny_overfit(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = _load_config(args.config, args.config_overrides)
    hierarchy_config = config["dataset"]["hierarchy"]
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
        anatomy_masks, roi_permutation = _select_roi_masks(
            model,
            batch,
            vision["image_patch_tokens"],
            anatomy_queries,
            args,
        )
    features = {
        "visual_tokens": vision["image_patch_tokens"].detach(),
        "finding_queries": finding_queries.detach(),
        "anatomy_masks": anatomy_masks.detach(),
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
    if not trainable_names or not all(
        name.startswith("finding_head.") for name in trainable_names
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
            model.finding_head.parameters(), args.max_grad_norm
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
    torch.save(model.finding_head.state_dict(), checkpoint_path)
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


def _intervention_statistics(model, logits, normal_logits, features):
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
    return {
        "loss": float(loss.float().item()),
        "macro_auroc": metrics["auroc"],
        "macro_auprc": metrics["auprc"],
        "mean_absolute_logit_delta": float(delta.mean().item()),
        "max_absolute_logit_delta": float(delta.max().item()),
    }


def _run_stage3_interventions(model, features, normal_aggregation, normal_fusion):
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
    variants["zero_finding_evidence"] = fusion.logits

    shuffled_logits = dict(features)
    shuffled_logits["finding_logits"] = features["finding_logits"][permutation]
    _, fusion, _ = _global_forward(model, shuffled_logits)
    variants["shuffled_finding_logits"] = fusion.logits
    variants["finding_only_gate_1"] = _forced_gate_logits(
        model, normal_aggregation, features, 1.0
    )
    variants["global_only_gate_0"] = _forced_gate_logits(
        model, normal_aggregation, features, 0.0
    )

    statistics = {
        name: _intervention_statistics(model, logits, normal_fusion.logits, features)
        for name, logits in variants.items()
    }
    return permutation.cpu().tolist(), variants, statistics


def run_stage3_tiny_overfit(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = _load_config(args.config, args.config_overrides)
    hierarchy_config = config["dataset"]["hierarchy"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model(config["model"], output_dir=str(output_dir))
    model = models["model"].to(device)
    _load_finding_head_checkpoint(model, args.finding_checkpoint)
    model.configure_hierarchy_training("global")

    dataset = HierarchyJsonlDataset(
        hierarchy_config["manifest_path"],
        hierarchy_config["image_root"],
        hierarchy_config["contract_path"],
        split=hierarchy_config.get("train_split", "train"),
        patch_grid_size=hierarchy_config["patch_grid_size"],
    )
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
        finding = model.finding_head(
            finding_queries,
            vision["image_patch_tokens"],
            batch["gt_anatomy_masks"],
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
    final = _global_statistics(
        final_fusion.logits,
        features["global_targets"],
        features["global_valid_mask"] & features["global_query_mask"],
        final_fusion.gates.squeeze(-1),
        final_aggregation,
    )
    final["loss"] = float(final_loss.float().item())
    with torch.no_grad():
        intervention_permutation, intervention_logits, interventions = (
            _run_stage3_interventions(model, features, final_aggregation, final_fusion)
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
    stage2.add_argument("--learning-rate", type=float, default=1e-2)
    stage2.add_argument("--weight-decay", type=float, default=0.0)
    stage2.add_argument("--max-grad-norm", type=float, default=10.0)
    stage2.add_argument("--log-every", type=int, default=10)
    stage2.add_argument("--seed", type=int, default=20260930)
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
    args = parser.parse_args()
    if args.experiment == "stage2-tiny-overfit":
        run_stage2_tiny_overfit(args)
    elif args.experiment == "stage3-tiny-overfit":
        run_stage3_tiny_overfit(args)


if __name__ == "__main__":
    main()
