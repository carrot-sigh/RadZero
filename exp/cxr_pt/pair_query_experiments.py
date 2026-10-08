import argparse
import json
import subprocess
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from common.utils import Config
from exp.cxr_pt.hierarchy_dataset import HierarchyJsonlDataset, hierarchy_collate_fn
from exp.cxr_pt.model import load_model
from exp.cxr_pt.hierarchy_metrics import (
    multilabel_metrics,
    paired_bootstrap_metric_delta,
)
from exp.cxr_pt.pair_query_state import (
    anatomy_state_diagnostics,
    save_static_anatomy_state,
    sha256_file,
    sha256_json,
)


def _load_config(config_path, config_overrides):
    arguments = SimpleNamespace(
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
    return Config(arguments).config


def _load_indices(path, expected_count=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = [int(value) for value in payload.get("indices", payload)]
    if len(indices) != len(set(indices)):
        raise ValueError("Pilot indices must be unique.")
    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(f"Expected {expected_count} indices, found {len(indices)}.")
    return indices


def _checkpoint_weights_path(path):
    path = Path(path)
    return path / "model.safetensors" if path.is_dir() else path


def _left_right_pairs(labels):
    index = {label: position for position, label in enumerate(labels)}
    pairs = []
    for label, left_index in index.items():
        if not label.startswith("left "):
            continue
        counterpart = "right " + label[len("left ") :]
        if counterpart in index:
            pairs.append((left_index, index[counterpart]))
    return pairs


def build_static_state(args):
    config = _load_config(args.config, args.config_overrides)
    hierarchy = config["model"]["model_config"]["hierarchy_config"]
    hierarchy["finding_connection_mode"] = "dynamic_pair_query"
    hierarchy["pretrained_anatomy_checkpoint"] = args.anatomy_checkpoint
    hierarchy.pop("static_anatomy_state_path", None)
    dataset_config = config["dataset"]["hierarchy"]
    dataset = HierarchyJsonlDataset(
        dataset_config["manifest_path"],
        dataset_config["image_root"],
        dataset_config["contract_path"],
        split=dataset_config.get("train_split", "train"),
        patch_grid_size=dataset_config["patch_grid_size"],
    )
    indices = _load_indices(args.indices, expected_count=args.expected_count)
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Pilot indices are outside the canonical train split.")
    models = load_model(config["model"], output_dir=str(Path(args.output).parent))
    model = models["model"].to(args.device).eval()
    for module_name in (*model.training_controller.ENCODER_MODULES, "anatomy_head"):
        getattr(model, module_name).eval()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(
            hierarchy_collate_fn,
            tokenizer=models["tokenizer"],
            image_processor=models["image_processor"],
            contract_path=dataset_config["contract_path"],
        ),
    )
    total = torch.zeros(model.num_anatomies, model.hidden_size, dtype=torch.float64)
    total_square = torch.zeros_like(total)
    diagnostic_states = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(args.device)
            encoded = {
                key: value.to(args.device)
                for key, value in batch["encoded_anatomy_prompts"].items()
            }
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                visual = model.forward_vision_model(pixel_values)["image_patch_tokens"]
                anatomy_queries = model._encode_prompt_bank(
                    encoded, model.num_anatomies, "anatomy"
                )
                states = model.anatomy_head(
                    anatomy_queries,
                    visual,
                    anatomy_query_mask=batch["anatomy_query_mask"].to(args.device),
                ).evidence_tokens
            states = states.float().cpu()
            total += states.double().sum(dim=0)
            total_square += states.double().square().sum(dim=0)
            remaining = args.diagnostic_samples - sum(
                value.size(0) for value in diagnostic_states
            )
            if remaining > 0:
                diagnostic_states.append(states[:remaining])
            count += states.size(0)
    if count != len(indices):
        raise RuntimeError(f"Static state counted {count} samples, expected {len(indices)}.")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0)
    checkpoint_weights = _checkpoint_weights_path(args.anatomy_checkpoint)
    vision_config = config["model"]["model_config"]["vision_config"]
    if not isinstance(vision_config, dict):
        vision_config = vars(vision_config)
    preprocessing = {
        "vision_config": vision_config,
        "patch_grid_size": dataset_config["patch_grid_size"],
        "image_processor": type(models["image_processor"]).__name__,
    }
    metadata = {
        "sample_count": count,
        "indices_sha256": sha256_file(args.indices),
        "checkpoint_sha256": sha256_file(checkpoint_weights),
        "checkpoint_path": str(Path(args.anatomy_checkpoint).resolve()),
        "preprocessing_sha256": sha256_json(preprocessing),
        "preprocessing": preprocessing,
        "seed": args.seed,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }
    metadata_path = save_static_anatomy_state(args.output, mean.float(), metadata)
    diagnostic_tensor = torch.cat(diagnostic_states)
    diagnostics = anatomy_state_diagnostics(
        diagnostic_tensor,
        mean.float(),
        _left_right_pairs(dataset.contract["anatomy"]["labels"]),
    )
    diagnostics.update(
        population_sample_count=count,
        diagnostic_sample_count=int(diagnostic_tensor.size(0)),
        per_anatomy_feature_mean=mean.float().tolist(),
        per_anatomy_feature_std=variance.sqrt().float().tolist(),
        preprocessing=preprocessing,
        seed=args.seed,
    )
    diagnostics_path = Path(args.output).with_suffix(".diagnostics.json")
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cache": args.output, "metadata": str(metadata_path), "diagnostics": str(diagnostics_path)}))


def prepare_configs(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.static_state).with_suffix(
        Path(args.static_state).suffix + ".json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not args.allow_subset_indices and sha256_file(args.train_indices) != metadata["indices_sha256"]:
        raise ValueError("Training indices do not match the static-state provenance.")
    if sha256_file(_checkpoint_weights_path(args.anatomy_checkpoint)) != metadata["checkpoint_sha256"]:
        raise ValueError("Anatomy checkpoint does not match the static-state provenance.")
    expected = {
        key: metadata[key]
        for key in (
            "sample_count",
            "indices_sha256",
            "checkpoint_sha256",
            "preprocessing_sha256",
            "state_source",
            "shape",
        )
    }
    steps = args.steps
    for short_name, mode in (
        ("QS", "static_pair_query"),
        ("QD", "dynamic_pair_query"),
    ):
        destination = output_dir / f"{short_name.lower()}_{steps}steps.yaml"
        content = {
            "model": {
                "model_config": {
                    "hierarchy_config": {
                        "finding_connection_mode": mode,
                        "pretrained_anatomy_checkpoint": str(Path(args.anatomy_checkpoint).resolve()),
                        "static_anatomy_state_path": str(Path(args.static_state).resolve()),
                        "static_anatomy_state_expected": expected,
                        "training": {
                            "teacher_forcing_start_ratio": 0.0,
                            "teacher_forcing_end_ratio": 0.0,
                        },
                        "training_schedule": [
                            {"stage": stage, "num_train_epochs": 100.0, "max_steps": steps}
                            for stage in ("finding", "global", "joint")
                        ],
                    }
                }
            },
            "dataset": {
                "hierarchy": {
                    "train_indices_path": str(Path(args.train_indices).resolve()),
                    "eval_indices_path": str(Path(args.valid_indices).resolve()),
                }
            },
            "train": {
                "seed": args.seed,
                "per_device_train_batch_size": args.per_device_batch_size,
                "per_device_eval_batch_size": args.per_device_batch_size,
                "gradient_accumulation_steps": 1,
                "evaluation_strategy": "steps",
                "eval_steps": args.eval_steps,
                "save_strategy": "steps",
                "save_steps": args.eval_steps,
                "save_total_limit": None,
                "load_best_model_at_end": False,
                "logging_steps": args.logging_steps,
                "report_to": "none",
            },
            "experiment": {
                "output_root_dir": str((output_dir / short_name).resolve()),
                "project": "pair_query_pilot",
                "user": short_name,
                "name": "run",
            },
        }
        import yaml

        destination.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
        print(destination)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _prediction_inputs(batch, device):
    excluded = {
        "sample_metadata",
        "anatomy_targets",
        "anatomy_boxes",
        "anatomy_annotation_mask",
        "gt_anatomy_masks",
        "finding_targets",
        "finding_valid_mask",
        "global_targets",
        "global_valid_mask",
    }
    return {
        key: _to_device(value, device)
        for key, value in batch.items()
        if key not in excluded
    }


def _fixed_cross_patient_donors(patient_ids, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(patient_ids), generator=generator).tolist()
    for shift in range(1, len(order)):
        donors = [order[(position + shift) % len(order)] for position in range(len(order))]
        if all(
            donor != receiver and patient_ids[donor] != patient_ids[receiver]
            for receiver, donor in enumerate(donors)
        ):
            return donors
    return None


def _load_full_checkpoint(model, checkpoint):
    from safetensors.torch import load_file

    path = Path(checkpoint)
    path = path / "model.safetensors" if path.is_dir() else path
    state = load_file(str(path))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _classification_summary(global_logits, finding_logits, targets, valid):
    global_metrics = multilabel_metrics(
        torch.sigmoid(global_logits),
        targets["global"],
        valid["global"],
        include_per_class=True,
    )
    finding_count = finding_logits.size(-1)
    finding_metrics = multilabel_metrics(
        torch.sigmoid(finding_logits).reshape(-1, finding_count),
        targets["finding"].reshape(-1, finding_count),
        valid["finding"].reshape(-1, finding_count),
        include_per_class=True,
    )
    global_metrics["valid_count"] = int(valid["global"].sum())
    global_metrics["valid_count_per_class"] = valid["global"].sum(0).tolist()
    finding_valid = valid["finding"].reshape(-1, finding_count)
    finding_metrics["valid_count"] = int(finding_valid.sum())
    finding_metrics["valid_count_per_class"] = finding_valid.sum(0).tolist()
    return {"global": global_metrics, "finding": finding_metrics}


def _save_attention_visualizations(dataset, dataset_indices, output, batch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attention = output.finding.attention_weights.detach().float().cpu()
    pair_mask = output.finding.pair_mask.detach().cpu()
    anatomy_labels = dataset.contract["anatomy"]["labels"]
    finding_labels = dataset.contract["finding"]["labels"]
    records = []
    for local_index, dataset_index in enumerate(dataset_indices[:4]):
        valid_positive = (
            pair_mask[local_index]
            & batch["finding_valid_mask"][local_index]
            & (batch["finding_targets"][local_index] == 1)
            & batch["anatomy_annotation_mask"][local_index].unsqueeze(-1)
        )
        locations = valid_positive.nonzero()
        if not locations.numel():
            continue
        anatomy_index, finding_index = locations[0].tolist()
        item = dataset[dataset_index]
        image = item["image"].convert("RGB")
        width, height = image.size
        values = attention[local_index, anatomy_index, finding_index]
        side = int(values.numel() ** 0.5)
        values = values.reshape(side, side)
        values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)
        alpha = Image.fromarray((values.numpy() * 180).astype(np.uint8)).resize(
            image.size, resample=Image.Resampling.BILINEAR
        )
        red = Image.new("RGBA", image.size, (255, 0, 0, 0))
        red.putalpha(alpha)
        overlay = Image.alpha_composite(image.convert("RGBA"), red)
        target = item["anatomy_targets"][anatomy_index].reshape(side, side)
        points = target.nonzero()
        if points.numel():
            y1, x1 = points.min(dim=0).values.tolist()
            y2, x2 = (points.max(dim=0).values + 1).tolist()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(
                (
                    x1 / side * width,
                    y1 / side * height,
                    x2 / side * width,
                    y2 / side * height,
                ),
                outline=(0, 255, 0, 255),
                width=max(2, width // 300),
            )
        path = output_dir / f"{local_index:02d}_{item['dicom_id']}.png"
        overlay.convert("RGB").save(path)
        records.append(
            {
                "dicom_id": item["dicom_id"],
                "anatomy_index": anatomy_index,
                "anatomy": anatomy_labels[anatomy_index],
                "finding_index": finding_index,
                "finding": finding_labels[finding_index],
                "path": str(path),
                "attention_definition": "mean across 8 cross-attention heads; not an exact single-path evidence decomposition",
                "green_box": "GT anatomy extent",
                "red_heatmap": "pair-query attention",
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return records


def evaluate_checkpoint(args):
    config = _load_config(args.config, args.config_overrides)
    dataset_config = config["dataset"]["hierarchy"]
    dataset = HierarchyJsonlDataset(
        dataset_config["manifest_path"],
        dataset_config["image_root"],
        dataset_config["contract_path"],
        split=dataset_config.get("eval_split", "valid"),
        patch_grid_size=dataset_config["patch_grid_size"],
    )
    indices = _load_indices(args.indices)
    if args.max_samples:
        indices = indices[: args.max_samples]
    models = load_model(config["model"], output_dir=args.output_dir)
    model = models["model"].to(args.device).eval()
    _load_full_checkpoint(model, args.checkpoint)
    subset = Subset(dataset, indices)
    collate = partial(
        hierarchy_collate_fn,
        tokenizer=models["tokenizer"],
        image_processor=models["image_processor"],
        contract_path=dataset_config["contract_path"],
    )
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    normal_global, normal_finding, states = [], [], []
    global_targets, global_valid, finding_targets, finding_valid = [], [], [], []
    metadata = {key: [] for key in ("dicom_id", "study_id", "patient_id", "split_index")}
    local = {"count": 0, "containment": 0.0, "uniform": 0.0, "pointing": 0.0}
    first_batch_audit = None
    visualizations = None
    with torch.no_grad():
        for batch in loader:
            inputs = _prediction_inputs(batch, args.device)
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                output = model(**inputs, return_loss=False)
            normal_global.append(output.fusion.logits.float().cpu())
            normal_finding.append(output.finding.finding_logits.float().cpu())
            states.append(output.anatomy.evidence_tokens.float().cpu())
            global_targets.append(batch["global_targets"])
            global_valid.append(batch["global_valid_mask"])
            finding_targets.append(batch["finding_targets"])
            valid_pairs = batch["finding_valid_mask"] & output.finding.pair_mask.cpu()
            finding_valid.append(valid_pairs)
            selected = (
                valid_pairs
                & (batch["finding_targets"] == 1)
                & batch["anatomy_annotation_mask"].unsqueeze(-1)
            )
            masks = batch["anatomy_targets"].to(output.finding.attention_weights.dtype)
            attention = output.finding.attention_weights.cpu()
            containment = (attention * masks.unsqueeze(2)).sum(-1)
            uniform = masks.mean(-1).unsqueeze(-1).expand_as(containment)
            peak = attention.argmax(-1)
            inside = masks.unsqueeze(2).expand_as(attention).gather(
                -1, peak.unsqueeze(-1)
            ).squeeze(-1)
            local["count"] += int(selected.sum())
            local["containment"] += float(containment[selected].sum())
            local["uniform"] += float(uniform[selected].sum())
            local["pointing"] += float(inside[selected].sum())
            for key in metadata:
                metadata[key].extend(batch["sample_metadata"][key])
            if visualizations is None:
                visualizations = _save_attention_visualizations(
                    dataset,
                    indices[: batch["pixel_values"].size(0)],
                    output,
                    batch,
                    Path(args.output_dir) / "attention_visualizations",
                )
            if first_batch_audit is None:
                full_inputs = {
                    key: _to_device(value, args.device)
                    for key, value in batch.items()
                    if key != "sample_metadata"
                }
                with torch.autocast(
                    device_type=torch.device(args.device).type,
                    dtype=torch.bfloat16,
                    enabled=torch.device(args.device).type == "cuda",
                ):
                    full_output = model(**full_inputs, return_loss=False)
                delta = (full_output.fusion.logits - output.fusion.logits).abs().max()
                first_batch_audit = {
                    "passed": bool(delta <= args.invariance_atol),
                    "max_abs_logit_delta": float(delta.float().item()),
                    "atol": args.invariance_atol,
                }
    normal_global = torch.cat(normal_global)
    normal_finding = torch.cat(normal_finding)
    states = torch.cat(states)
    targets = {"global": torch.cat(global_targets), "finding": torch.cat(finding_targets)}
    valid = {"global": torch.cat(global_valid), "finding": torch.cat(finding_valid)}
    normal = _classification_summary(normal_global, normal_finding, targets, valid)
    count = max(local["count"], 1)
    normal["anatomy_consistency"] = {
        "valid_positive_pair_count": local["count"],
        "containment": local["containment"] / count if local["count"] else None,
        "uniform_containment": local["uniform"] / count if local["count"] else None,
        "pointing_accuracy": local["pointing"] / count if local["count"] else None,
        "interpretation": "anatomy consistency, not lesion localization",
    }
    interventions = {}
    if model.finding_connection_mode == "dynamic_pair_query":
        donors = _fixed_cross_patient_donors(metadata["patient_id"], args.seed)
        variants = {
            "train_mean": model.pair_query_finding_head.static_anatomy_state.unsqueeze(0).expand_as(states),
        }
        if donors is not None:
            variants["cross_patient_same_anatomy"] = states[donors]
        labels = dataset.contract["anatomy"]["labels"]
        pairs = _left_right_pairs(labels)
        if pairs:
            swapped = states.clone()
            for left, right in pairs:
                swapped[:, left], swapped[:, right] = states[:, right], states[:, left]
            variants["qualified_left_right_swap"] = swapped
        for name, override in variants.items():
            variant_global, variant_finding = [], []
            attention_delta = evidence_delta = 0.0
            element_count = 0
            cursor = 0
            with torch.no_grad():
                for batch in loader:
                    size = batch["pixel_values"].size(0)
                    inputs = _prediction_inputs(batch, args.device)
                    with torch.autocast(
                        device_type=torch.device(args.device).type,
                        dtype=torch.bfloat16,
                        enabled=torch.device(args.device).type == "cuda",
                    ):
                        baseline = model(**inputs, return_loss=False)
                        changed = model(
                            **inputs,
                            anatomy_state_override=override[cursor : cursor + size].to(args.device),
                            return_loss=False,
                        )
                    variant_global.append(changed.fusion.logits.float().cpu())
                    variant_finding.append(changed.finding.finding_logits.float().cpu())
                    attention_delta += float(
                        (changed.finding.attention_weights - baseline.finding.attention_weights)
                        .abs().float().sum().item()
                    )
                    evidence_delta += float(
                        (changed.finding.evidence_tokens - baseline.finding.evidence_tokens)
                        .abs().float().sum().item()
                    )
                    element_count += changed.finding.attention_weights.numel()
                    cursor += size
            variant_global = torch.cat(variant_global)
            variant_finding = torch.cat(variant_finding)
            interventions[name] = _classification_summary(
                variant_global, variant_finding, targets, valid
            )
            interventions[name].update(
                mean_absolute_global_logit_delta=float(
                    (variant_global - normal_global).abs().mean().item()
                ),
                mean_absolute_finding_logit_delta=float(
                    (variant_finding - normal_finding).abs()[valid["finding"]].mean().item()
                ),
                mean_absolute_attention_delta=attention_delta / element_count,
                mean_absolute_evidence_delta=evidence_delta
                / (len(indices) * model.num_anatomies * model.num_findings * model.hidden_size),
            )
            if name == "qualified_left_right_swap":
                affected = torch.zeros_like(valid["finding"])
                for left, right in pairs:
                    affected[:, left] = True
                    affected[:, right] = True
                difference = (variant_finding - normal_finding).abs()
                related = valid["finding"] & affected
                unrelated = valid["finding"] & ~affected
                interventions[name]["related_pair_logit_delta"] = (
                    float(difference[related].mean().item()) if related.any() else None
                )
                interventions[name]["unrelated_pair_logit_delta"] = (
                    float(difference[unrelated].mean().item()) if unrelated.any() else None
                )
        if donors is None:
            interventions["cross_patient_same_anatomy"] = {
                "available": False,
                "reason": "No deterministic cross-patient derangement exists.",
            }
        interventions.setdefault(
            "qualified_left_right_swap",
            {"available": False, "reason": "No verified contract label pairs."},
        )
        if donors is not None:
            donor_subset = Subset(dataset, [indices[index] for index in donors])
            donor_loader = DataLoader(
                donor_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate,
            )
            shuffled_global, shuffled_finding = [], []
            attention_delta = evidence_delta = 0.0
            attention_count = evidence_count = 0
            with torch.no_grad():
                for receiver_batch, donor_batch in zip(loader, donor_loader):
                    receiver_inputs = _prediction_inputs(receiver_batch, args.device)
                    donor_inputs = _prediction_inputs(donor_batch, args.device)
                    with torch.autocast(
                        device_type=torch.device(args.device).type,
                        dtype=torch.bfloat16,
                        enabled=torch.device(args.device).type == "cuda",
                    ):
                        receiver = model(**receiver_inputs, return_loss=False)
                        donor = model(**donor_inputs, return_loss=False)
                        global_queries = model._encode_prompt_bank(
                            receiver_inputs["encoded_global_prompts"],
                            model.num_global_labels,
                            "global",
                        )
                        global_queries, _ = model._prepare_radzero_global_inputs(
                            global_queries, receiver.vision_outputs["vision_tokens"]
                        )
                        aggregation = model.finding_aggregator(
                            global_queries,
                            donor.finding.evidence_tokens,
                            finding_scores=donor.finding.finding_logits,
                            pair_mask=donor.finding.pair_mask,
                            global_query_mask=receiver_inputs["global_query_mask"],
                        )
                        fusion = model.evidence_fusion(
                            aggregation.evidence_tokens,
                            receiver.global_evidence.evidence_tokens,
                            global_queries,
                            global_query_mask=receiver_inputs["global_query_mask"],
                        )
                    shuffled_global.append(fusion.logits.float().cpu())
                    shuffled_finding.append(donor.finding.finding_logits.float().cpu())
                    attention_delta += float(
                        (donor.finding.attention_weights - receiver.finding.attention_weights)
                        .abs().float().sum().item()
                    )
                    evidence_delta += float(
                        (donor.finding.evidence_tokens - receiver.finding.evidence_tokens)
                        .abs().float().sum().item()
                    )
                    attention_count += receiver.finding.attention_weights.numel()
                    evidence_count += receiver.finding.evidence_tokens.numel()
            shuffled_global = torch.cat(shuffled_global)
            shuffled_finding = torch.cat(shuffled_finding)
            shuffled = _classification_summary(
                shuffled_global, shuffled_finding, targets, valid
            )
            shuffled.update(
                mean_absolute_global_logit_delta=float(
                    (shuffled_global - normal_global).abs().mean().item()
                ),
                mean_absolute_finding_logit_delta=float(
                    (shuffled_finding - normal_finding).abs()[valid["finding"]].mean().item()
                ),
                mean_absolute_attention_delta=attention_delta / attention_count,
                mean_absolute_evidence_delta=evidence_delta / evidence_count,
            )
            interventions["shuffled_full_finding_state"] = shuffled
        donor_path = Path(args.output_dir) / "cross_patient_donors.json"
        donor_path.parent.mkdir(parents=True, exist_ok=True)
        donor_path.write_text(
            json.dumps({"seed": args.seed, "donor_indices": donors}, indent=2) + "\n",
            encoding="utf-8",
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as destination:
        for index in range(len(indices)):
            destination.write(
                json.dumps(
                    {
                        **{key: metadata[key][index] for key in metadata},
                        "global_targets": targets["global"][index].tolist(),
                        "global_valid": valid["global"][index].tolist(),
                        "global_logits": normal_global[index].tolist(),
                    }
                )
                + "\n"
            )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "connection_mode": model.finding_connection_mode,
        "sample_count": len(indices),
        "normal": normal,
        "interventions": interventions,
        "no_gt_audit": first_batch_audit,
        "patient_metadata_available": all(metadata["patient_id"]),
        "predictions": str(prediction_path),
        "attention_visualizations": visualizations,
    }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)


def paired_compare(args):
    def load_predictions(path):
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        return {row["dicom_id"]: row for row in rows}

    first = load_predictions(args.first)
    second = load_predictions(args.second)
    if set(first) != set(second):
        raise ValueError("Paired prediction files contain different image IDs.")
    ordered_ids = sorted(first)
    for image_id in ordered_ids:
        if (
            first[image_id]["global_targets"] != second[image_id]["global_targets"]
            or first[image_id]["global_valid"] != second[image_id]["global_valid"]
        ):
            raise ValueError(f"Targets differ for paired image {image_id}.")
    first_scores = torch.sigmoid(
        torch.tensor([first[key]["global_logits"] for key in ordered_ids])
    )
    second_scores = torch.sigmoid(
        torch.tensor([second[key]["global_logits"] for key in ordered_ids])
    )
    targets = torch.tensor([first[key]["global_targets"] for key in ordered_ids])
    valid = torch.tensor(
        [first[key]["global_valid"] for key in ordered_ids], dtype=torch.bool
    )
    clusters = [first[key]["patient_id"] for key in ordered_ids]
    report = {
        "first": str(Path(args.first).resolve()),
        "second": str(Path(args.second).resolve()),
        "sample_count": len(ordered_ids),
        "patient_count": len(set(clusters)),
        "macro_auprc_delta": paired_bootstrap_metric_delta(
            first_scores,
            second_scores,
            targets,
            valid,
            clusters=clusters,
            samples=args.samples,
            seed=args.seed,
            metric="auprc",
        ),
        "macro_auroc_delta": paired_bootstrap_metric_delta(
            first_scores,
            second_scores,
            targets,
            valid,
            clusters=clusters,
            samples=args.samples,
            seed=args.seed,
            metric="auroc",
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


def frozen_checkpoint_audit(args):
    from safetensors.torch import load_file
    from exp.cxr_pt.model.checkpoint import migrate_released_radzero_state_dict

    checkpoint = _checkpoint_weights_path(args.checkpoint)
    trained = load_file(str(checkpoint))
    radzero = migrate_released_radzero_state_dict(
        load_file(str(_checkpoint_weights_path(args.radzero_checkpoint)))
    )
    anatomy = load_file(str(_checkpoint_weights_path(args.anatomy_checkpoint)))

    def compare(reference, prefixes):
        keys = sorted(
            key for key in reference if any(key.startswith(prefix) for prefix in prefixes)
        )
        missing = [key for key in keys if key not in trained]
        deltas = [
            float((trained[key].float() - reference[key].float()).abs().max().item())
            for key in keys
            if key in trained
        ]
        return {
            "tensor_count": len(keys),
            "missing": missing,
            "max_absolute_delta": max(deltas, default=0.0),
            "unchanged": not missing and max(deltas, default=0.0) == 0.0,
        }

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "radzero": compare(
            radzero, ("vision_model.", "text_model.", "align_transformer.")
        ),
        "anatomy_head": compare(anatomy, ("anatomy_head.",)),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


def select_best_checkpoint(args):
    stage_dir = Path(args.stage_dir)
    state_files = sorted(stage_dir.glob("checkpoint-*/trainer_state.json"))
    candidates = []
    for state_file in state_files:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        step = int(state_file.parent.name.split("-")[-1])
        entry = next(
            (
                row
                for row in state.get("log_history", [])
                if int(row.get("step", -1)) == step and args.metric in row
            ),
            None,
        )
        if entry is not None:
            candidates.append(
                {
                    "checkpoint": str(state_file.parent.resolve()),
                    "step": step,
                    "metric": args.metric,
                    "value": float(entry[args.metric]),
                }
            )
    if not candidates:
        raise ValueError(f"No checkpoints contain {args.metric} in {stage_dir}.")
    best = max(candidates, key=lambda item: item["value"])
    report = {"best": best, "candidates": candidates, "rule": "maximum"}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description="Pair-query hierarchy experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-static")
    build.add_argument("--config", default="exp/cxr_pt/config.yaml")
    build.add_argument("--config-overrides", nargs="+", default=["radzero.yaml", "hierarchy.yaml", "paths.yaml"])
    build.add_argument("--indices", required=True)
    build.add_argument("--anatomy-checkpoint", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--expected-count", type=int, default=5000)
    build.add_argument("--diagnostic-samples", type=int, default=512)
    build.add_argument("--batch-size", type=int, default=8)
    build.add_argument("--num-workers", type=int, default=4)
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--seed", type=int, default=42)
    prepare = subparsers.add_parser("prepare-configs")
    prepare.add_argument("--static-state", required=True)
    prepare.add_argument("--anatomy-checkpoint", required=True)
    prepare.add_argument("--train-indices", required=True)
    prepare.add_argument("--valid-indices", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--steps", type=int, default=1000)
    prepare.add_argument("--eval-steps", type=int, default=100)
    prepare.add_argument("--per-device-batch-size", type=int, default=8)
    prepare.add_argument("--logging-steps", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--allow-subset-indices", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="exp/cxr_pt/config.yaml")
    evaluate.add_argument("--config-overrides", nargs="+", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--indices", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--num-workers", type=int, default=4)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--invariance-atol", type=float, default=1e-3)
    compare = subparsers.add_parser("paired-compare")
    compare.add_argument("--first", required=True)
    compare.add_argument("--second", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--samples", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=42)
    audit = subparsers.add_parser("frozen-audit")
    audit.add_argument("--checkpoint", required=True)
    audit.add_argument("--radzero-checkpoint", required=True)
    audit.add_argument("--anatomy-checkpoint", required=True)
    audit.add_argument("--output", required=True)
    best = subparsers.add_parser("select-best")
    best.add_argument("--stage-dir", required=True)
    best.add_argument("--metric", default="eval_global_auprc")
    best.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build-static":
        build_static_state(args)
    elif args.command == "prepare-configs":
        prepare_configs(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args)
    elif args.command == "paired-compare":
        paired_compare(args)
    elif args.command == "frozen-audit":
        frozen_checkpoint_audit(args)
    else:
        select_best_checkpoint(args)


if __name__ == "__main__":
    main()
