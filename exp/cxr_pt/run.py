import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    enable_full_determinism,
    set_seed,
)

from common.code_snapshot import code_snapshot
from common.trainer import logger
from common.utils import Config, output_directory_setting, str2bool
from exp.cxr_pt.dataset import collate_fn, load_datasets
from exp.cxr_pt.hierarchy_dataset import HierarchyJsonlDataset, hierarchy_collate_fn
from exp.cxr_pt.inference.inference import Inference
from exp.cxr_pt.model import load_model
from exp.cxr_pt.trainer import CXRPreTrainer, validate_hierarchy_schedule


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_parameter_fingerprint(module):
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_hierarchy_initialization_audit(model, cfg):
    hierarchy = cfg["model"]["model_config"]["hierarchy_config"]
    schedule = validate_hierarchy_schedule(
        hierarchy.get("training_schedule"), allow_subset=True
    )
    module_names = (
        "anatomy_head",
        "finding_head",
        "pair_query_finding_head",
        "finding_aggregator",
        "global_branch",
        "evidence_fusion",
    )
    rows = {}
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        rows[name] = {
            "total_parameters": sum(p.numel() for p in module.parameters()),
            "initialization_sha256": _module_parameter_fingerprint(module),
            "phase_optimizer_parameters": {},
        }
    encoder_names = tuple(model.training_controller.ENCODER_MODULES)
    rows["frozen_radzero"] = {
        "total_parameters": sum(
            p.numel() for name in encoder_names for p in getattr(model, name).parameters()
        ),
        "initialization_sha256": hashlib.sha256(
            "".join(
                _module_parameter_fingerprint(getattr(model, name))
                for name in encoder_names
            ).encode("ascii")
        ).hexdigest(),
        "phase_optimizer_parameters": {},
    }
    phase_totals = {}
    for phase in schedule:
        stage = phase["stage"]
        configured = model.configure_hierarchy_training(stage)
        phase_totals[stage] = {
            "configured_modules": list(configured),
            "optimizer_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
        for name, row in rows.items():
            modules = (
                [getattr(model, encoder_name) for encoder_name in encoder_names]
                if name == "frozen_radzero"
                else [getattr(model, name)]
            )
            row["phase_optimizer_parameters"][stage] = sum(
                p.numel()
                for module in modules
                for p in module.parameters()
                if p.requires_grad
            )
    model.configure_hierarchy_training(schedule[0]["stage"])
    dataset_cfg = cfg["dataset"]["hierarchy"]
    file_provenance = {}
    for name, path in {
        "train_indices": dataset_cfg.get("train_indices_path"),
        "eval_indices": dataset_cfg.get("eval_indices_path"),
        "static_anatomy_state": hierarchy.get("static_anatomy_state_path"),
    }.items():
        if path:
            resolved = str(Path(path).resolve())
            file_provenance[name] = {
                "path": resolved,
                "sha256": _sha256_file(resolved),
            }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"])
        diff_sha256 = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        commit, diff_sha256 = None, None
    report = {
        "seed_set_before_model_construction": int(cfg["train"]["seed"]),
        "finding_connection_mode": hierarchy.get(
            "finding_connection_mode", "roi_constraint"
        ),
        "modules": rows,
        "phases": phase_totals,
        "file_provenance": file_provenance,
        "git_commit": commit,
        "git_diff_sha256": diff_sha256,
    }
    output = Path(cfg["train"]["output_dir"]) / "initialization_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _build_trainer(models, training_args, dataset, data_collator, cfg):
    callbacks = []
    if training_args.load_best_model_at_end:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg["experiment"]["early_stopping_patience"]
            )
        )
    return CXRPreTrainer(
        model=models["model"],
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("eval"),
        data_collator=data_collator,
        cfg=cfg,
        callbacks=callbacks,
        tokenizer=models["tokenizer"],
    )


def _run_hierarchy_schedule(models, dataset, data_collator, cfg, deepspeed):
    hierarchy_config = cfg["model"]["model_config"]["hierarchy_config"]
    schedule = validate_hierarchy_schedule(
        hierarchy_config.get("training_schedule"),
        allow_subset=bool(hierarchy_config.get("global_only", False))
        or hierarchy_config.get("finding_connection_mode", "roi_constraint")
        != "roi_constraint",
    )
    resume_checkpoint = cfg["experiment"].get("resume_from_checkpoint")
    resume_stage = cfg["experiment"].get("resume_stage")
    if resume_checkpoint and resume_stage not in {phase["stage"] for phase in schedule}:
        raise ValueError(
            "experiment.resume_stage must name the hierarchy phase for the resume "
            "checkpoint."
        )
    start_index = 0
    if resume_checkpoint:
        start_index = next(
            index
            for index, phase in enumerate(schedule)
            if phase["stage"] == resume_stage
        )

    trainer = None
    root_output_dir = cfg["train"]["output_dir"]
    for phase_index, phase in enumerate(schedule[start_index:], start=start_index):
        stage = phase["stage"]
        models["model"].configure_hierarchy_training(stage)
        phase_train_config = copy.deepcopy(cfg["train"])
        phase_train_config["output_dir"] = os.path.join(root_output_dir, stage)
        phase_train_config["num_train_epochs"] = phase["num_train_epochs"]
        phase_train_config["max_steps"] = phase.get("max_steps", -1)
        training_args = TrainingArguments(
            **phase_train_config,
            deepspeed=copy.deepcopy(deepspeed),
        )
        trainer = _build_trainer(models, training_args, dataset, data_collator, cfg)
        phase_resume = resume_checkpoint if phase_index == start_index else None
        logger.info(
            "Run hierarchy phase %s for %s epochs; output: %s",
            stage,
            phase["num_train_epochs"],
            phase_train_config["output_dir"],
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        phase_started = time.perf_counter()
        train_result = trainer.train(resume_from_checkpoint=phase_resume)
        phase_seconds = time.perf_counter() - phase_started
        is_world_process_zero = getattr(trainer, "is_world_process_zero", None)
        if callable(is_world_process_zero) and is_world_process_zero():
            named = dict(models["model"].named_parameters())
            trainable = {
                name: parameter
                for name, parameter in named.items()
                if parameter.requires_grad
            }
            optimizer_ids = {
                id(parameter)
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            optimizer_id_list = [
                id(parameter)
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            ]
            trainable_ids = {id(parameter) for parameter in trainable.values()}
            parameters_finite = all(
                bool(torch.isfinite(parameter.detach()).all())
                for parameter in trainable.values()
            )
            optimizer_state_finite = all(
                bool(torch.isfinite(value.detach()).all())
                for state in trainer.optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
            )
            audit = {
                "stage": stage,
                "configured_modules": list(
                    models["model"].configure_hierarchy_training(stage)
                ),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable.values()
                ),
                "optimizer_parameter_count": sum(
                    parameter.numel()
                    for parameter in named.values()
                    if id(parameter) in optimizer_ids
                ),
                "optimizer_parameter_names": sorted(
                    name for name, parameter in named.items() if id(parameter) in optimizer_ids
                ),
                "optimizer_parameter_ids_unique": len(optimizer_id_list)
                == len(set(optimizer_id_list)),
                "optimizer_matches_requires_grad": optimizer_ids == trainable_ids,
                "trainable_parameters_finite": parameters_finite,
                "optimizer_state_finite": optimizer_state_finite,
                "global_step": int(trainer.state.global_step),
                "wall_clock_seconds": phase_seconds,
                "seconds_per_optimizer_step": phase_seconds
                / max(int(trainer.state.global_step), 1),
                "trainer_train_runtime_seconds": train_result.metrics.get("train_runtime"),
                "trainer_seconds_per_optimizer_step": (
                    train_result.metrics.get("train_runtime", phase_seconds)
                    / max(int(trainer.state.global_step), 1)
                ),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None,
                "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else None,
                "world_size": int(trainer.args.world_size),
                "per_device_train_batch_size": int(trainer.args.per_device_train_batch_size),
                "gradient_accumulation_steps": int(trainer.args.gradient_accumulation_steps),
                "effective_batch_size": int(
                    trainer.args.world_size
                    * trainer.args.per_device_train_batch_size
                    * trainer.args.gradient_accumulation_steps
                ),
            }
            audit_path = Path(phase_train_config["output_dir"]) / "phase_audit.json"
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        resume_checkpoint = None
    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--cfg-path",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="path to configuration file.",
    )
    parser.add_argument(
        "--add_cfg_list",
        default=[],
        type=str,
        nargs="+",
        help="List of YAML files. The cfg will be overwritten in the given order.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of experiment. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--train",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the train mode to True or False",
    )
    parser.add_argument(
        "--inference",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the inference mode to True or False",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory for saving inference results",
    )
    parser.add_argument(
        "--compute_metric",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Compute metrics with inference results. Set to True or False",
    )
    parser.add_argument("--no_report", action="store_true", help="Skip report to wandb")
    args = parser.parse_args()

    return args


@record
def main():
    args = parse_args()
    cfg = Config(args).config

    output_directory_setting(cfg, logger)
    snapshot_dir = code_snapshot(
        save_dir=os.path.join(cfg["train"]["output_dir"], "snapshot"), cfg=cfg
    )

    deepspeed = dict(cfg["deepspeed"]) if cfg.get("deepspeed") else None

    seed = int(cfg["train"]["seed"])
    if cfg["train"].get("full_determinism", False):
        enable_full_determinism(seed)
    else:
        set_seed(seed)
    models = load_model(cfg["model"], output_dir=cfg["train"]["output_dir"])

    is_hierarchy = cfg["model"].get("model_class_name") == "HierarchicalCxrAlignModel"
    if is_hierarchy:
        _write_hierarchy_initialization_audit(models["model"], cfg)
        hierarchy_cfg = cfg["dataset"]["hierarchy"]
        train_dataset = HierarchyJsonlDataset(
                hierarchy_cfg["manifest_path"],
                hierarchy_cfg["image_root"],
                hierarchy_cfg["contract_path"],
                split=hierarchy_cfg.get("train_split", "train"),
                patch_grid_size=hierarchy_cfg["patch_grid_size"],
        )
        eval_dataset = HierarchyJsonlDataset(
                hierarchy_cfg["manifest_path"],
                hierarchy_cfg["image_root"],
                hierarchy_cfg["contract_path"],
                split=hierarchy_cfg.get("eval_split", "valid"),
                patch_grid_size=hierarchy_cfg["patch_grid_size"],
        )
        # Pilot runs may provide fixed local indices.  Subset is applied after
        # split filtering, so a train index can never address validation/test.
        def _subset(dataset_obj, path_key):
            path = hierarchy_cfg.get(path_key)
            if not path:
                return dataset_obj
            path = Path(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            indices = payload.get("indices", payload)
            if len(indices) == 0 or any(int(i) < 0 or int(i) >= len(dataset_obj) for i in indices):
                raise ValueError(f"Invalid hierarchy pilot indices in {path}.")
            return Subset(dataset_obj, [int(i) for i in indices])

        dataset = {
            "train": _subset(train_dataset, "train_indices_path"),
            "eval": _subset(eval_dataset, "eval_indices_path"),
        }
        data_collator = partial(
            hierarchy_collate_fn,
            tokenizer=models["tokenizer"],
            image_processor=models["image_processor"],
            contract_path=hierarchy_cfg["contract_path"],
        )
    else:
        dataset = load_datasets(
            cfg["dataset"],
            train=cfg["args"]["train"],
            inference=False,
        )
        data_collator = partial(
            collate_fn,
            tokenizer=models["tokenizer"],
            image_processor=models["image_processor"],
        )

    trainer = None
    if cfg["args"]["train"]:
        logger.info("run train")
        if is_hierarchy:
            trainer = _run_hierarchy_schedule(
                models, dataset, data_collator, cfg, deepspeed
            )
        else:
            training_args = TrainingArguments(**cfg["train"], deepspeed=deepspeed)
            trainer = _build_trainer(models, training_args, dataset, data_collator, cfg)
            trainer.train(
                resume_from_checkpoint=cfg["experiment"]["resume_from_checkpoint"]
            )

    if trainer is None:
        training_args = TrainingArguments(**cfg["train"], deepspeed=deepspeed)
        trainer = _build_trainer(models, training_args, dataset, data_collator, cfg)

    if cfg["args"]["inference"]:

        # checkpoint path
        if trainer.state.best_model_checkpoint is not None:
            checkpoint = trainer.state.best_model_checkpoint
        else:
            checkpoint = cfg["experiment"]["resume_from_checkpoint"]

        if is_hierarchy:
            trainer.inference(
                test_dataset=dataset["eval"],
                output_dir=os.path.join(
                    cfg["train"]["output_dir"], "inference", "hierarchy"
                ),
                resume_from_checkpoint=checkpoint,
            )
            return

        resume_from_checkpoint = trainer.inference_load_from_checkpoint(
            resume_from_checkpoint=checkpoint
        )

        if trainer.is_world_process_zero():
            models["model"].to(trainer.args.device)
            models["model"].to(torch.float32)

            data_root_dir = cfg["dataset"]["data_root"]
            save_root_dir = os.path.join(
                cfg["train"]["output_dir"],
                "inference",
                os.path.basename(snapshot_dir),
                os.path.basename(resume_from_checkpoint),
            )

            inference = Inference(
                cls_dataset=cfg["inference"]["cls_dataset"],
                det_dataset=cfg["inference"]["det_dataset"],
                seg_dataset=cfg["inference"]["seg_dataset"],
                data_root_dir=data_root_dir,
                batch_size=cfg["inference"]["batch_size"],
                num_workers=cfg["inference"]["num_workers"],
            )

            inference.classification(
                **models, save_root_dir=os.path.join(save_root_dir, "classification")
            )
            inference.grounding(
                **models, save_root_dir=os.path.join(save_root_dir, "grounding")
            )
            inference.segmentation(
                **models,
                save_root_dir=os.path.join(save_root_dir, "segmentation"),
                compute_pixel_level_auroc=cfg["inference"]["compute_pixel_level_auroc"],
            )

        else:
            pass


if __name__ == "__main__":
    main()
