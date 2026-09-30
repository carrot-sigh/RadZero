import argparse
import copy
import os
from functools import partial

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import EarlyStoppingCallback, TrainingArguments

from common.code_snapshot import code_snapshot
from common.trainer import logger
from common.utils import Config, output_directory_setting, str2bool
from exp.cxr_pt.dataset import collate_fn, load_datasets
from exp.cxr_pt.hierarchy_dataset import (
    HierarchyJsonlDataset,
    hierarchy_collate_fn,
)
from exp.cxr_pt.inference.inference import Inference
from exp.cxr_pt.model import load_model
from exp.cxr_pt.trainer import CXRPreTrainer, validate_hierarchy_schedule


def _build_trainer(models, training_args, dataset, data_collator, cfg):
    return CXRPreTrainer(
        model=models["model"],
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("eval"),
        data_collator=data_collator,
        cfg=cfg,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["experiment"]["early_stopping_patience"]
            )
        ],
        tokenizer=models["tokenizer"],
    )


def _run_hierarchy_schedule(models, dataset, data_collator, cfg, deepspeed):
    hierarchy_config = cfg["model"]["model_config"]["hierarchy_config"]
    schedule = validate_hierarchy_schedule(hierarchy_config.get("training_schedule"))
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
        phase_train_config["max_steps"] = -1
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
        trainer.train(resume_from_checkpoint=phase_resume)
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

    models = load_model(cfg["model"], output_dir=cfg["train"]["output_dir"])

    is_hierarchy = cfg["model"].get("model_class_name") == "HierarchicalCxrAlignModel"
    if is_hierarchy:
        hierarchy_cfg = cfg["dataset"]["hierarchy"]
        dataset = {
            "train": HierarchyJsonlDataset(
                hierarchy_cfg["manifest_path"],
                hierarchy_cfg["image_root"],
                hierarchy_cfg["contract_path"],
                split=hierarchy_cfg.get("train_split", "train"),
                patch_grid_size=hierarchy_cfg["patch_grid_size"],
            ),
            "eval": HierarchyJsonlDataset(
                hierarchy_cfg["manifest_path"],
                hierarchy_cfg["image_root"],
                hierarchy_cfg["contract_path"],
                split=hierarchy_cfg.get("eval_split", "valid"),
                patch_grid_size=hierarchy_cfg["patch_grid_size"],
            ),
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
