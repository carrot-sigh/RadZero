import unittest
from contextlib import nullcontext
import tempfile
from pathlib import Path
from types import MethodType
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from exp.cxr_pt.model.hierarchy_training import HierarchyTrainingController
from exp.cxr_pt.run import _run_hierarchy_schedule
from exp.cxr_pt.trainer import CXRPreTrainer, validate_hierarchy_schedule
from tests.test_hierarchical_model import HierarchyHarness, make_config


class _RepeatDataset(Dataset):
    def __init__(self, item, length=4):
        self.item = item
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.item


def _hierarchy_collator(items):
    output = {}
    for key in items[0]:
        if key.startswith("encoded_"):
            output[key] = items[0][key]
        else:
            output[key] = torch.stack([item[key] for item in items])
    return output


class HierarchyTrainerTest(unittest.TestCase):
    def setUp(self):
        self.model = HierarchyHarness(make_config(anatomies=3, findings=4, globals_=5))
        self.model.configure_hierarchy_training("anatomy")
        self.trainer = object.__new__(CXRPreTrainer)
        self.trainer.state = SimpleNamespace(global_step=17)
        batch_size, tokens, hidden = 2, 9, 8
        self.inputs = {
            "pixel_values": torch.randn(batch_size, tokens + 1, hidden),
            "encoded_anatomy_prompts": {"features": torch.randn(3, hidden * 2)},
            "encoded_finding_prompts": {"features": torch.randn(4, hidden)},
            "encoded_global_prompts": {"features": torch.randn(5, hidden)},
            "anatomy_targets": torch.randint(0, 2, (batch_size, 3, tokens)).float(),
            "anatomy_valid_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        }

    def test_hierarchy_schedule_requires_all_stages_in_order(self):
        schedule = [
            {"stage": stage, "num_train_epochs": 1}
            for stage in ("anatomy", "finding", "global", "joint")
        ]
        self.assertEqual(
            [phase["stage"] for phase in validate_hierarchy_schedule(schedule)],
            ["anatomy", "finding", "global", "joint"],
        )
        with self.assertRaisesRegex(ValueError, "must follow"):
            validate_hierarchy_schedule([schedule[1], schedule[0], *schedule[2:]])
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_hierarchy_schedule(
                [*schedule[:2], {"stage": "global", "num_train_epochs": 0}, schedule[3]]
            )

    def test_formal_schedule_runs_each_stage_with_fresh_trainer(self):
        configured_stages = []
        model = SimpleNamespace(configure_hierarchy_training=configured_stages.append)
        models = {"model": model, "tokenizer": object()}
        schedule = [
            {"stage": stage, "num_train_epochs": index + 1}
            for index, stage in enumerate(("anatomy", "finding", "global", "joint"))
        ]
        cfg = {
            "model": {
                "model_config": {"hierarchy_config": {"training_schedule": schedule}}
            },
            "train": {"output_dir": "/tmp/hierarchy", "num_train_epochs": 20},
            "experiment": {
                "resume_from_checkpoint": "/tmp/hierarchy/finding/checkpoint-2",
                "resume_stage": "finding",
                "early_stopping_patience": 1,
            },
        }
        trainers = []

        class FakeTrainer:
            def __init__(self):
                self.resumes = []

            def train(self, resume_from_checkpoint=None):
                self.resumes.append(resume_from_checkpoint)

        def build_trainer(*args, **kwargs):
            trainer = FakeTrainer()
            trainers.append(trainer)
            return trainer

        with patch(
            "exp.cxr_pt.run.TrainingArguments", side_effect=lambda **kwargs: kwargs
        ), patch("exp.cxr_pt.run._build_trainer", side_effect=build_trainer) as build:
            final_trainer = _run_hierarchy_schedule(
                models, {"train": object(), "eval": object()}, object(), cfg, None
            )

        self.assertEqual(configured_stages, ["finding", "global", "joint"])
        self.assertEqual(len(trainers), 3)
        self.assertIs(final_trainer, trainers[-1])
        self.assertEqual(
            trainers[0].resumes,
            ["/tmp/hierarchy/finding/checkpoint-2"],
        )
        self.assertEqual(trainers[1].resumes, [None])
        self.assertEqual(trainers[2].resumes, [None])
        phase_args = [call.args[1] for call in build.call_args_list]
        self.assertEqual(
            [args["output_dir"] for args in phase_args],
            [
                "/tmp/hierarchy/finding",
                "/tmp/hierarchy/global",
                "/tmp/hierarchy/joint",
            ],
        )

    def test_checkpoint_restores_stage_and_trainer_step(self):
        self.model.configure_hierarchy_training("finding")
        self.model.set_hierarchy_training_step(4321)
        state = self.model.state_dict()

        restored = HierarchyHarness(make_config(anatomies=3, findings=4, globals_=5))
        restored.load_state_dict(state)
        restored_trainer = object.__new__(CXRPreTrainer)
        restored_trainer.state = SimpleNamespace(global_step=4321)
        restored_inputs = dict(self.inputs)
        restored_inputs["gt_anatomy_masks"] = restored_inputs["anatomy_targets"]
        restored_inputs["finding_targets"] = torch.randint(0, 2, (2, 3, 4)).float()
        restored_inputs["finding_valid_mask"] = torch.ones(2, 3, 4, dtype=torch.bool)

        _, outputs = restored_trainer.compute_loss(
            restored, restored_inputs, return_outputs=True
        )
        self.assertEqual(restored.training_controller.stage, "finding")
        self.assertEqual(restored.training_controller.step.item(), 4321)
        self.assertIn("finding_loss", outputs["losses"])
        self.assertAlmostEqual(
            outputs["losses"]["teacher_forcing_ratio"].item(), 0.5679, places=4
        )

    def test_full_trainer_checkpoint_resume_restores_training_state(self):
        torch.manual_seed(53)
        model = HierarchyHarness(make_config(anatomies=3, findings=4, globals_=5))
        model.config.vision_config = {}
        model.config.text_config = {"use_text_projection": False}
        model.config.align_transformer_config = {"hidden_size": 8}
        model.configure_hierarchy_training("finding")
        sample = {
            "pixel_values": torch.randn(10, 8),
            "encoded_anatomy_prompts": {"features": torch.randn(3, 16)},
            "encoded_finding_prompts": {"features": torch.randn(4, 8)},
            "encoded_global_prompts": {"features": torch.randn(5, 8)},
            "anatomy_targets": torch.randint(0, 2, (3, 9)).float(),
            "anatomy_annotation_mask": torch.ones(3, dtype=torch.bool),
            "anatomy_query_mask": torch.ones(3, dtype=torch.bool),
            "gt_anatomy_masks": torch.randint(0, 2, (3, 9)).float(),
            "finding_targets": torch.randint(0, 2, (3, 4)).float(),
            "finding_valid_mask": torch.ones(3, 4, dtype=torch.bool),
            "global_targets": torch.randint(0, 2, (5,)).float(),
            "global_valid_mask": torch.ones(5, dtype=torch.bool),
        }
        dataset = _RepeatDataset(sample)
        with tempfile.TemporaryDirectory() as directory:
            common_args = dict(
                output_dir=directory,
                per_device_train_batch_size=2,
                learning_rate=1e-3,
                save_strategy="steps",
                save_steps=2,
                logging_strategy="no",
                evaluation_strategy="no",
                report_to=[],
                remove_unused_columns=False,
                disable_tqdm=True,
                use_cpu=True,
            )
            first_args = TrainingArguments(max_steps=2, **common_args)
            first = CXRPreTrainer(
                model=model,
                args=first_args,
                train_dataset=dataset,
                data_collator=_hierarchy_collator,
            )
            first.train()
            checkpoint = Path(directory) / "checkpoint-2"
            self.assertTrue((checkpoint / "optimizer.pt").is_file())
            self.assertTrue((checkpoint / "scheduler.pt").is_file())
            self.assertTrue((checkpoint / "trainer_state.json").is_file())
            saved_parameter = next(model.finding_head.parameters()).detach().clone()

            torch.manual_seed(59)
            restored_model = HierarchyHarness(
                make_config(anatomies=3, findings=4, globals_=5)
            )
            restored_model.config.vision_config = {}
            restored_model.config.text_config = {"use_text_projection": False}
            restored_model.config.align_transformer_config = {"hidden_size": 8}
            second_args = TrainingArguments(max_steps=4, **common_args)
            second = CXRPreTrainer(
                model=restored_model,
                args=second_args,
                train_dataset=dataset,
                data_collator=_hierarchy_collator,
            )
            second._load_from_checkpoint(str(checkpoint))
            torch.testing.assert_close(
                next(restored_model.finding_head.parameters()), saved_parameter
            )
            self.assertEqual(restored_model.training_controller.stage, "finding")

            second.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(second.state.global_step, 4)
            self.assertEqual(restored_model.training_controller.stage, "finding")
            optimizer_ids = {
                id(parameter)
                for group in second.optimizer.param_groups
                for parameter in group["params"]
            }
            expected_ids = {
                id(parameter)
                for parameter in restored_model.finding_head.parameters()
                if parameter.requires_grad
            }
            self.assertEqual(optimizer_ids, expected_ids)
            self.assertTrue(second.optimizer.state)
            self.assertEqual(second.lr_scheduler.last_epoch, 4)

    def test_tiny_anatomy_overfit_reduces_loss(self):
        torch.manual_seed(67)
        self.model.configure_hierarchy_training("anatomy")
        inputs = dict(self.inputs)
        inputs["anatomy_targets"] = torch.zeros_like(inputs["anatomy_targets"])
        optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=0.05,
        )
        initial_loss = None
        minimum_loss = None
        for step in range(120):
            self.trainer.state.global_step = step
            optimizer.zero_grad()
            loss = self.trainer.compute_loss(self.model, inputs)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            minimum_loss = (
                loss.item() if minimum_loss is None else min(minimum_loss, loss.item())
            )
        self.assertLess(minimum_loss, initial_loss * 0.8)

    def test_stage_selects_only_its_supervision_targets(self):
        for stage, expected in (
            (
                "anatomy",
                {"loss", "anatomy_loss", "anatomy_bce_loss", "anatomy_dice_loss"},
            ),
            ("finding", {"loss", "finding_loss"}),
        ):
            with self.subTest(stage=stage):
                self.model.configure_hierarchy_training(stage)
                if stage == "finding":
                    self.inputs["gt_anatomy_masks"] = self.inputs["anatomy_targets"]
                    self.inputs["finding_targets"] = torch.randint(
                        0, 2, (2, 3, 4)
                    ).float()
                    self.inputs["finding_valid_mask"] = torch.ones(
                        2, 3, 4, dtype=torch.bool
                    )
                _, outputs = self.trainer.compute_loss(
                    self.model, self.inputs, return_outputs=True
                )
                self.assertTrue(expected.issubset(outputs["losses"]))
                self.assertFalse(
                    {"anatomy_loss", "finding_loss", "global_loss"}.difference(expected)
                    & set(outputs["losses"])
                )

    def test_wrapped_model_preserves_stage_routing_and_step(self):
        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        wrapped = Wrapper(self.model)
        inputs = dict(self.inputs)
        inputs["finding_targets"] = torch.ones(2, 3, 4)
        inputs["finding_valid_mask"] = torch.ones(2, 3, 4, dtype=torch.bool)
        inputs["global_targets"] = torch.ones(2, 5)
        inputs["global_valid_mask"] = torch.ones(2, 5, dtype=torch.bool)

        _, outputs = self.trainer.compute_loss(wrapped, inputs, return_outputs=True)

        self.assertEqual(self.model.training_controller.step.item(), 17)
        self.assertIn("anatomy_loss", outputs["losses"])
        self.assertNotIn("finding_loss", outputs["losses"])
        self.assertNotIn("global_loss", outputs["losses"])

    def test_one_batch_backward_respects_anatomy_stage(self):
        loss = self.trainer.compute_loss(self.model, self.inputs)
        loss.backward()

        anatomy_grads = [
            parameter.grad
            for parameter in self.model.anatomy_head.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(anatomy_grads)
        self.assertTrue(all(gradient is not None for gradient in anatomy_grads))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in anatomy_grads)
        )
        self.assertFalse(
            any(
                parameter.grad is not None
                for parameter in self.model.finding_head.parameters()
            )
        )

    def test_optimizer_step_updates_only_anatomy_head_and_keeps_encoder_eval(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=0.1,
        )
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter)
            for parameter in self.model.anatomy_head.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimizer_ids, expected_ids)
        before_anatomy = {
            name: parameter.detach().clone()
            for name, parameter in self.model.anatomy_head.named_parameters()
        }
        before_frozen = {
            name: parameter.detach().clone()
            for module_name in ("vision_model", "text_model", "align_transformer")
            for name, parameter in getattr(self.model, module_name).named_parameters()
        }

        optimizer.zero_grad()
        self.trainer.compute_loss(self.model, self.inputs).backward()
        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(before_anatomy[name], parameter.detach())
                for name, parameter in self.model.anatomy_head.named_parameters()
            )
        )
        for module_name in ("vision_model", "text_model", "align_transformer"):
            module = getattr(self.model, module_name)
            self.assertFalse(module.training)
            for name, parameter in module.named_parameters():
                torch.testing.assert_close(
                    parameter.detach(), before_frozen[name], rtol=0, atol=0
                )

    def test_hierarchy_prediction_step_returns_metric_contract_without_gt_roi(self):
        self.trainer._prepare_inputs = lambda inputs: inputs
        self.trainer.compute_loss_context_manager = nullcontext
        inputs = dict(self.inputs)
        inputs.update(
            {
                "anatomy_annotation_mask": torch.ones(2, 3, dtype=torch.bool),
                "gt_anatomy_masks": self.inputs["anatomy_targets"],
                "finding_targets": torch.zeros(2, 3, 4),
                "finding_valid_mask": torch.ones(2, 3, 4, dtype=torch.bool),
                "global_targets": torch.zeros(2, 5),
                "global_valid_mask": torch.ones(2, 5, dtype=torch.bool),
            }
        )
        captured = {}
        original_forward = self.model.forward

        def capture_forward(model, *args, **kwargs):
            captured["has_gt_roi"] = "gt_anatomy_masks" in kwargs
            return original_forward(*args, **kwargs)

        self.model.forward = MethodType(capture_forward, self.model)
        loss, predictions, labels, losses = self.trainer.prediction_step(
            self.model, inputs, prediction_loss_only=False
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(captured["has_gt_roi"])
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(labels), 6)
        self.assertIn("anatomy_loss", losses)

    def test_standard_trainer_evaluate_reports_anatomy_metrics(self):
        sample = {
            "pixel_values": self.inputs["pixel_values"][0],
            "encoded_anatomy_prompts": self.inputs["encoded_anatomy_prompts"],
            "encoded_finding_prompts": self.inputs["encoded_finding_prompts"],
            "encoded_global_prompts": self.inputs["encoded_global_prompts"],
            "anatomy_targets": self.inputs["anatomy_targets"][0],
            "anatomy_annotation_mask": torch.tensor([True, False, True]),
            "anatomy_query_mask": torch.ones(3, dtype=torch.bool),
            "gt_anatomy_masks": self.inputs["anatomy_targets"][0],
            "finding_targets": torch.zeros(3, 4),
            "finding_valid_mask": torch.ones(3, 4, dtype=torch.bool),
            "global_targets": torch.zeros(5),
            "global_valid_mask": torch.ones(5, dtype=torch.bool),
        }
        with tempfile.TemporaryDirectory() as directory:
            trainer = CXRPreTrainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir=directory,
                    per_device_eval_batch_size=1,
                    report_to=[],
                    remove_unused_columns=False,
                    disable_tqdm=True,
                    use_cpu=True,
                ),
                eval_dataset=_RepeatDataset(sample, length=2),
                data_collator=_hierarchy_collator,
            )
            metrics = trainer.evaluate()
        self.assertIn("eval_anatomy_dice", metrics)
        self.assertIn("eval_anatomy_iou", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_loss"])))

    def test_stage_switch_rebuilds_optimizer_with_new_parameters(self):
        old_optimizer = torch.optim.SGD(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=0.01,
        )
        self.trainer.model = self.model
        self.trainer.optimizer = old_optimizer
        self.trainer.lr_scheduler = object()
        self.trainer.is_in_train = False
        self.trainer.callback_handler = SimpleNamespace(
            optimizer=old_optimizer, lr_scheduler=None
        )

        def create_optimizer_and_scheduler(trainer, num_training_steps):
            trainer.optimizer = torch.optim.SGD(
                [
                    parameter
                    for parameter in trainer.model.parameters()
                    if parameter.requires_grad
                ],
                lr=0.01,
            )
            trainer.lr_scheduler = SimpleNamespace(
                num_training_steps=num_training_steps
            )

        self.trainer.create_optimizer_and_scheduler = MethodType(
            create_optimizer_and_scheduler, self.trainer
        )
        self.trainer.configure_hierarchy_stage("finding", num_training_steps=20)
        optimizer_ids = {
            id(parameter)
            for group in self.trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter)
            for parameter in self.model.finding_head.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimizer_ids, expected_ids)
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.finding_head.named_parameters()
        }
        inputs = dict(self.inputs)
        inputs["gt_anatomy_masks"] = inputs["anatomy_targets"]
        inputs["finding_targets"] = torch.randint(0, 2, (2, 3, 4)).float()
        inputs["finding_valid_mask"] = torch.ones(2, 3, 4, dtype=torch.bool)
        self.trainer.optimizer.zero_grad()
        loss = self.trainer.compute_loss(self.model, inputs)
        loss.backward()
        self.trainer.optimizer.step()
        self.assertTrue(
            all(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in self.model.finding_head.named_parameters()
            )
        )

    def test_hierarchy_batch_routes_to_forward_and_reports_stage_metadata(self):
        loss, outputs = self.trainer.compute_loss(
            self.model, self.inputs, return_outputs=True
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(outputs["logits"].shape, (2, 5))
        self.assertEqual(self.model.training_controller.step.item(), 17)
        self.assertEqual(
            set(outputs["losses"]),
            {
                "loss",
                "anatomy_loss",
                "anatomy_bce_loss",
                "anatomy_dice_loss",
                "teacher_forcing_ratio",
                "hierarchy_stage_index",
            },
        )
        self.assertEqual(
            outputs["losses"]["hierarchy_stage_index"].item(),
            HierarchyTrainingController.STAGES.index("anatomy"),
        )
        self.assertAlmostEqual(
            outputs["losses"]["teacher_forcing_ratio"].item(), 0.9983, places=4
        )


if __name__ == "__main__":
    unittest.main()
