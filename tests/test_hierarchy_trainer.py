import unittest
from types import SimpleNamespace

import torch

from exp.cxr_pt.model.hierarchy_training import HierarchyTrainingController
from exp.cxr_pt.trainer import CXRPreTrainer
from tests.test_hierarchical_model import HierarchyHarness, make_config


class HierarchyTrainerTest(unittest.TestCase):
    def setUp(self):
        self.model = HierarchyHarness(make_config(anatomies=3, findings=4, globals_=5))
        self.model.configure_hierarchy_training("anatomy")
        self.trainer = object.__new__(CXRPreTrainer)
        self.trainer.state = SimpleNamespace(global_step=17)
        batch_size, tokens, hidden = 2, 7, 8
        self.inputs = {
            "pixel_values": torch.randn(batch_size, tokens + 1, hidden),
            "encoded_anatomy_prompts": {"features": torch.randn(3, hidden * 2)},
            "encoded_finding_prompts": {"features": torch.randn(4, hidden)},
            "encoded_global_prompts": {"features": torch.randn(5, hidden)},
            "anatomy_targets": torch.randint(0, 2, (batch_size, 3, tokens)).float(),
            "anatomy_valid_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        }

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
