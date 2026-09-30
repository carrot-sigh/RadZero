import unittest

import torch

from tests.test_hierarchical_model import HierarchyHarness, make_config


class HierarchyTrainingControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.model = HierarchyHarness(make_config())

    def test_each_stage_enables_only_its_owned_modules(self):
        expected = {
            "anatomy": {"anatomy_head"},
            "finding": {"finding_head"},
            "global": {
                "finding_aggregator",
                "global_branch",
                "evidence_fusion",
            },
            "joint": {
                "anatomy_head",
                "finding_head",
                "finding_aggregator",
                "global_branch",
                "evidence_fusion",
            },
        }
        hierarchy_modules = expected["joint"]

        for stage, trainable_names in expected.items():
            with self.subTest(stage=stage):
                configured = set(self.model.configure_hierarchy_training(stage))
                self.assertEqual(configured, trainable_names)
                for name in hierarchy_modules:
                    parameters = list(getattr(self.model, name).parameters())
                    if not parameters:
                        self.assertEqual(name, "global_branch")
                        continue
                    self.assertEqual(
                        any(parameter.requires_grad for parameter in parameters),
                        name in trainable_names,
                    )
                self.assertFalse(
                    any(
                        parameter.requires_grad
                        for parameter in self.model.loss_fns.parameters()
                    )
                )

    def test_frozen_encoders_stay_in_eval_mode_during_training(self):
        self.model.configure_hierarchy_training("anatomy")

        for name in self.model.training_controller.ENCODER_MODULES:
            self.assertFalse(getattr(self.model, name).training)

    def test_stage_and_step_survive_state_dict_round_trip(self):
        self.model.configure_hierarchy_training("global")
        self.model.set_hierarchy_training_step(7)
        state_dict = self.model.state_dict()

        restored = HierarchyHarness(make_config())
        restored.load_state_dict(state_dict)
        restored.configure_hierarchy_training()

        self.assertEqual(restored.training_controller.stage, "global")
        self.assertEqual(restored.training_controller.step.item(), 7)
        self.assertEqual(restored.get_teacher_forcing_ratio(), 0.9993)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in restored.global_branch.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in restored.anatomy_head.parameters()
            )
        )

    def test_teacher_forcing_schedule_is_linear_and_clamped(self):
        controller = self.model.training_controller

        self.assertEqual(controller.teacher_forcing_ratio(0), 1.0)
        self.assertEqual(controller.teacher_forcing_ratio(5000), 0.5)
        self.assertEqual(controller.teacher_forcing_ratio(10000), 0.0)
        self.assertEqual(controller.teacher_forcing_ratio(12000), 0.0)

    def test_model_selects_gt_or_predicted_masks_by_ratio(self):
        batch_size, anatomies, tokens, hidden_dim = 2, 3, 7, 8
        visual_tokens = torch.randn(batch_size, tokens, hidden_dim)
        anatomy_queries = torch.randn(anatomies, hidden_dim)
        finding_queries = torch.randn(4, hidden_dim)
        global_queries = torch.randn(5, hidden_dim)
        ground_truth = torch.zeros(batch_size, anatomies, tokens)
        self.model.train()

        gt_output = self.model.forward_hierarchy_from_features(
            visual_tokens,
            anatomy_queries,
            finding_queries,
            global_queries,
            global_visual_tokens=torch.randn(batch_size, tokens + 1, hidden_dim),
            gt_anatomy_masks=ground_truth,
            teacher_forcing_ratio=1.0,
        )
        predicted_output = self.model.forward_hierarchy_from_features(
            visual_tokens,
            anatomy_queries,
            finding_queries,
            global_queries,
            global_visual_tokens=torch.randn(batch_size, tokens + 1, hidden_dim),
            gt_anatomy_masks=ground_truth,
            teacher_forcing_ratio=0.0,
        )

        self.assertTrue(torch.all(gt_output.anatomy_masks_used == 0))
        torch.testing.assert_close(
            predicted_output.anatomy_masks_used,
            predicted_output.anatomy.roi_masks,
        )

    def test_explicit_override_and_teacher_forcing_are_mutually_exclusive(self):
        masks = torch.zeros(2, 3, 7)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.model.forward_hierarchy_from_features(
                torch.randn(2, 7, 8),
                torch.randn(3, 8),
                torch.randn(4, 8),
                torch.randn(5, 8),
                global_visual_tokens=torch.randn(2, 8, 8),
                anatomy_mask_override=masks,
                gt_anatomy_masks=masks,
            )

    def test_teacher_forcing_rejects_non_finite_gt_masks(self):
        masks = torch.zeros(2, 3, 7)
        masks[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "values in"):
            self.model.training_controller.select_anatomy_masks(
                torch.rand_like(masks), masks, training=True, ratio=1.0
            )


if __name__ == "__main__":
    unittest.main()
