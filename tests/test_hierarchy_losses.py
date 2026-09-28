import unittest

import torch

from exp.cxr_pt.model.hierarchy_losses import HierarchyLoss


class HierarchyLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.anatomy_logits = torch.randn(2, 3, 5, requires_grad=True)
        self.finding_logits = torch.randn(2, 3, 4, requires_grad=True)
        self.global_logits = torch.randn(2, 6, requires_grad=True)
        self.anatomy_targets = torch.randint(0, 2, (2, 3, 5)).float()
        self.finding_targets = torch.randint(0, 2, (2, 3, 4)).float()
        self.global_targets = torch.randint(0, 2, (2, 6)).float()

    def test_unknown_targets_are_excluded_from_loss_and_gradients(self):
        anatomy_valid = torch.ones_like(self.anatomy_targets, dtype=torch.bool)
        finding_valid = torch.ones_like(self.finding_targets, dtype=torch.bool)
        global_valid = torch.ones_like(self.global_targets, dtype=torch.bool)
        anatomy_valid[0, 1, 2] = False
        finding_valid[1, 2, 3] = False
        global_valid[0, 4] = False

        self.anatomy_targets[0, 1, 2] = torch.nan
        self.finding_targets[1, 2, 3] = -1
        self.global_targets[0, 4] = torch.nan
        loss_fn = HierarchyLoss()
        output = loss_fn(
            self.anatomy_logits,
            self.finding_logits,
            self.global_logits,
            anatomy_targets=self.anatomy_targets,
            finding_targets=self.finding_targets,
            global_targets=self.global_targets,
            anatomy_valid_mask=anatomy_valid,
            finding_valid_mask=finding_valid,
            global_valid_mask=global_valid,
        )
        clean_output = loss_fn(
            self.anatomy_logits,
            self.finding_logits,
            self.global_logits,
            anatomy_targets=torch.nan_to_num(self.anatomy_targets, nan=0.0),
            finding_targets=self.finding_targets.clamp_min(0),
            global_targets=torch.nan_to_num(self.global_targets, nan=0.0),
            anatomy_valid_mask=anatomy_valid,
            finding_valid_mask=finding_valid,
            global_valid_mask=global_valid,
        )

        torch.testing.assert_close(output.loss, clean_output.loss)
        output.loss.backward()

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(self.anatomy_logits.grad[0, 1, 2].item(), 0.0)
        self.assertEqual(self.finding_logits.grad[1, 2, 3].item(), 0.0)
        self.assertEqual(self.global_logits.grad[0, 4].item(), 0.0)

    def test_structural_masks_exclude_queries_tokens_and_pairs(self):
        anatomy_query_mask = torch.tensor([True, False, True])
        visual_token_mask = torch.tensor([[True, True, True, False, False], [True] * 5])
        finding_pair_mask = torch.ones_like(self.finding_targets, dtype=torch.bool)
        finding_pair_mask[:, 0, 1] = False
        global_query_mask = torch.tensor([True, False, True, True, True, False])

        output = HierarchyLoss()(
            self.anatomy_logits,
            self.finding_logits,
            self.global_logits,
            anatomy_targets=self.anatomy_targets,
            finding_targets=self.finding_targets,
            global_targets=self.global_targets,
            anatomy_query_mask=anatomy_query_mask,
            visual_token_mask=visual_token_mask,
            finding_pair_mask=finding_pair_mask,
            global_query_mask=global_query_mask,
        )
        output.loss.backward()

        self.assertTrue(torch.all(self.anatomy_logits.grad[:, 1] == 0))
        self.assertTrue(torch.all(self.anatomy_logits.grad[0, :, 3:] == 0))
        self.assertTrue(torch.all(self.finding_logits.grad[:, 0, 1] == 0))
        self.assertTrue(torch.all(self.global_logits.grad[:, [1, 5]] == 0))

    def test_empty_valid_batch_returns_differentiable_zero(self):
        valid_mask = torch.zeros_like(self.global_targets, dtype=torch.bool)
        targets = torch.full_like(self.global_targets, torch.nan)
        output = HierarchyLoss()(
            self.anatomy_logits,
            self.finding_logits,
            self.global_logits,
            global_targets=targets,
            global_valid_mask=valid_mask,
        )

        self.assertEqual(output.loss.item(), 0.0)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(self.global_logits.grad)
        self.assertTrue(torch.all(self.global_logits.grad == 0))

    def test_region_level_anatomy_valid_mask_expands_over_tokens(self):
        valid_mask = torch.ones(2, 3, dtype=torch.bool)
        valid_mask[0, 1] = False
        targets = self.anatomy_targets.clone()
        targets[0, 1] = torch.nan
        output = HierarchyLoss()(
            self.anatomy_logits,
            self.finding_logits,
            self.global_logits,
            anatomy_targets=targets,
            anatomy_valid_mask=valid_mask,
        )
        output.loss.backward()

        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.all(self.anatomy_logits.grad[0, 1] == 0))

    def test_loss_weights_form_the_reported_total(self):
        loss_fn = HierarchyLoss(
            anatomy_weight=2.0,
            finding_weight=3.0,
            global_weight=4.0,
            anatomy_bce_weight=0.5,
            anatomy_dice_weight=1.5,
        )
        output = loss_fn(
            self.anatomy_logits,
            self.finding_logits,
            self.global_logits,
            anatomy_targets=self.anatomy_targets,
            finding_targets=self.finding_targets,
            global_targets=self.global_targets,
        )
        expected = (
            2.0 * output.anatomy_loss
            + 3.0 * output.finding_loss
            + 4.0 * output.global_loss
        )

        torch.testing.assert_close(output.loss, expected)
        torch.testing.assert_close(
            output.anatomy_loss,
            0.5 * output.anatomy_bce_loss + 1.5 * output.anatomy_dice_loss,
        )

    def test_invalid_labeled_target_is_rejected(self):
        self.global_targets[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "finite values in"):
            HierarchyLoss()(
                self.anatomy_logits,
                self.finding_logits,
                self.global_logits,
                global_targets=self.global_targets,
            )


if __name__ == "__main__":
    unittest.main()
