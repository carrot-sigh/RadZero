import unittest

import torch

from types import SimpleNamespace

from exp.cxr_pt.hierarchy_metrics import (
    anatomy_iou_dice,
    bootstrap_multilabel_metrics,
    compute_hierarchy_trainer_metrics,
    export_hierarchy_output,
    evaluate_hierarchy_loader,
    evaluate_hierarchy_output,
    multilabel_metrics,
    pointing_game,
)


class HierarchyMetricsTest(unittest.TestCase):
    def test_bootstrap_multilabel_metrics_is_reproducible(self):
        scores = torch.tensor([[0.95, 0.05], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        targets = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
        first = bootstrap_multilabel_metrics(scores, targets, samples=50, seed=7)
        second = bootstrap_multilabel_metrics(scores, targets, samples=50, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["samples"], 50)
        self.assertGreater(first["ci95"]["auprc"]["valid_draws"], 0)
        self.assertLessEqual(first["ci95"]["auprc"]["valid_draws"], 50)

    def test_anatomy_metrics_respect_validity(self):
        targets = torch.tensor(
            [[[[1, 1], [0, 0]], [[0, 0], [0, 0]]]], dtype=torch.float
        )
        predictions = torch.tensor([[[[0.9, 0.8], [0.2, 0.1]], [[1, 1], [1, 1]]]])
        output = anatomy_iou_dice(
            predictions, targets, torch.tensor([[True, False]]), smooth=0.0
        )
        self.assertEqual(output["iou"].item(), 1.0)
        self.assertEqual(output["dice"].item(), 1.0)

    def test_pointing_game_ignores_invalid_regions(self):
        attention = torch.tensor([[[[0.9, 0.1], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]])
        targets = torch.tensor(
            [[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]], dtype=torch.float
        )
        output = pointing_game(attention, targets, torch.tensor([[True, False]]))
        self.assertEqual(output["pointing_accuracy"].item(), 1.0)

    def test_output_adapter_evaluates_and_exports_hierarchy_predictions(self):
        output = SimpleNamespace(
            anatomy=SimpleNamespace(
                roi_masks=torch.tensor([[[0.9, 0.1, 0.1, 0.1]]]),
                attention_weights=torch.zeros(1, 1, 4),
            ),
            finding=SimpleNamespace(
                finding_logits=torch.tensor([[[2.0, -2.0]]]),
                attention_weights=torch.zeros(1, 1, 2, 4),
                pair_mask=torch.ones(1, 1, 2, dtype=torch.bool),
            ),
            finding_aggregation=SimpleNamespace(
                attention_weights=torch.zeros(1, 1, 1, 2),
                null_attention=torch.zeros(1, 1),
            ),
            fusion=SimpleNamespace(
                logits=torch.tensor([[2.0]]),
                gates=torch.ones(1, 1, 1),
                fused_evidence=torch.zeros(1, 1, 2),
            ),
        )
        metrics = evaluate_hierarchy_output(
            output,
            anatomy_targets=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            finding_targets=torch.tensor([[[1.0, 0.0]]]),
            global_targets=torch.tensor([[1.0]]),
            anatomy_valid_mask=torch.tensor([[True]]),
            finding_valid_mask=torch.tensor([[[True, True]]]),
            global_valid_mask=torch.tensor([[True]]),
            grid_shape=(1, 4),
        )
        self.assertIn("anatomy_dice", metrics)
        self.assertIn("finding_f1", metrics)
        self.assertIn("global_f1", metrics)
        exported = export_hierarchy_output(output)
        self.assertEqual(tuple(exported["global_gates"].shape), (1, 1, 1))

    def test_loader_evaluation_restores_training_mode(self):
        from tests.test_hierarchical_model import HierarchyHarness, make_config

        model = HierarchyHarness(make_config(anatomies=1, findings=1, globals_=1))
        model.train()
        batch = {
            "pixel_values": torch.randn(2, 5, 8),
            "encoded_anatomy_prompts": {"features": torch.randn(1, 16)},
            "encoded_finding_prompts": {"features": torch.randn(1, 8)},
            "encoded_global_prompts": {"features": torch.randn(1, 8)},
            "anatomy_targets": torch.zeros(2, 1, 4),
            "anatomy_valid_mask": torch.ones(2, 1, dtype=torch.bool),
            "finding_targets": torch.zeros(2, 1, 1),
            "finding_valid_mask": torch.ones(2, 1, 1, dtype=torch.bool),
            "global_targets": torch.zeros(2, 1),
            "global_valid_mask": torch.ones(2, 1, dtype=torch.bool),
        }
        metrics = evaluate_hierarchy_loader(model, [batch])
        self.assertTrue(model.training)
        self.assertIn("anatomy_dice", metrics)
        self.assertIn("global_f1", metrics)

    def test_classification_metrics_ignore_unknown_labels(self):
        scores = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.9]])
        targets = torch.tensor([[1, 0], [0, -1], [1, 1]])
        valid = targets >= 0
        output = multilabel_metrics(scores, targets.clamp_min(0), valid)
        self.assertAlmostEqual(output["auroc"], 1.0)
        self.assertAlmostEqual(output["auprc"], 1.0)
        self.assertAlmostEqual(output["f1"], 1.0)

    def test_finding_metrics_use_only_valid_anatomy_finding_pairs(self):
        logits = torch.tensor(
            [
                [[8.0, -8.0], [-100.0, 100.0]],
                [[-8.0, 8.0], [100.0, -100.0]],
            ]
        )
        targets = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
            ]
        )
        valid = torch.tensor(
            [
                [[True, True], [False, False]],
                [[True, True], [False, False]],
            ]
        )
        output = SimpleNamespace(
            finding=SimpleNamespace(finding_logits=logits),
        )

        metrics = evaluate_hierarchy_output(
            output,
            finding_targets=targets,
            finding_valid_mask=valid,
        )

        self.assertEqual(metrics["finding_f1"], 1.0)
        self.assertEqual(metrics["finding_auroc"], 1.0)
        self.assertEqual(metrics["finding_auprc"], 1.0)
        self.assertEqual(metrics["finding_auroc_class_0"], 1.0)
        self.assertEqual(metrics["finding_auprc_class_0"], 1.0)
        self.assertEqual(metrics["finding_auroc_class_1"], 1.0)
        self.assertEqual(metrics["finding_auprc_class_1"], 1.0)

    def test_finding_metrics_omit_curve_values_without_both_classes(self):
        output = SimpleNamespace(
            finding=SimpleNamespace(finding_logits=torch.tensor([[[8.0]], [[7.0]]]))
        )
        metrics = evaluate_hierarchy_output(
            output,
            finding_targets=torch.ones(2, 1, 1),
            finding_valid_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        )

        self.assertNotIn("finding_auroc_class_0", metrics)
        self.assertNotIn("finding_auprc_class_0", metrics)

    def test_trainer_metric_adapter_excludes_unannotated_anatomy(self):
        predictions = (
            torch.tensor([[[0.9, 0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0]]]),
            torch.tensor([[[2.0], [2.0]]]),
            torch.tensor([[2.0]]),
        )
        labels = (
            torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
            torch.ones(1, 2, 1),
            torch.ones(1, 1),
            torch.tensor([[True, False]]),
            torch.ones(1, 2, 1, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
        )
        metrics = compute_hierarchy_trainer_metrics(
            SimpleNamespace(predictions=predictions, label_ids=labels)
        )
        self.assertEqual(metrics["anatomy_dice"], 1.0)
        self.assertEqual(metrics["anatomy_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
