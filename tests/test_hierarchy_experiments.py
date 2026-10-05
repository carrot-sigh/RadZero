import unittest
from collections import UserDict
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from exp.cxr_pt.hierarchy_experiments import (
    _comparison_interpretation,
    _finding_statistics,
    _module_state_unchanged,
    _canonical_split_indices,
    _module_state_max_abs_delta,
    _select_roi_masks,
    _to_device,
)


class HierarchyExperimentsTest(unittest.TestCase):
    def test_canonical_split_indices_are_reproducible_and_reloadable(self):
        # Python special methods are looked up on the class, so use a tiny class
        # rather than relying on an instance attribute for len().
        class DummyDataset:
            split = "train"

            def __len__(self):
                return 100

        dataset = DummyDataset()
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "indices.json"
            first = _canonical_split_indices(dataset, 8, 42, path)
            second = _canonical_split_indices(dataset, 8, 999, path)

            self.assertEqual(first, second)
            payload = path.read_text(encoding="utf-8")
            self.assertIn('"split": "train"', payload)

    def test_canonical_split_indices_reject_invalid_saved_file(self):
        class DummyDataset:
            split = "valid"

            def __len__(self):
                return 4

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "indices.json"
            path.write_text('{"indices": [0, 0]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                _canonical_split_indices(DummyDataset(), 2, 42, path)

    def test_to_device_handles_mapping_subclasses(self):
        batch = UserDict({"tokens": torch.ones(2), "nested": {"mask": torch.zeros(2)}})

        moved = _to_device(batch, torch.device("cpu"))

        self.assertEqual(moved["tokens"].device.type, "cpu")
        self.assertEqual(moved["nested"]["mask"].device.type, "cpu")

    def test_roi_modes_are_deterministic(self):
        masks = torch.arange(12, dtype=torch.float).reshape(3, 2, 2)
        batch = {"gt_anatomy_masks": masks}

        ground_truth, permutation = _select_roi_masks(
            None, batch, None, None, SimpleNamespace(roi_mode="gt")
        )
        torch.testing.assert_close(ground_truth, masks)
        self.assertIsNone(permutation)

        full, permutation = _select_roi_masks(
            None, batch, None, None, SimpleNamespace(roi_mode="full")
        )
        torch.testing.assert_close(full, torch.ones_like(masks))
        self.assertIsNone(permutation)

        shuffled, permutation = _select_roi_masks(
            None, batch, None, None, SimpleNamespace(roi_mode="shuffled")
        )
        self.assertEqual(permutation, [2, 0, 1])
        torch.testing.assert_close(shuffled, masks[torch.tensor(permutation)])

    def test_finding_statistics_use_only_valid_pairs(self):
        logits = torch.tensor([[[8.0, -8.0], [-100.0, 100.0]]])
        targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        valid = torch.tensor([[[True, True], [False, False]]])

        statistics = _finding_statistics(logits, targets, valid, ["a", "b"])

        self.assertEqual(statistics["valid_pair_count"], 2)
        self.assertEqual(statistics["positive_pair_count"], 1)
        self.assertEqual(statistics["negative_pair_count"], 1)
        self.assertEqual(statistics["macro_f1"], 0.5)

    def test_module_state_unchanged_detects_mutation(self):
        module = torch.nn.Linear(2, 2)
        snapshot = {
            name: value.detach().clone() for name, value in module.state_dict().items()
        }

        self.assertTrue(_module_state_unchanged(module, snapshot))
        with torch.no_grad():
            module.weight.add_(1.0)
        self.assertFalse(_module_state_unchanged(module, snapshot))
        self.assertAlmostEqual(_module_state_max_abs_delta(module, snapshot), 1.0)

    def test_pilot_comparison_uses_primary_metric_and_normal_intervention(self):
        methods = {
            name: {"global": {"macro_auroc": auroc, "macro_auprc": auprc}}
            for name, auroc, auprc in (
                ("A0", 0.70, 0.71), ("A1", 0.75, 0.76),
                ("B", 0.77, 0.75), ("C", 0.78, 0.79),
            )
        }
        interventions = {
            "normal": {"macro_auroc": 0.78, "macro_auprc": 0.79},
            "shuffled_full_finding_state": {
                "macro_auroc": 0.60, "macro_auprc": 0.61,
            },
        }

        deltas, mechanism, findings = _comparison_interpretation(
            methods, interventions
        )

        self.assertAlmostEqual(deltas["C_minus_B"]["macro_auprc"], 0.04)
        self.assertAlmostEqual(
            mechanism["shuffled_full_finding_state"]["macro_auprc"], 0.18
        )
        self.assertTrue(any("single" not in finding for finding in findings))


if __name__ == "__main__":
    unittest.main()
