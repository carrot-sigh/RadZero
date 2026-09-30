import unittest
from collections import UserDict
from types import SimpleNamespace

import torch

from exp.cxr_pt.hierarchy_experiments import (
    _finding_statistics,
    _module_state_unchanged,
    _select_roi_masks,
    _to_device,
)


class HierarchyExperimentsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
