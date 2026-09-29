import json
import tempfile
import unittest
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from exp.cxr_pt.hierarchy_contract import build_hierarchy_contract
from exp.cxr_pt.hierarchy_dataset import (
    HierarchyJsonlDataset,
    build_hierarchy_manifest,
    compute_anatomy_pos_weights,
    hierarchy_collate_fn,
    rasterize_box,
)
from tests.test_hierarchy_contract import (
    _make_chest_imagenome_archive,
    _make_mimic_metadata,
)


class FakeImageProcessor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.ones(len(images), 3, 8, 8)}


class FakeTokenizer:
    def __call__(self, labels, **kwargs):
        assert kwargs == {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        return {"input_ids": torch.arange(len(labels)).unsqueeze(1)}


class HierarchyDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.chest_archive = _make_chest_imagenome_archive(self.root)
        self.metadata_root = _make_mimic_metadata(self.root)
        self.contract_path = self.root / "contract.json"
        contract = build_hierarchy_contract(self.chest_archive, self.metadata_root)
        self.contract_path.write_text(json.dumps(contract))
        self.image_root = self.root / "images"
        image_path = self.image_root / "p1" / "p1" / "s10" / "train-image.jpg"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (100, 100)).save(image_path)
        test_path = self.image_root / "p2" / "p2" / "s20" / "test-image.jpg"
        test_path.parent.mkdir(parents=True)
        Image.new("RGB", (100, 100)).save(test_path)
        self.manifest_path = self.root / "hierarchy.jsonl"

    def test_manifest_dataset_and_collator_preserve_contract_order(self):
        counts = build_hierarchy_manifest(
            self.chest_archive,
            self.metadata_root,
            self.image_root,
            self.contract_path,
            self.manifest_path,
        )
        self.assertEqual(counts, {"train": 1, "valid": 1})

        dataset = HierarchyJsonlDataset(
            self.manifest_path,
            self.image_root,
            self.contract_path,
            split="train",
            patch_grid_size=4,
        )
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        self.assertEqual(item["anatomy_targets"].shape, (2, 16))
        self.assertEqual(item["finding_targets"].shape, (2, 1))
        self.assertEqual(item["global_targets"].shape, (3,))
        self.assertTrue(item["anatomy_annotation_mask"].all())
        self.assertTrue(item["anatomy_query_mask"].all())
        self.assertEqual(item["finding_targets"].tolist(), [[1.0], [0.0]])
        self.assertTrue(item["finding_valid_mask"].all())
        self.assertEqual(item["global_targets"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(item["global_valid_mask"].tolist(), [True, True, False])

        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=partial(
                hierarchy_collate_fn,
                tokenizer=FakeTokenizer(),
                image_processor=FakeImageProcessor(),
                contract_path=self.contract_path,
            ),
        )
        batch = next(iter(loader))
        self.assertEqual(batch["pixel_values"].shape, (1, 3, 8, 8))
        self.assertEqual(batch["gt_anatomy_masks"].shape, (1, 2, 16))
        self.assertEqual(batch["compatibility_mask"].tolist(), [[True], [True]])
        self.assertEqual(batch["encoded_global_prompts"]["input_ids"].shape, (3, 1))

    def test_anatomy_weights_use_rasterized_train_pixels(self):
        manifest = self.root / "weights.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "split": "train",
                    "image": "p1/p1/s10/train-image.jpg",
                    "boxes": {"lung": [0, 0, 50, 50]},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "split": "valid",
                    "image": "p2/p2/s20/test-image.jpg",
                    "boxes": {"lung": [0, 0, 100, 100]},
                }
            )
            + "\n"
        )
        stats = compute_anatomy_pos_weights(
            manifest, self.image_root, self.contract_path, "train", 4
        )
        self.assertEqual(stats["num_images"], 1)
        self.assertEqual(stats["positive_pixels"][0], 4)
        self.assertEqual(stats["negative_pixels"][0], 12)
        self.assertEqual(stats["suggested_pos_weight"][0], 13 / 5)
        self.assertIsNone(stats["suggested_pos_weight"][1])

    def test_rasterize_box_clips_to_patch_grid(self):
        mask = rasterize_box([-10, 25, 75, 120], 100, 100, 4, 4)
        expected = torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=torch.float,
        )
        self.assertTrue(torch.equal(mask, expected))


if __name__ == "__main__":
    unittest.main()
