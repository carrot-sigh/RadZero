import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from exp.cxr_pt.hierarchy_visualization import (
    denormalize_image,
    save_anatomy_mask_contact_sheet,
)


class HierarchyVisualizationTest(unittest.TestCase):
    def test_contact_sheet_renders_all_masks_and_marks_unannotated(self):
        tensor = torch.zeros(3, 8, 8)
        image = denormalize_image(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        masks = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_anatomy_mask_contact_sheet(
                image,
                masks,
                ["left lung", "right lung"],
                torch.tensor([True, False]),
                Path(directory) / "overlay.png",
                columns=2,
            )
            with Image.open(path) as rendered:
                self.assertEqual(rendered.size, (16, 36))


if __name__ == "__main__":
    unittest.main()
