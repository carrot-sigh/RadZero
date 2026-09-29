import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from exp.cxr_pt.hierarchy_contract import (
    build_hierarchy_contract,
    load_hierarchy_contract,
    validate_hierarchy_contract,
)


PREFIX = "chest-imagenome-dataset-1.0.0/"


def _write_csv(path, fieldnames, rows):
    with gzip.open(path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_chest_imagenome_archive(tmp_path):
    scene_path = tmp_path / "scene_graph.zip"
    scene = {
        "image_id": "train-image",
        "patient_id": 1,
        "study_id": 10,
        "objects": [
            {
                "bbox_name": "lung",
                "original_x1": 10,
                "original_y1": 20,
                "original_x2": 60,
                "original_y2": 80,
            },
            {
                "bbox_name": "heart",
                "original_x1": 30,
                "original_y1": 30,
                "original_x2": 70,
                "original_y2": 70,
            },
        ],
        "attributes": [
            {
                "bbox_name": "lung",
                "attributes": [
                    [
                        "anatomicalfinding|yes|opacity",
                        "anatomicalfinding|yes|fracture",
                        "anatomicalfinding|no|fracture",
                    ]
                ],
            },
            {
                "bbox_name": "heart",
                "attributes": [["anatomicalfinding|no|opacity"]],
            },
        ],
    }
    with ZipFile(scene_path, "w") as archive:
        archive.writestr("train-image_SceneGraph.json", json.dumps(scene))
        archive.writestr(
            "test-image_SceneGraph.json",
            json.dumps(
                {
                    "image_id": "test-image",
                    "patient_id": 2,
                    "study_id": 20,
                    "objects": [],
                    "attributes": [],
                }
            ),
        )

    outer_path = tmp_path / "chest-imagenome.zip"
    with ZipFile(outer_path, "w") as archive:
        archive.writestr(
            PREFIX + "semantics/objects_detectable_by_bbox_pipeline_v1.txt",
            "lung,heart,",
        )
        archive.writestr(
            PREFIX + "semantics/attribute_relations_v1.txt",
            "anatomicalfinding|yes|opacity,anatomicalfinding|yes|fracture,",
        )
        archive.writestr(
            PREFIX + "silver_dataset/splits/train.csv",
            "dicom_id\ntrain-image\n",
        )
        archive.writestr(
            PREFIX + "silver_dataset/splits/valid.csv",
            "dicom_id\ntest-image\n",
        )
        archive.writestr(PREFIX + "silver_dataset/splits/test.csv", "dicom_id\n")
        archive.write(scene_path, PREFIX + "silver_dataset/scene_graph.zip")
    return outer_path


def _make_mimic_metadata(tmp_path):
    metadata_root = tmp_path / "mimic"
    metadata_root.mkdir()
    _write_csv(
        metadata_root / "mimic-cxr-2.0.0-split.csv.gz",
        ["subject_id", "study_id", "dicom_id", "split"],
        [
            {
                "subject_id": "1",
                "study_id": "10",
                "dicom_id": "a",
                "split": "train",
            },
            {
                "subject_id": "2",
                "study_id": "20",
                "dicom_id": "b",
                "split": "validate",
            },
        ],
    )
    _write_csv(
        metadata_root / "mimic-cxr-2.0.0-chexpert.csv.gz",
        ["subject_id", "study_id", "No Finding", "Opacity", "Edema"],
        [
            {
                "subject_id": "1",
                "study_id": "10",
                "No Finding": "",
                "Opacity": "1.0",
                "Edema": "-1.0",
            },
            {
                "subject_id": "2",
                "study_id": "20",
                "No Finding": "1.0",
                "Opacity": "",
                "Edema": "0.0",
            },
        ],
    )
    return metadata_root


class HierarchyContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.tmp_path = Path(self.temporary_directory.name)

    def build_contract(self):
        return build_hierarchy_contract(
            _make_chest_imagenome_archive(self.tmp_path),
            _make_mimic_metadata(self.tmp_path),
        )

    def test_build_contract_freezes_training_policies(self):
        contract = self.build_contract()

        self.assertEqual(contract["anatomy"]["labels"], ["lung", "heart"])
        self.assertEqual(contract["finding"]["labels"], ["opacity"])
        self.assertEqual(
            contract["finding"]["excluded"],
            [
                {
                    "label": "fracture",
                    "reason": (
                        "no explicit yes/no annotation for a detectable anatomy "
                        "in train"
                    ),
                }
            ],
        )
        self.assertEqual(
            contract["finding"]["train_counts"]["opacity"],
            {"positive": 1, "negative": 1},
        )
        self.assertEqual(contract["finding"]["suggested_pos_weight"], [1.0])
        self.assertEqual(contract["compatibility"]["matrix"], [[1], [1]])
        self.assertEqual(contract["statistics"]["scene_graph_conflicts"], 1)

        self.assertEqual(
            contract["global"]["labels"], ["No Finding", "Opacity", "Edema"]
        )
        self.assertEqual(
            contract["global"]["train_counts"]["No Finding"], {"negative": 1}
        )
        self.assertEqual(contract["global"]["train_counts"]["Opacity"], {"positive": 1})
        self.assertEqual(contract["global"]["train_counts"]["Edema"], {"uncertain": 1})
        self.assertEqual(
            contract["statistics"]["global_study_split_counts"],
            {"train": 1, "validate": 1},
        )

    def test_load_contract_validates_dimensions(self):
        contract = self.build_contract()
        path = self.tmp_path / "contract.json"
        path.write_text(json.dumps(contract))

        self.assertEqual(load_hierarchy_contract(path), contract)

        contract["compatibility"]["matrix"][0] = []
        with self.assertRaisesRegex(ValueError, "matrix does not match"):
            validate_hierarchy_contract(contract)


if __name__ == "__main__":
    unittest.main()
