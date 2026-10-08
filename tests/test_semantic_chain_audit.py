import argparse
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from exp.cxr_pt.semantic_chain_audit.audit import _mapping_indexes, run
from exp.cxr_pt.semantic_chain_audit.facts import (
    Fact,
    JoinAudit,
    derive_distribution_fact,
    detect_cross_source_disagreement,
    detect_cross_level_reuse,
    normalize_state,
    resolve_same_source_states,
)
from exp.cxr_pt.semantic_chain_audit.hierarchy_exploration import _state_summary


def make_fact(source, state, locator="record:1", anatomy=None):
    return Fact(
        source=source,
        source_version="synthetic-fixture",
        annotation_quality="synthetic_fixture_not_real_evidence",
        source_record_locator=locator,
        subject_id="subject-fixture",
        study_id="study-fixture",
        image_id="image-fixture",
        report_id=None,
        split="train",
        annotation_granularity="anatomy" if anatomy else "study",
        raw_category="fixture",
        raw_concept="fixture concept",
        raw_state=state,
        normalized_concept="fixture concept",
        normalized_state=state,
        anatomy=anatomy,
        mapping_rule_id="fixture-rule",
    )


class StateTests(unittest.TestCase):
    def test_positive_negative_uncertain_unmentioned_unavailable(self):
        self.assertEqual(normalize_state("chexpert", "1.0"), "positive")
        self.assertEqual(normalize_state("chexpert", "0.0"), "explicit_negative")
        self.assertEqual(normalize_state("chexpert", "-1.0"), "uncertain")
        self.assertEqual(normalize_state("chexpert", ""), "unmentioned")
        self.assertEqual(
            normalize_state("chexpert", "", available=False), "unavailable"
        )

    def test_missing_does_not_become_negative(self):
        self.assertEqual(
            normalize_state("chest_imagenome_attribute", None), "unmentioned"
        )

    def test_same_source_conflict_preserves_conflict(self):
        self.assertEqual(
            resolve_same_source_states({"positive", "explicit_negative"}), "conflict"
        )

    def test_derived_parent_does_not_turn_partial_children_into_negative(self):
        self.assertEqual(_state_summary(set()), "unmentioned")
        self.assertEqual(_state_summary({"explicit_negative"}), "explicit_negative")
        self.assertEqual(_state_summary({"positive", "explicit_negative"}), "conflict")

    def test_cross_source_disagreement_does_not_overwrite(self):
        facts = [
            make_fact("chest_imagenome_attribute", "positive"),
            make_fact("chexpert", "explicit_negative", "record:2"),
        ]
        self.assertTrue(detect_cross_source_disagreement(facts))
        self.assertEqual(
            [fact.normalized_state for fact in facts], ["positive", "explicit_negative"]
        )


class ProvenanceTests(unittest.TestCase):
    def test_distribution_requires_both_positive_locations(self):
        left = make_fact("chest_imagenome_attribute", "positive", "left:1", "left_lung")
        missing_right = make_fact(
            "chest_imagenome_attribute", "unmentioned", "right:1", "right_lung"
        )
        self.assertIsNone(
            derive_distribution_fact(
                [left, missing_right],
                concept="bilateral_distribution",
                rule_id="fixture-bilateral-v1",
                required_anatomies=["left_lung", "right_lung"],
            )
        )
        right = make_fact(
            "chest_imagenome_attribute", "positive", "right:2", "right_lung"
        )
        derived = derive_distribution_fact(
            [left, right],
            concept="bilateral_distribution",
            rule_id="fixture-bilateral-v1",
            required_anatomies=["left_lung", "right_lung"],
        )
        self.assertTrue(derived.derived)
        self.assertEqual(set(derived.provenance), {"left:1", "right:2"})

    def test_cross_level_source_fact_reuse_is_detected(self):
        fact = make_fact("chest_imagenome_attribute", "positive")
        issues = detect_cross_level_reuse({"L1": [fact], "L2": [fact]})
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reused_level"], "L2")

    def test_mixed_label_is_not_split_by_mapping(self):
        mapping_path = (
            Path(__file__).parents[1]
            / "exp/cxr_pt/configs/semantic_chain_mapping.v1.json"
        )
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        attr, cue, chexpert, _, _ = _mapping_indexes(mapping)
        self.assertNotIn(("anatomicalfinding", "pulmonary edema/hazy opacity"), attr)
        self.assertIn("pulmonary edema/hazy opacity", mapping["mixed_terms"])

    def test_new_disease_screening_rejects_same_target_l2(self):
        mapping_path = (
            Path(__file__).parents[1]
            / "exp/cxr_pt/configs/semantic_chain_mapping.v1.json"
        )
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        _, _, chexpert, _, active = _mapping_indexes(mapping)
        self.assertIn(("global_label", "Pneumonia"), chexpert)
        self.assertIn(("global_label", "Pneumothorax"), chexpert)
        self.assertIn(("global_label", "Cardiomegaly"), chexpert)
        self.assertFalse(active[("pneumothorax", "L2")])
        self.assertFalse(active[("cardiomegaly", "L2")])
        pneumonia_pair = next(
            pair
            for pair in mapping["comparable_pairs"]
            if pair["comparison_id"] == "pneumonia_same_name"
        )
        self.assertEqual(pneumonia_pair["regional_category"], "disease")


class JoinTests(unittest.TestCase):
    def test_duplicate_and_expected_one_to_many(self):
        audit = JoinAudit()
        audit.add_image("image-1", "study-1", "subject-1", "train")
        audit.add_image("image-2", "study-1", "subject-1", "train")
        audit.add_image("image-1", "study-1", "subject-1", "train")
        summary = audit.summary()
        self.assertEqual(summary["duplicate_image_keys"], 1)
        self.assertEqual(summary["one_to_many_studies"], 1)
        self.assertEqual(summary["unexpected_many_to_many_studies"], 0)
        self.assertEqual(summary["unique_studies"], 1)

    def test_missing_key_is_retained_as_conflicting_duplicate(self):
        audit = JoinAudit()
        audit.add_image("", "study-1", "subject-1", "train")
        audit.add_image("", "study-2", "subject-2", "valid")
        self.assertEqual(audit.summary()["conflicting_image_keys"], 1)


class SyntheticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = Path(__file__).parents[1]
        self.mapping = self.repo / "exp/cxr_pt/configs/semantic_chain_mapping.v1.json"
        self._build_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def _build_fixture(self):
        manifest_rows = [
            {
                "dicom_id": "train-image-a",
                "study_id": "1001",
                "split": "train",
                "image": "p00/p00000001/s1001/train-image-a.jpg",
                "boxes": {"right lung": [0, 0, 90, 90]},
                "findings": [["right lung", "vascular congestion", 1]],
                "global": {"Edema": 1},
            },
            {
                "dicom_id": "train-image-b",
                "study_id": "1001",
                "split": "train",
                "image": "p00/p00000001/s1001/train-image-b.jpg",
                "boxes": {"left lung": [-1, 0, 101, 100]},
                "findings": [["left lung", "vascular congestion", 1]],
                "global": {"Edema": 1},
            },
            {
                "dicom_id": "valid-image",
                "study_id": "2001",
                "split": "valid",
                "image": "p00/p00000002/s2001/valid-image.jpg",
                "boxes": {"right lung": [0, 0, 100, 100]},
                "findings": [["right lung", "atelectasis", 1]],
                "global": {},
            },
            {
                "dicom_id": "test-image",
                "study_id": "3001",
                "split": "test",
                "image": "p00/p00000003/s3001/test-image.jpg",
                "boxes": {},
                "findings": [["right lung", "secret-test-concept", 1]],
                "global": {"Edema": 1},
            },
        ]
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
        )
        atomic_rows = [
            {
                "dicom_id": "train-image-a",
                "study_id": "1001",
                "subject_id": "00000001",
                "region": "right_lung",
                "category": "anatomicalfinding",
                "raw_label": "vascular congestion",
                "relation": "yes",
                "phrase_id": 0,
                "texture_cues": '[["texture|yes|interstitial"]]',
                "severity_cues": "[[]]",
                "temporal_cues": "[[]]",
            },
            {
                "dicom_id": "train-image-a",
                "study_id": "1001",
                "subject_id": "00000001",
                "region": "right_lung",
                "category": "anatomicalfinding",
                "raw_label": "pulmonary edema/hazy opacity",
                "relation": "yes",
                "phrase_id": 0,
                "texture_cues": '[["texture|yes|interstitial"]]',
                "severity_cues": "[[]]",
                "temporal_cues": "[[]]",
            },
            {
                "dicom_id": "valid-image",
                "study_id": "2001",
                "subject_id": "00000002",
                "region": "right_lung",
                "category": "anatomicalfinding",
                "raw_label": "atelectasis",
                "relation": "yes",
                "phrase_id": 0,
                "texture_cues": "[[]]",
                "severity_cues": "[[]]",
                "temporal_cues": "[[]]",
            },
            {
                "dicom_id": "test-image",
                "study_id": "3001",
                "subject_id": "00000003",
                "region": "right_lung",
                "category": "anatomicalfinding",
                "raw_label": "secret-test-concept",
                "relation": "yes",
                "phrase_id": 0,
                "texture_cues": "[[]]",
                "severity_cues": "[[]]",
                "temporal_cues": "[[]]",
            },
        ]
        self.atomic = self.root / "atomic.parquet"
        pq.write_table(pa.Table.from_pylist(atomic_rows), self.atomic)
        self.comparison_relations = self.root / "comparison_relations.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "dicom_id": "train-image-a",
                        "study_id": "1001",
                        "subject_id": "00000001",
                        "relationship_names": '["comparison|yes|improved"]',
                    },
                    {
                        "dicom_id": "test-image",
                        "study_id": "3001",
                        "subject_id": "00000003",
                        "relationship_names": '["comparison|yes|worsened"]',
                    },
                ]
            ),
            self.comparison_relations,
        )

        self.chexpert = self.root / "chexpert.csv.gz"
        labels = [
            "subject_id",
            "study_id",
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
        with gzip.open(self.chexpert, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=labels)
            writer.writeheader()
            writer.writerow(
                {
                    "subject_id": "00000001",
                    "study_id": "1001",
                    "Edema": "1.0",
                    "Atelectasis": "-1.0",
                }
            )
            writer.writerow(
                {
                    "subject_id": "00000002",
                    "study_id": "2001",
                    "Edema": "",
                    "Atelectasis": "0.0",
                }
            )
            writer.writerow(
                {"subject_id": "00000003", "study_id": "3001", "Edema": "1.0"}
            )

        self.metadata = self.root / "metadata.csv.gz"
        with gzip.open(self.metadata, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dicom_id", "Rows", "Columns"])
            writer.writeheader()
            for image in (
                "train-image-a",
                "train-image-b",
                "valid-image",
                "test-image",
            ):
                writer.writerow({"dicom_id": image, "Rows": 100, "Columns": 100})
        self.split = self.root / "split.csv.gz"
        with gzip.open(self.split, "wt", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["dicom_id", "study_id", "subject_id", "split"]
            )
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow(
                    {
                        "dicom_id": row["dicom_id"],
                        "study_id": row["study_id"],
                        "subject_id": _fixture_subject(row["image"]),
                        "split": row["split"],
                    }
                )
        self.outer = self.root / "chest-imagenome-dataset-1.0.0.zip"
        self.scene = self.root / "scene_graph.zip"
        self.reports = self.root / "reports.zip"
        for path in (self.outer, self.scene, self.reports):
            with ZipFile(path, "w") as archive:
                archive.writestr("fixture.txt", "synthetic fixture; not real evidence")

    def test_full_synthetic_run_deduplicates_study_and_excludes_test_semantics(self):
        output = self.root / "output"
        args = argparse.Namespace(
            manifest=self.manifest,
            atomic_annotations=self.atomic,
            comparison_relations=self.comparison_relations,
            chexpert=self.chexpert,
            metadata=self.metadata,
            mimic_split=self.split,
            chest_imagenome_zip=self.outer,
            scene_graph_zip=self.scene,
            reports_zip=self.reports,
            image_root=None,
            mapping=self.mapping,
            output_dir=output,
            sample_per_split=0,
            seed=7,
            batch_size=2,
            review_per_stratum=2,
        )
        summary = run(args)
        self.assertEqual(summary["scope"], "full_train_valid")
        self.assertEqual(summary["selected_images"], 3)
        self.assertEqual(summary["selected_studies"], 2)
        inventory_text = (output / "raw_concept_inventory.csv").read_text()
        self.assertNotIn("secret-test-concept", inventory_text)
        relationship_text = (output / "native_relationship_inventory.csv").read_text()
        self.assertIn("improved", relationship_text)
        self.assertNotIn("worsened", relationship_text)
        with (output / "candidate_chain_feasibility.csv").open() as handle:
            feasibility = list(csv.DictReader(handle))
        train_edema = next(
            row
            for row in feasibility
            if row["candidate_chain"] == "pulmonary_edema" and row["split"] == "train"
        )
        self.assertEqual(train_edema["denominator_studies"], "1")
        self.assertEqual(train_edema["l1_valid_studies"], "0")
        self.assertEqual(train_edema["all_three_valid_studies"], "0")
        join_quality = json.loads((output / "join_quality.json").read_text())
        self.assertGreater(join_quality["mapped_cue_scopes_excluded_by_config"], 0)
        config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(
            config["test_policy"],
            "structure counts only; labels excluded from semantic statistics",
        )


def _fixture_subject(image_path):
    return next(
        part[1:]
        for part in Path(image_path).parts
        if part.startswith("p") and len(part) == 9
    )


if __name__ == "__main__":
    unittest.main()
