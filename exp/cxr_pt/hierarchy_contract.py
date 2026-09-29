import argparse
import csv
import gzip
import io
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


CHEST_IMAGENOME_PREFIX = "chest-imagenome-dataset-1.0.0/"
GLOBAL_ID_COLUMNS = ("subject_id", "study_id")


def build_hierarchy_contract(chest_imagenome_zip, mimic_metadata_root):
    chest_imagenome_zip = Path(chest_imagenome_zip)
    mimic_metadata_root = Path(mimic_metadata_root)
    with ZipFile(chest_imagenome_zip) as archive:
        anatomies = _read_semantic_labels(
            archive, "objects_detectable_by_bbox_pipeline_v1.txt"
        )
        attribute_relations = _read_semantic_labels(
            archive, "attribute_relations_v1.txt"
        )
        candidate_findings = [
            value.split("|", 2)[2]
            for value in attribute_relations
            if value.startswith("anatomicalfinding|yes|")
        ]
        split_by_image = _read_chest_imagenome_splits(archive)
        with tempfile.TemporaryFile() as scene_graph_file:
            with archive.open(
                CHEST_IMAGENOME_PREFIX + "silver_dataset/scene_graph.zip"
            ) as source:
                shutil.copyfileobj(source, scene_graph_file)
            scene_graph_file.seek(0)
            scene_stats = _scan_scene_graphs(
                scene_graph_file,
                split_by_image,
                anatomies,
                candidate_findings,
            )

    findings = [
        finding
        for finding in candidate_findings
        if scene_stats["finding_counts"][finding]["positive"]
        or scene_stats["finding_counts"][finding]["negative"]
    ]
    excluded_findings = [
        {
            "label": finding,
            "reason": "no explicit yes/no annotation for a detectable anatomy in train",
        }
        for finding in candidate_findings
        if finding not in findings
    ]
    global_stats = _scan_global_labels(
        mimic_metadata_root, scene_stats["train_studies"]
    )
    compatibility = []
    positive_pair_counts = []
    negative_pair_counts = []
    for anatomy in anatomies:
        compatibility.append(
            [
                int(
                    scene_stats["pair_counts"][(anatomy, finding)]["positive"] > 0
                    or scene_stats["pair_counts"][(anatomy, finding)]["negative"] > 0
                )
                for finding in findings
            ]
        )
        positive_pair_counts.append(
            [
                scene_stats["pair_counts"][(anatomy, finding)]["positive"]
                for finding in findings
            ]
        )
        negative_pair_counts.append(
            [
                scene_stats["pair_counts"][(anatomy, finding)]["negative"]
                for finding in findings
            ]
        )

    contract = {
        "schema_version": 1,
        "sources": {
            "chest_imagenome_archive": chest_imagenome_zip.name,
            "mimic_metadata_release": mimic_metadata_root.name,
        },
        "anatomy": {
            "count": len(anatomies),
            "labels": anatomies,
            "policy": "official bbox-detectable objects",
        },
        "finding": {
            "count": len(findings),
            "labels": findings,
            "excluded": excluded_findings,
            "policy": {
                "positive": "anatomicalfinding|yes|LABEL",
                "negative": "anatomicalfinding|no|LABEL",
                "unknown": "missing or conflicting yes/no annotation",
            },
            "train_counts": {
                finding: dict(scene_stats["finding_counts"][finding])
                for finding in findings
            },
            "suggested_pos_weight": [
                _positive_weight(scene_stats["finding_counts"][finding])
                for finding in findings
            ],
            "pos_weight_policy": "(negative + 1) / (positive + 1), train only",
        },
        "global": {
            "count": len(global_stats["labels"]),
            "labels": global_stats["labels"],
            "policy": {
                "positive": "1.0",
                "negative": "0.0",
                "uncertain": "ignored (-1.0)",
                "unmentioned": "ignored (blank)",
                "no_finding_negative": (
                    "blank No Finding is negative only when another global label is 1.0"
                ),
            },
            "train_counts": global_stats["counts"],
            "suggested_pos_weight": [
                _positive_weight(global_stats["counts"][label])
                for label in global_stats["labels"]
            ],
            "pos_weight_policy": "(negative + 1) / (positive + 1), train only",
        },
        "compatibility": {
            "shape": [len(anatomies), len(findings)],
            "matrix": compatibility,
            "positive_pair_counts": positive_pair_counts,
            "negative_pair_counts": negative_pair_counts,
            "policy": "pair has at least one non-conflicting explicit train label",
        },
        "statistics": {
            "scene_graph_split_counts": dict(scene_stats["split_counts"]),
            "scene_graph_conflicts": scene_stats["conflicts"],
            "global_study_split_counts": global_stats["split_counts"],
        },
    }
    validate_hierarchy_contract(contract)
    return contract


def load_hierarchy_contract(path):
    contract = json.loads(Path(path).read_text())
    validate_hierarchy_contract(contract)
    return contract


def validate_hierarchy_contract(contract):
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported hierarchy contract schema version.")

    anatomies = contract["anatomy"]["labels"]
    findings = contract["finding"]["labels"]
    global_labels = contract["global"]["labels"]
    dimensions = (
        ("anatomy", anatomies),
        ("finding", findings),
        ("global", global_labels),
    )
    for name, labels in dimensions:
        if contract[name]["count"] != len(labels):
            raise ValueError(f"{name} count does not match its label list.")
        if len(labels) != len(set(labels)):
            raise ValueError(f"{name} labels must be unique.")

    compatibility = contract["compatibility"]
    expected_shape = [len(anatomies), len(findings)]
    if compatibility["shape"] != expected_shape:
        raise ValueError("Compatibility shape does not match label dimensions.")
    for key in ("matrix", "positive_pair_counts", "negative_pair_counts"):
        matrix = compatibility[key]
        if len(matrix) != len(anatomies) or any(
            len(row) != len(findings) for row in matrix
        ):
            raise ValueError(f"{key} does not match compatibility shape.")

    for name in ("finding", "global"):
        if len(contract[name]["suggested_pos_weight"]) != len(contract[name]["labels"]):
            raise ValueError(f"{name} positive weights do not match its labels.")


def _read_semantic_labels(archive, filename):
    text = archive.read(CHEST_IMAGENOME_PREFIX + "semantics/" + filename).decode()
    return [
        value.strip() for value in text.replace("\n", "").split(",") if value.strip()
    ]


def _read_chest_imagenome_splits(archive):
    split_by_image = {}
    for split in ("train", "valid", "test"):
        path = CHEST_IMAGENOME_PREFIX + f"silver_dataset/splits/{split}.csv"
        rows = csv.DictReader(io.TextIOWrapper(archive.open(path), encoding="utf-8"))
        for row in rows:
            split_by_image[row["dicom_id"]] = split
    return split_by_image


def _scan_scene_graphs(scene_graph_file, split_by_image, anatomies, findings):
    anatomy_set = set(anatomies)
    finding_set = set(findings)
    split_counts = Counter()
    finding_counts = defaultdict(Counter)
    pair_counts = defaultdict(Counter)
    conflicts = 0
    train_studies = set()
    with ZipFile(scene_graph_file) as scene_graphs:
        for info in scene_graphs.infolist():
            if not info.filename.endswith("_SceneGraph.json"):
                continue
            scene = json.loads(scene_graphs.read(info))
            split = split_by_image.get(scene["image_id"], "excluded")
            split_counts[split] += 1
            if split != "train":
                continue
            train_studies.add((str(scene["patient_id"]), str(scene["study_id"])))
            for region in scene.get("attributes", []):
                anatomy = region.get("bbox_name")
                if anatomy not in anatomy_set:
                    continue
                polarities = defaultdict(set)
                for group in region.get("attributes", []):
                    for raw_attribute in group:
                        parts = raw_attribute.split("|", 2)
                        if (
                            len(parts) == 3
                            and parts[0] == "anatomicalfinding"
                            and parts[2] in finding_set
                        ):
                            polarities[parts[2]].add(parts[1])
                for finding, values in polarities.items():
                    if values == {"yes"}:
                        outcome = "positive"
                    elif values == {"no"}:
                        outcome = "negative"
                    else:
                        conflicts += 1
                        continue
                    finding_counts[finding][outcome] += 1
                    pair_counts[(anatomy, finding)][outcome] += 1
    return {
        "split_counts": split_counts,
        "finding_counts": finding_counts,
        "pair_counts": pair_counts,
        "conflicts": conflicts,
        "train_studies": train_studies,
    }


def _scan_global_labels(mimic_metadata_root, hierarchy_train_studies):
    split_path = mimic_metadata_root / "mimic-cxr-2.0.0-split.csv.gz"
    label_path = mimic_metadata_root / "mimic-cxr-2.0.0-chexpert.csv.gz"
    study_split = {}
    with gzip.open(split_path, "rt", newline="") as split_file:
        for row in csv.DictReader(split_file):
            key = (row["subject_id"], row["study_id"])
            previous = study_split.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Study {key} occurs in multiple MIMIC splits.")

    counts = defaultdict(Counter)
    split_counts = Counter()
    with gzip.open(label_path, "rt", newline="") as label_file:
        rows = csv.DictReader(label_file)
        labels = [name for name in rows.fieldnames if name not in GLOBAL_ID_COLUMNS]
        for row in rows:
            key = (row["subject_id"], row["study_id"])
            split = study_split.get(key, "missing")
            split_counts[split] += 1
            if key not in hierarchy_train_studies:
                continue
            any_other_positive = any(
                row[label] == "1.0" for label in labels if label != "No Finding"
            )
            for label in labels:
                value = row[label]
                if value == "1.0":
                    counts[label]["positive"] += 1
                elif value == "0.0":
                    counts[label]["negative"] += 1
                elif label == "No Finding" and not value and any_other_positive:
                    counts[label]["negative"] += 1
                elif value == "-1.0":
                    counts[label]["uncertain"] += 1
                else:
                    counts[label]["unmentioned"] += 1
    return {
        "labels": labels,
        "counts": {label: dict(counts[label]) for label in labels},
        "split_counts": dict(split_counts),
    }


def _positive_weight(counts):
    positives = counts.get("positive", 0)
    negatives = counts.get("negative", 0)
    if positives == 0:
        return None
    return (negatives + 1) / (positives + 1)


def main():
    parser = argparse.ArgumentParser(description="Build the hierarchy label contract.")
    parser.add_argument("--chest-imagenome-zip", required=True)
    parser.add_argument("--mimic-metadata-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = build_hierarchy_contract(
        args.chest_imagenome_zip, args.mimic_metadata_root
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
