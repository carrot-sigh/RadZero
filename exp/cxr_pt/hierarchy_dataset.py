import argparse
import csv
import gzip
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .hierarchy_contract import (
    CHEST_IMAGENOME_PREFIX,
    _read_chest_imagenome_splits,
    load_hierarchy_contract,
)


def build_hierarchy_manifest(
    chest_imagenome_zip, mimic_metadata_root, image_root, contract_path, output_path
):
    """Join Chest ImaGenome scene graphs with MIMIC labels into JSONL rows."""
    contract = load_hierarchy_contract(contract_path)
    global_rows, mimic_splits = _load_global_rows(
        mimic_metadata_root, contract["global"]["labels"]
    )
    anatomy_set = set(contract["anatomy"]["labels"])
    finding_set = set(contract["finding"]["labels"])
    image_root = Path(image_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)

    with ZipFile(chest_imagenome_zip) as outer:
        chest_splits = _read_chest_imagenome_splits(outer)
        with tempfile.TemporaryFile() as scene_file:
            with outer.open(
                CHEST_IMAGENOME_PREFIX + "silver_dataset/scene_graph.zip"
            ) as source:
                shutil.copyfileobj(source, scene_file)
            scene_file.seek(0)
            with ZipFile(scene_file) as scenes, output_path.open("w") as output:
                for info in scenes.infolist():
                    if not info.filename.endswith("_SceneGraph.json"):
                        continue
                    scene = json.loads(scenes.read(info))
                    image_id = scene["image_id"]
                    split = chest_splits.get(image_id)
                    if split is None:
                        counts["excluded"] += 1
                        continue
                    key = (str(scene["patient_id"]), str(scene["study_id"]))
                    mimic_split = mimic_splits.get(key)
                    if mimic_split is None:
                        counts["missing_mimic_split"] += 1
                    elif _normalize_split(mimic_split) != split:
                        counts["mimic_split_mismatch"] += 1
                    global_targets = global_rows.get(key)
                    if global_targets is None:
                        counts["missing_global_labels"] += 1
                        global_targets = {}
                    relative_image = _mimic_image_path(key, image_id)
                    if not (image_root / relative_image).is_file():
                        counts["missing_image"] += 1
                        continue
                    row = _scene_to_manifest_row(
                        scene,
                        split,
                        relative_image,
                        global_targets,
                        anatomy_set,
                        finding_set,
                    )
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts[split] += 1
    return dict(counts)


def _load_global_rows(metadata_root, labels):
    metadata_root = Path(metadata_root)
    splits = {}
    with gzip.open(
        metadata_root / "mimic-cxr-2.0.0-split.csv.gz", "rt", newline=""
    ) as source:
        for row in csv.DictReader(source):
            key = (row["subject_id"], row["study_id"])
            previous = splits.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Study {key} occurs in multiple MIMIC splits.")

    global_rows = {}
    with gzip.open(
        metadata_root / "mimic-cxr-2.0.0-chexpert.csv.gz", "rt", newline=""
    ) as source:
        for row in csv.DictReader(source):
            key = (row["subject_id"], row["study_id"])
            any_other_positive = any(
                row[label] == "1.0" for label in labels if label != "No Finding"
            )
            targets = {}
            for label in labels:
                value = row[label]
                if value == "1.0":
                    targets[label] = 1
                elif value == "0.0" or (
                    label == "No Finding" and not value and any_other_positive
                ):
                    targets[label] = 0
            global_rows[key] = targets
    return global_rows, splits


def _normalize_split(split):
    return "valid" if split == "validate" else split


def _mimic_image_path(key, image_id):
    subject_id, study_id = key
    return str(
        Path(f"p{subject_id[:2]}")
        / f"p{subject_id}"
        / f"s{study_id}"
        / f"{image_id}.jpg"
    )


def _scene_to_manifest_row(
    scene, split, relative_image, global_targets, anatomy_set, finding_set
):
    boxes = {}
    for obj in scene.get("objects", []):
        anatomy = obj.get("bbox_name")
        if anatomy in anatomy_set:
            boxes[anatomy] = [
                obj["original_x1"],
                obj["original_y1"],
                obj["original_x2"],
                obj["original_y2"],
            ]

    findings = []
    for region in scene.get("attributes", []):
        anatomy = region.get("bbox_name")
        if anatomy not in boxes:
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
                findings.append([anatomy, finding, 1])
            elif values == {"no"}:
                findings.append([anatomy, finding, 0])

    return {
        "dicom_id": scene["image_id"],
        "study_id": str(scene["study_id"]),
        "split": split,
        "image": relative_image,
        "boxes": boxes,
        "findings": findings,
        "global": global_targets,
    }


class HierarchyJsonlDataset(Dataset):
    """Random-access hierarchy dataset backed by a compact JSONL manifest."""

    def __init__(
        self, manifest_path, image_root, contract_path, split, patch_grid_size
    ):
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.contract = load_hierarchy_contract(contract_path)
        self.split = split
        if isinstance(patch_grid_size, int):
            patch_grid_size = (patch_grid_size, patch_grid_size)
        self.patch_grid_size = tuple(patch_grid_size)
        self.anatomy_index = {
            label: index
            for index, label in enumerate(self.contract["anatomy"]["labels"])
        }
        self.finding_index = {
            label: index
            for index, label in enumerate(self.contract["finding"]["labels"])
        }
        self.global_index = {
            label: index
            for index, label in enumerate(self.contract["global"]["labels"])
        }
        self.offsets = []
        with self.manifest_path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if json.loads(line)["split"] == split:
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.manifest_path.open("rb") as source:
            source.seek(self.offsets[index])
            row = json.loads(source.readline())
        image = Image.open(self.image_root / row["image"]).convert("RGB")
        width, height = image.size
        num_anatomies = len(self.anatomy_index)
        num_findings = len(self.finding_index)
        num_globals = len(self.global_index)
        grid_height, grid_width = self.patch_grid_size

        anatomy_targets = torch.zeros(num_anatomies, grid_height, grid_width)
        anatomy_valid = torch.zeros(num_anatomies, dtype=torch.bool)
        for label, box in row["boxes"].items():
            anatomy_index = self.anatomy_index[label]
            anatomy_targets[anatomy_index] = rasterize_box(
                box, width, height, grid_width, grid_height
            )
            anatomy_valid[anatomy_index] = True

        finding_targets = torch.zeros(num_anatomies, num_findings)
        finding_valid = torch.zeros(num_anatomies, num_findings, dtype=torch.bool)
        for anatomy, finding, value in row["findings"]:
            anatomy_index = self.anatomy_index[anatomy]
            finding_index = self.finding_index[finding]
            finding_targets[anatomy_index, finding_index] = value
            finding_valid[anatomy_index, finding_index] = True

        global_targets = torch.zeros(num_globals)
        global_valid = torch.zeros(num_globals, dtype=torch.bool)
        for label, value in row["global"].items():
            global_index = self.global_index[label]
            global_targets[global_index] = value
            global_valid[global_index] = True

        return {
            "image": image,
            "anatomy_targets": anatomy_targets.flatten(1),
            "anatomy_valid_mask": anatomy_valid,
            "anatomy_query_mask": anatomy_valid,
            "finding_targets": finding_targets,
            "finding_valid_mask": finding_valid,
            "global_targets": global_targets,
            "global_valid_mask": global_valid,
        }


def compute_anatomy_pos_weights(
    manifest_path,
    image_root,
    contract_path,
    split,
    patch_grid_size,
    image_size_index=None,
):
    """Compute Laplace-smoothed anatomy pixel weights for one manifest split."""
    contract = load_hierarchy_contract(contract_path)
    labels = contract["anatomy"]["labels"]
    label_index = {label: index for index, label in enumerate(labels)}
    if isinstance(patch_grid_size, int):
        patch_grid_size = (patch_grid_size, patch_grid_size)
    grid_height, grid_width = patch_grid_size
    positive = np.zeros(len(labels), dtype=np.int64)
    valid = np.zeros(len(labels), dtype=np.int64)
    image_root = Path(image_root)
    num_rows = 0
    with Path(manifest_path).open() as source:
        for line in source:
            row = json.loads(line)
            if row["split"] != split:
                continue
            num_rows += 1
            if image_size_index is None:
                with Image.open(image_root / row["image"]) as image:
                    width, height = image.size
            else:
                width, height = image_size_index[row["dicom_id"]]
            for label, box in row["boxes"].items():
                index = label_index[label]
                mask = rasterize_box(box, width, height, grid_width, grid_height)
                positive[index] += int(mask.sum().item())
                valid[index] += mask.numel()
    negative = valid - positive
    weights = [
        float((negative[index] + 1) / (positive[index] + 1)) if valid[index] else None
        for index in range(len(labels))
    ]
    return {
        "split": split,
        "patch_grid_size": [grid_height, grid_width],
        "num_images": num_rows,
        "positive_pixels": positive.tolist(),
        "negative_pixels": negative.tolist(),
        "valid_pixels": valid.tolist(),
        "suggested_pos_weight": weights,
        "pos_weight_policy": "(negative + 1) / (positive + 1), rasterized train masks only",
    }


def rasterize_box(box, image_width, image_height, grid_width, grid_height):
    x1, y1, x2, y2 = box
    left = max(0, min(grid_width, math.floor(x1 / image_width * grid_width)))
    top = max(0, min(grid_height, math.floor(y1 / image_height * grid_height)))
    right = max(0, min(grid_width, math.ceil(x2 / image_width * grid_width)))
    bottom = max(0, min(grid_height, math.ceil(y2 / image_height * grid_height)))
    mask = torch.zeros(grid_height, grid_width)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 1
    return mask


def hierarchy_collate_fn(batch, tokenizer, image_processor, contract_path):
    contract = load_hierarchy_contract(contract_path)
    processed = image_processor([item["image"] for item in batch], return_tensors="pt")
    pixel_values = (
        processed if torch.is_tensor(processed) else processed["pixel_values"]
    )
    if torch.is_tensor(pixel_values):
        output = {"pixel_values": pixel_values}
    else:
        output = {"pixel_values": torch.as_tensor(np.asarray(pixel_values))}
    for key in (
        "anatomy_targets",
        "anatomy_valid_mask",
        "anatomy_query_mask",
        "finding_targets",
        "finding_valid_mask",
        "global_targets",
        "global_valid_mask",
    ):
        output[key] = torch.stack([item[key] for item in batch])
    output["gt_anatomy_masks"] = output["anatomy_targets"]
    output["global_query_mask"] = torch.ones(
        len(batch), contract["global"]["count"], dtype=torch.bool
    )
    output["compatibility_mask"] = torch.tensor(
        contract["compatibility"]["matrix"], dtype=torch.bool
    )
    for name, labels in (
        ("anatomy", contract["anatomy"]["labels"]),
        ("finding", contract["finding"]["labels"]),
        ("global", contract["global"]["labels"]),
    ):
        output[f"encoded_{name}_prompts"] = tokenizer(
            labels, padding=True, truncation=True, return_tensors="pt"
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Build hierarchy JSONL manifest.")
    parser.add_argument("--chest-imagenome-zip", required=True)
    parser.add_argument("--mimic-metadata-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    counts = build_hierarchy_manifest(
        args.chest_imagenome_zip,
        args.mimic_metadata_root,
        args.image_root,
        args.contract,
        args.output,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
