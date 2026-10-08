"""CLI for a RadGraph-free MIMIC-CXR semantic-chain feasibility audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import pyarrow.parquet as pq

from .facts import JoinAudit, normalize_state, resolve_same_source_states


ALLOWED_SEMANTIC_SPLITS = {"train", "valid"}
HISTORICAL_STUDIES = {"56045920", "59176408", "52411857"}
FACT_SCHEMA = [
    "source",
    "source_version",
    "annotation_quality",
    "source_record_locator",
    "subject_id",
    "study_id",
    "image_id",
    "report_id",
    "split",
    "annotation_granularity",
    "raw_category",
    "raw_concept",
    "raw_state",
    "normalized_concept",
    "normalized_state",
    "anatomy",
    "evidence_index",
    "mapping_rule_id",
    "derived",
    "provenance",
    "linkage_status",
]


def _subject_from_image_path(value: str) -> str | None:
    for part in Path(value).parts:
        if part.startswith("p") and len(part) == 9 and part[1:].isdigit():
            return part[1:]
    return None


def _sha256_prefix(path: Path, size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(size))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_clue(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path is not None else None, "available": False}
    stat = path.stat()
    return {
        "path": str(path),
        "available": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256_first_1mib": _sha256_prefix(path),
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames=None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


@dataclass
class InventoryCell:
    rows: int = 0
    studies: set[str] = field(default_factory=set)
    patients: set[str] = field(default_factory=set)
    anatomies: set[str] = field(default_factory=set)


class Inventory:
    def __init__(self) -> None:
        self.cells: dict[tuple[str, str, str, str, str], InventoryCell] = {}

    def add(
        self,
        source: str,
        split: str,
        category: str,
        concept: str,
        state: str,
        study: str,
        patient: str,
        anatomy: str | None = None,
    ) -> None:
        key = (source, split, category, concept, state)
        cell = self.cells.setdefault(key, InventoryCell())
        cell.rows += 1
        cell.studies.add(study)
        cell.patients.add(patient)
        if anatomy:
            cell.anatomies.add(anatomy)

    def rows(
        self,
        exported_findings: set[str],
        mapped_roles: dict[tuple[str, str, str], set[str]],
    ):
        output = []
        for key, cell in sorted(self.cells.items()):
            source, split, category, concept, state = key
            roles = mapped_roles.get((source, category, concept), set())
            output.append(
                {
                    "source": source,
                    "split": split,
                    "raw_category": category,
                    "raw_concept": concept,
                    "normalized_state": state,
                    "row_count": cell.rows,
                    "study_count": len(cell.studies),
                    "patient_count": len(cell.patients),
                    "anatomy_count": len(cell.anatomies),
                    "anatomies": "|".join(sorted(cell.anatomies)),
                    "in_current_41_finding_export": category == "anatomicalfinding"
                    and concept in exported_findings,
                    "candidate_roles": "|".join(sorted(roles)),
                    "mapping_need": (
                        "configured_review" if roles else "unmapped_raw_attribute"
                    ),
                }
            )
        return output


class MappedStates:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
        self.locators: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)

    def add(
        self,
        split: str,
        chain: str,
        level: str,
        study: str,
        rule_id: str,
        state: str,
        locator: str,
    ) -> None:
        key = (split, chain, level, study, rule_id)
        self.values[key].add(state)
        self.locators[key].add(locator)

    def level_states(
        self,
        split: str,
        chain: str,
        level: str,
        study: str,
        active_rules: set[str],
    ) -> list[str]:
        output = []
        for rule_id in active_rules:
            states = self.values.get((split, chain, level, study, rule_id))
            if states:
                # Study-level presence aggregation is not a conflict resolver:
                # a positive left region and negative right region may both be valid.
                if "positive" in states:
                    output.append("positive")
                elif "explicit_negative" in states:
                    output.append("explicit_negative")
                elif "uncertain" in states:
                    output.append("uncertain")
                elif "conflict" in states:
                    output.append("conflict")
                else:
                    output.append("unmentioned")
        return output


def _read_manifest(args, inventory: Inventory, exported_findings: set[str]):
    split_rows = Counter()
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    join = JoinAudit()
    with args.manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            split = str(row.get("split"))
            split_rows[split] += 1
            image = str(row.get("image", ""))
            subject = str(
                row.get("patient_id") or _subject_from_image_path(image) or ""
            )
            study = str(row.get("study_id", ""))
            dicom = str(row.get("dicom_id", ""))
            join.add_image(dicom, study, subject, split)
            if split in ALLOWED_SEMANTIC_SPLITS:
                candidates[split].append(
                    {
                        "dicom_id": dicom,
                        "study_id": study,
                        "subject_id": subject,
                        "split": split,
                        "image": image,
                        "boxes": row.get("boxes", {}),
                        "findings": row.get("findings", []),
                        "global": row.get("global", {}),
                        "line_number": line_number,
                    }
                )

    sampled = bool(args.sample_per_split)
    if sampled:
        rng = random.Random(args.seed)
        for split in sorted(candidates):
            if len(candidates[split]) > args.sample_per_split:
                candidates[split] = rng.sample(candidates[split], args.sample_per_split)

    records = [row for split in sorted(candidates) for row in candidates[split]]
    image_meta = {row["dicom_id"]: row for row in records}
    study_meta: dict[str, dict[str, str]] = {}
    regional_states: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    exported_global: dict[tuple[str, str], set[str]] = defaultdict(set)
    duplicate_finding_rows = 0
    for row in records:
        split, study, subject = row["split"], row["study_id"], row["subject_id"]
        prior = study_meta.setdefault(study, {"split": split, "subject_id": subject})
        if prior != {"split": split, "subject_id": subject}:
            raise ValueError(f"Study metadata conflict for study {study}")
        seen = set()
        for value in row["findings"]:
            if not isinstance(value, list) or len(value) != 3:
                continue
            anatomy, concept, raw_state = value
            key = (str(anatomy), str(concept), str(raw_state))
            if key in seen:
                duplicate_finding_rows += 1
            seen.add(key)
            state = normalize_state("chest_imagenome_export", raw_state)
            inventory.add(
                "chest_imagenome_export",
                split,
                "anatomicalfinding",
                str(concept),
                state,
                study,
                subject,
                str(anatomy),
            )
            exported_findings.add(str(concept))
            regional_states[(split, study, str(concept))].add(state)
        for concept, raw_state in row["global"].items():
            state = normalize_state("chexpert", raw_state)
            exported_global[(study, str(concept))].add(state)
    return {
        "split_rows": dict(split_rows),
        "records": records,
        "image_meta": image_meta,
        "study_meta": study_meta,
        "join": join,
        "regional_states": regional_states,
        "exported_global": exported_global,
        "duplicate_finding_rows": duplicate_finding_rows,
        "sampled": sampled,
    }


def _mapping_indexes(mapping):
    attr = defaultdict(list)
    cue = defaultdict(list)
    chexpert = defaultdict(list)
    mapped_roles = defaultdict(set)
    active = defaultdict(set)
    for rule in mapping["concepts"]:
        source = rule["source"]
        for term in rule["source_terms"]:
            indexed_rule = dict(rule)
            indexed_rule["_state_key"] = f"{rule['concept_id']}::{term}"
            key = (source, rule["source_category"], term)
            mapped_roles[key].add(
                f"{rule['candidate_chain']}:{rule['target_level']}:{rule['review_status']}"
            )
            target = {
                "chest_imagenome_attribute": attr,
                "chest_imagenome_cue": cue,
                "chexpert": chexpert,
            }[source]
            target[(rule["source_category"], term)].append(indexed_rule)
            if rule.get("use_for_coverage", True):
                active[(rule["candidate_chain"], rule["target_level"])].add(
                    indexed_rule["_state_key"]
                )
    return attr, cue, chexpert, mapped_roles, active


def _read_chexpert(args, manifest, mapping, inventory, mapped, chexpert_rules):
    study_meta = manifest["study_meta"]
    raw_states: dict[tuple[str, str], str] = {}
    raw_rows = 0
    linked_rows = 0
    duplicate_keys = 0
    seen_keys = set()
    with gzip.open(args.chexpert, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        labels = [
            name
            for name in reader.fieldnames or []
            if name not in {"subject_id", "study_id"}
        ]
        for row_number, row in enumerate(reader, 2):
            raw_rows += 1
            study = str(row["study_id"])
            subject = str(row["subject_id"])
            key = (subject, study)
            if key in seen_keys:
                duplicate_keys += 1
            seen_keys.add(key)
            meta = study_meta.get(study)
            if meta is None or meta["subject_id"] != subject:
                continue
            linked_rows += 1
            split = meta["split"]
            for label in labels:
                raw = row.get(label, "")
                state = normalize_state("chexpert", raw)
                raw_states[(study, label)] = state
                inventory.add(
                    "chexpert", split, "global_label", label, state, study, subject
                )
                for rule in chexpert_rules.get(("global_label", label), []):
                    mapped.add(
                        split,
                        rule["candidate_chain"],
                        rule["target_level"],
                        study,
                        rule["_state_key"],
                        state,
                        f"{args.chexpert.name}:row:{row_number}:label:{label}",
                    )
    return {
        "labels": labels,
        "raw_rows": raw_rows,
        "linked_study_rows": linked_rows,
        "duplicate_subject_study_keys": duplicate_keys,
        "states": raw_states,
    }


def _cue_at_phrase(value: Any, phrase_id: int) -> list[str]:
    if value is None:
        return []
    try:
        groups = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    if not isinstance(groups, list) or phrase_id < 0 or phrase_id >= len(groups):
        return []
    group = groups[phrase_id]
    if isinstance(group, str):
        return [group]
    return (
        [item for item in group if isinstance(item, str)]
        if isinstance(group, list)
        else []
    )


def _read_atomic(
    args, manifest, inventory, mapped, attr_rules, cue_rules, comparable_terms
):
    image_meta = manifest["image_meta"]
    columns = [
        "dicom_id",
        "study_id",
        "subject_id",
        "region",
        "category",
        "raw_label",
        "relation",
        "phrase_id",
        "texture_cues",
        "severity_cues",
        "temporal_cues",
    ]
    parquet = pq.ParquetFile(args.atomic_annotations)
    matched_rows = 0
    source_mismatches = 0
    exact_states: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    scope_locators: dict[tuple[str, str, str, str, str, str], str] = {}
    raw_regional: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    cue_scope_states: dict[tuple[str, str, str, str, int, str, str], set[str]] = (
        defaultdict(set)
    )
    cue_scope_locators: dict[tuple[str, str, str, str, int, str, str], str] = {}
    phrase_concepts: dict[tuple[str, str, str, str, int], set[str]] = defaultdict(set)
    seen_cues: set[tuple[str, str, int, str]] = set()
    mixed_positive_studies: set[tuple[str, str]] = set()
    historical: dict[str, list[dict[str, str]]] = defaultdict(list)
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        data = batch.to_pydict()
        for index, dicom_value in enumerate(data["dicom_id"]):
            dicom = str(dicom_value)
            meta = image_meta.get(dicom)
            if meta is None:
                continue
            matched_rows += 1
            study = str(data["study_id"][index])
            subject = str(data["subject_id"][index])
            if study != meta["study_id"] or subject != meta["subject_id"]:
                source_mismatches += 1
                continue
            split = meta["split"]
            category = str(data["category"][index])
            concept = str(data["raw_label"][index])
            raw_state = str(data["relation"][index])
            anatomy = str(data["region"][index])
            phrase_id = int(data["phrase_id"][index])
            phrase_concepts[(split, study, dicom, anatomy, phrase_id)].add(concept)
            try:
                state = normalize_state("chest_imagenome_attribute", raw_state)
            except ValueError:
                state = "unavailable"
            inventory.add(
                "chest_imagenome_attribute",
                split,
                category,
                concept,
                state,
                study,
                subject,
                anatomy,
            )
            locator = (
                f"scene_graph/{dicom}_SceneGraph.json:phrase:{data['phrase_id'][index]}"
            )
            if attr_rules.get((category, concept)) or concept in comparable_terms:
                scope = (split, study, dicom, anatomy, category, concept)
                exact_states[scope].add(state)
                scope_locators[scope] = locator
            if (
                category == "anatomicalfinding"
                and concept == "pulmonary edema/hazy opacity"
                and state == "positive"
            ):
                mixed_positive_studies.add((split, study))
            if study in HISTORICAL_STUDIES and len(historical[study]) < 200:
                if concept in comparable_terms or concept in {
                    "linear/patchy atelectasis",
                    "pulmonary edema/hazy opacity",
                }:
                    historical[study].append(
                        {
                            "study_id": study,
                            "stage": "raw_scene_graph_attribute",
                            "split": split,
                            "dicom_locator": locator,
                            "anatomy": anatomy,
                            "category": category,
                            "concept": concept,
                            "state": state,
                        }
                    )

            for cue_column, cue_category in (
                ("texture_cues", "texture"),
                ("severity_cues", "severity"),
                ("temporal_cues", "temporal"),
            ):
                for cue_raw in _cue_at_phrase(data[cue_column][index], phrase_id):
                    cue_key = (dicom, anatomy, phrase_id, cue_raw)
                    if cue_key in seen_cues:
                        continue
                    seen_cues.add(cue_key)
                    parts = cue_raw.split("|", 2)
                    if len(parts) != 3:
                        continue
                    raw_category, raw_relation, raw_concept = parts
                    if raw_category != cue_category:
                        continue
                    try:
                        cue_state = normalize_state("chest_imagenome_cue", raw_relation)
                    except ValueError:
                        cue_state = "unavailable"
                    inventory.add(
                        "chest_imagenome_cue",
                        split,
                        raw_category,
                        raw_concept,
                        cue_state,
                        study,
                        subject,
                        anatomy,
                    )
                    cue_locator = locator + f":{cue_column}"
                    for rule in cue_rules.get((raw_category, raw_concept), []):
                        cue_scope = (
                            split,
                            study,
                            dicom,
                            anatomy,
                            phrase_id,
                            raw_category,
                            rule["_state_key"],
                        )
                        cue_scope_states[cue_scope].add(cue_state)
                        cue_scope_locators[cue_scope] = cue_locator
    conflicts = [
        key
        for key, states in exact_states.items()
        if resolve_same_source_states(states) == "conflict"
    ]
    conflict_studies = defaultdict(set)
    for scope, states in exact_states.items():
        split, study, _, _, category, concept = scope
        resolved = resolve_same_source_states(states)
        rules = attr_rules.get((category, concept), [])
        if resolved == "conflict":
            for rule in rules:
                if rule.get("use_for_coverage", True):
                    conflict_studies[
                        (split, rule["candidate_chain"], rule["target_level"])
                    ].add(study)
            continue
        if concept in comparable_terms:
            raw_regional[(split, study, concept)].add(resolved)
        for rule in rules:
            mapped.add(
                split,
                rule["candidate_chain"],
                rule["target_level"],
                study,
                rule["_state_key"],
                resolved,
                scope_locators[scope],
            )
    cue_conflicts = 0
    excluded_mapped_cue_scopes = 0
    for scope, states in cue_scope_states.items():
        split, study, _, _, _, category, state_key = scope
        resolved = resolve_same_source_states(states)
        rules = [
            rule
            for (rule_category, _), indexed in cue_rules.items()
            if rule_category == category
            for rule in indexed
            if rule["_state_key"] == state_key
        ]
        if resolved == "conflict":
            cue_conflicts += 1
            for rule in rules:
                if rule.get("use_for_coverage", True):
                    conflict_studies[
                        (split, rule["candidate_chain"], rule["target_level"])
                    ].add(study)
            continue
        for rule in rules:
            phrase_scope = scope[:5]
            if set(rule.get("exclusions", [])) & phrase_concepts[phrase_scope]:
                excluded_mapped_cue_scopes += 1
                continue
            if (
                "explicit yes only" in rule["assertion_policy"]
                and resolved != "positive"
            ):
                continue
            mapped.add(
                split,
                rule["candidate_chain"],
                rule["target_level"],
                study,
                state_key,
                resolved,
                cue_scope_locators[scope],
            )
    return {
        "parquet_rows": parquet.metadata.num_rows,
        "matched_rows": matched_rows,
        "source_key_mismatches": source_mismatches,
        "cue_facts": len(seen_cues),
        "excluded_mapped_cue_scopes": excluded_mapped_cue_scopes,
        "exact_scope_conflicts": len(conflicts) + cue_conflicts,
        "mapped_conflict_studies": conflict_studies,
        "raw_regional": raw_regional,
        "mixed_positive_studies": mixed_positive_studies,
        "historical": historical,
    }


def _aggregate_region_state(states: set[str] | None) -> str:
    if not states:
        return "unmentioned"
    if "positive" in states:
        return "positive"
    if "explicit_negative" in states:
        return "explicit_negative"
    if "uncertain" in states:
        return "uncertain"
    return resolve_same_source_states(states)


def _candidate_rows(
    manifest, mapping, mapped, active_rules, chexpert_states, mapped_conflict_studies
):
    rows = []
    study_meta = manifest["study_meta"]
    for split in ("train", "valid"):
        studies = {
            study for study, meta in study_meta.items() if meta["split"] == split
        }
        patients = {study_meta[study]["subject_id"] for study in studies}
        for chain, chain_meta in mapping["chains"].items():
            level_sets: dict[str, dict[str, set[str]]] = {}
            level_valid: dict[str, set[str]] = {}
            row: dict[str, Any] = {
                "candidate_chain": chain,
                "display_name": chain_meta["display_name"],
                "split": split,
                "denominator_studies": len(studies),
                "denominator_patients": len(patients),
                "semantic_distinctness": chain_meta["semantic_distinctness"],
                "relationship_evidence": chain_meta["relationship_evidence"],
                "explicit_relation_studies": 0,
                "l3_independence": chain_meta["l3_independence"],
                "conclusion_grade": chain_meta["expected_conclusion"],
            }
            for level in ("L1", "L2", "L3"):
                states_by_study = {}
                valid = set()
                for study in studies:
                    states = set(
                        mapped.level_states(
                            split,
                            chain,
                            level,
                            study,
                            active_rules.get((chain, level), set()),
                        )
                    )
                    states_by_study[study] = states
                    if states & {"positive", "explicit_negative"}:
                        valid.add(study)
                level_sets[level] = states_by_study
                level_valid[level] = valid
                row[f"{level.lower()}_valid_studies"] = len(valid)
                row[f"{level.lower()}_valid_patients"] = len(
                    {study_meta[s]["subject_id"] for s in valid}
                )
                for state in ("positive", "explicit_negative", "uncertain"):
                    row[f"{level.lower()}_{state}_studies"] = sum(
                        state in values for values in states_by_study.values()
                    )
                row[f"{level.lower()}_conflict_studies"] = len(
                    mapped_conflict_studies.get((split, chain, level), set())
                )
                if level == "L3":
                    row["l3_unmentioned_studies"] = sum(
                        values == {"unmentioned"}
                        for values in states_by_study.values()
                    )
                    row["l3_unavailable_studies"] = sum(
                        not values or values == {"unavailable"}
                        for values in states_by_study.values()
                    )
            row["l1_l2_valid_studies"] = len(level_valid["L1"] & level_valid["L2"])
            row["l1_l3_valid_studies"] = len(level_valid["L1"] & level_valid["L3"])
            row["l2_l3_valid_studies"] = len(level_valid["L2"] & level_valid["L3"])
            all_three = level_valid["L1"] & level_valid["L2"] & level_valid["L3"]
            row["all_three_valid_studies"] = len(all_three)
            row["all_three_valid_patients"] = len(
                {study_meta[s]["subject_id"] for s in all_three}
            )
            row["at_least_two_valid_studies"] = sum(
                sum(study in level_valid[level] for level in ("L1", "L2", "L3")) >= 2
                for study in studies
            )
            l2_positive = {
                study
                for study, states in level_sets["L2"].items()
                if "positive" in states
            }
            l3_positive = {
                study
                for study, states in level_sets["L3"].items()
                if "positive" in states
            }
            l3_negative = {
                study
                for study, states in level_sets["L3"].items()
                if "explicit_negative" in states
            }
            row["l2_positive_l3_positive_studies"] = len(l2_positive & l3_positive)
            row["l2_positive_l3_explicit_negative_studies"] = len(
                l2_positive & l3_negative
            )
            rows.append(row)
    return rows


def _l2_pattern_rows(manifest, mapping, mapped, active_rules, chexpert_states):
    rows = []
    study_meta = manifest["study_meta"]
    for chain in mapping["chains"]:
        rules = active_rules.get((chain, "L2"), set())
        l3_labels = {
            term
            for rule in mapping["concepts"]
            if rule["candidate_chain"] == chain
            and rule["target_level"] == "L3"
            and rule["source"] == "chexpert"
            and rule.get("use_for_coverage", True)
            for term in rule["source_terms"]
        }
        if len(l3_labels) != 1:
            raise ValueError(
                f"Chain {chain!r} must configure exactly one active CheXpert L3 label."
            )
        l3_label = next(iter(l3_labels))
        for split in ("train", "valid"):
            counts = Counter()
            for study, meta in study_meta.items():
                if meta["split"] != split:
                    continue
                positive_terms = []
                for state_key in rules:
                    states = mapped.values.get(
                        (split, chain, "L2", study, state_key), set()
                    )
                    if "positive" in states:
                        positive_terms.append(state_key.split("::", 1)[-1])
                pattern = (
                    "|".join(sorted(positive_terms))
                    if positive_terms
                    else "<no_positive_l2>"
                )
                l3_state = chexpert_states.get((study, l3_label), "unavailable")
                counts[(pattern, l3_state)] += 1
            for (pattern, l3_state), count in sorted(counts.items()):
                rows.append(
                    {
                        "candidate_chain": chain,
                        "split": split,
                        "positive_l2_pattern": pattern,
                        "l3_state": l3_state,
                        "study_count": count,
                        "interpretation": "study co-occurrence; not an annotated relation",
                    }
                )
    return rows


def _read_native_relationships(args, manifest):
    parquet = pq.ParquetFile(args.comparison_relations)
    cells: dict[tuple[str, str], InventoryCell] = {}
    matched_rows = 0
    malformed_names = 0
    for batch in parquet.iter_batches(
        batch_size=args.batch_size,
        columns=["dicom_id", "study_id", "subject_id", "relationship_names"],
    ):
        data = batch.to_pydict()
        for index, dicom_value in enumerate(data["dicom_id"]):
            meta = manifest["image_meta"].get(str(dicom_value))
            if meta is None:
                continue
            matched_rows += 1
            try:
                names = json.loads(data["relationship_names"][index])
            except (TypeError, json.JSONDecodeError):
                malformed_names += 1
                continue
            if not isinstance(names, list):
                malformed_names += 1
                continue
            for name in names:
                if not isinstance(name, str):
                    malformed_names += 1
                    continue
                key = (meta["split"], name)
                cell = cells.setdefault(key, InventoryCell())
                cell.rows += 1
                cell.studies.add(meta["study_id"])
                cell.patients.add(meta["subject_id"])
    rows = []
    for (split, name), cell in sorted(cells.items()):
        parts = name.split("|", 2)
        rows.append(
            {
                "split": split,
                "raw_relationship": name,
                "relationship_category": parts[0] if parts else "",
                "raw_state": parts[1] if len(parts) > 1 else "",
                "raw_value": parts[2] if len(parts) > 2 else "",
                "row_count": cell.rows,
                "study_count": len(cell.studies),
                "patient_count": len(cell.patients),
                "native_semantics": "object-object comparison relation",
                "usable_as_l1_l2_or_l2_l3_support_edge": False,
                "raw_endpoints_exist": True,
                "flattened_parquet_preserves_endpoints": False,
            }
        )
    return {
        "parquet_rows": parquet.metadata.num_rows,
        "matched_rows": matched_rows,
        "malformed_relationship_names": malformed_names,
        "rows": rows,
    }


def _state_combinations(manifest, mapping, raw_regional, chexpert_states):
    rows = []
    study_meta = manifest["study_meta"]
    for pair in mapping["comparable_pairs"]:
        for split in ("train", "valid"):
            counts = Counter()
            studies = {
                study for study, meta in study_meta.items() if meta["split"] == split
            }
            for study in studies:
                regional = _aggregate_region_state(
                    raw_regional.get((split, study, pair["regional"]))
                )
                global_state = chexpert_states.get(
                    (study, pair["global"]), "unavailable"
                )
                counts[(regional, global_state)] += 1
            for (regional, global_state), count in sorted(counts.items()):
                rows.append(
                    {
                        "comparison_id": pair["comparison_id"],
                        "regional_concept": pair["regional"],
                        "global_concept": pair["global"],
                        "split": split,
                        "regional_state": regional,
                        "global_state": global_state,
                        "study_count": count,
                        "classification": (
                            "cross_source_directional_disagreement"
                            if {regional, global_state}
                            == {"positive", "explicit_negative"}
                            else "not_directional_disagreement"
                        ),
                    }
                )
    regional_positive_studies = defaultdict(set)
    for (split, study, _), states in raw_regional.items():
        if "positive" in states:
            regional_positive_studies[split].add(study)
    for split in ("train", "valid"):
        studies = {
            study for study, meta in study_meta.items() if meta["split"] == split
        }
        counts = Counter()
        for study in studies:
            any_regional_positive = study in regional_positive_studies[split]
            no_finding = chexpert_states.get((study, "No Finding"), "unavailable")
            counts[(any_regional_positive, no_finding)] += 1
        for (regional_positive, no_finding), count in sorted(counts.items(), key=str):
            rows.append(
                {
                    "comparison_id": "any_regional_positive_vs_no_finding",
                    "regional_concept": "any configured comparable regional finding",
                    "global_concept": "No Finding",
                    "split": split,
                    "regional_state": (
                        "positive" if regional_positive else "no_positive_observed"
                    ),
                    "global_state": no_finding,
                    "study_count": count,
                    "classification": (
                        "cooccurrence_requires_review"
                        if regional_positive and no_finding == "positive"
                        else "other"
                    ),
                }
            )
    return rows


def _join_source_availability(args, manifest):
    image_ids = set(manifest["image_meta"])
    studies = set(manifest["study_meta"])
    mimic_image_splits = {}
    duplicate_mimic_image_keys = 0
    with gzip.open(args.mimic_split, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            dicom = str(row["dicom_id"])
            if dicom not in image_ids:
                continue
            value = (str(row["study_id"]), str(row["subject_id"]), str(row["split"]))
            if dicom in mimic_image_splits:
                duplicate_mimic_image_keys += 1
            mimic_image_splits[dicom] = value
    mimic_split_mismatches = 0
    mimic_key_mismatches = 0
    for dicom, meta in manifest["image_meta"].items():
        source = mimic_image_splits.get(dicom)
        if source is None:
            continue
        study, subject, split = source
        if study != meta["study_id"] or subject != meta["subject_id"]:
            mimic_key_mismatches += 1
        normalized = "valid" if split == "validate" else split
        if normalized != meta["split"]:
            mimic_split_mismatches += 1

    scene_ids = set()
    with ZipFile(args.scene_graph_zip) as archive:
        for name in archive.namelist():
            if name.endswith("_SceneGraph.json"):
                scene_ids.add(Path(name).name.split("_SceneGraph.json", 1)[0])
    report_studies = set()
    reports_available = bool(args.reports_zip and args.reports_zip.is_file())
    if reports_available:
        study_pattern = re.compile(r"(?:^|/)s(\d+)\.txt$")
        with ZipFile(args.reports_zip) as archive:
            for name in archive.namelist():
                match = study_pattern.search(name)
                if match:
                    report_studies.add(match.group(1))
    return {
        "mimic_split_rows_linked": len(mimic_image_splits),
        "mimic_split_missing_images": len(image_ids - set(mimic_image_splits)),
        "mimic_split_duplicate_image_keys": duplicate_mimic_image_keys,
        "mimic_join_key_mismatches": mimic_key_mismatches,
        "mimic_vs_canonical_split_mismatched_images": mimic_split_mismatches,
        "scene_graph_members": len(scene_ids),
        "selected_images_without_scene_graph_member": len(image_ids - scene_ids),
        "report_members_with_study_locator": len(report_studies),
        "selected_studies_without_report_member": (
            len(studies - report_studies) if reports_available else None
        ),
        "report_archive_available": reports_available,
    }


def _export_state_audit(manifest, chexpert_states, labels):
    rows = []
    counts = Counter()
    for study, meta in manifest["study_meta"].items():
        other_positive = any(
            chexpert_states.get((study, label)) == "positive"
            for label in labels
            if label != "No Finding"
        )
        for label in labels:
            raw_state = chexpert_states.get((study, label), "unavailable")
            export_values = manifest["exported_global"].get((study, label))
            export_state = (
                resolve_same_source_states(export_values)
                if export_values
                else "omitted_from_export"
            )
            if raw_state in {"positive", "explicit_negative"}:
                expected = raw_state
            elif (
                label == "No Finding" and raw_state == "unmentioned" and other_positive
            ):
                expected = "explicit_negative"
            else:
                expected = "omitted_from_export"
            outcome = (
                "matches_project_policy" if export_state == expected else "mismatch"
            )
            counts[
                (meta["split"], label, raw_state, export_state, expected, outcome)
            ] += 1
    for key, count in sorted(counts.items()):
        split, label, raw_state, export_state, expected, outcome = key
        rows.append(
            {
                "split": split,
                "label": label,
                "raw_chexpert_state": raw_state,
                "current_export_state": export_state,
                "expected_under_documented_project_policy": expected,
                "outcome": outcome,
                "study_count": count,
            }
        )
    return rows


def _bbox_audit(args, manifest):
    dimensions = {}
    with gzip.open(args.metadata, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            dicom = str(row["dicom_id"])
            if dicom in manifest["image_meta"]:
                try:
                    dimensions[dicom] = (int(row["Columns"]), int(row["Rows"]))
                except (TypeError, ValueError):
                    continue
    counters = defaultdict(Counter)
    missing_images = Counter()
    for row in manifest["records"]:
        split = row["split"]
        dim = dimensions.get(row["dicom_id"])
        if args.image_root and not (args.image_root / row["image"]).is_file():
            missing_images[split] += 1
        for anatomy, box in row["boxes"].items():
            counters[split]["boxes"] += 1
            if not isinstance(box, list) or len(box) != 4:
                counters[split]["invalid_format"] += 1
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                counters[split]["degenerate"] += 1
            if dim is None:
                counters[split]["reference_dimensions_unavailable"] += 1
                continue
            width, height = dim
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                counters[split]["outside_image_extent"] += 1
            if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
                counters[split]["fully_outside"] += 1
    rows = []
    for split in ("train", "valid"):
        values = counters[split]
        rows.append(
            {
                "split": split,
                "image_count": sum(
                    row["split"] == split for row in manifest["records"]
                ),
                "bbox_count": values["boxes"],
                "invalid_format": values["invalid_format"],
                "degenerate": values["degenerate"],
                "outside_image_extent_before_loader_clamp": values[
                    "outside_image_extent"
                ],
                "fully_outside": values["fully_outside"],
                "reference_dimensions_unavailable": values[
                    "reference_dimensions_unavailable"
                ],
                "missing_image_files": missing_images[split],
                "coordinate_format": "absolute [x1,y1,x2,y2] in original JPG pixel coordinates",
                "loader_behavior": "rasterize_box clamps to patch-grid bounds using actual PIL image size",
            }
        )
    return rows


def _historical_trace(args, manifest, raw_scan, chexpert_states):
    output = []
    by_study = defaultdict(list)
    for row in manifest["records"]:
        if row["study_id"] in HISTORICAL_STUDIES:
            by_study[row["study_id"]].append(row)
    for study in sorted(HISTORICAL_STUDIES):
        rows = by_study.get(study, [])
        if not rows:
            output.append(
                {"study_id": study, "status": "not_in_selected_train_or_valid_scope"}
            )
            continue
        split = rows[0]["split"]
        output.extend(raw_scan["historical"].get(study, []))
        for row in rows:
            for anatomy, concept, value in row["findings"]:
                if value == 1 or concept in {
                    "atelectasis",
                    "lung opacity",
                    "pulmonary edema/hazy opacity",
                }:
                    output.append(
                        {
                            "study_id": study,
                            "stage": "current_manifest_export",
                            "split": split,
                            "dicom_locator": f"{args.manifest}:line:{row['line_number']}",
                            "anatomy": anatomy,
                            "concept": concept,
                            "state": normalize_state("chest_imagenome_export", value),
                        }
                    )
            for concept, value in row["global"].items():
                output.append(
                    {
                        "study_id": study,
                        "stage": "current_manifest_export_global",
                        "split": split,
                        "dicom_locator": f"{args.manifest}:line:{row['line_number']}",
                        "concept": concept,
                        "state": normalize_state("chexpert", value),
                    }
                )
                output.append(
                    {
                        "study_id": study,
                        "stage": "dataset_loader_target_valid_mask",
                        "split": split,
                        "concept": concept,
                        "state": f"target={value};valid_mask=true",
                    }
                )
        for concept in ("Atelectasis", "Edema", "No Finding", "Lung Opacity"):
            state = chexpert_states.get((study, concept), "unavailable")
            output.append(
                {
                    "study_id": study,
                    "stage": "raw_chexpert",
                    "split": split,
                    "concept": concept,
                    "state": state,
                }
            )
            if not any(concept in row["global"] for row in rows):
                output.append(
                    {
                        "study_id": study,
                        "stage": "dataset_loader_target_valid_mask",
                        "split": split,
                        "concept": concept,
                        "state": "target=0(default tensor);valid_mask=false",
                    }
                )
    return output


def _review_queue(
    args, manifest, mapping, mapped, active_rules, raw_scan, chexpert_states, combos
):
    candidates: dict[str, set[str]] = defaultdict(set)
    study_meta = manifest["study_meta"]
    for study, meta in study_meta.items():
        if meta["split"] != "train":
            continue
        for chain in mapping["chains"]:
            valid = {}
            for level in ("L1", "L2", "L3"):
                states = set(
                    mapped.level_states(
                        "train",
                        chain,
                        level,
                        study,
                        active_rules.get((chain, level), set()),
                    )
                )
                valid[level] = bool(states & {"positive", "explicit_negative"})
                if "conflict" in states or "uncertain" in states:
                    candidates["uncertain_or_internal_conflict"].add(study)
            if valid["L2"] and valid["L3"] and not valid["L1"]:
                candidates["l2_l3_without_l1"].add(study)
            if all(valid.values()):
                candidates["three_levels_present_requires_semantic_review"].add(study)
        if ("train", study) in raw_scan["mixed_positive_studies"]:
            candidates["mixed_label_present_do_not_split"].add(study)
    for row in combos:
        if (
            row["split"] == "train"
            and row["classification"] == "cross_source_directional_disagreement"
        ):
            # Counts do not retain identifiers; recover exact studies below.
            pair = next(
                (
                    p
                    for p in mapping["comparable_pairs"]
                    if p["comparison_id"] == row["comparison_id"]
                ),
                None,
            )
            if pair:
                for study, meta in study_meta.items():
                    if meta["split"] != "train":
                        continue
                    regional = _aggregate_region_state(
                        raw_scan["raw_regional"].get(("train", study, pair["regional"]))
                    )
                    global_state = chexpert_states.get(
                        (study, pair["global"]), "unavailable"
                    )
                    if {regional, global_state} == {"positive", "explicit_negative"}:
                        candidates["cross_source_directional_disagreement"].add(study)
    rng = random.Random(args.seed)
    local_rows = []
    summary_rows = []
    for reason, studies in sorted(candidates.items()):
        ordered = sorted(studies)
        chosen = rng.sample(ordered, min(args.review_per_stratum, len(ordered)))
        summary_rows.append(
            {
                "review_reason": reason,
                "eligible_train_studies": len(ordered),
                "sampled_studies": len(chosen),
                "review_type": (
                    "structured-label/report fidelity review"
                    if reason != "three_levels_present_requires_semantic_review"
                    else "semantic mapping review plus qualified image review"
                ),
            }
        )
        for study in chosen:
            images = [r for r in manifest["records"] if r["study_id"] == study]
            local_rows.append(
                {
                    "study_id": study,
                    "subject_id": study_meta[study]["subject_id"],
                    "dicom_ids": "|".join(sorted(r["dicom_id"] for r in images)),
                    "review_reason": reason,
                    "source_locators": "|".join(
                        f"scene_graph/{r['dicom_id']}_SceneGraph.json" for r in images
                    ),
                    "report_locator": f"p{study_meta[study]['subject_id'][:2]}/p{study_meta[study]['subject_id']}/s{study}.txt",
                    "question": "Is the structured assertion faithful, semantically distinct across levels, and visually valid when image review is required?",
                    "required_reviewer": "radiology-domain reviewer for image truth; data curator for extraction fidelity",
                }
            )
    return local_rows, summary_rows


def _resource_inventory(args, manifest, chexpert, raw_scan):
    resources = []
    for role, path in (
        ("current_hierarchy_manifest", args.manifest),
        ("flattened_chest_imagenome_attributes", args.atomic_annotations),
        ("flattened_chest_imagenome_comparison_relations", args.comparison_relations),
        ("mimic_chexpert_labels", args.chexpert),
        ("mimic_metadata", args.metadata),
        ("mimic_split", args.mimic_split),
        ("chest_imagenome_outer_archive", args.chest_imagenome_zip),
        ("chest_imagenome_scene_graph_archive", args.scene_graph_zip),
        ("mimic_reports_archive_optional", args.reports_zip),
    ):
        clue = _file_clue(path)
        clue["role"] = role
        resources.append(clue)
    inventory = {
        "resources": resources,
        "versions": {
            "chest_imagenome": "1.0.0 (archive path and internal release prefix)",
            "mimic_cxr_jpg": "2.1.0 path; metadata filenames retain 2.0.0 naming",
            "chexpert": "MIMIC-CXR-JPG release CSV; report-derived weak labels",
        },
        "counts": {
            "manifest_by_split": manifest["split_rows"],
            "selected_semantic_images": len(manifest["records"]),
            "selected_semantic_studies": len(manifest["study_meta"]),
            "current_export_finding_vocabulary": len(
                {
                    finding[1]
                    for row in manifest["records"]
                    for finding in row["findings"]
                    if isinstance(finding, list) and len(finding) == 3
                }
            ),
            "chexpert_rows_all_mimic": chexpert["raw_rows"],
            "chexpert_rows_linked_to_selected_scope": chexpert["linked_study_rows"],
            "atomic_parquet_rows_all_mimic_split": raw_scan["parquet_rows"],
            "atomic_rows_linked_to_selected_scope": raw_scan["matched_rows"],
        },
        "schemas": {
            "scene_graph": {
                "top_level": [
                    "StudyDateTime",
                    "StudyOrder",
                    "age_decile",
                    "attributes",
                    "gender",
                    "image_id",
                    "objects",
                    "patient_id",
                    "reason_for_exam",
                    "relationships",
                    "study_id",
                    "viewpoint",
                ],
                "object": [
                    "object_id",
                    "name",
                    "bbox_name",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "width",
                    "height",
                    "original_x1",
                    "original_y1",
                    "original_x2",
                    "original_y2",
                    "original_width",
                    "original_height",
                    "synsets",
                ],
                "attribute": [
                    "object_id",
                    "bbox_name",
                    "attributes",
                    "attributes_ids",
                    "phrase_IDs",
                    "phrases",
                    "sections",
                    "severity_cues",
                    "temporal_cues",
                    "texture_cues",
                ],
                "relationship": [
                    "relationship_id",
                    "subject_id",
                    "object_id",
                    "predicate",
                    "phrase",
                    "relationship_names",
                    "relationship_contexts",
                    "bbox_name",
                ],
            },
            "current_manifest": [
                "dicom_id",
                "study_id",
                "split",
                "image",
                "boxes",
                "findings",
                "global",
            ],
            "chexpert": ["subject_id", "study_id"] + chexpert["labels"],
            "fact_representation": FACT_SCHEMA,
        },
        "quality": {
            "scene_graph": "Chest ImaGenome silver_dataset: report-derived attributes and automated bbox pipeline",
            "gold_subset": "Official archive contains separate 500-study attribute and 1000-image bbox gold resources; not merged into silver facts in this audit",
            "chexpert": "Automated report labeler output; no independent expert adjudication asserted",
            "current_manifest": "Project export derived from silver scene graphs plus CheXpert",
        },
        "anonymous_structure_example": {
            "scene_graph": {
                "image_id": "<redacted>",
                "study_id": "<redacted>",
                "objects": [
                    {
                        "object_id": "<redacted>",
                        "bbox_name": "right lung",
                        "original_x1": 0,
                    }
                ],
                "attributes": [
                    {
                        "bbox_name": "right lung",
                        "attributes": [["anatomicalfinding|yes|lung opacity"]],
                        "phrase_IDs": ["<index>"],
                    }
                ],
                "relationships": [
                    {
                        "predicate": "<release relationship value>",
                        "relationship_names": ["comparison|yes|..."],
                    }
                ],
            }
        },
        "unconfirmed_assumptions": [
            "Auxiliary cue arrays are aligned to attribute phrase groups by list position; this audit marks mapped cues as derived/proposed.",
            "Chest ImaGenome relationship records are comparison relations, not evidence-to-interpretation support edges.",
            "No independent expert adjudication links CheXpert L3 to proposed L1/L2 nodes.",
        ],
    }
    return inventory


def _git_revision() -> dict[str, str]:
    def run(*command):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short"),
    }


def _report(
    args,
    resource_inventory,
    candidate_rows,
    join_quality,
    bbox_rows,
    state_rows,
    mapping,
):
    scope = (
        "SAMPLE/SMOKE RESULT"
        if args.sample_per_split
        else "FULL TRAIN/VALIDATION RESULT"
    )
    lines = [
        "# MIMIC-CXR RadGraph-free three-level semantic-chain audit",
        "",
        f"Execution scope: **{scope}**. Test labels were excluded from semantic statistics.",
        "",
        "## Direct, normalized, derived, and missing evidence",
        "",
        "- Direct source facts: Chest ImaGenome silver anatomy objects/attributes and raw CheXpert cells.",
        "- Normalized facts: source states mapped to positive, explicit_negative, uncertain, unmentioned, conflict, or unavailable without cross-source overwrite.",
        "- Derived facts: phrase-aligned auxiliary cues only; every such mapping is marked proposed and derived.",
        "- Human/expert gaps: independent morphology labels, evidence-to-interpretation edges, report fidelity review, and image truth adjudication.",
        "",
        "## Resource result",
        "",
        f"- Manifest split rows: `{resource_inventory['counts']['manifest_by_split']}`.",
        f"- Selected train/valid images: {resource_inventory['counts']['selected_semantic_images']}; studies: {resource_inventory['counts']['selected_semantic_studies']}.",
        f"- Chest ImaGenome atomic rows in full parquet: {resource_inventory['counts']['atomic_parquet_rows_all_mimic_split']}; linked in scope: {resource_inventory['counts']['atomic_rows_linked_to_selected_scope']}.",
        f"- CheXpert rows: {resource_inventory['counts']['chexpert_rows_all_mimic']}; linked study rows in scope: {resource_inventory['counts']['chexpert_rows_linked_to_selected_scope']}.",
        "",
        "## Source capability and export loss",
        "",
        "The raw Chest ImaGenome scene graph contains object IDs, anatomy names, anatomy bounding boxes, grouped attributes, phrase IDs/text links, cue arrays, and comparison relations. The current manifest retains anatomy boxes, 41 selected anatomical findings with explicit yes/no, and sparse re-encoded global labels. It drops object IDs, phrases/sections, raw attribute IDs, all non-selected categories, auxiliary cues, comparison relations, raw conflicts, and the distinction between CheXpert uncertain and blank because both are omitted from `global`.",
        "",
        "The anatomy boxes are not lesion boxes. Attributes are silver report-derived weak supervision attached to anatomy objects. CheXpert is study/report-level weak supervision copied to image rows; study counts below are deduplicated.",
        "",
        "## Candidate feasibility",
        "",
        "| Chain | Split | L1 valid | L2 valid | L3 valid | >=2 levels | All 3 | Relation | Conclusion |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in candidate_rows:
        lines.append(
            "| {display_name} | {split} | {l1_valid_studies} | {l2_valid_studies} | {l3_valid_studies} | {at_least_two_valid_studies} | {all_three_valid_studies} | co-occurrence only (0 explicit) | {conclusion_grade} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "A valid layer means at least one configured concept has an explicit positive or explicit negative source assertion. It is not restricted to positive cases. Counts use unique studies; patient counts and per-state counts are in `candidate_chain_feasibility.csv`. L1 auxiliary cues support no explicit-negative denominator, so absence remains unmentioned.",
            "",
            "## Relationship evidence",
            "",
            "Chest ImaGenome natively supplies anatomy-object to attribute association and object-object comparison records. Neither is an annotated support/explanation edge from L1 to L2 or L2 to L3. Candidate overlap therefore measures study co-occurrence, not reasoning-chain instances. The available data has concept-node supervision, while layer structure remains a research hypothesis.",
            "",
            "## Shortcut result",
            "",
        ]
    )
    for item in mapping["static_shortcuts"]:
        lines.append(f"- `{item['term']}`: {item['risk']}; {item['decision']}.")
    lines.extend(
        [
            "",
            "CheXpert L3 labels are not calculated as an OR of project L2 labels. This prevents one form of circular construction, but does not establish independent clinical evidence: both sources are derived from the same report, and same-name regional/global pairs have material lexical leakage risk.",
            "",
            "## Join and geometry",
            "",
            f"Join summary: `{join_quality}`.",
            "",
            f"BBox summary: `{bbox_rows}`. Out-of-extent coordinates are reported before the existing loader clamp and are not automatically classified as annotation errors. The loader rasterizes against actual PIL dimensions and clamps to the patch grid; no box was modified.",
            "",
            "## Outputs and boundaries",
            "",
            "Machine-readable tables accompany this report. `review_queue_local.csv` and `historical_trace_local.csv` contain controlled locators/identifiers and must remain local. No report text or new gold label was generated. No model was trained.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if mapping["policy"].get("test_semantics_forbidden") is not True:
        raise ValueError("Mapping must explicitly forbid test semantic use.")
    attr_rules, cue_rules, chexpert_rules, mapped_roles, active_rules = (
        _mapping_indexes(mapping)
    )
    inventory = Inventory()
    mapped = MappedStates()
    exported_findings: set[str] = set()

    manifest = _read_manifest(args, inventory, exported_findings)
    chexpert = _read_chexpert(
        args, manifest, mapping, inventory, mapped, chexpert_rules
    )
    comparable_terms = {pair["regional"] for pair in mapping["comparable_pairs"]}
    comparable_terms.update(mapping["mixed_terms"])
    raw_scan = _read_atomic(
        args,
        manifest,
        inventory,
        mapped,
        attr_rules,
        cue_rules,
        comparable_terms,
    )
    candidate_rows = _candidate_rows(
        manifest,
        mapping,
        mapped,
        active_rules,
        chexpert["states"],
        raw_scan["mapped_conflict_studies"],
    )
    l2_pattern_rows = _l2_pattern_rows(
        manifest, mapping, mapped, active_rules, chexpert["states"]
    )
    native_relationships = _read_native_relationships(args, manifest)
    state_rows = _state_combinations(
        manifest, mapping, raw_scan["raw_regional"], chexpert["states"]
    )
    bbox_rows = _bbox_audit(args, manifest)
    source_availability = _join_source_availability(args, manifest)
    export_state_rows = _export_state_audit(
        manifest, chexpert["states"], chexpert["labels"]
    )
    historical_rows = _historical_trace(args, manifest, raw_scan, chexpert["states"])
    review_local, review_summary = _review_queue(
        args,
        manifest,
        mapping,
        mapped,
        active_rules,
        raw_scan,
        chexpert["states"],
        state_rows,
    )
    join_quality = manifest["join"].summary()
    join_quality.update(
        {
            "duplicate_finding_rows_in_export": manifest["duplicate_finding_rows"],
            "atomic_source_key_mismatches": raw_scan["source_key_mismatches"],
            "native_relationship_rows_linked": native_relationships["matched_rows"],
            "malformed_native_relationship_names": native_relationships[
                "malformed_relationship_names"
            ],
            "audited_exact_scope_internal_conflicts": raw_scan["exact_scope_conflicts"],
            "mapped_cue_scopes_excluded_by_config": raw_scan[
                "excluded_mapped_cue_scopes"
            ],
            "chexpert_duplicate_subject_study_keys": chexpert[
                "duplicate_subject_study_keys"
            ],
            "selected_studies_without_chexpert_row": len(manifest["study_meta"])
            - chexpert["linked_study_rows"],
            "normal_study_to_multiple_image_is_error": False,
            **source_availability,
            "missing_image_files": sum(row["missing_image_files"] for row in bbox_rows),
        }
    )
    resource_inventory = _resource_inventory(args, manifest, chexpert, raw_scan)

    inventory_rows = inventory.rows(exported_findings, mapped_roles)
    _write_json(args.output_dir / "resource_inventory.json", resource_inventory)
    _write_json(
        args.output_dir / "fact_schema.json",
        {
            "fields": FACT_SCHEMA,
            "states": sorted(
                {
                    "positive",
                    "explicit_negative",
                    "uncertain",
                    "unmentioned",
                    "conflict",
                    "unavailable",
                }
            ),
        },
    )
    _write_csv(args.output_dir / "raw_concept_inventory.csv", inventory_rows)
    _write_csv(args.output_dir / "candidate_chain_feasibility.csv", candidate_rows)
    _write_csv(args.output_dir / "l2_pattern_l3_contingency.csv", l2_pattern_rows)
    _write_csv(
        args.output_dir / "native_relationship_inventory.csv",
        native_relationships["rows"],
    )
    _write_csv(args.output_dir / "state_combinations.csv", state_rows)
    _write_csv(args.output_dir / "export_state_audit.csv", export_state_rows)
    _write_csv(args.output_dir / "bbox_geometry.csv", bbox_rows)
    _write_csv(args.output_dir / "semantic_shortcuts.csv", mapping["static_shortcuts"])
    _write_csv(args.output_dir / "review_queue_summary.csv", review_summary)
    _write_csv(args.output_dir / "review_queue_local.csv", review_local)
    _write_csv(args.output_dir / "historical_trace_local.csv", historical_rows)
    _write_json(args.output_dir / "join_quality.json", join_quality)
    run_config = {
        "scope": "sample" if args.sample_per_split else "full_train_valid",
        "sample_per_split": args.sample_per_split,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "semantic_splits": ["train", "valid"],
        "test_policy": "structure counts only; labels excluded from semantic statistics",
        "inputs": {
            key: str(value)
            for key, value in vars(args).items()
            if isinstance(value, Path)
        },
        "mapping_rule_version": mapping["rule_version"],
        "code_sha256": _sha256_file(Path(__file__)),
        "mapping_sha256": _sha256_file(args.mapping),
        "git": _git_revision(),
        "counting_unit": "study unless a column explicitly says patient, image, bbox, row, or fact",
    }
    _write_json(args.output_dir / "run_config.json", run_config)
    report = _report(
        args,
        resource_inventory,
        candidate_rows,
        join_quality,
        bbox_rows,
        state_rows,
        mapping,
    )
    (args.output_dir / "audit_report.md").write_text(report, encoding="utf-8")
    summary = {
        "scope": run_config["scope"],
        "output_dir": str(args.output_dir),
        "selected_images": len(manifest["records"]),
        "selected_studies": len(manifest["study_meta"]),
        "candidate_rows": len(candidate_rows),
        "inventory_rows": len(inventory_rows),
        "review_queue_rows": len(review_local),
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit MIMIC-CXR three-level semantic-chain feasibility without RadGraph."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atomic-annotations", type=Path, required=True)
    parser.add_argument("--comparison-relations", type=Path, required=True)
    parser.add_argument("--chexpert", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mimic-split", type=Path, required=True)
    parser.add_argument("--chest-imagenome-zip", type=Path, required=True)
    parser.add_argument("--scene-graph-zip", type=Path, required=True)
    parser.add_argument("--reports-zip", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--review-per-stratum", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
