"""Explore label hierarchies without treating them as clinical reasoning chains."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .facts import normalize_state, resolve_same_source_states


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_manifest(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    image_split: dict[str, str] = {}
    study_subject: dict[str, str] = {}
    study_images: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row["split"])
            if split not in {"train", "valid"}:
                continue
            image = str(row["dicom_id"])
            study = str(row["study_id"])
            image_split[image] = split
            subject = str(row.get("patient_id") or row.get("subject_id") or "")
            if not subject:
                parts = Path(str(row.get("image", ""))).parts
                subject = next(
                    (part[1:] for part in parts if part.startswith("p") and len(part) == 9 and part[1:].isdigit()),
                    "",
                )
            study_subject.setdefault(study, subject)
            study_images[study].add(image)
    return image_split, study_subject, study_images


def _read_contract(path: Path) -> tuple[list[str], list[str]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    return contract["anatomy"]["labels"], contract["finding"]["labels"]


def _read_existing_rows(path: Path, keys: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if keys is None or row.get("candidate_chain") in keys]


def _augment_dimension_rows(path: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Reuse the prior raw inventory for cue/state coverage without rescanning cues."""
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["source"] not in {"chest_imagenome_cue", "chexpert"}:
                continue
            rows.append(
                {
                    "split": raw["split"],
                    "category": raw["source"] + ":" + raw["raw_category"],
                    "concept": raw["raw_concept"],
                    "study_count": raw["study_count"],
                    "positive_studies": raw["study_count"] if raw["normalized_state"] == "positive" else "0",
                    "explicit_negative_studies": raw["study_count"] if raw["normalized_state"] == "explicit_negative" else "0",
                    "uncertain_studies": raw["study_count"] if raw["normalized_state"] == "uncertain" else "0",
                    "role_assessment": "source-state inventory; cue is auxiliary/proposed, CheXpert is study/report weak supervision",
                }
            )
    return rows


def _scan_atomic(
    path: Path,
    image_split: dict[str, str],
    study_subject: dict[str, str],
    findings: set[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Build study/anatomy/finding states and a small raw dimension inventory."""
    scope_states: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    category_concepts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    category_states: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    rows = 0
    linked = 0
    parquet = pq.ParquetFile(path)
    columns = ["dicom_id", "study_id", "subject_id", "region", "category", "raw_label", "relation"]
    for batch in parquet.iter_batches(batch_size=65536, columns=columns):
        data = batch.to_pydict()
        for i, value in enumerate(data["dicom_id"]):
            rows += 1
            image = str(value)
            split = image_split.get(image)
            if split is None:
                continue
            linked += 1
            study = str(data["study_id"][i])
            anatomy = str(data["region"][i])
            category = str(data["category"][i])
            concept = str(data["raw_label"][i])
            try:
                state = normalize_state("chest_imagenome_attribute", data["relation"][i])
            except ValueError:
                state = "unavailable"
            category_concepts[(split, category, concept)].add(study)
            category_states[(split, category, concept, state)].add(study)
            if category == "anatomicalfinding" and concept in findings:
                scope_states[(split, study, anatomy, concept)].add(state)

    dimension_rows: list[dict[str, str]] = []
    for (split, category, concept), studies in sorted(category_concepts.items()):
        states = {
            state: len(category_states.get((split, category, concept, state), set()))
            for state in ("positive", "explicit_negative", "uncertain", "unmentioned", "unavailable")
        }
        role = {
            "anatomicalfinding": "regional finding or classification node",
            "disease": "diagnostic/report interpretation candidate; leakage risk",
            "device": "device category or spatial object candidate",
            "technicalassessment": "technical-quality or acquisition attribute",
            "texture": "direct texture attribute; sparse in this release",
            "severity": "attribute, not a semantic stage",
            "temporal": "attribute/comparison context, not an outcome",
            "laterality": "attribute/location qualifier",
            "nlp": "extraction/status metadata",
            "tubesandlines": "device finding",
        }.get(category, "raw attribute category")
        dimension_rows.append(
            {
                "split": split,
                "category": category,
                "concept": concept,
                "study_count": str(len(studies)),
                "positive_studies": str(states["positive"]),
                "explicit_negative_studies": str(states["explicit_negative"]),
                "uncertain_studies": str(states["uncertain"]),
                "role_assessment": role,
            }
        )
    return {
        "rows": rows,
        "linked_rows": linked,
        "scope_states": scope_states,
        "dimension_rows": dimension_rows,
    }, dimension_rows


def _state_summary(states: set[str]) -> str:
    if not states:
        return "unmentioned"
    return resolve_same_source_states(states)


def _anatomy_structure(
    scan: dict[str, Any], study_subject: dict[str, str], finding_count: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Evaluate a derived anatomy -> abnormality -> anatomy-finding structure."""
    scopes = scan["scope_states"]
    rows: list[dict[str, str]] = []
    pair_rows: list[dict[str, str]] = []
    by_study_anatomy: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for (split, study, anatomy, finding), states in scopes.items():
        state = _state_summary(states)
        by_study_anatomy[(split, study, anatomy)][finding] = state
        pair_rows.append(
            {
                "structure": "anatomy_to_finding",
                "split": split,
                "parent": anatomy,
                "child": finding,
                "edge_type": "located_at/anatomy association",
                "state": state,
                "derived": "false",
                "provenance": "Chest ImaGenome anatomicalfinding attribute",
            }
        )
    grouped: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"scopes": set(), "studies": set()}
    )
    for (split, study, anatomy), values in by_study_anatomy.items():
        child_states = list(values.values())
        if "positive" in child_states:
            parent = "positive"
        elif len(child_states) == finding_count and all(
            state == "explicit_negative" for state in child_states
        ):
            parent = "explicit_negative"
        elif any(s in {"explicit_negative", "uncertain", "conflict"} for s in child_states):
            parent = "partial_or_unknown"
        else:
            parent = "unmentioned"
        grouped[(split, parent)]["scopes"].add(f"{study}|{anatomy}")
        grouped[(split, parent)]["studies"].add(study)
    for (split, state), values in sorted(grouped.items()):
        rows.append(
            {
                "structure": "anatomy_taxonomy",
                "split": split,
                "node": "anatomy_has_any_configured_finding",
                "node_level": "derived_parent",
                "state": state,
                "scope_count": str(len(values["scopes"])),
                "study_count": str(len(values["studies"])),
                "patient_count": str(len({study_subject.get(study, "") for study in values["studies"] if study_subject.get(study)})),
                "direct_or_derived": "derived",
                "rule": "positive if any explicit-positive anatomy×finding child; negative only if the complete child vocabulary is explicitly negative; otherwise unknown",
            }
        )
    return rows, pair_rows


def _candidate_rows(existing: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = {"pulmonary_edema", "atelectasis", "pneumonia", "pneumothorax", "cardiomegaly"}
    output = []
    for row in existing:
        if row["candidate_chain"] not in keep:
            continue
        chain = row["candidate_chain"]
        output.append(
            {
                "structure": "regional_finding_to_report_target",
                "candidate_chain": chain,
                "split": row["split"],
                "l1_valid_studies": row["l1_valid_studies"],
                "l2_valid_studies": row["l2_valid_studies"],
                "l3_valid_studies": row["l3_valid_studies"],
                "l1_l2_valid_studies": row["l1_l2_valid_studies"],
                "l2_l3_valid_studies": row["l2_l3_valid_studies"],
                "all_three_valid_studies": row["all_three_valid_studies"],
                "all_three_valid_patients": row["all_three_valid_patients"],
                "derived_or_direct": "mixed; L1 cue may be derived/proposed, L2 regional attribute, L3 CheXpert weak supervision",
                "relationship_status": "anatomy association and report co-occurrence only; no support edge",
                "semantic_status": row["conclusion_grade"],
            }
        )
    return output


def _controlled_examples(scan: dict[str, Any], dimension_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for (split, study, anatomy, finding), states in sorted(scan["scope_states"].items()):
        state = _state_summary(states)
        key = (finding, state)
        if split != "train" or key in seen or state not in {"positive", "explicit_negative", "conflict"}:
            continue
        seen.add(key)
        examples.append(
            {
                "example_type": "direct_anatomy_finding",
                "split": split,
                "anonymous_anatomy": anatomy,
                "raw_attribute": finding,
                "raw_state": state,
                "hierarchy_node": f"({anatomy}, {finding})",
                "derived": "false",
                "interpretation": "direct regional leaf; no report-level diagnosis inferred",
            }
        )
        if len(examples) >= 8:
            break
    for raw in dimension_rows:
        if raw["split"] != "train" or raw["concept"] not in {"alveolar", "lucency", "pneumonia", "Cardiomegaly"}:
            continue
        examples.append(
            {
                "example_type": "source_dimension_boundary",
                "split": raw["split"],
                "anonymous_anatomy": "<study/report or phrase scope>",
                "raw_attribute": raw["category"] + ":" + raw["concept"],
                "raw_state": "see dimension inventory",
                "hierarchy_node": "proposed cue or report target",
                "derived": "cue=proposed; CheXpert=direct weak target",
                "interpretation": "not an independent semantic edge; target-name and same-report leakage must be reviewed",
            }
        )
        if len(examples) >= 14:
            break
    return examples


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    image_split, study_subject, study_images = _read_manifest(args.manifest)
    anatomies, findings = _read_contract(args.contract)
    scan, dimension_rows = _scan_atomic(args.atomic, image_split, study_subject, set(findings))
    dimension_rows = _augment_dimension_rows(args.raw_inventory, dimension_rows)
    existing = _read_existing_rows(args.candidate_csv)
    anatomy_nodes, anatomy_edges = _anatomy_structure(scan, study_subject, len(findings))
    candidate_rows = _candidate_rows(existing)

    dimensions = {
        "anatomy_objects": {"source": "scene_graph.objects", "count": 36, "direct": True, "recommended_role": "spatial locator; not an abnormality stage"},
        "anatomicalfinding": {"source": "scene_graph.attributes / atomic parquet", "count": len(findings), "direct": True, "recommended_role": "regional finding nodes and anatomy×finding leaves"},
        "disease": {"source": "scene_graph.attributes / atomic parquet", "direct": True, "recommended_role": "report-derived interpretation candidate; target leakage review required"},
        "device": {"source": "scene_graph.attributes", "direct": True, "recommended_role": "device taxonomy or spatial object"},
        "technicalassessment": {"source": "scene_graph.attributes", "direct": True, "recommended_role": "quality/acquisition attribute"},
        "texture_cues": {"source": "scene_graph attribute.texture_cues[phrase_ID]", "direct": False, "recommended_role": "phrase-aligned proposed evidence only; positive-only in configured mappings"},
        "severity_cues": {"source": "scene_graph attribute.severity_cues[phrase_ID]", "direct": False, "recommended_role": "modifier attribute, not a semantic stage"},
        "temporal_cues": {"source": "scene_graph attribute.temporal_cues[phrase_ID]", "direct": False, "recommended_role": "modifier/comparison context"},
        "sections_phrases": {"source": "scene_graph attribute.sections/phrases", "direct": True, "recommended_role": "evidence provenance and controlled review; not a gold target"},
        "comparison": {"source": "scene_graph.relationships / comparison parquet", "direct": True, "recommended_role": "temporal comparison edge; not diagnostic support"},
        "CheXpert": {"source": "MIMIC-CXR-JPG CheXpert CSV", "direct": True, "recommended_role": "study/report-level weak target; preserve uncertain/unmentioned"},
    }
    structure_config = {
        "schema_version": 1,
        "rule_version": "hierarchy-exploration-v1",
        "policy": {"discovery_split": "train", "evaluation_split": "valid", "test_semantics_forbidden": True, "missing_is_negative": False, "sources_are_not_merged": True},
        "structures": [
            {"id": "anatomy_taxonomy", "kind": "derived_classification", "path": ["anatomy object", "anatomy_has_any_configured_finding", "anatomy × finding"], "edges": ["anatomical_part_of", "located_at", "derived_aggregation"], "semantic_stage": False},
            {"id": "regional_finding_report_target", "kind": "weakly supervised regional-to-report DAG", "path": ["anatomy/location", "regional finding", "CheXpert report target"], "edges": ["located_at", "report_cooccurrence", "hypothesized_support"], "semantic_stage": "candidate only"},
            {"id": "descriptive_pattern_interpretation", "kind": "proposed semantic DAG", "path": ["phrase-aligned cue", "named imaging pattern", "report interpretation"], "edges": ["report_cooccurrence", "hypothesized_support"], "semantic_stage": "not established"},
            {"id": "comparison_context", "kind": "temporal/context graph", "path": ["current finding", "comparison relation", "prior/current context"], "edges": ["comparison"], "semantic_stage": False},
        ],
    }
    _write_json(args.output / "annotation_dimension_inventory.json", {"dimensions": dimensions, "counts": {"manifest_images": len(image_split), "manifest_studies": len(study_images), "atomic_rows": scan["rows"], "atomic_rows_linked": scan["linked_rows"]}})
    _write_csv(args.output / "annotation_dimension_values.csv", dimension_rows)
    _write_json(args.output / "hierarchy_structures.v1.json", structure_config)
    _write_csv(args.output / "hierarchy_node_inventory.csv", anatomy_nodes)
    _write_csv(args.output / "hierarchy_edge_inventory.csv", anatomy_edges)
    _write_csv(args.output / "hierarchy_candidate_coverage.csv", candidate_rows)
    _write_csv(args.output / "hierarchy_controlled_examples.csv", _controlled_examples(scan, dimension_rows))
    comparison_rows = [
        {"structure": "anatomy_taxonomy", "layer_meaning": "location and classification", "direct_or_derived": "direct leaves + derived parent", "coverage": "see hierarchy_node_inventory.csv", "synonym_risk": "low for taxonomy; not a clinical stage", "relation": "anatomy association", "intervention": "change child finding changes derived parent by rule", "original_three_stage_goal": "no"},
        {"structure": "regional_finding_report_target", "layer_meaning": "location → regional finding → report target", "direct_or_derived": "mixed", "coverage": "candidate coverage CSV", "synonym_risk": "high for same-name targets; moderate for others", "relation": "co-occurrence only", "intervention": "regional correction can change downstream consistency but target is not causally determined", "original_three_stage_goal": "conditional/no"},
        {"structure": "descriptive_pattern_interpretation", "layer_meaning": "phrase cue → pattern → interpretation", "direct_or_derived": "cue derived/proposed; pattern direct; target weak", "coverage": "pneumonia/edema candidates only", "synonym_risk": "high for alveolar/interstitial and diagnosis phrases", "relation": "no native support edge", "intervention": "requires independent morphology review", "original_three_stage_goal": "not established"},
        {"structure": "comparison_context", "layer_meaning": "temporal comparison context", "direct_or_derived": "direct relation", "coverage": "comparison inventory", "synonym_risk": "not applicable", "relation": "comparison only", "intervention": "changes temporal context, not diagnostic truth", "original_three_stage_goal": "no"},
    ]
    _write_csv(args.output / "hierarchy_structure_comparison.csv", comparison_rows)
    report = [
        "# 现有 MIMIC-CXR / Chest ImaGenome 层级化标签结构探索",
        "",
        "本报告是 train 发现、validation 固定方案评估的结构审计，不训练模型、不生成 gold labels，不把分类层级解释为临床因果链。",
        "",
        "## 结论",
        "",
        "1. 最稳妥、最接近现有标注的结构是 anatomy object → anatomy×finding，多父节点的空间定位/分类 DAG。叶节点直接来自 Chest ImaGenome anatomicalfinding；父节点只能规则派生，缺失不转阴性。",
        "2. anatomy/location → regional finding → CheXpert report target 可以作为弱监督 DAG，但边只有 located_at 和 report co-occurrence，不能称为完整证据→解释关系。",
        "3. phrase cue → pattern → interpretation 只有 proposed 节点监督。interstitial/alveolar/lucency 是 phrase-aligned 辅助 cue，positive-only 且非特异，需人工/影像验证。",
        "4. comparison relation 适合作为时间上下文边，不适合作为诊断解释层。",
        "",
        "## 标注维度",
        "",
        "原始 scene graph 实际包含 objects、attributes、phrase_IDs、phrases、sections、texture_cues、severity_cues、temporal_cues 和 relationships。当前导出保留 36 anatomy bbox、41 anatomical finding 和稀疏 CheXpert global，丢失大量 phrase/cue/关系来源信息。",
        "",
        "- anatomy objects：直接空间定位；不是异常层。",
        "- anatomicalfinding：直接区域 finding，适合作为 anatomy×finding 叶节点；没有报告级完整性。",
        "- disease：直接报告派生诊断词，适合作为候选解释节点，但有目标泄漏。",
        "- device：直接设备类别，可做设备 taxonomy 或位置对象；当前没有充分的设备末端位置评价层。",
        "- technicalassessment：技术/质量属性，不宜作为疾病语义层。",
        "- texture/severity/temporal cues：phrase 对齐辅助 cue 或修饰信息；不是自动 gold evidence。",
        "- comparison：improved/no change/worsened 的对象比较；不是 support/explanation 边。",
        "- CheXpert：study/report-level weak target；1/0/-1/blank 分别保留为 positive/explicit negative/uncertain/unmentioned。",
        "",
        "## 候选结构",
        "",
        "### 1. anatomy taxonomy（推荐）",
        "`anatomy object → anatomy_has_any_configured_finding → (anatomy, finding)`。这是分类/空间结构，不是三阶段临床推理。叶节点是直接属性；父节点 derived=true。positive 需要至少一个子节点显式 positive；negative 只有完整子词表全部显式 negative 才允许；否则 unknown/partial。",
        "",
        "### 2. regional finding → report target（条件推荐）",
        "`anatomy/location → regional finding → CheXpert target`。可用于多任务一致性和表示干预的研究假设，但不能把 CheXpert 当作区域 finding 的独立临床真值，也不能自动融合正负状态。",
        "",
        "### 3. descriptive cue → pattern → interpretation（暂不作为正式层级）",
        "Pneumonia 的 alveolar cue 与 airspace opacity/consolidation/infiltration 有节点覆盖，但 cue 非特异且同 phrase 可能含诊断词；Edema 和 Pneumothorax 同样存在 cue 泄漏和正例-only 问题。当前只能作为 proposed DAG，需人工补标独立形态证据。",
        "",
        "### 4. comparison context",
        "`current finding → comparison relation → prior/current context` 只表达时间比较上下文；不能推导总体诊断进展，也不能作为 L3。",
        "",
        "## 干预含义",
        "",
        "- anatomy taxonomy：修正一个 anatomy×finding 叶节点会按规则改变父节点；这是定义性派生，不是医学因果干预。",
        "- regional finding/report target：修正区域 finding 可改变与报告目标的 consistency 分析，但不能强制改变 CheXpert target。",
        "- cue/pattern/interpretation：修改 cue 只有在新增人工形态标签后才有明确监督意义；现有 cue 只能作为待审查变量。",
        "",
        "## 边界",
        "",
        "现有数据能整理出多个标签层，但只有 anatomy 分类结构是较稳定的直接/派生体系；没有任何候选被验证为三个独立语义阶段。缺失的独立形态证据、影像模式定义和层间支持关系不能由映射脚本制造。",
        "",
        "详细覆盖、节点、边和运行配置见同目录 CSV/JSON。test 未用于结构发现或语义覆盖。",
    ]
    (args.output / "hierarchy_exploration_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_json(args.output / "run_config.json", {"scope": "train_valid", "seed": args.seed, "inputs": {"manifest": str(args.manifest), "atomic": str(args.atomic), "contract": str(args.contract), "candidate_csv": str(args.candidate_csv)}, "test_policy": "test excluded from semantic discovery and coverage"})
    return {"output": str(args.output), "atomic_rows": scan["rows"], "linked_atomic_rows": scan["linked_rows"], "dimension_rows": len(dimension_rows), "candidate_rows": len(candidate_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore hierarchy structures in existing MIMIC-CXR/Chest ImaGenome labels.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atomic", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--raw-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    random.seed(args.seed)
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
