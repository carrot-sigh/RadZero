"""Source-preserving fact and state primitives used by the audit CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence


NORMALIZED_STATES = {
    "positive",
    "explicit_negative",
    "uncertain",
    "unmentioned",
    "conflict",
    "unavailable",
}


def normalize_state(source: str, raw_state: object, *, available: bool = True) -> str:
    """Normalize only source-defined assertions; missing never becomes negative."""
    if not available:
        return "unavailable"
    if source == "chexpert":
        if raw_state in (1, 1.0, "1", "1.0"):
            return "positive"
        if raw_state in (0, 0.0, "0", "0.0"):
            return "explicit_negative"
        if raw_state in (-1, -1.0, "-1", "-1.0"):
            return "uncertain"
        if raw_state is None or str(raw_state).strip() == "":
            return "unmentioned"
    elif source in {
        "chest_imagenome_attribute",
        "chest_imagenome_cue",
        "chest_imagenome_export",
    }:
        value = str(raw_state).strip().lower()
        if value in {"yes", "1", "1.0", "positive", "present"}:
            return "positive"
        if value in {"no", "0", "0.0", "negative", "absent"}:
            return "explicit_negative"
        if value in {"uncertain", "u", "maybe"}:
            return "uncertain"
        if raw_state is None or value == "":
            return "unmentioned"
    raise ValueError(f"Unsupported {source!r} state: {raw_state!r}")


def resolve_same_source_states(states: Iterable[str]) -> str:
    """Resolve one concept at one source/scope without discarding conflicts."""
    values = set(states)
    unknown = values - NORMALIZED_STATES
    if unknown:
        raise ValueError(f"Unknown normalized states: {sorted(unknown)}")
    explicit = values & {"positive", "explicit_negative"}
    if len(explicit) == 2 or "conflict" in values:
        return "conflict"
    if explicit:
        return next(iter(explicit))
    if "uncertain" in values:
        return "uncertain"
    if "unmentioned" in values:
        return "unmentioned"
    return "unavailable"


@dataclass(frozen=True)
class Fact:
    source: str
    source_version: str
    annotation_quality: str
    source_record_locator: str
    subject_id: str | None
    study_id: str | None
    image_id: str | None
    report_id: str | None
    split: str | None
    annotation_granularity: str
    raw_category: str
    raw_concept: str
    raw_state: str
    normalized_concept: str | None
    normalized_state: str
    anatomy: str | None = None
    evidence_index: str | None = None
    mapping_rule_id: str | None = None
    derived: bool = False
    provenance: tuple[str, ...] = ()
    linkage_status: str = "linked"

    def __post_init__(self) -> None:
        if self.normalized_state not in NORMALIZED_STATES:
            raise ValueError(f"Invalid normalized state: {self.normalized_state}")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["provenance"] = list(self.provenance)
        return value


def derive_distribution_fact(
    facts: Sequence[Fact],
    *,
    concept: str,
    rule_id: str,
    required_anatomies: Sequence[str],
) -> Fact | None:
    """Derive a positive distribution only from explicit positive locations.

    Missing contralateral evidence is never interpreted as negative. The result
    retains every input locator and is always marked as derived.
    """
    required = set(required_anatomies)
    positives = {
        fact.anatomy
        for fact in facts
        if fact.normalized_state == "positive" and fact.anatomy in required
    }
    if positives != required:
        return None
    first = facts[0]
    return Fact(
        source="derived_distribution",
        source_version=rule_id,
        annotation_quality="rule_derived_from_report_attributes",
        source_record_locator=";".join(
            sorted({f.source_record_locator for f in facts})
        ),
        subject_id=first.subject_id,
        study_id=first.study_id,
        image_id=first.image_id,
        report_id=first.report_id,
        split=first.split,
        annotation_granularity="study",
        raw_category="distribution_rule",
        raw_concept=concept,
        raw_state="positive",
        normalized_concept=concept,
        normalized_state="positive",
        mapping_rule_id=rule_id,
        derived=True,
        provenance=tuple(sorted({f.source_record_locator for f in facts})),
        linkage_status=first.linkage_status,
    )


def detect_cross_source_disagreement(facts: Sequence[Fact]) -> bool:
    """Return true for opposite explicit assertions from distinct sources."""
    by_source: dict[str, set[str]] = {}
    for fact in facts:
        if fact.normalized_state in {"positive", "explicit_negative"}:
            by_source.setdefault(fact.source, set()).add(fact.normalized_state)
    sources = list(by_source)
    return any(
        left != right
        for i, source in enumerate(sources)
        for other in sources[i + 1 :]
        for left in by_source[source]
        for right in by_source[other]
    )


def detect_cross_level_reuse(
    facts_by_level: Mapping[str, Sequence[Fact]],
) -> list[dict[str, str]]:
    """Identify one source fact or rule being reused across semantic levels."""
    seen: dict[tuple[str, str, str | None], str] = {}
    issues: list[dict[str, str]] = []
    for level, facts in facts_by_level.items():
        for fact in facts:
            key = (fact.source, fact.source_record_locator, fact.mapping_rule_id)
            previous = seen.setdefault(key, level)
            if previous != level:
                issues.append(
                    {
                        "source": fact.source,
                        "source_record_locator": fact.source_record_locator,
                        "mapping_rule_id": fact.mapping_rule_id or "",
                        "first_level": previous,
                        "reused_level": level,
                    }
                )
    return issues


@dataclass
class JoinAudit:
    """Small exact join audit with one-to-many and split-leakage accounting."""

    images: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    study_images: dict[str, set[str]] = field(default_factory=dict)
    study_subjects: dict[str, set[str]] = field(default_factory=dict)
    subject_splits: dict[str, set[str]] = field(default_factory=dict)
    duplicate_image_keys: int = 0
    conflicting_image_keys: int = 0

    def add_image(
        self, image_id: str, study_id: str, subject_id: str, split: str
    ) -> None:
        current = (study_id, subject_id, split)
        previous = self.images.get(image_id)
        if previous is not None:
            self.duplicate_image_keys += 1
            if previous != current:
                self.conflicting_image_keys += 1
            return
        self.images[image_id] = current
        self.study_images.setdefault(study_id, set()).add(image_id)
        self.study_subjects.setdefault(study_id, set()).add(subject_id)
        self.subject_splits.setdefault(subject_id, set()).add(split)

    def summary(self) -> dict[str, int]:
        return {
            "image_rows": len(self.images),
            "unique_studies": len(self.study_images),
            "unique_subjects": len(self.subject_splits),
            "duplicate_image_keys": self.duplicate_image_keys,
            "conflicting_image_keys": self.conflicting_image_keys,
            "one_to_many_studies": sum(len(v) > 1 for v in self.study_images.values()),
            "unexpected_many_to_many_studies": sum(
                len(v) > 1 for v in self.study_subjects.values()
            ),
            "subjects_crossing_splits": sum(
                len(v) > 1 for v in self.subject_splits.values()
            ),
        }
