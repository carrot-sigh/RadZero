"""Auditable semantic-chain checks and hierarchy exploration utilities."""

from .facts import (
    Fact,
    JoinAudit,
    derive_distribution_fact,
    detect_cross_source_disagreement,
    detect_cross_level_reuse,
    normalize_state,
    resolve_same_source_states,
)

__all__ = [
    "Fact",
    "JoinAudit",
    "derive_distribution_fact",
    "detect_cross_source_disagreement",
    "detect_cross_level_reuse",
    "normalize_state",
    "resolve_same_source_states",
]
