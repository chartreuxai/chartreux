"""Developer-owned purpose identifiers and prompt renderings."""

from __future__ import annotations

from types import MappingProxyType

from chartreux.core.dispatch.schema import PurposeDefinition, PurposeVocabulary

SHIPPED_PURPOSES: PurposeVocabulary = MappingProxyType({
    name: PurposeDefinition(description=description)
    for name, description in {
        "search": "Bounded searches, grep, and symbol or reference lookups.",
        "exploration": "Targeted code investigation and project exploration.",
        "verification": "Run proportionate tests, builds, and other verification; report only checks actually run.",
        "mechanical-edit": "Mechanical single-file edits with a known, bounded transformation.",
        "implementation": "All substantive implementation, including novel algorithmic reasoning, difficult refactoring, and broad-impact work.",
        "implementation-demanding-settled": "Demanding execution with a settled approach; a proactive escalation-implementor route, not only a failure-triggered route. State the reason for this route.",
        "design-analysis": "Architecture, cross-subsystem design, design refinements, and destructive-operation analysis; read-only advice, never implementation. Keep an advisor across related refinements.",
        "planning-analysis": "Analyze an approved design into bounded steps, dependencies, and acceptance checks; read-only advice, never implementation.",
        "review.quick": "A quick independent judgment in a fresh reviewer context, never the author or a reused advisor.",
        "review.standard": "Independent review against the approved design, plan, and acceptance checks; use a fresh reviewer, never the author. Do not give second-round reviewers earlier conclusions.",
        "review.deep": "Deep independent review through the configured composition of fresh reviewer contexts, with authorship-aware substitution. Never reuse an advisor as a reviewer or give second-round reviewers earlier conclusions; fail closed when a required independent seat is unavailable.",
    }.items()
})
