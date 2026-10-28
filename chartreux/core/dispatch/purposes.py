"""Developer-owned purpose identifiers, descriptions, and prompt renderings.

The purpose vocabulary survives the dispatch retirement: the composed
artifact renders its "Purpose meanings" table from :data:`SHIPPED_PURPOSES`,
and the purpose routes (``chartreux.core.dispatch.routes``) are keyed on the
same ids. The retired dispatch slot/policy schema (``DispatchPolicy``,
``DispatchSlot``, ``DispatchMode``, the shipped presets) lived in the deleted
``chartreux.core.dispatch.schema``/``presets`` modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

PurposeId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")]

#: The demanding implementation route (P1-c2): the only purpose that accepts
#: the structured difficulty field on a launch config.
DEMANDING_IMPLEMENTATION_PURPOSE = "implementation-demanding-settled"


class PurposeDefinition(BaseModel):
    """Developer-owned rendering for one purpose, including overlay additions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def description_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("purpose description must not be blank")
        return value


PurposeVocabulary = Mapping[PurposeId, PurposeDefinition]

SHIPPED_PURPOSES: PurposeVocabulary = MappingProxyType({
    name: PurposeDefinition(description=description)
    for name, description in {
        "research": "Research, source evaluation, and factual investigation.",
        "search": "Bounded searches, grep, and symbol or reference lookups.",
        "exploration": "Targeted code investigation and project exploration.",
        "verification": "Run proportionate tests, builds, and other verification; report only checks actually run.",
        "mechanical-edit": "Mechanical single-file edits with a known, bounded transformation.",
        "implementation": "All substantive implementation, including novel algorithmic reasoning, difficult refactoring, and broad-impact work.",
        DEMANDING_IMPLEMENTATION_PURPOSE: (
            "Exceptionally demanding, genuinely tricky work (novel algorithms, "
            "subtle invariants, concurrency) whose approach is already "
            "settled; ordinary substantive implementation goes to the default "
            "implementor. A settled or detailed approach alone never justifies "
            "this route; a merely large package is split at planning time so "
            "the default implementor can take the pieces."
        ),
        "design-analysis": "Architecture, cross-subsystem design, design refinements, and destructive-operation analysis; read-only advice, never implementation. Keep an advisor across related refinements.",
        "planning-analysis": "Analyze an approved design into bounded steps, dependencies, and acceptance checks; read-only advice, never implementation.",
        "review.quick": "A quick independent judgment in a fresh reviewer context, never the author or a reused advisor.",
        "review.standard": "Independent review against the approved design, plan, and acceptance checks; use a fresh reviewer, never the author. Do not give second-round reviewers earlier conclusions.",
        "review.deep": "Deep independent review composes the configured review seats (analytical, peer, execution, plus user-configured supplementary seats) as fresh contexts on distinct models; do not substitute a seat's model because of the implementor. Substitution (a fresh reviewer from a different model line than the material author) applies only to quick single-seat reviews. Never reuse an advisor as a reviewer or give second-round reviewers earlier conclusions; fail closed when a required independent seat is unavailable.",
    }.items()
})
