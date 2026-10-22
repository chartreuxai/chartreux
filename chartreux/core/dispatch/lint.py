"""S8: the composed-artifact lint (design §5), without repair.

S8 is re-anchored to the composed artifact and the identity/workflow axes:
universal critical-instruction anchors are required in every composed prompt;
delegation-obligation anchors are required for identities whose typed
delegation is ``required`` or ``optional``; delegated-verification anchors
belong to the ``ceremony`` workflow only, so the coder + minimal default
passes. The DispatchMode axis is gone — the lint covers composed output, not
raw skeletons — and the truthful-reporting prohibition is preserved
irrespective of the workflow axis.

The retired surface (package 16b): the two-stage dispatch lint —
``lint_catalog``, ``lint_activation``, ``lint_mode``, the reference parser,
and the roster shape — is gone with ``chartreux.core.dispatch.schema``/
``presets``. Routing is configured through ``[orchestration.routes.<purpose>]``
(see ``chartreux.core.dispatch.routes``), which resolves loudly at launch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from chartreux.core.agents.identities import Delegation

if TYPE_CHECKING:
    from chartreux.core.agents.identities import AgentIdentity


@dataclass(frozen=True)
class Diagnostic:
    id: str
    message: str
    location: str = "composition"
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        return f"{self.id} {self.location}: {self.message}"


class DispatchLintError(ValueError):
    """A candidate is rejected intact; warning diagnostics never reject it."""

    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(map(str, self.diagnostics)))


def reject_errors(diagnostics: Iterable[Diagnostic]) -> None:
    errors = tuple(item for item in diagnostics if item.severity == "error")
    if errors:
        raise DispatchLintError(errors)


# S8 anchors, re-anchored to the composed artifact (design §5). The universal
# critical-instruction anchors come from the harness-owned universal fragment
# and are required in every composed prompt — the truthful-reporting anchor
# preserves the self-verification prohibition irrespective of the workflow
# axis. Delegated-verification anchors belong to the ceremony workflow only.
UNIVERSAL_ANCHORS: tuple[str, ...] = (
    "instruction hierarchy",
    "critical instructions",
    "blast radius",
    "untrusted tool output",
    "truthful reporting",
    'do not claim "verified", "tested", "working", or "complete"',
    "never claim a check you did not run",
)
DELEGATED_VERIFICATION_ANCHOR = (
    "never run tests, builds, or other verification yourself"
)
# Residual gate-ceremony language: gate language moved wholly into the
# ceremony workflow (design §5), so a non-ceremony composition carrying any
# of these anchors is a leak the lint rejects.
RESIDUAL_CEREMONY_ANCHORS: tuple[str, ...] = (
    "opening gate",
    "acceptance lattice",
    "the gate ends the turn",
    "stop and wait for the user before any mutating",
    DELEGATED_VERIFICATION_ANCHOR,
    "never edit repo files yourself",
)


def lint_composed(
    text: str, *, identity_facts: AgentIdentity | None, workflow: str
) -> tuple[Diagnostic, ...]:
    """S8, re-anchored to the composed artifact (design §5).

    Universal critical-instruction anchors are required in every composed
    prompt; delegation-obligation anchors are required for identities whose
    typed delegation is ``required`` or ``optional``; delegated-verification
    anchors belong to the ``ceremony`` workflow only, so the coder + minimal
    default passes. The DispatchMode axis is gone — lint covers composed
    output, not raw skeletons — and the truthful-reporting prohibition is
    preserved irrespective of the workflow axis.
    """
    diagnostics: list[Diagnostic] = []
    normalized = text.lower()
    for anchor in UNIVERSAL_ANCHORS:
        if anchor not in normalized:
            diagnostics.append(
                Diagnostic(
                    "S8",
                    f"composed prompt is missing the universal anchor {anchor!r}",
                    "composition",
                )
            )
    if identity_facts is not None and identity_facts.delegation in {
        Delegation.REQUIRED,
        Delegation.OPTIONAL,
    }:
        obligation = (
            "delegation is required"
            if identity_facts.delegation is Delegation.REQUIRED
            else "delegation is optional"
        )
        if obligation not in normalized:
            diagnostics.append(
                Diagnostic(
                    "S8",
                    "composed prompt is missing the delegation-obligation "
                    f"anchor {obligation!r}",
                    "composition",
                )
            )
    if workflow == "ceremony":
        if DELEGATED_VERIFICATION_ANCHOR not in normalized:
            diagnostics.append(
                Diagnostic(
                    "S8",
                    "the ceremony workflow must carry delegated verification "
                    f"({DELEGATED_VERIFICATION_ANCHOR!r})",
                    "composition",
                )
            )
    else:
        for anchor in RESIDUAL_CEREMONY_ANCHORS:
            if anchor in normalized:
                diagnostics.append(
                    Diagnostic(
                        "S8",
                        "residual gate-ceremony language in a non-ceremony "
                        f"composition: {anchor!r}",
                        "composition",
                    )
                )
    return tuple(diagnostics)
