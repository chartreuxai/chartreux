"""Two-stage dispatch validation, without repair or profile-discovery recursion.

Curated prose uses ``slot `name` `` (or ``slots `a`, `b` ``),
``profile `name` ``, ``purpose `name` `` and ``@role`` references. Review
rows start with review purpose identifiers followed by a colon; failure rows
use ``class: target`` or a Markdown table. Legacy task examples are also read.
These parsers are lint-only: they are not a runtime routing authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Literal

from chartreux.core.dispatch.purposes import SHIPPED_PURPOSES
from chartreux.core.dispatch.schema import DispatchMode, DispatchPolicy

if TYPE_CHECKING:
    from chartreux.core.agents.models import AgentProfile
    from chartreux.core.model_catalog.resolver import ResolvedModel
    from chartreux.core.model_catalog.schema import ModelCatalog


@dataclass(frozen=True)
class Diagnostic:
    id: str
    message: str
    location: str = "dispatch"
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


@dataclass(frozen=True)
class References:
    slots: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    purposes: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()


_NAME = r"[a-zA-Z0-9_.-]+"
_MIN_TABLE_CELLS = 2
_MIN_CONTRAST_SLOTS = 2


def _named(text: str, kind: str) -> tuple[str, ...]:
    # A plural reference consumes only a contiguous quoted-name list, not the
    # rest of the sentence (which can contain unrelated inline code).
    pattern = rf"\b{kind}s?\s+(`{_NAME}`(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)`{_NAME}`)*)"
    return tuple(
        name
        for group in re.findall(pattern, text)
        for name in re.findall(rf"`({_NAME})`", group)
    )


def parse_references(text: str) -> References:
    """Extract explicit references, retaining duplicate seats for S6."""
    profiles = list(_named(text, "profile"))
    profiles.extend(re.findall(r'agent_type\s*=\s*["\']([^"\']+)["\']', text))
    # The compatibility block refers to the three profiles in a quoted list.
    profiles.extend(
        name
        for group in re.findall(
            rf"(`{_NAME}`(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)`{_NAME}`)*)\s+agent profiles?",
            text,
        )
        for name in re.findall(rf"`({_NAME})`", group)
    )
    purposes = list(_named(text, "purpose"))
    purposes.extend(
        name.rstrip(".") for name in re.findall(r"\breview\.[a-z][a-z0-9.-]*", text)
    )
    return References(
        (*_named(text, "slot"), *re.findall(rf"[Ii]f `({_NAME})` authored", text)),
        tuple(profiles),
        tuple(purposes),
        tuple(
            name
            for span in re.findall(r"`([^`]+)`", text)
            for name in re.findall(r"(?<![\w@])@([a-zA-Z0-9_.-]+)", span)
        ),
    )


def _blocks(policy: DispatchPolicy) -> Iterable[tuple[str, str]]:
    for name in ("instructions", "failure_routing", "compositions", "contrasts"):
        yield name, getattr(policy, name)


def lint_mode(mode: object) -> tuple[Diagnostic, ...]:
    if mode not in tuple(DispatchMode):
        return (Diagnostic("S9", "mode must be standalone or orchestrated"),)
    return ()


def _failure_rows(text: str) -> Iterable[tuple[str, str]]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if (
                len(cells) >= _MIN_TABLE_CELLS
                and cells[0].lower() not in {"class", "failure", "failure class"}
                and not re.fullmatch(r"[-: ]+", cells[0])
            ):
                yield cells[0], " | ".join(cells[1:])
        elif match := re.match(r"(?:[-*]\s+)?([\w.-]+):\s*(.*)", line):
            yield match[1], match[2]


def lint_catalog(
    policy: DispatchPolicy, catalog: ModelCatalog
) -> tuple[Diagnostic, ...]:
    """Stage 1: no registry. Unreferenced draft roles need not be runnable."""
    # Local imports avoid the loader/resolver cycle; no registry is constructed.
    from chartreux.core.model_catalog.loader import CatalogSnapshot
    from chartreux.core.model_catalog.resolver import (
        ModelResolutionError,
        ModelResolver,
    )

    diagnostics = list(lint_mode(policy.mode))
    resolver = ModelResolver(CatalogSnapshot(catalog, "dispatch-lint"))
    for name, slot in policy.slots.items():
        try:
            resolver.resolve(slot.role)
        except ModelResolutionError as exc:
            diagnostics.append(
                Diagnostic(
                    "S1",
                    str(exc),
                    f"slots.{name}.role",
                    "warning" if slot.role[1:] in catalog.roles else "error",
                )
            )
        if slot.implements == "never" and any(
            purpose.startswith("implementation") or purpose == "mechanical-edit"
            for purpose in slot.purposes
        ):
            diagnostics.append(
                Diagnostic(
                    "S4",
                    "implementation requires an implementing slot",
                    f"slots.{name}",
                )
            )
    for purpose in sorted(set(SHIPPED_PURPOSES) - set(policy.vocabulary)):
        diagnostics.append(
            Diagnostic(
                "S3", f"shipped purpose {purpose!r} cannot be removed", "vocabulary"
            )
        )
    for block, text in _blocks(policy):
        refs = parse_references(text)
        for kind, names, known in (
            ("slot", refs.slots, policy.slots),
            ("purpose", refs.purposes, policy.vocabulary),
            ("role", refs.roles, catalog.roles),
        ):
            for name in dict.fromkeys(names):
                # @role is a documented expression placeholder, not a binding.
                if kind == "role" and name == "role":
                    continue
                if name not in known:
                    diagnostics.append(
                        Diagnostic("R1", f"unknown {kind} {name!r}", block)
                    )
    classes: set[str] = set()
    for failure, target in _failure_rows(policy.failure_routing):
        refs = parse_references(target)
        if failure in classes:
            diagnostics.append(
                Diagnostic(
                    "S7", f"duplicate failure class {failure!r}", "failure_routing"
                )
            )
        classes.add(failure)
        if (
            not (refs.slots or refs.roles or re.search(r"\buser\b", target))
            or any(name not in policy.slots for name in refs.slots)
            or any(name not in catalog.roles for name in refs.roles)
        ):
            diagnostics.append(
                Diagnostic(
                    "S7",
                    f"unresolved failure target for {failure!r}",
                    "failure_routing",
                )
            )
    diagnostics.extend(_lint_compositions(policy))
    return tuple(diagnostics)


def _contrast_slots(policy: DispatchPolicy) -> set[str]:
    slots = set(parse_references(policy.contrasts).slots)
    # Preserve the verbatim legacy examples: a worker @scout override and a
    # worker default (implementation) represent distinct named launch seats.
    for match in re.finditer(
        r"task\(agent_type=\"([^\"]+)\"(.*?)(?=\. `|`\.|$)", policy.contrasts
    ):
        profile, body = match.groups()
        roles = re.findall(r'"@([a-zA-Z0-9_.-]+)"', body)
        for name, slot in policy.slots.items():
            if slot.profile == profile and (
                slot.role[1:] in roles if roles else "implementation" in slot.purposes
            ):
                slots.add(name)
    return slots


def lint_activation(
    policy: DispatchPolicy,
    profiles: Mapping[str, AgentProfile],
    *,
    shadowed_profiles: Iterable[str] = (),
) -> tuple[Diagnostic, ...]:
    """Stage 2: consume an already-discovered registry, never discover it here."""
    diagnostics: list[Diagnostic] = []
    for name in sorted(set(shadowed_profiles) & {"worker", "advisor", "reviewer"}):
        diagnostics.append(Diagnostic("S2", f"builtin profile {name!r} is shadowed"))
    for name, slot in policy.slots.items():
        if slot.profile not in profiles:
            diagnostics.append(
                Diagnostic(
                    "R1", f"unknown profile {slot.profile!r}", f"slots.{name}.profile"
                )
            )
    diagnostics.extend(_lint_escalation(policy))
    for block, text in _blocks(policy):
        refs = parse_references(text)
        for kind, names, known in (
            ("profile", refs.profiles, profiles),
            ("slot", refs.slots, policy.slots),
        ):
            for name in dict.fromkeys(names):
                if name not in known:
                    diagnostics.append(
                        Diagnostic("R1", f"unknown {kind} {name!r}", block)
                    )
    diagnostics.extend(_lint_compositions(policy, profiles))
    if policy.contrasts.strip() and len(_contrast_slots(policy)) < _MIN_CONTRAST_SLOTS:
        diagnostics.append(
            Diagnostic(
                "R2", "contrasts require at least two distinct slots", "contrasts"
            )
        )
    for clause in re.split(r"[;\n]|(?<=[.!?])\s+", policy.contrasts):
        refs = parse_references(clause)
        for name in refs.slots:
            slot = policy.slots.get(name)
            if slot is not None:
                for purpose in set(refs.purposes) - set(slot.purposes):
                    diagnostics.append(
                        Diagnostic(
                            "R3",
                            f"purpose {purpose!r} is not routed by slot {name!r}",
                            "contrasts",
                            "warning",
                        )
                    )
    return tuple(diagnostics)


def _lint_escalation(policy: DispatchPolicy) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    reason_required = (
        r"\b(?:state|provide|give|require|requires|requiring)\b[^.!?\n]*\breason\b"
    )
    for name, slot in policy.slots.items():
        if slot.implements != "escalation":
            continue
        for purpose in slot.purposes:
            if not re.search(
                reason_required, policy.vocabulary[purpose].description, re.I
            ):
                diagnostics.append(
                    Diagnostic(
                        "S5",
                        f"route {purpose!r} must require a reason",
                        f"slots.{name}",
                    )
                )
        for block, text in _blocks(policy):
            for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
                if name in parse_references(sentence).slots and not re.search(
                    reason_required, sentence, re.I
                ):
                    diagnostics.append(
                        Diagnostic(
                            "S5", f"route to {name!r} must require a reason", block
                        )
                    )
    return tuple(diagnostics)


def _lint_compositions(
    policy: DispatchPolicy, profiles: Mapping[str, AgentProfile] | None = None
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    if not re.search(r"(?m)^\s*review\.[\w.-]+", policy.compositions):
        diagnostics.append(
            Diagnostic("S6", "review compositions must be nonempty", "compositions")
        )
    for line in policy.compositions.splitlines():
        if not re.match(r"\s*review\.[\w.-]+", line):
            continue
        route = line.split(":", 1)[-1]
        seats = parse_references(re.split(r"\bIf\b", route, flags=re.I)[0]).slots
        if not seats or len(seats) != len(set(seats)):
            diagnostics.append(
                Diagnostic(
                    "S6",
                    "review composition must have nonempty, distinct seats",
                    "compositions",
                )
            )
        substitutions = re.findall(
            rf"[Ii]f `({_NAME})` authored.*?substitute(?:_if_author)? slot `({_NAME})`",
            route,
        )
        all_seats = (*seats, *(name for pair in substitutions for name in pair))
        for name in all_seats:
            slot = policy.slots.get(name)
            if (
                slot is None
                or not slot.review_eligible
                or (profiles is not None and slot.profile not in profiles)
            ):
                diagnostics.append(
                    Diagnostic(
                        "S6",
                        f"review seat {name!r} is unavailable or ineligible",
                        "compositions",
                    )
                )
        for original, substitute in substitutions:
            if original == substitute or (original in seats and substitute in seats):
                diagnostics.append(
                    Diagnostic(
                        "S6",
                        "authorship substitution must be a different, unoccupied seat",
                        "compositions",
                    )
                )
    return tuple(diagnostics)


def lint_rendered(text: str, mode: DispatchMode | str) -> tuple[Diagnostic, ...]:
    """WP4's reversion tripwire operates on rendered output, not templates."""
    diagnostics = list(lint_mode(mode))
    normalized = text.lower()
    verification = (
        "never run tests yourself" in normalized
        or "never run tests, builds, or other verification yourself" in normalized
    )
    honesty = (
        "never claim a check you did not run" in normalized
        or 'do not claim "verified", "tested", "working", or "complete" unless a corresponding execution step appears in the trajectory and you read its output.'
        in normalized
    )
    if not verification or not honesty:
        diagnostics.append(
            Diagnostic(
                "S8",
                "rendered output must prohibit self-verification and dishonest check claims",
            )
        )
    edit_prohibition = "never edit repo files yourself" in normalized
    if edit_prohibition != (mode == DispatchMode.ORCHESTRATED):
        diagnostics.append(
            Diagnostic(
                "S8", "rendered edit prohibition does not match the selected mode"
            )
        )
    return tuple(diagnostics)


@dataclass(frozen=True)
class RosterShape:
    bindings: frozenset[tuple[str, str | None]]
    slot_bindings: Mapping[str, tuple[str, str | None]] = field(default_factory=dict)
    failures: Mapping[str, str] = field(default_factory=dict)

    @property
    def single_model(self) -> bool:
        return len({model for model, _ in self.bindings}) == 1


def roster_shape(bindings: Iterable[ResolvedModel]) -> RosterShape:
    """Provider deployments never manufacture canonical model diversity."""
    return RosterShape(
        frozenset(
            (item.base_model, item.thinking or item.definition.thinking)
            for item in bindings
        )
    )
