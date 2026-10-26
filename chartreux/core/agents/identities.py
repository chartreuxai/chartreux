from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re


class Delegation(StrEnum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class ToolPosture(StrEnum):
    ALL_GRANTED = "all_granted"
    NON_MUTATING = "non_mutating"


@dataclass(frozen=True)
class AgentIdentity:
    id: str
    delegation: Delegation
    personal_tools: ToolPosture
    delegation_tools: ToolPosture
    safety: str
    display_name: str
    # Reviewed builtin tool names admitted BEYOND the personal posture's class
    # set (settlement decision 3, P1-h): an identity-scoped execution carve-out.
    # The posture table stays untouched — it is shared by every NON_MUTATING
    # selector (researcher, advisor, the orchestrator's personal axis, and every
    # delegation axis), so a table change would widen all of them. The carve-out
    # is personal-axis only (a reviewer cannot delegate its shell) and is
    # enforced against the shipped-builtin identity of the exact selected
    # class, so a foreign custom variant of a carved-out name fails closed.
    execution_carve_out: frozenset[str] = frozenset()


# The read-only identities still DELEGATE their non-mutating/control-plane set
# (their delegation_tools posture), so their declared delegation metadata is
# OPTIONAL — not NONE — matching the executable delegation behavior (design §3:
# no shipped identity sets none in v1).
_IDENTITIES = {
    "coder": AgentIdentity(
        "coder",
        Delegation.OPTIONAL,
        ToolPosture.ALL_GRANTED,
        ToolPosture.ALL_GRANTED,
        "full",
        "Coder",
    ),
    "orchestrator": AgentIdentity(
        "orchestrator",
        Delegation.REQUIRED,
        ToolPosture.NON_MUTATING,
        ToolPosture.ALL_GRANTED,
        "restricted",
        "Orchestrator",
    ),
    "researcher": AgentIdentity(
        "researcher",
        Delegation.OPTIONAL,
        ToolPosture.NON_MUTATING,
        ToolPosture.NON_MUTATING,
        "read-only",
        "Researcher",
    ),
    "reviewer": AgentIdentity(
        "reviewer",
        Delegation.OPTIONAL,
        ToolPosture.NON_MUTATING,
        ToolPosture.NON_MUTATING,
        "read-only",
        "Reviewer",
        # Exit gates verify (settlement decision 3): the reviewer may run
        # non-mutating verification commands through bash itself. Only the
        # bash surface is carved out — bash_start is job management, not
        # verification, and the write tools stay excluded — and bash's own
        # permission guards remain the mutation boundary.
        execution_carve_out=frozenset({"bash"}),
    ),
    "advisor": AgentIdentity(
        "advisor",
        Delegation.OPTIONAL,
        ToolPosture.NON_MUTATING,
        ToolPosture.NON_MUTATING,
        "read-only",
        "Advisor",
    ),
}


def get_builtin_identity(name: str) -> AgentIdentity | None:
    return _IDENTITIES.get(name)


def builtin_identities() -> tuple[AgentIdentity, ...]:
    return tuple(_IDENTITIES.values())


def builtin_identity_names() -> tuple[str, ...]:
    return tuple(sorted(_IDENTITIES))


class UnknownIdentityError(ValueError):
    """An ``[agent]`` identity selection that resolves to nothing known."""


# One resource-loading contract across prompts, workflows, and strategies
# (design §6): a custom identity name is a plain prompt file stem — the same
# id grammar the strict ResourceLoader enforces.
_CUSTOM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _custom_identity_resource_error(
    reference: str, name: str, user_prompts_dirs: Sequence[Path]
) -> UnknownIdentityError:
    dirs_hint = ", ".join(str(directory) for directory in user_prompts_dirs) or (
        "<no user prompts directories>"
    )
    return UnknownIdentityError(
        f"Unknown custom identity {reference!r}: no identity prompt "
        f"'{name}.md' exists in the user prompts directories ({dirs_hint}). "
        f"Create the prompt file, or select a built-in identity "
        f"({', '.join(builtin_identity_names())})."
    )


def validate_identity_selection(
    reference: str, *, user_prompts_dirs: Sequence[Path] = ()
) -> str:
    """Validate an identity selection, failing closed on unknown references.

    Design criterion 2: an identity that is neither a builtin, nor
    ``builtin:``-qualified-and-existing, nor a resolvable user resource is an
    actionable error — never an open default (S2 unknown-identity fail-open: a
    typo'd ``builtin:reviever`` or unknown bare name silently yielded an
    unrestricted posture).

    ``user:``-qualified selections are validated HERE, at loop construction,
    not deferred to prompt assembly (S2 round 2: unvalidated user:-qualified
    identities): the custom name must be a well-formed prompt file stem (no
    empty, traversal, or multi-colon forms), it must not reuse a reserved
    shipped name (reserved names cannot shadow shipped definitions), and its
    ``<name>.md`` prompt resource must exist in the user prompts directories.

    Returns the validated reference. Valid custom identities keep decision-8
    behavior (no harness-known posture): a ``user:``-qualified selection is an
    explicit custom qualification whose prompt file is the whole contract, and
    a bare non-builtin name must resolve to a ``<name>.md`` prompt in a user
    prompts directory.
    """
    builtin_hint = ", ".join(builtin_identity_names())
    if not isinstance(reference, str) or not reference:
        raise UnknownIdentityError(
            "Invalid [agent] identity: the selection must be non-empty text. "
            f"Built-in identities: {builtin_hint}."
        )
    if reference.startswith("builtin:"):
        name = reference.split(":", 1)[1]
        if get_builtin_identity(name) is None:
            raise UnknownIdentityError(
                f"Unknown built-in identity {reference!r}. "
                f"Available built-in identities: {builtin_hint}."
            )
        return reference
    if reference.startswith("user:"):
        # An explicitly qualified custom identity (design §4): the prompt
        # file is the whole contract, so it must exist — and the name must
        # be a plain file stem — at loop construction, not at deferred
        # prompt assembly (S2 round 2: unvalidated user:-qualified
        # identities).
        name = reference.split(":", 1)[1]
        if not name or not _CUSTOM_ID.fullmatch(name) or name in {".", ".."}:
            raise UnknownIdentityError(
                f"Malformed custom identity {reference!r}: the name after "
                "'user:' must be a plain prompt file name (letters, digits, "
                "'_' and '-', starting with a letter or digit) — no empty, "
                "traversal, or extra ':' forms."
            )
        if get_builtin_identity(name) is not None:
            # Reserved names cannot shadow shipped definitions (design §4;
            # package-1 acceptance): a user prompt may not reuse a built-in
            # identity name.
            raise UnknownIdentityError(
                f"Reserved identity name {reference!r}: {name!r} is a shipped "
                f"built-in identity; a custom prompt cannot shadow built-in "
                f"identities ({builtin_hint}). Rename the prompt file."
            )
        if not any(
            (directory / f"{name}.md").is_file() for directory in user_prompts_dirs
        ):
            raise _custom_identity_resource_error(reference, name, user_prompts_dirs)
        return reference
    if get_builtin_identity(reference) is not None:
        return reference
    # A bare non-builtin name is a custom identity reference: the SAME id
    # grammar as the ``user:``-qualified branch applies BEFORE the resource
    # existence check — a plain prompt file stem (no traversal, nested, or
    # multi-segment forms) that does not reuse a reserved shipped name — so
    # a matching file cannot launder a malformed reference through loop
    # construction (the ResourceLoader rejects these forms later, at prompt
    # assembly; S2 round 3 W4: the bare branch accepted any reference whose
    # file existed).
    if not _CUSTOM_ID.fullmatch(reference) or reference in {".", ".."}:
        raise UnknownIdentityError(
            f"Malformed custom identity {reference!r}: a bare custom identity "
            "must be a plain prompt file name (letters, digits, '_' and '-', "
            "starting with a letter or digit) — no empty, traversal, nested, "
            "or multi-segment forms."
        )
    if get_builtin_identity(reference) is not None:
        # Reserved names cannot shadow shipped definitions (design §4;
        # package-1 acceptance) — mirrors the ``user:`` branch. The builtin
        # check above already returned for shipped names, so this guards the
        # grammar contract itself.
        raise UnknownIdentityError(
            f"Reserved identity name {reference!r}: {reference!r} is a shipped "
            f"built-in identity; a custom prompt cannot shadow built-in "
            f"identities ({builtin_hint}). Rename the prompt file."
        )
    # The reference must resolve to a user prompt resource, otherwise the
    # selection is a typo and fails.
    if any(
        (directory / f"{reference}.md").is_file() for directory in user_prompts_dirs
    ):
        return reference
    dirs_hint = ", ".join(str(directory) for directory in user_prompts_dirs) or (
        "<no user prompts directories>"
    )
    raise UnknownIdentityError(
        f"Unknown identity {reference!r}: it is not a built-in identity "
        f"({builtin_hint}), and no custom identity prompt '{reference}.md' "
        f"exists in the user prompts directories ({dirs_hint}). Create the "
        f"prompt file, or select a built-in identity."
    )
