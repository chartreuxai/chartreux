"""Session safety-authority layer (v0.8.0 item A, package A-P0).

One surface: the session holds a single :class:`SafetyLevel` (SAFE | SMART |
DANGEROUS, amendment Q2). Every transition emits an
``authority_transition{from,to,actor,session_id,ts}`` event covering all six
ordered pairs (unguarded-mode-design.md, Audit). Children inherit the
parent's level as an explicit launch snapshot and never run the DANGEROUS
ceremony. Decided calls are remembered by identity so a decided call is never
re-decided. Audit records carry the authorizer tag of the level in force, and
activation is gated on a writable session audit sink (fail closed) at the
app-server seam.

A-P0 hard-codes the SAFE default and touches NEITHER config file: the
``[safety]`` section (incl. ``default_level``) lands wholly in A-P2, which
reads it through the stable :func:`initial_level` seam.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import BaseEvent

if TYPE_CHECKING:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

__all__ = [
    "AUTHORITY_TRANSITION_EVENT",
    "AUTHORITY_TRANSITION_PAIRS",
    "AuthorityActor",
    "AuthorityLaunchSnapshot",
    "AuthorityTransition",
    "DecidedCallIdentity",
    "DecidedCallRecord",
    "DecidedCallStore",
    "SafetyLevel",
    "SessionAuthorityState",
    "initial_level",
]


class SafetyLevel(StrEnum):
    """The session's single authority level (amendment Q2)."""

    SAFE = "safe"
    SMART = "smart"
    DANGEROUS = "dangerous"


class AuthorityActor(StrEnum):
    """Who performed a transition (transcript marker tags, amendment Q6)."""

    HUMAN = "human"
    CLASSIFIER = "classifier"
    DANGEROUS = "dangerous"
    POLICY = "policy"


#: Wire name of the transition event.
AUTHORITY_TRANSITION_EVENT = "authority_transition"

#: All six ordered pairs of distinct levels; every legal transition.
AUTHORITY_TRANSITION_PAIRS: tuple[tuple[SafetyLevel, SafetyLevel], ...] = tuple(
    (from_level, to_level)
    for from_level in SafetyLevel
    for to_level in SafetyLevel
    if from_level != to_level
)


class AuthorityTransition(BaseEvent):
    """``authority_transition{from,to,actor,session_id,ts}`` event.

    ``from``/``to`` are reserved words, so the fields alias them: serialized
    by alias the payload carries exactly ``from`` and ``to``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    from_level: SafetyLevel = Field(alias="from")
    to_level: SafetyLevel = Field(alias="to")
    actor: str
    session_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def event_name(self) -> str:
        return AUTHORITY_TRANSITION_EVENT


@dataclass(frozen=True, slots=True)
class DecidedCallIdentity:
    """Identity of one decided tool call: tool name plus a canonical-args key."""

    tool_name: str
    call_key: str

    @classmethod
    def of(cls, tool_name: str, args: BaseModel | None) -> DecidedCallIdentity:
        canonical = (args.model_dump_json() if args is not None else "").encode("utf-8")
        return cls(tool_name=tool_name, call_key=hashlib.sha256(canonical).hexdigest())

    def encode(self) -> dict[str, str]:
        return {"tool_name": self.tool_name, "call_key": self.call_key}

    @classmethod
    def decode(cls, payload: object) -> DecidedCallIdentity:
        if not isinstance(payload, dict):
            raise ValueError("a decided-call identity payload must be an object")
        tool_name = payload.get("tool_name")
        call_key = payload.get("call_key")
        if not isinstance(tool_name, str) or not isinstance(call_key, str):
            raise ValueError(
                "a decided-call identity payload needs string tool_name/call_key"
            )
        return cls(tool_name=tool_name, call_key=call_key)


@dataclass(frozen=True, slots=True)
class DecidedCallRecord:
    """A decided call and its audit authorizer tag (the level in force)."""

    identity: DecidedCallIdentity
    decision: str
    actor: str
    authorizer: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DecidedCallStore:
    """Decided-call identity memory: a decided call is never re-decided."""

    def __init__(self, records: Iterable[DecidedCallRecord] = ()) -> None:
        self._records: dict[tuple[str, str], DecidedCallRecord] = {
            (record.identity.tool_name, record.identity.call_key): record
            for record in records
        }

    def record(self, record: DecidedCallRecord) -> None:
        self._records[(record.identity.tool_name, record.identity.call_key)] = record

    def lookup(self, identity: DecidedCallIdentity) -> DecidedCallRecord | None:
        return self._records.get((identity.tool_name, identity.call_key))

    def __contains__(self, identity: DecidedCallIdentity) -> bool:
        return (identity.tool_name, identity.call_key) in self._records

    def records(self) -> tuple[DecidedCallRecord, ...]:
        return tuple(self._records.values())


@dataclass(frozen=True, slots=True)
class AuthorityLaunchSnapshot:
    """The explicit child launch snapshot: the parent's level at dispatch.

    Children inherit the level as runtime authority — never fabricated
    approvals — and never run the DANGEROUS-revoking ceremony themselves.
    """

    level: SafetyLevel = SafetyLevel.SAFE
    decided_calls: tuple[DecidedCallIdentity, ...] = ()

    def encode(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "decided_calls": [identity.encode() for identity in self.decided_calls],
        }

    @classmethod
    def decode(cls, payload: object) -> AuthorityLaunchSnapshot | None:
        """Strictly decode a JSON-safe snapshot; None when absent.

        A malformed payload raises (a legacy peer simply sends no key, which
        decodes to None and falls back to the hard-coded SAFE default).
        """
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("an authority launch snapshot payload must be an object")
        level = payload.get("level")
        try:
            level_value = SafetyLevel(level)
        except ValueError as exc:
            raise ValueError(
                f"an authority launch snapshot has an unknown level: {level!r}"
            ) from exc
        raw_calls = payload.get("decided_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError(
                "an authority launch snapshot decided_calls must be a list"
            )
        return cls(
            level=level_value,
            decided_calls=tuple(DecidedCallIdentity.decode(item) for item in raw_calls),
        )


class SessionAuthorityState:
    """Per-session authority state: current level, transition log, decided calls.

    The transition sink receives every emitted :class:`AuthorityTransition`;
    the loop wires it to its out-of-band event queue.
    """

    def __init__(
        self,
        *,
        session_id: str,
        level: SafetyLevel,
        decided_calls: DecidedCallStore | None = None,
        sink: Callable[[AuthorityTransition], None] | None = None,
    ) -> None:
        self._session_id = session_id
        self._level = level
        self._decided_calls = decided_calls or DecidedCallStore()
        self._sink = sink
        self._transitions: list[AuthorityTransition] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def level(self) -> SafetyLevel:
        return self._level

    def is_dangerous(self) -> bool:
        """Live DANGEROUS flag for the A-P1 enforcement-layer context seam."""
        return self._level is SafetyLevel.DANGEROUS

    @property
    def decided_calls(self) -> DecidedCallStore:
        return self._decided_calls

    @property
    def transitions(self) -> tuple[AuthorityTransition, ...]:
        return tuple(self._transitions)

    def transition(self, to: SafetyLevel, *, actor: str) -> AuthorityTransition | None:
        """Move to ``to``, emitting the transition event.

        All six ordered pairs of distinct levels are legal; a same-level
        request is a no-op and emits nothing.
        """
        if to == self._level:
            return None
        # Built via model_validate so the wire aliases ("from"/"to") are the
        # only construction keys — pydantic accepts them by alias here.
        event = AuthorityTransition.model_validate({
            "from": self._level,
            "to": to,
            "actor": actor,
            "session_id": self._session_id,
        })
        self._level = to
        self._transitions.append(event)
        if self._sink is not None:
            self._sink(event)
        return event

    def record_decided_call(
        self, identity: DecidedCallIdentity, *, decision: str, actor: str
    ) -> DecidedCallRecord:
        record = DecidedCallRecord(
            identity=identity,
            decision=decision,
            actor=actor,
            authorizer=self._level.value,
        )
        self._decided_calls.record(record)
        return record

    def launch_snapshot(self) -> AuthorityLaunchSnapshot:
        """Freeze the current authority for a child launch."""
        return AuthorityLaunchSnapshot(
            level=self._level,
            decided_calls=tuple(
                record.identity for record in self._decided_calls.records()
            ),
        )

    def audit_tag(self) -> dict[str, str]:
        """The audit authorizer tag for tool attempt/result records."""
        return {"authorizer": self._level.value}


def initial_level(config: ChartreuxConfigSchema) -> SafetyLevel:
    """Stable read seam for the session's initial :class:`SafetyLevel`.

    A-P0 hard-coded SAFE and read no config key; A-P2 lands the ``[safety]``
    section and reads ``safety.default_level`` through this seam. DANGEROUS is
    ceremony-only and is never an initial level: anything unexpected fails
    closed to SAFE.
    """
    try:
        section = getattr(config, "safety", None)
        default_level = getattr(section, "default_level", None)
    except Exception:
        return SafetyLevel.SAFE
    if default_level == SafetyLevel.SMART.value:
        return SafetyLevel.SMART
    return SafetyLevel.SAFE
