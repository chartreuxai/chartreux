"""Composed-ownership record and the one shared OSC status recompute.

Design of record: v0.7.9 point 5 (tasks/v0.7.9/session-status-design-final.md
§3.1-§3.3). The OSC record is owned, never accumulated: every status decision
recomputes through the single table below from six inputs — the callback
owner, the error owner, the main-turn state, the deferred owner-outcome
record keyed to the composed epoch, fresh child activity, and the releasing
owner's own outcome when invoked from a release. Sites never emit status
literals directly.

Ownership classes (design §3.1): a DURABLE taking — a main turn starting, a
prompt establishment, a root-turn error, an explicit reset, or session
rotation — advances the composed epoch and INVALIDATES any deferred record.
A TRANSIENT taking — a shell command starting or a callback opening — takes
the record for its lifetime and SAVES the deferred record (masked, not
discarded). Release precedence: a callback dismissal restores the saved
record (a dismissal carries no outcome of its own); a shell settle REPLACES
the saved record with the shell's own settlement outcome — the most recent
owner's terminal record wins — and the error-kind releasing outcome takes
the record as a persistent error owner.

The composed epoch is the deferral clock, never ``ProgramStatusEmitter``
``generation``: the generation advances on every emit call, including
deduplicated and guard-suppressed ones, so an owner that re-emits the same
composed state must not orphan a pending deferred outcome. The deferred
record is keyed to the composed epoch; a durable taking invalidates it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Owner-outcome states the deferred record can carry (rule 4 of the design's
# table). Errors persist through the error owner (rule 2), never as a
# deferred record.
_IDLE = "idle"

# Releasing-outcome states beyond the deferred pair: the error-kind release
# takes the record as the persistent rule-2 owner.
_ERROR = "error"


@dataclass(frozen=True)
class CallbackOwner:
    """Rule-1 owner: a presented question or permission prompt.

    A callback whose origin maps to no OSC kind never owned ``blocked``; the
    gathering site keeps such callbacks out (kind ``None``), matching the
    encoder's kind vocabulary.
    """

    kind: str
    message: str | None = None


@dataclass(frozen=True)
class DeferredRecord:
    """The saved owner-outcome record, keyed to the composed epoch."""

    state: str
    epoch: int


@dataclass(frozen=True)
class ReleaseOutcome:
    """A releasing owner's own terminal outcome (the sixth compose input)."""

    state: str
    message: str | None = None


@dataclass(frozen=True)
class ComposedDecision:
    """One composed status; state/kind/message map onto the OSC report."""

    state: str
    kind: str | None = None
    message: str | None = None


def compose_status(
    *,
    callback_owner: CallbackOwner | None,
    error_owner_message: str | None,
    turn_active: bool,
    child_active: bool,
    deferred: DeferredRecord | None,
    composed_epoch: int,
    releasing_outcome: ReleaseOutcome | None = None,
) -> ComposedDecision:
    """The one table-faithful recompute (design §3.1 rules 1-4).

    Rule 1: an open question or permission prompt owns ``blocked`` with its
    kind and message; background activity does not overwrite it. Rule 2: the
    current root error owns ``error`` and persists until a later owner
    replaces it. Rule 3: an active main turn — shell-job activity included —
    or an active background agent owns ``working``. Rule 4: the releasing
    owner's own outcome, then the saved deferred record (its epoch must be
    the composed epoch), then the ordinary prompt's ``idle``.
    """
    if callback_owner is not None:
        return ComposedDecision(
            "blocked", kind=callback_owner.kind, message=callback_owner.message
        )
    if error_owner_message is not None:
        return ComposedDecision("error", message=error_owner_message)
    if turn_active or child_active:
        return ComposedDecision("working")
    if releasing_outcome is not None:
        return ComposedDecision(
            releasing_outcome.state, message=releasing_outcome.message
        )
    if deferred is not None and deferred.epoch == composed_epoch:
        return ComposedDecision(deferred.state)
    return ComposedDecision(_IDLE)


class ProgramStatusComposition:
    """Composed-ownership state held beside the record-epoch system.

    ``epoch`` is the composed-ownership epoch (the deferral clock). It is
    distinct from the app's record epoch (error-token fencing) and from the
    emitter's generation: neither of those advances here.
    """

    def __init__(self) -> None:
        self.epoch = 0
        self._deferred: DeferredRecord | None = None
        self._error_message: str | None = None

    @property
    def deferred(self) -> DeferredRecord | None:
        """The saved owner-outcome record, or None while nothing defers."""
        return self._deferred

    @property
    def error_owner_message(self) -> str | None:
        """The persisted rule-2 error message, or None when no error owns."""
        return self._error_message

    def take(self, *, error: str | None = None) -> None:
        """DURABLE taking: advance the composed epoch and invalidate any
        deferred record.

        The error owner passes to the new owner: ``error`` names a root-turn
        error taking the record; None replaces any persisted error with the
        new owner (a valid turn start, a reset, a rotation).
        """
        self.epoch += 1
        self._deferred = None
        self._error_message = error

    def begin_transient(self) -> None:
        """TRANSIENT taking: save, do not discard, the deferred record.

        Nothing advances at the taking: the transient owner holds the record
        for its lifetime and release precedence restores or replaces the
        saved record when it settles.
        """

    def note_release(self, outcome: ReleaseOutcome | None) -> None:
        """Apply release precedence for a settling or releasing site.

        A dismissal (None) carries no outcome: the saved record stands for
        the restore. A done/idle outcome REPLACES the saved record — the most
        recent owner's terminal record wins — keyed to the composed epoch.
        An error-kind release keeps its message as the persistent rule-2
        owner instead of a deferred record: it survives child activity and
        child updates until a later owner replaces it.
        """
        if outcome is None:
            return
        if outcome.state == _ERROR:
            self._deferred = None
            self._error_message = outcome.message
            return
        self._deferred = DeferredRecord(state=outcome.state, epoch=self.epoch)

    def decision(
        self,
        *,
        callback_owner: CallbackOwner | None = None,
        turn_active: bool = False,
        child_active: bool = False,
        releasing_outcome: ReleaseOutcome | None = None,
    ) -> ComposedDecision:
        """Recompute through the shared table from current ownership."""
        return compose_status(
            callback_owner=callback_owner,
            error_owner_message=self._error_message,
            turn_active=turn_active,
            child_active=child_active,
            deferred=self._deferred,
            composed_epoch=self.epoch,
            releasing_outcome=releasing_outcome,
        )


__all__ = [
    "CallbackOwner",
    "ComposedDecision",
    "DeferredRecord",
    "ProgramStatusComposition",
    "ReleaseOutcome",
    "compose_status",
]
