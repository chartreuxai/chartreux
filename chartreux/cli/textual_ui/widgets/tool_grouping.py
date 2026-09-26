"""Pure policy for grouping tool activity in the conversation timeline.

The live event handler and history restoration must use this module rather than
encoding independent grouping rules.  Widgets only consume the resulting keys,
indicator values, and expansion state; this module deliberately has no Textual
dependency.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto

from chartreux.app_server.models import (
    CancelledEffectState,
    CompletedEffectState,
    EffectState,
    FailedEffectState,
    HookNoticeDetail,
    PublicEffectEntry,
    PublicHistoryEntry,
    PublicNoticeEntry,
    PublicReasoningEntry,
    SkippedEffectState,
)
from chartreux.utils.tool_presentation import ToolEffectKind


class GroupIndicator(StrEnum):
    """Widget-independent state for a settled tool-group indicator."""

    SUCCESS = auto()
    ERROR = auto()
    MUTED = auto()


@dataclass(frozen=True, slots=True)
class ToolGroupKey:
    """Stable identity of a group, anchored to its first creating entry."""

    first_entry_id: str


ManualCommandClassifier = Callable[[PublicEffectEntry], bool]

_NON_GROUPED_EFFECT_KINDS = frozenset({
    ToolEffectKind.FILE_EDIT,
    ToolEffectKind.FILE_WRITE,
})


def is_manual_shell_entry(entry: PublicEffectEntry) -> bool:
    """Return whether *entry* is a user-issued ``!<command>`` shell effect.

    Manual commands use the dedicated ``shell`` tool name.  Agent shell calls
    retain their backend-provided names (such as ``bash``), even though both
    share the SHELL effect kind.
    """
    return entry.detail.tool_name == "shell"


def entry_keeps_tool_group(
    entry: PublicHistoryEntry,
    *,
    manual_command_classifier: ManualCommandClassifier = is_manual_shell_entry,
) -> bool:
    """Whether *entry* belongs to a run of grouped tool activity.

    Effects, reasoning, and hook notices continue a group.  All message
    content (including user and assistant content) and other timeline entries
    are boundaries.  Manual commands and file edit/write effects are deliberately
    standalone entries.
    """
    if isinstance(entry, PublicEffectEntry):
        return (
            entry.detail.kind not in _NON_GROUPED_EFFECT_KINDS
            and not manual_command_classifier(entry)
        )
    return isinstance(entry, PublicReasoningEntry) or (
        isinstance(entry, PublicNoticeEntry)
        and isinstance(entry.detail, HookNoticeDetail)
    )


def starts_tool_group(
    entry: PublicHistoryEntry,
    previous_entry: PublicHistoryEntry | None,
    *,
    manual_command_classifier: ManualCommandClassifier = is_manual_shell_entry,
) -> bool:
    """Whether *entry* starts a group after *previous_entry* in timeline order."""
    return entry_keeps_tool_group(
        entry, manual_command_classifier=manual_command_classifier
    ) and (
        previous_entry is None
        or not entry_keeps_tool_group(
            previous_entry, manual_command_classifier=manual_command_classifier
        )
    )


def tool_group_key(first_entry: PublicHistoryEntry) -> ToolGroupKey:
    """Return the key for the first reasoning/effect entry that creates a group."""
    return ToolGroupKey(first_entry.id)


def effect_state_is_terminal(state: EffectState) -> bool:
    """Whether an effect state can settle a group indicator."""
    return isinstance(
        state,
        CompletedEffectState
        | FailedEffectState
        | CancelledEffectState
        | SkippedEffectState,
    )


def effect_state_to_indicator(state: EffectState) -> GroupIndicator:
    """Map a terminal effect result to the group summary's indicator."""
    if isinstance(state, FailedEffectState):
        return GroupIndicator.ERROR
    if isinstance(state, CompletedEffectState):
        return GroupIndicator.SUCCESS if state.display.success else GroupIndicator.ERROR
    if isinstance(state, SkippedEffectState | CancelledEffectState):
        return GroupIndicator.MUTED
    return GroupIndicator.SUCCESS


def effect_state_is_failure(state: EffectState) -> bool:
    """Whether *state* is an individually recoverable tool failure.

    A completed effect whose display reports ``success=False`` is a failure for
    presentation purposes just like an explicitly failed effect.
    """
    return isinstance(state, FailedEffectState) or (
        isinstance(state, CompletedEffectState) and not state.display.success
    )


@dataclass(slots=True)
class TimelineStatus:
    """Settled outcomes indexed by timeline position for one group.

    Calls may complete out of order in live rendering.  The indicator is always
    taken from the last *timeline* call that has a terminal state, never from a
    worst-outcome reduction of earlier calls.
    """

    _settled: dict[int, GroupIndicator] = field(default_factory=dict)

    def record_effect(self, timeline_index: int, state: EffectState) -> None:
        """Record or replace a call's state at its timeline position."""
        if effect_state_is_terminal(state):
            self._settled[timeline_index] = effect_state_to_indicator(state)
        else:
            self._settled.pop(timeline_index, None)

    def forget_effect(self, timeline_index: int) -> None:
        """Remove an outcome whose timeline entry is no longer retained."""
        self._settled.pop(timeline_index, None)

    @property
    def indicator(self) -> GroupIndicator | None:
        """The latest terminal call's indicator, or ``None`` while none settled."""
        if not self._settled:
            return None
        return self._settled[max(self._settled)]

    def failure_is_muted(self, timeline_index: int) -> bool:
        """Whether a failed call is resolved by a later successful call.

        This is intentionally independent of :attr:`indicator`: the summary is
        the last call's outcome, while an individual failure is muted only when
        a later call in the same group succeeded.
        """
        return self._settled.get(timeline_index) is GroupIndicator.ERROR and any(
            indicator is GroupIndicator.SUCCESS
            for index, indicator in self._settled.items()
            if index > timeline_index
        )


@dataclass(slots=True)
class ToolGroupExpansionState:
    """Presentation-owned collapsed/expanded state, keyed by group identity.

    Live and restored renderers query and update the same store.  The store,
    rather than a transient widget, owns a user's expansion choice.
    """

    default_collapsed: bool = True
    _collapsed: dict[ToolGroupKey, bool] = field(default_factory=dict)
    revision: int = 0
    reset_generation: int = 0
    _change_revisions: list[int] = field(default_factory=list)
    _changed_keys: list[ToolGroupKey] = field(default_factory=list)

    def _record_changes(self, keys: list[ToolGroupKey]) -> None:
        self.revision += 1
        self._change_revisions.extend([self.revision] * len(keys))
        self._changed_keys.extend(keys)

    def changed_keys_since(self, revision: int) -> set[ToolGroupKey]:
        return set(self._changed_keys[bisect_right(self._change_revisions, revision) :])

    def is_collapsed(self, key: ToolGroupKey) -> bool:
        """Return the stored choice, or the configured default for a new group."""
        return self._collapsed.get(key, self.default_collapsed)

    def register(self, key: ToolGroupKey) -> bool:
        return self._collapsed.setdefault(key, self.default_collapsed)

    def set_all_collapsed(self, collapsed: bool) -> None:
        changed = [key for key, value in self._collapsed.items() if value != collapsed]
        if self.default_collapsed != collapsed or changed:
            self._record_changes(changed)
        self.default_collapsed = collapsed
        for key in self._collapsed:
            self._collapsed[key] = collapsed

    @property
    def keys(self) -> frozenset[ToolGroupKey]:
        return frozenset(self._collapsed)

    def reset(self, *, default_collapsed: bool = True) -> None:
        self.default_collapsed = default_collapsed
        self._collapsed.clear()
        self._change_revisions.clear()
        self._changed_keys.clear()
        self.revision += 1
        self.reset_generation += 1

    def set_collapsed(self, key: ToolGroupKey, collapsed: bool) -> None:
        """Persist a group's expansion choice."""
        if self.is_collapsed(key) != collapsed:
            self._record_changes([key])
        self._collapsed[key] = collapsed
