from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field


@dataclass(slots=True)
class EntryExpansionState:
    """Session-local folding choices for reasoning and result entries."""

    default_collapsed: bool = True
    _collapsed: dict[str, bool] = field(default_factory=dict)
    revision: int = 0
    reset_generation: int = 0
    _change_revisions: list[int] = field(default_factory=list)
    _changed_ids: list[str] = field(default_factory=list)

    def is_collapsed(self, entry_id: str) -> bool:
        return self._collapsed.get(entry_id, self.default_collapsed)

    def register(self, entry_id: str) -> bool:
        return self._collapsed.setdefault(entry_id, self.default_collapsed)

    def set_collapsed(self, entry_id: str, collapsed: bool) -> None:
        if self.is_collapsed(entry_id) != collapsed:
            self.revision += 1
            self._change_revisions.append(self.revision)
            self._changed_ids.append(entry_id)
        self._collapsed[entry_id] = collapsed

    def set_all_collapsed(self, collapsed: bool) -> None:
        changed = [key for key, value in self._collapsed.items() if value != collapsed]
        if self.default_collapsed != collapsed or changed:
            self.revision += 1
            for key in changed:
                self._change_revisions.append(self.revision)
                self._changed_ids.append(key)
        self.default_collapsed = collapsed
        for key in self._collapsed:
            self._collapsed[key] = collapsed

    def changed_ids_since(self, revision: int) -> set[str]:
        return set(self._changed_ids[bisect_right(self._change_revisions, revision) :])

    @property
    def entry_ids(self) -> frozenset[str]:
        return frozenset(self._collapsed)

    def reset(self, *, default_collapsed: bool = True) -> None:
        self.default_collapsed = default_collapsed
        self._collapsed.clear()
        self._change_revisions.clear()
        self._changed_ids.clear()
        self.revision += 1
        self.reset_generation += 1
