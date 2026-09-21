from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import threading
from typing import overload

from chartreux.core.llm_models import LLMMessage, Role


class MessageList(Sequence[LLMMessage]):
    def __init__(self, initial: list[LLMMessage] | None = None) -> None:
        self._data: list[LLMMessage] = list(initial) if initial else []
        self._reset_hooks: list[Callable[[], None]] = []
        self._lock = threading.RLock()

    def append(self, msg: LLMMessage) -> None:
        self._data.append(msg)

    def insert(self, i: int, msg: LLMMessage) -> None:
        self._data.insert(i, msg)

    def extend(self, msgs: list[LLMMessage]) -> None:
        for msg in msgs:
            self.append(msg)

    def on_reset(self, hook: Callable[[], None]) -> None:
        """Register a callback that fires whenever the list is reset."""
        self._reset_hooks.append(hook)

    def reset(self, new: list[LLMMessage]) -> None:
        """Replace contents and fire reset hooks (e.g. RewindManager)."""
        with self._lock:
            self._data = list(new)
            for hook in self._reset_hooks:
                hook()

    def reset_preserving_system(self, tail: list[LLMMessage]) -> None:
        """Atomically keep existing system messages and replace the rest.

        Held under the same lock as ``update_system_prompt`` so a concurrent
        deferred-init thread cannot interleave its system-prompt insert with
        this read-then-replace and corrupt the list.
        """
        with self._lock:
            system = [m for m in self._data if m.role is Role.system]
            self._data = [*system, *tail]
            for hook in self._reset_hooks:
                hook()

    def update_system_prompt(self, new: str) -> None:
        """Replace the system prompt, or insert it if none exists yet.

        Under deferred init the prompt can land after messages were already
        appended, so insert at the front rather than clobber slot 0.
        """
        msg = LLMMessage(role=Role.system, content=new)
        with self._lock:
            if self._data and self._data[0].role == Role.system:
                self._data[0] = msg
            else:
                self._data.insert(0, msg)

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> LLMMessage: ...
    @overload
    def __getitem__(self, index: slice) -> list[LLMMessage]: ...
    def __getitem__(self, index: int | slice) -> LLMMessage | list[LLMMessage]:
        return self._data[index]

    def __iter__(self) -> Iterator[LLMMessage]:
        # Snapshot under the lock: only __iter__ hands element-by-element
        # control back to Python, so the deferred-init thread's insert(0)
        # could shift indices mid-iteration. Other read dunders are single
        # C-level ops and stay atomic under the GIL without locking.
        with self._lock:
            return iter(list(self._data))

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __bool__(self) -> bool:
        return bool(self._data)
