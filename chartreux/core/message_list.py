from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import threading
from typing import overload

from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.transcript_archive import ArchivedPrefixV1


class MessageList(Sequence[LLMMessage]):
    def __init__(self, initial: list[LLMMessage] | None = None) -> None:
        self._data: list[LLMMessage] = list(initial) if initial else []
        self._reset_hooks: list[Callable[[], None]] = []
        self._lock = threading.RLock()
        # M1 archive ownership: the epoch changes on list reset/rebind only
        # (never on cursor invalidation — that lives in the SessionLogger and
        # does not touch this list); the descriptor is the live archive state
        # this list owns, cleared alongside every rebind.
        self._message_epoch = 0
        self._archive_descriptor: ArchivedPrefixV1 | None = None

    @property
    def message_epoch(self) -> int:
        """The current message epoch; bumped by reset/rebind only."""
        return self._message_epoch

    @property
    def archive_descriptor(self) -> ArchivedPrefixV1 | None:
        """The archive descriptor this list currently owns, if any."""
        return self._archive_descriptor

    def set_archive_descriptor(self, descriptor: ArchivedPrefixV1) -> None:
        """Take ownership of ``descriptor`` for the current epoch.

        A descriptor from an older epoch is stale (its prefix belonged to a
        list that was reset/rebind since) and is rejected fail-closed.
        """
        with self._lock:
            if descriptor.message_epoch != self._message_epoch:
                raise ValueError(
                    f"archive descriptor epoch {descriptor.message_epoch} does not "
                    f"match the list epoch {self._message_epoch}"
                )
            self._archive_descriptor = descriptor

    def clear_archive_descriptor(self) -> None:
        """Drop the owned archive descriptor without changing the epoch."""
        with self._lock:
            self._archive_descriptor = None

    def publish_archive_trim(
        self, descriptor: ArchivedPrefixV1, replacements: Sequence[LLMMessage]
    ) -> None:
        """Publish sealed-prefix replacements same-slot, then take the descriptor.

        M1 §4 step 6 list side: exactly ``descriptor.archived_count``
        replacements are applied by absolute index over the non-system
        sequence — no slot is inserted or deleted and system messages are
        preserved — and the owned archive descriptor is set in the same lock
        scope. No ``reset()`` is called and no reset hook fires.
        """
        with self._lock:
            # Validate everything BEFORE mutating: a refused publication
            # leaves the list and the owned descriptor untouched.
            if descriptor.message_epoch != self._message_epoch:
                raise ValueError(
                    f"archive descriptor epoch {descriptor.message_epoch} does not "
                    f"match the list epoch {self._message_epoch}"
                )
            non_system_indexes = [
                index
                for index, message in enumerate(self._data)
                if message.role is not Role.system
            ]
            if descriptor.archived_count > len(non_system_indexes):
                raise ValueError(
                    f"descriptor archives {descriptor.archived_count} records but "
                    f"the list holds only {len(non_system_indexes)} non-system "
                    "messages"
                )
            if len(replacements) != descriptor.archived_count:
                raise ValueError(
                    f"trim publication replaces exactly one slot per archived "
                    f"record: got {len(replacements)} replacements for "
                    f"{descriptor.archived_count} sealed slots"
                )
            sealed_indexes = non_system_indexes[: descriptor.archived_count]
            for slot, replacement in zip(sealed_indexes, replacements, strict=True):
                self._data[slot] = replacement
            self._archive_descriptor = descriptor

    def append(self, msg: LLMMessage) -> None:
        self._data.append(msg)

    def insert(self, i: int, msg: LLMMessage) -> None:
        self._data.insert(i, msg)

    def replace(self, i: int, msg: LLMMessage) -> None:
        """Replace the message at ``i`` in place (PI1/T designed replacement).

        Used by the transcript recorder to swap a synthetic TOOL_NO_RESPONSE
        placeholder for the genuine result that supersedes it — same length,
        same position, no reset hooks fired.
        """
        self._data[i] = msg

    def extend(self, msgs: list[LLMMessage]) -> None:
        for msg in msgs:
            self.append(msg)

    def on_reset(self, hook: Callable[[], None]) -> None:
        """Register a callback that fires whenever the list is reset."""
        self._reset_hooks.append(hook)

    def reset(self, new: list[LLMMessage]) -> None:
        """Replace contents and fire reset hooks (e.g. RewindManager).

        A reset/rebind starts a new message epoch and drops any owned archive
        descriptor: the replacement list is complete, so no archived prefix
        belongs to it.
        """
        with self._lock:
            self._data = list(new)
            self._message_epoch += 1
            self._archive_descriptor = None
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
            # Resume rebind is a list rebind: new epoch, owned archive state
            # cleared (the rebound list holds complete messages).
            self._message_epoch += 1
            self._archive_descriptor = None
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
