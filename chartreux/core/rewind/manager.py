from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chartreux.core.checkpoints import Checkpointer, FileStore
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.message_list import MessageList
from chartreux.core.session.transcript_archive import (
    ArchivedPrefixV1,
    ArchiveIntegrityError,
    archived_prefix_chunks,
    hydrate_archived_messages,
)


class SaveMessages(Protocol):
    async def __call__(self, *, allow_empty: bool = False) -> None: ...


class RewindError(Exception):
    """Raised when a rewind operation fails."""


class RewindArchiveError(RewindError):
    """The sealed prefix could not be hydrated for a rewind/fork (M1 §3).

    A stale descriptor, an unavailable transcript, or archived bytes that fail
    integrity validation all fail closed: a rewind or fork never hands a
    partial, trimmed, or tombstoned history to model context.
    """


@dataclass(frozen=True, slots=True)
class ArchivedHistorySource:
    """The sealed prefix one rewind/fork operation hydrates from (M1 §3).

    Pairs the descriptor the message list owns with the transcript holding its
    archived bytes. Rare explicit rewind/fork operations may temporarily
    hydrate that prefix; ordinary saves never do.
    """

    descriptor: ArchivedPrefixV1
    transcript_path: Path

    @classmethod
    def for_messages(
        cls, messages: Sequence[LLMMessage], *, transcript_path: Path | None
    ) -> ArchivedHistorySource | None:
        """The source for ``messages``, or None when the list is unarchived.

        Raises:
            RewindArchiveError: The list's descriptor no longer matches its
                epoch, or no transcript is available to hydrate it.
        """
        descriptor = getattr(messages, "archive_descriptor", None)
        if descriptor is None:
            return None
        epoch = getattr(messages, "message_epoch", None)
        if epoch is None or descriptor.message_epoch != epoch:
            raise RewindArchiveError(
                "the archive descriptor no longer matches the message list "
                f"epoch ({descriptor.message_epoch} != {epoch})"
            )
        if transcript_path is None:
            raise RewindArchiveError(
                "the archived prefix cannot be hydrated: this session has no transcript"
            )
        return cls(descriptor=descriptor, transcript_path=transcript_path)

    def archived_messages(self) -> list[LLMMessage]:
        """The archived records, read and validated by the shared reader."""
        try:
            return hydrate_archived_messages(
                self.descriptor,
                b"".join(
                    archived_prefix_chunks(
                        self.transcript_path, self.descriptor.prefix_end_byte
                    )
                ),
            )
        except (ArchiveIntegrityError, OSError) as exc:
            raise RewindArchiveError(
                f"the archived transcript prefix could not be hydrated: {exc}"
            ) from exc


def archived_history_for(
    messages: Sequence[LLMMessage], session_logger: object
) -> ArchivedHistorySource | None:
    """The operation's archive source, resolved from the owning session logger."""
    if getattr(messages, "archive_descriptor", None) is None:
        return None
    transcript_path: Path | None = None
    if getattr(session_logger, "enabled", False) and (
        getattr(session_logger, "session_dir", None) is not None
    ):
        transcript_path = getattr(session_logger, "messages_filepath", None)
    return ArchivedHistorySource.for_messages(messages, transcript_path=transcript_path)


def complete_history(
    messages: Sequence[LLMMessage], archive: ArchivedHistorySource | None
) -> list[LLMMessage]:
    """The complete history of one rewind/fork operation (M1 §3).

    Sealed prefix slots are replaced by their hydrated archive records, so no
    consumer of this snapshot ever sees a trimmed payload or a tombstone, and
    a fresh logger is never handed one. Replacement is length-preserving, so
    absolute indexes — and the rewind targets resolved against them — stay
    valid, and the sealed message objects themselves are never mutated.
    """
    snapshot = list(messages)
    if archive is None:
        return snapshot
    archived = archive.archived_messages()
    slots = [
        index
        for index, message in enumerate(snapshot)
        if message.role is not Role.system
    ]
    if len(slots) < len(archived):
        raise RewindArchiveError(
            f"the archived prefix holds {len(archived)} record(s) but the message "
            f"list has only {len(slots)} non-system message(s)"
        )
    for slot, hydrated in zip(slots[: len(archived)], archived, strict=True):
        current = snapshot[slot]
        if (
            current.message_id is not None
            and hydrated.message_id is not None
            and current.message_id != hydrated.message_id
        ):
            raise RewindArchiveError(
                f"archived record {hydrated.message_id!r} does not match the sealed "
                f"slot {current.message_id!r}"
            )
        snapshot[slot] = hydrated
    return snapshot


class _RewindCancelled(Exception):
    def __init__(self, cancellation: asyncio.CancelledError) -> None:
        self.cancellation = cancellation


class RewindManager:
    """Impure shell over the shared Checkpointer: restores files from disk and
    truncates and forks the conversation to rewind to an earlier user message.
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        messages: MessageList,
        save_messages: SaveMessages,
        reset_session: Callable[[], Awaitable[None]],
        files: FileStore | None = None,
        *,
        fence_transcript: Callable[[], None] = lambda: None,
    ) -> None:
        self._checkpointer = checkpointer
        self._messages = messages
        self._save_messages = save_messages
        self._reset_session = reset_session
        self._files = files or FileStore()
        self._fence_transcript = fence_transcript
        self._is_rewinding = False
        self._messages.on_reset(self._on_messages_reset)

    def restorable_paths_at(self, message_index: int) -> list[str]:
        """Paths whose on-disk content would change if rewinding to this turn."""
        history = self._checkpointer.view()
        if not history.has_turn(message_index):
            return []
        plan = history.restore_plan_to_turn(message_index)
        return [
            path for path, target in plan.items() if self._files.read(path) != target
        ]

    def has_file_changes_at(self, message_index: int) -> bool:
        return bool(self.restorable_paths_at(message_index))

    def get_rewindable_messages(
        self, *, archive: ArchivedHistorySource | None = None
    ) -> list[tuple[int, str]]:
        """Return (message_index, content) for each user message."""
        messages = complete_history(self._messages, archive)
        return [
            (i, msg.content or "")
            for i, msg in enumerate(messages)
            if msg.role == Role.user and msg.content and not msg.injected
        ]

    def index_for_message_id(
        self, message_id: str, *, archive: ArchivedHistorySource | None = None
    ) -> int:
        """Resolve a rewindable user message id to its index.

        Raises:
            RewindError: If no non-injected user message carries this id.
        """
        messages = complete_history(self._messages, archive)
        for index, msg in enumerate(messages):
            if (
                msg.role == Role.user
                and not msg.injected
                and msg.message_id == message_id
            ):
                return index
        raise RewindError(f"No rewindable user message with id: {message_id}")

    async def rewind_to_message(
        self,
        message_index: int,
        *,
        restore_files: bool,
        inplace: bool = False,
        archive: ArchivedHistorySource | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """Rewind the session to the given user message index.

        Commit the conversation/session rewind, then restore files.

        File snapshots are memory-only. After resume, restoration covers only
        edits checkpointed in this process, not the historical transcript.

        - ``inplace=False`` (default, fork): save the full history under the
          current session, truncate, then fork to a fresh session so the
          original conversation is preserved as a parent.
        - ``inplace=True``: truncate first, then persist the truncated history
          under the *same* session. The rewound turns are dropped for good and
          no new session is created.

        ``archive`` (M1 §3) hydrates the session's sealed prefix from the
        transcript first, so a rewind never exposes trimmed payloads or
        tombstones to model context; the snapshot stays length-preserving, so
        ``message_index`` is unchanged.

        Returns a tuple of (message_content, restore_errors, restored_paths).

        Raises:
            RewindError: If the message index is invalid or not a user message.
        """
        return await self._rewind_legacy(
            message_index, restore_files=restore_files, inplace=inplace, archive=archive
        )

    async def _rewind_legacy(
        self,
        message_index: int,
        *,
        restore_files: bool,
        inplace: bool,
        archive: ArchivedHistorySource | None = None,
    ) -> tuple[str, list[str], list[str]]:
        # Hydration happens before anything exposes history to model context:
        # the truncation installs this snapshot and a fork hands it on. The
        # rollback list is the same complete snapshot, so a failed rewind can
        # never leave the session holding a trimmed prefix.
        messages: Sequence[LLMMessage] = complete_history(self._messages, archive)
        if message_index < 0 or message_index >= len(messages):
            raise RewindError(f"Invalid message index: {message_index}")

        user_msg = messages[message_index]
        if user_msg.role != Role.user:
            raise RewindError(f"Message at index {message_index} is not a user message")

        message_content = user_msg.content or ""
        restore_errors: list[str] = []
        restored_paths: list[str] = []

        restore_plan = None
        if restore_files:
            restore_plan = self._checkpointer.view().restore_plan_to_turn(message_index)

        # A fork's parent must be durable before any restoration mutates disk.
        if not inplace:
            await self._save_messages()

        original_messages = list(messages)
        try:
            with self._checkpointer.atomic():
                self._reset_messages(messages[:message_index])
                try:
                    if inplace:
                        await self._save_messages(allow_empty=True)
                    else:
                        await self._reset_session()
                except asyncio.CancelledError as exc:
                    # Convert cancellation while inside ``atomic`` so checkpoint
                    # rollback completes before cancellation reaches the caller.
                    raise _RewindCancelled(exc) from exc
                # Keep the recovery log intact until conversation persistence or
                # session reset has completed successfully.
                self._checkpointer.drop_turns_from(message_index)
        except _RewindCancelled as exc:
            self._reset_messages(original_messages)
            raise exc.cancellation
        except Exception:
            self._reset_messages(original_messages)
            raise

        if restore_plan is not None:
            restore_errors, restored_paths = self._files.apply(restore_plan)

        return message_content, restore_errors, restored_paths

    def _reset_messages(self, messages: Sequence[LLMMessage]) -> None:
        self._fence_transcript()
        self._is_rewinding = True
        try:
            self._messages.reset(list(messages))
        finally:
            self._is_rewinding = False

    def _on_messages_reset(self) -> None:
        """Called when the message list is reset, such as on session switch or clear.

        Skipped while rewinding: the rewind's own _truncate handles the log. In
        every other case the checkpoint log is cleared. When a turn was open,
        re-open one so the remaining tool loop can keep recording snapshots.
        """
        if self._is_rewinding:
            return
        was_open = self._checkpointer.has_open_turn
        self._checkpointer.clear()
        if was_open:
            self._checkpointer.begin_turn(len(self._messages))
