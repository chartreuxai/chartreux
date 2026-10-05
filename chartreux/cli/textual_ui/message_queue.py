from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4
from weakref import WeakSet

from textual.widget import Widget

from chartreux.app_server.models import (
    ImageAttachment,
    PreparedPrompt,
    PublicMessageEntry,
    PublicQueuedTurn,
    PublicTurnQueue,
    SessionImageContentBlock,
    SessionTextContentBlock,
    TurnUserInputEntry,
)
from chartreux.app_server.protocol import AppServerResponseError
from chartreux.cli.textual_ui.widgets.messages import QueueHeaderMessage, UserMessage
from chartreux.observability.logging import logger
from chartreux.utils.image_uri import image_attachment_from_session_block

if TYPE_CHECKING:
    from chartreux.cli.commands import Command


# Queued prompts merged into one turn join with a blank line, matching how the
# app server renders a restored multi-block user entry.
_MERGE_SEPARATOR = "\n\n"


def _queued_turn_user_entry(queued_turn: PublicQueuedTurn) -> TurnUserInputEntry | None:
    return next(
        (
            entry
            for entry in reversed(queued_turn.entries)
            if isinstance(entry, TurnUserInputEntry)
        ),
        None,
    )


def _queued_image_attachment(block: SessionImageContentBlock) -> ImageAttachment:
    return image_attachment_from_session_block(block)


@dataclass(frozen=True, slots=True)
class _QueuedPrompt:
    content: str
    skill_name: str | None = None
    prepared_prompt: PreparedPrompt | None = None
    message_entry_id: str | None = None


@dataclass(frozen=True)
class QueuePorts:
    mount_and_scroll: Callable[..., Awaitable[None]]
    current_turn_queue: Callable[[], PublicTurnQueue]
    enqueue_turn: Callable[..., Awaitable[PublicQueuedTurn]]
    replace_queued_turn: Callable[..., Awaitable[PublicQueuedTurn | None]]
    remove_queued_turn: Callable[[str], Awaitable[bool]]
    resume_turn_queue: Callable[[], Awaitable[PublicTurnQueue]]
    steer_turn: Callable[..., Awaitable[None]]
    turn_has_started: Callable[[str], bool]
    set_loading_queue_count: Callable[[int], None]
    get_show_message_timestamps: Callable[[], bool] = lambda: True
    history_message: Callable[[str], PublicMessageEntry | None] = lambda _id: None


@dataclass(slots=True)
class _Pending:
    prompt: _QueuedPrompt
    widget: UserMessage


@dataclass(slots=True)
class _MergedEntry:
    prompt: _QueuedPrompt
    widget: UserMessage


@dataclass(slots=True)
class _MergedTurn:
    """The single server queue item that all busy-time prompts merge into.

    Each queued prompt keeps its own ``UserMessage`` widget so the queue
    selection/edit UX can navigate, edit, and remove prompts individually, but
    they all share one ``item_id``: the CLI folds every prompt into that item
    with ``queue/replace`` so the server promotes them together as one turn.
    ``item_id`` is ``None`` until the first enqueue returns and when failed
    steering recovery preserves a local-only block.
    """

    entries: list[_MergedEntry]
    message_entry_id: str
    item_id: str | None = None


@dataclass(slots=True)
class _SteeringDelivery:
    block: _MergedTurn
    expected_turn_id: str
    idempotency_key: str
    require_waiting_only: bool = True
    ambiguous: bool = False


class QueueController:  # noqa: PLR0904
    """Merge busy-time prompts into one app-server turn.

    The app server keeps a FIFO queue and promotes one item at a time, which the
    desktop app relies on for turn-by-turn delivery. The CLI wants queued prompts
    delivered together, so it keeps a single queue item and folds each new prompt
    into it with ``queue/replace``. The server queue behaviour is unchanged: it
    still promotes exactly one (already-merged) item, and each queued prompt keeps
    its own widget so it stays individually editable and removable.
    """

    def __init__(self, ports: QueuePorts) -> None:
        self._ports = ports
        self._server_queue = PublicTurnQueue()
        self._merged: _MergedTurn | None = None
        self._restored: list[_MergedTurn] = []
        # Optimistic prompts sent while idle: rendered as normal messages and
        # promoted immediately, so they own their own queue item and never merge.
        self._optimistic: dict[str, _Pending] = {}
        self._header: QueueHeaderMessage | None = None
        # Serialize mutations so app-server events (sync / turn_started) cannot
        # interleave with an in-flight enqueue or replace.
        self._lock = asyncio.Lock()
        self._pending_enqueues = 0
        self._message_widgets: dict[str, WeakSet[UserMessage]] = {}
        self._posted_at: dict[str, datetime | None] = {}
        # Ambiguous deliveries remain immutable and non-promotable, but are
        # exposed to queue selection so the user can discard a stuck delivery.
        self._unresolved_steering: list[_SteeringDelivery] = []

    def reconcile_history_entry(self, entry: PublicMessageEntry) -> None:
        """Consume canonical history, before or after a queue promotion event.

        The owner wires this to EventHandler.on_user_message and supplies
        history_message for promotion-time lookup. Weak references also reach already-promoted
        widgets without retaining evicted transcript pages.
        """
        if entry.role != "user":
            return
        self._posted_at[entry.id] = entry.posted_at
        for widget in self._message_widgets.get(entry.id, ()):
            if widget.history_entry_id == entry.id:
                widget.reconcile_timestamp(entry.posted_at)
                widget.set_show_message_timestamps(
                    self._ports.get_show_message_timestamps()
                )

    def _track_message(self, widget: UserMessage) -> None:
        entry_id = widget.history_entry_id
        widget.set_show_message_timestamps(self._ports.get_show_message_timestamps())
        if entry_id is None:
            return
        self._message_widgets.setdefault(entry_id, WeakSet()).add(widget)
        if (entry := self._ports.history_message(entry_id)) is not None:
            self.reconcile_history_entry(entry)
        elif entry_id in self._posted_at:
            widget.reconcile_timestamp(self._posted_at[entry_id])

    @property
    def header(self) -> QueueHeaderMessage | None:
        return self._header

    @property
    def paused(self) -> bool:
        return self._server_queue.paused

    @property
    def has_server_work(self) -> bool:
        return bool(self) or self._pending_enqueues > 0

    @property
    def has_removable(self) -> bool:
        return any(block.entries for block in self._manageable_blocks())

    @property
    def has_unresolved_steering(self) -> bool:
        return bool(self._unresolved_steering)

    def __bool__(self) -> bool:
        return (
            bool(self._blocks())
            or bool(self._optimistic)
            or self.has_unresolved_steering
        )

    def __len__(self) -> int:
        return sum(len(block.entries) for block in self._manageable_blocks()) + len(
            self._optimistic
        )

    def pin_target(self, messages_area: Widget) -> Widget | None:
        target: Widget | None = self._header
        if target is None:
            target = self._first_pending_widget()
        if target is not None and target.parent is messages_area:
            return target
        return None

    def quit_warning_extra(self) -> str:
        if not self:
            return ""
        count = len(self)
        plural = "s" if count != 1 else ""
        return f"{count} queued message{plural} will be discarded"

    def notify_busy_changed(self) -> None:
        self._push_loading_queue_count()

    @contextmanager
    def reserve_enqueue(self) -> Iterator[None]:
        """Keep this submission visible as server work until it is finished."""
        self._pending_enqueues += 1
        try:
            yield
        finally:
            self._pending_enqueues -= 1

    async def enqueue_prompt(
        self,
        content: str,
        *,
        skill_name: str | None = None,
        prepared_prompt: PreparedPrompt | None = None,
        optimistic_start: bool = False,
    ) -> None:
        prompt = _QueuedPrompt(
            content=content,
            skill_name=skill_name,
            prepared_prompt=prepared_prompt,
            message_entry_id=str(uuid4()),
        )
        with self.reserve_enqueue():
            async with self._lock:
                if optimistic_start and not self:
                    await self._enqueue_optimistic(prompt)
                else:
                    await self._enqueue_merged(prompt)

    async def sync_server_queue(self, queue: PublicTurnQueue) -> None:
        async with self._lock:
            self._server_queue = queue.model_copy(deep=True)
            # Incremental queue updates never remove the merged block: promotion
            # pops the queue item immediately before ``TurnStarted``, so a missing
            # item does not mean the prompt was discarded. ``turn_started`` clears
            # a promoted block and ``clear_server_queue`` clears a reset one.
            known = set(self._optimistic)
            known.update(
                block.item_id for block in self._blocks() if block.item_id is not None
            )
            for item in queue.items:
                if item.id in known:
                    continue
                if self._ports.turn_has_started(item.id):
                    continue
                # A queued item we do not track yet: restore it as one block.
                # The CLI only ever creates a single item, so this is the resume
                # path, where the merged item is one combined user entry.
                await self._restore_server_item(item)
                known.add(item.id)

            if self._header is not None:
                self._header.set_paused(self.paused)
            await self._remove_header_if_empty()
            self._push_loading_queue_count()

    async def clear_server_queue(self) -> None:
        """Forget queued prompts after a server-side session reset."""
        async with self._lock:
            widgets: list[UserMessage] = []
            if self._merged is not None:
                widgets.extend(entry.widget for entry in self._merged.entries)
            for restored in self._restored:
                widgets.extend(entry.widget for entry in restored.entries)
            widgets.extend(pending.widget for pending in self._optimistic.values())

            self._server_queue = PublicTurnQueue()
            self._merged = None
            self._restored.clear()
            self._optimistic.clear()
            self._message_widgets.clear()
            self._posted_at.clear()

            removed: set[int] = set()
            for widget in widgets:
                if id(widget) in removed:
                    continue
                removed.add(id(widget))
                await widget.remove()

            await self._remove_header_if_empty()
            self._push_loading_queue_count()

    async def turn_started(self, queue_item_id: str | None) -> None:
        async with self._lock:
            await self._turn_started_locked(queue_item_id)

    async def pop_last(self) -> bool:
        async with self._lock:
            blocks = self._manageable_blocks()
            if not blocks:
                return False
            merged = blocks[-1]
            if merged.item_id is not None and self._ports.turn_has_started(
                merged.item_id
            ):
                return False
            return await self._drop_entry_locked(merged, len(merged.entries) - 1)

    async def steer_prepared(
        self,
        content: str,
        *,
        prepared_prompt: PreparedPrompt,
        expected_turn_id: str,
        skill_name: str | None = None,
    ) -> bool:
        """Admit only this new prepared prompt, without touching the backlog."""
        async with self._lock:
            entry_id = str(uuid4())
            prompt = _QueuedPrompt(
                content, skill_name, prepared_prompt.model_copy(deep=True), entry_id
            )
            widget = self._build_widget(prompt, history_entry_id=entry_id)
            self._track_message(widget)
            delivery = _SteeringDelivery(
                _MergedTurn([_MergedEntry(prompt, widget)], entry_id),
                expected_turn_id,
                f"steer:{entry_id}:{uuid4()}",
            )
            self._unresolved_steering.append(delivery)
            await self._ports.mount_and_scroll(widget)
            return await self._deliver_prepared_locked(delivery)

    async def retry_unresolved_steering(self) -> bool:
        async with self._lock:
            sent = False
            for delivery in list(self._unresolved_steering):
                sent = await self._deliver_prepared_locked(delivery) or sent
            return sent

    async def _deliver_prepared_locked(self, delivery: _SteeringDelivery) -> bool:
        block = delivery.block
        try:
            await self._ports.steer_turn(
                self._server_text(block.entries),
                self._server_images(block.entries) or None,
                block.message_entry_id,
                require_waiting_only=delivery.require_waiting_only,
                expected_turn_id=delivery.expected_turn_id,
                idempotency_key=delivery.idempotency_key,
            )
        except asyncio.CancelledError:
            delivery.ambiguous = True
            self._push_loading_queue_count()
            raise
        except Exception as error:
            if (
                isinstance(error, AppServerResponseError)
                and isinstance(error.error.data, dict)
                and error.error.data.get("steerCommitted") is False
            ):
                # Definitive rejection only. Keep this as a separate trailing
                # block so older prompts are neither replaced nor reordered.
                delivery.ambiguous = False
                self._unresolved_steering.remove(delivery)
                if delivery.require_waiting_only or self._merged is not None:
                    self._restored.append(block)
                else:
                    self._merged = block
                await self._ensure_header()
                await self._reenqueue_after_failed_steer(block)
                if block.item_id is not None and self._ports.turn_has_started(
                    block.item_id
                ):
                    await self._turn_started_locked(block.item_id)
                self._push_loading_queue_count()
                raise
            delivery.ambiguous = True
            self._push_loading_queue_count()
            raise RuntimeError(
                "Steering delivery could not be confirmed; your prompt was kept "
                "unresolved. Retry uses the same delivery receipt; queue remove "
                "discards the entire unresolved delivery without undoing accepted context."
            ) from error
        self._unresolved_steering.remove(delivery)
        for entry in block.entries:
            self._track_message(entry.widget)
            await entry.widget.set_pending(False)
        await self._remove_header_if_empty()
        self._push_loading_queue_count()
        return True

    async def steer_pending(self, *, expected_turn_id: str) -> bool:
        """Remove the backlog before explicit steering, retaining a retry receipt.

        Only definitive pre-commit rejection re-enqueues the block. Ambiguous
        outcomes stay outside the promotable/editable queue until receipt replay.
        """
        async with self._lock:
            merged = self._merged
            if merged is None or merged.item_id is None:
                return False
            if self._ports.turn_has_started(merged.item_id):
                await self._turn_started_locked(merged.item_id)
                return False
            removed = await self._ports.remove_queued_turn(merged.item_id)
            self._server_queue = self._ports.current_turn_queue().model_copy(deep=True)
            if not removed or self._ports.turn_has_started(merged.item_id):
                # Promoted or dropped between the guard and the remove: let the
                # normal turn-start path finalize it instead of steering, so it
                # is never delivered both as a steer and as its own turn.
                if self._ports.turn_has_started(merged.item_id):
                    await self._turn_started_locked(merged.item_id)
                return False
            merged.item_id = None
            self._merged = None
            delivery = _SteeringDelivery(
                merged,
                expected_turn_id,
                f"steer:{merged.message_entry_id}:{uuid4()}",
                require_waiting_only=False,
            )
            self._unresolved_steering.append(delivery)
            return await self._deliver_prepared_locked(delivery)

    async def _reenqueue_after_failed_steer(self, merged: _MergedTurn) -> None:
        # Use a fresh idempotency key: the original one (derived from
        # ``message_entry_id``) was consumed by the now-removed item, so reusing
        # it would be rejected. ``message_entry_id`` stays the same for rewind.
        queued_turn = await self._ports.enqueue_turn(
            self._server_text(merged.entries),
            message_entry_id=merged.message_entry_id,
            images=self._server_images(merged.entries) or None,
            idempotency_key=str(uuid4()),
        )
        merged.item_id = queued_turn.id
        self._server_queue = self._ports.current_turn_queue().model_copy(deep=True)

    def queue_item_texts(self) -> list[tuple[int, str]]:
        return [
            (index, entry.prompt.content) for index, entry in enumerate(self._entries())
        ]

    @property
    def widgets(self) -> list[UserMessage]:
        return [entry.widget for entry in self._entries()]

    async def pop_at(self, index: int) -> bool:
        async with self._lock:
            located = self._entry_at(index)
            if located is None:
                return False
            merged, entry_index = located
            if merged.item_id is not None and self._ports.turn_has_started(
                merged.item_id
            ):
                return False
            return await self._drop_entry_locked(merged, entry_index)

    def is_unresolved_prompt(self, index: int) -> bool:
        located = self._entry_at(index)
        return located is not None and any(
            delivery.block is located[0] for delivery in self._unresolved_steering
        )

    async def update_prompt(
        self,
        queue_index: int,
        content: str,
        *,
        prepared_prompt: PreparedPrompt | None = None,
    ) -> bool:
        async with self._lock:
            located = self._entry_at(queue_index)
            if located is None:
                return False
            merged, entry_index = located
            if any(delivery.block is merged for delivery in self._unresolved_steering):
                # Editing would change the payload tied to the replay receipt.
                return False
            if merged.item_id is not None and self._ports.turn_has_started(
                merged.item_id
            ):
                await self._turn_started_locked(merged.item_id)
                return False
            entry = merged.entries[entry_index]
            edited = _MergedEntry(
                replace(entry.prompt, content=content, prepared_prompt=prepared_prompt),
                entry.widget,
            )
            candidate = list(merged.entries)
            candidate[entry_index] = edited
            if not await self._replace_entries_locked(candidate, merged):
                return False
            entry.widget.update_content(content)
            self._push_loading_queue_count()
            return True

    async def resume(self) -> None:
        async with self._lock:
            if self._server_queue.paused:
                self._server_queue = await self._ports.resume_turn_queue()
            if self._header is not None:
                self._header.set_paused(self.paused)

    # -- internal helpers (all run under ``self._lock``) -------------------

    def _blocks(self) -> list[_MergedTurn]:
        return [*([self._merged] if self._merged is not None else []), *self._restored]

    def _manageable_blocks(self) -> list[_MergedTurn]:
        return [*self._blocks(), *(d.block for d in self._unresolved_steering)]

    def _entries(self) -> list[_MergedEntry]:
        return [entry for block in self._manageable_blocks() for entry in block.entries]

    def _entry_at(self, index: int) -> tuple[_MergedTurn, int] | None:
        if index < 0:
            return None
        for block in self._manageable_blocks():
            if index < len(block.entries):
                return block, index
            index -= len(block.entries)
        return None

    async def _enqueue_optimistic(self, prompt: _QueuedPrompt) -> None:
        images = (
            prompt.prepared_prompt.images if prompt.prepared_prompt is not None else []
        )
        widget = UserMessage(
            prompt.content,
            pending=False,
            history_entry_id=prompt.message_entry_id,
            images=images or None,
        )
        self._track_message(widget)
        await self._ports.mount_and_scroll(widget)
        pending = _Pending(prompt, widget)
        try:
            queued_turn = await self._ports.enqueue_turn(
                self._server_text_of(prompt),
                message_entry_id=prompt.message_entry_id,
                images=self._server_images_of(prompt) or None,
            )
        except Exception:
            await widget.remove()
            raise
        self._optimistic[queued_turn.id] = pending
        self._server_queue = self._ports.current_turn_queue().model_copy(deep=True)
        if self._ports.turn_has_started(queued_turn.id):
            await self._turn_started_locked(queued_turn.id)
        self._push_loading_queue_count()

    async def _enqueue_merged(self, prompt: _QueuedPrompt) -> None:
        merged = self._merged
        if (
            merged is not None
            and merged.item_id is not None
            and self._ports.turn_has_started(merged.item_id)
        ):
            # The current block promoted before this prompt arrived; finalize it
            # and start a fresh block for the next turn.
            await self._turn_started_locked(merged.item_id)
            merged = self._merged

        if merged is None or merged.item_id is None:
            await self._create_merged(prompt)
            return

        # Later prompts stay individually editable but share the first prompt's
        # server history entry (the merged item has one entry_id), so they get no
        # history id of their own -- a unique id would be a dangling rewind
        # target once the turn starts.
        widget = self._build_widget(prompt)
        await self._ports.mount_and_scroll(widget, after=self._last_widget())
        # Commit the new prompt to the merged item only if the replace lands.
        # If the item promotes or is removed during the round-trip, the new
        # prompt is not part of that turn, so it must not be un-pended with it.
        candidate = [*merged.entries, _MergedEntry(prompt, widget)]
        try:
            replaced = await self._replace_entries_locked(candidate)
        except Exception:
            # The widget is mounted but not committed to the merged item; drop it
            # so a failed replace does not leave an untracked pending prompt.
            await widget.remove()
            raise
        if replaced:
            self._push_loading_queue_count()
            return
        await widget.remove()
        # The previous block started or was removed mid-replace: queue this
        # prompt as a fresh block so it is not lost.
        await self._create_merged(prompt)

    async def _create_merged(self, prompt: _QueuedPrompt) -> None:
        await self._ensure_header()
        retained = self._merged
        if retained is not None:
            # A failed steering recovery can retain a block that has no server
            # item. Append to it and enqueue the combined block rather than
            # replacing it, which would abandon the retained user content.
            widget = self._build_widget(prompt)
            await self._ports.mount_and_scroll(widget, after=self._last_widget())
            entries = [*retained.entries, _MergedEntry(prompt, widget)]
            try:
                queued_turn = await self._ports.enqueue_turn(
                    self._server_text(entries),
                    message_entry_id=retained.message_entry_id,
                    images=self._server_images(entries) or None,
                    idempotency_key=str(uuid4()),
                )
            except Exception:
                await widget.remove()
                raise
            retained.entries = entries
            retained.item_id = queued_turn.id
            self._relink_merged()
            self._server_queue = self._ports.current_turn_queue().model_copy(deep=True)
            if self._ports.turn_has_started(queued_turn.id):
                await self._turn_started_locked(queued_turn.id)
            self._push_loading_queue_count()
            return

        # The first prompt owns the merged item's server history entry, so its
        # widget keeps that id for rewind and history lookups.
        widget = self._build_widget(prompt, history_entry_id=prompt.message_entry_id)
        await self._ports.mount_and_scroll(widget, after=self._header)
        merged = _MergedTurn(
            entries=[_MergedEntry(prompt, widget)],
            message_entry_id=prompt.message_entry_id or str(uuid4()),
        )
        self._merged = merged
        self._relink_merged()
        try:
            queued_turn = await self._ports.enqueue_turn(
                self._server_text(merged.entries),
                message_entry_id=merged.message_entry_id,
                images=self._server_images(merged.entries) or None,
            )
        except Exception:
            await widget.remove()
            self._merged = None
            await self._remove_header_if_empty()
            raise
        merged.item_id = queued_turn.id
        self._server_queue = self._ports.current_turn_queue().model_copy(deep=True)
        if self._ports.turn_has_started(queued_turn.id):
            await self._turn_started_locked(queued_turn.id)
        self._push_loading_queue_count()

    async def _drop_entry_locked(self, merged: _MergedTurn, index: int) -> bool:
        delivery = next(
            (d for d in self._unresolved_steering if d.block is merged), None
        )
        if delivery is not None:
            # Discard the entire immutable delivery, not a changed retry payload.
            # This cannot retract context that the server may already have seen.
            self._unresolved_steering.remove(delivery)
            for entry in merged.entries:
                await entry.widget.remove()
            await self._remove_header_if_empty()
            self._push_loading_queue_count()
            return True
        entry = merged.entries[index]
        if len(merged.entries) == 1:
            return await self._remove_merged_locked(merged)
        candidate = [e for i, e in enumerate(merged.entries) if i != index]
        if not await self._replace_entries_locked(candidate, merged):
            return False
        await entry.widget.remove()
        self._push_loading_queue_count()
        return True

    async def _replace_entries_locked(
        self, entries: list[_MergedEntry], merged: _MergedTurn | None = None
    ) -> bool:
        merged = merged or self._merged
        if merged is None:
            return False
        if merged.item_id is None:
            merged.entries = entries
            self._relink_merged(merged)
            return True
        queued_turn = await self._ports.replace_queued_turn(
            merged.item_id,
            self._server_text(entries),
            message_entry_id=merged.message_entry_id,
            images=self._server_images(entries) or None,
        )
        self._server_queue = self._ports.current_turn_queue().model_copy(deep=True)
        if queued_turn is not None:
            merged.entries = entries
            self._relink_merged(merged)
            return True
        # not_found: the item started or was removed.
        if self._ports.turn_has_started(merged.item_id):
            await self._turn_started_locked(merged.item_id)
        else:
            await self._discard_merged(merged)
        self._push_loading_queue_count()
        return False

    async def _remove_merged_locked(self, merged: _MergedTurn) -> bool:
        if merged.item_id is None:
            await self._discard_merged(merged)
            self._push_loading_queue_count()
            return True
        removed = await self._ports.remove_queued_turn(merged.item_id)
        self._server_queue = self._ports.current_turn_queue().model_copy(deep=True)
        if not removed or self._ports.turn_has_started(merged.item_id):
            return False
        await self._discard_merged(merged)
        self._push_loading_queue_count()
        return True

    async def _discard_merged(self, merged: _MergedTurn) -> None:
        if merged is self._merged:
            self._merged = None
        else:
            self._restored.remove(merged)
        for entry in merged.entries:
            await entry.widget.remove()
        await self._remove_header_if_empty()

    async def _turn_started_locked(self, queue_item_id: str | None) -> None:
        if queue_item_id is None:
            return
        pending = self._optimistic.pop(queue_item_id, None)
        if pending is not None:
            self._track_message(pending.widget)
            await pending.widget.set_pending(False)
            await self._reset_header_position()
            self._push_loading_queue_count()
            return
        merged = next(
            (block for block in self._blocks() if block.item_id == queue_item_id), None
        )
        if merged is None:
            return
        for entry in merged.entries:
            self._track_message(entry.widget)
            await entry.widget.set_pending(False)
        if merged is self._merged:
            self._merged = None
        else:
            self._restored.remove(merged)
        await self._reset_header_position()
        self._push_loading_queue_count()

    async def _restore_server_item(self, queued_turn: PublicQueuedTurn) -> None:
        user_entry = _queued_turn_user_entry(queued_turn)
        if user_entry is None:
            return
        content = _MERGE_SEPARATOR.join(
            block.text
            for block in user_entry.content
            if isinstance(block, SessionTextContentBlock)
        )
        images = [
            _queued_image_attachment(block)
            for block in user_entry.content
            if isinstance(block, SessionImageContentBlock)
        ]
        prompt = _QueuedPrompt(
            content,
            prepared_prompt=PreparedPrompt(
                display_text=content, prompt_text=content, images=images
            ),
            message_entry_id=user_entry.entry_id,
        )
        widget = UserMessage(
            content,
            pending=True,
            history_entry_id=user_entry.entry_id,
            images=images or None,
        )
        merged = self._merged
        if merged is not None:
            restored = _MergedTurn(
                entries=[_MergedEntry(prompt, widget)],
                message_entry_id=user_entry.entry_id or str(uuid4()),
                item_id=queued_turn.id,
            )
            after = self._last_widget()
            await self._ports.mount_and_scroll(widget, after=after)
            self._restored.append(restored)
            self._relink_merged(restored)
            return

        self._merged = _MergedTurn(
            entries=[_MergedEntry(prompt, widget)],
            message_entry_id=user_entry.entry_id or str(uuid4()),
            item_id=queued_turn.id,
        )
        self._relink_merged()
        await self._ensure_header()
        await self._ports.mount_and_scroll(widget, after=self._header)

    def _build_widget(
        self, prompt: _QueuedPrompt, *, history_entry_id: str | None = None
    ) -> UserMessage:
        images = (
            prompt.prepared_prompt.images if prompt.prepared_prompt is not None else []
        )
        return UserMessage(
            prompt.content,
            pending=True,
            history_entry_id=history_entry_id,
            images=images or None,
        )

    def _relink_merged(self, merged: _MergedTurn | None = None) -> None:
        """Render the merged prompts as one visual block.

        Each queued prompt keeps its own widget, but consecutive prompts in the
        same merged turn hide the separator between them and mark themselves as
        continuations, so they read as a single grouped message (matching the
        legacy queued-prompt rendering) even though they stay individually
        selectable, editable, and removable.
        """
        merged = merged or self._merged
        if merged is None:
            return
        widgets = [entry.widget for entry in merged.entries]
        last = len(widgets) - 1
        # The merged turn has one server history entry. Only the first widget
        # is rewindable; re-assign so popping the oldest prompt does not lose
        # the id (later widgets are mounted with history_entry_id=None).
        rewind_id = merged.message_entry_id
        for index, widget in enumerate(widgets):
            widget.set_follows_previous(index > 0)
            widget.set_show_separator(index == last)
            widget.history_entry_id = rewind_id if index == 0 else None
            self._track_message(widget)

    @staticmethod
    def _server_text_of(prompt: _QueuedPrompt) -> str:
        prepared = prompt.prepared_prompt
        return prepared.prompt_text if prepared is not None else prompt.content

    @staticmethod
    def _server_images_of(prompt: _QueuedPrompt) -> list[ImageAttachment]:
        prepared = prompt.prepared_prompt
        return list(prepared.images) if prepared is not None else []

    def _server_text(self, entries: list[_MergedEntry]) -> str:
        return _MERGE_SEPARATOR.join(
            self._server_text_of(entry.prompt) for entry in entries
        )

    def _server_images(self, entries: list[_MergedEntry]) -> list[ImageAttachment]:
        images: list[ImageAttachment] = []
        for entry in entries:
            images.extend(self._server_images_of(entry.prompt))
        return images

    def _last_widget(self) -> UserMessage | QueueHeaderMessage | None:
        blocks = self._blocks()
        if blocks and blocks[-1].entries:
            return blocks[-1].entries[-1].widget
        return self._header

    def _first_pending_widget(self) -> UserMessage | None:
        blocks = self._blocks()
        if blocks and blocks[0].entries:
            return blocks[0].entries[0].widget
        if self._optimistic:
            return next(iter(self._optimistic.values())).widget
        return None

    async def _ensure_header(self) -> None:
        if self._header is not None:
            return
        header = QueueHeaderMessage(paused=self.paused)
        self._header = header
        await self._ports.mount_and_scroll(header)

    async def _reset_header_position(self) -> None:
        await self._remove_header()
        first_pending = self._first_pending_widget()
        if first_pending is None:
            return
        header = QueueHeaderMessage(paused=self.paused)
        self._header = header
        await self._ports.mount_and_scroll(header, before=first_pending)

    async def _remove_header_if_empty(self) -> None:
        if self or self._header is None:
            return
        await self._remove_header()

    async def _remove_header(self) -> None:
        if self._header is None:
            return
        header = self._header
        self._header = None
        if header.parent is not None:
            await header.remove()

    def _push_loading_queue_count(self) -> None:
        self._ports.set_loading_queue_count(len(self))


@dataclass(frozen=True)
class SideChannelPorts:
    """Callbacks for side-channel slash command execution."""

    invoke_command: Callable[[str, Command, str, str], Awaitable[bool]]


@dataclass(slots=True)
class SideChannelItem:
    cmd_name: str
    command: Command
    cmd_args: str
    display_text: str


class SideChannelController:
    """Run one allowlisted slash command alongside the active job."""

    def __init__(self, ports: SideChannelPorts) -> None:
        self._ports = ports
        self._task: asyncio.Task | None = None
        self._enabled = True

    def __bool__(self) -> bool:
        return self.draining

    def __len__(self) -> int:
        return 1 if self.draining else 0

    @property
    def draining(self) -> bool:
        return self._task is not None and not self._task.done()

    def enqueue(
        self, cmd_name: str, command: Command, cmd_args: str, display_text: str
    ) -> bool:
        if not self._enabled or self.draining:
            return False
        item = SideChannelItem(cmd_name, command, cmd_args, display_text)
        self._task = asyncio.create_task(self._run(item))
        return True

    async def _run(self, item: SideChannelItem) -> None:
        try:
            await self._ports.invoke_command(
                item.cmd_name, item.command, item.cmd_args, item.display_text
            )
        except Exception:
            logger.exception("Side-channel command failed")
        finally:
            self._task = None

    async def shutdown(self) -> None:
        self._enabled = False
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
