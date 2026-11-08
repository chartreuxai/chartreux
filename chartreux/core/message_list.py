from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import count
import threading
from typing import overload

from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.transcript_archive import ArchivedPrefixV1

#: Monotonic per-instance source identity. A request envelope names the list it
#: was frozen from, and ``id()`` is reusable once the list is collected.
_source_identities = count(1)


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """The frozen provenance of one request-preparation source (S1-A1).

    Minted per attempt at the coordinator's input build, after the candidate
    model is selected. The identity/version/epoch triple names the exact list
    state the request was built from; the model key names why two prepared
    requests are not interchangeable — ``supports_images`` gates the image
    admission and ``alias`` identifies the wire model. No field is refreshed in
    place: a later attempt mints a new envelope.
    """

    source_identity: int
    version: int
    epoch: int
    model_alias: str
    supports_images: bool


def freeze_request_provenance(
    messages: Sequence[LLMMessage], *, model_alias: str, supports_images: bool
) -> tuple[tuple[LLMMessage, ...], RequestEnvelope | None]:
    """Freeze ``messages`` for one request attempt, with provenance if owned.

    A ``MessageList`` freezes atomically — snapshot, version and epoch read as
    one triple — and yields its envelope; an unprovenanced sequence (a model
    context projection, a sideband list) keeps the naive snapshot and reports
    no envelope.
    """
    if not isinstance(messages, MessageList):
        return tuple(messages), None
    frozen, version, epoch = messages.freeze_for_request()
    return frozen, RequestEnvelope(
        source_identity=messages.source_identity,
        version=version,
        epoch=epoch,
        model_alias=model_alias,
        supports_images=supports_images,
    )


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
        # S1-A1 request provenance: a SEPARATE mutation lock, ordered AFTER
        # ``_lock`` (a ``_lock`` holder may take it; nothing takes ``_lock``
        # while holding it — ``freeze_for_request`` takes it alone), so each
        # structural change and the version bump describing it are one atomic
        # step and a freeze can never pair fresh content with a stale version.
        # The version counts retained-content mutations; the epoch counts list
        # identities and stays the reset/rebind signal.
        self._mutation_lock = threading.RLock()
        self._message_version = 0
        self._source_identity = next(_source_identities)

    @property
    def message_epoch(self) -> int:
        """The current message epoch; bumped by reset/rebind only."""
        return self._message_epoch

    @property
    def message_version(self) -> int:
        """The retained-content mutation version.

        Bumped by every mutation of the retained messages — the six structural
        writers, ``extend`` per appended message, and the post-retention
        in-place stamp — and never by a read or a refused mutation.
        """
        return self._message_version

    @property
    def source_identity(self) -> int:
        """This list's stable identity: minted once, unique across instances."""
        return self._source_identity

    def freeze_for_request(self) -> tuple[tuple[LLMMessage, ...], int, int]:
        """Snapshot the retained messages with the version and epoch they carry.

        Read under the mutation lock alone: every structural writer holds that
        lock across its change and its version increment (reset/rebind hold it
        across the content swap and the epoch bump), so the triple is
        consistent for the whole snapshot. The snapshot is shallow — the
        message objects stay live — and a post-retention in-place write goes
        through ``stamp_turn_duration``, which bumps the version.
        """
        with self._mutation_lock:
            return tuple(self._data), self._message_version, self._message_epoch

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
            # The slot writes and the version bump are one step under the
            # mutation lock; every refusal above already left both untouched.
            with self._mutation_lock:
                for slot, replacement in zip(sealed_indexes, replacements, strict=True):
                    self._data[slot] = replacement
                self._archive_descriptor = descriptor
                self._message_version += 1

    def append(self, msg: LLMMessage) -> None:
        with self._mutation_lock:
            self._data.append(msg)
            self._message_version += 1

    def insert(self, i: int, msg: LLMMessage) -> None:
        with self._mutation_lock:
            self._data.insert(i, msg)
            self._message_version += 1

    def replace(self, i: int, msg: LLMMessage) -> None:
        """Replace the message at ``i`` in place (PI1/T designed replacement).

        Used by the transcript recorder to swap a synthetic TOOL_NO_RESPONSE
        placeholder for the genuine result that supersedes it — same length,
        same position, no reset hooks fired.
        """
        with self._mutation_lock:
            self._data[i] = msg
            self._message_version += 1

    def extend(self, msgs: list[LLMMessage]) -> None:
        # Per appended message: each append is its own retained-content
        # mutation, so a partially applied extend is visible as such.
        for msg in msgs:
            self.append(msg)

    def stamp_turn_duration(self, message_id: str, duration: float) -> bool:
        """Stamp ``turn_duration`` on the retained prose owner ``message_id``.

        The one post-retention in-place write in production (P1-CENSUS v2): the
        message object is already in the retained list, so this accessor takes
        the mutation lock and bumps the request version — a request frozen
        before the stamp can never be served after it. The owner is the LAST
        retained assistant message carrying content under that id (the
        recorder keeps the genuinely superseding record last). Returns whether
        an owner was found; a miss mutates nothing.
        """
        with self._mutation_lock:
            for message in reversed(self._data):
                if (
                    message.message_id == message_id
                    and message.role is Role.assistant
                    and message.content
                ):
                    message.turn_duration = duration
                    self._message_version += 1
                    return True
        return False

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
            with self._mutation_lock:
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
            with self._mutation_lock:
                system = [m for m in self._data if m.role is Role.system]
                self._data = [*system, *tail]
                # Resume rebind is a list rebind: new epoch, owned archive
                # state cleared (the rebound list holds complete messages).
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
            with self._mutation_lock:
                if self._data and self._data[0].role == Role.system:
                    self._data[0] = msg
                else:
                    self._data.insert(0, msg)
                self._message_version += 1

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
