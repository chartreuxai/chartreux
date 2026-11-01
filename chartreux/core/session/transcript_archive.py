"""Archived transcript prefix: descriptor, sealing primitives, and the shared READ interface.

M1 (message-list RAM trim) archives a sealed prefix of ``messages.jsonl`` in
place: the JSONL stays complete on disk, and an in-memory descriptor records
which byte prefix of it is archived. This module owns the descriptor type, the
sealing surface (eligible-prefix rule, seal receipt, seal refusal), and the
single archive-reading surface. Later subwaves (P4c1/P4c2) consume the read
interface below and must not re-implement archive reading.

The read interface is exported but UNWIRED in production: nothing calls it
until a later subwave. The descriptor itself is process-local — it is never
persisted ("persist no watermark": it describes RAM eviction, not a disk
format).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from chartreux.core.llm_models import LLMMessage, Role
from chartreux.utils.durable_io import read_chunks

TRIM_MODES: tuple[str, str] = ("payloads", "tombstones")
ARCHIVE_DESCRIPTOR_VERSION = 1
_SHA256_HEX_LENGTH = 64
_HEX_DIGITS = "0123456789abcdef"


class ArchiveIntegrityError(ValueError):
    """The archived bytes do not match the descriptor; reading fails closed."""


def _require_int(name: str, value: int) -> None:
    # bool is an int subclass; a descriptor field must be a real int.
    if type(value) is not int or value < 0:
        raise ValueError(
            f"ArchivedPrefixV1.{name} must be a non-negative int, got {value!r}"
        )


@dataclass(frozen=True)
class ArchivedPrefixV1:
    """In-memory descriptor of the archived prefix of ``messages.jsonl``.

    Fields per the M1 storage model: ``message_epoch`` changes on list
    reset/rebind (never on cursor invalidation); ``archived_count`` is the
    number of non-system JSONL records archived; ``prefix_end_byte`` is the
    byte immediately after those records; ``prefix_sha256`` hashes their exact
    persisted bytes; ``boundary_message_id`` is the compaction envelope
    authorizing the archive; ``trim_mode`` is ``payloads`` or ``tombstones``.
    """

    session_id: str
    message_epoch: int
    archived_count: int
    prefix_end_byte: int
    prefix_sha256: str
    boundary_message_id: str | None
    trim_mode: Literal["payloads", "tombstones"]

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError(
                f"ArchivedPrefixV1.session_id must be a non-empty str, got {self.session_id!r}"
            )
        _require_int("message_epoch", self.message_epoch)
        _require_int("archived_count", self.archived_count)
        _require_int("prefix_end_byte", self.prefix_end_byte)
        if self.archived_count > 0 and self.prefix_end_byte == 0:
            raise ValueError(
                "ArchivedPrefixV1 with archived_count > 0 must have prefix_end_byte > 0"
            )
        if self.archived_count == 0 and self.prefix_end_byte != 0:
            raise ValueError(
                "ArchivedPrefixV1 with archived_count == 0 must have prefix_end_byte == 0"
            )
        if (
            not isinstance(self.prefix_sha256, str)
            or len(self.prefix_sha256) != _SHA256_HEX_LENGTH
            or any(c not in _HEX_DIGITS for c in self.prefix_sha256)
        ):
            raise ValueError(
                "ArchivedPrefixV1.prefix_sha256 must be 64 lowercase hex chars, "
                f"got {self.prefix_sha256!r}"
            )
        if self.boundary_message_id is not None and (
            not isinstance(self.boundary_message_id, str)
            or not self.boundary_message_id
        ):
            raise ValueError(
                "ArchivedPrefixV1.boundary_message_id must be None or a non-empty str, "
                f"got {self.boundary_message_id!r}"
            )
        if self.trim_mode not in TRIM_MODES:
            raise ValueError(
                f"ArchivedPrefixV1.trim_mode must be one of {TRIM_MODES}, got {self.trim_mode!r}"
            )

    @property
    def version(self) -> int:
        return ARCHIVE_DESCRIPTOR_VERSION


@dataclass(frozen=True)
class ArchiveSaveSnapshot:
    """The archive side of a save snapshot.

    Captured with every accepted save: the logger-retained descriptor paired
    with the message epoch of the list that was saved, so a stale descriptor
    (one whose epoch no longer matches the list) is visible at save time.
    """

    descriptor: ArchivedPrefixV1 | None
    message_epoch: int | None

    @property
    def is_authoritative(self) -> bool:
        """True when a retained descriptor matches the saved list's epoch."""
        return (
            self.descriptor is not None
            and self.message_epoch is not None
            and self.descriptor.message_epoch == self.message_epoch
        )


class ArchiveSealError(ValueError):
    """Sealing was refused; NO receipt is issued. Fail closed.

    ``violations`` carries the transcript-integrity violations when the
    refusal came from the strict pre-seal validation of the closed prefix.
    """

    def __init__(self, reason: str, *, violations: tuple[Any, ...] = ()) -> None:
        self.reason = reason
        self.violations: tuple[Any, ...] = tuple(violations)
        super().__init__(reason)


@dataclass(frozen=True)
class ArchiveSealReceipt:
    """The durable-persist receipt authorizing RAM eviction of a sealed prefix.

    Issued by the logger's sealing operation only AFTER the live transcript
    was durably persisted and the sealed prefix bytes were verified on disk
    against ``descriptor``. A returned save call is not authorization; this
    receipt is.
    """

    descriptor: ArchivedPrefixV1
    sealed_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ArchivedPrefixV1):
            raise ValueError(
                f"ArchiveSealReceipt.descriptor must be an ArchivedPrefixV1, "
                f"got {self.descriptor!r}"
            )
        if not isinstance(self.sealed_at, str) or not self.sealed_at:
            raise ValueError(
                f"ArchiveSealReceipt.sealed_at must be a non-empty str, got {self.sealed_at!r}"
            )

    @property
    def session_id(self) -> str:
        return self.descriptor.session_id

    @property
    def message_epoch(self) -> int:
        return self.descriptor.message_epoch

    @property
    def archived_count(self) -> int:
        return self.descriptor.archived_count

    @property
    def prefix_end_byte(self) -> int:
        return self.descriptor.prefix_end_byte

    @property
    def prefix_sha256(self) -> str:
        return self.descriptor.prefix_sha256

    @property
    def boundary_message_id(self) -> str | None:
        return self.descriptor.boundary_message_id


@dataclass(frozen=True)
class EligibleArchivePrefix:
    """The closed prefix eligible for sealing (M1's eligible-prefix rule).

    ``messages`` holds the non-system messages of the closed prefix, ending at
    a complete settled round strictly before the newest compaction boundary;
    ``boundary_message_id`` is that newest boundary envelope's message id.
    """

    messages: tuple[LLMMessage, ...]
    boundary_message_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple) or not all(
            isinstance(message, LLMMessage) for message in self.messages
        ):
            raise ValueError(
                f"EligibleArchivePrefix.messages must be a tuple of LLMMessage, "
                f"got {self.messages!r}"
            )
        if any(message.role is Role.system for message in self.messages):
            raise ValueError(
                "an eligible archive prefix never contains system messages"
            )
        if self.boundary_message_id is not None and (
            not isinstance(self.boundary_message_id, str)
            or not self.boundary_message_id
        ):
            raise ValueError(
                "EligibleArchivePrefix.boundary_message_id must be None or a "
                f"non-empty str, got {self.boundary_message_id!r}"
            )


def _last_settled_round_end(region: Sequence[LLMMessage]) -> int:
    """Length of the longest prefix of ``region`` ending at a complete settled round.

    A round starts at a user message and completes where the next user message
    begins (at the region end, the newest boundary begins the next round). A
    round is settled when it contains an assistant message and no assistant
    tool call opened inside it is left unanswered.
    """
    last_end = 0
    in_round = False
    has_assistant = False
    open_calls: set[str] = set()
    for index, message in enumerate(region):
        if message.role is Role.user:
            if in_round and has_assistant and not open_calls:
                last_end = index
            in_round = True
            has_assistant = False
            open_calls = set()
        elif message.role is Role.assistant:
            has_assistant = True
            if message.tool_calls:
                open_calls.update(
                    call.id for call in message.tool_calls if call.id is not None
                )
        elif message.role is Role.tool and message.tool_call_id is not None:
            open_calls.discard(message.tool_call_id)
    if region and in_round and has_assistant and not open_calls:
        last_end = len(region)
    return last_end


def eligible_archive_prefix(
    messages: Sequence[LLMMessage],
) -> EligibleArchivePrefix | None:
    """The closed prefix eligible for sealing, or None when nothing is eligible.

    M1's eligible-prefix rule, operationalized:

    - strictly before the newest compaction boundary (the boundary envelope
      itself stays live);
    - excluding system messages (the persisted transcript holds non-system
      records only, so the prefix is taken over non-system messages);
    - ending at a complete settled round;
    - excluding the active invocation (everything after the newest boundary —
      where the active invocation lives — is never eligible; a boundary
      appended mid-invocation leaves an unsettled trailing round that the
      settled-round rule trims back).
    """
    non_system = tuple(
        message for message in messages if message.role is not Role.system
    )
    boundary_position: int | None = None
    boundary_message_id: str | None = None
    for index, message in enumerate(non_system):
        if message.context_boundary == "compaction":
            boundary_position = index
            boundary_message_id = message.message_id
    if boundary_position is None:
        return None
    region = non_system[:boundary_position]
    end = _last_settled_round_end(region)
    if end == 0:
        return None
    return EligibleArchivePrefix(
        messages=region[:end], boundary_message_id=boundary_message_id
    )


def _parse_record_line(line: bytes) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except ValueError as exc:
        raise ArchiveIntegrityError(
            f"archived transcript line is not JSON: {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise ArchiveIntegrityError("archived transcript line is not a JSON object")
    return record


def iter_archived_records(
    descriptor: ArchivedPrefixV1, chunks: Iterable[bytes]
) -> Iterator[dict[str, Any]]:
    """Stream the archived records named by ``descriptor`` from ``chunks``.

    The exact bytes are hashed and counted as they stream; each complete line
    is parsed and yielded one at a time, so the archived records are never
    materialized whole by this interface. Reading fails closed with
    ``ArchiveIntegrityError`` when the stream holds more bytes than
    ``prefix_end_byte``, more records than ``archived_count``, a line that is
    not a JSON object, a trailing partial line, or when the final byte count,
    sha256, or record count does not match the descriptor.
    """
    hasher = hashlib.sha256()
    total = 0
    count = 0
    buf = b""
    for chunk in chunks:
        if not chunk:
            continue
        if total + len(chunk) > descriptor.prefix_end_byte:
            raise ArchiveIntegrityError(
                f"archived bytes exceed prefix_end_byte={descriptor.prefix_end_byte}"
            )
        hasher.update(chunk)
        total += len(chunk)
        buf += chunk
        while True:
            newline = buf.find(b"\n")
            if newline < 0:
                break
            line, buf = buf[:newline], buf[newline + 1 :]
            record = _parse_record_line(line)
            count += 1
            if count > descriptor.archived_count:
                raise ArchiveIntegrityError(
                    f"archived records exceed archived_count={descriptor.archived_count}"
                )
            yield record
    if buf:
        raise ArchiveIntegrityError("archived bytes end inside a record line")
    if total != descriptor.prefix_end_byte:
        raise ArchiveIntegrityError(
            f"archived byte count {total} != prefix_end_byte={descriptor.prefix_end_byte}"
        )
    if hasher.hexdigest() != descriptor.prefix_sha256:
        raise ArchiveIntegrityError("archived bytes do not match prefix_sha256")
    if count != descriptor.archived_count:
        raise ArchiveIntegrityError(
            f"archived record count {count} != archived_count={descriptor.archived_count}"
        )


def read_archived_records(
    descriptor: ArchivedPrefixV1, archived_bytes: bytes
) -> list[dict[str, Any]]:
    """Read and validate the archived JSONL records named by ``descriptor``.

    ``archived_bytes`` are the exact persisted prefix bytes (the first
    ``prefix_end_byte`` bytes of ``messages.jsonl``). Any mismatch with the
    descriptor raises ``ArchiveIntegrityError`` — never a partial read.
    """
    return list(iter_archived_records(descriptor, (archived_bytes,)))


def hydrate_archived_messages(
    descriptor: ArchivedPrefixV1, archived_bytes: bytes
) -> list[LLMMessage]:
    """Hydrate the archived records into messages, validated as above.

    Uses the same ``LLMMessage.model_validate`` hydration as the trusted
    session loader, so an archived record round-trips exactly like a
    cold-resumed one.
    """
    return [
        LLMMessage.model_validate(record)
        for record in read_archived_records(descriptor, archived_bytes)
    ]


def archived_prefix_chunks(path: Path, limit: int) -> Iterator[bytes]:
    """Stream the first ``limit`` bytes of ``path`` in bounded chunks.

    The file is never held whole; the stream stops at the limit so the shared
    read interface sees exactly the archived prefix bytes.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError(f"limit must be a non-negative int, got {limit!r}")
    remaining = limit
    for chunk in read_chunks(path):
        if remaining <= 0:
            return
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        remaining -= len(chunk)
        yield chunk


def verify_archived_prefix_on_disk(descriptor: ArchivedPrefixV1, path: Path) -> None:
    """Prove the sealed prefix is durably on disk exactly as ``descriptor`` says.

    Streams the first ``prefix_end_byte`` bytes of the transcript at ``path``
    through the shared read interface above; any byte-count, record-count,
    structure, or hash mismatch raises ``ArchiveIntegrityError`` — never a
    partial verification. Sealing consumes this to issue its receipt only
    after the durable persist.
    """
    try:
        for _record in iter_archived_records(
            descriptor, archived_prefix_chunks(path, descriptor.prefix_end_byte)
        ):
            pass
    except OSError as exc:
        raise ArchiveIntegrityError(
            f"cannot read the archived prefix at {path}: {exc}"
        ) from exc
