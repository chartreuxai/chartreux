"""P4c1 test support: inject a sealed archive prefix into a live session.

Rewind/fork hydration (M1 §3) is reached by INJECTING an archive descriptor
plus the trimmed RAM copies of its sealed messages directly — the packages
here never depend on P4b's replacement model or on P4d's trim activation.
The descriptor is built from real transcript bytes, so the shared archive
read interface validates what production would have sealed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.transcript_archive import ArchivedPrefixV1


def record_line(message: LLMMessage) -> bytes:
    """One persisted transcript line, exactly as the save path writes it."""
    return (
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_transcript(path: Path, messages: list[LLMMessage]) -> None:
    path.write_bytes(b"".join(record_line(message) for message in messages))


def descriptor_for_prefix(
    transcript_path: Path,
    archived_count: int,
    *,
    session_id: str,
    message_epoch: int,
    trim_mode: str = "payloads",
) -> ArchivedPrefixV1:
    """The descriptor of the first ``archived_count`` records of a transcript."""
    lines = [line + b"\n" for line in transcript_path.read_bytes().split(b"\n")[:-1]]
    assert len(lines) >= archived_count
    prefix = b"".join(lines[:archived_count])
    return ArchivedPrefixV1(
        session_id=session_id,
        message_epoch=message_epoch,
        archived_count=archived_count,
        prefix_end_byte=len(prefix),
        prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        boundary_message_id=None,
        trim_mode=trim_mode,  # type: ignore[arg-type]
    )


def trim_prefix_payloads(messages: MessageList, archived_count: int) -> None:
    """Drop ``reasoning_payloads`` from the sealed slots (M1 ``payloads`` trim).

    Length-preserving, so absolute indexes stay valid; the transcribed message
    identity (role, message_id) is untouched.
    """
    sealed = 0
    for index, message in enumerate(list(messages)):
        if message.role is Role.system:
            continue
        if sealed >= archived_count:
            break
        messages.replace(index, message.model_copy(update={"reasoning_payloads": None}))
        sealed += 1


def tombstone_prefix(messages: MessageList, archived_count: int) -> None:
    """Replace the sealed slots with payload-empty tombstones (M1 ``tombstones``).

    Length-preserving and identity-preserving (role, message_id, injected),
    every payload graph released — the shape a rewind/fork must never hand on.
    """
    sealed = 0
    for index, message in enumerate(list(messages)):
        if message.role is Role.system:
            continue
        if sealed >= archived_count:
            break
        messages.replace(
            index,
            LLMMessage(
                role=message.role,
                message_id=message.message_id,
                injected=message.injected,
            ),
        )
        sealed += 1
