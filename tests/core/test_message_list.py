from __future__ import annotations

import threading

import pytest

from chartreux.core.agent_loop._transcript_integrity import archive_reference
from chartreux.core.llm_models import LLMMessage, Role, archived_replacement
from chartreux.core.message_list import MessageList
from chartreux.core.session.transcript_archive import ArchivedPrefixV1


def _descriptor(archived_count: int, message_epoch: int = 0) -> ArchivedPrefixV1:
    return ArchivedPrefixV1(
        session_id="session-1",
        message_epoch=message_epoch,
        archived_count=archived_count,
        prefix_end_byte=8 * archived_count,
        prefix_sha256="0" * 64,
        boundary_message_id="boundary-1",
        trim_mode="payloads",
    )


def test_update_system_prompt_replaces_existing_system_slot() -> None:
    messages = MessageList(
        initial=[
            LLMMessage(role=Role.system, content="old"),
            LLMMessage(role=Role.user, content="hi"),
        ]
    )

    messages.update_system_prompt("new")

    assert len(messages) == 2
    assert messages[0].role == Role.system
    assert messages[0].content == "new"
    assert messages[1].content == "hi"


def test_update_system_prompt_inserts_without_clobbering_when_no_system() -> None:
    messages = MessageList(
        initial=[
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]
    )

    messages.update_system_prompt("system prompt")

    assert len(messages) == 3
    assert messages[0].role == Role.system
    assert messages[1].content == "Hello"
    assert messages[2].content == "Hi there!"


def test_update_system_prompt_inserts_into_empty_list() -> None:
    messages = MessageList()

    messages.update_system_prompt("system prompt")

    assert len(messages) == 1
    assert messages[0].role == Role.system


def test_reset_preserving_system_keeps_system_and_replaces_tail() -> None:
    messages = MessageList(
        initial=[
            LLMMessage(role=Role.system, content="sys"),
            LLMMessage(role=Role.user, content="old user"),
        ]
    )

    tail = [
        LLMMessage(role=Role.user, content="loaded user"),
        LLMMessage(role=Role.assistant, content="loaded assistant"),
    ]
    messages.reset_preserving_system(tail)

    assert [m.role for m in messages] == [Role.system, Role.user, Role.assistant]
    assert messages[0].content == "sys"
    assert messages[1].content == "loaded user"
    assert messages[2].content == "loaded assistant"


def test_reset_preserving_system_without_existing_system() -> None:
    messages = MessageList(initial=[LLMMessage(role=Role.user, content="old")])

    tail = [LLMMessage(role=Role.user, content="loaded")]
    messages.reset_preserving_system(tail)

    assert [m.role for m in messages] == [Role.user]
    assert messages[0].content == "loaded"


def test_reset_preserving_system_is_safe_under_concurrent_prompt_updates() -> None:
    # Regression: on resume the deferred-init thread updates the system prompt
    # while the resume coroutine replaces the message tail. Without shared
    # locking the read-then-replace raced with insert(0) and could raise
    # "list changed size during iteration".
    messages = MessageList(initial=[LLMMessage(role=Role.user, content="old")])
    tail = [LLMMessage(role=Role.user, content="loaded")]
    errors: list[BaseException] = []
    stop = threading.Event()

    def hammer_prompt() -> None:
        try:
            while not stop.is_set():
                messages.update_system_prompt("prompt")
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=hammer_prompt)
    worker.start()
    try:
        for _ in range(2000):
            messages.reset_preserving_system(tail)
    finally:
        stop.set()
        worker.join()

    assert errors == []
    assert messages[0].role == Role.system
    assert messages[-1].content == "loaded"


def test_publish_archive_trim_replaces_same_slot_and_survives_list_extraction() -> None:
    # M1 replacement model: archived message copies carry the PRIVATE archive
    # reference, so converting the MessageList to a plain list does not lose
    # archived status.
    descriptor = _descriptor(archived_count=2)
    original_user = LLMMessage(role=Role.user, content="ask")
    original_assistant = LLMMessage(
        role=Role.assistant,
        content="answer",
        reasoning_payloads=[{"payload": "opaque-reasoning"}],
    )
    messages = MessageList([
        LLMMessage(role=Role.system, content="sys"),
        original_user,
        original_assistant,
    ])
    replacements = [
        archived_replacement(original_user, descriptor),
        archived_replacement(original_assistant, descriptor),
    ]

    messages.publish_archive_trim(descriptor, replacements)

    # Same-slot replacement: no slot inserted or deleted, system preserved.
    assert len(messages) == 3
    assert messages[0].role is Role.system
    assert messages.archive_descriptor is descriptor
    # list() extraction keeps the private reference on the copies.
    extracted = list(messages)
    assert archive_reference(extracted[0]) is None
    assert archive_reference(extracted[1]) is descriptor
    assert archive_reference(extracted[2]) is descriptor
    # The payloads trim released only the reasoning payload graph...
    assert extracted[2].reasoning_payloads is None
    assert extracted[2].content == "answer"
    # ...and the private marker never serializes.
    assert "_archive_reference" not in extracted[2].model_dump(mode="json")
    # The originals are never mutated.
    assert original_assistant.reasoning_payloads == [{"payload": "opaque-reasoning"}]
    assert archive_reference(original_assistant) is None


def test_publish_archive_trim_rejects_wrong_replacement_count() -> None:
    descriptor = _descriptor(archived_count=2)
    messages = MessageList([
        LLMMessage(role=Role.user, content="u1"),
        LLMMessage(role=Role.assistant, content="a1"),
    ])

    with pytest.raises(ValueError, match="exactly one slot per archived record"):
        messages.publish_archive_trim(
            descriptor, [archived_replacement(messages[0], descriptor)]
        )

    # A refused publication leaves the list and the owned descriptor untouched.
    assert messages.archive_descriptor is None
    assert archive_reference(messages[0]) is None
    assert messages[0].content == "u1"


def test_publish_archive_trim_rejects_stale_epoch_descriptor() -> None:
    descriptor = _descriptor(archived_count=1, message_epoch=7)
    messages = MessageList([LLMMessage(role=Role.user, content="u1")])

    with pytest.raises(ValueError, match="does not match the list epoch"):
        messages.publish_archive_trim(
            descriptor, [archived_replacement(messages[0], descriptor)]
        )

    assert messages.archive_descriptor is None
    assert messages[-1].content == "u1"


def test_iteration_is_stable_under_concurrent_prompt_insert() -> None:
    # Regression: during deferred init a reader iterates the list (e.g. a
    # role-filtered count) while the init thread inserts the system prompt at
    # index 0. __iter__ must snapshot so readers never see a duplicate, a
    # miss, or "list changed size during iteration".
    non_system = 50
    messages = MessageList(
        initial=[LLMMessage(role=Role.user, content=f"m{i}") for i in range(non_system)]
    )
    errors: list[BaseException] = []
    stop = threading.Event()

    def toggle_prompt() -> None:
        try:
            while not stop.is_set():
                messages.update_system_prompt("prompt")
                messages.reset([m for m in messages if m.role is not Role.system])
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=toggle_prompt)
    worker.start()
    try:
        for _ in range(2000):
            counted = sum(1 for m in messages if m.role is Role.user)
            assert counted == non_system
    finally:
        stop.set()
        worker.join()

    assert errors == []
