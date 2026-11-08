"""S1-A1: the request version, the mutation lock order, and the envelope.

The settled contract under test: every retained-content mutation of a
``MessageList`` bumps one version counter under a separate mutation lock that
is ordered AFTER ``_lock``; ``freeze_for_request`` returns the snapshot, the
version and the epoch as one atomic triple; the coordinator mints a frozen
request envelope per attempt carrying the source identity, that triple and the
model key; unprovenanced sequences keep the naive interface; a refused
mutation never bumps.
"""

from __future__ import annotations

import threading

import pytest

from chartreux.core.agent_loop._completion_coordinator import CompletionCoordinator
from chartreux.core.agent_loop.llm_gateway import messages_for_backend
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm_models import LLMMessage, Role, archived_replacement
from chartreux.core.message_list import (
    MessageList,
    RequestEnvelope,
    freeze_request_provenance,
)
from chartreux.core.session.transcript_archive import ArchivedPrefixV1
from tests.conftest import build_test_vibe_config


def _user(index: int) -> LLMMessage:
    return LLMMessage(role=Role.user, content=f"m{index}", message_id=f"m-{index}")


def _assistant(index: int) -> LLMMessage:
    return LLMMessage(role=Role.assistant, content=f"a{index}", message_id=f"a-{index}")


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


def _model(alias: str = "alias-x", *, supports_images: bool = True) -> ModelConfig:
    return ModelConfig(
        name="wire", provider="test", alias=alias, supports_images=supports_images
    )


def _coordinator() -> CompletionCoordinator:
    """A coordinator whose input build runs its real code.

    ``build_completion_inputs`` reads only the config getter, the session-id
    getter, the backend-metadata assembler and the max-tokens getter; the
    attempt path is outside this package and is never reached here.
    """
    config = build_test_vibe_config(
        providers=[ProviderConfig(name="test", api_base="https://test.invalid")]
    )
    coordinator = CompletionCoordinator.__new__(CompletionCoordinator)
    coordinator._config_getter = lambda: config
    coordinator._session_id_getter = lambda: "session-1"
    coordinator._backend_metadata = lambda call_type: {}
    coordinator._max_tokens_getter = lambda: None
    coordinator._last_request_envelope = None
    return coordinator


def test_message_list_version_bumps_on_all_six_mutation_classes() -> None:
    # The six mutation entry points of the settled contract each describe one
    # retained-content mutation and bump the version by exactly one. ``extend``
    # counts once here because it routes through ``append``; its per-append
    # count is pinned by its own node.
    messages = MessageList([_user(0), _assistant(0)])
    assert messages.message_version == 0

    # Reads never bump: iteration, freezing and the plain dunders.
    assert [message.message_id for message in messages] == ["m-0", "a-0"]
    messages.freeze_for_request()
    assert len(messages) == 2
    assert messages.message_version == 0

    messages.append(_user(1))
    assert messages.message_version == 1

    messages.insert(0, _user(2))
    assert messages.message_version == 2

    messages.replace(0, _user(3))
    assert messages.message_version == 3

    messages.extend([_assistant(1)])
    assert messages.message_version == 4

    descriptor = _descriptor(archived_count=1)
    messages.publish_archive_trim(
        descriptor, [archived_replacement(messages[0], descriptor)]
    )
    assert messages.message_version == 5

    messages.update_system_prompt("system prompt")
    assert messages.message_version == 6
    assert messages[0].role is Role.system

    # Reset/rebind is an epoch event, not a retained-content mutation: the
    # epoch moves and the freeze triple carries the new one, so a request
    # frozen before the rebind can never match after it.
    before = messages.freeze_for_request()
    messages.reset([_user(9)])
    after = messages.freeze_for_request()
    assert after[2] == before[2] + 1
    assert after[0] != before[0]


def test_message_list_version_failed_op_never_bumps() -> None:
    messages = MessageList([_user(0), _assistant(0)])
    before = messages.message_version

    with pytest.raises(IndexError):
        messages.replace(len(messages), _user(1))
    with pytest.raises(IndexError):
        messages.replace(-len(messages) - 1, _user(1))
    assert messages.message_version == before

    descriptor = _descriptor(archived_count=1)
    with pytest.raises(ValueError, match="exactly one slot per archived record"):
        messages.publish_archive_trim(descriptor, [])
    oversized = _descriptor(archived_count=5)
    with pytest.raises(ValueError, match="the list holds only"):
        messages.publish_archive_trim(
            oversized, [archived_replacement(messages[0], oversized)] * 5
        )
    stale = _descriptor(archived_count=1, message_epoch=7)
    with pytest.raises(ValueError, match="does not match the list epoch"):
        messages.publish_archive_trim(stale, [archived_replacement(messages[0], stale)])
    assert messages.message_version == before
    assert [message.message_id for message in messages] == ["m-0", "a-0"]

    # The post-retention stamp is a mutation too, so a stamp that finds no
    # eligible prose owner mutates nothing and bumps nothing.
    assert messages.stamp_turn_duration("missing", 1.5) is False
    assert messages.stamp_turn_duration("m-0", 1.5) is False  # not an assistant
    content_free = LLMMessage(role=Role.assistant, content="", message_id="a-empty")
    messages.append(content_free)
    stamped = messages.message_version
    assert messages.stamp_turn_duration("a-empty", 1.5) is False  # no prose
    assert messages.message_version == stamped
    assert content_free.turn_duration is None


def test_message_list_version_freeze_atomic_under_concurrent_writers() -> None:
    # Every bump describes exactly one structural change: a version separated
    # from its content swap would let a freeze pair fresh messages with a stale
    # version (or the reverse). The writer replaces slot 0 with the message
    # whose id encodes the version it will carry, so any tearing is visible.
    messages = MessageList([_user(0), _assistant(0)])
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for index in range(1, 5001):
                messages.replace(0, _user(index))
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    worker = threading.Thread(target=writer)
    worker.start()
    observations = 0
    previous_version = -1
    try:
        while not stop.is_set() or previous_version < 5000:
            snapshot, version, epoch = messages.freeze_for_request()
            assert len(snapshot) == 2
            assert snapshot[0].message_id == f"m-{version}"
            assert snapshot[1].message_id == "a-0"
            assert version >= previous_version
            assert epoch == messages.message_epoch == 0
            previous_version = version
            observations += 1
            if observations > 100_000:  # pragma: no cover - runaway guard
                break
    finally:
        stop.set()
        worker.join()

    assert errors == []
    assert observations > 0
    assert messages.message_version >= previous_version

    # The post-retention in-place write is a mutation with no structural
    # change: the snapshot stays identical while the version moves, so a
    # request frozen before the stamp can never be served after it.
    retained = MessageList([_user(0), _assistant(0)])
    frozen_before, version_before, _ = retained.freeze_for_request()
    assert retained.stamp_turn_duration("a-0", 12.5) is True
    frozen_after, version_after, _ = retained.freeze_for_request()
    assert version_after == version_before + 1
    assert frozen_after == frozen_before
    assert retained[1].turn_duration == 12.5


def test_message_list_version_lock_order_liveness() -> None:
    # The mutation lock is ordered AFTER ``_lock``: a ``_lock`` holder may take
    # it, and the freeze takes it alone. The two writer classes and the freeze
    # therefore interleave without a cycle — a freeze that reached back for
    # ``_lock`` while holding the mutation lock would deadlock here.
    messages = MessageList([_user(0)])
    errors: list[BaseException] = []
    iterations = 500

    def prompt_writer() -> None:
        try:
            for _ in range(iterations):
                messages.update_system_prompt("prompt")
        except BaseException as exc:
            errors.append(exc)

    def rebinder() -> None:
        try:
            for _ in range(iterations):
                messages.reset_preserving_system([_user(1)])
        except BaseException as exc:
            errors.append(exc)

    writers = [
        threading.Thread(target=prompt_writer),
        threading.Thread(target=rebinder),
    ]
    for worker in writers:
        worker.start()
    try:
        for _ in range(iterations * 4):
            snapshot, _, epoch = messages.freeze_for_request()
            # The two writers only ever prepend a system prompt or rebind the
            # tail to one user message, so any consistent snapshot is short.
            assert len(snapshot) <= 2
            assert epoch == messages.message_epoch
    finally:
        for worker in writers:
            worker.join(timeout=2)

    assert [worker.is_alive() for worker in writers] == [False, False]
    assert errors == []
    assert messages[0].role is Role.system


def test_message_list_version_envelope_capture_via_production_method() -> None:
    # The envelope is minted by the production input build, per attempt, after
    # the candidate model is selected: source identity + version + epoch + the
    # model key the attempt resolved.
    model = _model(supports_images=True)
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()

    inputs = coordinator.build_completion_inputs(
        model=model, messages=messages, tools=None, tool_choice=None
    )

    envelope = coordinator.last_request_envelope
    assert isinstance(envelope, RequestEnvelope)
    assert envelope.source_identity == messages.source_identity
    assert envelope.version == messages.message_version == 0
    assert envelope.epoch == messages.message_epoch == 0
    assert envelope.model_alias == "alias-x"
    assert envelope.supports_images is True
    # The request carries the frozen snapshot, which is the same transcript the
    # naive projection consumes.
    assert inputs.messages == tuple(messages)
    assert [
        message.message_id for message in messages_for_backend(inputs.messages, model)
    ] == ["m-0", "a-0"]

    # A later attempt mints a different envelope: the capture is frozen, never
    # refreshed in place.
    messages.append(_user(1))
    coordinator.build_completion_inputs(
        model=model, messages=messages, tools=None, tool_choice=None
    )
    second = coordinator.last_request_envelope
    assert isinstance(second, RequestEnvelope)
    assert second.version == 1
    assert second.source_identity == envelope.source_identity
    assert envelope.version == 0

    # The model key belongs to the envelope: the same list state under another
    # model is another envelope.
    coordinator.build_completion_inputs(
        model=_model("alias-y", supports_images=False),
        messages=messages,
        tools=None,
        tool_choice=None,
    )
    other = coordinator.last_request_envelope
    assert isinstance(other, RequestEnvelope)
    assert other.model_alias == "alias-y"
    assert other.supports_images is False
    assert other.version == second.version


def test_message_list_version_naive_path_preserved_for_plain_sequences() -> None:
    plain = [_user(0), _assistant(0)]
    frozen, envelope = freeze_request_provenance(
        plain, model_alias="alias-x", supports_images=False
    )
    assert envelope is None
    assert frozen == tuple(plain)
    assert frozen is not plain

    # The naive interface survives the production path: an unprovenanced
    # sequence keeps the plain snapshot and reports no envelope, and a build
    # over it does not leave an earlier envelope looking current.
    coordinator = _coordinator()
    owned = MessageList([_user(0)])
    coordinator.build_completion_inputs(
        model=_model(), messages=owned, tools=None, tool_choice=None
    )
    assert coordinator.last_request_envelope is not None
    naive_inputs = coordinator.build_completion_inputs(
        model=_model(), messages=plain, tools=None, tool_choice=None
    )
    assert coordinator.last_request_envelope is None
    assert naive_inputs.messages == tuple(plain)

    # A MessageList afterwards freezes normally again: no state leaks from the
    # naive path.
    frozen_owned, owned_envelope = freeze_request_provenance(
        owned, model_alias="alias-x", supports_images=False
    )
    assert isinstance(owned_envelope, RequestEnvelope)
    assert frozen_owned == (owned[0],)
    assert owned_envelope.version == owned.message_version


def test_message_list_version_bumps_per_append_on_extend() -> None:
    messages = MessageList([_user(0)])
    before = messages.message_version
    batch = [_user(1), _assistant(1), _user(2)]

    messages.extend(batch)

    # Each appended message is its own retained-content mutation, so a
    # partially applied extend is visible as such.
    assert messages.message_version == before + len(batch)
    assert [message.message_id for message in messages] == ["m-0", "m-1", "a-1", "m-2"]

    # An empty extend is no mutation at all.
    unchanged = messages.message_version
    messages.extend([])
    assert messages.message_version == unchanged
