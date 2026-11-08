"""S1-A2: exact-snapshot memoization of the request preparation.

The settled contract under test: the per-loop gateway keeps ONE slot holding
the last successfully prepared request projection, keyed by (source identity,
captured version, captured epoch, model alias, supports_images); a hit is
authorized only after re-reading the LIVE version/epoch from the source
MessageList at service time, with the key read, the live re-read and the value
publish in one synchronous stretch (no await between check and publish);
failures are never cached; on ANY doubt the naive path runs. The carrier is
populated by the real coordinator mint site (``build_completion_inputs``), so
the production path hits end-to-end — proven by the replay test driving the
real coordinator + gateway.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import threading

import pytest

from chartreux.core.agent_loop._completion_coordinator import CompletionCoordinator
from chartreux.core.agent_loop._loop import AgentLoop, requires_init
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
    messages_for_backend,
)
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm_models import LLMMessage, Role, archived_replacement
from chartreux.core.message_list import MessageList
from chartreux.core.session.transcript_archive import ArchivedPrefixV1
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.secret_redaction import ScrubPolicy
from tests.conftest import build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _user(index: int) -> LLMMessage:
    return LLMMessage(role=Role.user, content=f"m{index}", message_id=f"m-{index}")


def _assistant(index: int) -> LLMMessage:
    return LLMMessage(role=Role.assistant, content=f"a{index}", message_id=f"a-{index}")


def _model(alias: str = "alias-x", *, supports_images: bool = True) -> ModelConfig:
    return ModelConfig(
        name="wire", provider="test", alias=alias, supports_images=supports_images
    )


def _coordinator() -> CompletionCoordinator:
    """A coordinator whose input build runs its real production code.

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


def _build(
    coordinator: CompletionCoordinator,
    messages: Sequence[LLMMessage],
    model: ModelConfig | None = None,
) -> CompletionInputs:
    """Mint the inputs through the real production build (the carrier site)."""
    return coordinator.build_completion_inputs(
        model=model or _model(), messages=messages, tools=None, tool_choice=None
    )


def _resources(backend: FakeBackend) -> CallResources:
    return CallResources(
        backend=backend, stats=AgentStats(), process_message=lambda m: m
    )


def _spy_prep(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the naive preparations the gateway actually performs."""
    calls: list[int] = []
    real = messages_for_backend

    def counting(
        messages: Sequence[LLMMessage], active_model: ModelConfig
    ) -> Sequence[LLMMessage]:
        calls.append(1)
        return real(messages, active_model)

    monkeypatch.setattr(
        "chartreux.core.agent_loop.llm_gateway.messages_for_backend", counting
    )
    return calls


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


def _mutate(messages: MessageList, kind: str) -> None:
    """One mutation event of each class the version counts."""
    if kind == "append":
        messages.append(_user(10))
    elif kind == "insert":
        messages.insert(0, _user(10))
    elif kind == "replace":
        messages.replace(0, _user(10))
    elif kind == "extend":
        messages.extend([_assistant(10), _user(11)])
    elif kind == "archive_trim":
        descriptor = _descriptor(1)
        messages.publish_archive_trim(
            descriptor, [archived_replacement(messages[0], descriptor)]
        )
    elif kind == "system_prompt":
        messages.update_system_prompt("a system prompt")
    elif kind == "stamp":
        assert messages.stamp_turn_duration("a-0", 1.5) is True
    elif kind == "reset":
        messages.reset([_user(20)])
    else:  # pragma: no cover - guarded by the parametrize list
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "mutation",
    [
        "append",
        "insert",
        "replace",
        "extend",
        "archive_trim",
        "system_prompt",
        "stamp",
        "reset",
    ],
)
def test_request_prep_memo_hit_matches_naive_after_every_mutation_class(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Byte-identical successful outputs AND identical typed failures across
    # cache-hit vs naive, exercised after EVERY mutation event class: the
    # first service prepares naively; the replay of the same inputs is a HIT
    # serving the identical projection without recomputation (or re-raising
    # the identical typed failure, never cached); after a mutation, a fresh
    # build mints a new envelope whose moved key MISSES and recomputes, and
    # ITS replay hits. (archive_trim's replacement carries an archive marker,
    # so that class's post-mutation projection fails closed in the naive path
    # too — the memo must reproduce that typed failure identically.)
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()
    gateway = LLMGateway()

    def naive_outcome(inputs: CompletionInputs) -> tuple[str, object]:
        try:
            return (
                "ok",
                [m.model_dump() for m in messages_for_backend(inputs.messages, model)],
            )
        except Exception as error:
            return ("raise", (type(error), str(error)))

    def service_twice(
        inputs: CompletionInputs,
    ) -> tuple[object, object, tuple[str, object]]:
        outcome = naive_outcome(inputs)
        if outcome[0] == "ok":
            first = gateway._prepare_backend_messages(inputs)
            second = gateway._prepare_backend_messages(inputs)
            dump = [m.model_dump() for m in first]
            assert dump == outcome[1]
            assert [m.model_dump() for m in second] == outcome[1]
            return first, second, outcome
        error_type, error_text = outcome[1]  # type: ignore[misc]
        with pytest.raises(error_type) as first_error:
            gateway._prepare_backend_messages(inputs)
        with pytest.raises(error_type) as second_error:
            gateway._prepare_backend_messages(inputs)
        assert str(first_error.value) == str(second_error.value) == error_text
        return None, None, outcome

    inputs = _build(coordinator, messages, model)
    first, second, outcome = service_twice(inputs)
    added = len(calls)
    if outcome[0] == "ok":
        assert second is first  # HIT: served from the slot, no recomputation
        assert added == 1
    else:
        assert added == 2  # failures never cached: both services recomputed

    _mutate(messages, mutation)

    inputs2 = _build(coordinator, messages, model)
    first2, second2, outcome2 = service_twice(inputs2)
    assert len(calls) - added == (1 if outcome2[0] == "ok" else 2)
    if outcome2[0] == "ok":
        assert second2 is first2  # the fresh envelope's replay HITS
        if outcome[0] == "ok":
            assert first2 is not first  # moved key: a MISS, not the stale slot
        assert list(first2) == list(  # type: ignore[arg-type]
            messages_for_backend(inputs2.messages, model)
        )


def test_request_prep_memo_in_place_mutation_between_attempts_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The post-retention in-place stamp between two attempts of the SAME
    # inputs: the accessor bumps the version, so the second attempt MUST miss
    # and recompute — never serve the stale slot.
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()
    gateway = LLMGateway()
    inputs = _build(coordinator, messages, model)
    prep = gateway._prepare_backend_messages(inputs)
    assert calls == [1]

    assert messages.stamp_turn_duration("a-0", 12.5) is True
    again = gateway._prepare_backend_messages(inputs)
    assert again is not prep
    assert calls == [1, 1]
    assert list(again) == list(messages_for_backend(inputs.messages, model))

    inputs2 = _build(coordinator, messages, model)
    prep2 = gateway._prepare_backend_messages(inputs2)
    assert prep2 is not again
    assert gateway._prepare_backend_messages(inputs2) is prep2
    assert calls == [1, 1, 1]


def test_request_prep_memo_live_version_drift_after_capture_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Structural mutation after capture, same inputs object: the live re-read
    # sees the bump, so the memo check MISSES and the naive path recomputes.
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()
    gateway = LLMGateway()
    inputs = _build(coordinator, messages, model)
    prep = gateway._prepare_backend_messages(inputs)
    assert calls == [1]

    messages.append(_user(1))
    drifted = gateway._prepare_backend_messages(inputs)
    assert drifted is not prep
    assert calls == [1, 1]
    assert list(drifted) == list(messages_for_backend(inputs.messages, model))


def test_request_prep_memo_epoch_reset_between_attempts_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reset-mid-prepare: the epoch moves, so both the stale inputs (live epoch
    # != captured epoch) and a fresh build (moved key) miss and recompute.
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()
    gateway = LLMGateway()
    inputs = _build(coordinator, messages, model)
    prep = gateway._prepare_backend_messages(inputs)
    assert calls == [1]

    messages.reset([_user(20)])
    assert messages.message_epoch == 1
    drifted = gateway._prepare_backend_messages(inputs)
    assert drifted is not prep
    assert calls == [1, 1]

    inputs2 = _build(coordinator, messages, model)
    prep2 = gateway._prepare_backend_messages(inputs2)
    assert prep2 is not drifted
    assert gateway._prepare_backend_messages(inputs2) is prep2
    assert calls == [1, 1, 1]


def test_request_prep_memo_cross_thread_daemon_writer_lands_then_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The deferred-init daemon's writer shape: update_system_prompt from
    # ANOTHER thread after capture. Once the write lands, the next check MUST
    # miss — the version bump is a plain attribute write, visible to the live
    # re-read (CPython GIL), and no memo synchronization is needed beyond the
    # @requires_init fence plus this re-read.
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()
    gateway = LLMGateway()
    inputs = _build(coordinator, messages, model)
    prep = gateway._prepare_backend_messages(inputs)
    assert calls == [1]

    def daemon() -> None:
        messages.update_system_prompt("deferred init prompt")

    worker = threading.Thread(target=daemon)
    worker.start()
    worker.join()
    assert messages.message_version == 1

    landed = gateway._prepare_backend_messages(inputs)
    assert landed is not prep
    assert calls == [1, 1]

    inputs2 = _build(coordinator, messages, model)
    prep2 = gateway._prepare_backend_messages(inputs2)
    assert gateway._prepare_backend_messages(inputs2) is prep2
    assert calls == [1, 1, 1]


@pytest.mark.asyncio
async def test_request_prep_memo_loop_task_writer_between_attempts_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A different loop task mutating between the two attempts of ONE logical
    # call (the bounded empty replay reuses the same inputs): the replay's
    # memo check re-reads the live version, MISSES, and recomputes — never a
    # served stale result (the wire still carries the frozen snapshot).
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()
    gateway = LLMGateway()
    inputs = _build(coordinator, messages, model)

    class YieldingBackend(FakeBackend):
        async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0)  # let the writer task run mid-call
            return await super().complete(**kwargs)

    backend = YieldingBackend([
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content="answer")],
    ])

    async def writer() -> None:
        await asyncio.sleep(0)
        messages.append(_user(99))

    task = asyncio.create_task(writer())
    result = await gateway.complete(inputs, _resources(backend))
    await task
    assert result.message.content == "answer"
    assert len(backend.requests_messages) == 2
    # Two transport attempts, but the preparation ran TWICE: the replay
    # missed via the live re-read. Both attempts carried the frozen snapshot.
    assert calls == [1, 1]
    assert backend.requests_messages[0] == backend.requests_messages[1]
    assert [m.message_id for m in backend.requests_messages[1]] == ["m-0", "a-0"]


@pytest.mark.asyncio
async def test_request_prep_memo_production_path_hit_through_coordinator_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # END-TO-END production-path HIT: the real coordinator mint site
    # (build_completion_inputs) populates the carrier, and the bounded empty
    # replay inside gateway.complete reuses the SAME inputs — the replay's
    # preparation is a memo HIT on the production path, not a hand-built
    # envelope.
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages = MessageList([_user(0), _assistant(0)])
    coordinator = _coordinator()
    inputs = _build(coordinator, messages, model)
    assert inputs.live_source is not None
    assert inputs.live_source.source is messages
    assert inputs.live_source.envelope == coordinator.last_request_envelope

    backend = FakeBackend([
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content="answer")],
    ])
    gateway = LLMGateway()
    result = await gateway.complete(inputs, _resources(backend))
    assert result.message.content == "answer"
    assert len(backend.requests_messages) == 2
    # ONE preparation for TWO transport attempts: the replay HIT the memo.
    assert calls == [1]
    assert backend.requests_messages[0] == backend.requests_messages[1]


def test_request_prep_memo_unprovenanced_inputs_stay_naive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unprovenanced sequences (a plain list through the coordinator, or a
    # hand-built CompletionInputs like the utility-completion shape at
    # utility_completion.py) keep the naive interface: no carrier, no slot.
    calls = _spy_prep(monkeypatch)
    model = _model()
    coordinator = _coordinator()
    gateway = LLMGateway()
    plain = [_user(0), _assistant(0)]

    inputs = _build(coordinator, plain, model)
    assert inputs.live_source is None
    first = gateway._prepare_backend_messages(inputs)
    second = gateway._prepare_backend_messages(inputs)
    assert first is not second
    assert calls == [1, 1]
    assert gateway._prepared_slot is None

    direct = CompletionInputs(
        model=model,
        provider_name="test",
        emits_finish_reason=True,
        messages=tuple(plain),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=None,
    )
    assert direct.live_source is None
    gateway._prepare_backend_messages(direct)
    gateway._prepare_backend_messages(direct)
    assert calls == [1, 1, 1, 1]
    assert gateway._prepared_slot is None


def test_request_prep_memo_single_slot_replaced_and_model_fingerprint_keyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Single-slot semantics + the model fingerprint in the key: a newer source
    # replaces the slot, and another alias / supports_images is another key.
    calls = _spy_prep(monkeypatch)
    model = _model()
    messages_a = MessageList([_user(0)])
    messages_b = MessageList([_user(1)])
    coordinator = _coordinator()
    gateway = LLMGateway()

    inputs_a = _build(coordinator, messages_a, model)
    prep_a = gateway._prepare_backend_messages(inputs_a)
    inputs_b = _build(coordinator, messages_b, model)
    gateway._prepare_backend_messages(inputs_b)
    again_a = gateway._prepare_backend_messages(inputs_a)
    assert again_a is not prep_a
    assert calls == [1, 1, 1]

    other_model = _model("alias-y")
    inputs_y = _build(coordinator, messages_a, other_model)
    prep_y = gateway._prepare_backend_messages(inputs_y)
    assert prep_y is not again_a
    assert calls == [1, 1, 1, 1]

    no_images = _model(supports_images=False)
    inputs_ni = _build(coordinator, messages_a, no_images)
    prep_ni = gateway._prepare_backend_messages(inputs_ni)
    assert prep_ni is not prep_y
    assert calls == [1, 1, 1, 1, 1]
    assert gateway._prepare_backend_messages(inputs_ni) is prep_ni
    assert calls == [1, 1, 1, 1, 1]


@pytest.mark.asyncio
async def test_request_prep_memo_requires_init_fence_covers_every_completion_entry() -> (
    None
):
    # The production soundness fence, asserted rather than assumed: every
    # production completion entry is @requires_init-wrapped (its code object
    # is exactly the requires_init wrapper's), and the wrapper AWAITS deferred
    # init BEFORE the method body — so no production memo check can run while
    # the off-loop deferred-init daemon writer is still pending.
    entries = (
        "act",
        "inject_user_context",
        "compact",
        "relocate",
        "reload_with_initial_messages",
        "clear_history",
    )
    for name in entries:
        method = getattr(AgentLoop, name)
        raw = method.__wrapped__
        assert raw is not None and raw.__name__ == name, name
        assert requires_init(raw).__code__ is method.__code__, name

    class _FenceProbe:
        def __init__(self) -> None:
            self.scrub_policy = ScrubPolicy()
            self.order: list[str] = []

        async def wait_until_ready(self) -> None:
            self.order.append("ready")

        @requires_init
        async def entry(self) -> None:
            self.order.append("body")

    probe = _FenceProbe()
    await probe.entry()
    assert probe.order == ["ready", "body"]
