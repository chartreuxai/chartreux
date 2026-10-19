"""R0 — root-agent protection characterization (plan v0.5.6-recovery, package R0).

The OD2 finding: the ROOT agent already has more protection than the incident
narrative implied. This file characterizes the CURRENT tree's root-level
behavior against the R0 acceptance bullets. It is characterization only —
if a behavior fails, the finding is reported with a separately scoped repair
proposal; no failover policy is changed here.

Anchors (re-verified against the tree at 50a21b6):

- Every AgentLoop, root included, builds a CompletionCoordinator
  (``chartreux/core/agent_loop/_loop.py:610-663``); candidate selection has no
  subagent exclusion (``_completion_coordinator.py:276-299``).
- The availability registry is root-owned and shared with children
  (``chartreux/core/model_catalog/availability.py:90-91``;
  ``ConfigOrchestrator.copy`` passes ``availability_registry`` through).
- A user-initiated retry is the only turn start that clears the committed
  base's shared cooldowns (``_loop.py:1643-1646``; F6 four-path matrix in
  ``tests/recovery/test_injected_retry_flag.py``).
- Fresh-process root session resume restores the committed identity and fails
  closed on an unusable one (``chartreux/app_server/_runtime.py:614-680``,
  ``_resume_identity`` at ``:1815-1852``).

Coverage map (acceptance bullet -> tests):

1. Root retries and eligible same-base failover work BEFORE semantic output;
   NO transparent replay after streamed output -> ``TestRootFailoverBeforeOutput``.
2. Shared cooldown/probe behavior covers root/child interaction ->
   ``TestSharedCooldownRootChild``.
3. All-deployments-unavailable becomes a VISIBLE terminal state -> the
   app-server surface pin lives in ``tests/app_server/test_session_spawn.py``
   and the TUI pin in ``tests/cli/textual_ui/test_app_server_requests.py``.
4. Manual /retry is ONE injected new turn with cooldown-reset behavior
   characterized -> the TUI pin lives in
   ``tests/cli/textual_ui/test_app_server_requests.py``; the cooldown-reset
   matrix itself is WP0 F6's ``tests/recovery/test_injected_retry_flag.py``.
5. Incomplete-stream continuation stays bounded -> the TUI pin lives in
   ``tests/cli/textual_ui/test_app_server_requests.py``.
6. Fresh-process resume restores committed identity/state but NEVER callbacks
   or session-only grants -> ``TestFreshProcessResume``.
7. Same-endpoint alternatives are NEVER claimed as outage independence ->
   ``TestSameEndpointNotIndependent``.
8. Controlled transports + fake clocks only -> every test here uses
   ``FakeBackend``/``ControlledTransport`` (scripted, no network) and the
   opt-in ``fake_clock``/``controlled_transport`` fixtures from
   ``tests/recovery/conftest.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import json
from pathlib import Path
import time
from types import MethodType
from typing import cast

import httpx
import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._utils import public_error
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import (
    TurnErrorCode,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    ProtocolErrorCode,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop._loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig
from chartreux.core.dangerous_commands import RepositoryScope
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.model_catalog.availability import (
    AllDeploymentsUnavailableError,
    ExclusionReason,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_models import canonical_endpoint_identity
from chartreux.questions import UserAnswer
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import ControlledTransport, FakeClock, RetryableFault
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend

_MODEL_A = ModelConfig(name="model-a", provider="mistral", alias="model-a")
_MODEL_B = ModelConfig(name="model-b", provider="mistral", alias="model-b")


def _two_deployment_snapshot() -> CatalogSnapshot:
    """One base model with two deployments on DIFFERENT endpoints."""
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": "https://first.invalid"},
                "test-second": {"api_base": "https://second.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {"provider": "test-first", "name": "first"},
                        {"provider": "test-second", "name": "second"},
                    ]
                }
            },
            "roles": {},
        }),
        "test-revision",
    )


def _same_endpoint_snapshot() -> CatalogSnapshot:
    """Two providers on ONE shared endpoint, for two base models.

    ``base`` has one deployment per provider (same-endpoint alternatives);
    ``other`` adds the second base model on the same endpoint so cross-model
    rate-limit evidence can arm an endpoint-wide correlation cooldown.
    """
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-a": {"api_base": "https://shared.invalid"},
                "test-b": {"api_base": "https://shared.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {"provider": "test-a", "name": "base-a"},
                        {"provider": "test-b", "name": "base-b"},
                    ]
                },
                "other": {"deployments": [{"provider": "test-b", "name": "other-b"}]},
            },
            "roles": {},
        }),
        "test-revision",
    )


def _single_deployment_snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": "https://first.invalid"}},
            "models": {
                "base": {"deployments": [{"provider": "test-first", "name": "first"}]}
            },
            "roles": {},
        }),
        "test-revision",
    )


def _root_agent(
    snapshot: CatalogSnapshot, backend: FakeBackend, *, streaming: bool = False
) -> AgentLoop:
    """A ROOT agent loop (never a subagent) on the given catalog."""
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    return build_test_agent_loop(
        config=config, backend=backend, enable_streaming=streaming
    )


def _failover_backends(
    agent: AgentLoop, first: FakeBackend, second: FakeBackend
) -> None:
    def backend_for_attempt(
        _self: AgentLoop, model: ModelConfig, _budget: object
    ) -> FakeBackend:
        return {"test-first": first, "test-second": second}[model.provider]

    agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]


def _connect_backend_error(provider: str = "test-first") -> BackendError:
    """A production-faithful transport fault (D1/L): the BackendError carries
    its httpx cause, so the classifier honestly resolves CONNECTION and the
    deployment cooldown arms.
    """
    exc = BackendError(
        provider=provider,
        endpoint=f"https://{provider}.invalid/chat",
        status=None,
        reason="network boom",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="first",
        payload_summary=PayloadSummary(
            model="first",
            message_count=1,
            approx_chars=10,
            temperature=0,
            has_tools=False,
            tool_choice=None,
        ),
    )
    exc.__cause__ = httpx.ConnectError("network boom")
    return exc


async def _consume(events: AsyncIterator[object]) -> list[object]:
    return [event async for event in events]


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the server-side state")
        await asyncio.sleep(0.01)


def _assistant_contents(agent: AgentLoop) -> list[str]:
    return [
        message.content
        for message in agent.messages
        if message.role is Role.assistant and isinstance(message.content, str)
    ]


class TestRootFailoverBeforeOutput:
    """Acceptance 1: root retries and eligible same-base failover work
    BEFORE semantic output; NO transparent replay after streamed output.

    The OD2 key finding pinned end-to-end through the public root turn API
    (``AgentLoop.act``): same-base deployment failover is NOT child-only —
    the root loop's CompletionCoordinator offers the same-base alternatives.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_root_failover_before_output_serves_same_base_alternative(
        self, streaming: bool, controlled_transport: Callable[..., ControlledTransport]
    ) -> None:
        agent = _root_agent(
            _two_deployment_snapshot(), FakeBackend(), streaming=streaming
        )
        assert agent.is_subagent is False
        # A controlled transport records its attempt BEFORE applying the
        # scripted outcome, so the attempt counts are exact.
        first = controlled_transport([_connect_backend_error("test-first")])
        second = FakeBackend([mock_llm_chunk(content="ok")])
        _failover_backends(agent, first, second)
        mark = agent.completion_metadata_mark()

        await _consume(agent.act("hi"))

        # The root turn completed through the eligible same-base alternative:
        # exactly one assistant message, committed to the second deployment.
        assert _assistant_contents(agent) == ["ok"]
        committed = agent.committed_model
        assert committed is not None
        assert (committed.provider, committed.wire_name) == ("test-second", "second")
        # Both deployments saw exactly one attempt — the failed one and the
        # failover target; no retry storm, no duplicated output.
        assert len(first.requests_messages) == 1
        assert len(second.requests_messages) == 1
        metadata = agent.completion_metadata_since(mark)
        providers_used = cast(list[list[str]], metadata["providers_used"])
        assert providers_used == [["test-first", "test-second"]]
        notices = cast(list[dict[str, str]], metadata["switch_notices"])
        assert len(notices) == 1
        notice = notices[0]
        assert notice["base_model"] == "base"
        assert notice["old_provider"] == "test-first"
        assert notice["new_provider"] == "test-second"
        assert isinstance(notice["reason"], str) and notice["reason"]

    @pytest.mark.asyncio
    async def test_root_in_call_retry_succeeds_before_output(
        self, controlled_transport: Callable[..., ControlledTransport]
    ) -> None:
        """The in-call retry budget: a transient transport fault before any
        semantic output is retried inside the same logical call and the root
        turn still completes.
        """
        backend = controlled_transport([
            RetryableFault(_connect_backend_error("test-first")),
            [mock_llm_chunk(content="ok")],
        ])
        agent = _root_agent(_single_deployment_snapshot(), backend)

        await _consume(agent.act("hi"))

        # Two transport attempts, one logical call, one assistant message: the
        # retry happened before semantic output, invisibly to the turn.
        assert len(backend.requests_messages) == 2
        assert _assistant_contents(agent) == ["ok"]
        committed = agent.committed_model
        assert committed is not None
        assert committed.provider == "test-first"

    @pytest.mark.asyncio
    async def test_root_never_replays_after_streamed_output(
        self, controlled_transport: Callable[..., ControlledTransport]
    ) -> None:
        """NO transparent replay after streamed output: once content has
        streamed, a mid-stream failure surfaces as an error — the partial
        output stays, the identity is unchanged, the cooldown arms, and the
        same-base alternative is NOT silently used to replay.
        """
        agent = _root_agent(_two_deployment_snapshot(), FakeBackend(), streaming=True)
        original_identity = agent.committed_model

        # One attempt: stream the partial content, then fault mid-stream.
        first = controlled_transport([
            [
                mock_llm_chunk(content="partial", stop_reason=None),
                _connect_backend_error("test-first"),
            ]
        ])
        second = FakeBackend([mock_llm_chunk(content="never")])
        _failover_backends(agent, first, second)
        mark = agent.completion_metadata_mark()

        with pytest.raises(RuntimeError, match="API error"):
            await _consume(agent.act("hi"))

        # The same-base alternative was NOT attempted: no transparent replay
        # after streamed output.
        assert len(second.requests_messages) == 0
        assert len(first.requests_messages) == 1
        # The partial output is preserved as the turn's transcript tail.
        assert agent.messages[-1].role is Role.assistant
        assert agent.messages[-1].content == "partial"
        # The committed identity did not move to the alternative.
        assert agent.committed_model == original_identity
        assert agent.committed_model is not None
        assert agent.committed_model.provider == "test-first"
        # No failover switch was recorded for the interrupted attempt.
        assert agent.completion_metadata_since(mark)["switch_notices"] == []
        # The interrupted deployment's cooldown armed (shared registry).
        registry = agent.config_orchestrator.availability_registry
        assert registry.cooldown_until("base", "test-first") is not None


class TestSharedCooldownRootChild:
    """Acceptance 2: shared cooldown/probe behavior covers root/child
    interaction — the availability registry is root-owned and shared with
    children (``AvailabilityRegistry`` docstring; ``ConfigOrchestrator.copy``
    passes the registry through to child orchestrators).
    """

    def _single_model_config(self) -> ChartreuxConfigSchema:
        """A one-base-model catalog so the root and its ``worker`` child
        commit the SAME base model and share its cooldown key.
        """
        from tests.conftest import single_model_catalog

        snapshot = CatalogSnapshot(single_model_catalog(), "test-fixture")
        return ChartreuxConfigSchema.model_validate(
            {"active_model": "solo-model"}, context={"catalog_snapshot": snapshot}
        ).attach_catalog_snapshot(snapshot)

    @pytest.mark.asyncio
    async def test_child_failure_cooldown_blocks_root_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Child -> root: a child turn's transport failure arms the SHARED
        cooldown, and the root's next turn is blocked at admission — zero root
        transport attempts, AllDeploymentsUnavailable naming the cooldown.
        """
        config = self._single_model_config()
        parent_backend = FakeBackend([mock_llm_chunk(content="parent ok")])
        parent = build_test_agent_loop(config=config, backend=parent_backend)
        child_backend = FakeBackend(exception_to_raise=_connect_backend_error("solo"))
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
        )
        child = await AgentRuntimeFactory().create_child(parent, "worker")
        try:
            await child.wait_until_ready()
            # The child shares the root-owned registry object itself.
            assert (
                child.config_orchestrator.availability_registry
                is parent.config_orchestrator.availability_registry
            )
            committed = child.committed_model
            assert committed is not None
            assert committed.base_model == "solo-model"

            with pytest.raises(RuntimeError, match="API error"):
                await _consume(child.act("work"))

            registry = parent.config_orchestrator.availability_registry
            assert registry.cooldown_until("solo-model", "solo") is not None

            # The root's next turn is blocked at admission by the CHILD's
            # failure: no root transport attempt exists at all.
            with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
                await _consume(parent.act("hi"))
            assert exc_info.value.base_model == "solo-model"
            assert [
                (exclusion.provider, exclusion.reason)
                for exclusion in exc_info.value.exclusions
            ] == [("solo", ExclusionReason.COOLDOWN)]
            assert len(parent_backend.requests_messages) == 0
        finally:
            await child.aclose()
            await parent.aclose()

    @pytest.mark.asyncio
    async def test_root_probe_in_flight_blocks_child_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        controlled_transport: Callable[..., ControlledTransport],
        fake_clock: FakeClock,
    ) -> None:
        """Root -> child: the registry's single-flight recovery probe is
        shared. While the root's turn holds the probe in flight (cooldown
        expired, probe claimed), a child turn for the same base is NOT
        admitted — the probe is not double-booked across root and child.
        """
        config = self._single_model_config()
        parent_backend = controlled_transport([
            _connect_backend_error("solo"),
            [mock_llm_chunk(content="parent ok")],
        ])
        parent = build_test_agent_loop(
            config=config, backend=parent_backend, enable_streaming=True
        )
        registry = parent.config_orchestrator.availability_registry
        child_backend = FakeBackend([mock_llm_chunk(content="child ok")])
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
        )
        child = await AgentRuntimeFactory().create_child(parent, "worker")
        try:
            await child.wait_until_ready()

            # Turn 1 fails and arms the shared cooldown.
            with pytest.raises(RuntimeError, match="API error"):
                await _consume(parent.act("one"))
            assert registry.cooldown_until("solo-model", "solo") is not None

            # Move the registry onto the fake clock past the cooldown; turn 2
            # claims the single-flight probe and gates in flight.
            fake_clock.value = time.monotonic()
            registry._clock = fake_clock
            fake_clock.advance(31.0)
            started, release = parent_backend.gate_attempt(1)
            turn_two = asyncio.create_task(_consume(parent.act("two")))
            await asyncio.wait_for(started.wait(), timeout=5)

            # Read-only check: the probe is held by the root's in-flight turn,
            # so the shared key is unavailable to anyone else right now.
            assert registry.is_available("solo-model", "solo") is False

            # The child's turn for the same base is blocked at admission by
            # the root's in-flight probe: zero child transport attempts.
            with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
                await _consume(child.act("work"))
            assert [
                (exclusion.provider, exclusion.reason)
                for exclusion in exc_info.value.exclusions
            ] == [("solo", ExclusionReason.COOLDOWN)]
            assert len(child_backend.requests_messages) == 0

            # The root's turn completes and its success clears the cooldown.
            release.set()
            await asyncio.wait_for(turn_two, timeout=5)
            assert _assistant_contents(parent) == ["parent ok"]
            assert registry.cooldown_until("solo-model", "solo") is None
        finally:
            await child.aclose()
            await parent.aclose()


class TestFreshProcessResume:
    """Acceptance 6: fresh-process resume restores committed identity/state
    but NEVER callbacks or session-only grants.

    The fresh-process open is the blueprint resume
    (``_runtime.py:614-680``): a new AgentLoop is built from the persisted
    session. Callbacks live in the per-process turns service and session-only
    grants live in the per-process in-memory stores, so neither can survive;
    the stored committed identity fails closed when unusable
    (``_resume_identity``, ``_runtime.py:1815-1852``).
    """

    def _saved_session_config(self, tmp_path: Path) -> ChartreuxConfigSchema:
        return build_test_vibe_config(
            active_model="model-b",
            models=[_MODEL_A, _MODEL_B],
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
        )

    async def _persist_target_session(self, tmp_path: Path) -> str:
        saved = build_test_agent_loop(config=self._saved_session_config(tmp_path))
        saved.messages.append(LLMMessage(role=Role.user, content="target transcript"))
        saved.messages.append(LLMMessage(role=Role.assistant, content="target answer"))
        await saved.persist_empty_session()
        target_session_id = saved.session_id
        await saved.aclose()
        return target_session_id

    @pytest.mark.asyncio
    async def test_resume_restores_committed_identity_and_state(
        self, tmp_path: Path
    ) -> None:
        target_session_id = await self._persist_target_session(tmp_path)

        # A fresh process: a new loop (fresh registry, fresh stores) opens the
        # persisted session by id.
        resumed_loop = build_test_agent_loop(
            config=self._saved_session_config(tmp_path)
        )
        session = await attach_test_app_server_session(
            start_test_app_server(resumed_loop), resume_session_id=target_session_id
        )
        try:
            assert resumed_loop.session_id == target_session_id
            # Committed identity restored.
            committed = resumed_loop.committed_model
            assert committed is not None
            assert committed.base_model == "model-b"
            assert committed.provider == "mistral"
            assert committed.wire_name == "model-b"
            # Conversation state restored (the system prompt rides along at
            # the head of the message list; the transcript follows it).
            transcript = [
                message.content
                for message in resumed_loop.messages
                if message.role in (Role.user, Role.assistant)
            ]
            assert transcript == ["target transcript", "target answer"]
        finally:
            await session.close()
            await resumed_loop.aclose()

    @pytest.mark.asyncio
    async def test_resume_fails_closed_on_unusable_committed_identity(
        self, tmp_path: Path
    ) -> None:
        """The stored identity fails closed: a fresh process whose catalog
        lacks the committed model refuses the resume with an actionable error
        instead of loading the session unpinned.
        """
        target_session_id = await self._persist_target_session(tmp_path)

        resuming_config = build_test_vibe_config(
            models=[_MODEL_A],
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
        )
        resumed_loop = build_test_agent_loop(config=resuming_config)
        with pytest.raises(AppServerResponseError) as exc_info:
            await attach_test_app_server_session(
                start_test_app_server(resumed_loop), resume_session_id=target_session_id
            )
        assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        assert "model-b" in exc_info.value.error.message
        await resumed_loop.aclose()

    @pytest.mark.asyncio
    async def test_resume_never_restores_callbacks_or_session_grants(
        self, tmp_path: Path
    ) -> None:
        """Process 1 leaves a question callback OPEN and holds session-only
        grants (a root grant and a dangerous-command session grant). The
        fresh process resumes the persisted session and gets NONE of them:
        no callback state, the old callback id is unknown, and both grant
        stores start empty.
        """
        target_session_id = await self._persist_target_session(tmp_path)

        # --- Process 1: open a question callback and hold session grants. ---
        question_payload = {
            "question": "Authorize this?",
            "options": [{"label": "Yes"}, {"label": "No"}],
        }
        question_chunk = mock_llm_chunk(
            content="",
            tool_calls=[
                ToolCall(
                    id="question-1",
                    index=0,
                    function=FunctionCall(
                        name="ask_user_question",
                        arguments=json.dumps({"questions": [question_payload]}),
                    ),
                )
            ],
        )
        process1_backend = FakeBackend([question_chunk])
        process1_config = build_test_vibe_config(
            active_model="model-b",
            models=[_MODEL_A, _MODEL_B],
            enabled_tools=["ask_user_question"],
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
        )
        process1_loop = build_test_agent_loop(
            config=process1_config,
            backend=process1_backend,
            enable_streaming=True,
            user_input_capability=True,
        )
        client_transport, server_transport = memory_transport_pair()
        server1 = build_test_app_server(process1_loop, server_transport)
        session1 = await AppServerSession.start(
            AppServerClient(client_transport, run_peer=server1.serve),
            client_info=ClientInfo(name="r0-resume-p1", version="0"),
            capabilities=ClientCapabilities(callback_kinds=["user_input"]),
            resume_session_id=target_session_id,
        )
        act_task: asyncio.Task[list[object]] | None = None
        try:
            act_task = asyncio.create_task(_consume(session1.act("ask me")))
            turns1 = legacy_backend(server1).session.turns
            await _wait_until(lambda: session1.state.active_callbacks != [])
            open_callbacks = list(turns1._callbacks)
            assert len(open_callbacks) == 1
            old_callback_id = open_callbacks[0]
            assert session1.state.active_callbacks[0].callback_id == old_callback_id

            # Session-only grants held by process 1's loop (in-memory only).
            assert process1_loop.apply_root_grant(tmp_path) is True
            assert process1_loop._session_root_grants.roots == {tmp_path.resolve()}
            process1_loop._dangerous_command_grants.grant_session({
                (
                    "publish",
                    RepositoryScope(git_dir=tmp_path / ".git", work_tree=tmp_path),
                )
            })
            assert (
                process1_loop._dangerous_command_grants.dangerous_command_authorities()
                != ()
            )
        finally:
            # Process death: the turn is abandoned mid-callback; nothing about
            # the in-flight callback or the grants is persisted.
            if act_task is not None:
                act_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await act_task
            await session1.close()
            await process1_loop.aclose()

        # --- Process 2: a fresh process resumes the persisted session. ---
        resumed_loop = build_test_agent_loop(
            config=self._saved_session_config(tmp_path)
        )
        client_transport2, server_transport2 = memory_transport_pair()
        server2 = build_test_app_server(resumed_loop, server_transport2)
        session2 = await AppServerSession.start(
            AppServerClient(client_transport2, run_peer=server2.serve),
            client_info=ClientInfo(name="r0-resume-p2", version="0"),
            capabilities=ClientCapabilities(callback_kinds=["user_input"]),
            resume_session_id=target_session_id,
        )
        try:
            # Identity and state restored... (process 1's abandoned turn
            # persisted its own messages after the restored transcript; the
            # persisted pair is restored in order at its head.)
            committed = resumed_loop.committed_model
            assert committed is not None
            assert committed.base_model == "model-b"
            transcript = [
                message.content
                for message in resumed_loop.messages
                if message.role in (Role.user, Role.assistant)
            ]
            assert transcript[:2] == ["target transcript", "target answer"]
            # ...but NEVER the callbacks: no callback state server- or
            # client-side, and the old callback id is unknown.
            assert legacy_backend(server2).session.turns._callbacks == {}
            assert session2.state.active_callbacks == []
            with pytest.raises(AppServerResponseError) as exc_info:
                await session2.respond_to_callback(
                    old_callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[
                                UserAnswer(question="Authorize this?", answer="Yes")
                            ]
                        )
                    ),
                )
            assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND
            assert "Callback not found" in exc_info.value.error.message
            # ...and NEVER the session-only grants: both stores start empty.
            assert resumed_loop._session_root_grants.roots == set()
            assert (
                resumed_loop._dangerous_command_grants.dangerous_command_authorities()
                == ()
            )
        finally:
            await session2.close()
            await resumed_loop.aclose()


class TestSameEndpointNotIndependent:
    """Acceptance 7: same-endpoint alternatives are NEVER claimed as outage
    independence. Failover between same-base deployments is permitted even
    when they share one endpoint (it is the same base), but an endpoint-wide
    correlation cooldown excludes ALL of them — the harness never treats the
    second deployment as independent of the first's endpoint outage — and the
    failover switch notice carries no independence claim.
    """

    @pytest.mark.asyncio
    async def test_same_base_failover_works_across_providers_on_one_endpoint(
        self, controlled_transport: Callable[..., ControlledTransport]
    ) -> None:
        """Baseline: same-base failover to a same-endpoint alternative IS
        permitted while the endpoint is healthy — the exclusion machinery
        keys on deployment identity, not on the endpoint alone.
        """
        agent = _root_agent(_same_endpoint_snapshot(), FakeBackend())

        def backend_for_attempt(
            _self: AgentLoop, model: ModelConfig, _budget: object
        ) -> FakeBackend:
            return {"test-a": first, "test-b": second}[model.provider]

        first = controlled_transport([_connect_backend_error("test-a")])
        second = FakeBackend([mock_llm_chunk(content="ok")])
        agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]

        await _consume(agent.act("hi"))

        assert _assistant_contents(agent) == ["ok"]
        committed = agent.committed_model
        assert committed is not None
        assert (committed.provider, committed.wire_name) == ("test-b", "base-b")
        assert len(first.requests_messages) == 1
        assert len(second.requests_messages) == 1

    @pytest.mark.asyncio
    async def test_endpoint_correlation_excludes_same_endpoint_alternatives(
        self,
    ) -> None:
        """An endpoint-wide correlation cooldown (cross-model rate-limit
        evidence on ONE canonical endpoint) excludes EVERY deployment of that
        endpoint — the root turn fails with AllDeploymentsUnavailable naming
        both same-endpoint alternatives, with zero transport attempts. The
        same-endpoint alternative does NOT rescue the turn: it is not outage
        independence.
        """
        agent = _root_agent(_same_endpoint_snapshot(), FakeBackend())
        backend = cast(FakeBackend, agent._injected_backend)
        registry = agent.config_orchestrator.availability_registry
        raw_endpoint = "https://shared.invalid"
        endpoint = canonical_endpoint_identity(raw_endpoint)
        assert endpoint == "shared.invalid"

        # Cross-model rate-limit evidence on the one shared endpoint: two
        # distinct base models, one identical structured signature. The raw
        # URL is canonicalized (to the host identity) before it keys the
        # correlation state.
        failure = FailureInfo(
            category=FailureCategory.RATE_LIMIT,
            status=429,
            error_code="rate_limit_exceeded",
        )
        registry.record_failure(
            "base", "test-a", failure, endpoint=raw_endpoint, signature="rl-1"
        )
        registry.record_failure(
            "other", "test-b", failure, endpoint=raw_endpoint, signature="rl-1"
        )
        assert registry.endpoint_correlation_active(endpoint) is True

        with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
            await _consume(agent.act("hi"))

        # Both same-endpoint deployments are excluded by the ENDPOINT-wide
        # cooldown (checked before the per-deployment chain), and the visible
        # error names the model and each deployment's exclusion reason.
        assert exc_info.value.base_model == "base"
        assert [
            (exclusion.provider, exclusion.wire_name, exclusion.reason)
            for exclusion in exc_info.value.exclusions
        ] == [
            ("test-a", "base-a", ExclusionReason.ENDPOINT_COOLDOWN),
            ("test-b", "base-b", ExclusionReason.ENDPOINT_COOLDOWN),
        ]
        assert len(backend.requests_messages) == 0
        surfaced = public_error(exc_info.value)
        assert surfaced.code == TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE
        assert "base-a" in surfaced.message
        assert "base-b" in surfaced.message
        assert "endpoint-cooldown" in surfaced.message

    @pytest.mark.asyncio
    async def test_failover_switch_notice_carries_no_independence_claim(self) -> None:
        """The switch notice (surfaced to the orchestrator via
        ``metadata.switch_notices``) records base model, old/new provider and
        reason — and nothing else. It never claims endpoint or outage
        independence for the alternative.
        """
        agent = _root_agent(_two_deployment_snapshot(), FakeBackend())
        first = FakeBackend(exception_to_raise=_connect_backend_error("test-first"))
        second = FakeBackend([mock_llm_chunk(content="ok")])
        _failover_backends(agent, first, second)
        mark = agent.completion_metadata_mark()

        await _consume(agent.act("hi"))

        notices = cast(
            list[dict[str, str]],
            agent.completion_metadata_since(mark)["switch_notices"],
        )
        assert len(notices) == 1
        notice = notices[0]
        # Exactly these four fields — no endpoint, independence, or outage
        # claim field exists on the notice.
        assert set(notice) == {"base_model", "old_provider", "new_provider", "reason"}
        rendered = json.dumps(notice).lower()
        assert "independen" not in rendered
        assert "outage" not in rendered
        assert "endpoint" not in rendered
