"""Protocol provenance for user-input callbacks.

``UserInputCallbackDetail.origin`` carries the trusted provenance of a
user-input request across the app-server boundary: the server fills it from
the core event, so a client can tell a model-originated question from a
runtime approval prompt without inspecting the question text.

The addition is FORWARD-compatible only. The wire models are strict
(``extra="forbid"``) and the field serializes as an explicit
``"origin": null`` rather than being omitted, so a client built against an
older protocol version rejects the new key. Client and server must therefore
run the same protocol version; no client-side test can make an older model
accept the key, which is why the compat note is pinned by the strictness test
below rather than by a round trip.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from typing import Literal

from acp.schema import ElicitationFormSessionMode
from pydantic import BaseModel, Field, ValidationError
import pytest

from chartreux.acp.agent import _build_elicitation_schema
from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server._turns import ChildQuestionRefusedError
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    OpenCallbackState,
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    DangerousCommandAuthorityReadResponse,
    DangerousCommandDenylistBlockWire,
    DangerousCommandGrantScopeWire,
    DangerousCommandGrantWire,
    DangerousCommandRevokeParams,
    InitializeParams,
    InitializeResponse,
    ServerInfo,
    remote_exec_attestation,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.dangerous_commands import (
    REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY,
    REMOTE_EXEC_OPERATION_ID,
    REMOTE_EXEC_SSH_TRUST_CAPABILITY,
)
from chartreux.core.events import BaseEvent, UserInputOrigin, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.questions import QuestionChoice, UserQuestion, UserQuestionRequest
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    build_test_app_server,
    create_test_app_server_session,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend


def _request() -> UserQuestionRequest:
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Ship it?",
                options=[QuestionChoice(label="Yes"), QuestionChoice(label="No")],
            )
        ]
    )


def _ask_question_call() -> ToolCall:
    return ToolCall(
        id="question-1",
        index=0,
        function=FunctionCall(
            name="ask_user_question",
            arguments=json.dumps({
                "questions": [
                    {
                        "question": "Ship it?",
                        "options": [{"label": "Yes"}, {"label": "No"}],
                    }
                ]
            }),
        ),
    )


def _push_call() -> ToolCall:
    return ToolCall(
        id="push-1",
        index=0,
        function=FunctionCall(
            name="bash", arguments=json.dumps({"command": "git push"})
        ),
    )


def _callback_entry(detail: UserInputCallbackDetail) -> PublicCallbackEntry:
    return PublicCallbackEntry(
        id="entry-1",
        session_id="session-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        callback_id="callback-1",
        title="User input required",
        detail=detail,
        state=OpenCallbackState(),
    )


def _elicitation_mode(detail: UserInputCallbackDetail) -> ElicitationFormSessionMode:
    return ElicitationFormSessionMode(
        session_id="session-1",
        tool_call_id=detail.related_entry_id,
        requested_schema=_build_elicitation_schema(detail.request),
    )


def test_origin_field_defaults_to_none() -> None:
    detail = UserInputCallbackDetail(request=_request())

    assert "origin" in UserInputCallbackDetail.model_fields
    assert detail.origin is None


def test_wire_serialization_emits_an_explicit_null_origin() -> None:
    plain_json = _callback_entry(
        UserInputCallbackDetail(request=_request())
    ).model_dump_json(by_alias=True)
    assert '"origin":null' in plain_json
    assert json.loads(plain_json)["detail"]["origin"] is None

    typed_json = _callback_entry(
        UserInputCallbackDetail(request=_request(), origin="model_question")
    ).model_dump_json(by_alias=True)
    assert json.loads(typed_json)["detail"]["origin"] == "model_question"


def test_old_payload_without_origin_still_parses() -> None:
    """Backward read: a payload from a server that never sent origin parses."""
    payload = {
        "kind": "user_input",
        "request": _request().model_dump(mode="json", by_alias=True),
        "relatedEntryId": "tool-1",
        "unattendedExpiry": None,
    }

    detail = validate_wire(UserInputCallbackDetail, payload)

    assert detail.origin is None
    assert detail.related_entry_id == "tool-1"


def test_strict_models_keep_the_addition_forward_compatible_only() -> None:
    """Compat note: only same-version clients accept the new key.

    An older client model has no ``origin`` field and inherits
    ``extra="forbid"``, so it rejects the new payload exactly as this test
    rejects an unknown key. Nothing on this side can change that; the
    requirement is a same-version client and server.
    """
    new_payload = {
        "kind": "user_input",
        "request": _request().model_dump(mode="json", by_alias=True),
        "relatedEntryId": "tool-1",
        "unattendedExpiry": None,
        "origin": "model_question",
    }
    assert validate_wire(UserInputCallbackDetail, new_payload).origin == (
        "model_question"
    )
    assert UserInputCallbackDetail.model_config.get("extra") == "forbid"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_wire(UserInputCallbackDetail, {**new_payload, "futureKey": None})


def test_acp_elicitation_request_is_unchanged_by_a_present_origin() -> None:
    plain = UserInputCallbackDetail(request=_request(), related_entry_id="tool-1")
    typed = UserInputCallbackDetail(
        request=_request(), related_entry_id="tool-1", origin="runtime_approval"
    )

    plain_mode = _elicitation_mode(plain).model_dump(mode="json", by_alias=True)
    typed_mode = _elicitation_mode(typed).model_dump(mode="json", by_alias=True)

    assert typed_mode == plain_mode
    assert "origin" not in json.dumps(typed_mode)
    assert typed.related_entry_id == plain.related_entry_id == "tool-1"


@pytest.mark.asyncio
async def test_model_question_callback_carries_model_question_origin() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_ask_question_call()])],
        [mock_llm_chunk(content="Done")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)
    callback: PublicCallbackEntry | None = None

    try:
        async for event in session.act("ask me"):
            if isinstance(event, CallbackRequested):
                callback = event.callback
                detail = callback.detail
                assert isinstance(detail, UserInputCallbackDetail)
                assert detail.origin == "model_question"
                assert detail.related_entry_id == "question-1"
                await session.respond_to_callback(
                    callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="Ship it?", answer="Yes")]
                        )
                    ),
                )
    finally:
        await session.close()
        await agent_loop.aclose()

    assert callback is not None


@pytest.mark.asyncio
async def test_runtime_approval_callback_carries_runtime_approval_origin(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_push_call()])],
        [mock_llm_chunk(content="Done")],
    ])
    agent_loop = build_test_agent_loop(
        backend=backend, enable_streaming=True, cwd=project, user_input_capability=True
    )
    session = await create_test_app_server_session(agent_loop)
    callback: PublicCallbackEntry | None = None

    try:
        async for event in session.act("Exercise an approval prompt"):
            if isinstance(event, CallbackRequested):
                callback = event.callback
                detail = callback.detail
                assert isinstance(detail, UserInputCallbackDetail)
                # The runtime approval origin is stamped by the shell policy,
                # not by the model's arguments.
                assert detail.request.questions[0].header == "Dangerous command"
                assert detail.origin == "runtime_approval"
                await session.respond_to_callback(
                    callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="grant", answer="Deny")]
                        )
                    ),
                )
    finally:
        await session.close()
        await agent_loop.aclose()

    assert callback is not None


@pytest.mark.asyncio
async def test_child_origin_model_question_refused_at_admission(tmp_path: Path) -> None:
    """D2: a child loop's model question never opens a user-input callback.

    The refusal must land at admission, so it is asserted where a callback
    could not be opened: the controller has neither an active turn nor a
    projector, and ``_open_callback`` refuses to run without both. A request
    that is still refused here, with its waiter settled, can only have been
    stopped before the callback path.

    The waiter is the production broker future the asking tool awaits, so the
    denial the model reads is the one asserted below — no hang, no silent
    no-op, and no authority state moved.
    """
    seeded_root = tmp_path / "seeded-root"
    seeded_root.mkdir()
    config = build_test_vibe_config(
        dangerous_command_grants_by_project={str(tmp_path): ["git.push"]}
    )
    agent_loop = build_test_agent_loop(config=config, user_input_capability=True)
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await AppServerSession.start(
        AppServerClient(client_transport, run_peer=server.serve),
        client_info=ClientInfo(name="child-origin-test", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    turns = legacy_backend(server).session.turns
    agent_loop.apply_root_grant(seeded_root)
    roots_before = set(agent_loop._session_root_grants.roots)
    denied_before = set(agent_loop._session_root_grants.denied_roots)
    approvals_before = dict(agent_loop.config.dangerous_command_grants_by_project)
    broker = agent_loop._request_broker
    queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
    broker.bind(queue)
    agent_loop._is_subagent = True
    waiter: asyncio.Task[BaseModel] | None = None
    try:
        waiter = asyncio.create_task(
            broker.request_user_input(
                _request(), "question-1", origin=UserInputOrigin.MODEL_QUESTION
            )
        )
        event = await asyncio.wait_for(queue.get(), timeout=5)
        assert isinstance(event, UserInputRequestEvent)
        assert event.origin is UserInputOrigin.MODEL_QUESTION
        assert turns._active_turn is None
        assert turns._projector is None

        await turns._handle_core_request(event)

        with pytest.raises(ChildQuestionRefusedError) as refused:
            await asyncio.wait_for(waiter, timeout=5)
        assert isinstance(refused.value, RuntimeError)
        assert "not available to subagents" in str(refused.value)
        assert "blocker" in str(refused.value)

        # No callback was opened and no root question or approval changed.
        assert turns._callbacks == {}
        assert not [
            entry for entry in turns._history if isinstance(entry, PublicCallbackEntry)
        ]
        assert seeded_root.resolve() in roots_before
        assert set(agent_loop._session_root_grants.roots) == roots_before
        assert set(agent_loop._session_root_grants.denied_roots) == denied_before
        assert approvals_before
        assert (
            dict(agent_loop.config.dangerous_command_grants_by_project)
            == approvals_before
        )
    finally:
        if waiter is not None and not waiter.done():
            waiter.cancel()
        broker.unbind(queue)
        await session.close()
        await agent_loop.aclose()


# --- S4b1b: strict-client compatibility of the remote-exec attestation entries
#
# The attestation entries (remote_exec/network_locality,
# remote_exec/ssh_trust_context) are REQUEST-SIDE only: they ride in
# initialize.params.capabilities.client_tools. Both directions of the strict-
# client rule are pinned here, each documented:
#   (a) old server / new client REQUEST: a pre-S4b1b server model — a closed
#       Literal without the new entries — rejects a new client's initialize
#       request that carries them. The wire change is forward-compatible only
#       in the other direction (an old request read by a new server), which is
#       pinned by the second node below.
#   (b) old client / new server RESPONSE: no response model echoes the
#       capability channel and every response model keeps extra="forbid", so a
#       strict old client rejects any new response key. The entries never
#       appear on the wire in the response direction at all.


class _LegacyClientCapabilities(ProtocolModel):
    """A pre-S4b1b server's view of the client_tools channel (closed Literal)."""

    callback_kinds: list[str] = Field(default_factory=list)
    client_tools: list[Literal["filesystem/read", "filesystem/write", "terminal"]] = (
        Field(default_factory=list)
    )
    disabled_notifications: list[str] = Field(default_factory=list)


class _LegacyInitializeParams(ProtocolModel):
    client_info: ClientInfo
    capabilities: _LegacyClientCapabilities = Field(
        default_factory=_LegacyClientCapabilities
    )


def _initialize_payload(client_tools: list) -> dict:
    return InitializeParams(
        client_info=ClientInfo(name="test", version="1"),
        capabilities=ClientCapabilities(client_tools=client_tools),
    ).model_dump(mode="json", by_alias=True)


_ATTESTING_TOOLS = [
    "terminal",
    REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY,
    REMOTE_EXEC_SSH_TRUST_CAPABILITY,
]


def test_s4b1b_strict_old_server_rejects_attestation_entries_in_the_request() -> None:
    """Direction (a), new client -> old server: the closed pre-S4b1b Literal
    fails validation on the attestation entries, while the new server model
    accepts the same payload and classifies it execution-bound.
    """
    with pytest.raises(ValidationError):
        validate_wire(_LegacyInitializeParams, _initialize_payload(_ATTESTING_TOOLS))

    parsed = validate_wire(InitializeParams, _initialize_payload(_ATTESTING_TOOLS))
    attestation = remote_exec_attestation(parsed.capabilities)
    assert attestation.execution_bound
    assert attestation.network_locality
    assert attestation.ssh_trust_context


def test_s4b1b_old_request_without_attestation_entries_still_parses() -> None:
    """Direction (a) backward read, old client -> new server: a pre-S4b1b
    initialize request (no attestation entries) parses under BOTH models; the
    new model classifies it non-attesting, so every older client hard-denies
    remote execution.
    """
    payload = _initialize_payload(["terminal"])
    legacy = validate_wire(_LegacyInitializeParams, payload)
    assert legacy.capabilities.client_tools == ["terminal"]

    parsed = validate_wire(InitializeParams, payload)
    attestation = remote_exec_attestation(parsed.capabilities)
    assert not attestation.execution_bound
    assert not attestation.network_locality
    assert not attestation.ssh_trust_context


def test_s4b1b_strict_old_client_rejects_any_new_response_key() -> None:
    """Direction (b), new server -> old client response: the attestation
    entries are request-side only — InitializeResponse carries no capability
    echo and keeps extra="forbid", so a strict old client rejects an injected
    key, exactly as this test rejects it on the new model.
    """
    assert "client_tools" not in InitializeResponse.model_fields
    assert InitializeResponse.model_config.get("extra") == "forbid"
    payload = InitializeResponse(
        server_info=ServerInfo(name="chartreux", version="0.7.1")
    ).model_dump(mode="json", by_alias=True)
    assert validate_wire(InitializeResponse, payload).server_info.name == "chartreux"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_wire(InitializeResponse, {**payload, "clientTools": ["terminal"]})


# --- S4b3: strict-client compatibility of the destination-scoped revoke wire
#
# The destination parameter on policy/dangerous_commands/revoke and the
# destination-aware projection of policy/dangerous_commands/read both follow the
# 0.7.0 strict-client convention: the models keep extra="forbid" and a defaulted
# option serializes as an explicit null, so each direction is simulated here and
# documented, exactly as the attestation carriers above do:
#   (a) old server / new client REQUEST: a pre-S4b3 server model — revoke params
#       without a ``destination`` field — rejects a new client's revoke request,
#       including the whole-scope one, because the defaulted field still appears
#       on the wire as ``"destination": null``. The new model accepts both.
#   (b) old client / new server RESPONSE: a pre-S4b3 client model — grant wires
#       without ``destination`` and a read response without ``savedDestinations``
#       — rejects the new server's read response. The other direction (an old
#       server's response read by the new client) stays green, because every new
#       field defaults to None.

_DESTINATION = "deploy@203.0.113.10:22"


class _LegacyRevokeParams(ProtocolModel):
    """A pre-S4b3 server's view of policy/dangerous_commands/revoke params."""

    session_id: str
    operation_id: str
    expected_live_generation: int = Field(ge=0)
    expected_disk_revision: str | None = None
    user_initiated: Literal[True]


class _LegacyGrantWire(ProtocolModel):
    """A pre-S4b3 client's view of one dangerous-command authority entry."""

    operation_id: str
    scope: DangerousCommandGrantScopeWire


class _LegacyAuthorityReadResponse(ProtocolModel):
    """A pre-S4b3 client's view of the authority read response."""

    project: str
    live_authority_generation: int
    live_grants: list[_LegacyGrantWire]
    accepted_saved_grants: list[_LegacyGrantWire]
    saved_operations: list[str] | None = None
    disk_revision: str | None = None
    blocking_denylist: list[DangerousCommandDenylistBlockWire]


def _revoke_payload(destination: str | None) -> dict:
    return DangerousCommandRevokeParams(
        session_id="session-1",
        operation_id=REMOTE_EXEC_OPERATION_ID,
        expected_live_generation=3,
        expected_disk_revision="rev-1",
        destination=destination,
        user_initiated=True,
    ).model_dump(mode="json", by_alias=True)


def test_s4b3_strict_old_server_rejects_the_destination_request_field() -> None:
    """Direction (a), new client -> old server: the pre-S4b3 revoke model has no
    ``destination`` field and keeps extra="forbid", so it rejects BOTH forms of
    the new request — the destination-scoped one and the whole-scope one, whose
    defaulted field still serializes as an explicit null. The new server model
    accepts both, and the destination survives canonicalization on the wire.
    """
    scoped = _revoke_payload(_DESTINATION)
    assert scoped["destination"] == _DESTINATION
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_wire(_LegacyRevokeParams, scoped)
    assert (
        validate_wire(DangerousCommandRevokeParams, scoped).destination == _DESTINATION
    )

    # The whole-scope revoke does not escape the same rejection: the field is
    # present as null, which an older strict server refuses just the same.
    whole = _revoke_payload(None)
    assert whole["destination"] is None
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_wire(_LegacyRevokeParams, whole)
    assert validate_wire(DangerousCommandRevokeParams, whole).destination is None
    assert (
        validate_wire(DangerousCommandRevokeParams, whole).operation_id
        == REMOTE_EXEC_OPERATION_ID
    )


def test_s4b3_strict_old_client_rejects_the_new_response_keys() -> None:
    """Direction (b), new server -> old client response: the destination-aware
    projection adds ``savedDestinations`` to the read response and
    ``destination`` to every grant wire. The pre-S4b3 client model rejects the
    new payload, while the new client reads an older server's response (no new
    keys) with both fields defaulting to None.
    """
    scope = DangerousCommandGrantScopeWire(git_dir="/repo/.git", work_tree="/repo")
    response = DangerousCommandAuthorityReadResponse(
        project="/repo",
        live_authority_generation=3,
        live_grants=[
            DangerousCommandGrantWire(
                operation_id=REMOTE_EXEC_OPERATION_ID,
                scope=scope,
                destination=_DESTINATION,
            )
        ],
        accepted_saved_grants=[],
        saved_operations=[],
        saved_destinations=[_DESTINATION],
        disk_revision="rev-1",
        blocking_denylist=[],
    ).model_dump(mode="json", by_alias=True)
    assert response["savedDestinations"] == [_DESTINATION]
    assert response["liveGrants"][0]["destination"] == _DESTINATION
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_wire(_LegacyAuthorityReadResponse, response)

    # The backward read stays green: an older server's response parses under
    # the new model, and the absent destination projection is an explicit None
    # rather than a silent whole-scope claim.
    legacy = {
        "project": "/repo",
        "liveAuthorityGeneration": 3,
        "liveGrants": [
            {
                "operationId": REMOTE_EXEC_OPERATION_ID,
                "scope": {"gitDir": "/repo/.git", "workTree": "/repo"},
            }
        ],
        "acceptedSavedGrants": [],
        "savedOperations": [],
        "diskRevision": "rev-1",
        "blockingDenylist": [],
    }
    assert validate_wire(_LegacyAuthorityReadResponse, legacy).project == "/repo"
    parsed = validate_wire(DangerousCommandAuthorityReadResponse, legacy)
    assert parsed.live_grants[0].destination is None
    assert parsed.saved_destinations is None
