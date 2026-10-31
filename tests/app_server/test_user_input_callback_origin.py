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

import json
from pathlib import Path
import subprocess

from acp.schema import ElicitationFormSessionMode
from pydantic import ValidationError
import pytest

from chartreux.acp.agent import _build_elicitation_schema
from chartreux.app_server._model import validate_wire
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
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.questions import QuestionChoice, UserQuestion, UserQuestionRequest
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
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
