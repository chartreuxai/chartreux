from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from chartreux.app_server import _runtime as runtime_module
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import PublicMessageEntry
from chartreux.app_server.protocol import ProtocolErrorCode
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.agent_loop.test_agent_wait_steering import wait_call
from tests.app_server.test_subagents import BlockingBackend, _background_result
from tests.cli.textual_ui.test_message_queue_ui import _wait_until
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.parametrize(
    "mixed,lost_response", [(False, False), (False, True), (True, False)]
)
async def test_wait_submission_reaches_real_parent_before_children_complete(
    monkeypatch: pytest.MonkeyPatch, size, mixed: bool, lost_response: bool
) -> None:
    backend = FakeBackend()
    parent = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["task", "wait_for_agent", "bash"],
            tools={"bash": {"permission": "always"}},
        ),
        backend=backend,
    )
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        runtime_module,
        "AgentLoop",
        lambda **kwargs: AgentLoop(backend=child_backend, **kwargs),
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(parent, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    app = build_test_chartreux_app(
        agent_loop=parent, app_server=lambda: attach_test_app_server_session(client)
    )
    try:
        async with app.run_test(size=size) as pilot:
            await app._session_ready.wait()
            await app.app_server.resources.runtime.wait_until_ready()
            root = legacy_backend(server)
            root.session.turns.link_subagent = AsyncMock()
            children = []
            for index in range(2):
                launch = await _background_result(
                    root.children,
                    TaskArgs(task=f"child {index}"),
                    InvokeContext(
                        tool_call_id=f"launch-{index}", session_id=parent.session_id
                    ),
                )
                assert launch.agent_id is not None and launch.run_id is not None
                children.append((launch.agent_id, launch.run_id))
            await asyncio.wait_for(child_backend.started.wait(), 3)
            calls = [wait_call(agent_id, i) for i, (agent_id, _) in enumerate(children)]
            unrelated_entered, unrelated_release = asyncio.Event(), asyncio.Event()
            if mixed:
                calls.append(
                    ToolCall(
                        id="unrelated",
                        index=2,
                        function=FunctionCall(
                            name="bash", arguments='{"command": "printf unrelated"}'
                        ),
                    )
                )
                original_collect = parent._run_post_tool_hooks

                async def hold_unrelated(call, **kwargs):
                    if call.call_id == "unrelated":
                        unrelated_entered.set()
                        await unrelated_release.wait()
                    async for event in original_collect(call, **kwargs):
                        yield event

                monkeypatch.setattr(parent, "_run_post_tool_hooks", hold_unrelated)
            backend._streams.extend([
                [mock_llm_chunk(tool_calls=calls)],
                [mock_llm_chunk(content="continued")],
            ])
            await app._handle_user_message("start")
            if mixed:
                assert await _wait_until(pilot, unrelated_entered.is_set, timeout=5)
                assert not app.app_server.waiting_only
                await app._dispatch_submitted_value("mixed stays queued")
                assert not any(
                    m.content == "mixed stays queued" for m in parent.messages
                )
                assert len(app.app_server.turn_queue.items) == 1
                assert root.session.turns.active_turn is not None
                assert parent.outstanding_wait_call_ids(
                    root.session.turns.active_turn.id
                )
                unrelated_release.set()
            assert await _wait_until(
                pilot, lambda: app.app_server.waiting_only, timeout=5
            )
            assert root.session.turns.active_turn is not None
            turn_id = root.session.turns.active_turn.id
            older_queue = None
            if not lost_response:
                await app._enqueue_prompt_with_resources("older queued instructions")
                await pilot.pause()
                older_queue = app.app_server.turn_queue.model_copy(deep=True)
                original_stream = backend.complete_streaming

                async def hold_continuation(**kwargs):
                    # The first provider call has already finished. Keep the
                    # continuation alive so promotion cannot hide backlog edits.
                    await asyncio.Event().wait()
                    async for chunk in original_stream(**kwargs):
                        yield chunk

                monkeypatch.setattr(backend, "complete_streaming", hold_continuation)
            # Dismissing a locally owned console is not an interruption.
            await app.action_toggle_debug_console()
            assert app._debug_console is not None
            app._debug_console.query_one("#debug-console-log").focus()
            await pilot.pause()
            await pilot.press("escape")
            assert await _wait_until(pilot, lambda: app._debug_console is None)
            assert app.app_server.waiting_only
            original_prepare = app.app_server.resources.workspace.prepare_prompt
            prepare = AsyncMock(wraps=original_prepare)
            monkeypatch.setattr(
                app.app_server.resources.workspace, "prepare_prompt", prepare
            )
            original_steer = app._queue._ports.steer_turn
            requests = []

            async def steer(*args, **kwargs):
                requests.append((args, kwargs.copy()))
                await original_steer(*args, **kwargs)
                if lost_response and len(requests) == 1:
                    raise OSError("response lost after commit")

            from dataclasses import replace

            app._queue._ports = replace(app._queue._ports, steer_turn=steer)
            await app._dispatch_submitted_value("new instructions")
            assert prepare.await_count == 1
            assert not child_backend.release.is_set()
            assert not child_backend.stopped.is_set()
            assert (
                len([
                    m
                    for m in parent.messages
                    if m.role is Role.user and m.content == "new instructions"
                ])
                == 1
            )
            assert requests[0][1]["require_waiting_only"] is True
            assert requests[0][1]["expected_turn_id"] == turn_id
            if lost_response:
                assert app._queue.has_unresolved_steering
                assert not app.app_server.turn_queue.items
                assert await _wait_until(
                    pilot, lambda: not app.app_server.turn_active, timeout=5
                )
                assert await app._steer_queued_now()
                assert requests[0] == requests[1]
                assert not app._queue.has_unresolved_steering
                assert (
                    len([m for m in parent.messages if m.content == "new instructions"])
                    == 1
                )
            else:
                assert app.app_server.turn_queue == older_queue
                expected_backlog = (["mixed stays queued"] if mixed else []) + [
                    "older queued instructions"
                ]
                assert [
                    widget.get_content() for widget in app._queue.widgets
                ] == expected_backlog
                assert app._queue.widgets[0].pending
                assert root.session.turns.active_turn is not None
                assert root.session.turns.active_turn.id == turn_id
                # Only top-level Escape interrupts, not the local dismissal.
                await pilot.press("escape")
                assert await _wait_until(
                    pilot, lambda: not app.app_server.turn_active, timeout=5
                )
                assert app.app_server.turn_queue.paused
            await pilot.pause()
            entry_id = requests[0][0][2]
            canonical = next(
                entry
                for entry in app.app_server.history
                if isinstance(entry, PublicMessageEntry) and entry.id == entry_id
            )
            tracked = list(app._queue._message_widgets[entry_id])
            assert len(tracked) == 1
            assert tracked[0].posted_at == canonical.posted_at
            assert not tracked[0].pending
            for child in children:
                assert await root.children.get_agent_result(*child) is None
            child_backend.release.set()
            for child in children:
                result = await asyncio.wait_for(root.children.wait_for_agent(*child), 5)
                assert result.response == "Child completed"
    finally:
        child_backend.release.set()


@pytest_asyncio.fixture(params=[(80, 24), (120, 36)])
async def waiting_app(monkeypatch: pytest.MonkeyPatch, request):
    backend = FakeBackend()
    parent = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["task", "wait_for_agent"]),
        backend=backend,
    )
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        runtime_module,
        "AgentLoop",
        lambda **kwargs: AgentLoop(backend=child_backend, **kwargs),
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(parent, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    app = build_test_chartreux_app(
        agent_loop=parent, app_server=lambda: attach_test_app_server_session(client)
    )
    try:
        async with app.run_test(size=request.param) as pilot:
            await app._session_ready.wait()
            await app.app_server.resources.runtime.wait_until_ready()
            root = legacy_backend(server)
            root.session.turns.link_subagent = AsyncMock()
            launch = await _background_result(
                root.children,
                TaskArgs(task="child work"),
                InvokeContext(tool_call_id="launch", session_id=parent.session_id),
            )
            assert launch.agent_id is not None and launch.run_id is not None
            await asyncio.wait_for(child_backend.started.wait(), 3)
            backend._streams.extend([
                [mock_llm_chunk(tool_calls=[wait_call(launch.agent_id)])],
                [mock_llm_chunk(content="continued")],
                [mock_llm_chunk(content="promoted")],
            ])
            await app._handle_user_message("start")
            assert await _wait_until(pilot, lambda: app.app_server.waiting_only)
            yield app, pilot, parent, backend, child_backend, root, launch
    finally:
        child_backend.release.set()


@pytest.mark.asyncio
async def test_full_app_stale_admission_preserves_prompt_without_cancelling_successor(
    monkeypatch: pytest.MonkeyPatch, waiting_app
) -> None:
    app, pilot, parent, backend, child_backend, root, launch = waiting_app
    assert root.session.turns.active_turn is not None
    old_id = root.session.turns.active_turn.id
    entered, release = asyncio.Event(), asyncio.Event()
    original_prepare = app.app_server.resources.workspace.prepare_prompt

    async def prepare(*args, **kwargs):
        prepared = await original_prepare(*args, **kwargs)
        if args[0] == "stale instructions":
            entered.set()
            await release.wait()
        return prepared

    monkeypatch.setattr(app.app_server.resources.workspace, "prepare_prompt", prepare)
    submission = asyncio.create_task(
        app._dispatch_submitted_value("stale instructions")
    )
    await asyncio.wait_for(entered.wait(), 3)
    # The UI captured old_id before preparation. Finish that real turn and
    # start a successor with a live wait before releasing the prepared request.
    await app.app_server.inject_user_context("finish old", require_active_turn=True)
    await root.session.turns.wait_for_operation(old_id)
    assert await _wait_until(pilot, lambda: not app.app_server.turn_active)
    backend._streams.insert(
        0,
        [
            mock_llm_chunk(
                tool_calls=[
                    wait_call(launch.agent_id).model_copy(
                        update={"id": "successor-wait"}
                    )
                ]
            )
        ],
    )
    await app._handle_user_message("successor")
    assert await _wait_until(pilot, lambda: app.app_server.waiting_only)
    assert root.session.turns.active_turn is not None
    successor_id = root.session.turns.active_turn.id
    assert successor_id != old_id
    cancel = Mock(wraps=parent.cancel_outstanding_waits_for_steering)
    monkeypatch.setattr(parent, "cancel_outstanding_waits_for_steering", cancel)
    errors = []
    original_steer = app._queue._ports.steer_turn

    async def steer(*args, **kwargs):
        assert kwargs["expected_turn_id"] == old_id
        try:
            await original_steer(*args, **kwargs)
        except Exception as error:
            errors.append(error)
            raise

    app._queue._ports = replace(app._queue._ports, steer_turn=steer)
    release.set()
    await asyncio.wait_for(submission, 5)
    assert len(errors) == 1
    assert errors[0].error.code is ProtocolErrorCode.STALE_TURN
    cancel.assert_not_called()
    assert parent.outstanding_wait_call_ids(successor_id) == ("successor-wait",)
    assert not child_backend.stopped.is_set()
    assert not any(m.content == "stale instructions" for m in parent.messages)
    assert [w.get_content() for w in app._queue.widgets] == ["stale instructions"]
    assert app._queue.widgets[0].pending
    assert len(app.app_server.turn_queue.items) == 1
    assert not app._queue.has_unresolved_steering


@pytest.mark.asyncio
async def test_full_app_promotion_race_delivers_rejected_admission_once(
    monkeypatch: pytest.MonkeyPatch, waiting_app
) -> None:
    app, pilot, parent, _, child_backend, root, launch = waiting_app
    assert root.session.turns.active_turn is not None
    turn_id = root.session.turns.active_turn.id
    entered, release = asyncio.Event(), asyncio.Event()
    original_steer = app._queue._ports.steer_turn
    requests = []

    async def steer(*args, **kwargs):
        requests.append((args, kwargs))
        entered.set()
        await release.wait()
        await original_steer(*args, **kwargs)

    app._queue._ports = replace(app._queue._ports, steer_turn=steer)
    submission = asyncio.create_task(app._dispatch_submitted_value("promote once"))
    await asyncio.wait_for(entered.wait(), 3)
    child_backend.release.set()
    await asyncio.wait_for(
        root.children.wait_for_agent(launch.agent_id, launch.run_id), 5
    )
    await root.session.turns.wait_for_operation(turn_id)
    assert await _wait_until(pilot, lambda: not app.app_server.turn_active)
    release.set()
    await asyncio.wait_for(submission, 5)
    assert await _wait_until(
        pilot,
        lambda: (
            any(m.content == "promote once" for m in parent.messages)
            and not app.app_server.turn_active
        ),
    )
    assert len(requests) == 1
    entry_id = requests[0][0][2]
    assert len([m for m in parent.messages if m.content == "promote once"]) == 1
    assert (
        len([
            e
            for e in app.app_server.history
            if isinstance(e, PublicMessageEntry) and e.id == entry_id
        ])
        == 1
    )
    assert not app.app_server.turn_queue.items
    assert not app._queue.has_server_work
    assert not next(iter(app._queue._message_widgets[entry_id])).pending
    result = await root.children.get_agent_result(launch.agent_id, launch.run_id)
    assert result is not None and result.response == "Child completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_first", [False, True])
async def test_full_app_child_completion_and_steering_emit_one_wait_result(
    monkeypatch: pytest.MonkeyPatch, waiting_app, completion_first: bool
) -> None:
    app, pilot, parent, _, child_backend, root, launch = waiting_app
    entered, release = asyncio.Event(), asyncio.Event()
    finalized, finish_result = asyncio.Event(), asyncio.Event()
    original_inject = parent.inject_user_context
    original_collect = parent._run_post_tool_hooks

    async def inject(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_inject(*args, **kwargs)

    async def collect(call, **kwargs):
        if call.call_id == launch.agent_id:
            finalized.set()
            await finish_result.wait()
        async for event in original_collect(call, **kwargs):
            yield event

    monkeypatch.setattr(parent, "inject_user_context", inject)
    monkeypatch.setattr(parent, "_run_post_tool_hooks", collect)
    submission = asyncio.create_task(
        app._dispatch_submitted_value("racing instructions")
    )
    await asyncio.wait_for(entered.wait(), 3)
    if completion_first:
        child_backend.release.set()
        await asyncio.wait_for(finalized.wait(), 5)
        # The blocking wait completed but its result finalization still owns the
        # batch. Final server validation must reject the stale waiting-only hint.
        release.set()
        await asyncio.wait_for(submission, 5)
        assert len(app.app_server.turn_queue.items) == 1
        finish_result.set()
    else:
        release.set()
        await asyncio.wait_for(finalized.wait(), 5)
        await asyncio.wait_for(submission, 5)
        child_backend.release.set()
        finish_result.set()
    assert await _wait_until(pilot, lambda: not app.app_server.turn_active)
    results = [
        m
        for m in parent.messages
        if m.role is Role.tool and m.tool_call_id == launch.agent_id
    ]
    assert len(results) == 1
    assert results[0].tool_result is not None
    assert results[0].tool_result.cancelled is not completion_first
    assert len([m for m in parent.messages if m.content == "racing instructions"]) == 1
    assert not app.app_server.turn_queue.items
    assert not app._queue.has_unresolved_steering
    result = await asyncio.wait_for(
        root.children.wait_for_agent(launch.agent_id, launch.run_id), 5
    )
    assert result.response == "Child completed"
    assert (
        await root.children.get_agent_result(launch.agent_id, launch.run_id) is result
    )


@pytest.mark.asyncio
async def test_wait_completion_consumes_queue_item_while_edit_preserves_copy_once(
    waiting_app,
) -> None:
    app, pilot, parent, _, child_backend, root, launch = waiting_app
    await app._enqueue_prompt_with_resources("older queued instructions")
    await pilot.pause()
    assert app._chat_input_container is not None
    body = app._chat_input_container._body
    assert body is not None
    await pilot.press("up", "enter")
    assert body._queue_in_edit_mode
    assert body.input_widget is not None
    body.input_widget.load_text("edited after completion")

    child_backend.release.set()
    await asyncio.wait_for(
        root.children.wait_for_agent(launch.agent_id, launch.run_id), 5
    )
    assert await _wait_until(
        pilot,
        lambda: not app.app_server.turn_queue.items and not app.app_server.turn_active,
    )
    assert (
        len([m for m in parent.messages if m.content == "older queued instructions"])
        == 1
    )
    assert body._queue_in_edit_mode
    await pilot.press("enter")
    assert body._queue_edit_consumed
    assert not any(m.content == "edited after completion" for m in parent.messages)
    await pilot.press("enter")
    assert await _wait_until(
        pilot,
        lambda: any(m.content == "edited after completion" for m in parent.messages),
    )
    assert (
        len([m for m in parent.messages if m.content == "edited after completion"]) == 1
    )
    assert not app._queue.has_unresolved_steering
