from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
import json
from typing import cast

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserInputRequestEvent,
    UserMessageEvent,
)
from chartreux.core.hooks.manager import HooksManager
from chartreux.core.hooks.models import HookToolDenial, HookToolInputRewrite
from chartreux.core.llm.format import ResolvedToolCall
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins.task import Task
from chartreux.core.tools.builtins.todo import TodoItem
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_tool import FakeTool

type QuestionHandler = Callable[[UserInputRequestEvent], Awaitable[UserQuestionResult]]


async def act_and_collect_events(
    agent_loop: AgentLoop, prompt: str, question_handler: QuestionHandler | None = None
) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    async for event in agent_loop.act(prompt):
        events.append(event)
        if not isinstance(event, UserInputRequestEvent):
            continue
        try:
            result = (
                await question_handler(event)
                if question_handler is not None
                else UserQuestionResult(answers=[], cancelled=True)
            )
        except BaseException as exc:
            agent_loop.reject_request(event.request_id, exc)
            continue
        agent_loop.resolve_user_input_request(event.request_id, result)
    return events


def make_question_tool_call(call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(
            name="ask_user_question",
            arguments='{"questions": [{"question": "Which runtime?", "options": [{"label": "Python"}, {"label": "Node"}]}]}',
        ),
    )


def make_question_loop(tool_calls: list[ToolCall]) -> AgentLoop:
    return build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["ask_user_question"]),
        backend=FakeBackend([
            [mock_llm_chunk(content="Which runtime?", tool_calls=tool_calls)],
            [mock_llm_chunk(content="Done.")],
        ]),
    )


async def answer_question(_event: UserInputRequestEvent) -> UserQuestionResult:
    return UserQuestionResult(
        answers=[UserAnswer(question="Which runtime?", answer="Python")]
    )


def tool_result(events: list[BaseEvent]) -> ToolResultEvent:
    return next(event for event in events if isinstance(event, ToolResultEvent))


def make_config(
    todo_permission: ToolPermission = ToolPermission.ALWAYS,
) -> ChartreuxConfigSchema:
    return build_test_vibe_config(
        enabled_tools=["todo"], tools={"todo": {"permission": todo_permission.value}}
    )


def make_todo_tool_call(
    call_id: str, index: int = 0, arguments: str | None = None
) -> ToolCall:
    args = arguments if arguments is not None else '{"action": "read"}'
    return ToolCall(
        id=call_id, index=index, function=FunctionCall(name="todo", arguments=args)
    )


def make_agent_loop(
    *, todo_permission: ToolPermission = ToolPermission.ALWAYS, backend: FakeBackend
) -> AgentLoop:
    agent_loop = build_test_agent_loop(
        config=make_config(todo_permission=todo_permission), backend=backend
    )
    return agent_loop


@pytest.mark.parametrize(
    ("initial_input", "rewritten_input", "expected_input"),
    [
        ({"task": "original"}, {"task": "rewritten"}, {"task": "rewritten"}),
        (
            {"task": "original", "config": {"model": "small"}},
            {"task": "original", "config": {"model": "large"}},
            {"task": "original", "config": {"model": "large"}},
        ),
    ],
)
def test_pre_tool_task_rewrite_preserves_sparse_launch_config(
    initial_input: dict[str, object],
    rewritten_input: dict[str, object],
    expected_input: dict[str, object],
) -> None:
    agent_loop = build_test_agent_loop()
    tool_call = ResolvedToolCall(
        tool_name="task",
        tool_class=Task,
        validated_args=TaskArgs.model_validate(initial_input),
        call_id="task-1",
    )

    rewritten = agent_loop._apply_tool_input_rewrite(
        tool_call, HookToolInputRewrite(hook_name="rewrite", tool_input=rewritten_input)
    )

    assert not isinstance(rewritten, HookToolDenial)
    _, tool_input = rewritten
    assert tool_input == expected_input


@pytest.mark.asyncio
async def test_single_tool_call_executes_when_permission_is_always() -> None:
    mocked_tool_call_id = "call_1"
    tool_call = make_todo_tool_call(mocked_tool_call_id)
    backend = FakeBackend([
        [mock_llm_chunk(content="Let me check your todos.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="I retrieved 0 todos.")],
    ])
    agent_loop = make_agent_loop(backend=backend)

    events = await act_and_collect_events(agent_loop, "What's my todo list?")

    assert [type(e) for e in events] == [
        UserMessageEvent,
        AssistantEvent,
        ToolCallEvent,
        ToolResultEvent,
        AssistantEvent,
    ]
    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[1], AssistantEvent)
    assert events[1].content == "Let me check your todos."
    assert isinstance(events[2], ToolCallEvent)
    assert events[2].tool_name == "todo"
    assert isinstance(events[3], ToolResultEvent)
    assert events[3].error is None
    assert events[3].skipped is False
    assert events[3].result is not None
    assert isinstance(events[4], AssistantEvent)
    assert events[4].content == "I retrieved 0 todos."
    # check conversation history
    tool_msgs = [m for m in agent_loop.messages if m.role == Role.tool]
    assert len(tool_msgs) == 1
    assert tool_msgs[-1].tool_call_id == mocked_tool_call_id
    assert "total_count" in (tool_msgs[-1].content or "")


@pytest.mark.asyncio
async def test_tool_call_executes_without_approval_at_ask_permission() -> None:
    agent_loop = make_agent_loop(
        todo_permission=ToolPermission.ASK,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Let me check your todos.",
                    tool_calls=[make_todo_tool_call("call_2")],
                )
            ],
            [mock_llm_chunk(content="I retrieved 0 todos.")],
        ]),
    )

    events = await act_and_collect_events(agent_loop, "What's my todo list?")

    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[1], AssistantEvent)
    assert isinstance(events[2], ToolCallEvent)
    assert events[2].tool_name == "todo"
    assert isinstance(events[3], ToolResultEvent)
    assert not any("approval" in type(event).__name__.lower() for event in events)
    result = tool_result(events)
    assert result.skipped is False
    assert result.error is None
    assert result.result is not None
    assert result.cancelled is False
    assert agent_loop.stats.tool_calls_rejected == 0
    assert agent_loop.stats.tool_calls_agreed == 1
    assert agent_loop.stats.tool_calls_succeeded == 1


@pytest.mark.asyncio
async def test_user_question_answered_by_request_event() -> None:
    agent_loop = make_question_loop([make_question_tool_call("call_3")])
    events = await act_and_collect_events(agent_loop, "Which runtime?", answer_question)
    assert len([e for e in events if isinstance(e, UserInputRequestEvent)]) == 1

    assert isinstance(events[0], UserMessageEvent)
    result = tool_result(events)
    assert result.skipped is False
    assert result.error is None
    assert result.result is not None
    assert agent_loop.stats.tool_calls_agreed == 1
    assert agent_loop.stats.tool_calls_rejected == 0
    assert agent_loop.stats.tool_calls_succeeded == 1


@pytest.mark.asyncio
async def test_user_question_cancelled_by_request_event() -> None:
    agent_loop = make_question_loop([make_question_tool_call("call_4")])
    events = await act_and_collect_events(agent_loop, "Which runtime?")

    assert isinstance(events[0], UserMessageEvent)
    result = tool_result(events)
    assert result.skipped is False
    assert result.error is None
    assert isinstance(result.result, UserQuestionResult)
    assert result.result.cancelled is True
    assert result.result.answers == []
    assert result.presentation is not None
    assert result.presentation.display.success is False
    assert agent_loop.stats.tool_calls_rejected == 0
    assert agent_loop.stats.tool_calls_agreed == 1


@pytest.mark.asyncio
async def test_tool_call_skipped_when_permission_is_never() -> None:
    agent_loop = make_agent_loop(
        todo_permission=ToolPermission.NEVER,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Let me check your todos.",
                    tool_calls=[make_todo_tool_call("call_never")],
                )
            ],
            [mock_llm_chunk(content="Tool is disabled.")],
        ]),
    )

    events = await act_and_collect_events(agent_loop, "What's my todo list?")

    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[3], ToolResultEvent)
    assert events[3].skipped is True
    assert events[3].error is None
    assert events[3].result is None
    assert events[3].skip_reason is not None
    assert "disabled by policy" in events[3].skip_reason.lower()
    tool_msgs = [
        m for m in agent_loop.messages if m.role == Role.tool and m.name == "todo"
    ]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "todo"
    assert events[3].skip_reason in (tool_msgs[-1].content or "")
    assert agent_loop.stats.tool_calls_rejected == 1
    assert agent_loop.stats.tool_calls_agreed == 0
    assert agent_loop.stats.tool_calls_succeeded == 0


@pytest.mark.asyncio
async def test_repeated_calls_do_not_mutate_configured_permission() -> None:
    agent_loop = make_agent_loop(
        todo_permission=ToolPermission.ASK,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="First check.",
                    tool_calls=[make_todo_tool_call("call_first")],
                )
            ],
            [mock_llm_chunk(content="First done.")],
            [
                mock_llm_chunk(
                    content="Second check.",
                    tool_calls=[make_todo_tool_call("call_second")],
                )
            ],
            [mock_llm_chunk(content="Second done.")],
        ]),
    )
    original_tools = agent_loop.config.model_dump()["tools"]

    events1 = await act_and_collect_events(agent_loop, "First request")
    events2 = await act_and_collect_events(agent_loop, "Second request")

    tool_config_todo = agent_loop.tool_manager.get_tool_config("todo")
    assert tool_config_todo.permission is ToolPermission.ASK
    tool_config_help = agent_loop.tool_manager.get_tool_config("bash")
    assert tool_config_help.permission is not ToolPermission.ALWAYS
    assert agent_loop.config.model_dump()["tools"] == original_tools
    assert isinstance(events1[0], UserMessageEvent)
    first_result = tool_result(events1)
    assert first_result.skipped is False
    assert first_result.result is not None
    assert isinstance(events2[0], UserMessageEvent)
    second_result = tool_result(events2)
    assert second_result.skipped is False
    assert second_result.result is not None
    assert agent_loop.stats.tool_calls_rejected == 0
    assert agent_loop.stats.tool_calls_succeeded == 2


@pytest.mark.asyncio
async def test_tool_call_with_invalid_action() -> None:
    tool_call = make_todo_tool_call("call_5", arguments='{"action": "invalid_action"}')
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Let me check your todos.", tool_calls=[tool_call]
                )
            ],
            [mock_llm_chunk(content="I encountered an error with the action.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "What's my todo list?")

    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[3], ToolResultEvent)
    assert events[3].error is not None
    assert events[3].result is None
    assert "tool_error" in events[3].error.lower()
    assert agent_loop.stats.tool_calls_failed == 1


@pytest.mark.asyncio
async def test_tool_call_with_duplicate_todo_ids() -> None:
    duplicate_todos = [
        TodoItem(id="duplicate", content="Task 1"),
        TodoItem(id="duplicate", content="Task 2"),
    ]
    tool_call = make_todo_tool_call(
        "call_6",
        arguments=json.dumps({
            "action": "write",
            "todos": [t.model_dump() for t in duplicate_todos],
        }),
    )
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Let me write todos.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="I couldn't write todos with duplicate IDs.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Add todos")

    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[3], ToolResultEvent)
    assert events[3].error is not None
    assert events[3].result is None
    assert "unique" in events[3].error.lower()
    assert agent_loop.stats.tool_calls_failed == 1


@pytest.mark.asyncio
async def test_tool_call_with_exceeding_max_todos() -> None:
    many_todos = [TodoItem(id=f"todo_{i}", content=f"Task {i}") for i in range(150)]
    tool_call = make_todo_tool_call(
        "call_7",
        arguments=json.dumps({
            "action": "write",
            "todos": [t.model_dump() for t in many_todos],
        }),
    )
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Let me write todos.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="I couldn't write that many todos.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Add todos")

    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[3], ToolResultEvent)
    assert events[3].error is not None
    assert events[3].result is None
    assert "100" in events[3].error
    assert agent_loop.stats.tool_calls_failed == 1


@pytest.mark.asyncio
async def test_tool_call_can_be_interrupted() -> None:
    """Test that tool calls can be interrupted via asyncio.CancelledError.

    When a tool raises CancelledError, the error is captured as a cancellation event
    and the agent loop stops gracefully after the current tool batch completes.
    """
    tool_call = ToolCall(
        id="call_8", index=0, function=FunctionCall(name="stub_tool", arguments="{}")
    )
    config = build_test_vibe_config(enabled_tools=["stub_tool"])
    agent_loop = build_test_agent_loop(
        config=config,
        backend=FakeBackend([
            [mock_llm_chunk(content="Let me use the tool.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Tool execution completed.")],
        ]),
    )
    # no dependency injection available => monkey patch
    agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool
    stub_tool_instance = agent_loop.tool_manager.get("stub_tool")
    assert isinstance(stub_tool_instance, FakeTool)
    stub_tool_instance._exception_to_raise = asyncio.CancelledError()

    events: list[BaseEvent] = []
    async for ev in agent_loop.act("Execute tool"):
        events.append(ev)

    tool_result_event = next(
        (e for e in events if isinstance(e, ToolResultEvent)), None
    )
    assert tool_result_event is not None
    assert tool_result_event.error is not None
    assert "execution interrupted by user" in tool_result_event.error.lower()
    assert agent_loop.stats.tool_calls_failed == 1

    # Agent loop should stop after cancellation — no second LLM turn
    assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
    assert len(assistant_events) == 1


class _RecordingHooksManager:
    def __init__(self) -> None:
        self.invoked: list[str] = []
        self.cancel_on_pre = False

    def reset_retry_count(self) -> None:
        return

    async def run(self, invocation: object) -> AsyncGenerator[object, None]:
        self.invoked.append(str(getattr(invocation, "hook_event_name", "")))
        if self.cancel_on_pre and self.invoked[-1] == "pre_tool":
            raise asyncio.CancelledError()
        return
        yield


def _install_recording_hooks(agent_loop: AgentLoop) -> _RecordingHooksManager:
    recorder = _RecordingHooksManager()
    agent_loop._hooks_manager = cast(HooksManager, recorder)
    return recorder


@pytest.mark.asyncio
async def test_post_tool_does_not_fire_when_cancel_lands_before_tool_execution() -> (
    None
):
    agent_loop = make_agent_loop(
        todo_permission=ToolPermission.ASK,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Let me check your todos.",
                    tool_calls=[make_todo_tool_call("call_cancel_pre")],
                )
            ],
            [mock_llm_chunk(content="Cancelled.")],
        ]),
    )
    recorder = _install_recording_hooks(agent_loop)
    recorder.cancel_on_pre = True

    events = await act_and_collect_events(agent_loop, "What's my todo list?")

    # The pre-tool hook interrupts the batch before a tool is invoked.
    assert not any(isinstance(e, ToolResultEvent) for e in events)
    assert agent_loop.stats.tool_calls_agreed == 0
    assert agent_loop.stats.tool_calls_succeeded == 0
    assert len([e for e in events if isinstance(e, AssistantEvent)]) == 1

    assert "pre_tool" in recorder.invoked
    assert "post_tool" not in recorder.invoked


@pytest.mark.asyncio
async def test_post_tool_fires_when_user_cancels_question() -> None:
    agent_loop = make_question_loop([make_question_tool_call("call_user_cancel")])
    recorder = _install_recording_hooks(agent_loop)

    events = await act_and_collect_events(agent_loop, "Which runtime?")

    result = tool_result(events)
    assert result.presentation is not None
    assert result.presentation.display.success is False
    assert isinstance(result.result, UserQuestionResult)
    assert result.result.cancelled is True
    assert "pre_tool" in recorder.invoked
    # Unlike obsolete pre-execution approval, the question is an executed tool.
    assert "post_tool" in recorder.invoked


@pytest.mark.asyncio
async def test_post_tool_does_not_fire_when_permission_is_never() -> None:
    agent_loop = make_agent_loop(
        todo_permission=ToolPermission.NEVER,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Let me check your todos.",
                    tool_calls=[make_todo_tool_call("call_never")],
                )
            ],
            [mock_llm_chunk(content="OK.")],
        ]),
    )
    recorder = _install_recording_hooks(agent_loop)

    events = await act_and_collect_events(agent_loop, "What's my todo list?")

    tool_result = next((e for e in events if isinstance(e, ToolResultEvent)), None)
    assert tool_result is not None
    assert tool_result.skipped is True

    assert "pre_tool" in recorder.invoked
    assert "post_tool" not in recorder.invoked


@pytest.mark.asyncio
async def test_post_tool_fires_when_cancel_lands_during_tool_execution() -> None:
    tool_call = ToolCall(
        id="call_cancel_mid",
        index=0,
        function=FunctionCall(name="stub_tool", arguments="{}"),
    )
    config = build_test_vibe_config(enabled_tools=["stub_tool"])
    agent_loop = build_test_agent_loop(
        config=config,
        backend=FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Done.")],
        ]),
    )
    agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool
    stub_tool_instance = agent_loop.tool_manager.get("stub_tool")
    assert isinstance(stub_tool_instance, FakeTool)
    stub_tool_instance._exception_to_raise = asyncio.CancelledError()

    recorder = _install_recording_hooks(agent_loop)

    async for _ev in agent_loop.act("Execute tool"):
        pass

    assert "post_tool" in recorder.invoked


@pytest.mark.asyncio
async def test_fill_missing_tool_responses_inserts_placeholders() -> None:
    agent_loop = build_test_agent_loop(
        config=make_config(), backend=FakeBackend(mock_llm_chunk(content="ok"))
    )
    tool_calls_messages = [
        make_todo_tool_call("tc1", index=0),
        make_todo_tool_call("tc2", index=1),
    ]
    assistant_msg = LLMMessage(
        role=Role.assistant, content="Calling tools...", tool_calls=tool_calls_messages
    )
    agent_loop.messages.reset([
        agent_loop.messages[0],
        assistant_msg,
        # only one tool responded: the second is missing
        LLMMessage(
            role=Role.tool, tool_call_id="tc1", name="todo", content="Retrieved 0 todos"
        ),
    ])

    await act_and_collect_events(agent_loop, "Proceed")

    tool_msgs = [m for m in agent_loop.messages if m.role == Role.tool]
    assert any(m.tool_call_id == "tc2" for m in tool_msgs)
    # find placeholder message for tc2
    placeholder = next(m for m in tool_msgs if m.tool_call_id == "tc2")
    assert placeholder.name == "todo"
    assert (
        placeholder.content
        == "<user_cancellation>Tool execution interrupted - no response available</user_cancellation>"
    )


@pytest.mark.asyncio
async def test_parallel_tool_calls_produce_correct_events() -> None:
    """Two tool calls in one LLM response should execute in parallel and produce correct events."""
    tool_call_1 = make_todo_tool_call("call_p1", index=0)
    tool_call_2 = make_todo_tool_call("call_p2", index=1)
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="Let me check two things.",
                tool_calls=[tool_call_1, tool_call_2],
            )
        ],
        [mock_llm_chunk(content="Both done.")],
    ])
    agent_loop = make_agent_loop(backend=backend)

    events = await act_and_collect_events(agent_loop, "Check two things")

    event_types = [type(e) for e in events]
    # UserMessage, Assistant, ToolCall, ToolCall, then two ToolResults (order may vary), then Assistant
    assert event_types[0] is UserMessageEvent
    assert event_types[1] is AssistantEvent
    # Both ToolCallEvents emitted upfront
    assert event_types[2] is ToolCallEvent
    assert event_types[3] is ToolCallEvent
    tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert {e.tool_call_id for e in tool_call_events} == {"call_p1", "call_p2"}
    # Both ToolResultEvents present
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_result_events) == 2
    assert {e.tool_call_id for e in tool_result_events} == {"call_p1", "call_p2"}
    for tool_result in tool_result_events:
        assert tool_result.error is None
        assert tool_result.skipped is False
        assert tool_result.result is not None
    # Final assistant message
    assert event_types[-1] is AssistantEvent
    assert isinstance(events[-1], AssistantEvent)
    assert events[-1].content == "Both done."
    # Verify conversation history has both tool responses
    tool_msgs = [m for m in agent_loop.messages if m.role == Role.tool]
    assert {m.tool_call_id for m in tool_msgs} == {"call_p1", "call_p2"}
    assert agent_loop.stats.tool_calls_succeeded == 2


@pytest.mark.asyncio
async def test_parallel_user_questions_both_receive_answers() -> None:
    question_calls: list[str] = []

    async def handler(event: UserInputRequestEvent) -> UserQuestionResult:
        question_calls.append(event.tool_call_id)
        return await answer_question(event)

    agent_loop = make_question_loop([
        make_question_tool_call("call_a1", index=0),
        make_question_tool_call("call_a2", index=1),
    ])
    events = await act_and_collect_events(agent_loop, "Check two things", handler)

    assert set(question_calls) == {"call_a1", "call_a2"}
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_result_events) == 2
    for tool_result in tool_result_events:
        assert tool_result.error is None
        assert tool_result.skipped is False
        assert tool_result.result is not None
    assert agent_loop.stats.tool_calls_agreed == 2
    assert agent_loop.stats.tool_calls_succeeded == 2


@pytest.mark.asyncio
async def test_parallel_question_responses_are_correlated_by_request_id() -> None:
    agent_loop = make_question_loop([
        make_question_tool_call(f"call_s{i}", index=i) for i in range(3)
    ])
    requests: list[UserInputRequestEvent] = []
    results: list[ToolResultEvent] = []
    async for event in agent_loop.act("Go"):
        if isinstance(event, UserInputRequestEvent):
            requests.append(event)
            if len(requests) == 3:
                assert len({request.request_id for request in requests}) == 3
                for request in reversed(requests):
                    agent_loop.resolve_user_input_request(
                        request.request_id,
                        UserQuestionResult(
                            answers=[
                                UserAnswer(
                                    question="Which runtime?",
                                    answer=request.tool_call_id,
                                )
                            ]
                        ),
                    )
        elif isinstance(event, ToolResultEvent):
            results.append(event)

    assert len(results) == 3
    for result in results:
        assert isinstance(result.result, UserQuestionResult)
        assert result.result.answers[0].answer == result.tool_call_id
    assert agent_loop.stats.tool_calls_agreed == 3
    assert agent_loop.stats.tool_calls_succeeded == 3


@pytest.mark.asyncio
async def test_parallel_questions_with_mixed_answer_and_cancellation() -> None:
    async def handler(event: UserInputRequestEvent) -> UserQuestionResult:
        if event.tool_call_id == "call_yes":
            return await answer_question(event)
        return UserQuestionResult(answers=[], cancelled=True)

    agent_loop = make_question_loop([
        make_question_tool_call("call_yes", index=0),
        make_question_tool_call("call_no", index=1),
    ])
    events = await act_and_collect_events(agent_loop, "Go", handler)

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 2
    by_id = {e.tool_call_id: e for e in results}
    assert by_id["call_yes"].error is None
    assert by_id["call_yes"].cancelled is False
    assert isinstance(by_id["call_yes"].result, UserQuestionResult)
    assert by_id["call_yes"].result.answers[0].answer == "Python"
    assert by_id["call_no"].error is None
    assert by_id["call_no"].presentation is not None
    assert by_id["call_no"].presentation.display.success is False
    assert isinstance(by_id["call_no"].result, UserQuestionResult)
    assert by_id["call_no"].result.answers == []
    assert agent_loop.stats.tool_calls_agreed == 2
    assert agent_loop.stats.tool_calls_rejected == 0


@pytest.mark.asyncio
async def test_parallel_three_tools_all_succeed() -> None:
    """Three parallel tool calls should all complete successfully."""
    tool_calls = [make_todo_tool_call(f"call_t{i}", index=i) for i in range(3)]
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Three tools.", tool_calls=tool_calls)],
            [mock_llm_chunk(content="All three done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Go")

    tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_call_events) == 3
    assert len(tool_result_events) == 3
    assert {e.tool_call_id for e in tool_call_events} == {
        "call_t0",
        "call_t1",
        "call_t2",
    }
    assert {e.tool_call_id for e in tool_result_events} == {
        "call_t0",
        "call_t1",
        "call_t2",
    }
    for tool_result in tool_result_events:
        assert tool_result.error is None
        assert tool_result.result is not None
    assert agent_loop.stats.tool_calls_succeeded == 3
    tool_msgs = [m for m in agent_loop.messages if m.role == Role.tool]
    assert len(tool_msgs) == 3


@pytest.mark.asyncio
async def test_parallel_one_tool_error_does_not_block_others() -> None:
    """If one parallel tool fails, the other should still succeed."""
    tc_good = make_todo_tool_call("call_good", index=0)
    tc_bad = make_todo_tool_call(
        "call_bad", index=1, arguments='{"action": "invalid_action"}'
    )
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Two tools.", tool_calls=[tc_good, tc_bad])],
            [mock_llm_chunk(content="Done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Go")

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2
    results_by_id = {e.tool_call_id: e for e in tool_results}
    assert results_by_id["call_good"].error is None
    assert results_by_id["call_good"].result is not None
    assert results_by_id["call_bad"].error is not None
    assert results_by_id["call_bad"].result is None
    assert agent_loop.stats.tool_calls_succeeded == 1
    assert agent_loop.stats.tool_calls_failed == 1


@pytest.mark.asyncio
async def test_parallel_calls_execute_without_approval_handler() -> None:
    tc1 = make_todo_tool_call("call_nc1", index=0)
    tc2 = make_todo_tool_call("call_nc2", index=1)
    agent_loop = make_agent_loop(
        todo_permission=ToolPermission.ASK,
        backend=FakeBackend([
            [mock_llm_chunk(content="Two tools.", tool_calls=[tc1, tc2])],
            [mock_llm_chunk(content="Cannot proceed.")],
        ]),
    )

    events = await act_and_collect_events(agent_loop, "Go")

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2
    for tool_result in tool_results:
        assert tool_result.skipped is False
        assert tool_result.result is not None
        assert tool_result.error is None
    assert agent_loop.stats.tool_calls_rejected == 0
    assert agent_loop.stats.tool_calls_succeeded == 2


@pytest.mark.asyncio
async def test_parallel_all_permission_never() -> None:
    tc1 = make_todo_tool_call("call_nv1", index=0)
    tc2 = make_todo_tool_call("call_nv2", index=1)
    agent_loop = make_agent_loop(
        todo_permission=ToolPermission.NEVER,
        backend=FakeBackend([
            [mock_llm_chunk(content="Two tools.", tool_calls=[tc1, tc2])],
            [mock_llm_chunk(content="Both disabled.")],
        ]),
    )

    events = await act_and_collect_events(agent_loop, "Go")

    assert not any("approval" in type(event).__name__.lower() for event in events)
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2
    for tool_result in tool_results:
        assert tool_result.skipped is True
        assert "disabled by policy" in (tool_result.skip_reason or "").lower()
    assert agent_loop.stats.tool_calls_rejected == 2


@pytest.mark.asyncio
async def test_parallel_tool_call_events_emitted_before_results() -> None:
    """All ToolCallEvents must appear before any ToolResultEvent in the event stream."""
    tool_calls = [make_todo_tool_call(f"call_o{i}", index=i) for i in range(3)]
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Three tools.", tool_calls=tool_calls)],
            [mock_llm_chunk(content="Done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Go")

    last_call_idx = max(i for i, e in enumerate(events) if isinstance(e, ToolCallEvent))
    first_result_idx = min(
        i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)
    )
    assert last_call_idx < first_result_idx


@pytest.mark.asyncio
async def test_parallel_conversation_history_has_all_tool_messages() -> None:
    """All parallel tool results must appear in the conversation messages."""
    tool_calls = [make_todo_tool_call(f"call_h{i}", index=i) for i in range(4)]
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Four tools.", tool_calls=tool_calls)],
            [mock_llm_chunk(content="All four done.")],
        ])
    )

    await act_and_collect_events(agent_loop, "Go")

    tool_msgs = [m for m in agent_loop.messages if m.role == Role.tool]
    assert {m.tool_call_id for m in tool_msgs} == {
        "call_h0",
        "call_h1",
        "call_h2",
        "call_h3",
    }
    assert agent_loop.stats.tool_calls_succeeded == 4


@pytest.mark.asyncio
async def test_pending_injected_message_continues_loop_after_tool_result() -> None:
    tool_call = make_todo_tool_call("call_inject")
    backend = FakeBackend([
        [mock_llm_chunk(content="Let me check.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Acting on the injected guidance.")],
    ])
    agent_loop = make_agent_loop(backend=backend)

    events: list[BaseEvent] = []
    async for event in agent_loop.act("Go"):
        events.append(event)
        if isinstance(event, ToolResultEvent):
            agent_loop._pending_injected_messages.append(
                LLMMessage(role=Role.user, content="updated context", injected=True)
            )

    assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
    assert len(assistant_events) == 2

    injected_msgs = [
        m for m in agent_loop.messages if m.role == Role.user and m.injected
    ]
    assert any("updated context" in (m.content or "") for m in injected_msgs)

    last_assistant = next(
        m for m in reversed(agent_loop.messages) if m.role == Role.assistant
    )
    assert "Acting on the injected guidance" in (last_assistant.content or "")
