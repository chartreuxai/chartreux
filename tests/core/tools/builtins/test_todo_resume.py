from __future__ import annotations

import json

from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.tools.builtins.todo import TodoState


def _write_call(call_id: str, todos: list[dict[str, str]]) -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        tool_calls=[
            ToolCall(
                id=call_id,
                index=0,
                function=FunctionCall(
                    name="todo",
                    arguments=json.dumps({"action": "write", "todos": todos}),
                ),
            )
        ],
    )


def _success(call_id: str) -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name="todo",
        tool_result=PersistedToolResult(output={"todos": []}),
    )


def test_replay_restores_only_successful_todo_writes() -> None:
    todos = [{"id": "kept", "content": "Keep this"}]

    state = TodoState.replay([
        _write_call("successful", todos),
        _success("successful"),
        _write_call("failed", [{"id": "lost", "content": "Do not restore"}]),
        LLMMessage(role=Role.tool, tool_call_id="failed", name="todo"),
    ])

    assert [todo.id for todo in state.todos] == ["kept"]


def test_replay_successful_empty_write_clears_todos() -> None:
    state = TodoState.replay([
        _write_call("set", [{"id": "old", "content": "Old"}]),
        _success("set"),
        _write_call("clear", []),
        _success("clear"),
    ])

    assert state.todos == []


def test_replay_uses_todo_state_preserved_at_compaction_boundary() -> None:
    state = TodoState.replay([
        LLMMessage(
            role=Role.user,
            content="compacted",
            injected=True,
            context_boundary="compaction",
            tool_result=PersistedToolResult(
                output={"todo_state": {"todos": [{"id": "saved", "content": "Saved"}]}}
            ),
        ),
        _write_call("later", [{"id": "later", "content": "Later"}]),
        LLMMessage(role=Role.tool, tool_call_id="later", name="todo"),
    ])

    assert [todo.id for todo in state.todos] == ["saved"]
