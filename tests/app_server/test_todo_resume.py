from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import JsonValue
import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config


@pytest.mark.asyncio
async def test_in_place_resume_restores_successful_todo_write(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    todos = [{"id": "resume", "content": "Restore me"}]
    saved.messages.extend([
        LLMMessage(
            role=Role.assistant,
            tool_calls=[
                ToolCall(
                    id="todo-write",
                    index=0,
                    function=FunctionCall(
                        name="todo",
                        arguments=json.dumps({"action": "write", "todos": todos}),
                    ),
                )
            ],
        ),
        LLMMessage(
            role=Role.tool,
            name="todo",
            tool_call_id="todo-write",
            tool_result=PersistedToolResult(
                output=cast(dict[str, JsonValue], {"todos": todos})
            ),
        ),
    ])
    await saved._save_messages()
    session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config)
    try:
        await AgentRuntimeFactory().resume_root(source, session_id)
        assert [todo.id for todo in source.tool_manager.get("todo").state.todos] == [
            "resume"
        ]
    finally:
        await source.aclose()
