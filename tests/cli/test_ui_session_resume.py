from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import OptionList

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server.models import CompletedEffectState
from chartreux.cli.textual_ui.widgets.compact import CompactMessage
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from chartreux.cli.textual_ui.widgets.model_picker import ModelPickerApp
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from chartreux.core.config import (
    ChartreuxConfigSchema,
    ModelConfig,
    SessionLoggingConfig,
)
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.session.session_loader import SessionLoader
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
    wait_until,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_ui_displays_messages_when_resuming_session(
    vibe_config: ChartreuxConfigSchema,
) -> None:
    """Test that messages are properly displayed when resuming a session."""
    agent_loop = build_test_agent_loop(config=vibe_config)

    # Simulate a previous session with messages
    user_msg = LLMMessage(role=Role.user, content="Hello, how are you?")
    assistant_msg = LLMMessage(
        role=Role.assistant,
        content="I'm doing well, thank you!",
        tool_calls=[
            ToolCall(
                id="tool_call_1",
                index=0,
                function=FunctionCall(
                    name="read", arguments='{"file_path": "test.txt"}'
                ),
            )
        ],
    )
    tool_result_msg = LLMMessage(
        role=Role.tool,
        content="File content here",
        name="read",
        tool_call_id="tool_call_1",
    )

    agent_loop.messages.extend([user_msg, assistant_msg, tool_result_msg])

    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        # Wait for the app to initialize and rebuild history
        await pilot.pause(0.5)

        # Verify user message is displayed
        user_messages = app.query(UserMessage)
        assert len(user_messages) == 1
        assert user_messages[0]._content == "Hello, how are you?"

        # Verify assistant message is displayed
        assistant_messages = app.query(AssistantMessage)
        assert len(assistant_messages) == 1
        assert assistant_messages[0]._content == "I'm doing well, thank you!"

        # Verify tool call message is displayed
        tool_call_messages = app.query(ToolCallMessage)
        assert len(tool_call_messages) == 1
        assert tool_call_messages[0]._tool_name == "read"

        # Verify tool result message is displayed
        tool_result_messages = app.query(ToolResultMessage)
        assert len(tool_result_messages) == 1
        assert tool_result_messages[0].tool_name == "read"
        state = tool_result_messages[0]._state
        assert isinstance(state, CompletedEffectState)
        assert state.output_text == "File content here"


@pytest.mark.asyncio
async def test_ui_does_not_display_messages_when_only_system_messages_exist(
    vibe_config: ChartreuxConfigSchema,
) -> None:
    """Test that no messages are displayed when only system messages exist."""
    agent_loop = build_test_agent_loop(config=vibe_config)

    # Only system messages
    system_msg = LLMMessage(role=Role.system, content="System prompt")
    agent_loop.messages.append(system_msg)

    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.5)

        # Verify no user or assistant messages are displayed
        user_messages = app.query(UserMessage)
        assert len(user_messages) == 0

        assistant_messages = app.query(AssistantMessage)
        assert len(assistant_messages) == 0


@pytest.mark.asyncio
async def test_ui_displays_multiple_user_assistant_turns(
    vibe_config: ChartreuxConfigSchema,
) -> None:
    """Test that multiple conversation turns are properly displayed."""
    agent_loop = build_test_agent_loop(config=vibe_config)

    # Multiple conversation turns
    messages = [
        LLMMessage(role=Role.user, content="First question"),
        LLMMessage(role=Role.assistant, content="First answer"),
        LLMMessage(role=Role.user, content="Second question"),
        LLMMessage(role=Role.assistant, content="Second answer"),
    ]

    agent_loop.messages.extend(messages)

    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.5)

        # Verify all messages are displayed
        user_messages = app.query(UserMessage)
        assert len(user_messages) == 2
        assert user_messages[0]._content == "First question"
        assert user_messages[1]._content == "Second question"

        assistant_messages = app.query(AssistantMessage)
        assert len(assistant_messages) == 2
        assert assistant_messages[0]._content == "First answer"
        assert assistant_messages[1]._content == "Second answer"


@pytest.mark.asyncio
async def test_ui_displays_compaction_checkpoint_when_resuming_session(
    vibe_config: ChartreuxConfigSchema,
) -> None:
    agent_loop = build_test_agent_loop(config=vibe_config)
    agent_loop.messages.extend([
        LLMMessage(role=Role.user, content="Before compaction"),
        LLMMessage(
            role=Role.user,
            content="Compacted context",
            injected=True,
            context_boundary="compaction",
        ),
        LLMMessage(role=Role.assistant, content="After compaction"),
    ])
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.5)

        assert [message._content for message in app.query(UserMessage)] == [
            "Before compaction"
        ]
        assert [message._content for message in app.query(AssistantMessage)] == [
            "After compaction"
        ]
        assert [message.get_content() for message in app.query(CompactMessage)] == [
            "Compaction completed."
        ]


class _RecordingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.models: list[ModelConfig] = []

    async def complete(self, *, model: ModelConfig, **kwargs: object):
        self.models.append(model)
        return await super().complete(model=model, **kwargs)


@pytest.mark.asyncio
async def test_selected_model_survives_exit_and_fresh_resume_without_new_turn(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    models = [
        ModelConfig(name="alpha-wire", provider="mistral", alias="alpha"),
        ModelConfig(name="beta-wire", provider="mistral", alias="beta"),
    ]
    config = build_test_vibe_config(
        active_model="alpha", models=models, session_logging=logging
    )
    saved = build_test_agent_loop(config=config, backend=FakeBackend(), cwd=tmp_path)
    await saved.persist_empty_session()
    session_id = saved.session_id
    session_dir = saved.session_logger.session_dir
    assert session_dir is not None

    app = build_test_chartreux_app(agent_loop=saved)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_model()
        await wait_until(pilot, lambda: bool(app.query(ModelPickerApp)))
        picker = app.query_one(ModelPickerApp)
        await wait_until(pilot, lambda: picker.query_one(OptionList).has_focus)
        await pilot.press("down", "enter")
        await wait_until(pilot, lambda: app.config.active_model.alias == "beta")
        await wait_until(
            pilot,
            lambda: (
                saved.committed_model is not None
                and saved.committed_model.base_model == "beta"
            ),
        )

    selected_identity = saved.committed_model
    assert selected_identity is not None
    await saved.aclose()

    durable_metadata = SessionLoader.load_metadata(session_dir)
    assert durable_metadata.launch_config is not None
    assert durable_metadata.launch_config.version == 2
    assert durable_metadata.launch_config.committed_model == selected_identity

    resumed_backend = _RecordingBackend()
    resumed = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model="alpha", models=models, session_logging=logging
        ),
        backend=resumed_backend,
        cwd=tmp_path,
    )
    try:
        await AgentRuntimeFactory().resume_root(resumed, session_id)
        assert resumed.committed_model == selected_identity

        async for _ in resumed.act("continue"):
            pass

        assert [model.alias for model in resumed_backend.models] == ["beta"]
    finally:
        await resumed.aclose()


@pytest.mark.asyncio
async def test_ui_displays_messages_when_resuming_in_dangerous_directory(
    monkeypatch: pytest.MonkeyPatch, vibe_config: ChartreuxConfigSchema
) -> None:
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.app.is_dangerous_directory",
        lambda: (True, "You are in the home directory"),
    )

    agent_loop = build_test_agent_loop(config=vibe_config)
    agent_loop.messages.extend([
        LLMMessage(role=Role.user, content="Hello from a previous run"),
        LLMMessage(role=Role.assistant, content="Welcome back!"),
    ])

    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.5)

        user_messages = app.query(UserMessage)
        assistant_messages = app.query(AssistantMessage)

        assert len(user_messages) == 1
        assert user_messages[0]._content == "Hello from a previous run"
        assert len(assistant_messages) == 1
        assert assistant_messages[0]._content == "Welcome back!"
