from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from chartreux.app_server._shell import restored_shell_effect_state, shell_effect_state
from chartreux.app_server.models import (
    CancelledEffectState,
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    FailedEffectState,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    ShellEffectDetail,
    ShellEffectInput,
)
from chartreux.app_server.protocol import ShellRunResponse
from chartreux.cli.textual_ui.widgets.collapsible import CollapsibleSection
from chartreux.cli.textual_ui.widgets.tools import ToolResultMessage
from chartreux.core.llm_models import ManualShellContext


class _ShellResultApp(App[None]):
    def __init__(self, entry: PublicEffectEntry) -> None:
        super().__init__()
        self._entry = entry
        self.result: ToolResultMessage | None = None

    def compose(self) -> ComposeResult:
        yield Vertical(id="root")

    async def on_mount(self) -> None:
        self.result = ToolResultMessage(self._entry)
        await self.query_one("#root", Vertical).mount(self.result)


def _entry(
    state: FailedEffectState | CancelledEffectState | CompletedEffectState,
) -> PublicEffectEntry:
    return PublicEffectEntry(
        id="shell-1",
        session_id="session-1",
        turn_id="turn-1",
        created_at=1,
        updated_at=2,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="shell",
        detail=ShellEffectDetail(
            tool_name="shell",
            input=ShellEffectInput(command="failing-command"),
            display=EffectCallDisplay(
                summary="shell: failing-command",
                verb="Running",
                message="failing-command",
                settled_verb="Ran",
                settled_message="failing-command",
                status_text="Running command",
            ),
        ),
        state=state,
    )


async def _render(
    state: FailedEffectState | CancelledEffectState | CompletedEffectState,
) -> list[str]:
    app = _ShellResultApp(_entry(state))
    async with app.run_test() as pilot:
        await pilot.pause()
        result = app.result
        assert result is not None
        sections = list(result.query(CollapsibleSection))
        for section in sections:
            section.set_collapsed(False)
        if sections:
            await pilot.pause()
        return [
            rendered.plain
            for widget in result.query(Static)
            if isinstance(rendered := widget.render(), Content)
        ]


@pytest.mark.asyncio
async def test_failed_manual_shell_shows_output_and_exit_status() -> None:
    state = FailedEffectState(
        error={"message": "Command exited with status 7"},  # type: ignore[arg-type]
        output_text="stdout line\nstderr line",
        display=EffectResultDisplay(
            success=False, message="Command exited with status 7"
        ),
    )

    rendered = await _render(state)

    assert any("Command exited with status 7" in text for text in rendered)
    assert "stdout line\nstderr line" in rendered
    assert rendered.count("stdout line\nstderr line") == 1


@pytest.mark.asyncio
async def test_failed_manual_shell_shows_stderr_only_output() -> None:
    rendered = await _render(
        FailedEffectState(
            error={"message": "Command exited with status 1"},  # type: ignore[arg-type]
            output_text="stderr only",
            display=EffectResultDisplay(
                success=False, message="Command exited with status 1"
            ),
        )
    )

    assert "stderr only" in rendered


@pytest.mark.asyncio
async def test_failed_manual_shell_shows_empty_output() -> None:
    rendered = await _render(
        FailedEffectState(
            error={"message": "Command exited with status 1"},  # type: ignore[arg-type]
            output_text="",
            display=EffectResultDisplay(
                success=False, message="Command exited with status 1"
            ),
        )
    )

    assert "(no output)" in rendered


@pytest.mark.asyncio
async def test_timed_out_manual_shell_retains_available_output() -> None:
    rendered = await _render(
        FailedEffectState(
            error={"message": "Command timed out"},  # type: ignore[arg-type]
            output_text="before timeout",
            display=EffectResultDisplay(success=False, message="Command timed out"),
        )
    )

    assert "before timeout" in rendered


@pytest.mark.asyncio
async def test_interrupted_manual_shell_retains_available_output() -> None:
    rendered = await _render(
        CancelledEffectState(
            reason="Command interrupted",
            output_text="before interruption",
            display=EffectResultDisplay(success=False, message="Command interrupted"),
        )
    )

    assert "before interruption" in rendered


@pytest.mark.asyncio
async def test_successful_manual_shell_output_is_expanded_once() -> None:
    rendered = await _render(
        CompletedEffectState(
            output={
                "stdout": "visible output",
                "stderr": "",
                "output": "visible output",
            },
            output_text="visible output",
            display=EffectResultDisplay(success=True, message="Ran command"),
        )
    )

    assert rendered.count("visible output") == 1

    result = ShellRunResponse(
        operation_id="shell-1",
        command="failing-command",
        cwd="/workspace",
        stdout="[bold]literal[/bold]",
        stderr="",
        exit_code=4,
    )
    live = shell_effect_state(result, output_text="[bold]literal[/bold]", duration_ms=1)
    restored = restored_shell_effect_state(
        ManualShellContext(
            operation_id="shell-1",
            command="failing-command",
            cwd="/workspace",
            stdout="[bold]literal[/bold]",
            exit_code=4,
            output_text="[bold]literal[/bold]",
            duration_ms=1,
            created_at=1,
        )
    )
    assert isinstance(live, FailedEffectState)
    assert isinstance(restored, FailedEffectState)

    live_rendered = await _render(live)
    restored_rendered = await _render(restored)

    assert live_rendered == restored_rendered
    assert live_rendered.count("[bold]literal[/bold]") == 1
