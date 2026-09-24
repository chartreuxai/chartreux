from __future__ import annotations

from textual.app import App, ComposeResult
from textual.pilot import Pilot

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectResultDisplay,
    GenericEffectDetail,
    PublicEntryGenerationStatus,
)
from chartreux.app_server.protocol import (
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    AgentTranscriptToolStatus,
)
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ReasoningMessage,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from chartreux.utils.tool_presentation import EffectCallDisplay
from tests.snapshots.snap_compare import SnapCompare


def _text_entry(
    entry_id: str,
    text: str,
    kind: AgentTranscriptEntryKind,
    index: int,
    **payload: object,
) -> AgentTranscriptEntry:
    values: dict[str, object] = {
        "entry_id": entry_id,
        "kind": kind,
        "display_text": text,
        "digest": f"{index:064x}",
        "created_at": index,
        "updated_at": index,
        "generation_status": PublicEntryGenerationStatus.COMPLETED,
        "title": entry_id,
        **payload,
    }
    return AgentTranscriptEntry.model_validate(values)


def _tool_entry(
    entry_id: str, index: int, *, result_entry: bool
) -> AgentTranscriptEntry:
    name = "read_file"
    display = EffectCallDisplay(
        summary="read_file(src/example.py)",
        verb="Reading",
        message="src/example.py",
        settled_verb="Read",
        settled_message="src/example.py",
        status_text="Reading src/example.py",
    )
    return _text_entry(
        entry_id,
        "Example file contents",
        AgentTranscriptEntryKind.TOOL_RESULT
        if result_entry
        else AgentTranscriptEntryKind.TOOL_CALL,
        index,
        title=name,
        tool_name=name,
        tool_call_id="call-shared",
        arguments={"file_path": "src/example.py"},
        result={"content": "Example file contents"},
        output_text="Example file contents",
        status=AgentTranscriptToolStatus.COMPLETED,
        detail=GenericEffectDetail(
            tool_name=name, input={"file_path": "src/example.py"}, display=display
        ),
        state=CompletedEffectState(
            output={"content": "Example file contents"},
            output_text="Example file contents",
            display=EffectResultDisplay(
                success=True, verb="Read", message="src/example.py"
            ),
        ),
    )


class _SnapshotSource:
    def __init__(self) -> None:
        self.response = AgentTranscriptGetResponse(
            state=AgentTranscriptState.AVAILABLE,
            entries=[
                _text_entry(
                    "user",
                    "Please inspect **this image**.",
                    AgentTranscriptEntryKind.USER_TEXT,
                    0,
                    attachment_names=["diagram.png"],
                    attachment_count=1,
                ),
                _text_entry(
                    "assistant",
                    "I will inspect the image and read `src/example.py`.",
                    AgentTranscriptEntryKind.ASSISTANT_TEXT,
                    1,
                ),
                _text_entry(
                    "reasoning",
                    "First I will read the file, then summarize its contents.",
                    AgentTranscriptEntryKind.REASONING,
                    2,
                ),
                _tool_entry("call", 3, result_entry=False),
                _tool_entry("result", 4, result_entry=True),
            ],
            oldest_cursor="snapshot-cursor",
            has_more=False,
        )

    async def read_agent_transcript(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse:
        return self.response


class AgentTranscriptViewerSnapshotApp(App[None]):
    CSS_PATH = "../../chartreux/cli/textual_ui/app.tcss"

    def __init__(self) -> None:
        super().__init__()
        self._tools_collapsed = False
        self.viewer = AgentTranscriptViewer(
            _SnapshotSource(), "agent-snapshot", profile="reviewer"
        )

    def compose(self) -> ComposeResult:
        yield self.viewer


def test_snapshot_agent_transcript_viewer_renders_all_entry_kinds(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.1)
        viewer = pilot.app.query_one(AgentTranscriptViewer)
        if not viewer._known_entries:
            viewer.action_refresh()
        for _ in range(20):
            await pilot.pause(0.1)
            if (
                len(viewer.query(UserMessage)) == 1
                and len(viewer.query(AssistantMessage)) == 1
                and len(viewer.query(ReasoningMessage)) == 1
                and len(viewer.query(ToolCallMessage)) == 1
                and len(viewer.query(ToolResultMessage)) == 1
            ):
                return
        raise AssertionError("agent transcript did not populate before snapshot")

    assert snap_compare(
        "test_ui_snapshot_agent_transcript.py:AgentTranscriptViewerSnapshotApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )
