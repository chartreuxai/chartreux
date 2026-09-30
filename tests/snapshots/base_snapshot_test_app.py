from __future__ import annotations

import weakref

from rich.style import Style
from textual.widgets.text_area import TextAreaTheme

from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input import ChatTextArea
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_history_file,
)
from tests.snapshots.snapshot_event_loop import install_snapshot_wake
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend


def default_config(**kwargs) -> ChartreuxConfigSchema:
    """Default configuration for snapshot testing.
    Remove as much interference as possible from the snapshot comparison, in order to get a clean pixel-to-pixel comparison.
    - Injects a fake backend to prevent (or stub) LLM calls.
    - Disables the banner animation.
    - Forces a value for the displayed workdir
    - Hides the chat input cursor (as the blinking animation is not deterministic).
    - Shows thinking nodes by default; pass show_thinking_nodes=False to hide them.
    - Pins the model set to the shared on-disk test seed so the banner renders a
      stable single model regardless of the schema's evolving built-in defaults.
    """
    # Use the session-scoped model override to keep banner snapshots stable while
    # the persisted main default lives in the catalog's orchestrator preset.
    kwargs.setdefault("active_model", "glm-5-3")
    kwargs.setdefault("show_thinking_nodes", True)
    return build_test_vibe_config(
        disable_welcome_banner_animation=True,
        displayed_workdir="/test/workdir",
        **kwargs,
    )


class BaseSnapshotTestApp(ChartreuxApp):
    CSS_PATH = "../../chartreux/cli/textual_ui/app.tcss"

    def __init__(
        self,
        config: ChartreuxConfigSchema | None = None,
        backend: FakeBackend | None = None,
        agent_loop: AgentLoop | None = None,
        **kwargs,
    ):
        agent_loop_kwargs: dict = {}
        if "mcp_registry" in kwargs:
            agent_loop_kwargs["mcp_registry"] = kwargs.pop("mcp_registry")

        resolved_agent_loop = agent_loop or build_test_agent_loop(
            config=config or default_config(),
            enable_streaming=bool(kwargs.get("enable_streaming", False)),
            backend=backend or FakeBackend(),
            **agent_loop_kwargs,
        )

        history_file = kwargs.pop("history_file", None)
        history_dir = None
        if history_file is None:
            history_file, history_dir = make_test_history_file()

        super().__init__(
            history_file=history_file,
            app_server=lambda: create_test_app_server_session(resolved_agent_loop),
            **kwargs,
        )
        if history_dir is not None:
            weakref.finalize(self, history_dir.cleanup)

    async def on_load(self):
        install_snapshot_wake()
        await super().on_load()

    async def on_ready(self):
        # on_ready is called once all the on_mount in the MRO chain have been called
        # https://textual.textualize.io/api/events/#textual.events.Ready
        self._hide_chat_input_cursor()

    def _hide_chat_input_cursor(self) -> None:
        text_area = self.query_one(ChatTextArea)
        hidden_cursor_theme = TextAreaTheme(name="hidden_cursor", cursor_style=Style())
        text_area.register_theme(hidden_cursor_theme)
        text_area.theme = "hidden_cursor"
