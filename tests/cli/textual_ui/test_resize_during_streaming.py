"""Terminal resize while chunks are actively arriving in a streaming turn."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import time
from typing import Any

import pytest

from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, ErrorMessage
from chartreux.core.llm_models import LLMChunk
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class GatedStreamingBackend(FakeBackend):
    """Emit one gated chunk at a time so a resize can land mid-stream."""

    def __init__(self, contents: list[str]) -> None:
        super().__init__()
        self._contents = list(contents)
        self.emitted = 0
        self.stream_started = asyncio.Event()
        self._release = asyncio.Event()

    def release_next_chunk(self) -> None:
        self._release.set()

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        self.stream_started.set()
        for content in self._contents:
            await self._release.wait()
            self._release = asyncio.Event()
            yield mock_llm_chunk(content=content)
            self.emitted += 1


async def _wait_until(pilot, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


@pytest.mark.asyncio
async def test_resize_mid_stream_keeps_transcript_coherent() -> None:
    contents = [f"chunk-{index:02d} " for index in range(12)]
    full_text = "".join(contents)
    backend = GatedStreamingBackend(contents)
    app = build_test_chartreux_app(
        agent_loop=build_test_agent_loop(backend=backend, enable_streaming=True)
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause()

        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("stream a long reply"))
        assert await _wait_until(pilot, backend.stream_started.is_set)

        for index in range(len(contents)):
            if index == 4:
                # Shrink the terminal while a chunk is still in flight.
                backend.release_next_chunk()
                assert app.event_handler is not None
                assert app.event_handler.current_streaming_message is not None
                await pilot.resize_terminal(60, 20)
                continue
            backend.release_next_chunk()
            assert await _wait_until(pilot, lambda i=index: backend.emitted == i + 1)
            if index == 8:
                # Grow between two chunks, still mid-stream.
                await pilot.resize_terminal(130, 44)

        assert await _wait_until(pilot, lambda: backend.emitted == len(contents))
        assert await _wait_until(pilot, lambda: not app._agent_job_active())
        await pilot.pause()

        # No crash and no error surfaced by the resizes.
        assert list(app.query(ErrorMessage)) == []

        # Exactly one assistant message holding the full streamed text: the
        # resizes duplicated nothing and lost nothing.
        assistant_messages = list(app.query(AssistantMessage))
        assert [message.get_content() for message in assistant_messages] == [full_text]

        # The resize-driven transcript reconcile settled.
        transcript = app._transcript
        assert transcript._reconcile_task is None or transcript._reconcile_task.done()
        assert not transcript._reconcile_pending
        assert not transcript._reconcile_active
