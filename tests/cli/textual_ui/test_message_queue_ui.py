from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
import time

import pytest
from textual.app import App

from chartreux.app_server._turns import TurnController
from chartreux.app_server.models import (
    PreparedPrompt,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    PublicQueuedTurn,
    PublicTurnQueue,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
    SessionTextContentBlock,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from chartreux.cli.textual_ui.widgets.loading import INTERRUPTING_LOADING_STATUS
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    InterruptMessage,
    QueueHeaderMessage,
    SlashCommandMessage,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.tools import ToolResultMessage
from chartreux.observability.logging import set_config_log_level, set_session_override
from chartreux.ui.widgets.theme_picker import ThemePickerApp, sorted_theme_names
from tests.cli.textual_ui.test_message_queue import _queue_controller, _server_turn
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@pytest.fixture(autouse=True)
def _reset_log_level_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    set_session_override(None)
    set_config_log_level(None)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DEBUG_MODE", raising=False)
    yield
    set_session_override(None)
    set_config_log_level(None)


@pytest.fixture
def chartreux_app() -> ChartreuxApp:
    return build_test_chartreux_app()


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return False


async def _press_escape_and_wait_for_settle(pilot, app, timeout: float = 5.0) -> None:
    """Interrupt the active turn and wait for the interrupt to settle.

    Wait for the observable postconditions because the interrupt can clear and
    set its internal event before this coroutine resumes under parallel load.
    """
    assert await _wait_until(pilot, lambda: app.app_server.turn_active, timeout=timeout)
    await pilot.press("escape")
    assert await _wait_until(
        pilot,
        lambda: app.app_server.turn_queue.paused and not app._agent_job_active(),
        timeout=timeout,
    )


class _BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__([[mock_llm_chunk(content="done")]] * 4)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await self.release.wait()
        return await super().complete(**kwargs)


def _blocked_app() -> tuple[ChartreuxApp, _BlockingBackend]:
    backend = _BlockingBackend()
    agent_loop = build_test_agent_loop(backend=backend)
    return build_test_chartreux_app(agent_loop=agent_loop), backend


def _shell_result_output(widget: ToolResultMessage) -> str:
    entry = widget._entry
    state = getattr(entry, "state", None)
    return getattr(state, "output_text", "") or ""


def _queued_texts(app: ChartreuxApp) -> list[str]:
    return [
        block.text
        for queued_turn in app.app_server.turn_queue.items
        for entry in queued_turn.entries
        if entry.role == "user"
        for block in entry.content
        if isinstance(block, SessionTextContentBlock)
    ]


async def _wait_for_queued(pilot, app: ChartreuxApp, texts: list[str]) -> None:
    """Wait until the server queue holds exactly ``texts``, in order.

    Prefer this over ``len(app._queue) == N`` / ``turn_queue.items``: the client
    count is optimistic (it also counts in-flight submits and a running-but-not-
    yet-surfaced turn) and the projection lags, so a count can be satisfied
    before the intended follow-ups are actually enqueued server-side -- letting a
    following interrupt race ahead of the enqueue. Queue content only reflects
    prompts the server has accepted.
    """
    assert await _wait_until(pilot, lambda: _queued_texts(app) == texts)


@pytest.mark.asyncio
async def test_concurrent_idle_prompts_are_serialized_by_server_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause(0.1)

        original_agent_job_active = app._agent_job_active
        forced_idle_checks = 0

        def first_two_checks_are_idle() -> bool:
            nonlocal forced_idle_checks
            if forced_idle_checks < 2:
                forced_idle_checks += 1
                return False
            return original_agent_job_active()

        with monkeypatch.context() as patcher:
            patcher.setattr(app, "_agent_job_active", first_two_checks_are_idle)
            await asyncio.gather(
                app._handle_user_message("first"), app._handle_user_message("second")
            )

        assert await _wait_until(pilot, backend.started.is_set)
        assert await _wait_until(
            pilot, lambda: len(app.app_server.turn_queue.items) == 1
        )
        assert await _wait_until(
            pilot, lambda: app.app_server.state.latest_turn is not None
        )
        assert app.app_server.state.latest_turn is not None
        assert app.app_server.state.latest_turn.queue_item_id is not None
        assert await _wait_until(pilot, lambda: app._loading_widget is not None)
        assert list(app.query(ErrorMessage)) == []

        backend.release.set()
        assert await _wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )

    assert {request[-1].content for request in backend.requests_messages} == {
        "first",
        "second",
    }


@pytest.mark.parametrize(
    "ordering", ["history-first", "promotion-first", "history-before-restore"]
)
@pytest.mark.asyncio
async def test_restored_queue_reconciles_by_id_across_orderings(ordering: str) -> None:
    posted = datetime(2026, 7, 12, 14, 32, tzinfo=UTC)
    canonical = PublicMessageEntry(
        id="canonical",
        session_id="session",
        role="user",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        content=[TextContentBlock(text="restored")],
        posted_at=posted,
    )
    app = App()
    async with app.run_test() as pilot:

        async def mount(widget, **_kwargs) -> None:
            await app.mount(widget)

        controller = _queue_controller(mount_and_scroll=mount)
        if ordering == "history-before-restore":
            controller.reconcile_history_entry(canonical)
        await controller.sync_server_queue(
            PublicTurnQueue(items=[_server_turn("queued", "restored", "canonical")])
        )
        widget = controller.widgets[0]
        assert widget.pending and widget.header.posted_at is None
        if ordering == "history-first":
            controller.reconcile_history_entry(canonical)
            assert widget.header.posted_at is None
        await controller.turn_started("queued")
        if ordering == "promotion-first":
            assert widget.header.posted_at is None
            controller.reconcile_history_entry(canonical)
        assert widget.header.posted_at == posted
        controller.reconcile_history_entry(canonical)
        await controller.turn_started("queued")
        assert widget.header.posted_at == posted
        await widget.remove()
        # A remounted history widget receives the same authority, not a new clock.
        restored = UserMessage("restored", history_entry_id="canonical")
        controller._track_message(restored)
        await app.mount(restored)
        await pilot.pause()
        assert restored.header.posted_at == posted


@pytest.mark.asyncio
async def test_retained_records_render_as_pending_queue_items() -> None:
    """C2: B.2a's pending state reaches the TUI, not only the wire.

    A retained record is a ready user turn the server holds beside the FIFO. The
    CLI renders it as a pending queued prompt so the user sees the work that
    waits, and when an admission gives it a queue place the same block follows
    the entry instead of being shown twice.
    """
    app = App()
    async with app.run_test():

        async def mount(widget, **_kwargs) -> None:
            await app.mount(widget)

        controller = _queue_controller(mount_and_scroll=mount)
        await controller.sync_server_queue(
            PublicTurnQueue(retained=[_server_turn("op-1", "record text", "op-1")])
        )
        assert [widget.get_content() for widget in controller.widgets] == [
            "record text"
        ]
        widget = controller.widgets[0]
        assert widget.pending
        assert isinstance(controller.header, QueueHeaderMessage)
        assert len(controller) == 1

        # The record takes the place an admission freed: one queue identity, one
        # rendered prompt -- the block follows the entry, it is not duplicated.
        await controller.sync_server_queue(
            PublicTurnQueue(items=[_server_turn("queued-1", "record text", "op-1")])
        )
        assert [widget.get_content() for widget in controller.widgets] == [
            "record text"
        ]
        assert controller.widgets[0] is widget
        await controller.turn_started("queued-1")
        assert controller.widgets == []
        assert controller.header is None


@pytest.mark.asyncio
async def test_editing_a_retained_record_keeps_the_pending_record_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RV-FIX-6: an edit must not hide a retained record that is still pending.

    A retained record waits beside the FIFO, so ``queue/replace`` has no queue
    item to rewrite and answers not_found. The edit therefore becomes its own
    queued prompt: the pending record keeps its visible place instead of being
    discarded while the server still holds it as ready user work.
    """
    app = build_test_chartreux_app()
    record = _server_turn("op-1", "record text", "op-1")
    queue = PublicTurnQueue(retained=[record])
    replace_calls: list[tuple[str, str]] = []

    async def enqueue_turn(
        content: str, *, message_entry_id: str | None = None, **_kwargs: object
    ) -> PublicQueuedTurn:
        nonlocal queue
        turn = _server_turn(f"queued-{len(queue.items) + 1}", content, message_entry_id)
        queue = PublicTurnQueue(items=[*queue.items, turn], retained=queue.retained)
        return turn

    async def replace_queued_turn(
        queue_item_id: str, content: str, **_kwargs: object
    ) -> PublicQueuedTurn | None:
        replace_calls.append((queue_item_id, content))
        # A retained record owns no queue item: the server answers not_found.
        return None

    # Queue selection is only offered while an agent job runs, which is exactly
    # when a retained record waits beside a full queue.
    monkeypatch.setattr(app, "_agent_job_active", lambda: True)

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        app._queue._ports = replace(
            app._queue._ports,
            enqueue_turn=enqueue_turn,
            replace_queued_turn=replace_queued_turn,
            current_turn_queue=lambda: queue,
        )
        await app._queue.sync_server_queue(PublicTurnQueue(retained=[record]))
        assert await _wait_until(pilot, lambda: len(app._queue) == 1)
        record_widget = app._queue.widgets[0]
        assert record_widget.get_content() == "record text"
        assert record_widget.pending

        app.query_one(ChatTextArea).focus()
        await pilot.press("up")
        assert await _wait_until(
            pilot, lambda: app._queue_selected_widget is record_widget
        )
        app.query_one(ChatInputContainer).post_message(
            ChatInputContainer.QueueEditSubmitted("edited")
        )

        # The TUI queue keeps both prompts pending: the record is not discarded
        # and its widget is the same block, not a re-created copy.
        assert await _wait_until(
            pilot,
            lambda: (
                app._queue.queue_item_texts() == [(0, "edited"), (1, "record text")]
            ),
        )
        assert record_widget in app._queue.widgets
        assert record_widget.pending
        assert record_widget.get_content() == "record text"
        assert not list(app.query(ErrorMessage))

        # The backend read agrees: the record still waits beside the FIFO with
        # its original text, and the edited text took a queue place of its own.
        backend_queue = app._queue._ports.current_turn_queue()
        assert [item.id for item in backend_queue.retained] == ["op-1"]
        assert backend_queue.retained[0].entries[0].content == [
            SessionTextContentBlock(text="record text")
        ]
        assert [
            entry.content for item in backend_queue.items for entry in item.entries
        ] == [[SessionTextContentBlock(text="edited")]]
        # The record was never offered to a replace that cannot land.
        assert replace_calls == []


@pytest.mark.asyncio
async def test_no_queue_header_when_empty(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test():
        headers = list(chartreux_app.query(QueueHeaderMessage))
        assert headers == []


@pytest.mark.asyncio
async def test_mount_and_scroll_ignores_streaming_message_during_shutdown(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        messages_area = chartreux_app._messages_area
        message = AssistantMessage("late response")
        messages_area._closing = True
        try:
            await chartreux_app._mount_and_scroll(message)
        finally:
            messages_area._closing = False

        assert not message.is_mounted


@pytest.mark.asyncio
async def test_queued_prompts_merge_into_one_turn() -> None:
    app, backend = _blocked_app()

    def history_message(entry_id: str) -> PublicMessageEntry | None:
        return next(
            (
                entry
                for entry in (app.app_server.state.history or [])
                if isinstance(entry, PublicMessageEntry) and entry.id == entry_id
            ),
            None,
        )

    app._queue._ports = replace(app._queue._ports, history_message=history_message)
    async with app.run_test() as pilot:
        assert app.event_handler is not None
        app.event_handler.on_user_message = app._queue.reconcile_history_entry
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)

        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        # The two queued prompts keep separate widgets but merge into a single
        # server queue item, so they are delivered as one turn, not one each.
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])
        assert len(app.app_server.turn_queue.items) == 1
        assert [
            message._content for message in app.query(UserMessage) if message.pending
        ] == ["first queued", "second queued"]
        queued_widgets = [
            message for message in app.query(UserMessage) if message.pending
        ]
        assert all(message.header.posted_at is None for message in queued_widgets)

        backend.release.set()
        assert await _wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )
        assert await _wait_until(
            pilot,
            lambda: (
                len([
                    message
                    for message in app.query(AssistantMessage)
                    if message.get_content() == "done"
                ])
                == 2
            ),
        )

        canonical = history_message(queued_widgets[0].history_entry_id or "")
        assert canonical is not None and canonical.posted_at is not None
        assert queued_widgets[0].header.posted_at == canonical.posted_at
        assert queued_widgets[1].header.posted_at is None
        assert not queued_widgets[1].header.display
        app._queue.reconcile_history_entry(canonical)
        assert queued_widgets[0].header.posted_at == canonical.posted_at

    assert [request[-1].content for request in backend.requests_messages] == [
        "block queue",
        "first queued\n\nsecond queued",
    ]


@pytest.mark.asyncio
async def test_finalize_does_not_tear_down_promoted_follow_up_turn() -> None:
    class GatedBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__([
                [mock_llm_chunk(content="first done")],
                [mock_llm_chunk(content="second done")],
            ])
            self.started = [asyncio.Event(), asyncio.Event()]
            self.release = [asyncio.Event(), asyncio.Event()]
            self.calls = 0

        async def complete(self, **kwargs):
            index = self.calls
            self.calls += 1
            self.started[index].set()
            await self.release[index].wait()
            return await super().complete(**kwargs)

    backend = GatedBackend()
    app = build_test_chartreux_app(agent_loop=build_test_agent_loop(backend=backend))

    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("first"))
        assert await _wait_until(pilot, backend.started[0].is_set)
        chat_input.post_message(ChatInputContainer.Submitted("second"))
        assert await _wait_until(pilot, lambda: len(app._queue) == 1)

        backend.release[0].set()
        assert await _wait_until(pilot, backend.started[1].is_set)
        assert await _wait_until(pilot, lambda: app._loading_widget is not None)
        assert await _wait_until(pilot, lambda: app.app_server.turn_active)

        backend.release[1].set()
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )
        assert await _wait_until(
            pilot,
            lambda: any(
                "second done" in message.get_content()
                for message in app.query(AssistantMessage)
            ),
        )


@pytest.mark.asyncio
async def test_submit_during_queue_promotion_enqueues_behind_promoted_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, backend = _blocked_app()
    promotion_waiting = asyncio.Event()
    release_promotion = asyncio.Event()
    promotion_count = 0
    original_emit_queue_updated = TurnController._emit_queue_updated

    async def gate_promoted_queue_update(self: TurnController) -> None:
        nonlocal promotion_count
        await original_emit_queue_updated(self)
        active_turn = self.active_turn
        if (
            active_turn is not None
            and active_turn.queue_item_id is not None
            and not self.queue_state.items
        ):
            promotion_count += 1
            if promotion_count == 2:
                promotion_waiting.set()
                await release_promotion.wait()

    monkeypatch.setattr(
        TurnController, "_emit_queue_updated", gate_promoted_queue_update
    )

    try:
        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInputContainer)
            chat_input.post_message(ChatInputContainer.Submitted("first"))
            assert await _wait_until(pilot, backend.started.is_set)
            chat_input.post_message(ChatInputContainer.Submitted("second"))
            assert await _wait_until(pilot, lambda: len(app._queue) == 1)

            backend.release.set()
            assert await _wait_until(pilot, promotion_waiting.is_set)
            assert await _wait_until(pilot, lambda: not app._agent_job_active())
            assert app._queue.has_server_work

            chat_input.post_message(ChatInputContainer.Submitted("third"))
            assert await _wait_until(pilot, lambda: _queued_texts(app) == ["third"])

            release_promotion.set()
            assert await _wait_until(pilot, lambda: len(backend.requests_messages) == 3)
            assert await _wait_until(
                pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
            )
    finally:
        release_promotion.set()

    assert [request[-1].content for request in backend.requests_messages] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_ctrl_c_removes_newest_server_prompt() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        await pilot.press("ctrl+c")
        await _wait_for_queued(pilot, app, ["first queued"])

        backend.release.set()
        assert await _wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        assert await _wait_until(
            pilot,
            lambda: (
                not app._agent_job_active()
                and len(app._queue) == 0
                and len(app.query(AssistantMessage)) == 2
            ),
        )

    assert backend.requests_messages[-1][-1].content == "first queued"


@pytest.mark.asyncio
async def test_prompt_submitted_while_shell_runs_is_queued(
    chartreux_app: ChartreuxApp,
) -> None:
    """Item-5 A.4: prompts and shell inputs share one FIFO queue."""
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 2"
        await pilot.press("enter")
        assert await _wait_until(
            pilot, lambda: chartreux_app._bash_task is not None, timeout=2.0
        )

        chat_input.value = "after shell"
        await pilot.press("enter")

        assert chat_input.value == ""
        await _wait_for_queued(pilot, chartreux_app, ["after shell"])
        assert not any(
            "cannot be queued" in notification.message
            for notification in chartreux_app._notifications
        )

        await pilot.press("escape")
        assert await _wait_until(
            pilot, lambda: chartreux_app._bash_task is None, timeout=5.0
        )


@pytest.mark.asyncio
async def test_interrupt_gives_immediate_loading_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A running turn with the loading spinner visible.
    *Do*: Press Escape with the server interrupt held open.
    *Assert*: The loading status flips to "Interrupting" immediately, before the
    cancel unwinds, so the keypress is acknowledged right away.
    """
    app, backend = _blocked_app()
    release_interrupt = asyncio.Event()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()

        original_interrupt = app.app_server.interrupt

        async def gated_interrupt() -> None:
            await release_interrupt.wait()
            await original_interrupt()

        monkeypatch.setattr(app.app_server, "interrupt", gated_interrupt)

        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block turn"))
        assert await _wait_until(pilot, backend.started.is_set)
        assert await _wait_until(pilot, lambda: app._loading_widget is not None)
        assert await _wait_until(
            pilot, lambda: app.app_server.turn_active and not app._queue
        )

        await pilot.press("escape")

        # Feedback must land while the (gated) cancel is still in flight and the
        # loading widget is still on screen -- not only after it tears down.
        assert await _wait_until(
            pilot,
            lambda: (
                app._loading_widget is not None
                and app._loading_widget._base_status == INTERRUPTING_LOADING_STATUS
            ),
        )

        release_interrupt.set()
        assert await _wait_until(pilot, lambda: bool(list(app.query(InterruptMessage))))


@pytest.mark.asyncio
async def test_interrupt_terminates_client_driven_turn() -> None:
    """Interrupting a client-owned turn (retry/auto-retry) pauses the queue.

    Coverage for the interrupt path when a client ``_agent_task`` is live: the
    server turn must end and the queued follow-up must stay paused rather than
    promote and run. (In this in-process harness closing the client stream also
    ends the server turn; the unconditional server interrupt in ``_interrupt_turn``
    hardens the case where it does not, e.g. under the real runtime.)
    """
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()

        # Drive a client-owned turn, mirroring the retry / auto-retry path.
        app._agent_task = asyncio.create_task(app._handle_turn("client turn"))
        assert await _wait_until(pilot, backend.started.is_set)

        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("queued follow-up"))
        await _wait_for_queued(pilot, app, ["queued follow-up"])

        await pilot.press("escape")

        # The server turn must reach a terminal state and pause the queue rather
        # than staying IN_PROGRESS and promoting the follow-up.
        assert await _wait_until(
            pilot, lambda: not app.app_server.turn_active, timeout=5.0
        )
        assert await _wait_until(
            pilot, lambda: app.app_server.turn_queue.paused, timeout=5.0
        )
        assert _queued_texts(app) == ["queued follow-up"]


@pytest.mark.asyncio
async def test_empty_enter_resumes_server_prompts_after_interrupt() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app.app_server.turn_queue.paused)
        assert _queued_texts(app) == ["first queued\n\nsecond queued"]

        chat_input.value = ""
        await pilot.press("enter")
        assert await _wait_until(
            pilot,
            lambda: (
                not app.app_server.turn_queue.paused
                and len(backend.requests_messages) == 1
                and len(app._queue) == 0
            ),
            timeout=5.0,
        )

    assert [request[-1].content for request in backend.requests_messages] == [
        "first queued\n\nsecond queued"
    ]


@pytest.mark.asyncio
async def test_empty_bash_does_not_resume_server_prompts_after_interrupt() -> None:
    """*Prepare*: An interrupted turn with one queued prompt left paused.
    *Do*: Submit a bare shell prefix while the queue is paused.
    *Assert*: The refused record is queued and the queued prompt stays paused.
    """
    # Prepare
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        assert await _wait_until(pilot, app._agent_job_active)
        chat_input.post_message(ChatInputContainer.Submitted("queued prompt"))
        await _wait_for_queued(pilot, app, ["queued prompt"])

        await _press_escape_and_wait_for_settle(pilot, app)
        assert app.app_server.turn_queue.paused
        assert not app._agent_job_active()

        # Do
        chat_input.value = "!"
        await pilot.press("enter")
        # Item 5 B.1: the bare `!` still reaches the shell resource, so its
        # refused record exists -- but it executes nothing and is not the
        # user's queued work, so it does not resume the paused queue.
        assert await _wait_until(
            pilot,
            lambda: (
                len(_queued_texts(app)) == 2
                and "Manual `!` command result from the user." in _queued_texts(app)[1]
            ),
        )
        await pilot.pause(0.1)

        # Assert
        assert app.app_server.turn_queue.paused
        assert _queued_texts(app)[0] == "queued prompt"
        assert "Shell command cannot be empty" in _queued_texts(app)[1]
        assert backend.calls == 1
        # The submission is consumed: the record is the command's whole effect,
        # so a restored input would let a re-submit mint a second one.
        assert chat_input.value == ""
        assert len(_queued_texts(app)) == 2


@pytest.mark.asyncio
async def test_empty_bash_consumes_the_submission_while_agent_busy() -> None:
    """*Prepare*: A blocked turn, so the epoch is busy.
    *Do*: Submit a bare shell prefix while the agent is busy.
    *Assert*: The submission is consumed and exactly one refusal record is queued.
    """
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        assert await _wait_until(
            pilot, lambda: app.app_server.turn_active and not app._queue
        )

        chat_input.value = "!"
        await pilot.press("enter")
        # The refused record is the whole effect of the bare `!`; the input is
        # not handed back, so the record cannot be produced twice.
        assert await _wait_until(
            pilot,
            lambda: any(
                "Manual `!` command result from the user." in text
                for text in _queued_texts(app)
            ),
        )
        assert chat_input.value == ""
        assert not any(
            "cannot be queued" in notification.message
            for notification in app._notifications
        )
        assert (
            len([
                text
                for text in _queued_texts(app)
                if "Manual `!` command result from the user." in text
            ])
            == 1
        )
        backend.release.set()


@pytest.mark.asyncio
async def test_rewind_skips_paused_server_prompt() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("queued prompt"))
        await _wait_for_queued(pilot, app, ["queued prompt"])

        await _press_escape_and_wait_for_settle(pilot, app)
        assert app.app_server.turn_queue.paused
        assert not app._agent_job_active()

        await pilot.press("escape", "escape")
        assert await _wait_until(
            pilot, lambda: app._rewind_highlighted_widget is not None
        )

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "block queue"


@pytest.mark.asyncio
async def test_bash_submitted_during_running_bash_is_queued(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 2"
        await pilot.press("enter")

        assert await _wait_until(
            pilot, lambda: chartreux_app._bash_task is not None, timeout=2.0
        )

        chat_input.value = "!echo queued"
        await pilot.press("enter")

        assert chat_input.value == ""
        assert not any(
            "cannot be queued" in notification.message
            for notification in chartreux_app._notifications
        )
        # Accepted as user input: it runs after the first command instead of
        # being refused, whether it is still queued or already executed.
        assert await _wait_until(
            pilot,
            lambda: any(
                "queued" in str(_shell_result_output(widget))
                for widget in chartreux_app.query(ToolResultMessage)
                if widget.tool_name == "shell"
            ),
            timeout=5.0,
        )
        await pilot.press("escape")
        assert await _wait_until(
            pilot, lambda: chartreux_app._bash_task is None, timeout=5.0
        )


@pytest.mark.asyncio
async def test_bash_command_queued_while_agent_busy() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        assert await _wait_until(
            pilot, lambda: app.app_server.turn_active and not app._queue
        )

        chat_input.post_message(ChatInputContainer.Submitted("!echo queued"))

        assert await _wait_until(
            pilot,
            lambda: (
                [
                    entry.command
                    for item in app.app_server.turn_queue.items
                    for entry in item.entries
                    if entry.role == "shell"
                ]
                == ["echo queued"]
            ),
        )
        assert chat_input.value == ""
        assert not any(
            "cannot be queued" in notification.message
            for notification in app._notifications
        )
        backend.release.set()


@pytest.mark.asyncio
async def test_slash_command_rejected_while_agent_busy() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        assert await _wait_until(
            pilot, lambda: app.app_server.turn_active and not app._queue
        )

        chat_input.post_message(ChatInputContainer.Submitted("/clear"))

        assert await _wait_until(
            pilot,
            lambda: any(
                "Slash commands cannot be queued" in notification.message
                for notification in app._notifications
            ),
        )
        assert chat_input.value == "/clear"
        assert not app._queue
        backend.release.set()


def test_slash_command_message_displays_command_with_leading_slash() -> None:
    # The migrated role label is "You", so the slash stays in the content.
    assert SlashCommandMessage("/clear").get_content() == "/clear"
    assert SlashCommandMessage("model sonnet").get_content() == "/model sonnet"


@pytest.mark.asyncio
async def test_theme_selected_while_idle_persists_immediately(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        picker = ThemePickerApp(
            theme_names=sorted_theme_names(), current_theme=chartreux_app.config.theme
        )
        await chartreux_app._switch_from_input(picker)
        assert await _wait_until(
            pilot, lambda: bool(list(chartreux_app.query(ThemePickerApp))), timeout=2.0
        )

        target = "light" if chartreux_app.config.theme != "light" else "dark"
        chartreux_app.post_message(ThemePickerApp.ThemeSelected(target))

        assert await _wait_until(
            pilot, lambda: chartreux_app.config.theme == target, timeout=2.0
        )
        assert not chartreux_app._queue


@pytest.mark.asyncio
async def test_quit_warning_shows_queue_count() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("a"))
        chat_input.post_message(ChatInputContainer.Submitted("b"))
        assert await _wait_until(pilot, lambda: len(app._queue) == 2)

        warning = app._queue.quit_warning_extra()
        assert warning == "2 queued messages will be discarded"

        assert await app._queue.pop_last()
        warning = app._queue.quit_warning_extra()
        assert warning == "1 queued message will be discarded"

        assert await app._queue.pop_last()
        assert app._queue.quit_warning_extra() == ""
        backend.release.set()


@pytest.mark.asyncio
async def test_theme_command_rejected_while_agent_busy() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        assert await _wait_until(
            pilot, lambda: app.app_server.turn_active and not app._queue
        )

        chat_input.post_message(ChatInputContainer.Submitted("/theme"))

        assert await _wait_until(
            pilot,
            lambda: any(
                "Slash commands cannot be queued" in notification.message
                for notification in app._notifications
            ),
        )
        assert chat_input.value == "/theme"
        assert not list(app.query(ThemePickerApp))
        assert not app._queue
        backend.release.set()


@pytest.mark.asyncio
async def test_side_channel_exit_not_rejected_while_busy(
    chartreux_app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 2"
        await pilot.press("enter")

        assert await _wait_until(
            pilot, lambda: chartreux_app._bash_task is not None, timeout=2.0
        )

        exit_calls: list[dict] = []

        def fake_exit(self: ChartreuxApp, **kwargs: object) -> None:
            exit_calls.append(kwargs)

        monkeypatch.setattr(ChartreuxApp, "_exit_app", fake_exit)

        chat_input.post_message(ChatInputContainer.Submitted("/exit"))

        assert await _wait_until(pilot, lambda: len(exit_calls) > 0, timeout=2.0)
        assert not any(
            "cannot be queued" in n.message for n in chartreux_app._notifications
        )

        await pilot.press("escape")
        await _wait_until(pilot, lambda: chartreux_app._bash_task is None, timeout=5.0)


def _gate_turn_promotion(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Hold the first queued promotion open until the returned event is set.

    Freezes the enqueue -> promote window so a just-submitted prompt can be
    inspected before it becomes the running turn. Only gates when there is
    actually something to promote, so startup is unaffected.
    """
    release = asyncio.Event()
    original = TurnController._promote_next
    gated = {"done": False}

    async def gated_promote(self: TurnController) -> None:
        if not gated["done"] and self._turn_queue.peek_next() is not None:
            gated["done"] = True
            await release.wait()
        await original(self)

    monkeypatch.setattr(TurnController, "_promote_next", gated_promote)
    return release


@pytest.mark.asyncio
async def test_idle_submit_does_not_flash_as_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: An idle session (no turn running, empty queue).
    *Do*: Submit a prompt while its promotion is held open.
    *Assert*: It renders as a normal message, never with queued styling.
    """
    app = build_test_chartreux_app()
    release_promotion = _gate_turn_promotion(monkeypatch)
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()

        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("solo message"))
        assert await _wait_until(
            pilot, lambda: len(app.app_server.turn_queue.items) == 1
        )
        await pilot.pause(0.1)

        # In the enqueue -> promote window the prompt must not look queued.
        messages = [
            message
            for message in app.query(UserMessage)
            if message.get_content() == "solo message"
        ]
        assert messages, "the submitted message should be mounted"
        assert all(not message.pending for message in messages)
        assert list(app.query(QueueHeaderMessage)) == []

        release_promotion.set()
        assert await _wait_until(
            pilot,
            lambda: not app._agent_job_active() and len(app._queue) == 0,
            timeout=5.0,
        )


@pytest.mark.asyncio
async def test_idle_submit_shows_loading_before_turn_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: An idle session with promotion held open.
    *Do*: Submit a prompt.
    *Assert*: The loading indicator appears during the enqueue -> promote gap,
    not only once the server emits TurnStarted.
    """
    app = build_test_chartreux_app()
    release_promotion = _gate_turn_promotion(monkeypatch)
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()

        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("solo message"))

        # Feedback must appear while the promotion is still gated (before the
        # TurnStarted event would otherwise mount the loading widget).
        assert await _wait_until(pilot, lambda: app._loading_widget is not None)
        assert not app.app_server.turn_active

        release_promotion.set()
        assert await _wait_until(
            pilot,
            lambda: not app._agent_job_active() and len(app._queue) == 0,
            timeout=5.0,
        )


@pytest.mark.asyncio
async def test_idle_submit_is_in_flight_and_interruptible_before_turn_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: An idle submit whose promotion is held open (enqueue->promote gap).
    *Do*: Press Escape during that gap.
    *Assert*: The app treats the turn as in flight (interruptible, not idle), and
    the interrupt takes effect once the turn surfaces rather than being a no-op.
    """
    app, backend = _blocked_app()
    release_promotion = _gate_turn_promotion(monkeypatch)
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()

        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("solo message"))
        assert await _wait_until(
            pilot, lambda: len(app.app_server.turn_queue.items) == 1
        )

        # Consolidated in-flight state: busy/interruptible even though the server
        # turn has not started yet.
        assert app._agent_job_active()
        assert not app.app_server.turn_active

        # Escape in the gap must not be a no-op that falls through to rewind.
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert app._rewind_highlighted_widget is None

        # Let the turn surface; the queued interrupt then takes effect.
        release_promotion.set()
        assert await _wait_until(
            pilot, lambda: bool(list(app.query(InterruptMessage))), timeout=5.0
        )
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active(), timeout=5.0
        )
        backend.release.set()


@pytest.mark.asyncio
async def test_aborted_submit_clears_in_flight_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An aborted prompt (prep returns None) must not leave a stuck busy spinner."""
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()

        async def _abort(_content: str) -> None:
            return None

        monkeypatch.setattr(app, "_prepare_prompt_or_abort", _abort)
        await app._handle_user_message("please abort")
        await pilot.pause(0.05)

        assert not app._pending_turn
        assert not app._agent_job_active()
        assert app._loading_widget is None


def _gate_turn_finalize(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Hold the interrupted turn open until the returned event is set.

    Gating ``_finalize_turn`` keeps the turn ``IN_PROGRESS`` (so ``turn_active``
    stays True and the queue is not yet paused) after the cancellation is
    requested, deterministically reproducing the Esc -> Enter race window.
    """
    release = asyncio.Event()
    original = TurnController._finalize_turn

    async def gated_finalize(self: TurnController, *args: object, **kwargs: object):
        await release.wait()
        return await original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(TurnController, "_finalize_turn", gated_finalize)
    return release


@pytest.mark.asyncio
async def test_fast_escape_then_empty_enter_resumes_paused_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A running turn with two queued prompts.
    *Do*: Interrupt and press empty Enter before the pause propagates.
    *Assert*: The queue resumes and both prompts run (the Enter is not lost).
    """
    app, backend = _blocked_app()
    release_finalize = _gate_turn_finalize(monkeypatch)
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        # The two prompts merge into a single queued item.
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        # Interrupt: finalize is gated, so the turn stays active and the queue
        # is not paused yet -- the exact stale-state window the bug hit.
        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app.app_server.turn_active)
        assert not app.app_server.turn_queue.paused

        # Fast empty Enter while the interrupt has not settled.
        chat_input.value = ""
        await pilot.press("enter")
        await pilot.pause(0.1)

        # Let the interrupt settle; the deferred Enter must resume the queue.
        release_finalize.set()
        assert await _wait_until(
            pilot,
            lambda: (
                not app.app_server.turn_queue.paused
                and len(backend.requests_messages) == 1
                and len(app._queue) == 0
            ),
            timeout=5.0,
        )

    assert [request[-1].content for request in backend.requests_messages] == [
        "first queued\n\nsecond queued"
    ]


@pytest.mark.asyncio
async def test_fast_escape_then_message_starts_new_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A running turn with an empty queue.
    *Do*: Interrupt and submit a new message before the interrupt settles.
    *Assert*: The message runs as a normal turn, never a trapped queued item.
    """
    app, backend = _blocked_app()
    release_finalize = _gate_turn_finalize(monkeypatch)
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block turn"))
        assert await _wait_until(pilot, backend.started.is_set)

        # Interrupt: finalize gated so the turn is still active on submit.
        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app.app_server.turn_active)

        # Submit a new instruction during the unsettled interrupt window.
        chat_input.value = "do this instead"
        await pilot.press("enter")
        await pilot.pause(0.1)

        release_finalize.set()
        assert await _wait_until(
            pilot,
            lambda: (
                not app.app_server.turn_queue.paused
                and any(
                    request[-1].content == "do this instead"
                    for request in backend.requests_messages
                )
                and len(app._queue) == 0
            ),
            timeout=5.0,
        )

        # It ran directly, never left behind as a queued/pending message.
        assert _queued_texts(app) == []


@pytest.mark.asyncio
async def test_fast_escape_then_two_messages_preserve_fifo_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A running turn with an empty queue.
    *Do*: Interrupt and submit two messages during the unsettled window.
    *Assert*: They run in submit order (deferred dispatch stays FIFO).
    """
    app, backend = _blocked_app()
    release_finalize = _gate_turn_finalize(monkeypatch)
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block turn"))
        assert await _wait_until(pilot, backend.started.is_set)

        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app.app_server.turn_active)

        # Two rapid submits while the interrupt has not settled: both defer.
        chat_input.post_message(ChatInputContainer.Submitted("first message"))
        chat_input.post_message(ChatInputContainer.Submitted("second message"))
        await pilot.pause(0.1)

        release_finalize.set()
        assert await _wait_until(
            pilot,
            lambda: (
                not app.app_server.turn_queue.paused
                and {request[-1].content for request in backend.requests_messages}
                == {"first message", "second message"}
                and len(app._queue) == 0
            ),
            timeout=5.0,
        )

    assert [request[-1].content for request in backend.requests_messages] == [
        "first message",
        "second message",
    ]


@pytest.mark.asyncio
async def test_noop_interrupt_settles_and_does_not_stall_next_submit() -> None:
    """*Prepare*: An idle app (no active turn), settle window opened by Escape.
    *Do*: Run the interrupt worker; it finds nothing to interrupt.
    *Assert*: The window is settled so a following submit does not wait out the
    full _await_interrupt_settled timeout.
    """
    app, _backend = _blocked_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause(0.1)

        # The Escape handler opens the settle window before the worker runs; the
        # turn then finishes (here: was already idle), so _interrupt_turn is a
        # no-op. Nothing else will emit a turn/finalize event to settle it.
        app._begin_interrupt_settle()
        assert app._interrupt_pending
        await app._interrupt_turn()

        # The no-op interrupt must close the window itself.
        assert not app._interrupt_pending
        assert app._interrupt_settled.is_set()

        # A racing submit must not wait out the 30s timeout.
        await asyncio.wait_for(app._await_interrupt_settled(), timeout=1.0)


@pytest.mark.asyncio
async def test_noop_interrupt_settles_even_with_unpaused_queue_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A no-op interrupt while accepted queue items are not paused.
    *Do*: Run the interrupt worker; it finds nothing to interrupt.
    *Assert*: It settles anyway. This path never issues interrupt(), so the
    server never pauses -- waiting for a pause (as _maybe_settle_interrupt does)
    would strand a racing submit until the queue drains or the 30s timeout.
    """
    app, _backend = _blocked_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause(0.1)

        # Simulate accepted-but-unpaused queue items with no active turn: the
        # exact state where the conditional settle would keep the window open.
        queue_cls = type(app._queue)
        monkeypatch.setattr(queue_cls, "has_removable", property(lambda self: True))
        monkeypatch.setattr(queue_cls, "paused", property(lambda self: False))
        assert not app._agent_job_active()

        app._begin_interrupt_settle()
        await app._interrupt_turn()

        assert not app._interrupt_pending
        assert app._interrupt_settled.is_set()
        await asyncio.wait_for(app._await_interrupt_settled(), timeout=1.0)


async def _steer_spy(app: ChartreuxApp) -> list[str]:
    """Record the content of every steer (turn/steer) the app issues."""
    steered: list[str] = []
    original = app.app_server.inject_user_context

    async def spy(content: str, **kwargs):
        steered.append(content)
        return await original(content, **kwargs)

    app.app_server.inject_user_context = spy  # type: ignore[method-assign]
    return steered


@pytest.mark.asyncio
async def test_ctrl_enter_steers_queued_prompts_into_active_turn() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)

        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        steered = await _steer_spy(app)
        app.query_one(ChatTextArea).focus()
        await pilot.pause()
        await pilot.press("ctrl+enter")

        # The turn is still blocked, so the queue can only empty via steering,
        # not via normal promotion (which happens after a turn ends).
        assert await _wait_until(pilot, lambda: _queued_texts(app) == [])
        assert steered == ["first queued\n\nsecond queued"]
        assert not any(message.pending for message in app.query(UserMessage))
        assert not list(app.query(QueueHeaderMessage))

        backend.release.set()
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )


@pytest.mark.asyncio
async def test_empty_enter_steers_queued_prompts_into_active_turn() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)

        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        steered = await _steer_spy(app)
        chat_input.value = ""
        app.query_one(ChatTextArea).focus()
        await pilot.pause()
        await pilot.press("enter")

        assert await _wait_until(pilot, lambda: _queued_texts(app) == [])
        assert steered == ["first queued\n\nsecond queued"]
        assert not any(message.pending for message in app.query(UserMessage))
        assert not list(app.query(QueueHeaderMessage))

        backend.release.set()
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )


@pytest.mark.asyncio
async def test_fast_escape_then_ctrl_enter_resumes_paused_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, backend = _blocked_app()
    release_finalize = _gate_turn_finalize(monkeypatch)
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app.app_server.turn_active)

        chat_input.value = ""
        app.query_one(ChatTextArea).focus()
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause(0.1)

        release_finalize.set()
        assert await _wait_until(
            pilot,
            lambda: (
                not app.app_server.turn_queue.paused
                and len(backend.requests_messages) == 1
                and len(app._queue) == 0
            ),
            timeout=5.0,
        )

    assert [request[-1].content for request in backend.requests_messages] == [
        "first queued\n\nsecond queued"
    ]


@pytest.mark.asyncio
async def test_steer_after_ended_turn_does_not_drop_queue() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        async def ended_turn_inject(*_args: object, **_kwargs: object) -> list[object]:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.CONFLICT,
                    message="No active turn",
                    data={"steerCommitted": False},
                )
            )

        app.app_server.inject_user_context = ended_turn_inject  # type: ignore[method-assign]

        # The turn ended, so the steer raises; the block is re-enqueued (fresh
        # idempotency key) rather than dropped, and no error is surfaced.
        assert not await app._steer_queued_now()
        assert not list(app.query(ErrorMessage))
        assert app._queue.has_removable
        assert [widget.get_content() for widget in app._queue.widgets] == [
            "first queued",
            "second queued",
        ]
        assert all(message.pending for message in app._queue.widgets)

        backend.release.set()
        # The follow-ups were never delivered mid-turn, so they promote and run
        # as the next turn instead of being dropped.
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )

    assert [request[-1].content for request in backend.requests_messages] == [
        "block queue",
        "first queued\n\nsecond queued",
    ]


@pytest.mark.asyncio
async def test_steer_requests_invoked_skill_injection() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        await _wait_for_queued(pilot, app, ["first queued"])

        calls: list[dict[str, object]] = []
        original = app.app_server.inject_user_context

        async def spy(content: str, **kwargs: object) -> list[object]:
            calls.append({"content": content, **kwargs})
            return await original(content, **kwargs)  # type: ignore[arg-type]

        app.app_server.inject_user_context = spy  # type: ignore[method-assign]

        assert await app._steer_queued_now()

        # A steered queued skill must load its body, matching what promotion
        # would do, so the steer requests skill injection.
        assert calls and calls[0]["inject_invoked_skill"] is True

        backend.release.set()
        assert await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_unresolved_steering_queue_edit_does_not_copy_and_remove_discards(
    size: tuple[int, int],
) -> None:
    app = build_test_chartreux_app()

    async def disconnected(*args, **kwargs):
        raise OSError("delivery unknown")

    async with app.run_test(size=size) as pilot:
        app._queue._ports = replace(app._queue._ports, steer_turn=disconnected)
        with pytest.raises(RuntimeError, match="unresolved"):
            await app._queue.steer_prepared(
                "stuck",
                prepared_prompt=PreparedPrompt(
                    display_text="stuck", prompt_text="stuck"
                ),
                expected_turn_id="retired",
            )
        chat_input = app.query_one(ChatInputContainer)
        app.query_one(ChatTextArea).focus()
        await pilot.press("up")
        assert await _wait_until(pilot, lambda: app._queue_selected_widget is not None)
        chat_input.post_message(ChatInputContainer.QueueEditSubmitted("changed"))
        assert await _wait_until(pilot, lambda: bool(app.query(ErrorMessage)))
        assert app._queue.queue_item_texts() == [(0, "stuck")]
        assert not app.app_server.turn_queue.items
        chat_input.post_message(ChatInputContainer.QueueRemoveRequested())
        assert await _wait_until(pilot, lambda: not app._queue)
        assert not app._queue.has_unresolved_steering


@pytest.mark.asyncio
async def test_empty_steer_is_noop_while_navigating_queue() -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("block queue"))
        assert await _wait_until(pilot, backend.started.is_set)
        chat_input.post_message(ChatInputContainer.Submitted("first queued"))
        chat_input.post_message(ChatInputContainer.Submitted("second queued"))
        await _wait_for_queued(pilot, app, ["first queued\n\nsecond queued"])

        # Enter queue navigation (selection) mode; empty Enter there is a queue
        # action, not a steer, so steering must be a no-op (matching Ctrl+Enter).
        app.query_one(ChatTextArea).focus()
        await pilot.press("up")
        body = app.query(ChatInputContainer)[0]._body
        assert await _wait_until(pilot, lambda: body is not None and body.in_queue_mode)

        assert not await app._steer_queued_now()
        assert _queued_texts(app) == ["first queued\n\nsecond queued"]
        assert all(message.pending for message in app._queue.widgets)

        backend.release.set()
        assert await _wait_until(pilot, lambda: not app.app_server.turn_active)
