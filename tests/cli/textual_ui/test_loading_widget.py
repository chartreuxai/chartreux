from __future__ import annotations

import pytest

from chartreux.cli.textual_ui.widgets.loading import (
    DEFAULT_LOADING_STATUS,
    INTERRUPTING_LOADING_STATUS,
    THINKING_LOADING_STATUS,
    LoadingWidget,
)
from tests.cli.textual_ui.test_message_queue_ui import _wait_until
from tests.cli.textual_ui.test_wait_steering_app import waiting_app as waiting_app


def test_status_is_semantic_and_uses_running_wording() -> None:
    widget = LoadingWidget(status="Discovering models")
    assert widget.status == "Discovering models"
    assert widget._build_status_text() == "Running: Discovering models…"
    widget.set_status(THINKING_LOADING_STATUS)
    assert widget.status == THINKING_LOADING_STATUS


def test_interrupting_status_sticks_against_late_streaming_updates() -> None:
    """Once interrupting, late streaming status updates must not overwrite it.

    A turn keeps streaming until the cancel propagates, and the event handler
    drives set_status("Thinking"/"Generating") on those events. Without a latch
    they clobber the "Interrupting" label, so the interrupt looks ignored.
    """
    widget = LoadingWidget()
    widget.set_status(INTERRUPTING_LOADING_STATUS)

    widget.set_status(THINKING_LOADING_STATUS)
    widget.set_status(DEFAULT_LOADING_STATUS)
    widget.set_status("Reading file")

    assert widget._base_status == INTERRUPTING_LOADING_STATUS


def test_status_updates_apply_before_interrupting() -> None:
    widget = LoadingWidget()
    widget.set_status(THINKING_LOADING_STATUS)
    assert widget._base_status == THINKING_LOADING_STATUS
    widget.set_status(INTERRUPTING_LOADING_STATUS)
    assert widget._base_status == INTERRUPTING_LOADING_STATUS


def test_action_required_status_holds_while_preserving_latest_progress() -> None:
    widget = LoadingWidget(status="Running command")

    widget.begin_action_required("Waiting for approval to run command")
    widget.set_status(DEFAULT_LOADING_STATUS)

    assert widget.base_status == "Waiting for approval to run command"
    assert widget._pause_start is not None

    widget.end_action_required()

    assert widget.base_status == DEFAULT_LOADING_STATUS
    assert widget._pause_start is None


def test_next_action_required_status_does_not_reset_saved_progress() -> None:
    widget = LoadingWidget(status="Running command")

    widget.begin_action_required("Waiting for first approval")
    widget.begin_action_required("Waiting for second approval")
    widget.end_action_required()

    assert widget.base_status == "Running command"


def test_queue_hint_labels_next_turn_submission() -> None:
    widget = LoadingWidget()
    widget.set_queue_count(2)

    hint = widget._format_hint(10)

    assert "Enter" in hint
    assert "queues next turn" in hint
    assert "to cancel last queued message" in hint


def test_hint_without_queue_labels_first_submission() -> None:
    widget = LoadingWidget()

    hint = widget._format_hint(10)

    assert "Enter" in hint
    assert "queues next turn" in hint


@pytest.mark.parametrize("queued_count", [0, 2])
def test_waiting_only_hint_labels_current_turn_steering(queued_count: int) -> None:
    widget = LoadingWidget()
    widget.set_queue_count(queued_count)
    widget.set_waiting_only(True)

    hint = widget._format_hint(10)

    assert "Enter" in hint
    assert "steers current turn" in hint
    assert "queues next turn" not in hint
    if queued_count:
        assert "to cancel last queued message" in hint

    widget.set_waiting_only(False)
    assert "queues next turn" in widget._format_hint(10)


@pytest.mark.asyncio
async def test_published_waiting_only_state_updates_loading_hint(waiting_app) -> None:
    app, pilot, *_ = waiting_app

    def waiting_hint_visible() -> bool:
        loading = app._loading_widget
        return (
            loading is not None
            and loading.hint_widget is not None
            and "steers current turn" in str(loading.hint_widget.content)
        )

    assert await _wait_until(pilot, waiting_hint_visible)
    await app._enqueue_prompt_with_resources("older queued input")
    await pilot.pause()
    assert waiting_hint_visible()
    loading = app._loading_widget
    assert loading is not None
    assert "to cancel last queued message" in loading._format_hint(0)
