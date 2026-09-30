from __future__ import annotations

from unittest.mock import patch

from chartreux.cli.textual_ui.widgets.context_progress import (
    ContextProgress,
    TokenState,
)


def test_context_progress_skips_updates_when_rendered_text_is_unchanged() -> None:
    widget = ContextProgress()

    with patch.object(widget, "update", wraps=widget.update) as update:
        widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=12_001))
        widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=12_999))
        assert update.call_count == 1
        assert update.call_args.args == ("ctx 12k/200k",)

        widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=14_000))

    assert update.call_count == 2
    assert update.call_args.args == ("ctx 14k/200k",)


def test_context_progress_omits_unavailable_usage() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=0))

    assert str(widget.render()) == ""


def test_context_progress_marks_unknown_usage_after_compaction() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=-1))

    assert str(widget.render()) == ""


def test_context_progress_uses_compact_integer_format_for_used_tokens() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=200_000, current_tokens=12_500))

    assert str(widget.render()) == "ctx 12k/200k"


def test_context_progress_uses_compact_integer_k_format() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=568_000, current_tokens=170_000))

    assert str(widget.render()) == "ctx 170k/568k"


def test_context_progress_uses_compact_integer_m_format() -> None:
    widget = ContextProgress()

    widget.watch_tokens(TokenState(max_tokens=40_000_000, current_tokens=35_900_000))

    assert str(widget.render()) == "ctx 35.9M/40.0M"
