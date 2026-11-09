"""The consent modal's full field set and its mandatory re-preview discipline.

``PlanConsentScreen`` is the last surface the user sees before a plan handoff
replaces the conversation, so it must show the whole design §1 field set —
episode identity, lifecycle state, path, full content, extracted summary, byte
count, SHA-256 consent digest, executor selection — with Cancel as the default.
Consent applies to the bytes last shown: a draft that changes between preview
and confirm is re-previewed and needs a fresh confirm.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button

from chartreux.cli.textual_ui.screens.plan_consent import (
    CANCEL_BUTTON_ID,
    START_BUTTON_ID,
    PlanConsentScreen,
)
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.utils.io import decode_safe

_EPISODE_ID = "ep-0123456789abcdef"
_SUMMARY = "Do the thing carefully."


def _plan_text(*, steps: int = 60, summary: str = _SUMMARY) -> bytes:
    """A plan document with CRLF newlines, so the decode is provably lossy."""
    step_lines = "".join(f"- step {index}\r\n" for index in range(1, steps + 1))
    return (
        f"# Plan\r\n\r\n## Summary\r\n\r\n{summary}\r\n\r\n## Steps\r\n\r\n{step_lines}"
    ).encode()


_RAW = _plan_text()


class _ModalApp(App[None]):
    def __init__(self, screen: PlanConsentScreen) -> None:
        super().__init__()
        self.config = SimpleNamespace(ascii_chrome=False)  # pyright: ignore[reportAttributeAccessIssue]
        self.screen_under_test = screen
        self.result: bool | None = None

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic("host")

    def on_mount(self) -> None:
        self.push_screen(self.screen_under_test, callback=self._remember)

    def _remember(self, result: bool | None) -> None:
        self.result = result


def _screen(
    path: Path | str,
    *,
    raw: bytes | None = None,
    episode_id: str | None = _EPISODE_ID,
    lifecycle_state: str | None = "draft",
    summary: str = _SUMMARY,
    snapshot_consent: bool = False,
) -> PlanConsentScreen:
    return PlanConsentScreen(
        plan_path=str(path),
        summary=summary,
        executor_model="mistral-large-4",
        executor_thinking="medium",
        executor_identity="orchestrator",
        executor_display_name="Mistral Large",
        episode_id=episode_id,
        lifecycle_state=lifecycle_state,
        document_bytes=raw,
        snapshot_consent=snapshot_consent,
    )


def _text(screen: PlanConsentScreen, selector: str) -> str:
    return str(screen.query_one(selector, NoMarkupStatic).content)


def _draft(tmp_path: Path) -> Path:
    path = tmp_path / "plan.md"
    path.write_bytes(_RAW)
    return path


class TestConsentFieldSet:
    @pytest.mark.asyncio
    async def test_the_modal_renders_the_full_design_field_set(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert _text(screen, "#plan-consent-episode") == _EPISODE_ID
            assert _text(screen, "#plan-consent-state") == "draft"
            assert _text(screen, "#plan-consent-path") == str(path)
            assert _text(screen, "#plan-consent-content") == decode_safe(_RAW).text
            assert _text(screen, "#plan-consent-summary") == _SUMMARY
            assert _text(screen, "#plan-consent-bytes") == f"{len(_RAW)} bytes"
            assert _text(screen, "#plan-consent-digest") == (
                hashlib.sha256(_RAW).hexdigest()
            )
            executor = _text(screen, "#plan-consent-executor")
            assert "Mistral Large" in executor
            assert "(mistral-large-4)" in executor
            assert "thinking medium" in executor
            assert "identity orchestrator" in executor

    @pytest.mark.asyncio
    async def test_the_digest_is_over_the_raw_bytes_not_the_decoded_text(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            decoded = decode_safe(_RAW).text
            assert "\r" not in decoded
            # The display is the decode; the digest is not.
            assert _text(screen, "#plan-consent-content") == decoded
            assert _text(screen, "#plan-consent-digest") == (
                hashlib.sha256(_RAW).hexdigest()
            )
            assert _text(screen, "#plan-consent-digest") != (
                hashlib.sha256(decoded.encode()).hexdigest()
            )
            assert _text(screen, "#plan-consent-bytes") == f"{len(_RAW)} bytes"

    @pytest.mark.asyncio
    async def test_the_whole_document_is_shown_in_the_scrolling_body(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            body = screen.query_one("#plan-consent-body", VerticalScroll)
            content = screen.query_one("#plan-consent-content", NoMarkupStatic)
            assert content.parent is body
            assert body.max_scroll_y > 0
            rendered = str(content.content)
            assert rendered.startswith("# Plan")
            assert "- step 1" in rendered
            assert "- step 60" in rendered

    @pytest.mark.asyncio
    async def test_cancel_is_focused_by_default_and_escape_dismisses(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app.focused is screen.query_one(f"#{CANCEL_BUTTON_ID}", Button)
            await pilot.press("escape")
            await pilot.pause()
            assert app.result is False

    @pytest.mark.asyncio
    async def test_the_summary_is_extracted_from_the_previewed_bytes(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW, summary="a summary from somewhere else")
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert _text(screen, "#plan-consent-summary") == _SUMMARY

    @pytest.mark.asyncio
    async def test_missing_identity_inputs_render_as_unresolved(
        self, tmp_path: Path
    ) -> None:
        screen = _screen(
            tmp_path / "missing.md", raw=None, episode_id=None, lifecycle_state=None
        )
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert _text(screen, "#plan-consent-episode") == "unresolved"
            assert _text(screen, "#plan-consent-state") == "unresolved"
            assert _text(screen, "#plan-consent-content") == "unresolved"
            assert _text(screen, "#plan-consent-bytes") == "unresolved"
            assert _text(screen, "#plan-consent-digest") == "unresolved"
            assert _text(screen, "#plan-consent-summary") == _SUMMARY


class TestConsentRePreviewOnChange:
    @pytest.mark.asyncio
    async def test_an_unchanged_document_confirms_and_dismisses(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert app.result is True

    @pytest.mark.asyncio
    async def test_a_changed_document_re_previews_instead_of_confirming(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        changed = _plan_text(steps=3)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(changed)
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert app.result is None
            assert _text(screen, "#plan-consent-content") == decode_safe(changed).text
            assert _text(screen, "#plan-consent-bytes") == f"{len(changed)} bytes"
            assert _text(screen, "#plan-consent-digest") == (
                hashlib.sha256(changed).hexdigest()
            )
            assert "changed since the preview" in _text(screen, "#plan-consent-notice")

    @pytest.mark.asyncio
    async def test_consent_applies_to_the_bytes_last_shown(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        first = _plan_text(steps=5)
        second = _plan_text(steps=2)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(first)
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert app.result is None
            assert _text(screen, "#plan-consent-digest") == (
                hashlib.sha256(first).hexdigest()
            )
            path.write_bytes(second)
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert app.result is None
            assert _text(screen, "#plan-consent-digest") == (
                hashlib.sha256(second).hexdigest()
            )
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert app.result is True

    @pytest.mark.asyncio
    async def test_the_re_preview_returns_focus_to_cancel(self, tmp_path: Path) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(_plan_text(steps=3))
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert app.focused is screen.query_one(f"#{CANCEL_BUTTON_ID}", Button)

    @pytest.mark.asyncio
    async def test_the_extracted_summary_follows_a_changed_document(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(_plan_text(steps=3, summary="A different summary."))
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert _text(screen, "#plan-consent-summary") == "A different summary."

    @pytest.mark.asyncio
    async def test_an_unreadable_document_refuses_at_confirm(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.unlink()
            await pilot.click(f"#{START_BUTTON_ID}")
            # Longer than the Button's active-effect window, whose "-active"
            # state ignores a press that lands inside it.
            await pilot.pause(0.3)
            assert app.result is None
            assert "Cannot verify the plan document" in _text(
                screen, "#plan-consent-notice"
            )


class TestConsentRefusalMatrix:
    """design §4.2 (:234): empty and summary-less bytes are never consentable."""

    @pytest.mark.asyncio
    async def test_an_empty_draft_refuses_at_confirm(self, tmp_path: Path) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(b"")
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            # The changed bytes are re-previewed before anything is refused.
            assert app.result is None
            assert "changed since the preview" in _text(screen, "#plan-consent-notice")
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            notice = _text(screen, "#plan-consent-notice")
            assert "is empty" in notice
            # The refusal names its recovery.
            assert "/plan rebind <path>" in notice
            assert screen.confirmed is None

    @pytest.mark.asyncio
    async def test_a_summary_less_draft_refuses_at_confirm(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        summary_less = b"# Plan\n\n## Steps\n\n- only a step\n"
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(summary_less)
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            notice = _text(screen, "#plan-consent-notice")
            assert "no readable ## Summary section" in notice
            assert "/plan rebind <path>" in notice
            assert screen.confirmed is None

    @pytest.mark.asyncio
    async def test_an_empty_summary_section_refuses_at_confirm(
        self, tmp_path: Path
    ) -> None:
        # ``extract_plan_summary`` returns "" for a PRESENT but empty section,
        # and the server refuses that exactly like a missing one; the modal must
        # not confirm bytes the server would refuse.
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        empty_summary = b"# Plan\r\n\r\n## Summary\r\n\r\n## Steps\r\n\r\n- a step\r\n"
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(empty_summary)
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            notice = _text(screen, "#plan-consent-notice")
            assert "no readable ## Summary section" in notice
            assert "/plan rebind <path>" in notice
            assert screen.confirmed is None

    @pytest.mark.asyncio
    async def test_a_recovered_valid_draft_confirms_after_the_re_preview(
        self, tmp_path: Path
    ) -> None:
        path = _draft(tmp_path)
        screen = _screen(path, raw=_RAW)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            path.write_bytes(b"")
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            repaired = _plan_text(steps=4)
            path.write_bytes(repaired)
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is True
            assert screen.confirmed is not None
            assert screen.confirmed.document_bytes == repaired


class TestSnapshotConsent:
    """The recovery re-offer's bytes ARE the verified snapshot (design §4.2)."""

    @pytest.mark.asyncio
    async def test_snapshot_consent_confirms_without_a_readable_draft(
        self, tmp_path: Path
    ) -> None:
        # The draft is gone entirely: the re-offer leg must not consult it.
        screen = _screen(tmp_path / "gone.md", raw=_RAW, snapshot_consent=True)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is True
            assert screen.confirmed is not None
            assert screen.confirmed.document_bytes == _RAW
            assert screen.confirmed.sha256 == hashlib.sha256(_RAW).hexdigest()

    @pytest.mark.asyncio
    async def test_snapshot_consent_still_refuses_empty_bytes(
        self, tmp_path: Path
    ) -> None:
        screen = _screen(tmp_path / "gone.md", raw=b"", snapshot_consent=True)
        app = _ModalApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.click(f"#{START_BUTTON_ID}")
            await pilot.pause(0.3)
            assert app.result is None
            assert screen.confirmed is None
            assert "is empty" in _text(screen, "#plan-consent-notice")
