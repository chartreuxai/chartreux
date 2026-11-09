"""The ``/plan accept`` consent modal: the episode, its bytes, and the executor.

One explicit decision before the handoff replaces the conversation: start
execution with the snapshotted executor selection, or cancel and stay in plan
mode. The modal previews the whole decision — the episode's identity and
lifecycle state, the selected document's path and full content, its extracted
``## Summary``, the exact byte count and SHA-256 consent digest, and the
executor selection. Cancel is the default action.

Consent applies to the bytes last shown: a draft that changed between preview
and confirm is re-previewed in place, focus returns to Cancel, and starting
needs a fresh confirm. The digest is always taken over the RAW file bytes,
while the displayed content is ``decode_safe``'s decoding of those same bytes,
so a lossy or newline-normalizing decode can never change what is consented
to. Nothing here resolves or re-reads the executor selection — the app
snapshots it at modal open and the handoff applies exactly what this screen
displays. The theme row appears only when the mode has a theme, and never
rides the handoff.

Two consent shapes exist, told apart by ``snapshot_consent``:

* an ordinary accept previews the bound DRAFT, so confirm re-reads the draft
  and compares digests (P2.5);
* the recovery re-offer previews the episode's VERIFIED surviving snapshot.
  Post-consent the draft is never consulted again (design §4.2/§4.4), so that
  shape confirms the previewed bytes alone and never re-reads the draft.

Both shapes refuse empty and summary-less bytes (design §4.2's refusal matrix,
:234): consenting to a document with nothing to execute is never allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button

from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.utils.io import decode_safe

CANCEL_BUTTON_ID = "plan-consent-cancel"
START_BUTTON_ID = "plan-consent-start"

# The design's consent field set is rendered in full; a value the modal was not
# given is stated as unresolved rather than silently dropping its row.
UNRESOLVED = "unresolved"
_SUMMARY_MISSING = "(the document has no readable ## Summary section)"

_CHANGED_NOTICE = (
    "The plan document changed since the preview; consent applies to the bytes "
    "last shown. Review the content and the digest, then start again — or cancel."
)
_UNREADABLE_NOTICE = (
    "Cannot verify the plan document: it cannot be read now. Nothing was "
    "started. Cancel and reopen the plan."
)
# The refusal matrix (design §4.2, :234): empty and summary-less bytes are not
# consentable. Every refusal names the recovery the matrix names.
_EMPTY_NOTICE = (
    "The plan document is empty, so there is nothing to execute. Nothing was "
    "started. Cancel, write the plan (or re-bind another document with "
    "/plan rebind <path>), then start again."
)
_SUMMARYLESS_NOTICE = (
    "The plan document has no readable ## Summary section, so there is nothing "
    "to execute. Nothing was started. Cancel, add a ## Summary section (or "
    "re-bind another document with /plan rebind <path>), then start again."
)


@dataclass(frozen=True, slots=True)
class ConfirmedConsent:
    """The bytes the modal last showed and the user confirmed.

    The handoff carries exactly these bytes: a caller must never re-read the
    draft after confirmation, because a change between the modal's check and
    the request would consent to bytes the user never saw.
    """

    document_bytes: bytes
    sha256: str
    summary: str


class PlanConsentScreen(ModalScreen[bool]):
    """Confirm start of the executor session, or cancel and keep planning."""

    CSS_PATH = "plan_consent.tcss"
    # Default-No: the affirmative action replaces the conversation, so the
    # first focus is Cancel and Escape always cancels.
    AUTO_FOCUS = f"#{CANCEL_BUTTON_ID}"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True)
    ]

    def __init__(  # noqa: PLR0913 - the design's field set is the interface
        self,
        *,
        plan_path: str,
        summary: str,
        executor_model: str,
        executor_thinking: str,
        executor_identity: str,
        executor_display_name: str,
        theme: str | None = None,
        episode_id: str | None = None,
        lifecycle_state: str | None = None,
        document_bytes: bytes | None = None,
        snapshot_consent: bool = False,
    ) -> None:
        super().__init__(id="plan-consent-screen")
        self._plan_path = plan_path
        self._executor_model = executor_model
        self._executor_thinking = executor_thinking
        self._executor_identity = executor_identity
        self._executor_display_name = executor_display_name
        self._theme = theme
        self._episode_id = episode_id
        self._lifecycle_state = lifecycle_state
        # The recovery re-offer's preview is the episode's VERIFIED surviving
        # snapshot, so the draft has no say in the confirmation at all.
        self._snapshot_consent = snapshot_consent
        # The previewed bytes ARE the consent surface: the caller's retained
        # snapshot when it supplies one, otherwise the draft as it stands at
        # modal open. No bytes means there is no preview to verify (the
        # pre-integrity call shape), never a substitute re-read of the inputs.
        self._preview_bytes = (
            document_bytes if document_bytes is not None else _read_raw_bytes(plan_path)
        )
        self._summary = (
            summary
            if self._preview_bytes is None
            else (_extract_summary(self._preview_bytes) or _SUMMARY_MISSING)
        )
        self._notice = ""
        self._confirmed: ConfirmedConsent | None = None

    @property
    def confirmed(self) -> ConfirmedConsent | None:
        """The confirmed bytes; set only when a start actually dismissed."""
        return self._confirmed

    def compose(self) -> ComposeResult:
        with Vertical(id="plan-consent-dialog"):
            yield NoMarkupStatic("Start plan execution", id="plan-consent-title")
            with VerticalScroll(id="plan-consent-body"):
                yield NoMarkupStatic(self._notice, id="plan-consent-notice")
                yield NoMarkupStatic("Episode", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._episode_id or UNRESOLVED,
                    classes="plan-consent-value",
                    id="plan-consent-episode",
                )
                yield NoMarkupStatic("Lifecycle", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._lifecycle_state or UNRESOLVED,
                    classes="plan-consent-value",
                    id="plan-consent-state",
                )
                yield NoMarkupStatic("Plan document", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._plan_path,
                    classes="plan-consent-value",
                    id="plan-consent-path",
                )
                if self._theme:
                    yield NoMarkupStatic("Theme", classes="plan-consent-label")
                    yield NoMarkupStatic(
                        self._theme,
                        classes="plan-consent-value",
                        id="plan-consent-theme",
                    )
                yield NoMarkupStatic("Content", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._content_text(),
                    classes="plan-consent-value",
                    id="plan-consent-content",
                )
                yield NoMarkupStatic("Summary", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._summary,
                    classes="plan-consent-value",
                    id="plan-consent-summary",
                )
                yield NoMarkupStatic("Size", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._size_text(),
                    classes="plan-consent-value",
                    id="plan-consent-bytes",
                )
                yield NoMarkupStatic("SHA-256", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._digest_text(),
                    classes="plan-consent-value",
                    id="plan-consent-digest",
                )
                yield NoMarkupStatic("Executor", classes="plan-consent-label")
                yield NoMarkupStatic(
                    self._executor_line(),
                    classes="plan-consent-value",
                    id="plan-consent-executor",
                )
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('Tab')} Focus  {shortcut('Enter')} Activate  "
                    f"{shortcut('Esc')} Cancel"
                ),
                id="plan-consent-hint",
            )
            with Horizontal(id="plan-consent-actions"):
                yield Button("Cancel", id=CANCEL_BUTTON_ID)
                yield Button("Start execution", id=START_BUTTON_ID)

    def _content_text(self) -> str:
        if self._preview_bytes is None:
            return UNRESOLVED
        return decode_safe(self._preview_bytes).text

    def _size_text(self) -> str:
        if self._preview_bytes is None:
            return UNRESOLVED
        return f"{len(self._preview_bytes)} bytes"

    def _digest_text(self) -> str:
        if self._preview_bytes is None:
            return UNRESOLVED
        return _digest(self._preview_bytes)

    def _executor_line(self) -> str:
        model = (
            self._executor_model
            if self._executor_display_name == self._executor_model
            else f"{self._executor_display_name} ({self._executor_model})"
        )
        separator = chrome_glyph("metadata_separator")
        return (
            f"{model}  {separator}  thinking {self._executor_thinking}  "
            f"{separator}  identity {self._executor_identity}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == START_BUTTON_ID:
            self.action_start()
        else:
            self.action_cancel()

    def action_start(self) -> None:
        """Confirm, but only for the consentable bytes last shown (design §4.4)."""
        if self._snapshot_consent:
            # The recovery leg carries the episode's verified snapshot; the
            # draft is never consulted post-consent (design §4.2/§4.4).
            self._refuse_or_confirm()
            return
        current = _read_raw_bytes(self._plan_path)
        if self._preview_bytes is None:
            # No preview ever existed, so there is nothing to compare against;
            # this is the pre-integrity call shape. A document that became
            # readable since is shown and needs a fresh confirm.
            if current is None:
                self._confirm_and_dismiss()
                return
            self._re_preview(current)
            return
        if current is None:
            self._set_notice(_UNREADABLE_NOTICE)
            return
        if _digest(current) == _digest(self._preview_bytes):
            self._refuse_or_confirm()
            return
        self._re_preview(current)

    def _refuse_or_confirm(self) -> None:
        """Apply the refusal matrix to the shown bytes, then confirm them."""
        notice = _consent_refusal(self._preview_bytes)
        if notice is not None:
            self._set_notice(notice)
            return
        self._confirm_and_dismiss()

    def _confirm_and_dismiss(self) -> None:
        """Carry the bytes last shown out with the confirmation."""
        if self._preview_bytes is not None:
            self._confirmed = ConfirmedConsent(
                document_bytes=self._preview_bytes,
                sha256=_digest(self._preview_bytes),
                summary=self._summary,
            )
        self.dismiss(True)

    def _re_preview(self, raw: bytes) -> None:
        """Show the changed bytes and require a fresh confirm; never dismiss."""
        self._preview_bytes = raw
        self._summary = _extract_summary(raw) or _SUMMARY_MISSING
        self._set_notice(_CHANGED_NOTICE)
        self.query_one("#plan-consent-content", NoMarkupStatic).update(
            self._content_text()
        )
        self.query_one("#plan-consent-summary", NoMarkupStatic).update(self._summary)
        self.query_one("#plan-consent-bytes", NoMarkupStatic).update(self._size_text())
        self.query_one("#plan-consent-digest", NoMarkupStatic).update(
            self._digest_text()
        )
        # The notice rides the top of the scroll body: bring it into view so the
        # changed bytes are reviewed from the start.
        self.query_one("#plan-consent-body", VerticalScroll).scroll_home(animate=False)
        # Consent is a deliberate choice about newly shown bytes: the default
        # action goes back to Cancel so a held Enter cannot carry over.
        self.query_one(f"#{CANCEL_BUTTON_ID}", Button).focus()

    def _set_notice(self, notice: str) -> None:
        self._notice = notice
        self.query_one("#plan-consent-notice", NoMarkupStatic).update(notice)

    def action_cancel(self) -> None:
        self.dismiss(False)


def _read_raw_bytes(path: str) -> bytes | None:
    """Read the draft's raw bytes; None when it cannot be read."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def _digest(raw: bytes) -> str:
    """The consent digest: SHA-256 over the raw file bytes, never the decode."""
    return hashlib.sha256(raw).hexdigest()


def _extract_summary(raw: bytes) -> str | None:
    """Extract ``## Summary`` from the previewed bytes.

    Imported lazily: the textual-UI boundary forbids a static
    ``chartreux.core`` import, and the core extractor is the one authority for
    the section's meaning.
    """
    plan_mode = import_module("chartreux.core.plan_mode")
    return plan_mode.extract_plan_summary(decode_safe(raw).text)


def _consent_refusal(raw: bytes | None) -> str | None:
    """The refusal for shown bytes the design's matrix forbids consenting to.

    A bound draft that is missing/unreadable/empty/summary-less is refused
    (design §4.2, :234): the executor would be seeded with a document that has
    nothing to execute. ``None`` means the bytes are consentable.

    The summary test is falsiness, not ``is None``: ``extract_plan_summary``
    returns ``None`` for an absent heading and ``""`` for a heading with no
    text, and the server refuses BOTH — an empty section is as unexecutable as
    no section, so the modal must not confirm what the server would refuse.
    """
    if raw is None:
        return _UNREADABLE_NOTICE
    if not raw.strip():
        return _EMPTY_NOTICE
    if not _extract_summary(raw):
        return _SUMMARYLESS_NOTICE
    return None
