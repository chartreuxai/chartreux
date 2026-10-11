"""Shell Permissions panel: inspect and revoke dangerous-command authority.

The panel is the Settings entry for the ``shell_permissions`` deferred
descriptor. It distinguishes live session grants, saved project grants (with
the disk revision), and blocking explicit denylist entries, and revokes one
operation at a time with the expected live generation and disk revision from
the last inspection. Live and saved outcomes are reported separately; a
conflict or uncertain outcome marks the list stale and asks for a refresh
instead of claiming an unconditional revocation. The panel writes no config
and makes no authority decisions — it drives the P5 endpoints through the
protocol-only ``ShellPermissionsService``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shell_permissions import (
    ShellPermissionsGrant,
    ShellPermissionsInspection,
    ShellPermissionsRevocation,
    ShellPermissionsService,
)
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

MIN_MODAL_WIDTH = 84
MIN_MODAL_HEIGHT = 28
STALE_SUFFIX = " The list is stale; press R to refresh."
DOUBLE_CLICK = 2

_LIVE_STATUS_TEXT = {
    "removed": "live session authority removed",
    "conflict": "live authority changed since inspection; nothing was revoked",
    "skipped": "live removal skipped",
    "failed": "live removal failed",
}
_SAVED_STATUS_TEXT = {
    "removed": "saved project grant removed from your user config",
    "conflict": "saved grant changed on disk; the saved side was not touched",
    "skipped": "no saved removal attempted",
    "not_applicable": "never saved (consequential operations are not persisted)",
    "failed": "saved removal failed",
}


@dataclass(frozen=True)
class ShellPermissionsResult:
    """Dismissal payload: the last revocation outcome, honestly worded."""

    feedback: str


class ShellPermissionsOptionList(NavigableOptionList):
    """Bounded section rows; movement never wraps or transfers focus."""

    def focus_on_click(self) -> bool:
        return False

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()

    def _move_bounded(self, direction: int, distance: int = 1) -> None:
        valid = [i for i, option in enumerate(self.options) if not option.disabled]
        if not valid:
            self.highlighted = None
            return
        current = self.highlighted
        if current not in valid:
            self.highlighted = valid[0 if direction > 0 else -1]
            return
        ordinal = valid.index(current)
        self.highlighted = valid[
            max(0, min(len(valid) - 1, ordinal + direction * distance))
        ]

    def action_cursor_down(self) -> None:
        self._move_bounded(1)

    def action_cursor_up(self) -> None:
        self._move_bounded(-1)

    def action_page_down(self) -> None:
        self._move_bounded(1, max(1, self.scrollable_content_region.height))

    def action_page_up(self) -> None:
        self._move_bounded(-1, max(1, self.scrollable_content_region.height))

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        screen = self.screen
        if not isinstance(screen, ShellPermissionsApp):
            return
        if not screen._begin_pointer_event(event) or screen._busy:
            return
        index = event.style.meta.get("option")
        if index is None or self.get_option_at_index(index).disabled:
            return
        self.highlighted = index
        self.focus()
        if self.id == "shell-permissions-actions":
            if event.chain == 1:
                screen._pointer_command_started = True
                self.action_select()
        elif event.chain == DOUBLE_CLICK:
            screen._pointer_command_started = True
            self.action_select()


class ShellPermissionsActions(ShellPermissionsOptionList):
    """Fixed Enter-only commands, also available by single click."""


class ShellPermissionsApp(ModalScreen[ShellPermissionsResult]):
    """Inspect and revoke this session's dangerous-command authority."""

    CSS_PATH = "shell_permissions_app.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("tab", "next_group", show=False, priority=True),
        Binding("shift+tab", "previous_group", show=False, priority=True),
        Binding("escape", "close", "Back", show=False, priority=True),
        Binding("r", "refresh", "Refresh", show=False),
    ]

    def __init__(self, service: ShellPermissionsService) -> None:
        super().__init__(id="shell-permissions-screen")
        self.service = service
        self._inspection: ShellPermissionsInspection | None = None
        self._load_error: str | None = None
        self._busy = False
        self._busy_label = "Inspecting authority"
        self._stale = False
        self._outcome_feedback = ""
        self._feedback_class = ""
        self._pointer_command_started = False

    def compose(self) -> ComposeResult:
        with Vertical(id="shell-permissions-content"):
            yield NoMarkupStatic("Shell Permissions", id="shell-permissions-title")
            pending = NoMarkupStatic(
                "Action pending · finish editing, then Esc to answer",
                id="shell-permissions-pending-action",
            )
            pending.display = False
            yield pending
            yield NoMarkupStatic("", id="shell-permissions-project")
            yield ShellPermissionsOptionList(id="shell-permissions-grants")
            yield ShellPermissionsActions(
                Option("Refresh", id="refresh"),
                Option("Back", id="back"),
                id="shell-permissions-actions",
            )
            yield NoMarkupStatic("", id="shell-permissions-feedback")
            yield NoMarkupStatic("", id="shell-permissions-hint")

    def on_mount(self) -> None:
        self.query_one("#shell-permissions-pending-action").display = bool(
            getattr(self.app, "_pending_callbacks", None)
            or getattr(self.app, "_pending_local_question", None)
        )
        self._resize_surface()
        self.query_one("#shell-permissions-grants", ShellPermissionsOptionList).focus()
        self._render_rows()
        self.run_worker(
            self._inspect(),
            group="shell-permissions",
            exclusive=True,
            name="shell-permissions-inspect",
        )

    def on_resize(self) -> None:
        self._resize_surface()

    def _resize_surface(self) -> None:
        content = self.query_one("#shell-permissions-content", Vertical)
        width, height = self.size
        fullscreen = width < MIN_MODAL_WIDTH or height < MIN_MODAL_HEIGHT
        content.set_class(fullscreen, "fullscreen")
        content.styles.width = "100%" if fullscreen else min(92, width - 2)
        content.styles.height = "100%" if fullscreen else max(1, height - 2)
        content.border_title = "" if fullscreen else "Shell Permissions"

    def _begin_pointer_event(self, event: events.Click) -> bool:
        if event.chain == 1:
            self._pointer_command_started = False
        return not self._pointer_command_started

    # -- focus groups ----------------------------------------------------

    def _visible_focus_groups(self) -> list[Widget]:
        return [
            widget
            for widget in (
                self.query_one("#shell-permissions-grants", OptionList),
                self.query_one("#shell-permissions-actions", OptionList),
            )
            if widget.display and widget.can_focus
        ]

    def _cycle_group(self, direction: int) -> None:
        groups = self._visible_focus_groups()
        if not groups:
            return
        index = (
            groups.index(self.focused)
            if self.focused in groups
            else (-1 if direction > 0 else 0)
        )
        groups[(index + direction) % len(groups)].focus()
        self._update_hint()

    def action_next_group(self) -> None:
        self._cycle_group(1)

    def action_previous_group(self) -> None:
        self._cycle_group(-1)

    # -- rendering -------------------------------------------------------

    def _heading(self, title: str, key: str) -> Option:
        return Option(
            Content.assemble((title, "$foreground bold")),
            id=f"head:{key}",
            disabled=True,
        )

    def _empty(self, key: str) -> Option:
        return Option(
            Content.assemble(("None", "$text-muted")), id=f"state:{key}", disabled=True
        )

    def _grant_rows(
        self, grants: tuple[ShellPermissionsGrant, ...], *, source: str, row_prefix: str
    ) -> list[Option]:
        by_operation: dict[str, list[str]] = {}
        for grant in grants:
            by_operation.setdefault(grant.operation_id, []).append(grant.repository)
        rows: list[Option] = []
        for operation_id, repositories in by_operation.items():
            scope = repositories[0] + (
                f" (+{len(repositories) - 1} more)" if len(repositories) > 1 else ""
            )
            rows.append(
                Option(
                    Content.assemble(
                        (operation_id, ""),
                        (f" — {source}", "$text-muted"),
                        (f" · {scope}", "$text-muted"),
                    ),
                    id=f"{row_prefix}:{operation_id}",
                )
            )
        return rows

    def _denylist_rows(self, inspection: ShellPermissionsInspection) -> list[Option]:
        rows: list[Option] = []
        for block in inspection.blocking_denylist:
            operations = ", ".join(block.operations) or "dangerous git operations"
            rows.append(
                Option(
                    Content.assemble(
                        (block.pattern, "$error"),
                        (f" — blocks {operations}", "$text-muted"),
                        (" (absolute veto; no grant overrides this)", "$text-muted"),
                    ),
                    id=f"deny:{block.pattern}",
                )
            )
        return rows

    def _saved_rows(self, inspection: ShellPermissionsInspection) -> list[Option]:
        rows = self._grant_rows(
            inspection.saved_grants, source="saved for this project", row_prefix="saved"
        )
        rows.extend(
            Option(
                Content.assemble(
                    (operation_id, ""),
                    (" — on disk only; not accepted by this session", "$text-muted"),
                ),
                id=f"disk:{operation_id}",
            )
            for operation_id in inspection.disk_only_operations
        )
        if inspection.saved_operations_unavailable:
            rows.append(
                Option(
                    Content.assemble((
                        "Saved state unavailable on disk",
                        "$text-muted",
                    )),
                    id="state:saved-unavailable",
                    disabled=True,
                )
            )
        return rows

    def _render_rows(self) -> None:
        grants = self.query_one("#shell-permissions-grants", ShellPermissionsOptionList)
        highlighted = (
            str(grants.highlighted_option.id)
            if grants.highlighted_option is not None and grants.highlighted_option.id
            else None
        )
        rows: list[Option] = []
        inspection = self._inspection
        if inspection is not None:
            rows.append(self._heading("Live session grants", "live"))
            rows.extend(
                self._grant_rows(
                    inspection.live_grants, source="live session", row_prefix="live"
                )
                or [self._empty("live")]
            )
            rows.append(self._heading("Saved project grants", "saved"))
            rows.extend(self._saved_rows(inspection) or [self._empty("saved")])
            rows.append(self._heading("Blocked by explicit denylist", "denylist"))
            rows.extend(self._denylist_rows(inspection) or [self._empty("denylist")])
        elif self._load_error is not None:
            rows.append(
                Option(
                    Content.assemble((
                        "Authority unavailable — press R to retry",
                        "$text-muted",
                    )),
                    id="state:unavailable",
                    disabled=True,
                )
            )
        else:
            rows.append(
                Option(
                    Content.assemble((
                        "Inspecting dangerous-command authority…",
                        "$text-muted",
                    )),
                    id="state:inspecting",
                    disabled=True,
                )
            )
        grants.clear_options()
        grants.add_options(rows)
        identifiers = [str(row.id) for row in grants.options if row.id is not None]
        if highlighted in identifiers:
            grants.highlighted = identifiers.index(highlighted)
        else:
            grants.highlighted = next(
                (i for i, row in enumerate(grants.options) if not row.disabled), None
            )
        if grants.highlighted is not None:
            grants.scroll_to_highlight()
        self._update_project()
        self._update_hint()

    def _update_project(self) -> None:
        widget = self.query_one("#shell-permissions-project", NoMarkupStatic)
        inspection = self._inspection
        if inspection is None:
            widget.update(
                "Project: unknown · authority inspection unavailable"
                if self._load_error is not None
                else "Project: unknown · inspecting…"
            )
            return
        text = f"Project: {inspection.project}"
        if inspection.disk_revision is not None:
            text += f" · saved revision {inspection.disk_revision}"
        if self._stale:
            text += " · list stale (press R to refresh)"
        widget.update(text)

    def _render_feedback(self) -> None:
        widget = self.query_one("#shell-permissions-feedback", NoMarkupStatic)
        rendered = self._outcome_feedback + (STALE_SUFFIX if self._stale else "")
        widget.update(rendered)
        widget.set_class(self._feedback_class == "warning", "warning")
        widget.set_class(self._feedback_class == "error", "error")
        widget.set_class(self._feedback_class == "success", "success")

    def _update_hint(self) -> None:
        widget = self.query_one("#shell-permissions-hint", NoMarkupStatic)
        if self._busy:
            widget.update(shortcut_hint(f"Running: {self._busy_label}"))
            return
        actions = self.query_one("#shell-permissions-actions", OptionList)
        if actions.has_focus:
            action = actions.highlighted_option
            enter_label = str(action.prompt) if action is not None else "Select"
        else:
            enter_label = "Select"
            grants = self.query_one("#shell-permissions-grants", OptionList)
            option = grants.highlighted_option
            option_id = str(option.id) if option is not None and option.id else ""
            if option_id.startswith("deny:"):
                enter_label = "Explain veto"
            elif option_id.startswith(("live:", "saved:", "disk:")):
                enter_label = "Revoke"
        widget.update(
            shortcut_hint(
                f"{shortcut('↑↓/jk')} navigate  {shortcut('Enter')} {enter_label}  "
                f"{shortcut('R')} refresh  {shortcut('Tab')} groups  "
                f"{shortcut('Esc')} back"
            )
        )

    # -- data flow -------------------------------------------------------

    async def _inspect(self) -> None:
        self._busy = True
        self._busy_label = "Inspecting authority"
        self._update_hint()
        try:
            inspection = await self.service.inspect()
        except Exception as exc:
            self._inspection = None
            self._load_error = str(exc)
            self._outcome_feedback = (
                f"Failed: Could not inspect dangerous-command authority: {exc}"
            )
            self._feedback_class = "error"
            self._render_rows()
        else:
            self._inspection = inspection
            self._load_error = None
            self._stale = False
            if self._outcome_feedback.startswith("Failed: Could not inspect"):
                self._outcome_feedback = ""
                self._feedback_class = ""
            self._render_rows()
        finally:
            self._busy = False
            self._render_feedback()
            self._update_hint()

    def action_refresh(self) -> None:
        if self._busy:
            return
        self.run_worker(
            self._inspect(),
            group="shell-permissions",
            exclusive=True,
            name="shell-permissions-inspect",
        )

    def _explain_veto(self, pattern: str) -> None:
        self._outcome_feedback = (
            f"! Warning: '{pattern}' is an explicit denylist entry — an absolute "
            "veto. No live or saved grant overrides it; remove the entry from "
            "your config's shell denylist to allow the operation."
        )
        self._feedback_class = "warning"
        self._render_feedback()
        self._update_hint()

    def _revoke_operation(self, operation_id: str) -> None:
        inspection = self._inspection
        if self._busy or inspection is None:
            return
        self._busy = True
        self._busy_label = f"Revoking {operation_id}"
        self._update_hint()
        self.run_worker(
            self._revoke(
                operation_id,
                inspection.live_authority_generation,
                inspection.disk_revision,
            ),
            group="shell-permissions",
            exclusive=True,
            name="shell-permissions-revoke",
        )

    async def _revoke(
        self,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
    ) -> None:
        try:
            outcome = await self.service.revoke(
                operation_id=operation_id,
                expected_live_generation=expected_live_generation,
                expected_disk_revision=expected_disk_revision,
            )
        except Exception as exc:
            self._outcome_feedback = f"Failed: Revoking {operation_id} failed: {exc}"
            self._feedback_class = "error"
            self._busy = False
            self._render_feedback()
            self._update_hint()
            return
        self._stale = outcome.conflicted
        self._report(outcome)
        if outcome.conflicted:
            # A stale expectation rejected the removal: keep the pre-revocation
            # list, mark it stale, and ask for a refresh instead of claiming
            # the authority is gone.
            self._render_rows()
            self._busy = False
            self._update_hint()
        else:
            # Reconcile the list with the post-revocation authority.
            await self._inspect()

    def _report(self, outcome: ShellPermissionsRevocation) -> None:
        operation_id = outcome.operation_id
        live_text = _LIVE_STATUS_TEXT[outcome.live_status]
        if outcome.live_status == "conflict" and outcome.live_generation is not None:
            live_text += f" (now generation {outcome.live_generation})"
        saved_text = _SAVED_STATUS_TEXT[outcome.saved_status]
        uncertain = (
            outcome.conflicted
            or "failed" in {outcome.live_status, outcome.saved_status}
            or outcome.warning is not None
        )
        message = f"Revoking {operation_id}: {live_text}; {saved_text}."
        if outcome.warning is not None:
            message += f" {outcome.warning}"
        if uncertain:
            self._outcome_feedback = f"! Warning: {message}"
            self._feedback_class = "warning"
        else:
            self._outcome_feedback = f"{chrome_glyph('success')} {message}"
            self._feedback_class = "success"
        self._render_feedback()
        self._update_hint()

    # -- selection and dismissal -----------------------------------------

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        self._update_hint()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        self._update_hint()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "shell-permissions-actions":
            if event.option.id == "refresh":
                self.action_refresh()
            elif event.option.id == "back":
                self.action_close()
            return
        if self._busy or event.option.disabled or event.option.id is None:
            return
        option_id = str(event.option.id)
        if option_id.startswith("deny:"):
            self._explain_veto(option_id[len("deny:") :])
            return
        for prefix in ("live:", "saved:", "disk:"):
            if option_id.startswith(prefix):
                self._revoke_operation(option_id[len(prefix) :])
                return

    def action_close(self) -> None:
        if self._busy:
            self._update_hint()
            return
        self.dismiss(ShellPermissionsResult(self._outcome_feedback))
