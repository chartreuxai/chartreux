"""Session/picker flow controller for the TUI app (W4b C3, stage 9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from textual.widget import Widget

from chartreux.app_server.models import (
    COMMITTED_MODEL_RECOVERY_ISSUE_FILE,
    PublicHistoryEntry,
    PublicSession,
    UserQuestionResult,
)
from chartreux.app_server.protocol import AppServerResponseError, ProtocolErrorCode
from chartreux.cli.textual_ui.bottom_panel import BottomApp
from chartreux.cli.textual_ui.widgets.branch_created_message import BranchCreatedMessage
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.load_more import HistoryLoadMoreRequested
from chartreux.cli.textual_ui.widgets.messages import (
    ErrorMessage,
    SlashCommandMessage,
    UserCommandMessage,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.rewind_fork_message import RewindForkMessage
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupExpansionState
from chartreux.cli.textual_ui.windowing import (
    HISTORY_RESUME_TAIL_MESSAGES,
    create_resume_plan,
)
from chartreux.cli.textual_ui.windowing.transcript import TranscriptAnchor
from chartreux.observability.logging import logger
from chartreux.ui.chrome_glyphs import chrome_glyph

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.app import ChartreuxApp
    from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp


@dataclass(slots=True)
class _PickerState:
    """Mutable state owned by the in-app session picker.

    ``previewing`` is True while a session's history is displayed in the
    transcript area instead of the live session's history. ``preview_session_id``
    is the session whose history is currently shown (None when not previewing).
    """

    previewing: bool = False
    preview_session_id: str | None = None

    def preview_is_current(self, session_id: str) -> bool:
        return self.preview_session_id == session_id

    def clear_preview(self) -> None:
        self.preview_session_id = None

    def exit_preview(self) -> bool:
        """Clear preview state; returns True if a preview was active."""
        was_previewing = self.previewing
        self.previewing = False
        self.preview_session_id = None
        return was_previewing


class SessionFlowController:
    """Owns the session-picker preview/origin state and the resume flow for one app.

    Plain object (no mixins): the app binds itself late via ``_bind_app``
    (never constructor-captured), and collaborators are invoked as
    ``self._app._method(...)`` at call time so instance patches keep
    intercepting. The app keeps the ``app._picker`` alias (a get-only
    forwarding property) plus forwarding properties for the origin fields, so
    direct-attribute consumers (tests, the transcript presenter) see no change.
    """

    # Late-bound host reference, assigned via _bind_app after construction.
    _app: ChartreuxApp

    def __init__(self) -> None:
        self.picker = _PickerState()
        self.picker_origin_anchor: TranscriptAnchor | None = None
        self.picker_origin_restoring = False
        self.picker_origin_range: tuple[int, int] | None = None
        self.picker_origin_focus: Widget | None = None
        self.resume_adopted = False
        self.rewind_mode = False
        self.rewind_highlighted_widget: UserMessage | None = None

    def _bind_app(self, app: ChartreuxApp) -> None:
        """Bind the host app after construction."""
        self._app = app

    async def on_session_picker_app_session_highlighted(
        self, event: SessionPickerApp.SessionHighlighted
    ) -> None:
        session_id = event.session_id
        if session_id is None:
            return
        # Current session is already displayed — don't reload unless we navigated
        # away first (in which case picker.previewing is True).
        if session_id == self._app.app_server.session_id and not self.picker.previewing:
            return
        # Record the intended preview target so a slower earlier request can't
        # overwrite the screen after the highlight moved on or resume started.
        self.picker.preview_session_id = session_id
        self.picker.previewing = True
        self._app._reset_ui_state()
        with self._app.batch_update():
            await self._app._load_more.hide()
            await self._app._messages_area.remove_children()
            await self._app._messages_area.mount(
                UserCommandMessage(f"Preview `{session_id[:8]}` loading (not active)…")
            )
        try:
            history = await self._app.app_server.resources.sessions.get_session_history(
                session_id
            )
        except Exception as exc:
            logger.exception("get_session_history failed for %s", session_id)
            if self.picker.preview_is_current(session_id):
                self.picker.previewing = True
                self._app._reset_ui_state()
                with self._app.batch_update():
                    await self._app._load_more.hide()
                    await self._app._messages_area.remove_children()
                    await self._app._messages_area.mount(
                        ErrorMessage(
                            f"Preview `{session_id[:8]}` failed to load: {exc}. "
                            "No history is shown; choose another session or cancel."
                        )
                    )
            return
        if not self.picker.preview_is_current(session_id):
            return
        await self.apply_picker_preview(session_id, history)

    async def apply_picker_preview(
        self, session_id: str, history: list[PublicHistoryEntry]
    ) -> None:
        self.picker.previewing = True
        self._app._preview_tool_group_expansion_state = ToolGroupExpansionState(
            default_collapsed=self._app._tools_collapsed
        )
        self._app._preview_entry_expansion_state = EntryExpansionState(
            default_collapsed=self._app._tools_collapsed
        )
        self._app._reset_ui_state()
        plan = create_resume_plan(history, HISTORY_RESUME_TAIL_MESSAGES)
        with self._app.batch_update():
            await self._app._load_more.hide()
            await self._app._messages_area.remove_children()
            if not self.picker.preview_is_current(session_id):
                return
            await self._app._messages_area.mount(
                UserCommandMessage(
                    f"Preview `{session_id[:8]}` (not active): "
                    + ("history loaded" if history else "empty history")
                )
            )
            if plan is not None:
                await self._app._mount_history_batch(
                    plan.tail_entries,
                    self._app._messages_area,
                    start_index=plan.tail_start_index,
                )
        if not self.picker.preview_is_current(session_id):
            return
        if plan is not None:
            self._app.call_after_refresh(self._app._chat_widget.anchor)
            self._app._windowing.set_backfill(plan.backfill_entries)
            await self._app._load_more.set_visible(
                self._app._messages_area,
                visible=self._app._has_older_history,
                remaining=self._app._history_backfill_remaining,
            )

    def exit_picker_preview(self) -> bool:
        was_previewing = self.picker.exit_preview()
        if was_previewing:
            self._app._tools_collapsed = (
                self._app._entry_expansion_state.default_collapsed
            )
        return was_previewing

    async def exit_picker_to_input(self) -> None:
        await self._app._switch_to_input_app()
        # Both exits (cancel, delete-last) return to the live input.
        self._app._mark_session_ready()
        if self._app._show_resume_picker:
            self._app._show_resume_picker = False
            self._app._startup_prompt_processed = True
        if self.exit_picker_preview():
            await self._app._rebuild_transcript_from_current_session()
            await self.restore_picker_history_range()
            return
        self._app.run_worker(
            self._app._show_custom_tools_deprecation_warning_after_initial_history(),
            exclusive=False,
        )

    async def on_session_picker_app_cancelled(
        self, event: SessionPickerApp.Cancelled
    ) -> None:
        self.picker_origin_restoring = True
        try:
            await self.exit_picker_to_input()
            await self._app._mount_and_scroll(UserCommandMessage("Resume cancelled."))
            await self.restore_picker_origin()
        finally:
            self._app.call_after_refresh(self.finish_picker_origin_restore)

    def finish_picker_origin_restore(self) -> None:
        self.picker_origin_restoring = False
        self._app._remember_transcript_anchor()

    async def restore_picker_origin(self) -> None:
        anchor = self.picker_origin_anchor
        self.picker_origin_anchor = None
        self.picker_origin_range = None
        if anchor is not None:
            self._app._transcript_following = anchor.following
            await self._app._transcript.restore_anchor(
                self._app._messages_area, anchor, check_revision=False
            )
        target = self.picker_origin_focus
        self.picker_origin_focus = None
        if target is not None and target.is_mounted and target.display:
            target.focus()

    async def restore_picker_history_range(self) -> None:
        await self._app._transcript_presenter.restore_picker_history_range()

    def build_picker(
        self, sessions: list[PublicSession], *, loading: bool = False
    ) -> SessionPickerApp:
        from chartreux.cli.textual_ui.app import _get_session_picker_app_class

        sessions = sorted(sessions, key=lambda s: s.updated_at, reverse=True)
        return _get_session_picker_app_class()(
            sessions=sessions,
            latest_messages={
                session.id: session.title or session.preview for session in sessions
            },
            current_session_id=self._app.app_server.session_id,
            cwd=self._app.app_server.cwd,
            loading=loading,
        )

    async def show_session_picker(
        self, command_message: SlashCommandMessage | None = None, **kwargs: Any
    ) -> None:
        # Mount the picker and erase the slash command message in one repaint so
        # the transition from command menu → picker is visually instant.
        picker = self.build_picker([], loading=True)
        self.picker_origin_anchor = self._app._transcript.capture_anchor(
            self._app._messages_area,
            following=self._app._transcript_following
            and self._app._chat_widget.scroll_y >= self._app._chat_widget.max_scroll_y,
        )
        self.picker_origin_range = (
            (
                self._app._transcript.admitted_start_index,
                self._app._transcript.admitted_end_index,
            )
            if self._app._transcript.unit_ids
            else None
        )
        self.picker_origin_focus = self._app.screen.focused
        with self._app.batch_update():
            if command_message is not None:
                await command_message.remove()
            await self._app._switch_from_input(picker)

        try:
            local_sessions = await self._app.app_server.resources.sessions.list(
                self._app.app_server.cwd
            )
        except Exception as e:
            logger.exception("Failed to discover saved sessions")
            picker.set_loading(False, error=str(e))
            return

        if not picker.is_mounted:
            # The picker was closed while sessions were loading.
            return
        picker.load_sessions(
            local_sessions
            if self._app.app_server.resources.runtime.session_log.enabled
            else [],
            {
                session.id: session.title or session.preview
                for session in local_sessions
            },
        )

    async def on_session_picker_app_session_selected(
        self, event: SessionPickerApp.SessionSelected
    ) -> None:
        was_previewing = self._app._exit_picker_preview()
        await self._app._switch_to_input_app()
        # Mark ready in the finally (covers both the failure return and success),
        # but only after _resume_local_session so a turn can't dispatch mid-rebind.
        try:
            await self._app._resume_local_session(event.session_id)
        except Exception as e:
            logger.exception("Failed to resume session %s", event.session_id)
            if self._app._show_resume_picker:
                self._app._show_resume_picker = False
                self._app._startup_prompt_processed = True
            if not self.resume_adopted and was_previewing:
                try:
                    await self._app._rebuild_transcript_from_current_session()
                except Exception:
                    logger.exception(
                        "Failed to rebuild transcript after resume failure"
                    )
            if self.resume_adopted:
                # Recovery is installed by _resume_local_session, not the old preview.
                return
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Session not changed; failed to load `{event.session_id[:8]}`: {e}",
                    collapsed=self._app._tools_collapsed,
                )
            )
            return
        else:
            if self._app._show_resume_picker:
                self._app._show_resume_picker = False
                await self._app._process_startup_prompt_when_available()
        finally:
            self._app._mark_session_ready()

    async def resume_local_session(self, session_id: str) -> None:
        self._app._exit_picker_preview()
        await self._app._request_agent_transcript_close(restore_focus=False)
        self.resume_adopted = False
        self._app._resume_ui_ready.clear()
        self._app._stop_usage_subscription()
        try:
            await self._app.app_server.resume(
                session_id, on_adopt=lambda: setattr(self, "resume_adopted", True)
            )
            logger.debug("Resume stage=rpc-returned session_id=%s", session_id)
            await self.reset_presentation_after_resume()
            logger.debug("Resume stage=presentation-reset session_id=%s", session_id)
            if self._app._chat_input_container:
                self._app._chat_input_container.set_custom_border(None)
            self._app._refresh_banner()
            self._app._refresh_context_progress()
            await self._app._rebuild_transcript_from_current_session()
            logger.debug("Resume stage=transcript-rebuilt session_id=%s", session_id)
            await self._app._queue.sync_server_queue(self._app.app_server.turn_queue)
            logger.debug("Resume stage=queue-synced session_id=%s", session_id)
            await self._app._mount_and_scroll(
                UserCommandMessage(f"Resumed session `{session_id[:8]}`")
            )
            await self.surface_committed_model_recovery()
            self._app.run_worker(self._app._finish_resume_notices(), exclusive=False)
        except BaseException as exc:
            if self.resume_adopted:
                # Adoption is not reversible. Remove old-session presentation even
                # when resource refresh, rebuild, queue sync or cancellation fails.
                try:
                    await self.show_adopted_resume_recovery(exc)
                except Exception:
                    logger.exception("Failed to display adopted-session recovery")
            raise
        finally:
            self._app._resume_ui_ready.set()
            self._app._sync_usage_subscription()

    async def show_adopted_resume_recovery(self, exc: BaseException) -> None:
        self._app._active_callback = None
        self._app._pending_callbacks.clear()
        self._app._indicate_pending_action()
        try:
            await self._app._queue.clear_server_queue()
        except Exception:
            logger.exception("Failed to clear old-session queue presentation")
        self._app._reset_ui_state()
        with self._app.batch_update():
            await self._app._load_more.hide()
            await self._app._messages_area.remove_children()
            await self._app._messages_area.mount(
                ErrorMessage(
                    f"Session `{self._app.app_server.session_id[:8]}` is active, but its "
                    f"presentation did not finish loading: {exc}. "
                    "Do not rely on old content. Reopen /sessions to retry loading "
                    "or restart to recover this session."
                )
            )
        self._app._refresh_banner()

    async def surface_committed_model_recovery(self) -> None:
        """Surface a resumed session whose committed model left the catalog.

        The transcript is already loaded; the runtime issue reported by the
        server routes the user to the model picker so the next turn runs on an
        explicitly chosen model instead of a silent default.

        The resume has already succeeded at this point, so a failure while
        surfacing (refresh, notice, or picker) must not abort the mount path:
        log it and leave the session usable.
        """
        try:
            runtime = self._app.app_server.resources.runtime
            # The resume response carries no runtime snapshot, so read one before
            # inspecting issues; runtime/updated may still be in flight.
            await runtime.refresh()
            issue = next(
                (
                    issue
                    for issue in runtime.issues
                    if issue.file == COMMITTED_MODEL_RECOVERY_ISSUE_FILE
                ),
                None,
            )
            if issue is None:
                return
            await self._app._mount_and_scroll(
                ErrorMessage(issue.message, collapsed=self._app._tools_collapsed)
            )
            await self._app._switch_to_model_picker_app()
        except Exception:
            logger.exception("Failed to surface the committed-model recovery notice")

    async def reset_presentation_after_resume(self) -> None:
        """Discard presentation state owned by the previously attached session."""
        self._app._entry_expansion_state.reset(
            default_collapsed=self._app._tools_collapsed
        )
        self._app._tool_group_expansion_state.reset(
            default_collapsed=self._app._tools_collapsed
        )
        async with self._app._turn_ui_lock():
            self._app._turn_ui_generation += 1
            self._app._interrupt_requested = False
            self._app._settle_interrupt()
            self._app._turn_supervisor._clear_pending_turn()

            self._app._active_callback = None
            self._app._callback_delivery_uncertain = False
            self._app._pending_callbacks.clear()
            self._app._indicate_pending_action()
            if (
                self._app._pending_local_question is not None
                and not self._app._pending_local_question.done()
            ):
                self._app._pending_local_question.set_result(
                    UserQuestionResult(answers=[], cancelled=True)
                )

            if self._app.event_handler is not None:
                await self._app.event_handler.finalize_streaming()
                self._app.event_handler.stop_current_tool_call(cancelled=True)
                self._app.event_handler.stop_current_compact()

            await self._app._remove_loading_widget()
            self._app._loading_widget = None
            await self._app._queue.clear_server_queue()
            self._app._queue.notify_busy_changed()

    async def finish_resume_notices(self) -> None:
        """Wait for a resumed session's deferred init to settle, then show
        post-init notices — and surface a late init failure.

        Fast resume returns from ``session/resume`` before MCP init
        completes; the server finishes it in ``finish_resume_root`` and then
        emits ``runtime/updated``. Showing notices right after the RPC would
        race that init and read incomplete runtime state, and a background
        failure would go unsurfaced. Re-arm readiness against the now-resumed
        runtime so notices reflect the settled state and failures reach the UI.
        """
        if self._app._post_init_notices_shown:
            return
        try:
            await self._app.app_server.resources.runtime.wait_until_ready()
        except Exception as e:
            if isinstance(e, AppServerResponseError) and e.error.code in {
                ProtocolErrorCode.CONFLICT,
                ProtocolErrorCode.NOT_FOUND,
            }:
                # A newer resume replaced the root and now owns readiness/UI;
                # its own watcher will surface notices for the resumed runtime.
                logger.info("Resume readiness watch superseded by a newer resume")
                return
            logger.exception("Background initialization failed after resume")
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Background initialization failed: {e}",
                    collapsed=self._app._tools_collapsed,
                )
            )
            self._app._refresh_banner()
            return
        await self._app._show_post_init_notices_once()
        self._app._refresh_banner()

    async def auto_resume_on_startup(self) -> None:
        await self._app._mount_and_scroll(
            UserCommandMessage(f"Resuming session{chrome_glyph('truncation')}")
        )
        session_id: str | None = None
        try:
            session_id = self._app._resume_session_id
            if session_id is None:
                # --continue: resolve server-side so the tty-scoped last-session
                # pointer is preferred over the most recently updated session,
                # matching the pre-fast-resume _find_session_to_continue behavior.
                session_id = await self._app.app_server.resources.sessions.resolve_continue_session(
                    self._app.app_server.cwd
                )
                if session_id is None:
                    await self._app._mount_and_scroll(
                        UserCommandMessage("No previous sessions found.")
                    )
                    await self._app._show_custom_tools_deprecation_warning_after_initial_history()
                    await self._app._process_startup_prompt_when_available()
                    return
            await self._app._resume_local_session(session_id)
        except Exception as e:
            logger.exception(
                "Failed to auto-resume session %s", session_id or "<unknown>"
            )
            if not self.resume_adopted:
                try:
                    await self._app._rebuild_transcript_from_current_session()
                except Exception:
                    logger.exception(
                        "Failed to rebuild transcript after resume failure"
                    )
            await self._app._mount_and_scroll(
                ErrorMessage(
                    (
                        f"Session `{self._app.app_server.session_id[:8]}` is active but "
                        f"presentation is incomplete: {e}"
                        if self.resume_adopted
                        else f"Session not changed; failed to resume: {e}"
                    ),
                    collapsed=self._app._tools_collapsed,
                )
            )
        finally:
            self._app._resume_session_id = None
            self._app._continue_latest = False
            # Covers --resume, --continue, and no-sessions-found (all flow here).
            # Must run before _process_startup_prompt so an injected prompt can
            # dispatch.
            self._app._mark_session_ready()
            # _resume_local_session spawns _finish_resume_notices as a worker,
            # but the startup prompt must fire regardless of whether notices
            # have settled — _process_startup_prompt_when_available guards on
            # the two flags cleared above, so the sequence is safe.
            await self._app._process_startup_prompt_when_available()

    async def on_session_picker_app_session_delete_requested(
        self, event: SessionPickerApp.SessionDeleteRequested
    ) -> None:
        from chartreux.cli.textual_ui.app import _get_session_picker_app_class

        if event.session_id == self._app.app_server.session_id:
            self.clear_pending_session_delete(event.option_id)
            await self._app._mount_and_scroll(
                ErrorMessage(
                    "Deleting the current session is not supported.",
                    collapsed=self._app._tools_collapsed,
                )
            )
            return

        try:
            await self._app.app_server.resources.sessions.delete(event.session_id)
        except Exception as e:
            self.clear_pending_session_delete(event.option_id)
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Failed to delete session: {e}",
                    collapsed=self._app._tools_collapsed,
                )
            )
            return

        try:
            picker = self._app.query_one(_get_session_picker_app_class())
        except Exception:
            picker = None

        if picker is not None:
            picker.remove_session(event.option_id)

        await self._app._mount_and_scroll(
            UserCommandMessage(f"Deleted session `{event.session_id[:8]}`.")
        )

        if picker is not None and not picker.has_sessions:
            await self._app._exit_picker_to_input()
            await self._app._mount_and_scroll(
                UserCommandMessage("No saved sessions left for this directory.")
            )

    def clear_pending_session_delete(self, option_id: str) -> None:
        from chartreux.cli.textual_ui.app import _get_session_picker_app_class

        try:
            self._app.query_one(_get_session_picker_app_class()).clear_pending_delete(
                option_id
            )
        except Exception:
            pass

    async def rename_session(self, cmd_args: str = "", **kwargs: Any) -> None:
        title = cmd_args.strip()
        if not title:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    "Usage: /rename <title>", collapsed=self._app._tools_collapsed
                )
            )
            return

        try:
            renamed_title = await self._app.app_server.resources.sessions.rename(title)
        except Exception as e:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Failed to rename session: {e}",
                    collapsed=self._app._tools_collapsed,
                )
            )
            return

        self._app._on_session_title_changed(renamed_title)
        await self._app._mount_and_scroll(
            UserCommandMessage(f'Session renamed to "{renamed_title}".')
        )

    async def branch_session(self, cmd_args: str = "", **kwargs: Any) -> None:
        old_session_id = self._app.app_server.session_id
        try:
            response = await self._app.app_server.resources.sessions.fork(
                entry_id=None, attach=False
            )
        except Exception as e:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Failed to branch session: {e}",
                    collapsed=self._app._tools_collapsed,
                )
            )
            return

        await self._app._mount_and_scroll(
            BranchCreatedMessage(
                old_session_id=old_session_id, new_session_id=response.state.session.id
            )
        )

    def _get_user_message_widgets(self) -> list[UserMessage]:
        """Return all UserMessage widgets currently visible in #messages.

        Only includes completed messages with a public history id (i.e. real
        user messages, not queued prompts or slash-command echo messages).
        """
        return [
            child
            for child in self._app._messages_area.children
            if (
                isinstance(child, UserMessage)
                and child.history_entry_id is not None
                and not child.pending
            )
        ]

    def rewind_prev(self) -> None:
        if self._app._agent_job_active():
            return

        user_widgets = self._get_user_message_widgets()
        if not user_widgets:
            return

        if not self.rewind_mode:
            self.rewind_mode = True
            target = user_widgets[-1]
        elif self.rewind_highlighted_widget is not None:
            try:
                idx = user_widgets.index(self.rewind_highlighted_widget)
            except ValueError:
                idx = len(user_widgets)
            if idx <= 0:
                self._app.run_worker(
                    self._rewind_prev_at_top(), group="rewind", exclusive=False
                )
                return
            target = user_widgets[idx - 1]
        else:
            target = user_widgets[-1]

        self._app.run_worker(
            self._select_rewind_widget(target), group="rewind", exclusive=False
        )

    async def _rewind_prev_at_top(self) -> None:
        """Handle navigating past the topmost visible user message."""
        if self._app._load_more.widget is not None and self._app._has_older_history:
            await self._app.on_history_load_more_requested(HistoryLoadMoreRequested())
            user_widgets = self._get_user_message_widgets()
            if user_widgets and self.rewind_highlighted_widget is not None:
                # Find the current highlighted widget in the refreshed list
                # and select the one above it
                try:
                    idx = user_widgets.index(self.rewind_highlighted_widget)
                except ValueError:
                    idx = 0
                if idx > 0:
                    await self._select_rewind_widget(user_widgets[idx - 1])
                    return
        # No load more or already first message: scroll to top
        self._app.call_after_refresh(self._app._chat_widget.scroll_home, animate=False)

    def rewind_next(self) -> None:
        if not self.rewind_mode:
            return

        if self.rewind_highlighted_widget is None:
            return

        user_widgets = self._get_user_message_widgets()
        try:
            idx = user_widgets.index(self.rewind_highlighted_widget)
        except ValueError:
            return
        if idx >= len(user_widgets) - 1:
            return

        self._app.run_worker(
            self._select_rewind_widget(user_widgets[idx + 1]),
            group="rewind",
            exclusive=False,
        )

    async def _select_rewind_widget(self, widget: UserMessage) -> None:
        """Highlight the given user message widget and show the rewind panel."""
        entry_id = widget.history_entry_id
        if entry_id is None or widget.pending:
            return
        try:
            has_file_changes = await (
                self._app.app_server.resources.sessions.rewind_has_file_changes(
                    entry_id
                )
            )
        except AppServerResponseError as exc:
            self._app.notify(exc.error.message, severity="error")
            if self.rewind_highlighted_widget is None:
                self.rewind_mode = False
            return

        if self.rewind_highlighted_widget is not None:
            self.rewind_highlighted_widget.remove_class("rewind-selected")

        widget.add_class("rewind-selected")
        self.rewind_highlighted_widget = widget

        await self._switch_to_rewind_app(
            widget.get_content(), has_file_changes=has_file_changes
        )

        chat = self._app._chat_widget
        self._app.call_after_refresh(
            chat.scroll_to_widget, widget, animate=False, top=True
        )

    async def _switch_to_rewind_app(
        self, message_preview: str, *, has_file_changes: bool
    ) -> None:
        """Show the rewind action panel at the bottom."""
        from chartreux.cli.textual_ui.app import _get_rewind_app_class

        if self._app._current_bottom_app == BottomApp.Rewind:
            # Reuse existing widget if the option set hasn't changed
            try:
                existing = self._app.query_one(_get_rewind_app_class())
                if existing.has_file_changes == has_file_changes:
                    existing.update_preview(message_preview)
                    return
                await existing.remove()
            except Exception:
                pass

            rewind_app = _get_rewind_app_class()(
                message_preview=message_preview, has_file_changes=has_file_changes
            )
            bottom_container = self._app.query_one("#bottom-app-container")
            self._app._current_bottom_app = BottomApp.Rewind
            await bottom_container.mount(rewind_app)
            self._app.call_after_refresh(rewind_app.focus)
        else:
            rewind_app = _get_rewind_app_class()(
                message_preview=message_preview, has_file_changes=has_file_changes
            )
            await self._app._switch_from_input(rewind_app)

    def _clear_rewind_state(self) -> None:
        if self.rewind_highlighted_widget is not None:
            self.rewind_highlighted_widget.remove_class("rewind-selected")
            self.rewind_highlighted_widget = None
        self.rewind_mode = False

    async def exit_rewind_mode(self) -> None:
        """Exit rewind mode and restore the input panel."""
        self._clear_rewind_state()
        await self._app._switch_to_input_app()

    async def execute_rewind(self, *, restore_files: bool, inplace: bool) -> None:
        if not self.rewind_mode or self.rewind_highlighted_widget is None:
            return

        target_widget = self.rewind_highlighted_widget
        entry_id = target_widget.history_entry_id
        if entry_id is None:
            return

        old_session_id = self._app.app_server.session_id
        await self._app._request_agent_transcript_close(restore_focus=False)
        try:
            result = await self._app.app_server.resources.sessions.rewind(
                entry_id, restore_files=restore_files, inplace=inplace
            )
            await self._app._queue.clear_server_queue()
            await self._app.app_server.resources.refresh()
        except AppServerResponseError as exc:
            self._app.notify(exc.error.message, severity="error")
            return

        message_content = result.message
        restore_errors = result.restore_errors

        for error in restore_errors:
            self._app.notify(error, severity="warning")

        self._clear_rewind_state()
        await self._app._switch_to_input_app()
        await self._app._reset_message_widgets()
        await self._app._resume_history_from_messages()
        if not inplace:
            await self._mount_rewind_fork_message(old_session_id)
        if self._app._chat_input_container:
            self._app._chat_input_container.value = message_content

    async def _mount_rewind_fork_message(self, old_session_id: str) -> None:
        """Mount the fork notice after a non-inplace rewind."""
        await self._app._mount_and_scroll(
            RewindForkMessage(
                old_session_id=old_session_id,
                new_session_id=self._app.app_server.session_id,
            )
        )
