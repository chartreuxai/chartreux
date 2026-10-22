"""Commit, reload, and confirmation operations with an injected host."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from typing import Any

from textual.widgets import Input

from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
    ProviderWorkbenchResult,
)
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.providers.management_state import (
    ConnectionDraft,
    ManagementState,
    usable_canonical_models,
)
from chartreux.ui.providers.workbench_types import WorkbenchView
from chartreux.ui.providers.workbench_widgets import BrowserList, ModelChecklist


class WorkbenchOperations:
    """Commit, reload, and confirmation operations; all state stays on the host.

    Plain object (no mixins): the screen constructs it in ``__init__`` with
    ``host=self`` and is its late-bound host — every collaboration is
    ``self._host._method(...)`` at call time, never a constructor-captured
    bound method. The commit/reload/confirmation state (``_busy``,
    ``_commit_task``, ``_write_task``, ``_reload_failed``, ``_confirm*``,
    ``_after_commit``, ``_setup_ready``, ``_dismissed``) stays on the screen;
    the controller mutates it through the host exactly where the screen's own
    code did.

    Pinned invariants, moved verbatim: the commit shield/finally is preserved
    exactly — the save-then-recheck preflight loop, the saved-then-cancelled
    adoption, and the finally-block destination switch stay one block each,
    never split; ``on_unmount`` stays an async handler on the screen whose
    body moves here, keeping the ``asyncio.shield`` shutdown reconciliation
    await structure; ``confirm_no_body`` keeps calling the connection delegate
    ``_select_action`` through the host, so the TWP:78-79 instance patch
    stays interceptable; the mode decisions route through the flow policy
    (``host._flow``, converted at 18.3a).
    """

    def __init__(self, *, host: Any) -> None:
        self._host = host

    def apply(self, state: ManagementState) -> None:
        self.start_commit(state, create=False)

    def request_commit(self) -> None:
        if self._host._busy:
            return
        if self._host.state is None:
            first = next(iter(self._host.snapshot.catalog.providers), None)
            if first is None:
                self._host._set_feedback(
                    "catalog", "Configure a provider first.", "warning"
                )
                return
            self._host.state = ManagementState.from_snapshot(self._host.snapshot, first)
        if self._host._flow.commit_destination() == "presets":
            self._host._open_presets()
            return
        self.apply(self._host.state)

    def set_busy(self, busy: bool) -> None:
        self._host._busy = busy
        for widget in self._host.query("Input, SelectionList, OptionList"):
            widget.disabled = busy
        if busy:
            self._host._feedback_kind = "running"
        if not busy:
            too_small = (
                self._host.size.width < self._host.MIN_WIDTH
                or self._host.size.height < self._host.MIN_HEIGHT
            )
            self._host.query_one("#workbench").display = not too_small
            self._host.query_one("#wb-small").display = too_small
        self._host._update_help()

    def start_commit(  # noqa: PLR0911, PLR0912
        self, state: ManagementState, *, create: bool
    ) -> None:
        if self._host._busy or state is not self._host.state or self._host._dismissed:
            return
        if self._host._reload_failed:
            self._host._message = (
                "Retry Runtime Reload before saving more catalog changes."
            )
            self._host._update_help()
            return
        if self._host._detail is not None:
            self._host._save_detail()
            if self._host._detail is not None:
                return
        if self._host._editing is not None:
            self._host._accept_editor()
            if self._host._editing is not None:
                return
        try:
            changes = state.changes(
                mode=self._host._flow.commit_mode(self._host.mode),
                credential_resolver=self._host._credential_value,
            )
            if (
                self._host._after_commit == "models"
                and self._host._add is not None
                and self._host._add.preset == "existing"
            ):
                # Editing a saved connection is its own checkpoint. Keep model
                # selections in the draft until the Models forward action.
                changes = replace(changes, models={}, roles=None)
            if create:
                if self._host._add is None:
                    return
                provider = self._host.snapshot.catalog.providers.get(state.provider_id)
                is_mistral_preset = (
                    state.provider_id == "mistral"
                    and self._host._add.preset == "mistral"
                )
                if provider is not None and not is_mistral_preset:
                    raise ValueError("Provider already exists; choose another name.")
                connection_changed = provider is not None and (
                    state.connection != ConnectionDraft.from_definition(provider)
                )
                if is_mistral_preset and provider is not None:
                    customized = (
                        state.provider_id in self._host.snapshot.overlaid_providers
                    )
                    if (
                        customized
                        and connection_changed
                        and not self._host._mistral_overwrite_approved
                    ):
                        self._host._confirm = "overwrite-mistral"
                        self._host._update_help()
                        return
                if provider is None or connection_changed:
                    # A new provider persists its full staged definition, and a
                    # changed preset overrides the shipped one. An unchanged
                    # preset over an existing definition persists nothing: the
                    # provider already lives in the base catalog, and writing
                    # its table into the overlay would flip overlaid_providers
                    # provenance and block first-run materialization of the
                    # default template.
                    changes = replace(
                        changes,
                        provider={
                            **state.catalog.providers[state.provider_id].model_dump(),
                            **changes.provider_patches.get(state.provider_id, {}),
                        },
                    )
            elif not state.dirty:
                self._host._feedback_kind = "info"
                self._host._message = "No catalog changes to apply."
                self._host._update_help()
                return
            if not self._host._connection_approved:
                affected = [
                    name
                    for name, connection in state.connections.items()
                    if name in self._host.snapshot.overlaid_providers
                    and name in self._host.snapshot.catalog.providers
                    and not (create and name == state.provider_id)
                    and connection
                    != connection.from_definition(
                        self._host.snapshot.catalog.providers[name]
                    )
                ]
                if affected:
                    self._host._connection_scopes = affected
                    self._host._confirm = "replace-connection"
                    self._host._update_help()
                    return
            # The draft is mutable, including nested deployment patches. Never hand
            # any of its containers to the writer thread.
            changes = deepcopy(
                replace(changes, expected_revision=self._host.snapshot.revision)
            )
        except ValueError as exc:
            self._host._set_feedback("catalog", str(exc), "error")
            return
        self._host._message = f"Saving{chrome_glyph('running')}"
        self._host._feedback_kind = "running"
        self._host._mistral_overwrite_approved = False
        self._host._connection_approved = False
        self._host._commit_return = self._host._frame()
        self.set_busy(True)
        self._host._commit_task = asyncio.create_task(
            self.commit(changes, state, create=create)
        )

    async def commit(  # noqa: PLR0912, PLR0915
        self, changes: CatalogChanges, state: ManagementState, *, create: bool
    ) -> None:
        previously_usable = usable_canonical_models(
            self._host.snapshot,
            self._host._persisted_credential_value,
            allowed_models=self._host.allowed_models,
            thinking_overrides=self._host.thinking_overrides,
        )
        write = asyncio.create_task(
            asyncio.to_thread(self._host.catalog_writer.apply_changes, changes)
        )
        self._host._write_task = write
        saved = False
        try:
            # Shield the thread operation even when a screen worker is cancelled.
            while True:
                try:
                    result = await asyncio.shield(write)
                    break
                except asyncio.CancelledError:
                    if write.done():
                        result = write.result()
                        break
            if isinstance(result, CatalogValidationError):
                self._host._set_feedback(
                    "catalog",
                    f"Failed to {'create' if create else 'apply'}: {result.message}",
                    "error",
                )
                return
            saved = True
            self._host._changed |= result.changed
            self._host.snapshot = result.snapshot
            state.begin_discovery()
            self._host.state = ManagementState.from_snapshot(
                result.snapshot, state.provider_id
            )
            if result.changed:
                self._host._notify_models_saved(previously_usable, frozenset())
            if create:
                self._host._stage = None
                self._host._add = None
                self._host._add_checkpoint = None
                self._host._add_state = None
                self._host._pre_add_state = None
                self._host._navigation.clear()
                self._host._provider_open = False
                self._host._models_list_open = False
                self._host._view = WorkbenchView.PROVIDERS
                self._host._sync_view()
                self._host._browser_cursor = state.provider_id
                self._host.query_one("#wb-providers", BrowserList).highlighted = None
            await self.reload_catalog()
            if not self._host._reload_failed:
                self._host._set_feedback(
                    "catalog",
                    "Provider created." if create else "Saved catalog changes.",
                    "success",
                )
            self._host._refresh_browser()
            if create:
                self._host.state = None
                self._host._view = WorkbenchView.PROVIDERS
                self._host._refresh_browser()
                browser = self._host.query_one("#wb-providers", BrowserList)
                ids = [str(option.id) for option in browser.options]
                if state.provider_id in ids:
                    browser.highlighted = ids.index(state.provider_id)
                self._host._activate_view(
                    WorkbenchView.PROVIDERS,
                    focus_id="wb-providers",
                    selected_id=state.provider_id,
                )
                browser.focus()
                self._host._defer_focus(browser)
            else:
                self._host._refresh_actions(highlight="apply")
                self._host._refresh_models()
                if self._host._models_view:
                    self._host._refresh_catalog()
        except asyncio.CancelledError:
            if saved:
                self._host._reload_failed = True
                self._host._warning = "Runtime reload failed."
                self._host._message = "Saved; Reload Failed."
            else:
                raise
        except Exception:
            if saved:
                self._host._reload_failed = True
                self._host._warning = "Runtime reload failed."
                self._host._message = "Saved; Reload Failed."
            else:
                self._host._set_feedback(
                    "catalog",
                    "Failed to create provider. Draft preserved."
                    if create
                    else "Failed to apply catalog changes. Draft preserved.",
                    "error",
                )
        finally:
            if self._host._message.startswith(("Failed", "Saved; Reload Failed")):
                self._host._feedback_kind = "error"
            self._host._write_task = None
            self.set_busy(False)
            if saved and create and not self._host._reload_failed:
                browser = self._host.query_one("#wb-providers", BrowserList)
                browser.focus()
                self._host._defer_focus(browser)
            frame = self._host._commit_return
            self._host._commit_return = None
            if frame is not None and (not create or not saved):
                self._host._restore_frame(frame)
            destination = self._host._after_commit
            self._host._after_commit = None
            if saved and not self._host._reload_failed:
                if destination == "models":
                    self._host.state = ManagementState.from_snapshot(
                        self._host.snapshot, state.provider_id
                    )
                    if self._host._connection_pending is not None:
                        self._host.state.pending = self._host._connection_pending
                        self._host._connection_pending = None
                    self._host._stage = "models"
                    self._host._models_list_open = True
                    self._host._refresh_models()
                    self._host._activate_view(
                        WorkbenchView.MODELS, focus_id="wb-models"
                    )
                    self._host._defer_focus(
                        self._host.query_one("#wb-models", ModelChecklist)
                    )
                    self._host.run_worker(
                        self._host._discovery_request(self._host.state),
                        group="workbench-discovery",
                        exclusive=True,
                    )
                elif destination == "another":
                    self._host._start_add()
                elif destination == "presets":
                    self._host._open_presets()
                elif destination == "finish":
                    self._host._setup_ready = True
                    self.finish()

    async def reload_catalog(self) -> None:
        try:
            reload = await self._host.config.reload_catalog_and_config()
        except Exception:
            reload = None
        if reload is None or reload.snapshot is None:
            self._host._reload_failed = True
            self._host._warning = (
                reload.message if reload is not None else None
            ) or "Runtime reload failed."
            self._host._set_feedback("reload", "Saved; Reload Failed.", "error")
        else:
            self._host._reload_failed = False
            self._host._warning = None
            self._host.snapshot = reload.snapshot
            if self._host.state:
                self._host.state = ManagementState.from_snapshot(
                    reload.snapshot, self._host.state.provider_id
                )
            self._host._set_feedback("reload", "Runtime reloaded.", "success")
        self._host._refresh_actions()
        self._host._update_help()

    def retry_reload(self) -> None:
        if self._host._busy or not self._host._reload_failed:
            return
        self._host._message = f"Reloading{chrome_glyph('running')}"
        self._host._feedback_kind = "running"
        self.set_busy(True)
        self._host._commit_task = asyncio.create_task(self.retry_reload_task())

    async def retry_reload_task(self) -> None:
        try:
            await self.reload_catalog()
        finally:
            self.set_busy(False)

    async def unmount_body(self) -> None:
        self._host._unmounted = True
        if self._host._commit_task is not None and not self._host._commit_task.done():
            await asyncio.shield(self._host._commit_task)

    def restore_confirmation_focus(self) -> None:
        """Return focus to the control that opened a completed confirmation."""
        opener_id = self._host._confirm_opener
        self._host._confirm_opener = None
        if opener_id:
            opener = self._host.query(f"#{opener_id}")
            if opener and opener.first().display:
                opener.first().focus()
                self._host._restore_group_position(
                    self._host._confirm_position, restore_focus=True
                )
                self._host._confirm_position = None
                return
        for widget_id in ("wb-catalog", "wb-actions", "wb-providers"):
            widget = self._host.query_one(f"#{widget_id}")
            if widget.display and getattr(widget, "can_focus", False):
                widget.focus()
                return

    def confirm_yes_body(self) -> None:
        if self._host._busy:
            return
        if not self._host._confirm:
            return
        decision = self._host._confirm
        self._host._confirm = None
        self._host._sync_view()
        state = self._host.state
        if decision == "replace-connection" and state:
            self._host._connection_approved = True
            self.start_commit(state, create=self._host._stage == "models")
        elif decision == "overwrite-mistral" and state:
            self._host._mistral_overwrite_approved = True
            self.start_commit(state, create=True)
        elif decision == "shared-key":
            self._host._save_key(
                self._host.query_one(
                    f"#{self._host._credential_input_id}", Input
                ).value.strip()
            )
        elif decision in {"add-discard", "add-discard-simple"}:
            if self._host.state:
                self._host.state.begin_discovery()
            self._host.state = self._host._pre_add_state
            self._host._pre_add_state = None
            self._host._add_state = None
            self._host._add = None
            self._host._add_checkpoint = None
            self._host._stage = None
            self._host._navigation.clear()
            self._host._provider_open = False
            self._host._models_list_open = False
            self._host._view = WorkbenchView.PROVIDERS
            self._host._refresh_browser()
            self._host._activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            self._host._restore_transient_opener(self._host._add_opener)
            self._host._add_opener = None
        elif decision == "discard" and state:
            state.begin_discovery()
            self._host.state = ManagementState.from_snapshot(
                self._host.snapshot, state.provider_id
            )
            self._host._refresh_actions()
            self._host._refresh_browser()
            if self._host._models_view:
                self._host._refresh_catalog()
        elif decision == "close":
            self.finish()
        elif decision.startswith("existing:") and state:
            wire = decision.partition(":")[2]
            item = state.pending[wire]
            state.pending[wire] = replace(item, decision="add_existing")
            self._host._refresh_actions(highlight=f"collision:{wire}")
        self._host._update_help()
        if decision == "discard" or decision.startswith("existing:"):
            self.restore_confirmation_focus()

    def confirm_continue_body(self) -> None:
        if (
            self._host._busy
            or self._host._confirm != "add-discard"
            or self._host._add_state is None
        ):
            return
        self._host._confirm = None
        self._host._sync_view()
        self._host._confirm_opener = None
        self._host._connection_action("continue")

    def confirm_detail_save_body(self) -> None:
        if self._host._busy or self._host._confirm != "detail-discard":
            return
        self._host._confirm = None
        self._host._sync_view()
        self._host._confirm_opener = None
        self._host._save_detail()

    def confirm_detail_discard_body(self) -> None:
        if self._host._busy or self._host._confirm != "detail-discard":
            return
        self._host._confirm = None
        self._host._sync_view()
        self._host._confirm_opener = None
        self._host._close_detail()

    def confirm_no_body(self) -> None:
        if self._host._busy:
            return
        if not self._host._confirm:
            return
        decision = self._host._confirm
        self._host._confirm = None
        self._host._sync_view()
        if decision == "shared-key":
            self._host._restore_view_focus(self._host._credential_input_id)
        elif self._host._confirm_opener:
            self._host._restore_view_focus(self._host._confirm_opener)
        self._host._restore_group_position(
            self._host._confirm_position, restore_focus=True
        )
        self._host._confirm_position = None
        self._host._confirm_opener = None
        if decision.startswith("existing:") and self._host.state:
            wire = decision.partition(":")[2]
            item = self._host.state.pending[wire]
            self._host.state.pending[wire] = replace(item, decision="separate")
            self._host._select_action(f"collision:{wire}")
        self._host._update_help()

    def finish(self) -> None:
        if self._host._busy:
            return
        self._host._dismissed = True
        if self._host.state:
            self._host.state.begin_discovery()
        self._host.dismiss(
            ProviderWorkbenchResult(
                "completed"
                if self._host._flow.finish_outcome(
                    self._host._setup_ready, self._host._changed
                )
                else "cancelled",
                changed=self._host._changed,
                warning=self._host._warning,
            )
        )
