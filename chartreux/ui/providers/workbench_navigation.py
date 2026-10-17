"""Navigation controller for the provider workbench with an injected host."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from textual import events
from textual.widget import Widget
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.selection_list import Selection

from chartreux.ui.providers.workbench_types import (
    NavigationFrame,
    WorkbenchView,
    _GroupPosition,
)
from chartreux.ui.providers.workbench_widgets import WorkbenchList
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList


class WorkbenchNavigation:
    """View-stack and focus navigation; all state stays on the host.

    Plain object (no mixins): the screen constructs it in ``__init__`` with
    ``host=self`` and is its late-bound host — every collaboration is
    ``self._host._method(...)`` at call time, never a constructor-captured
    bound method. The frame stack (``_navigation``), group positions, and
    focus state stay on the screen; the controller mutates them through the
    host exactly where the screen's own code did.

    Pinned invariants, moved verbatim: the ``_restore_group_position``
    epoch/owner fences (the ``finish_restore`` closure captures the host), and
    the ``_defer_focus`` ownership tuple (the ``restore`` closure).
    """

    def __init__(self, *, host: Any) -> None:
        self._host = host

    def can_focus_group(self, group_id: str | None) -> bool:
        if (
            self._host._unmounted
            or self._host._busy
            or self._host._dismissed
            or group_id is None
        ):
            return False
        widget = self._host.query_one(f"#{group_id}")
        if (
            not widget.can_focus
            or not widget.display
            or not widget.visible
            or widget.disabled
            or any(not ancestor.display for ancestor in widget.ancestors)
        ):
            return False
        if self._host._confirm:
            return group_id == "wb-confirm-actions"
        return True

    def can_activate(self, group_id: str | None) -> bool:
        """Resolve interaction ownership before pointer focus can dismiss help."""
        if not self.can_focus_group(group_id):
            return False
        if self._host._confirm:
            return group_id == "wb-confirm-actions"
        if self._host._help_open:
            return False
        if self._host._editing or self._host._view == WorkbenchView.EDITOR:
            return group_id == "wb-input"
        return True

    def pointer_focus(self, group_id: str | None) -> bool:
        if not self.can_focus_group(group_id):
            return False
        if self._host._help_open and group_id != "wb-help":
            self._host._help_dismiss_click = group_id
            self._host._help_open = False
            self._host._help_opener = None
            self._host._update_help()
        return True

    def consume_help_click(self, group_id: str | None) -> bool:
        dismissed = self._host._help_dismiss_click == group_id
        self._host._help_dismiss_click = None
        return dismissed

    def focus_groups(self) -> list[str]:
        groups = {
            WorkbenchView.PROVIDERS: ["wb-providers", "wb-root-actions"],
            WorkbenchView.ACTIONS: ["wb-actions", "wb-provider-operations"],
            WorkbenchView.CONNECTION: ["wb-actions", "wb-connection-actions"],
            WorkbenchView.MODELS: ["wb-models", "wb-models-actions", "wb-help"],
            WorkbenchView.CATALOG: [
                "wb-catalog-filter",
                "wb-catalog",
                "wb-catalog-actions",
                "wb-help",
            ],
            WorkbenchView.DETAIL: ["wb-detail-fields", "wb-detail-actions"],
            WorkbenchView.PRESETS: ["wb-presets", "wb-presets-actions"],
            WorkbenchView.PRESET_EDITOR: [
                "wb-preset-editor",
                "wb-preset-editor-actions",
            ],
        }.get(self._host._view, [])
        if self._host._help_open and "wb-help" not in groups:
            groups = [*groups, "wb-help"]
        return [
            group
            for group in groups
            if self._host.query_one(f"#{group}").can_focus
            and self._host.query_one(f"#{group}").display
            and self._host.query_one(f"#{group}").visible
            and all(
                ancestor.display
                for ancestor in self._host.query_one(f"#{group}").ancestors
            )
            and not self._host.query_one(f"#{group}").disabled
        ]

    def cycle_group_check(self) -> bool:
        if self._host._busy or self._host._confirm:
            return True
        return (
            bool(self.focus_groups())
            and not self._host._editing
            and self._host._view != WorkbenchView.EDITOR
        )

    def cycle_group_body(self, direction: int) -> None:
        if self._host._busy:
            return
        if self._host._confirm:
            self._host.query_one("#wb-confirm-actions").focus()
            return
        groups = self.focus_groups()
        if not groups:
            return
        self._host._navigation_epoch += 1
        current = self._host.focused.id if self._host.focused else None
        index = (
            groups.index(current) if current in groups else (-1 if direction > 0 else 0)
        )
        self.capture_group_position()
        target = self._host.query_one(f"#{groups[(index + direction) % len(groups)]}")
        if target.id != "wb-help":
            self._host._help_open = False
            self._host._help_opener = None
        position = self._host._positions.get((
            self._host._view,
            target.id or "",
            self.group_context(self._host._view, target.id or ""),
        ))
        if position is not None:
            self.restore_group_position(position, restore_focus=True)
        else:
            target.focus(scroll_visible=False)
        self._host._update_help()

    def group_context(  # noqa: PLR0911 - each view owns its semantic context
        self, view: WorkbenchView, group_id: str
    ) -> tuple[str, ...]:
        if group_id == "wb-help" and self._host._help_focus_id:
            origin: tuple[WorkbenchView, tuple[str, ...]] | None = (
                self._host._group_contents_context.get(self._host._help_focus_id)
            )
            if origin:
                return (origin[0].value, self._host._help_focus_id, *origin[1])
        provider = self._host.state.provider_id if self._host.state else ""
        if view in {
            WorkbenchView.ACTIONS,
            WorkbenchView.CONNECTION,
            WorkbenchView.MODELS,
        }:
            if self._host._add is not None and self._host._stage == "connection":
                provider = (
                    self._host._add.provider_id or f"draft:{self._host._draft_session}"
                )
            return (provider, self._host._stage or "management")
        if view == WorkbenchView.CATALOG:
            return (
                (self._host._model_filter or "all",) if group_id == "wb-catalog" else ()
            )
        if view == WorkbenchView.DETAIL:
            return (
                provider,
                self._host._detail_preview_wire or self._host._detail or "",
            )
        if view == WorkbenchView.PRESET_EDITOR:
            return (self._host._preset_role or "",)
        if view in {WorkbenchView.PICKER, WorkbenchView.DEPLOYMENTS}:
            return (
                "preset"
                if self._host._preset_field
                else "deployment"
                if self._host._picker
                else "model",
                self._host._preset_role or provider,
                self._host._preset_field
                or self._host._detail
                or self._host._detail_cursor
                or "",
            )
        if view == WorkbenchView.EDITOR:
            group = self._host._help_focus_id or "wb-actions"
            origin = self._host._group_contents_context.get(group)
            if origin:
                return (origin[0].value, group, *origin[1], self._host._editing or "")
            if self._host._navigation:
                opener = self._host._navigation[-1]
                return (
                    opener.view.value,
                    opener.focus_id or "",
                    *opener.context,
                    self._host._editing or "",
                )
        return ()

    def row_identity(self, widget: OptionList, index: int) -> str | None:
        option = widget.get_option_at_index(index)
        if isinstance(widget, SelectionList):
            if widget.id == "wb-models" and index < len(
                self._host._model_row_identities
            ):
                return self._host._model_row_identities[index]
            return str(cast(Selection[str], option).value)
        return str(option.id) if option.id is not None else None

    # Keep the identity-neighbour repair algorithm aligned with SettingsScreen's
    # _capture_group_position/_restore_group_position (and MCPApp's equivalent).
    def capture_group_position(
        self, widget: Widget | None = None
    ) -> _GroupPosition | None:
        widget = widget or self._host.focused
        if widget is None or widget.id is None:
            return None
        owner = self._host._group_contents_context.get(widget.id)
        if isinstance(widget, Input) or widget.id == "wb-help":
            if not widget.display:
                return None
            view = (
                WorkbenchView.EDITOR if isinstance(widget, Input) else self._host._view
            )
            owner = (view, self.group_context(view, widget.id))
            self._host._group_contents_context[widget.id] = owner
        if owner is None:
            if not widget.display:
                return None
            owner = (self._host._view, self.group_context(self._host._view, widget.id))
            self._host._group_contents_context[widget.id] = owner
        view, context = owner
        identities = (
            tuple(
                identity
                for index, option in enumerate(widget.options)
                if not option.disabled
                and (identity := self.row_identity(widget, index)) is not None
            )
            if isinstance(widget, OptionList)
            else ()
        )
        selected = (
            self.row_identity(widget, widget.highlighted)
            if isinstance(widget, OptionList) and widget.highlighted is not None
            else None
        )
        position = _GroupPosition(
            view,
            widget.id,
            context,
            identities,
            identities.index(selected) if selected in identities else 0,
            widget.scroll_x,
            widget.scroll_y,
            widget.cursor_position if isinstance(widget, Input) else None,
        )
        self._host._positions[(view, widget.id, context)] = position
        return position

    def position_before_rebuild(
        self, view: WorkbenchView, group: str
    ) -> _GroupPosition | None:
        widget = self._host.query_one(f"#{group}")
        context = self.group_context(view, group)
        owner = self._host._group_contents_context.get(group)
        previous = self.capture_group_position(widget) if owner is not None else None
        position = self._host._positions.get((view, group, context))
        if owner != (view, context) and isinstance(widget, OptionList):
            widget.highlighted = None
            if group == "wb-catalog" and position is None and previous is not None:
                position = replace(previous, view=view, context=context)
        self._host._group_contents_context[group] = (view, context)
        return position

    def restore_group_position(
        self, position: _GroupPosition | None, *, restore_focus: bool = False
    ) -> None:
        if position is None:
            return
        widget = self._host.query_one(f"#{position.group}")
        if self._host._group_contents_context.get(position.group) != (
            position.view,
            position.context,
        ):
            return
        if isinstance(widget, OptionList):
            valid = [
                i for i, option in enumerate(widget.options) if not option.disabled
            ]
            ids = {self.row_identity(widget, i): i for i in valid}
            for ordinal in sorted(
                range(len(position.identities)),
                key=lambda i: (abs(i - position.ordinal), i < position.ordinal),
            ):
                if position.identities[ordinal] in ids:
                    widget.highlighted = ids[position.identities[ordinal]]
                    break
            else:
                widget.highlighted = (
                    valid[min(position.ordinal, len(valid) - 1)] if valid else None
                )
        if (
            restore_focus
            and self._host._view == position.view
            and widget.display
            and widget.can_focus
            and self.can_focus_group(position.group)
        ):
            if isinstance(widget, WorkbenchList) and widget.id != "wb-confirm-actions":
                self._host._help_focus_id = widget.id
            self._host.set_focus(widget, scroll_visible=False)

        focused = self._host.focused
        view, epoch = self._host._view, self._host._navigation_epoch
        owner = (
            self._host._confirm,
            self._host._editing,
            self._host._help_open,
            self._host._busy,
        )

        def finish_restore() -> None:
            if self._host._unmounted:
                return
            if (
                self._host._dismissed
                or self._host._navigation_epoch != epoch
                or self._host._view != view
                or self._host.focused is not focused
                or (
                    self._host._confirm,
                    self._host._editing,
                    self._host._help_open,
                    self._host._busy,
                )
                != owner
            ):
                return
            if not widget.display or any(
                not ancestor.display for ancestor in widget.ancestors
            ):
                return
            if self._host._group_contents_context.get(position.group) != (
                position.view,
                position.context,
            ):
                return
            if self.group_context(position.view, position.group) != position.context:
                return
            if (
                restore_focus
                and widget.has_focus
                and isinstance(widget, Input)
                and position.cursor_position is not None
            ):
                widget.cursor_position = position.cursor_position
            widget.scroll_to(
                x=position.scroll_x,
                y=position.scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )
            if isinstance(widget, OptionList):
                widget.scroll_to_highlight()

        self._host.call_after_refresh(finish_restore)

    def frame(self) -> NavigationFrame:
        focused = self._host.focused
        selected: str | None = None
        scroll = 0
        if focused is not None:
            selected_index = getattr(focused, "highlighted", None)
            if selected_index is not None:
                if isinstance(focused, SelectionList):
                    selected_option = focused.get_option_at_index(selected_index)
                    selected = str(cast(Selection[str], selected_option).value)
                else:
                    selected_option = getattr(focused, "highlighted_option", None)
                    if selected_option is not None and selected_option.id is not None:
                        selected = str(selected_option.id)
                scroll = int(getattr(getattr(focused, "scroll_offset", None), "y", 0))
        position = self.capture_group_position()
        return NavigationFrame(
            self._host._view,
            focused.id if focused is not None else None,
            selected,
            scroll,
            self._host.filter_text,
            self._host._model_filter,
            self._host._detail_cursor,
            self._host._stage,
            self._host.state.provider_id if self._host.state is not None else None,
            position.context if position else (),
            position,
        )

    def push_view(self, view: WorkbenchView) -> None:
        self._host._navigation_epoch += 1
        self._host._navigation.append(self.frame())
        self._host._view = view

    def activate_view(
        self,
        view: WorkbenchView,
        *,
        focus_id: str | None = None,
        selected_id: str | None = None,
    ) -> None:
        """Display one view and restore focus only after all controls are updated."""
        self._host._navigation_epoch += 1
        self.capture_group_position()
        self._host._view = view
        primary = self._host._sync_view()
        self.restore_view_focus(
            focus_id or ("wb-input" if primary == "editor" else f"wb-{primary}"),
            selected_id=selected_id,
        )

    def defer_focus(self, widget: Widget) -> None:
        view, stage, epoch = (
            self._host._view,
            self._host._stage,
            self._host._navigation_epoch,
        )
        context = self.group_context(view, widget.id or "")
        owner = (
            self._host._confirm,
            self._host._editing,
            self._host._help_open,
            self._host._busy,
        )

        def restore() -> None:
            if (
                not self._host.is_mounted
                or self._host._dismissed
                or self._host._unmounted
            ):
                return
            if not widget.display or any(
                not ancestor.display for ancestor in widget.ancestors
            ):
                return
            if (
                self._host._navigation_epoch == epoch
                and self._host._view == view
                and self._host._stage == stage
                and (
                    self._host._confirm,
                    self._host._editing,
                    self._host._help_open,
                    self._host._busy,
                )
                == owner
                and self.group_context(view, widget.id or "") == context
            ):
                widget.focus()

        self._host.call_after_refresh(restore)

    def restore_view_focus(
        self, widget_id: str, *, selected_id: str | None = None
    ) -> None:
        """Restore an opener only after the shared projection has been applied."""
        if not widget_id.startswith("wb-"):
            widget_id = f"wb-{widget_id}"
        if self._host._confirm:
            widget_id = "wb-confirm-actions"
        widget = self._host.query_one(f"#{widget_id}")
        if (
            not widget.display
            or not widget.visible
            or any(not ancestor.display for ancestor in widget.ancestors)
        ):
            return
        if selected_id is not None and isinstance(widget, OptionList):
            ids = [
                str(cast(Selection[str], option).value)
                if isinstance(widget, SelectionList)
                else str(option.id)
                for option in widget.options
            ]
            if selected_id in ids:
                widget.highlighted = ids.index(selected_id)
        if getattr(widget, "can_focus", False) or hasattr(widget, "focus"):
            self._host.set_focus(widget)
        if selected_id is None:
            self.restore_group_position(
                self._host._positions.get((
                    self._host._view,
                    widget_id,
                    self.group_context(self._host._view, widget_id),
                ))
            )
        if frame_scroll := getattr(self._host, "_restore_scroll", None):
            try:
                widget.scroll_y = frame_scroll
            except (AttributeError, TypeError):
                pass

    def pop_view(self) -> bool:
        if not self._host._navigation:
            return False
        frame = self._host._navigation.pop()
        # Provider typing is a local browser filter; keep changes made while
        # a nested model/editor view was open.
        self._host._model_filter = frame.model_filter
        self._host._detail_cursor = frame.detail_cursor
        self._host._stage = frame.stage
        if frame.view == WorkbenchView.CONNECTION:
            self._host._models_list_open = False
            if self._host.state is not None:
                self._host.state.begin_discovery()
        self._host._view = frame.view
        self._host._return_focus = frame.focus_id
        self.restore_frame(frame)
        return True

    def restore_frame(self, frame: NavigationFrame) -> None:
        self._host._restore_scroll = frame.scroll
        if (
            frame.view != WorkbenchView.CONNECTION
            and frame.provider_id is not None
            and self._host.state is not None
            and frame.provider_id in self._host.state.catalog.providers
        ):
            self._host.state.for_provider(frame.provider_id)
        if frame.view == WorkbenchView.PROVIDERS:
            self._host._refresh_browser()
            self.activate_view(
                frame.view,
                focus_id=frame.focus_id or "wb-providers",
                selected_id=frame.selected_id,
            )
        elif frame.view == WorkbenchView.CATALOG:
            self._host._refresh_catalog()
            self.activate_view(
                frame.view,
                focus_id=(
                    frame.focus_id
                    if frame.focus_id in {"wb-catalog", "wb-catalog-filter"}
                    or (
                        frame.focus_id == "wb-catalog-actions"
                        and self._host.state
                        and self._host.state.dirty
                    )
                    else "wb-catalog"
                ),
                selected_id=frame.selected_id,
            )
        elif frame.view == WorkbenchView.MODELS:
            self._host._refresh_models()
            self.activate_view(
                frame.view,
                focus_id=(
                    frame.focus_id
                    if frame.focus_id in {"wb-models", "wb-models-actions"}
                    else "wb-models"
                ),
                selected_id=frame.selected_id,
            )
        elif frame.view in {WorkbenchView.ACTIONS, WorkbenchView.CONNECTION}:
            if frame.view == WorkbenchView.CONNECTION:
                self._host._refresh_add_connection(highlight=frame.selected_id)
            else:
                self._host._refresh_actions(highlight=frame.selected_id)
            self.activate_view(
                frame.view,
                focus_id=frame.focus_id or "wb-actions",
                selected_id=frame.selected_id,
            )
        else:
            # Nested pickers, editors, confirmations, and preset selection all
            # retain their widgets; restore their recorded control directly.
            self.activate_view(
                frame.view, focus_id=frame.focus_id, selected_id=frame.selected_id
            )
        if (
            frame.view in {WorkbenchView.MODELS, WorkbenchView.CATALOG}
            and frame.focus_id
        ):
            widget = self._host.query_one(f"#{frame.focus_id}")
            widget.scroll_to(y=frame.scroll, animate=False, force=True)
        self.restore_group_position(frame.position)
        self._host._restore_scroll = 0

    def restore_transient_opener(self, frame: NavigationFrame | None) -> None:
        if frame is None:
            return
        self._host._sync_view()
        self.restore_view_focus(frame.focus_id or "wb-providers")
        self.restore_group_position(frame.position, restore_focus=True)

    def return_from_editor(self) -> None:
        opener = self._host._editor_opener
        self._host._editor_opener = None
        self.restore_transient_opener(opener)

    def help_body(self) -> None:
        if self._host._busy or self._host._confirm:
            return
        if self._host._help_open:
            self._host._help_open = False
            opener = self._host._help_opener
            self._host._help_opener = None
            self._host._update_help()
            self.restore_transient_opener(opener)
        else:
            self._host._help_opener = self.frame()
            self._host._help_open = True
            self._host._update_help()

    def back_body(self) -> None:  # noqa: PLR0911, PLR0912, PLR0915
        if self._host._busy:
            self._host._update_help()
            return
        if self._host._confirm:
            self._host.action_confirm_no()
            return
        if self._host._help_open:
            self.help_body()
            return
        if self._host._editing:
            self._host._key_after_env = False
            row = self._host._detail_edit_row
            self._host._editing = None
            self._host._detail_edit_row = None
            self._host.query_one("#wb-input", Input).value = ""
            if row:
                self._host._refresh_detail_fields(highlight=row)
            self.return_from_editor()
            self._host._update_help()
            return
        if self._host._preset_field is not None:
            self.capture_group_position()
            field = self._host._preset_field
            self._host._preset_field = None
            self._host._refresh_preset_editor(highlight=field)
            if not self.pop_view():
                self.activate_view(
                    WorkbenchView.PRESET_EDITOR,
                    focus_id="wb-preset-editor",
                    selected_id=field,
                )
            return
        if self._host._view == WorkbenchView.PRESET_EDITOR:
            self._host._close_preset_editor()
            return
        if self._host._view == WorkbenchView.PRESETS:
            if (
                self._host._flow.escape_unwinds_customize()
                and self._host._customize_presets
            ):
                self._host._customize_presets = False
                self._host._refresh_presets(highlight="customize")
                self._host.query_one("#wb-presets").focus()
                return
            self._host._stage = None
            opener = self._host._presets_opener
            self._host._presets_opener = None
            if opener and opener.focus_id:
                self.restore_frame(opener)
            else:
                self._host._refresh_browser()
                self.activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            return
        if (
            self._host.filter_text
            and not self._host._provider_open
            and not self._host._stage
            and not self._host._models_view
            and not self._host._picker
        ):
            self._host.filter_text = ""
            self._host._refresh_browser()
            return
        if self._host._protocol_picker:
            self._host._protocol_picker = False
            if not self.pop_view():
                self.activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
            return
        if self.can_clear_catalog_filter():
            self._host._model_filter = None
            self._host._detail_cursor = None
            self._host._refresh_catalog()
            self._host.set_focus(
                self._host.query_one("#wb-catalog-filter", NavigableOptionList)
            )
            self._host._update_help()
            return
        if self._host._picker:
            self._host._close_picker()
            return
        if (
            self._host._models_view
            and not self._host._detail
            and not self._host._editing
        ):
            self._host._models_view = False
            if not self.pop_view():
                self._host._provider_open = False
                self.activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            self._host._update_help()
            return
        if self._host._detail:
            if self._host._detail_is_modified():
                self._host._confirm = "detail-discard"
            else:
                self._host._close_detail()
        elif self._host._stage == "models" and self._host._models_list_open:
            self._host._models_list_open = False
            if not self.pop_view():
                self.activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        elif self._host._stage:
            self._host._back_add()
        elif self._host.filter_text and self._host.state is None:
            self._host.filter_text = ""
            self._host._refresh_browser()
        elif self._host._models_list_open:
            self._host._models_list_open = False
            if not self.pop_view():
                self.activate_view(WorkbenchView.ACTIONS, focus_id="wb-actions")
        elif self._host.state:
            if self._host.query_one("#wb-actions").display:
                if not self._host.state.dirty:
                    self._host.state.begin_discovery()
                    self._host.state = None
                self._host._provider_open = False
                self.activate_view(WorkbenchView.PROVIDERS, focus_id="wb-providers")
            elif self._host.state.dirty:
                self._host._confirm = "close"
            else:
                self._host.state = None
                self._host._finish()
        else:
            self._host._finish()
        self._host._update_help()

    def can_clear_catalog_filter(self) -> bool:
        return bool(
            self._host._models_view
            and self._host._view == WorkbenchView.CATALOG
            and self._host._model_filter
            and not self._host._detail
            and not self._host._editing
            and not self._host._picker
        )

    def resize_body(self, event: events.Resize) -> None:
        if self._host.is_mounted:
            too_small = (
                event.size.width < self._host.MIN_WIDTH
                or event.size.height < self._host.MIN_HEIGHT
            )
            workbench = self._host.query_one("#workbench")
            if too_small and workbench.display:
                self._host._resize_frame = self.frame()
                self._host._resize_owner = (
                    self._host._confirm,
                    self._host._editing,
                    self._host._help_open,
                )
            workbench.display = not too_small
            self._host.query_one("#wb-small").display = too_small
            if self._host._busy or too_small:
                return
            if not too_small:
                if self._host._models_view:
                    self._host._refresh_catalog(preserve_filter_cursor=True)
                elif self._host._models_list_open:
                    self._host._refresh_models()
                elif self._host._stage == "connection":
                    if (
                        not self._host._editing
                        and not self._host._confirm
                        and not self._host._detail
                    ):
                        self._host._refresh_add_connection()
                elif self._host._provider_open or self._host._stage == "models":
                    if (
                        not self._host._editing
                        and not self._host._confirm
                        and not self._host._detail
                    ):
                        self._host._refresh_actions()
                else:
                    self._host._refresh_browser()
                self._host._update_help()
                frame, self._host._resize_frame = self._host._resize_frame, None
                if (
                    frame is not None
                    and frame.view == self._host._view
                    and frame.stage == self._host._stage
                    and self._host._resize_owner
                    == (self._host._confirm, self._host._editing, self._host._help_open)
                ):
                    self.restore_transient_opener(frame)
                    if frame.focus_id:
                        self.defer_focus(self._host.query_one(f"#{frame.focus_id}"))
