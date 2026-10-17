"""Write/persistence controller for the settings screen (W5 11.4a, stage 11)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import JsonValue

from chartreux.app_server.protocol import SettingDescriptorWire
from chartreux.cli.textual_ui.widgets.settings_lists import SettingsOptionList
from chartreux.ui.chrome_glyphs import chrome_glyph

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.screens.settings import SettingsScreen


class SettingsWriteController:
    """Owns the settings screen's save path (``_start_write``/``_write``/``_write_changes``).

    Plain object (no mixins): the screen constructs it in ``__init__`` and is
    its late-bound host — every collaboration is ``self._screen._method(...)``
    at call time, never a constructor-captured bound method. No state fields
    move: everything the save path touches (``_busy``, ``_needs_refresh``,
    ``_error``/``_error_path``/``_unresolved``, ``service``, ``snapshot``,
    ``catalog``/``fields``) is surface-global and stays on the screen, read and
    written through the host exactly where the screen's own code did.

    Pinned invariants, moved verbatim: the synchronous ``_busy`` reservation in
    ``start_write`` (queued activations cannot race worker startup) and the
    ``finally`` refocus predicate in ``write_changes``.
    """

    def __init__(self, screen: SettingsScreen) -> None:
        self._screen = screen

    def start_write(
        self,
        item: SettingDescriptorWire,
        changes: Mapping[str, JsonValue | list[str] | None],
    ) -> None:
        if not self._screen._can_mutate():
            return
        # Reserve synchronously: queued activations cannot race worker startup.
        self._screen._busy = True
        self._screen._error_path = item.path
        self._screen._error = f"{chrome_glyph('running')} Running: Saving {item.label}"
        self._screen._update_help()
        self._screen._update_hint()
        self._screen.run_worker(
            self.write_changes(item, changes, reserved=True), group="settings-write"
        )

    async def write(
        self, item: SettingDescriptorWire, value: JsonValue | list[str] | None
    ) -> None:
        await self.write_changes(item, {item.path: value})

    async def write_changes(
        self,
        item: SettingDescriptorWire,
        changes: Mapping[str, JsonValue | list[str] | None],
        *,
        reserved: bool = False,
    ) -> None:
        if self._screen._busy and not reserved:
            return
        self._screen._busy = True
        self._screen._error_path = item.path
        self._screen._error = f"{chrome_glyph('running')} Running: Saving {item.label}"
        self._screen._update_help()
        self._screen._update_hint()
        try:
            if self._screen._needs_refresh:
                self._screen._error_path = item.path
                self._screen._error = (
                    f"{chrome_glyph('error')} Failed: Could not save settings; "
                    "close and reopen Settings to reconcile external changes."
                )
                self._screen._unresolved[item.path] = self._screen._error
                self._screen._update_help()
                return
            outcome = await self._screen.service.save(
                {
                    path: list[JsonValue](value) if isinstance(value, list) else value
                    for path, value in changes.items()
                },
                self._screen.snapshot.user_revision,
            )
            if outcome.persistence == "not_saved":
                if outcome.error == "conflict":
                    self._screen._needs_refresh = True
                self._screen._error_path = item.path
                self._screen._error = (
                    f"{chrome_glyph('error')} Failed: Could not save {item.label}: "
                    f"{outcome.error or 'write failed'}."
                )
                self._screen._unresolved[item.path] = self._screen._error
                self._screen._update_help()
                return
            # End drafts only after persistence, including saved-but-unknown.
            if self._screen._expanded == item.path:
                self._screen._collapse(restore_opener=False)
            if outcome.snapshot is None:
                self._screen._needs_refresh = True
                self._screen._error = (
                    "! Warning: Settings saved; current state unknown. "
                    "Reopen Settings before editing."
                )
                self._screen._unresolved[item.path] = self._screen._error
                self._screen._update_help()
                return
            self._screen.snapshot = outcome.snapshot
            self._screen.catalog = self._screen.snapshot.catalog
            self._screen.fields = {
                field.path: field for field in self._screen.snapshot.fields
            }
            self._screen._unresolved.pop(item.path, None)
            self._screen._error = (
                f"{chrome_glyph('success')} Saved: {item.label} updated"
            )
            if outcome.shadowed:
                self._screen._error = (
                    "! Warning: Saved but shadowed by a higher layer: "
                    + ", ".join(outcome.shadowed)
                )
            if (
                outcome.persistence == "durability_uncertain"
                or outcome.application == "failed"
            ):
                self._screen._error = (
                    "! Warning: Settings saved, but durability or application "
                    "is uncertain."
                )
            if self._screen._error.startswith("! Warning:"):
                self._screen._unresolved[item.path] = self._screen._error
            self._screen._refresh_options()
        except Exception as exc:
            self._screen._error_path = item.path
            self._screen._error = (
                f"{chrome_glyph('error')} Failed: Could not save {item.label}: {exc}"
            )
            self._screen._unresolved[item.path] = self._screen._error
            self._screen._update_help()
        finally:
            self._screen._busy = False
            if (
                self._screen._expanded is None
                and self._screen._editing is None
                and self._screen._confirmation is None
            ):
                self._screen.query_one("#settings-options", SettingsOptionList).focus()
            self._screen._update_hint()
