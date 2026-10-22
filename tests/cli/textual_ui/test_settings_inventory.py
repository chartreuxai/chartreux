"""Inventory checklist matcher parity, toggle, overlapping filters, patterns, unchecks, resets, and save-failure tests for the settings app suite."""

from __future__ import annotations

from typing import Literal, cast

import pytest
from textual.widgets import Input, OptionList

from chartreux.app_server._resources import _inventory_item_states
from chartreux.cli.textual_ui.screens.settings import (
    SettingsChecklist,
    SettingsOptionList,
    SettingsScreen,
    _draft_values,
    inventory_item_state,
    inventory_name_matches,
    toggle_inventory_name,
)
from chartreux.core.config.settings_catalog import TASK_IDENTITY_DENYLIST_PATH
from chartreux.core.utils.matching import name_matches
from chartreux.ui.settings_service import SettingsSaveOutcome
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui._settings_app_support import (
    FakeService,
    Harness,
    focus_action,
    navigation_payload,
)


@pytest.mark.parametrize("category", ["tools", "skills", "identities"])
@pytest.mark.asyncio
async def test_checklist_glyphs_do_not_rely_on_color(
    category: Literal["tools", "skills", "identities"],
) -> None:
    service = FakeService()
    if category == "identities":
        field = next(
            f for f in service.snapshot.fields if f.path == TASK_IDENTITY_DENYLIST_PATH
        )
    else:
        field = next(
            f for f in service.snapshot.fields if f.path == f"disabled_{category}"
        )
    field.saved_value = [service.snapshot.inventories[category][1]]
    field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*f"inventory_{category}", "enter")
        await pilot.pause()
        checklist = cast(SettingsScreen, pilot.app.screen).query_one(SettingsChecklist)
        checked = list(checklist.render_line(0))
        unchecked = list(checklist.render_line(1))
        assert "".join(segment.text for segment in checked).startswith("▸ [■]")
        assert "".join(segment.text for segment in unchecked).startswith("  [ ]")
        assert checked[0].style is not None
        assert checked[0].style.reverse and checked[0].style.bold
        assert unchecked[0].style is not None
        assert not unchecked[0].style.reverse


@pytest.mark.parametrize("name", ["bash", "read_file", "BASH"])
@pytest.mark.parametrize(
    "entries",
    [[], ["Bash"], ["READ_*"], ["re:^read_.*"], ["re:["], ["  "], ["re:.*file"]],
)
def test_inventory_matcher_agrees_with_server(name: str, entries: list[str]) -> None:
    assert inventory_name_matches(name, entries) == name_matches(name, entries)
    state = _inventory_item_states(
        {"tools": [name], "skills": [], "identities": []}, {"enabled_tools": entries}
    )["tools"][name]
    assert inventory_item_state(name, entries, []) == (
        state.effective,
        state.pattern_driven,
    )


def test_toggle_inventory_mode_and_last_item() -> None:
    assert toggle_inventory_name("bash", [], [], is_enabled=True) == ([], ["bash"])
    assert toggle_inventory_name("bash", [], ["bash"], is_enabled=False) == ([], [])
    assert toggle_inventory_name("bash", ["bash"], ["read_file"], is_enabled=True) == (
        [],
        ["read_file", "bash"],
    )
    assert toggle_inventory_name("bash", ["bash"], ["BASH"], is_enabled=False) == (
        ["bash"],
        [],
    )
    assert toggle_inventory_name("bash", ["read_file"], ["BASH"], is_enabled=False) == (
        ["read_file", "bash"],
        [],
    )
    for category in ("skills",):
        assert toggle_inventory_name(
            "bash", ["bash"], ["bash"], is_enabled=True, category=category
        ) == ([], ["bash"])
    assert toggle_inventory_name("bash", ["bash"], [], is_enabled=True) == (
        [],
        ["bash"],
    )
    assert toggle_inventory_name("bash", ["read_file"], [], is_enabled=False) == (
        ["read_file", "bash"],
        [],
    )


@pytest.mark.parametrize("category", ["tools", "skills", "identities"])
@pytest.mark.parametrize(
    ("enabled", "disabled", "effective", "pattern_driven"),
    [
        (["bash"], ["BASH"], False, False),
        (["bash"], ["bas*"], False, True),
        (["bas*"], ["bash"], False, True),
        (["bas*"], ["b*"], False, True),
        (["read_*"], ["bas*"], False, True),
        (["bash"], ["re:^bas.*"], False, True),
        (["bash"], ["read_*"], True, False),
    ],
)
def test_inventory_overlapping_filters_agree_with_server(
    category: Literal["tools", "skills", "identities"],
    enabled: list[str],
    disabled: list[str],
    effective: bool,
    pattern_driven: bool,
) -> None:
    if category == "identities":
        effective = not inventory_name_matches("bash", disabled)
        pattern_driven = any(
            entry.lower() != "bash" and inventory_name_matches("bash", [entry])
            for entry in disabled
        )
    elif category != "tools":
        effective = inventory_name_matches("bash", enabled)
        pattern_driven = any(
            entry.lower() != "bash" and inventory_name_matches("bash", [entry])
            for entry in enabled
        )
    settings: dict[str, object] = (
        {TASK_IDENTITY_DENYLIST_PATH: disabled}
        if category == "identities"
        else {f"enabled_{category}": enabled, f"disabled_{category}": disabled}
    )
    state = _inventory_item_states({category: ["bash"]}, settings)[category]["bash"]
    assert (state.effective, state.pattern_driven) == (effective, pattern_driven)
    widget_category: Literal["tools", "skills", "agents"] = (
        "tools" if category == "identities" else category
    )
    assert inventory_item_state(
        "bash",
        [] if category == "identities" else enabled,
        disabled,
        category=widget_category,
    ) == (effective, pattern_driven)


@pytest.mark.parametrize(
    ("enabled", "disabled", "locked"),
    [
        (["bash", "read_file"], ["BASH"], False),
        (["bash", "read_file"], ["bas*"], True),
        (["bas*", "read_file"], ["bash"], True),
        (["bas*", "read_file"], ["b*"], True),
    ],
)
@pytest.mark.asyncio
async def test_inventory_overlapping_tools_save_and_reopen(
    enabled: list[str], disabled: list[str], locked: bool
) -> None:
    service = FakeService()
    for field in service.snapshot.fields:
        if field.path in {"enabled_tools", "disabled_tools"}:
            values = enabled if field.path == "enabled_tools" else disabled
            field.effective_value = list(values)
            field.saved_value = list(values)
            field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.selected == ["read_file"]
        assert screen._inventory_item_state("bash") == (False, locked)
        await pilot.press("space")
        assert ("bash" in checklist.selected) is not locked
        await pilot.press("down", "space")
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [
            {"enabled_tools": enabled[:-1]}
            if locked
            else {"enabled_tools": ["bash"], "disabled_tools": []}
        ]
        assert (
            service.snapshot.inventory_states["tools"]["bash"].effective is not locked
        )
        await pilot.press("enter")
        await pilot.pause()
        assert checklist.selected == ([] if locked else ["bash"])
        assert screen._inventory_item_state("bash") == (not locked, locked)


@pytest.mark.asyncio
async def test_inventory_unchanged_default_does_not_write() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert screen._expanded is None
        assert service.saved == []


@pytest.mark.asyncio
async def test_inventory_toggle_commit_discard_and_pattern_lock() -> None:
    service = FakeService()
    for field in service.snapshot.fields:
        if field.path == "enabled_tools":
            field.effective_value = ["bash", "read_*"]
            field.saved_value = ["bash", "read_*"]
            field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.selected == ["bash", "read_file"]
        assert (
            "pattern"
            in str(screen.query_one("#settings-help", NoMarkupStatic).content).lower()
        )
        await pilot.press("down", "space")
        assert checklist.selected == ["bash", "read_file"]
        await pilot.press("up", "space", "escape")
        assert screen._confirmation is not None and not service.saved
        assert "Cancel preserves both saved lists and the current draft" in str(
            screen.query_one("#settings-confirmation-text", NoMarkupStatic).content
        )
        await pilot.press("enter")  # Cancel discard; Apply explicitly persists.
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert screen._expanded is None
        assert service.saved == [{"enabled_tools": ["read_*"]}]


@pytest.mark.asyncio
async def test_inventory_last_allowed_item_restores_default() -> None:
    service = FakeService()
    field = next(
        f for f in service.snapshot.fields if f.path == TASK_IDENTITY_DENYLIST_PATH
    )
    field.effective_value = ["coder"]
    field.saved_value = ["coder"]
    field.saved_explicit = True
    service.snapshot.inventory_states["identities"]["coder"].effective = False
    service.snapshot.inventory_states["identities"]["coder"].default_effective = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_identities", "enter")
        await pilot.pause()
        checklist = cast(SettingsScreen, pilot.app.screen).query_one(SettingsChecklist)
        assert checklist.selected == ["researcher"]
        await pilot.press("space")
        assert checklist.selected == ["coder", "researcher"]
        focus_action(cast(SettingsScreen, pilot.app.screen), "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{TASK_IDENTITY_DENYLIST_PATH: []}]
        await pilot.press("enter")
        await pilot.pause()
        assert checklist.selected == ["coder", "researcher"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_inventory_add_pattern_space_does_not_activate(
    size: tuple[int, int],
) -> None:
    service = FakeService()
    async with Harness(service).run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_skills", "enter", "space")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        actions = focus_action(screen, "add-pattern")
        assert (
            actions.highlighted_option
            and actions.highlighted_option.id == "add-pattern"
        )
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["review"]}
        assert screen._inventory_draft is not None
        draft = {key: list(values) for key, values in screen._inventory_draft.items()}
        snapshot = service.snapshot.model_dump()
        expanded = screen._expanded
        selected = list(checklist.selected)
        highlighted = checklist.highlighted

        await pilot.press("space")
        await pilot.pause()

        assert pilot.app.screen is screen
        assert screen._expanded == expanded
        assert screen._editing is None
        assert checklist.display and actions.has_focus
        assert not screen.query_one("#settings-editor").display
        assert checklist.highlighted == highlighted
        assert checklist.selected == selected
        assert screen._inventory_draft == draft
        assert service.snapshot.model_dump() == snapshot
        assert not service.saved and not service.revisions

        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#settings-input", Input).has_focus
        assert screen.query_one("#settings-editor").display
        assert screen._inventory_draft == draft
        assert service.snapshot.model_dump() == snapshot
        assert not service.saved and not service.revisions


@pytest.mark.asyncio
async def test_inventory_pattern_add_delete_and_reset() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_skills", "enter", "space")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#settings-input", Input).has_focus
        await pilot.press(*"re:^my-", "enter")
        await pilot.pause()
        assert service.saved == []
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["review", "re:^my-"]}
        checklist = screen.query_one(SettingsChecklist)
        added = (screen._inventory_draft or {})["disabled"][-1]
        assert checklist.highlighted is not None
        assert (
            checklist.get_option_at_index(checklist.highlighted).value
            == f"\x00pattern:disabled:{added.token}"
        )
        assert (
            checklist.scroll_y
            <= (checklist.highlighted or 0)
            < checklist.scroll_y + checklist.scrollable_content_region.height
        )
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"disabled_skills": ["review", "re:^my-"]}]
        await pilot.press("enter", "down", "down")
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.get_option_at_index(
            checklist.highlighted or 0
        ).value.startswith("\x00pattern:")
        await pilot.press("ctrl+d", "down", "enter")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["review"]}
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved[-1] == {"disabled_skills": ["review"]}
        await pilot.press("enter", "ctrl+r", "enter")
        assert len(service.saved) == 2
        await pilot.press("ctrl+r", "down", "enter")
        await pilot.pause()
        assert service.saved[-1] == {"enabled_skills": None, "disabled_skills": None}
        assert all(
            not field.saved_explicit
            for field in service.snapshot.fields
            if field.path in {"enabled_skills", "disabled_skills"}
        )


@pytest.mark.asyncio
async def test_pattern_add_highlights_and_reveals_new_row_in_long_checklist() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        assert screen._inventory_draft is not None
        screen._inventory_draft["disabled"] = screen._make_draft([
            f"existing-{index}-*" for index in range(40)
        ])
        screen._render_checklist()
        focus_action(screen, "add-pattern")
        await pilot.press("enter", *"new-pattern-*", "enter")
        await pilot.pause()
        entry = screen._inventory_draft["disabled"][-1]
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.highlighted is not None
        assert checklist.get_option_at_index(checklist.highlighted).value == (
            f"\x00pattern:disabled:{entry.token}"
        )
        assert checklist.scroll_y > 0
        assert (
            checklist.scroll_y
            <= checklist.highlighted
            < checklist.scroll_y + checklist.scrollable_content_region.height
        )
        assert not service.saved


@pytest.mark.asyncio
async def test_inventory_default_consecutive_unchecks_stay_unchecked() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter", "space", "down", "space")
        screen = cast(SettingsScreen, pilot.app.screen)
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["bash", "read_file"]}
        assert screen.query_one(SettingsChecklist).selected == []
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"disabled_tools": ["bash", "read_file"]}]


@pytest.mark.asyncio
async def test_inventory_pattern_deletion_recomputes_and_unlocks() -> None:
    service = FakeService()
    field = next(
        field for field in service.snapshot.fields if field.path == "disabled_tools"
    )
    field.saved_value = ["read_*"]
    field.saved_explicit = True
    field.effective_value = ["read_*"]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        checklist = screen.query_one(SettingsChecklist)
        assert checklist.selected == ["bash"]
        await pilot.press("down", "space")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["read_*"]}
        await pilot.press("down", "ctrl+d", "down", "enter")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": []}
        assert checklist.selected == ["bash", "read_file"]
        await pilot.press("down", "space")
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": ["read_file"]}


@pytest.mark.asyncio
async def test_inventory_project_values_not_copied_to_user() -> None:
    service = FakeService()
    field = next(
        field for field in service.snapshot.fields if field.path == "disabled_tools"
    )
    field.effective_value = ["project_*"]
    field.origin = "project"
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        assert {
            side: _draft_values(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        } == {"enabled": [], "disabled": []}
        await pilot.press("space")
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        assert service.saved == [{"disabled_tools": ["bash"]}]
        assert "project_*" not in str(service.saved)


@pytest.mark.asyncio
async def test_inventory_reset_uses_original_target_after_highlight_moves() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools")
        screen = cast(SettingsScreen, pilot.app.screen)
        screen.action_remove_override()
        assert screen._confirmation is not None
        options = screen.query_one(SettingsOptionList)
        options._query = "inventory_"
        screen._refresh_options()
        options.highlighted = next(
            i
            for i, option in enumerate(options.options)
            if option.id == "inventory_skills"
        )
        await pilot.press("down", "enter")
        await pilot.pause()
        assert service.saved == [{"enabled_tools": None, "disabled_tools": None}]


@pytest.mark.asyncio
async def test_inventory_reset_closes_open_pattern_editor() -> None:
    service = FakeService()
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_tools", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        assert screen.query_one("#settings-input", Input).has_focus
        screen.action_remove_override()
        await pilot.pause()
        assert not screen.query_one("#settings-editor").display
        await pilot.press("down", "enter")
        await pilot.pause()
        assert not screen.query_one("#settings-editor").display
        assert service.saved == [{"enabled_tools": None, "disabled_tools": None}]


@pytest.mark.asyncio
async def test_inventory_save_failure_visible_in_help() -> None:
    service = FakeService()
    service.outcome = SettingsSaveOutcome(
        "not_saved", "unchanged", error="invalid re: pattern"
    )
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press(*"inventory_identities", "enter")
        screen = cast(SettingsScreen, pilot.app.screen)
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(*"re:[", "enter")
        focus_action(screen, "apply")
        await pilot.press("enter")
        await pilot.pause()
        screen = cast(SettingsScreen, pilot.app.screen)
        assert service.saved == [{TASK_IDENTITY_DENYLIST_PATH: ["re:["]}]
        assert "invalid re: pattern" in str(
            screen.query_one("#settings-help", NoMarkupStatic).content
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["enter", "space", "click"])
@pytest.mark.parametrize("row", ["pattern", "locked", "state"])
async def test_wp4_inert_inventory_rows(row: str, gesture: str) -> None:
    service = FakeService()
    field = next(f for f in service.snapshot.fields if f.path == "disabled_tools")
    if row != "state":
        field.saved_value = ["read_*"]
        field.saved_explicit = True
    else:
        service.snapshot.inventories["tools"] = []
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        checklist = screen.query_one(SettingsChecklist)
        checklist.highlighted = 2 if row == "pattern" else 1 if row == "locked" else 0
        await pilot.pause()
        before = navigation_payload(screen)
        if gesture == "click":
            await pilot.click(checklist, offset=(5, checklist.highlighted or 0))
        else:
            await pilot.press(gesture)
        assert navigation_payload(screen) == before
        assert not service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("failure", ["not_saved", "conflict", "exception", "unknown"])
async def test_wp4_inventory_failure_retains_identity_opener_and_scroll(
    reset: bool, failure: str
) -> None:
    class FailingService(FakeService):
        async def save(
            self, changed_leaves: dict, expected_revision: str | None
        ) -> SettingsSaveOutcome:
            if failure == "exception":
                self.saved.append(changed_leaves)
                raise RuntimeError("transport failed")
            return await super().save(changed_leaves, expected_revision)

    service = FailingService()
    service.outcome = (
        SettingsSaveOutcome("saved", "applied")
        if failure == "unknown"
        else SettingsSaveOutcome(
            "not_saved",
            "unchanged",
            error="conflict" if failure == "conflict" else "write failed",
        )
    )
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(30)]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter", "enter")
        checklist = screen.query_one(SettingsChecklist)
        checklist.highlighted = 20
        await pilot.pause()
        checklist.scroll_to(y=18, animate=False, immediate=True, force=True)
        await pilot.pause()
        position = screen._capture_group_position(checklist)
        draft = {
            side: list(entries)
            for side, entries in (screen._inventory_draft or {}).items()
        }
        identities = [o.id for o in checklist.options]
        if reset:
            focus_action(screen, "add-pattern")
            await pilot.press("enter", "ctrl+r")
            assert screen._editing is None  # Explicit editor-reset remains supported.
            await pilot.press("down", "enter")
        else:
            focus_action(screen, "apply")
            await pilot.press("enter")
        await pilot.pause()
        assert len(service.saved) == 1 and not screen._busy
        if failure == "unknown":
            assert screen._inventory_draft is None and screen._expanded is None
            assert screen._needs_refresh and "current state unknown" in screen._error
        else:
            assert (
                screen._inventory_draft == draft
                and screen._expanded == "inventory_tools"
            )
            assert [o.id for o in checklist.options] == identities
            assert position is not None and checklist.scroll_y == position.scroll_y
            actions = screen.query_one("#settings-actions", OptionList)
            assert actions.has_focus and actions.highlighted_option is not None
            assert actions.highlighted_option.id == (
                "add-pattern" if reset else "apply"
            )
            assert screen._needs_refresh is (failure == "conflict")
            if failure == "conflict":
                before = navigation_payload(screen)
                screen._toggle_inventory("member-1")
                screen._save_inventory(
                    next(d for d in screen.catalog if d.path == screen._expanded)
                )
                assert navigation_payload(screen) == before and len(service.saved) == 1


@pytest.mark.asyncio
async def test_review_added_pattern_bookmark_reentry() -> None:
    service = FakeService()
    field = next(f for f in service.snapshot.fields if f.path == "disabled_tools")
    field.saved_value = [f"existing-{index}-*" for index in range(40)]
    field.saved_explicit = True
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        focus_action(screen, "add-pattern")
        await pilot.press("enter")
        screen.query_one("#settings-input", Input).value = "custom-*"
        await pilot.press("enter")
        checklist = screen.query_one(SettingsChecklist)
        opener = checklist.highlighted_option
        assert opener is not None and str(opener.id).startswith("pattern:")
        added = (screen._inventory_draft or {})["disabled"][-1]
        assert opener.id == f"pattern:inventory_tools:disabled:{added.token}"
        scroll = checklist.scroll_y
        assert scroll > 0
        await pilot.press("shift+tab")
        assert checklist.has_focus and checklist.highlighted_option is opener
        assert checklist.scroll_y == scroll
        assert checklist.highlighted is not None
        assert (
            checklist.scroll_y
            <= checklist.highlighted
            < checklist.scroll_y + checklist.scrollable_content_region.height
        )
        await pilot.press("shift+tab", "tab")
        assert checklist.has_focus and checklist.highlighted_option is opener
        assert checklist.scroll_y == scroll
        assert not service.saved


@pytest.mark.asyncio
async def test_review_double_cancel_preserves_pattern_opener() -> None:
    service = FakeService()
    field = next(f for f in service.snapshot.fields if f.path == "disabled_tools")
    field.saved_value = ["long-pattern-" * 40]
    field.saved_explicit = True
    service.snapshot.inventories["tools"] = [f"member-{i}" for i in range(20)]
    async with Harness(service).run_test(size=(80, 24)) as pilot:
        screen = cast(SettingsScreen, pilot.app.screen)
        await pilot.press(*"inventory_tools", "enter")
        checklist = screen.query_one(SettingsChecklist)
        checklist.highlighted = checklist.option_count - 1
        await pilot.pause()
        # Inspect earlier members without changing the selected pattern opener.
        checklist.scroll_to(y=0, animate=False, immediate=True, force=True)
        await pilot.pause()
        opener = checklist.highlighted_option
        assert opener is not None and str(opener.id).startswith("pattern:")
        payload = navigation_payload(screen)
        await pilot.press("ctrl+d")
        assert screen._confirmation is not None
        await pilot.click("#settings-confirmation-actions", offset=(3, 0), times=2)
        await pilot.pause()
        assert screen._confirmation is None and checklist.has_focus
        assert checklist.highlighted_option is opener
        assert navigation_payload(screen) == payload and not service.saved
