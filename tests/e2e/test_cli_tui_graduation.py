"""Full-TUI graduation integration at the compaction-event boundary.

The second model starts disabled and is enabled and saved through the workbench.
Every slot role binds the mock model except ``@heavy``, which binds the disabled
second model, so the bound roster is single-model until that save transitions
it. A typed compaction event is injected instead of forcing real token overflow;
the signal truth tables live in tests/ui/providers/test_graduation.py. Here the
real app handles the event, idle boundary, clickable notice, persisted dismissal
and presets navigation. A second journey pins a fresh two-model install (the
shipped roster binds both default models), where no save latches and no nudge
appears. A third journey binds a session whose live roster is empty while the
workbench snapshot pre-save roster has one identity: the live empty roster
wins and a 0->2 transition never latches. This is an in-process Textual
journey, not a real-model or manual live pass.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
from textual.widgets import OptionList, SelectionList

from chartreux.app_server.events import SessionCompacted
from chartreux.app_server.models import SessionLogSummary
from chartreux.app_server.protocol import SessionCompactedParams
from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.ui.providers.graduation import GraduationStore
from chartreux.ui.providers.management_state import ManagementState
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen, WorkbenchView
from tests.conftest import build_test_chartreux_app
from tests.e2e.common import write_e2e_config


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("customize", [False, True], ids=["dismiss", "customize"])
async def test_graduation_idle_notice_and_persistent_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, theme: str, customize: bool
) -> None:
    home = tmp_path / "graduation-home"
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_KEYRING", "1")
    write_e2e_config(home, "http://127.0.0.1:1/v1")
    config_path = home / "config.toml"
    config_path.write_text(
        f'theme = "{theme}"\ndisable_welcome_banner_animation = true\n'
    )
    catalog_path = home / "models.toml"
    # A genuine single-model roster: every slot role binds the mock model except
    # @heavy, which binds the disabled second model and stays unbound until the
    # workbench save enables it.
    catalog_path.write_text(
        catalog_path.read_text()
        + '\n[roles.scout]\nmodel = "mock-model"\nthinking = "off"\n'
        + '\n[roles.heavy]\nmodel = "second-model"\nthinking = "off"\n'
        + '\n[models.second-model]\nthinking = "off"\n'
        + '[[models.second-model.deployments]]\nprovider = "mock-provider"\n'
        + 'name = "second-model"\ndisabled = true\n'
        + '\n[dispatch]\nmode = "standalone"\n'
    )
    assert not GraduationStore().state.second_model_saved
    config = (await build_default_orchestrator()).config
    app = build_test_chartreux_app(config=config)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        notice = app.query_one("#graduation-notice")
        assert not notice.display
        assert app.config.dispatch_mode == "standalone"
        # The session-bound roster is single-model while the second model is
        # disabled: @heavy stays unbound.
        assert app.config.bound_roster == ["mock-model"]
        # A modal is a non-idle boundary even when the model is not streaming.
        await app._show_providers()
        await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        screen = app.screen
        providers = screen.query_one("#wb-providers", OptionList)
        providers.highlighted = next(
            i
            for i, option in enumerate(providers.options)
            if option.id == "mock-provider"
        )
        providers.focus()
        await pilot.press("enter")
        await pilot.pause()
        actions = screen.query_one("#wb-provider-operations", OptionList)
        actions.highlighted = next(
            i for i, option in enumerate(actions.options) if option.id == "models"
        )
        actions.focus()
        await pilot.press("enter")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = next(
            i
            for i in range(models.option_count)
            if models.get_option_at_index(i).value == "second-model"
        )
        models.focus()
        await pilot.press("space")
        await pilot.pause()
        assert "second-model" in models.selected
        # Return to provider actions and explicitly persist the toggle.
        await pilot.press("escape")
        await pilot.pause()
        operations = screen.query_one("#wb-provider-operations", OptionList)
        operations.highlighted = next(
            i for i, option in enumerate(operations.options) if option.id == "apply"
        )
        operations.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert GraduationStore().state.second_model_saved
        assert not tomllib.loads(catalog_path.read_text())["models"]["second-model"][
            "deployments"
        ][0]["disabled"]
        assert not notice.display  # second-model save alone is not enough
        # Inject a typed live compaction notification, not a mocked UI action.
        app._refresh_status_for_event(
            SessionCompacted(
                SessionCompactedParams(
                    session_id=app.app_server.session_id,
                    old_session_id=app.app_server.session_id,
                    event_id=1,
                    emitted_at=1,
                    state=app.app_server.state,
                    session_log=SessionLogSummary(enabled=False),
                    summary_length=10,
                )
            )
        )
        await pilot.pause()
        assert not notice.display
        assert not GraduationStore().state.shown
        for _ in range(5):
            if len(app.screen_stack) == 1:
                break
            await pilot.press("escape")
            await pilot.pause()
        # Provider close is the actual reevaluation path, not a presentation mock.
        assert notice.display
        assert GraduationStore().state.shown
        focused = app.screen.focused
        app._maybe_show_graduation()
        assert app.screen.focused is focused
        # Rich click spans in the actual 80-column TUI notice.
        await pilot.click("#graduation-notice", offset=(36 if customize else 48, 0))
        await pilot.pause()
        assert not notice.display
        assert GraduationStore().state.dismissed
        assert app.config.dispatch_mode == "standalone"
        assert (
            tomllib.loads(catalog_path.read_text())["dispatch"]["mode"] == "standalone"
        )
        if customize:
            screen = app.screen
            assert isinstance(screen, ProviderWorkbenchScreen)
            assert screen.view == WorkbenchView.PRESETS
            assert "preset:orchestrator" in {
                option.id
                for option in screen.query_one("#wb-presets", OptionList).options
            }
            await pilot.press("escape")
            await pilot.pause()
    # A fresh app reloads dismissal; a new need signal cannot show it again.
    restarted = build_test_chartreux_app(config=config)
    async with restarted.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        restarted._graduation.state.compacted()
        restarted._maybe_show_graduation()
        assert not restarted.query_one("#graduation-notice").display
        assert restarted._graduation.state.dismissed


@pytest.mark.asyncio
async def test_fresh_two_model_install_never_latches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh R1 install binds both default models; no save latches, no nudge."""
    home = tmp_path / "fresh-home"
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_KEYRING", "1")
    home.mkdir(parents=True)
    (home / "config.toml").write_text("disable_welcome_banner_animation = true\n")
    # No models.toml: the shipped catalog binds glm-5-3 and mistral-large-4.
    assert not GraduationStore().state.second_model_saved
    config = (await build_default_orchestrator()).config
    app = build_test_chartreux_app(config=config)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        notice = app.query_one("#graduation-notice")
        assert not notice.display
        assert app.config.dispatch_mode == "standalone"
        # The projection sorts; a non-None list means a policy is attached.
        assert app.config.bound_roster == ["glm-5-3", "mistral-large-4"]
        # Entering the provider key in Provider Settings makes both default
        # models usable at once, but the roster already bound two identities.
        monkeypatch.delenv("MISTRAL_API_KEY")
        await app._show_providers()
        await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        screen = app.screen
        screen.state = ManagementState.from_snapshot(screen.snapshot, "mistral")
        screen._save_key("fake-key")
        await pilot.pause()
        assert not GraduationStore().state.second_model_saved
        for _ in range(5):
            if len(app.screen_stack) == 1:
                break
            await pilot.press("escape")
            await pilot.pause()
        # A compaction after the key entry still cannot show the nudge.
        app._refresh_status_for_event(
            SessionCompacted(
                SessionCompactedParams(
                    session_id=app.app_server.session_id,
                    old_session_id=app.app_server.session_id,
                    event_id=1,
                    emitted_at=1,
                    state=app.app_server.state,
                    session_log=SessionLogSummary(enabled=False),
                    summary_length=10,
                )
            )
        )
        await pilot.pause()
        assert not notice.display
        assert not GraduationStore().state.shown
        assert not GraduationStore().state.second_model_saved
        assert app._graduation.state.need_signal


@pytest.mark.asyncio
async def test_empty_live_roster_never_latches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session bound with no slots keeps its live empty roster: no 0->2 latch."""
    home = tmp_path / "empty-roster-home"
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_KEYRING", "1")
    home.mkdir(parents=True)
    (home / "config.toml").write_text("disable_welcome_banner_animation = true\n")
    catalog_path = home / "models.toml"
    base = (
        '[providers."mock-provider"]\napi_base = "http://127.0.0.1:1/v1"\n'
        'api_key_env_var = "MISTRAL_API_KEY"\nbackend = "generic"\n\n'
        '[models."mock-model"]\nthinking = "off"\ntemperature = 0.2\n\n'
        '[[models."mock-model".deployments]]\nprovider = "mock-provider"\n'
        'name = "mock-model"\nsupports_images = false\n\n'
        '[roles."orchestrator"]\ndescription = "E2E main assistant preset"\n'
        'model = "mock-model"\nthinking = "high"\n'
    )
    # At session start no slot role resolves: the live policy attaches with an
    # empty roster.
    catalog_path.write_text(
        base
        + '\n[roles.scout]\nmodel = "missing-model"\nthinking = "off"\n'
        + '\n[roles.worker]\nmodel = "missing-model"\nthinking = "off"\n'
        + '\n[roles.heavy]\nmodel = "missing-model"\nthinking = "off"\n'
        + '\n[dispatch]\nmode = "standalone"\n'
    )
    config = (await build_default_orchestrator()).config
    app = build_test_chartreux_app(config=config)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        notice = app.query_one("#graduation-notice")
        assert not notice.display
        assert app.config.dispatch_mode == "standalone"
        assert app.config.bound_roster == []
        # The catalog gains one bound identity before the workbench opens, so
        # the workbench snapshot pre-save roster has one identity while the
        # live roster stays empty.
        catalog_path.write_text(
            base
            + '\n[roles.scout]\nmodel = "mock-model"\nthinking = "off"\n'
            + '\n[roles.worker]\nmodel = "mock-model"\nthinking = "off"\n'
            + '\n[roles.heavy]\nmodel = "second-model"\nthinking = "off"\n'
            + '\n[models.second-model]\nthinking = "off"\n'
            + '[[models.second-model.deployments]]\nprovider = "mock-provider"\n'
            + 'name = "second-model"\ndisabled = true\n'
            + '\n[dispatch]\nmode = "standalone"\n'
        )
        await app._show_providers()
        await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        screen = app.screen
        assert screen._bound_roster_identities(screen.snapshot) == frozenset({
            "mock-model"
        })
        screen.state = ManagementState.from_snapshot(screen.snapshot, "mock-provider")
        state = screen.state
        assert state is not None
        state.toggle("second-model", True)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        # The live roster is empty: a 0->2 transition never latches, even
        # though the workbench snapshot pre-save roster had one identity.
        assert not GraduationStore().state.second_model_saved
        for _ in range(5):
            if len(app.screen_stack) == 1:
                break
            await pilot.press("escape")
            await pilot.pause()
        app._refresh_status_for_event(
            SessionCompacted(
                SessionCompactedParams(
                    session_id=app.app_server.session_id,
                    old_session_id=app.app_server.session_id,
                    event_id=1,
                    emitted_at=1,
                    state=app.app_server.state,
                    session_log=SessionLogSummary(enabled=False),
                    summary_length=10,
                )
            )
        )
        await pilot.pause()
        assert not notice.display
        assert not GraduationStore().state.shown
        assert not GraduationStore().state.second_model_saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "launch_kind", ["verification", "shared-role", "implementation"]
)
async def test_limit_stops_require_implementation_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_kind: str
) -> None:
    from dataclasses import replace

    from chartreux.app_server._projection import project_launch_slot_purposes
    from chartreux.app_server.events import AgentsUpdate, ClientProjection
    from chartreux.app_server.protocol import (
        AgentSummaryModel,
        AgentsUpdateParams,
        Notification,
        RunStopReason,
    )
    from chartreux.core.dispatch.presets import STANDALONE_PRESET
    from chartreux.core.launch_types import LaunchConfig
    from tests.conftest import build_test_agent_loop

    home = tmp_path / "attempt-home"
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    write_e2e_config(home, "http://127.0.0.1:1/v1")
    config = (await build_default_orchestrator()).config
    # Keep @worker unambiguous for the positive control; the shared-role overlay
    # then intentionally makes verification indistinguishable from implementation.
    slots = dict(STANDALONE_PRESET.slots)
    if launch_kind == "shared-role":
        slots["mechanical"] = slots["mechanical"].model_copy(update={"role": "@worker"})
    config.attach_catalog_snapshot(
        replace(
            config.catalog_snapshot,
            dispatch=STANDALONE_PRESET.model_copy(update={"slots": slots}),
        )
    )
    loop = build_test_agent_loop(config=config)
    loop.launch_overrides = LaunchConfig(
        model="@scout" if launch_kind == "verification" else "@worker"
    )
    try:
        purposes = project_launch_slot_purposes(loop, "worker")
        if launch_kind == "shared-role":
            assert purposes == []
        app = build_test_chartreux_app(config=config)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            projection = ClientProjection(app.app_server.state)
            for run_id in ("run-1", "run-2"):
                agent = AgentSummaryModel(
                    agent_id="worker-1",
                    profile="worker",
                    availability="running",
                    current_run_id=run_id,
                    current_run_status="running",
                    current_task_summary="same task",
                    slot_purposes=purposes,
                )
                for stopped in (False, True):
                    if stopped:
                        agent.current_run_status = "completed"
                        agent.availability = "idle"
                        agent.stop_reason = RunStopReason.BUDGET_EXCEEDED
                    params = AgentsUpdateParams(
                        session_id=app.app_server.session_id,
                        event_id=projection.last_event_id + 1,
                        emitted_at=1,
                        agents=[agent],
                    )
                    event = projection.consume(
                        Notification(
                            method="agents/update",
                            params=params.model_dump(mode="json", by_alias=True),
                        )
                    )
                    assert isinstance(event, AgentsUpdate)
                    app._refresh_status_for_event(event)
            assert app._graduation.state.need_signal is (
                launch_kind == "implementation"
            )
    finally:
        await loop.aclose()
