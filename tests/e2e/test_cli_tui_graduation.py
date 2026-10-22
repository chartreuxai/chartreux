"""Full-TUI graduation integration at the compaction-event boundary.

The second model starts disabled and is enabled and saved through the workbench;
no roster shape drives a suggestion anymore. A typed compaction event is
injected instead of forcing real token overflow; the signal truth tables live in
tests/ui/providers/test_graduation.py. Here the real app handles the event, idle
boundary, clickable notice, persisted dismissal and presets navigation. A
second journey pins a fresh two-model install (the shipped roster binds both
default models), where key entry and catalog saves never signal and only the
compaction does. A third journey binds a session whose live roster is empty
while the workbench snapshot pre-save roster has one identity: the live empty
roster stays authoritative and the roster-transitioning save still never
signals. This is an in-process Textual journey, not a real-model or manual
live pass.
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
    assert not GraduationStore().state.need_signal
    config = (await build_default_orchestrator()).config
    app = build_test_chartreux_app(config=config)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        notice = app.query_one("#graduation-notice")
        assert not notice.display
        # The dispatch posture is retired from the config view (package 16b);
        # the legacy [dispatch] table in models.toml is reported loudly at
        # load and never merged.
        assert not hasattr(app.config, "dispatch_mode")
        assert not hasattr(app.config, "bound_roster")
        diagnostics = "\n".join(config.catalog_snapshot.dispatch_diagnostics)
        assert "The legacy [dispatch] table in" in diagnostics
        assert "[orchestration.routes.<purpose>]" in diagnostics
        # The roster is single-model while the second model is disabled:
        # @heavy stays unbound until the workbench save enables it.
        deployments = config.catalog_snapshot.catalog.models["second-model"].deployments
        assert deployments[0].disabled
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
        assert not GraduationStore().state.need_signal
        assert not tomllib.loads(catalog_path.read_text())["models"]["second-model"][
            "deployments"
        ][0]["disabled"]
        assert not notice.display  # a roster-transitioning save alone is not enough
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
        await pilot.click("#graduation-notice", offset=(56 if customize else 66, 0))
        await pilot.pause()
        assert not notice.display
        assert GraduationStore().state.dismissed
        assert not hasattr(app.config, "dispatch_mode")
        # Catalog writes preserve the legacy [dispatch] table byte-for-byte.
        assert (
            tomllib.loads(catalog_path.read_text())["dispatch"]["mode"] == "standalone"
        )
        if customize:
            screen = app.screen
            assert isinstance(screen, ProviderWorkbenchScreen)
            assert screen.view == WorkbenchView.PRESETS
            assert "preset:main" in {
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
async def test_fresh_two_model_install_never_signals_from_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh R1 install binds both default models; only compaction signals."""
    home = tmp_path / "fresh-home"
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_KEYRING", "1")
    home.mkdir(parents=True)
    (home / "config.toml").write_text("disable_welcome_banner_animation = true\n")
    # No models.toml: the shipped catalog binds glm-5-3 and mistral-large-4.
    assert not GraduationStore().state.need_signal
    config = (await build_default_orchestrator()).config
    app = build_test_chartreux_app(config=config)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        notice = app.query_one("#graduation-notice")
        assert not notice.display
        # The dispatch posture is retired from the config view (package 16b).
        assert not hasattr(app.config, "dispatch_mode")
        assert not hasattr(app.config, "bound_roster")
        # The shipped catalog's role bindings cover both default models.
        role_models = {
            role.model for role in config.catalog_snapshot.catalog.roles.values()
        }
        assert role_models == {"glm-5-3", "mistral-large-4"}
        # Entering the provider key in Provider Settings makes both default
        # models usable at once, but no save ever drives a suggestion.
        monkeypatch.delenv("MISTRAL_API_KEY")
        await app._show_providers()
        await pilot.pause()
        assert isinstance(app.screen, ProviderWorkbenchScreen)
        screen = app.screen
        screen.state = ManagementState.from_snapshot(screen.snapshot, "mistral")
        screen._save_key("fake-key")
        await pilot.pause()
        assert not GraduationStore().state.need_signal
        for _ in range(5):
            if len(app.screen_stack) == 1:
                break
            await pilot.press("escape")
            await pilot.pause()
        # A compaction after the key entry is the failure-driven signal: the
        # nudge appears once the app is idle.
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
        assert app._graduation.state.need_signal
        assert notice.display
        assert GraduationStore().state.shown


@pytest.mark.asyncio
async def test_empty_live_roster_save_never_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session bound with no slots keeps its live empty roster; saves never signal."""
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
        '[roles."main"]\ndescription = "E2E main assistant preset"\n'
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
        # The dispatch posture is retired from the config view (package 16b);
        # the legacy [dispatch] table is reported loudly, never merged.
        assert not hasattr(app.config, "dispatch_mode")
        assert not hasattr(app.config, "bound_roster")
        diagnostics = "\n".join(config.catalog_snapshot.dispatch_diagnostics)
        assert "The legacy [dispatch] table in" in diagnostics
        # At session start no dispatch-slot role resolves: the slot roles all
        # bind a model the catalog does not define.
        catalog = config.catalog_snapshot.catalog
        slot_role_models = {
            catalog.roles[name].model for name in ("scout", "worker", "heavy")
        }
        assert slot_role_models == {"missing-model"}
        assert "missing-model" not in catalog.models
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
        # The workbench snapshot's pre-save roster has one bound identity
        # (the bound-roster surface itself is retired with package 16b): the
        # slot roles bind the mock model, and the second model is disabled.
        snapshot_catalog = screen.snapshot.catalog
        role_models = {role.model for role in snapshot_catalog.roles.values()}
        assert role_models == {"mock-model", "second-model"}
        assert snapshot_catalog.models["second-model"].deployments[0].disabled
        screen.state = ManagementState.from_snapshot(screen.snapshot, "mock-provider")
        state = screen.state
        assert state is not None
        state.toggle("second-model", True)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        # The live roster is empty: the roster-transitioning save still never
        # signals, whatever the workbench snapshot pre-save roster held.
        assert not GraduationStore().state.need_signal
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
        assert app._graduation.state.need_signal
        assert notice.display
        assert GraduationStore().state.shown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "launch_kind", ["verification", "shared-role", "implementation"]
)
async def test_limit_stops_require_implementation_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_kind: str
) -> None:
    from chartreux.app_server._projection import project_launch_purposes
    from chartreux.app_server.events import AgentsUpdate, ClientProjection
    from chartreux.app_server.protocol import (
        AgentSummaryModel,
        AgentsUpdateParams,
        Notification,
        RunStopReason,
    )
    from chartreux.core.subagents import derive_agent_name
    from tests.conftest import build_test_agent_loop

    home = tmp_path / "attempt-home"
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    write_e2e_config(home, "http://127.0.0.1:1/v1")
    config = (await build_default_orchestrator()).config
    loop = build_test_agent_loop(config=config)
    # The launch purpose recorded at resolution drives the projection (the
    # dispatch slot purposes are retired with package 16b): a verification
    # launch carries its mechanical purpose, a direct launch carries none,
    # and an implementation launch carries the implementation purpose.
    loop.launch_purpose = {
        "verification": "verification",
        "shared-role": None,
        "implementation": "implementation",
    }[launch_kind]
    try:
        purposes = project_launch_purposes(loop)
        if launch_kind == "shared-role":
            assert purposes == []
        app = build_test_chartreux_app(config=config)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            projection = ClientProjection(app.app_server.state)
            for run_id in ("run-1", "run-2"):
                agent = AgentSummaryModel(
                    agent_id="worker-1",
                    identity="coder",
                    availability="running",
                    current_run_id=run_id,
                    current_run_status="running",
                    current_task_summary="same task",
                    launch_purposes=purposes,
                    name=derive_agent_name(
                        agent_id="worker-1",
                        identity="coder",
                        identity_display_label=None,
                        base_model=None,
                        effective_model=None,
                    ),
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
