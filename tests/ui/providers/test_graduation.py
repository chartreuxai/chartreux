from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.inline_notice import InlineNotice
from chartreux.core.model_catalog.contracts import ProviderWorkbenchResult
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.setup.onboarding import OnboardingApp
from chartreux.ui.providers.graduation import GraduationState, GraduationStore
from chartreux.ui.providers.management_state import (
    ManagementState,
    usable_canonical_models,
)
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.conftest import build_test_vibe_config
from tests.ui.providers._workbench_support import Host, setup


def test_model_saved_never_gates() -> None:
    state = GraduationState()
    for before, after, saved in (
        (frozenset({"a"}), frozenset({"a", "b"}), frozenset({"b"})),
        (frozenset({"a", "b"}), frozenset({"a", "b"}), frozenset({"a", "b"})),
        (frozenset(), frozenset({"a", "b"}), frozenset({"a", "b"})),
        (frozenset({"a"}), frozenset({"a"}), frozenset({"a"})),
        (frozenset({"a"}), frozenset({"a", "b"}), frozenset()),
    ):
        state.model_saved(before, after, saved)
        state.model_saved(before, after, saved, succeeded=False, linked=True)
    assert not state.need_signal
    assert not state.eligible(idle=True, headless=False)


def test_eligibility_gates() -> None:
    state = GraduationState()
    assert not state.eligible(idle=True, headless=False)
    state.compacted()
    assert state.eligible(idle=True, headless=False)
    assert not state.eligible(idle=False, headless=False)
    assert not state.eligible(idle=True, headless=True)
    state.shown = True
    assert not state.eligible(idle=True, headless=False)
    state.shown = False
    state.dismissed = True
    assert not state.eligible(idle=True, headless=False)


def test_failure_dedup_scope_budget_and_exclusions() -> None:
    state = GraduationState()
    for _ in range(3):
        state.implementation_failed("task", "run1", attempt_budgeted=True)
    state.implementation_failed("other-task", "run2", attempt_budgeted=True)
    for excluded in ("replayed", "cancelled", "transient"):
        state.implementation_failed(
            "task", "excluded", attempt_budgeted=True, **{excluded: True}
        )
    state.implementation_failed("task", "unbudgeted", attempt_budgeted=False)
    assert not state.eligible(idle=True, headless=False)
    state.implementation_failed("task", "run2", attempt_budgeted=True)
    assert state.eligible(idle=True, headless=False)
    assert len(state.failures["task"]) == 2


def test_replayed_compaction_excluded() -> None:
    state = GraduationState()
    state.compacted(replayed=True)
    assert not state.eligible(idle=True, headless=False)


def test_compaction_signal_is_not_persisted(tmp_path: Path) -> None:
    path = tmp_path / "graduation.toml"
    store = GraduationStore(path)
    assert not path.exists()
    store.state.compacted()
    assert store.save()
    resumed = GraduationStore(path)
    assert not resumed.state.need_signal
    resumed.state.shown = True
    resumed.state.dismissed = True
    assert resumed.save()
    restarted = GraduationStore(path)
    restarted.state.compacted()
    assert not restarted.state.eligible(idle=True, headless=False)
    assert restarted.state.dismissed


def test_interleaved_stores_preserve_durable_flags(tmp_path: Path) -> None:
    path = tmp_path / "graduation.toml"
    first = GraduationStore(path)
    stale = GraduationStore(path)
    first.state.shown = True
    first.state.dismissed = True
    assert first.save()
    stale.state.compacted()
    assert stale.save()
    resumed = GraduationStore(path)
    assert resumed.state.shown
    assert resumed.state.dismissed
    assert stale.state.dismissed


def test_invalid_file_is_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "graduation.toml"
    path.write_text('shown = "invalid"\n')
    store = GraduationStore(path)
    assert not store.save()
    assert path.read_text() == 'shown = "invalid"\n'


def test_persisted_roster_latch_from_older_release_is_ignored(tmp_path: Path) -> None:
    # A store persisted by an older release carries a roster-size latch key the
    # v0.5.7 state no longer reads: it neither signals on its own nor blocks a
    # fresh failure-driven signal.
    path = tmp_path / "graduation.toml"
    path.write_text("second_model_saved = true\n")
    store = GraduationStore(path)
    assert not store.state.eligible(idle=True, headless=False)
    store.state.compacted()
    assert store.state.eligible(idle=True, headless=False)


def with_slot_roles(data: dict, *, single_model: bool = False) -> dict:
    """Bind the shipped slot roles so the roster tracks models a and b.

    ``@worker``/``@scout`` bind ``a`` and ``@heavy`` binds ``b``, so the bound
    roster is single-model while ``b`` is disabled and multi-model once the
    second model is enabled. ``single_model`` drops ``a`` so no slot binds.
    """
    if single_model:
        data["models"].pop("a")
    data["roles"]["worker"] = {"model": "a", "thinking": "medium"}
    data["roles"]["scout"] = {"model": "a", "thinking": "low"}
    data["roles"]["heavy"] = {"model": "b", "thinking": "high"}
    return data


def recording_graduation(
    screen: ProviderWorkbenchScreen,
) -> tuple[
    GraduationState, list[tuple[frozenset[str], frozenset[str], frozenset[str]]]
]:
    graduation = GraduationState()
    calls: list[tuple[frozenset[str], frozenset[str], frozenset[str]]] = []

    def record(
        roster_before: frozenset[str],
        roster_after: frozenset[str],
        saved_models: frozenset[str],
    ) -> None:
        calls.append((roster_before, roster_after, saved_models))
        graduation.model_saved(roster_before, roster_after, saved_models)

    screen.on_models_saved = record
    return graduation, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("single_model", [False, True])
async def test_roster_transition_reported_without_signal(single_model: bool) -> None:
    screen, services = setup()
    data = with_slot_roles(
        services.catalog.catalog.model_dump(), single_model=single_model
    )
    data["models"]["b"]["deployments"][0]["disabled"] = True
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "disabled")
    screen.snapshot = services.catalog
    graduation, calls = recording_graduation(screen)
    assert len(usable_canonical_models(screen.snapshot, services.resolve_key)) == (
        0 if single_model else 1
    )
    async with Host(screen).run_test() as pilot:
        state = ManagementState.from_snapshot(screen.snapshot, "two")
        screen.state = state
        state.toggle("b", True)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        assert services.writes
        assert calls
        # The dispatch roster is retired (package 16b): the callback receives
        # empty rosters and the saved-models delta, and never signals.
        assert calls[-1][0] == frozenset()
        assert calls[-1][1] == frozenset()
        assert "b" in calls[-1][2]
        assert not graduation.need_signal


@pytest.mark.asyncio
async def test_fresh_two_model_roster_never_signals() -> None:
    """A fresh two-model install: key entry and catalog saves never signal."""
    screen, services = setup()
    data = with_slot_roles(services.catalog.catalog.model_dump())
    data["providers"]["two"]["api_key_env_var"] = "MISSING_KEY"
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "two-model")
    screen.snapshot = services.catalog
    graduation, calls = recording_graduation(screen)
    # The bound-roster surface is retired (package 16b): the equivalent fact is
    # that the catalog role bindings cover both models.
    role_models = {role.model for role in screen.snapshot.catalog.roles.values()}
    assert role_models == {"a", "b"}
    async with Host(screen).run_test() as pilot:
        screen.state = ManagementState.from_snapshot(screen.snapshot, "two")
        screen._save_key("test-value")
        await pilot.pause()
        assert services.keys == [("MISSING_KEY", "test-value")]
        assert calls
        state = screen.state
        assert state is not None
        state.toggle("b", False)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        assert services.writes
        assert not graduation.need_signal


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", [False, True])
async def test_session_only_save_is_not_a_durable_roster_save(
    persisted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.core.model_catalog.contracts import CredentialSaveResult

    screen, services = setup()
    if not persisted:
        monkeypatch.setattr(
            services,
            "save_key",
            lambda env, value: CredentialSaveResult("session_only"),
        )
    data = with_slot_roles(services.catalog.catalog.model_dump())
    data["providers"]["two"]["api_key_env_var"] = "MISSING_KEY"
    data["models"]["b"]["deployments"][0]["disabled"] = True
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "missing-key")
    screen.snapshot = services.catalog
    graduation, calls = recording_graduation(screen)
    async with Host(screen).run_test() as pilot:
        screen.state = ManagementState.from_snapshot(screen.snapshot, "two")
        screen._save_key("test-value")
        await pilot.pause()
        assert services.keys == ([("MISSING_KEY", "test-value")] if persisted else [])
        # Only a durable save reports through the onboarding callback.
        assert bool(calls) is persisted
        state = screen.state
        assert state is not None
        state.toggle("b", True)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        assert services.writes
        assert calls
        assert not graduation.need_signal


@pytest.mark.asyncio
async def test_session_only_provenance_survives_workbench_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.core.model_catalog.contracts import CredentialSaveResult

    initial_screen, services = setup()
    data = with_slot_roles(services.catalog.catalog.model_dump())
    data["providers"]["two"]["api_key_env_var"] = "REOPEN_TEST_KEY"
    data["models"]["b"]["deployments"][0]["disabled"] = True
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "disabled")
    graduation, _calls = recording_graduation(initial_screen)
    owner = SimpleNamespace(
        config=SimpleNamespace(enable_system_trust_store=False),
        _graduation_models_saved=initial_screen.on_models_saved,
        _mount_and_scroll=AsyncMock(),
    )
    # Use the real app initialization and opening path, but a lightweight UI host.
    ChartreuxApp._init_cached_widgets(owner)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.app._ProviderCredentials", lambda: services
    )
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.app._ProviderConfigService", lambda app: services
    )
    monkeypatch.setattr(
        "chartreux.core.model_catalog.loader.CatalogStore", lambda: services
    )
    monkeypatch.setattr(
        "chartreux.core.model_catalog.loader.load_catalog", lambda: services.catalog
    )

    def session_only(env: str, value: str) -> CredentialSaveResult:
        monkeypatch.setenv(env, value)
        return CredentialSaveResult("session_only")

    monkeypatch.setattr(services, "save_key", session_only)
    screens: list[ProviderWorkbenchScreen] = []
    host = Host(initial_screen)
    async with host.run_test() as pilot:
        await host.pop_screen()

        async def visit(screen: ProviderWorkbenchScreen) -> ProviderWorkbenchResult:
            screens.append(screen)
            await host.push_screen(screen)
            await pilot.pause()
            screen.state = ManagementState.from_snapshot(screen.snapshot, "two")
            state = screen.state
            if len(screens) == 1:
                screen._save_key("test-value")
                assert owner._session_only_credential_envs == {"REOPEN_TEST_KEY"}
            else:
                assert screen is not screens[0]
                state.toggle("b", True)
                await screen._commit(state.changes(), state, create=False)
                await pilot.pause()
                assert services.writes
                assert not graduation.need_signal
                # Persisting the key ends its runtime-only provenance.
                monkeypatch.setattr(
                    services,
                    "save_key",
                    lambda env, value: CredentialSaveResult("saved"),
                )
                screen._save_key("test-value")
                assert not owner._session_only_credential_envs
            await host.pop_screen()
            return ProviderWorkbenchResult("cancelled")

        owner.push_screen_wait = visit
        await ChartreuxApp._wait_for_provider_management(owner)  # type: ignore[arg-type]
        assert len(screens) == 1
        assert not graduation.need_signal
        await ChartreuxApp._wait_for_provider_management(owner)  # type: ignore[arg-type]
        assert len(screens) == 2
        assert not graduation.need_signal
        assert owner._mount_and_scroll.await_count == 2


@pytest.mark.asyncio
async def test_onboarding_models_saved_never_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _screen, services = setup()
    monkeypatch.setattr(
        "chartreux.setup.onboarding.load_catalog", lambda: services.catalog
    )
    app = OnboardingApp(
        config=build_test_vibe_config(), config_service=services, credentials=services
    )
    app._graduation = GraduationStore(tmp_path / "graduation.toml")

    async def complete_workbench(
        workbench: ProviderWorkbenchScreen,
    ) -> ProviderWorkbenchResult:
        assert workbench.on_models_saved == app._graduation_models_saved
        assert workbench.on_models_saved is not None
        # The shared flow reports the bound-roster transition around the save.
        workbench.on_models_saved(
            frozenset({"a"}), frozenset({"a", "b"}), frozenset({"b"})
        )
        return ProviderWorkbenchResult("completed", changed=True)

    app.push_screen_wait = AsyncMock(side_effect=complete_workbench)
    app.exit = lambda result: None  # type: ignore[method-assign]
    await app._run_workbench()
    assert app.push_screen_wait.await_count == 1
    assert not app._graduation.state.need_signal
    assert not app._graduation.state.eligible(idle=True, headless=False)


def graduation_handler_owner(path: Path) -> SimpleNamespace:
    """A minimal ChartreuxApp stand-in for the graduation save handler."""
    return SimpleNamespace(_graduation=GraduationStore(path))


@pytest.mark.parametrize(
    "roster_before,roster_after",
    [
        (frozenset(), frozenset()),
        (frozenset({"a"}), frozenset({"a", "b"})),
        (frozenset(), frozenset({"a", "b"})),
    ],
)
def test_graduation_models_saved_never_signals(
    roster_before: frozenset[str], roster_after: frozenset[str], tmp_path: Path
) -> None:
    # No roster shape drives a suggestion anymore: the handler records the
    # transition and never signals (the roster-size latch is retired).
    owner = graduation_handler_owner(tmp_path / "graduation.toml")
    ChartreuxApp._graduation_models_saved(
        owner,  # type: ignore[arg-type]
        roster_before,
        roster_after,
        frozenset({"b"}),
    )
    assert not owner._graduation.state.need_signal
    assert not owner._graduation.state.eligible(idle=True, headless=False)


class PresentationApp(App):
    _maybe_show_graduation = ChartreuxApp._maybe_show_graduation
    action_graduation_dismiss = ChartreuxApp.action_graduation_dismiss
    action_graduation_customize = ChartreuxApp.action_graduation_customize

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._graduation = GraduationStore(path)
        self._graduation.state.compacted()
        self._graduation_notice = InlineNotice(id="graduation-notice")
        self._graduation_notice.styles.height = "auto"
        # The retired dispatch_mode posture is gone; the presentation flow
        # reads no config dispatch fields.
        self.config = SimpleNamespace()
        self._active_callback = None
        self._agent_summaries = []
        self.busy = True
        self._show_providers = AsyncMock()

    def _is_busy(self) -> bool:
        return self.busy

    def compose(self) -> ComposeResult:
        yield self._graduation_notice
        yield Input(id="prompt")


@pytest.mark.asyncio
async def test_presentation_idle_once_dismiss_and_focus(tmp_path: Path) -> None:
    app = PresentationApp(tmp_path / "graduation.toml")
    async with app.run_test() as pilot:
        await pilot.pause()
        focused = app.screen.focused
        app._maybe_show_graduation()
        assert not app._graduation_notice.display
        app.busy = False
        app._maybe_show_graduation()
        assert app._graduation_notice.display
        assert "Budgeted implementation attempts failed repeatedly." in str(
            app._graduation_notice.render()
        )
        assert app.screen.focused is focused
        await pilot.pause()
        await pilot.click("#graduation-notice", offset=(66, 0))
        await pilot.pause()
        app._maybe_show_graduation()
        assert not app._graduation_notice.display
        assert GraduationStore(app._graduation.path).state.dismissed


@pytest.mark.asyncio
async def test_customize_presets_after_failure_signal(tmp_path: Path) -> None:
    app = PresentationApp(tmp_path / "graduation.toml")
    app.busy = False
    await app.action_graduation_customize()
    app._show_providers.assert_awaited_once_with(initial_view="presets")
