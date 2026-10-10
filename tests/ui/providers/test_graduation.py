from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.inline_notice import InlineNotice
from chartreux.core.dispatch.schema import DispatchMode
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
from tests.ui.providers.test_workbench import Host, setup


def eligible(state: GraduationState, **kwargs: object) -> bool:
    return state.eligible(
        mode=kwargs.get("mode", DispatchMode.STANDALONE),  # type: ignore[arg-type]
        idle=kwargs.get("idle", True),  # type: ignore[arg-type]
        headless=kwargs.get("headless", False),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("before", "after", "save", "need", "expected"),
    [
        (frozenset({"a"}), frozenset({"a", "b"}), True, True, True),
        (frozenset({"a"}), frozenset({"a", "b"}), False, True, False),
        (frozenset({"a"}), frozenset({"a", "b"}), True, False, False),
        (frozenset({"a", "b"}), frozenset({"a", "b"}), True, True, False),
        (frozenset({"a"}), frozenset({"a"}), True, True, False),
    ],
)
def test_truth_table(
    before: frozenset[str],
    after: frozenset[str],
    save: bool,
    need: bool,
    expected: bool,
) -> None:
    state = GraduationState()
    if save:
        state.model_saved(before, after, frozenset({"b"}))
    if need:
        state.compacted()
    assert eligible(state) is expected


@pytest.mark.parametrize("need_first", [True, False])
def test_both_orders(need_first: bool) -> None:
    state = GraduationState()
    if need_first:
        state.compacted()
    state.model_saved(frozenset({"a"}), frozenset({"a", "b"}), frozenset({"b"}))
    if not need_first:
        state.compacted()
    assert eligible(state)
    assert not eligible(state, idle=False)
    assert not eligible(state, headless=True)
    assert not eligible(state, mode=DispatchMode.ORCHESTRATED)


@pytest.mark.parametrize("succeeded,linked", [(False, False), (True, True)])
def test_failed_or_linked_save(succeeded: bool, linked: bool) -> None:
    state = GraduationState(need_signal=True)
    state.model_saved(
        frozenset({"a"}),
        frozenset({"a", "b"}),
        frozenset({"b"}),
        succeeded=succeeded,
        linked=linked,
    )
    assert not eligible(state)


@pytest.mark.parametrize(
    ("before", "after", "saved", "expected"),
    [
        (frozenset({"a"}), frozenset({"a", "b"}), frozenset({"b"}), True),
        (frozenset({"a"}), frozenset({"b", "c"}), frozenset({"b"}), True),
        (frozenset({"a", "b"}), frozenset({"a", "b"}), frozenset({"b"}), False),
        (frozenset({"a", "b"}), frozenset({"a", "b", "c"}), frozenset({"c"}), False),
        (frozenset(), frozenset({"a", "b"}), frozenset({"a", "b"}), False),
        (frozenset({"a"}), frozenset({"a"}), frozenset({"a"}), False),
        (frozenset({"a"}), frozenset({"a", "b"}), frozenset(), False),
        (frozenset({"a"}), frozenset({"a", "b"}), frozenset({"c"}), False),
    ],
    ids=[
        "single-to-two",
        "single-to-other-two",
        "already-two",
        "two-to-three",
        "empty-to-two",
        "no-transition",
        "nothing-saved",
        "saved-outside-roster",
    ],
)
def test_roster_transition_gate(
    before: frozenset[str], after: frozenset[str], saved: frozenset[str], expected: bool
) -> None:
    state = GraduationState(need_signal=True)
    state.model_saved(before, after, saved)
    assert state.second_model_saved is expected
    assert eligible(state) is expected


def test_deployments_and_thinking_are_not_models() -> None:
    state = GraduationState(need_signal=True)
    # Slots bound to one canonical identity through different deployments or
    # thinking levels stay a single-model roster.
    state.model_saved(frozenset({"a"}), frozenset({"a"}), frozenset({"a"}))
    assert not eligible(state)
    state.model_saved(frozenset({"a"}), frozenset({"a", "b"}), frozenset())
    assert not eligible(state)


def test_failure_dedup_scope_budget_and_exclusions() -> None:
    state = GraduationState(second_model_saved=True)
    for _ in range(3):
        state.implementation_failed("task", "run1", attempt_budgeted=True)
    state.implementation_failed("other-task", "run2", attempt_budgeted=True)
    for excluded in ("replayed", "cancelled", "transient"):
        state.implementation_failed(
            "task", "excluded", attempt_budgeted=True, **{excluded: True}
        )
    state.implementation_failed("task", "unbudgeted", attempt_budgeted=False)
    assert not eligible(state)
    state.implementation_failed("task", "run2", attempt_budgeted=True)
    assert eligible(state)
    assert len(state.failures["task"]) == 2


def test_replayed_compaction_excluded() -> None:
    state = GraduationState(second_model_saved=True)
    state.compacted(replayed=True)
    assert not eligible(state)


def test_dismissal_and_shown_persist(tmp_path: Path) -> None:
    path = tmp_path / "graduation.toml"
    store = GraduationStore(path)
    assert not path.exists()
    store.state.second_model_saved = True
    assert store.save()
    resumed = GraduationStore(path)
    assert resumed.state.second_model_saved
    resumed.state.shown = True
    resumed.state.dismissed = True
    assert resumed.save()
    restarted = GraduationStore(path)
    restarted.state.compacted()
    assert not eligible(restarted.state)
    assert restarted.state.dismissed


def test_interleaved_stores_preserve_durable_flags(tmp_path: Path) -> None:
    path = tmp_path / "graduation.toml"
    first = GraduationStore(path)
    stale = GraduationStore(path)
    first.state.shown = True
    first.state.dismissed = True
    assert first.save()
    stale.state.second_model_saved = True
    assert stale.save()
    resumed = GraduationStore(path)
    assert resumed.state.shown
    assert resumed.state.dismissed
    assert resumed.state.second_model_saved
    assert stale.state.dismissed


def test_invalid_file_is_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "graduation.toml"
    path.write_text('shown = "invalid"\n')
    store = GraduationStore(path)
    assert not store.save()
    assert path.read_text() == 'shown = "invalid"\n'


def test_persisted_latch_remains_eligible(tmp_path: Path) -> None:
    # A store persisted by an older release keeps its latch: the re-gate never
    # re-evaluates a persisted second_model_saved flag.
    path = tmp_path / "graduation.toml"
    path.write_text("second_model_saved = true\n")
    store = GraduationStore(path)
    store.state.compacted()
    assert eligible(store.state)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("single_model", [False, True])
async def test_existing_model_becomes_usable(single_model: bool) -> None:
    screen, services = setup()
    data = with_slot_roles(
        services.catalog.catalog.model_dump(), single_model=single_model
    )
    data["models"]["b"]["deployments"][0]["disabled"] = True
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "disabled")
    screen.snapshot = services.catalog
    graduation = GraduationState()
    screen.on_models_saved = graduation.model_saved
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
        assert graduation.second_model_saved is not single_model


@pytest.mark.asyncio
async def test_fresh_two_model_roster_never_latches() -> None:
    """A fresh two-model install: key entry and catalog saves never latch."""
    screen, services = setup()
    data = with_slot_roles(services.catalog.catalog.model_dump())
    data["providers"]["two"]["api_key_env_var"] = "MISSING_KEY"
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "two-model")
    screen.snapshot = services.catalog
    graduation = GraduationState()
    screen.on_models_saved = graduation.model_saved
    assert screen._bound_roster_identities(screen.snapshot) == frozenset({"a", "b"})
    async with Host(screen).run_test() as pilot:
        screen.state = ManagementState.from_snapshot(screen.snapshot, "two")
        screen._save_key("test-value")
        await pilot.pause()
        assert services.keys == [("MISSING_KEY", "test-value")]
        assert not graduation.second_model_saved
        # A catalog save against the already two-identity roster does not latch.
        state = screen.state
        assert state is not None
        state.toggle("b", False)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        assert services.writes
        assert not graduation.second_model_saved


@pytest.mark.asyncio
@pytest.mark.parametrize("single_model", [False, True])
@pytest.mark.parametrize("persisted", [False, True])
async def test_credential_fixed_later(
    single_model: bool, persisted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.core.model_catalog.contracts import CredentialSaveResult

    screen, services = setup()
    if not persisted:
        monkeypatch.setattr(
            services,
            "save_key",
            lambda env, value: CredentialSaveResult("session_only"),
        )
    data = with_slot_roles(
        services.catalog.catalog.model_dump(), single_model=single_model
    )
    data["providers"]["two"]["api_key_env_var"] = "MISSING_KEY"
    data["models"]["b"]["deployments"][0]["disabled"] = True
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "missing-key")
    screen.snapshot = services.catalog
    graduation = GraduationState()
    screen.on_models_saved = graduation.model_saved
    async with Host(screen).run_test() as pilot:
        screen.state = ManagementState.from_snapshot(screen.snapshot, "two")
        screen._save_key("test-value")
        await pilot.pause()
        assert services.keys == ([("MISSING_KEY", "test-value")] if persisted else [])
        assert not graduation.second_model_saved
        # Enabling the second model transitions the roster; the saved
        # credential is what makes it usable at that save.
        state = screen.state
        assert state is not None
        state.toggle("b", True)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        assert services.writes
        assert graduation.second_model_saved is (persisted and not single_model)


@pytest.mark.asyncio
async def test_session_only_key_excluded_from_later_catalog_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.core.model_catalog.contracts import CredentialSaveResult

    screen, services = setup()
    data = with_slot_roles(services.catalog.catalog.model_dump())
    data["providers"]["two"]["api_key_env_var"] = "MISSING_KEY"
    data["models"]["b"]["deployments"][0]["disabled"] = True
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "disabled")
    screen.snapshot = services.catalog
    graduation = GraduationState()
    screen.on_models_saved = graduation.model_saved

    def session_only(env: str, value: str) -> CredentialSaveResult:
        monkeypatch.setenv(env, value)
        return CredentialSaveResult("session_only")

    monkeypatch.setattr(services, "save_key", session_only)
    async with Host(screen).run_test() as pilot:
        screen.state = ManagementState.from_snapshot(screen.snapshot, "two")
        screen._save_key("test-value")
        state = screen.state
        assert state is not None
        state.toggle("b", True)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        assert services.writes
        assert not graduation.second_model_saved
        # A later durable credential save qualifies the model at the next
        # roster-transitioning save.
        monkeypatch.setattr(
            services, "save_key", lambda env, value: CredentialSaveResult("saved")
        )
        screen._save_key("test-value")
        state = screen.state
        assert state is not None
        state.toggle("b", False)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        state = screen.state
        assert state is not None
        state.toggle("b", True)
        await screen._commit(state.changes(), state, create=False)
        await pilot.pause()
        assert graduation.second_model_saved


@pytest.mark.asyncio
async def test_session_only_key_excluded_after_workbench_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.core.model_catalog.contracts import CredentialSaveResult

    initial_screen, services = setup()
    data = with_slot_roles(services.catalog.catalog.model_dump())
    data["providers"]["two"]["api_key_env_var"] = "REOPEN_TEST_KEY"
    data["models"]["b"]["deployments"][0]["disabled"] = True
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "disabled")
    graduation = GraduationState()
    owner = SimpleNamespace(
        config=SimpleNamespace(enable_system_trust_store=False),
        _graduation_models_saved=graduation.model_saved,
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
                assert not graduation.second_model_saved
                # Persisting the key ends its runtime-only provenance.
                monkeypatch.setattr(
                    services,
                    "save_key",
                    lambda env, value: CredentialSaveResult("saved"),
                )
                screen._save_key("test-value")
                assert not owner._session_only_credential_envs
                # The durable key qualifies the model at the next
                # roster-transitioning save: re-enabling the second model.
                state = screen.state
                assert state is not None
                state.toggle("b", False)
                await screen._commit(state.changes(), state, create=False)
                await pilot.pause()
                state = screen.state
                assert state is not None
                state.toggle("b", True)
                await screen._commit(state.changes(), state, create=False)
                await pilot.pause()
                assert graduation.second_model_saved
            await host.pop_screen()
            return ProviderWorkbenchResult("cancelled")

        owner.push_screen_wait = visit
        await ChartreuxApp._wait_for_provider_management(owner)  # type: ignore[arg-type]
        assert len(screens) == 1
        assert not graduation.second_model_saved
        await ChartreuxApp._wait_for_provider_management(owner)  # type: ignore[arg-type]
        assert len(screens) == 2
        assert graduation.second_model_saved
        assert owner._mount_and_scroll.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "after", "saved_models", "expected"),
    [
        (frozenset({"a"}), frozenset({"a", "b"}), frozenset({"b"}), True),
        (frozenset({"a", "b"}), frozenset({"a", "b"}), frozenset({"a", "b"}), False),
        (frozenset(), frozenset({"a", "b"}), frozenset({"a", "b"}), False),
    ],
    ids=["single-model-transition", "fresh-two-model-roster", "empty-roster"],
)
async def test_onboarding_roster_transition_latch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    before: frozenset[str],
    after: frozenset[str],
    saved_models: frozenset[str],
    expected: bool,
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
        assert not app._graduation.state.second_model_saved
        assert workbench.on_models_saved == app._graduation_models_saved
        assert workbench.on_models_saved is not None
        # The shared flow reports the bound-roster transition around the save.
        workbench.on_models_saved(before, after, saved_models)
        return ProviderWorkbenchResult("completed", changed=True)

    app.push_screen_wait = AsyncMock(side_effect=complete_workbench)
    app.exit = lambda result: None  # type: ignore[method-assign]
    await app._run_workbench()
    assert app.push_screen_wait.await_count == 1
    assert app._graduation.state.second_model_saved is expected
    assert GraduationStore(app._graduation.path).state.second_model_saved is expected


def graduation_handler_owner(
    bound_roster: list[str] | None, path: Path
) -> SimpleNamespace:
    """A minimal ChartreuxApp stand-in for the graduation save handler."""
    return SimpleNamespace(
        config=SimpleNamespace(bound_roster=bound_roster),
        _graduation=GraduationStore(path),
    )


def test_graduation_empty_live_roster_never_latches(tmp_path: Path) -> None:
    # A session whose bound policy attached with no bound slots must use the
    # live (empty) roster: the snapshot fallback must not turn a live 0->2
    # transition into a latch.
    owner = graduation_handler_owner([], tmp_path / "graduation.toml")
    ChartreuxApp._graduation_models_saved(
        owner,  # type: ignore[arg-type]
        frozenset({"a"}),  # workbench pre-save roster: one identity
        frozenset({"a", "b"}),  # post-save roster: two identities
        frozenset({"b"}),
    )
    assert not owner._graduation.state.second_model_saved


def test_graduation_prefers_live_roster_over_snapshot(tmp_path: Path) -> None:
    # The live single-model roster is authoritative even when the workbench's
    # pre-save snapshot already bound two identities.
    owner = graduation_handler_owner(["a"], tmp_path / "graduation.toml")
    ChartreuxApp._graduation_models_saved(
        owner,  # type: ignore[arg-type]
        frozenset({"a", "b"}),
        frozenset({"a", "b", "c"}),
        frozenset({"c"}),
    )
    assert owner._graduation.state.second_model_saved


def test_graduation_falls_back_without_attached_policy(tmp_path: Path) -> None:
    # No bound policy (pre-session): the workbench's pre-save resolution is
    # the fallback source.
    owner = graduation_handler_owner(None, tmp_path / "graduation.toml")
    ChartreuxApp._graduation_models_saved(
        owner,  # type: ignore[arg-type]
        frozenset({"a"}),
        frozenset({"a", "b"}),
        frozenset({"b"}),
    )
    assert owner._graduation.state.second_model_saved


class PresentationApp(App):
    _maybe_show_graduation = ChartreuxApp._maybe_show_graduation
    action_graduation_dismiss = ChartreuxApp.action_graduation_dismiss
    action_graduation_customize = ChartreuxApp.action_graduation_customize

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._graduation = GraduationStore(path)
        self._graduation.state.second_model_saved = True
        self._graduation.state.compacted()
        self._graduation_notice = InlineNotice(id="graduation-notice")
        self._graduation_notice.styles.height = "auto"
        self.config = SimpleNamespace(dispatch_mode="standalone")
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
        assert app.screen.focused is focused
        assert app.config.dispatch_mode == "standalone"
        await pilot.pause()
        await pilot.click("#graduation-notice", offset=(47, 0))
        await pilot.pause()
        app._maybe_show_graduation()
        assert not app._graduation_notice.display
        assert GraduationStore(app._graduation.path).state.dismissed
        assert app.config.dispatch_mode == "standalone"


@pytest.mark.asyncio
async def test_customize_presets_without_mode_change(tmp_path: Path) -> None:
    app = PresentationApp(tmp_path / "graduation.toml")
    app.busy = False
    await app.action_graduation_customize()
    app._show_providers.assert_awaited_once_with(initial_view="presets")
    assert app.config.dispatch_mode == "standalone"
