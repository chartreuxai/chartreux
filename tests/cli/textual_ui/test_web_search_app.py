from __future__ import annotations

from typing import cast

import pytest
from textual.widgets import Button, Input, OptionList

from chartreux.cli.commands import CommandRegistry
from chartreux.cli.textual_ui.screens.settings import SettingsOptionList, SettingsScreen
from chartreux.ui.settings_service import (
    SettingsConfigResource,
    SettingsReloadOutcome,
    SettingsSaveOutcome,
    SettingsService,
)
from chartreux.ui.web_search import WebSearchScreen
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.cli.textual_ui.web_search_fixture import (
    FakeCredentials,
    FakeSettingsService,
    WebSearchHarness,
    make_snapshot,
)


def _screen(app: WebSearchHarness) -> WebSearchScreen:
    return cast(WebSearchScreen, app.screen)


def _text(screen: WebSearchScreen, widget_id: str) -> str:
    return str(screen.query_one(widget_id, NoMarkupStatic).content)


def _highlight(providers: OptionList, provider: str) -> None:
    providers.highlighted = next(
        index for index, option in enumerate(providers.options) if option.id == provider
    )


def test_web_search_command_is_registered() -> None:
    registry = CommandRegistry()
    assert registry.parse_command("/web-search") is not None
    assert registry.commands["web-search"].handler == "_show_web_search"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_provider", ["auto", "mistral"])
async def test_standalone_mistral_alias_preserves_raw_provider_and_overrides(
    raw_provider: str,
) -> None:
    service = FakeSettingsService(
        make_snapshot({
            "provider": raw_provider,
            "api_key_env_var": "CUSTOM_MISTRAL_KEY",
            "base_url": "https://mistral.example.invalid",
        })
    )
    app = WebSearchHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        assert [option.id for option in providers.options] == [
            "mistral",
            "exa",
            "brave",
            "duckduckgo",
        ]
        assert "(*)" in str(providers.get_option("mistral").prompt)
        _highlight(providers, "mistral")
        await pilot.press("space")
        assert screen._draft["provider"] == raw_provider
        assert screen._draft["api_key_env_var"] == "CUSTOM_MISTRAL_KEY"
        assert screen._draft["base_url"] == "https://mistral.example.invalid"
        assert not screen._dirty_settings()
        assert screen._forced_reset == set()
        assert service.saved == []
        screen.query_one("#websearch-input-timeout", Input).value = "42"
        await pilot.pause()
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert service.saved[0][0] == {"tools.web_search.timeout": 42}


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_provider", ["auto", "mistral", "unknown"])
async def test_onboarding_requires_explicit_fallback_selection(
    raw_provider: str,
) -> None:
    service = FakeSettingsService(make_snapshot({"provider": raw_provider}))
    app = WebSearchHarness(service, mode="onboarding")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        assert [option.id for option in providers.options] == [
            "exa",
            "brave",
            "duckduckgo",
        ]
        assert all("( )" in str(option.prompt) for option in providers.options)
        assert screen.query_one("#websearch-save", Button).disabled
        providers.focus()
        await pilot.press("enter")
        assert screen._draft["provider"] == raw_provider
        assert not screen._dirty_settings()
        assert service.saved == []
        _highlight(providers, "exa")
        await pilot.press("space")
        assert screen._draft["provider"] == "exa"
        assert screen._forced_reset == {"api_key_env_var", "base_url"}
        assert not screen.query_one("#websearch-save", Button).disabled


@pytest.mark.asyncio
async def test_onboarding_rechecks_selected_fallback_after_fresh_read() -> None:
    service = FakeSettingsService(
        make_snapshot({"provider": "duckduckgo"}, readiness="ready")
    )
    app = WebSearchHarness(service, mode="onboarding")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        service.snapshot = make_snapshot({"provider": "auto"}, readiness="ready")
        screen = _screen(app)
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, WebSearchScreen)
        assert screen.query_one("#websearch-save", Button).disabled
        assert "Saved provider changed" in _text(screen, "#websearch-message")
        assert service.saved == []


@pytest.mark.asyncio
async def test_onboarding_missing_key_requires_explicit_skip() -> None:
    service = FakeSettingsService(make_snapshot(readiness="missing_key"))
    app = WebSearchHarness(service, mode="onboarding")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        assert [option.id for option in providers.options] == [
            "exa",
            "brave",
            "duckduckgo",
        ]
        assert all("( )" in str(option.prompt) for option in providers.options)
        assert "Choose Exa, Brave or DuckDuckGo" in _text(
            screen, "#websearch-readiness"
        )
        assert str(screen.query_one("#websearch-save", Button).label) == "Finish setup"
        assert screen.query_one("#websearch-save", Button).disabled
        assert not screen.query_one("#websearch-credential").display
        screen.query_one("#websearch-skip", Button).press()
        await pilot.pause()
        assert app.result == "skip"
        assert service.saved == []


@pytest.mark.asyncio
async def test_onboarding_keyless_provider_finishes_after_fresh_read() -> None:
    service = FakeSettingsService(
        make_snapshot({"provider": "duckduckgo"}, readiness="ready")
    )
    app = WebSearchHarness(service, mode="onboarding")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        _screen(app).query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert app.result == "finish"


@pytest.mark.asyncio
async def test_onboarding_finish_rechecks_stale_readiness_and_pending_key() -> None:
    service = FakeSettingsService(make_snapshot({"provider": "exa"}, readiness="ready"))
    app = WebSearchHarness(service, mode="onboarding")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        screen.query_one("#websearch-key", Input).value = "replacement-secret"
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, WebSearchScreen)
        assert "Save or clear" in _text(screen, "#websearch-message")
        screen.query_one("#websearch-key", Input).value = ""
        service.snapshot = make_snapshot({"provider": "exa"}, readiness="missing_key")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, WebSearchScreen)
        assert "API key" in _text(screen, "#websearch-message")


@pytest.mark.asyncio
async def test_onboarding_finish_rejects_fallback_projection() -> None:
    service = FakeSettingsService(make_snapshot({"provider": "exa"}, readiness="ready"))
    app = WebSearchHarness(service, mode="onboarding")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        fallback = make_snapshot({"provider": "exa"}, readiness="ready")
        assert fallback.web_search is not None
        fallback = fallback.model_copy(
            update={
                "web_search": fallback.web_search.model_copy(
                    update={
                        "fields": [
                            field.model_copy(update={"origin": "live config"})
                            for field in fallback.web_search.fields
                        ]
                    }
                )
            }
        )
        service.snapshot = fallback
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, WebSearchScreen)
        assert screen._needs_refresh
        assert "could not be verified" in _text(screen, "#websearch-message")


@pytest.mark.asyncio
async def test_onboarding_dirty_skip_confirms_and_saved_key_survives() -> None:
    service = FakeSettingsService()
    credentials = FakeCredentials()
    app = WebSearchHarness(service, credentials, mode="onboarding")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        providers.focus()
        await pilot.press("space")
        screen.query_one("#websearch-key", Input).value = "saved-secret"
        screen.query_one("#websearch-save-key", Button).press()
        screen.query_one("#websearch-input-timeout", Input).value = "42"
        screen.query_one("#websearch-skip", Button).press()
        await pilot.pause()
        assert screen.query_one("#websearch-confirmation").display
        assert app.result is None
        screen.query_one("#websearch-discard", Button).press()
        await pilot.pause()
        assert app.result == "skip"
        assert credentials.saved == [("EXA_API_KEY", "saved-secret")]
        assert service.saved == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (80, 48)])
async def test_form_has_separate_focus_and_saved_provider_and_masks_key(
    size: tuple[int, int],
) -> None:
    app = WebSearchHarness()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        assert providers.has_focus
        assert screen.query_one("#websearch-content").region.size == screen.size
        assert "Effective: Mistral" in _text(screen, "#websearch-current-provider")
        assert "Configured" not in _text(screen, "#websearch-readiness")
        key = screen.query_one("#websearch-key", Input)
        assert key.password
        key.value = "secret-should-not-render"
        await pilot.pause()
        assert "secret-should-not-render" not in str(app.export_screenshot())


@pytest.mark.asyncio
async def test_provider_switch_resets_custom_env_and_endpoint_as_explicit_empty_sets() -> (
    None
):
    snapshot = make_snapshot({
        "provider": "mistral",
        "api_key_env_var": "CUSTOM_KEY",
        "base_url": "https://custom.example/v1",
    })
    service = FakeSettingsService(snapshot)
    credentials = FakeCredentials({"EXA_API_KEY": "existing"})
    app = WebSearchHarness(service, credentials)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        await pilot.pause()
        assert screen._draft["provider"] == "exa"
        assert screen._draft["api_key_env_var"] == ""
        assert screen._draft["base_url"] == ""
        assert "EXA_API_KEY" in _text(screen, "#websearch-selected-env")
        assert "resets custom" in _text(screen, "#websearch-provider-note")
        assert "Mistral" in _text(screen, "#websearch-current-provider")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert len(service.saved) == 1
        changes, revision = service.saved[0]
        assert revision == "revision-1"
        assert changes == {
            "tools.web_search.provider": "exa",
            "tools.web_search.api_key_env_var": "",
            "tools.web_search.base_url": "",
        }
        assert isinstance(app.screen, WebSearchScreen)
        assert "saved to user config" in _text(screen, "#websearch-message")


@pytest.mark.asyncio
async def test_enter_accepts_selected_provider_not_highlighted_row() -> None:
    app = WebSearchHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        assert screen._draft["provider"] == "exa"
        _highlight(providers, "brave")
        await pilot.press("enter")
        assert screen._draft["provider"] == "exa"
        assert screen.query_one("#websearch-key", Input).has_focus


@pytest.mark.asyncio
async def test_api_key_save_is_separate_and_survives_discard() -> None:
    service = FakeSettingsService()
    credentials = FakeCredentials()
    app = WebSearchHarness(service, credentials)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        screen.query_one("#websearch-key", Input).value = "test-key"
        screen.query_one("#websearch-save-key", Button).press()
        await pilot.pause()
        assert credentials.saved == [("MISTRAL_API_KEY", "test-key")]
        assert service.saved == []
        assert screen.query_one("#websearch-key", Input).value == ""
        assert "Credential available" in _text(screen, "#websearch-readiness")
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        providers.focus()
        await pilot.press("space")
        await pilot.press("escape")
        await pilot.pause()
        assert screen.query_one("#websearch-confirmation").display
        assert "remain saved" in _text(screen, "#websearch-confirmation-text")
        screen.query_one("#websearch-discard", Button).press()
        await pilot.pause()
        assert credentials.keys["MISTRAL_API_KEY"] == "test-key"
        assert service.saved == []


@pytest.mark.asyncio
async def test_session_only_key_result_names_scope() -> None:
    credentials = FakeCredentials()
    credentials.save_status = "session_only"
    app = WebSearchHarness(credentials=credentials)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        screen.query_one("#websearch-key", Input).value = "session-key"
        screen.query_one("#websearch-save-key", Button).press()
        await pilot.pause()
        assert "this session" in _text(screen, "#websearch-key-result")
        assert screen.query_one("#websearch-key", Input).value == ""


@pytest.mark.asyncio
async def test_rejected_key_save_keeps_masked_input_for_retry() -> None:
    credentials = FakeCredentials()
    credentials.save_status = "invalid_env_var"
    app = WebSearchHarness(credentials=credentials)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        key_input = screen.query_one("#websearch-key", Input)
        key_input.value = "retry-secret"
        screen.query_one("#websearch-save-key", Button).press()
        await pilot.pause()
        assert key_input.password and key_input.value == "retry-secret"
        assert "Could not save API key" in _text(screen, "#websearch-key-result")
        assert "retry-secret" not in str(app.export_screenshot())


@pytest.mark.asyncio
async def test_invalid_saved_provider_can_be_repaired() -> None:
    app = WebSearchHarness(
        FakeSettingsService(make_snapshot({"provider": "invalid"}, readiness="invalid"))
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        assert "Invalid (invalid)" in _text(screen, "#websearch-current-provider")
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "brave")
        await pilot.press("space")
        assert screen._draft["provider"] == "brave"


@pytest.mark.asyncio
async def test_sanitized_invalid_fields_require_explicit_repair() -> None:
    service = FakeSettingsService(
        make_snapshot(
            {"base_url": "[redacted]", "permission": "[invalid]"},
            readiness="invalid",
            invalid_fields=["tools.web_search.base_url", "tools.web_search.permission"],
        )
    )
    app = WebSearchHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        assert screen._advanced
        assert screen.query_one("#websearch-input-base_url", Input).value == ""
        assert "base_url, permission" in _text(screen, "#websearch-provider-note")
        assert "/open-config-file" in _text(screen, "#websearch-provider-note")
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert "Repair invalid saved fields" in _text(screen, "#websearch-message")
        assert service.saved == []


@pytest.mark.asyncio
async def test_conflict_locks_editing_until_refresh_and_keeps_draft() -> None:
    service = FakeSettingsService()
    service.outcome = SettingsSaveOutcome("not_saved", "unchanged", error="conflict")
    app = WebSearchHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert screen._needs_refresh
        assert screen._draft["provider"] == "exa"
        assert providers.disabled
        assert "Not saved: conflict" in _text(screen, "#websearch-message")
        assert len(service.saved) == 1
        screen.query_one("#websearch-refresh", Button).press()
        await pilot.pause()
        assert not screen._needs_refresh
        assert screen._draft["provider"] == "auto"
        assert "draft discarded" in _text(screen, "#websearch-message")
        assert len(service.saved) == 1


@pytest.mark.asyncio
async def test_saved_read_failure_never_rewrites_stale_revision() -> None:
    service = FakeSettingsService()
    service.outcome = SettingsSaveOutcome(
        "saved", "failed", snapshot=None, error="snapshot_unknown"
    )
    app = WebSearchHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert screen._needs_refresh and screen._runtime_failed
        assert "Saved to user config" in _text(screen, "#websearch-message")
        assert screen.query_one("#websearch-save", Button).disabled
        assert len(service.saved) == 1
        screen.query_one("#websearch-refresh", Button).press()
        await pilot.pause()
        assert screen.query_one("#websearch-retry", Button).display
        screen.query_one("#websearch-retry", Button).press()
        await pilot.pause()
        assert service.retry_count == 1
        assert not screen._runtime_failed
        assert len(service.saved) == 1


@pytest.mark.asyncio
async def test_runtime_retry_waits_for_new_draft_and_never_rewrites() -> None:
    service = FakeSettingsService()
    service.outcome = SettingsSaveOutcome("saved", "unchanged", service.snapshot)
    app = WebSearchHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        assert screen._runtime_failed
        assert "runtime unchanged" in _text(screen, "#websearch-message")
        assert screen.query_one("#websearch-retry", Button).display
        _highlight(providers, "brave")
        providers.focus()
        await pilot.press("space")
        assert screen._dirty_settings()
        assert screen.query_one("#websearch-retry", Button).disabled
        await screen._retry_runtime()
        assert service.retry_count == 0
        assert screen._draft["provider"] == "brave"
        assert len(service.saved) == 1


@pytest.mark.asyncio
async def test_reload_success_then_read_failure_is_not_reported_as_reload_failure() -> (
    None
):
    class Resource:
        def __init__(self) -> None:
            self.reload_count = 0

        async def reload(self, *, reload_runtime: bool = True) -> None:
            assert reload_runtime
            self.reload_count += 1

        async def read_settings(self) -> None:
            raise OSError("projection unavailable")

    resource = Resource()
    service = SettingsService(cast(SettingsConfigResource, resource))
    outcome = await service.retry_runtime()
    assert outcome == SettingsReloadOutcome(
        True, True, snapshot=None, error="snapshot_unknown"
    )
    assert resource.reload_count == 1


@pytest.mark.asyncio
async def test_reload_success_then_ui_failure_preserves_applied_runtime() -> None:
    class Resource:
        def __init__(self) -> None:
            self.reload_count = 0

        async def reload(self, *, reload_runtime: bool = True) -> None:
            assert reload_runtime
            self.reload_count += 1

        async def read_settings(self):
            return make_snapshot()

    async def failing_ui() -> None:
        raise RuntimeError("UI theme refresh failed")

    resource = Resource()
    service = SettingsService(
        cast(SettingsConfigResource, resource), apply_ui=failing_ui
    )
    outcome = await service.retry_runtime()
    assert outcome.runtime_applied
    assert not outcome.ui_applied
    assert outcome.snapshot is not None
    assert outcome.error == "ui_update_failed"
    assert resource.reload_count == 1


@pytest.mark.asyncio
async def test_runtime_reload_read_failure_requires_refresh_without_rewrite() -> None:
    service = FakeSettingsService()
    service.outcome = SettingsSaveOutcome("saved", "failed", service.snapshot)
    service.retry_outcome = SettingsReloadOutcome(
        True, True, snapshot=None, error="snapshot_unknown"
    )
    app = WebSearchHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        screen.query_one("#websearch-retry", Button).press()
        await pilot.pause()
        assert service.retry_count == 1
        assert not screen._runtime_failed
        assert screen._needs_refresh
        assert "applied in runtime; settings read failed" in _text(
            screen, "#websearch-message"
        )
        assert screen.query_one("#websearch-refresh", Button).display
        assert not screen.query_one("#websearch-retry", Button).display
        assert len(service.saved) == 1


@pytest.mark.asyncio
async def test_runtime_reload_ui_failure_offers_ui_retry_without_rewrite() -> None:
    service = FakeSettingsService()
    service.outcome = SettingsSaveOutcome("saved", "failed", service.snapshot)
    service.retry_outcome = SettingsReloadOutcome(
        True, False, service.snapshot, "ui_update_failed"
    )
    app = WebSearchHarness(service)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = _screen(app)
        providers = screen.query_one("#websearch-providers", OptionList)
        _highlight(providers, "exa")
        await pilot.press("space")
        screen.query_one("#websearch-save", Button).press()
        await pilot.pause()
        screen.query_one("#websearch-retry", Button).press()
        await pilot.pause()
        assert service.retry_count == 1
        assert not screen._runtime_failed
        assert screen._ui_refresh_failed
        retry = screen.query_one("#websearch-retry", Button)
        assert retry.display and str(retry.label) == "Retry UI refresh"
        retry.press()
        await pilot.pause()
        assert service.retry_count == 1
        assert service.ui_retry_count == 1
        assert not screen._ui_refresh_failed
        assert len(service.saved) == 1


@pytest.mark.asyncio
async def test_settings_link_opens_web_search_and_restores_selected_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.conftest import build_test_chartreux_app
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator) -> SimpleNamespace:
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        assert await app._handle_command("/settings")
        await pilot.pause()
        await pilot.press(*"tools/web_search", "enter")
        for _ in range(30):
            if isinstance(app.screen, WebSearchScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, WebSearchScreen)
        await pilot.pause()
        assert not app.screen._dirty_settings(), (
            app.screen._draft,
            app.screen._touched,
        )
        await pilot.press("escape")
        for _ in range(30):
            if isinstance(app.screen, SettingsScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        options = app.screen.query_one(SettingsOptionList)
        assert options._query == "tools/web_search"
        assert options.highlighted_option is not None
        assert options.highlighted_option.id == "tools/web_search"
        assert options.has_focus


@pytest.mark.asyncio
async def test_direct_web_search_route_returns_to_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.conftest import build_test_chartreux_app
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator) -> SimpleNamespace:
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        assert await app._handle_command("/web-search")
        for _ in range(30):
            if isinstance(app.screen, WebSearchScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, WebSearchScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not any(
            isinstance(screen, WebSearchScreen) for screen in app.screen_stack
        )
        assert not any(
            isinstance(screen, SettingsScreen) for screen in app.screen_stack
        )
