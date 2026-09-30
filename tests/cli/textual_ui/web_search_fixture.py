"""Offline web search settings screen fixture for UI tests and captures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from pydantic import JsonValue
from textual.app import App

from chartreux.app_server.protocol import (
    SettingLeafWire,
    SettingsReadResponse,
    WebSearchSettingsWire,
)
from chartreux.cli.textual_ui.screens.web_search import WebSearchScreen
from chartreux.cli.textual_ui.settings_service import (
    SettingsReloadOutcome,
    SettingsSaveOutcome,
    SettingsService,
)
from chartreux.ui.providers.contracts import CredentialSaveResult

VALUES: dict[str, JsonValue] = {
    "permission": "ask",
    "provider": "auto",
    "api_key_env_var": None,
    "base_url": None,
    "timeout": 120,
    "max_results": 5,
    "model": "mistral-vibe-cli-with-tools",
}
DEFAULT_ENVS = {
    "auto": "MISTRAL_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "exa": "EXA_API_KEY",
    "brave": "BRAVE_SEARCH_API_KEY",
    "duckduckgo": None,
}


def make_snapshot(
    values: Mapping[str, JsonValue] | None = None,
    *,
    readiness: Literal["ready", "missing_key", "invalid"] = "missing_key",
    revision: str | None = "revision-1",
    invalid_fields: list[str] | None = None,
) -> SettingsReadResponse:
    effective = {**VALUES, **(values or {})}
    web = WebSearchSettingsWire(
        fields=[
            SettingLeafWire(
                path=f"tools.web_search.{name}",
                effective_value=value,
                origin="default",
                saved_explicit=False,
            )
            for name, value in effective.items()
        ],
        invalid_fields=invalid_fields or [],
        readiness=readiness,
        readiness_message=(
            "MISTRAL_API_KEY is missing" if readiness == "missing_key" else None
        ),
        credential_env_var="MISTRAL_API_KEY",
        default_credential_env_vars=DEFAULT_ENVS,
    )
    return SettingsReadResponse(
        fields=[],
        catalog=[],
        user_layer="user" if revision else None,
        user_revision=revision,
        web_search=web,
    )


class FakeCredentials:
    def __init__(self, keys: Mapping[str, str] | None = None) -> None:
        self.keys = dict(keys or {})
        self.saved: list[tuple[str, str]] = []
        self.save_status: Literal["saved", "session_only", "invalid_env_var"] = "saved"

    def resolve_key(self, env_var: str) -> str | None:
        return self.keys.get(env_var)

    def save_key(self, env_var: str, key: str) -> CredentialSaveResult:
        self.saved.append((env_var, key))
        if self.save_status in {"saved", "session_only"}:
            self.keys[env_var] = key
        return CredentialSaveResult(self.save_status)


class FakeSettingsService:
    def __init__(self, snapshot: SettingsReadResponse | None = None) -> None:
        self.snapshot = snapshot or make_snapshot()
        self.saved: list[tuple[dict[str, JsonValue | None], str | None]] = []
        self.outcome: SettingsSaveOutcome | None = None
        self.read_error: Exception | None = None
        self.retry_error: Exception | None = None
        self.retry_count = 0
        self.ui_retry_count = 0
        self.retry_outcome: SettingsReloadOutcome | None = None
        self.ui_outcome: SettingsReloadOutcome | None = None

    async def read(self) -> SettingsReadResponse:
        if self.read_error:
            raise self.read_error
        return self.snapshot

    async def save(
        self,
        changed_leaves: Mapping[str, JsonValue | None],
        expected_revision: str | None,
    ) -> SettingsSaveOutcome:
        self.saved.append((dict(changed_leaves), expected_revision))
        if self.outcome is not None:
            return self.outcome
        assert self.snapshot.web_search is not None
        new_fields = []
        for field in self.snapshot.web_search.fields:
            if field.path in changed_leaves:
                value = changed_leaves[field.path]
                field = field.model_copy(
                    update={
                        "effective_value": value,
                        "origin": "user",
                        "saved_explicit": True,
                        "saved_value": value,
                    }
                )
            new_fields.append(field)
        web = self.snapshot.web_search.model_copy(update={"fields": new_fields})
        self.snapshot = self.snapshot.model_copy(
            update={"web_search": web, "user_revision": "revision-2"}
        )
        return SettingsSaveOutcome("saved", "applied", self.snapshot)

    async def retry_runtime(self) -> SettingsReloadOutcome:
        self.retry_count += 1
        if self.retry_error:
            return SettingsReloadOutcome(False, False, error=str(self.retry_error))
        return self.retry_outcome or SettingsReloadOutcome(True, True, self.snapshot)

    async def retry_ui(self) -> SettingsReloadOutcome:
        self.ui_retry_count += 1
        return self.ui_outcome or SettingsReloadOutcome(True, True, self.snapshot)


class WebSearchHarness(App[None]):
    def __init__(
        self,
        service: FakeSettingsService | None = None,
        credentials: FakeCredentials | None = None,
        *,
        mode: Literal["standalone", "onboarding"] = "standalone",
    ) -> None:
        super().__init__()
        self.service = service or FakeSettingsService()
        self.credentials = credentials or FakeCredentials()
        self.mode: Literal["standalone", "onboarding"] = mode
        self.result: str | None = None

    def on_mount(self) -> None:
        self.push_screen(
            WebSearchScreen(
                cast(SettingsService, self.service),
                self.service.snapshot,
                credentials=self.credentials,
                mode=self.mode,
            ),
            callback=lambda result: setattr(self, "result", result),
        )
