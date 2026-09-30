"""Web search settings boundary for onboarding before an agent runtime exists."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from chartreux.app_server._web_search_settings import project_web_search_settings
from chartreux.app_server.protocol import SettingsReadResponse
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayerError
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch, PatchOp, RemoveOperationPatch
from chartreux.core.tools.builtins.web_search import (
    SearchProviderDiagnostic,
    WebSearchConfig,
    effective_web_search_config,
)
from chartreux.ui.settings_service import SettingsReloadOutcome, SettingsSaveOutcome

_PREFIX = "tools.web_search."


class OnboardingWebSearchSettings:
    """Revision checked search saves and safe reads using the setup orchestrator."""

    def __init__(self, orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]) -> None:
        self.orchestrator = orchestrator

    async def read(self) -> SettingsReadResponse:
        snapshot = self.orchestrator.copy()
        layers: list[tuple[str, dict[str, Any]]] = []
        user_layer: str | None = None
        user_revision: str | None = None
        user_unavailable = False
        for layer in snapshot.layers:
            if isinstance(layer, UserConfigLayer):
                user_layer = layer.name
            try:
                loaded = await layer.load(force=True)
            except ConfigLayerError:
                if isinstance(layer, UserConfigLayer):
                    user_unavailable = True
                continue
            layers.append((layer.name, loaded.model_dump(mode="json")))
            if isinstance(layer, UserConfigLayer):
                user_revision = layer.fingerprint
        fallback = user_unavailable
        if fallback:
            config = self.orchestrator.config
        else:
            try:
                config = (await snapshot.preview_candidate()).config
            except ConfigLayerError:
                config = self.orchestrator.config
                fallback = True
        return SettingsReadResponse(
            fields=[],
            web_search=project_web_search_settings(
                config,
                layers,
                user_layer=user_layer,
                user_unavailable=user_unavailable,
                fallback=fallback,
            ),
            user_layer=user_layer,
            user_revision=None if user_unavailable else user_revision,
        )

    async def save(  # noqa: PLR0911 - each persistence outcome needs its own status
        self,
        changed_leaves: Mapping[str, JsonValue | None],
        expected_revision: str | None,
    ) -> SettingsSaveOutcome:
        if not changed_leaves:
            return SettingsSaveOutcome("not_saved", "unchanged", error="no_changes")
        before = await self.read()
        if before.user_revision is None or before.user_layer is None:
            return SettingsSaveOutcome(
                "not_saved", "unchanged", before, error="view_only"
            )
        if expected_revision is None or expected_revision != before.user_revision:
            return SettingsSaveOutcome(
                "not_saved", "unchanged", before, error="conflict"
            )
        valid_paths = (
            {field.path for field in before.web_search.fields}
            if before.web_search is not None
            else set()
        )
        operations: list[PatchOp] = []
        for path, value in changed_leaves.items():
            if path not in valid_paths or not path.startswith(_PREFIX):
                raise ValueError(f"Unknown web search setting: {path}")
            pointer = "/" + path.replace(".", "/")
            if value is None:
                operations.append(
                    RemoveOperationPatch(
                        path=pointer, target_layer_name=before.user_layer
                    )
                )
            else:
                try:
                    WebSearchConfig.model_validate({path.removeprefix(_PREFIX): value})
                except ValueError:
                    return SettingsSaveOutcome(
                        "not_saved", "unchanged", before, error="validation"
                    )
                operations.append(
                    AddOperationPatch(
                        path=pointer, value=value, target_layer_name=before.user_layer
                    )
                )

        async def preflight(candidate: Any) -> None:
            if isinstance(
                effective_web_search_config(candidate.config), SearchProviderDiagnostic
            ):
                raise ValueError("Invalid web search settings")

        result = await self.orchestrator.save(
            operations,
            target="user",
            expected_revision=expected_revision,
            reason="onboarding web search settings",
            preflight=preflight,
        )
        if result.persistence == "not_saved":
            return SettingsSaveOutcome(
                "not_saved", result.application, error=result.error or "save_failed"
            )
        try:
            after = await self.read()
        except Exception:
            return SettingsSaveOutcome(
                result.persistence, result.application, error="snapshot_unknown"
            )
        previous = (
            {field.path: field for field in before.web_search.fields}
            if before.web_search
            else {}
        )
        current = (
            {field.path: field for field in after.web_search.fields}
            if after.web_search
            else {}
        )
        shadowed = tuple(
            path
            for path in changed_leaves
            if path in previous
            and path in current
            and current[path].saved_explicit
            and current[path].origin != after.user_layer
            and current[path].effective_value == previous[path].effective_value
        )
        return SettingsSaveOutcome(
            result.persistence, result.application, after, shadowed, result.error
        )

    async def retry_runtime(self) -> SettingsReloadOutcome:
        try:
            await self.orchestrator.reload()
        except Exception as error:
            return SettingsReloadOutcome(False, False, error=str(error))
        return await self.retry_ui()

    async def retry_ui(self) -> SettingsReloadOutcome:
        try:
            return SettingsReloadOutcome(True, True, await self.read())
        except Exception as error:
            return SettingsReloadOutcome(True, True, error=f"snapshot_unknown: {error}")
