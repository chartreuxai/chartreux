"""Protocol-only Settings service boundary for the future bottom-panel UI."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from chartreux.app_server.protocol import (
    ConfigWriteOpWire,
    ConfigWriteResponse,
    SettingsReadResponse,
)


class SettingsConfigResource(Protocol):
    async def read_settings(self) -> SettingsReadResponse: ...

    async def reload(self, *, reload_runtime: bool = True) -> object: ...

    async def write(
        self,
        ops: list[ConfigWriteOpWire],
        *,
        reason: str,
        target: Literal["user"],
        expected_revision: str | None,
    ) -> ConfigWriteResponse: ...


@dataclass(frozen=True)
class SettingsSaveOutcome:
    persistence: Literal["not_saved", "saved", "durability_uncertain"]
    application: Literal["unchanged", "applied", "failed"]
    snapshot: SettingsReadResponse | None = None
    shadowed: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class SettingsReloadOutcome:
    runtime_applied: bool
    ui_applied: bool
    snapshot: SettingsReadResponse | None = None
    error: str | None = None


class SettingsService:
    def __init__(
        self,
        config: SettingsConfigResource,
        *,
        apply_ui: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._apply_ui = apply_ui

    async def read(self) -> SettingsReadResponse:
        return await self._config.read_settings()

    async def retry_runtime(self) -> SettingsReloadOutcome:
        """Reload saved configuration without repeating its write."""
        try:
            await self._config.reload(reload_runtime=True)
        except Exception as exc:
            return SettingsReloadOutcome(False, False, error=str(exc))
        return await self.retry_ui()

    async def retry_ui(self) -> SettingsReloadOutcome:
        """Refresh UI and projection after runtime has already applied."""
        ui_applied = True
        error = None
        if self._apply_ui is not None:
            try:
                await self._apply_ui()
            except Exception:
                ui_applied = False
                error = "ui_update_failed"
        try:
            snapshot = await self.read()
        except Exception:
            snapshot = None
            if error is None:
                error = "snapshot_unknown"
        return SettingsReloadOutcome(True, ui_applied, snapshot, error)

    async def save(  # noqa: PLR0911, PLR0912
        self,
        changed_leaves: Mapping[str, JsonValue | None],
        expected_revision: str | None,
    ) -> SettingsSaveOutcome:
        """None removes the user override; all other values are explicit sets."""
        if expected_revision is None:
            return SettingsSaveOutcome("not_saved", "unchanged", error="view_only")
        if not changed_leaves:
            return SettingsSaveOutcome("not_saved", "unchanged", error="no_changes")
        before = await self.read()
        if before.user_revision is None or before.user_layer is None:
            return SettingsSaveOutcome(
                "not_saved", "unchanged", before, error="view_only"
            )
        if before.user_revision != expected_revision:
            return SettingsSaveOutcome(
                "not_saved", "unchanged", before, error="conflict"
            )
        ops: list[ConfigWriteOpWire] = []
        editable = {
            descriptor.path: descriptor
            for descriptor in before.catalog
            if descriptor.kind not in {"link", "deferred"}
            and descriptor.control != "toggle_inventory"
        }
        web_search_paths = (
            {field.path for field in before.web_search.fields}
            if before.web_search is not None
            else set()
        )
        for path, value in changed_leaves.items():
            descriptor = editable.get(path)
            if descriptor is None:
                backing = before.backing_settings.get(path)
                if path in web_search_paths:
                    pass  # The app server validates the complete merged search config.
                elif backing is None or not path.startswith(("enabled_", "disabled_")):
                    raise ValueError(f"Unknown settings leaf: {path}")
                elif value is not None:
                    backing.validate_value(value)
            elif value is not None:
                descriptor.validate_value(value)
            ops.append(
                ConfigWriteOpWire(
                    op="remove" if value is None else "set",
                    path="/" + "/".join(path.split(".")),
                    value=value,
                    target_layer=before.user_layer,
                )
            )
        response = await self._config.write(
            ops,
            reason="settings UI save",
            target="user",
            expected_revision=expected_revision,
        )
        # Never propagate response.fields or response.saved_values: string entries
        # in the write projection are redacted. Read authoritative leaves instead.
        if response.persistence == "not_saved":
            error = (
                "conflict"
                if any("conflict" in failure.lower() for failure in response.failures)
                else response.failures[0]
                if response.failures
                else "save_failed"
            )
            return SettingsSaveOutcome("not_saved", response.application, error=error)
        try:
            after = await self.read()
        except Exception:
            # The write has already committed. Do not turn a failed projection into
            # a failed save or expose a stale revision for a second write.
            if response.application == "applied" and self._apply_ui is not None:
                try:
                    await self._apply_ui()
                except Exception:
                    return SettingsSaveOutcome(
                        response.persistence,
                        "failed",
                        error="ui_update_failed_snapshot_unknown",
                    )
            return SettingsSaveOutcome(
                response.persistence, response.application, error="snapshot_unknown"
            )
        previous = {field.path: field for field in before.fields}
        current = {field.path: field for field in after.fields}
        if before.web_search is not None:
            previous.update({field.path: field for field in before.web_search.fields})
        if after.web_search is not None:
            current.update({field.path: field for field in after.web_search.fields})
        shadowed = tuple(
            path
            for path in changed_leaves
            if path in current
            and path in previous
            and current[path].saved_explicit
            and current[path].origin != after.user_layer
            and current[path].effective_value == previous[path].effective_value
        )
        if response.application == "applied" and self._apply_ui is not None:
            try:
                await self._apply_ui()
            except Exception:
                return SettingsSaveOutcome(
                    response.persistence,
                    "failed",
                    after,
                    shadowed,
                    error="ui_update_failed",
                )
        return SettingsSaveOutcome(
            response.persistence, response.application, after, shadowed
        )
