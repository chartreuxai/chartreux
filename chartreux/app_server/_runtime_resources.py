from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import logging
from typing import Literal

from chartreux.app_server._model import validate_wire
from chartreux.app_server.client_state import ClientSessionState
from chartreux.app_server.config import (
    THINKING_LEVELS,
    ConfigView,
    ProxySettingsView,
    ThinkingLevel,
)
from chartreux.app_server.connection import AppServerResourceConnection
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    ConfigIssue,
    DebugLogPage,
    IdentityView,
    MCPState,
    SessionLogSummary,
    SkillSummary,
    ToolSummary,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ConfigFieldsReadParams,
    ConfigFieldsReadResponse,
    ConfigMutationResponse,
    ConfigProxyReadParams,
    ConfigProxyReadResponse,
    ConfigProxyWriteParams,
    ConfigReloadParams,
    ConfigSchemaReadParams,
    ConfigSchemaReadResponse,
    ConfigWriteOpWire,
    ConfigWriteParams,
    ConfigWriteResponse,
    DiagnosticsLogsReadParams,
    DiagnosticsLogsReadResponse,
    EmptyResponse,
    IdentityReadParams,
    IdentityReadResponse,
    Notification,
    PolicyReadParams,
    PolicyReadResponse,
    PolicyReplaceParams,
    PolicyReplaceResponse,
    PolicyToolReplacement,
    ProtocolError,
    ProtocolErrorCode,
    RootsReadParams,
    RootsReadResponse,
    RootsReplaceParams,
    RootsReplaceResponse,
    RuntimeReadParams,
    RuntimeReadResponse,
    RuntimeSnapshot,
    RuntimeUpdatedParams,
    SessionReadyWaitParams,
    SessionReadyWaitResponse,
)

logger = logging.getLogger(__name__)


def _escape_json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


class ConfigResource:
    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self._connection = connection
        self._state = state
        self._subscribers: list[Callable[[ConfigView], None]] = []

    @property
    def current(self) -> ConfigView:
        return self._state.config

    def subscribe(self, callback: Callable[[ConfigView], None]) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def publish_change(self, previous: ConfigView) -> None:
        if previous == self.current:
            return
        for callback in list(self._subscribers):
            try:
                callback(self.current)
            except (Exception, asyncio.CancelledError):
                # State is already accepted. Observers cannot hide a successful
                # response or prevent delivery to the remaining subscribers.
                logging.getLogger(__name__).warning("Config subscriber failed")

    def _apply_runtime(self, snapshot: RuntimeSnapshot) -> None:
        previous = self.current
        self._state.apply_runtime(snapshot)
        self.publish_change(previous)

    async def read_schema(self) -> ConfigSchemaReadResponse:
        client = await self._connection.connect()
        return validate_wire(
            ConfigSchemaReadResponse,
            await client.request("config/schema", ConfigSchemaReadParams()),
        )

    async def read_fields(self) -> ConfigFieldsReadResponse:
        client = await self._connection.connect()
        return validate_wire(
            ConfigFieldsReadResponse,
            await client.request(
                "config/fields/read",
                ConfigFieldsReadParams(session_id=self._state.session_id),
            ),
        )

    async def write(
        self,
        ops: list[ConfigWriteOpWire],
        *,
        reason: str,
        reload_runtime: bool = False,
        target: Literal["session", "user", "project"] = "session",
        expected_revision: str | None = None,
    ) -> ConfigWriteResponse:
        client = await self._connection.connect()
        response = validate_wire(
            ConfigWriteResponse,
            await client.request(
                "config/write",
                ConfigWriteParams(
                    session_id=self._state.session_id,
                    ops=ops,
                    reason=reason,
                    reload_runtime=reload_runtime,
                    target=target,
                    expected_revision=expected_revision,
                ),
            ),
        )
        if response.application == "applied" or (
            not response.rejected and not response.failures
        ):
            self._apply_runtime(response.runtime)
        return response

    async def update(
        self,
        changes: Mapping[str, object],
        *,
        target_layer: str | None = None,
        reload_runtime: bool = False,
    ) -> None:
        ops = [
            ConfigWriteOpWire.model_validate({
                "op": "set",
                "path": f"/{key}",
                "value": value,
                "target_layer": target_layer,
            })
            for key, value in changes.items()
        ]
        response = await self.write(
            ops, reason="app-server config update", reload_runtime=reload_runtime
        )
        if response.rejected:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INVALID_PARAMS,
                    message="Invalid configuration edit",
                )
            )
        if response.failures:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INTERNAL_ERROR,
                    message="; ".join(response.failures),
                )
            )

    async def set_thinking(self, level: ThinkingLevel) -> None:
        alias = self.current.active_model.alias
        if level not in THINKING_LEVELS or not any(
            model.alias == alias for model in self.current.models
        ):
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INVALID_PARAMS,
                    message="Invalid thinking level or model alias",
                )
            )
        response = await self.write(
            [
                ConfigWriteOpWire(
                    op="set",
                    path=f"/thinking_overrides/{_escape_json_pointer_token(alias)}",
                    value=level,
                    target_layer="overrides",
                )
            ],
            reason="app-server thinking update",
        )
        if response.rejected:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INVALID_PARAMS,
                    message="Invalid configuration edit",
                )
            )
        if response.failures:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INTERNAL_ERROR,
                    message="; ".join(response.failures),
                )
            )

    async def read_roots(self) -> RootsReadResponse:
        client = await self._connection.connect()
        return validate_wire(
            RootsReadResponse,
            await client.request(
                "policy/roots/read", RootsReadParams(session_id=self._state.session_id)
            ),
        )

    async def replace_roots(
        self, *, roots: list[str], expected_revision: str, user_initiated: bool
    ) -> RootsReplaceResponse:
        if user_initiated is not True:
            raise ValueError("Root replacement requires an explicit user action")
        client = await self._connection.connect()
        response = validate_wire(
            RootsReplaceResponse,
            await client.request(
                "policy/roots/replace",
                RootsReplaceParams(
                    session_id=self._state.session_id,
                    roots=roots,
                    expected_revision=expected_revision,
                    user_initiated=True,
                ),
            ),
        )
        self._apply_runtime(response.runtime)
        return response

    async def read_policy(self) -> PolicyReadResponse:
        client = await self._connection.connect()
        return validate_wire(
            PolicyReadResponse,
            await client.request(
                "config/policy/read",
                PolicyReadParams(session_id=self._state.session_id),
            ),
        )

    async def replace_policy(
        self,
        *,
        source: str,
        expected_revision: str,
        tools: dict[str, PolicyToolReplacement],
        user_initiated: bool,
    ) -> PolicyReplaceResponse:
        """Explicit user action only; replaces one source for this session tree."""
        if user_initiated is not True:
            raise ValueError("Policy replacement requires an explicit user action")
        client = await self._connection.connect()
        response = validate_wire(
            PolicyReplaceResponse,
            await client.request(
                "config/policy/replace",
                PolicyReplaceParams(
                    session_id=self._state.session_id,
                    source=source,
                    expected_revision=expected_revision,
                    tools=tools,
                    scope="session",
                    user_initiated=True,
                ),
            ),
        )
        self._apply_runtime(response.runtime)
        return response

    async def reload(self, *, reload_runtime: bool = True) -> ConfigMutationResponse:
        client = await self._connection.connect()
        response = validate_wire(
            ConfigMutationResponse,
            await client.request(
                "config/reload",
                ConfigReloadParams(
                    session_id=self._state.session_id, reload_runtime=reload_runtime
                ),
            ),
        )
        self._apply_runtime(response.runtime)
        return response

    async def read_proxy(self) -> ProxySettingsView:
        client = await self._connection.connect()
        response = validate_wire(
            ConfigProxyReadResponse,
            await client.request(
                "config/proxy/read",
                ConfigProxyReadParams(session_id=self._state.session_id),
            ),
        )
        return response.settings

    async def update_proxy(self, changes: Mapping[str, str | None]) -> None:
        client = await self._connection.connect()
        validate_wire(
            EmptyResponse,
            await client.request(
                "config/proxy/write",
                ConfigProxyWriteParams(
                    session_id=self._state.session_id, changes=dict(changes)
                ),
            ),
        )


class IdentityResource:
    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self._connection = connection
        self._state = state
        self._current: IdentityView | None = None

    @property
    def current(self) -> IdentityView | None:
        return self._current

    async def read(self) -> IdentityView | None:
        self._current = None
        client = await self._connection.connect()
        response = validate_wire(
            IdentityReadResponse,
            await client.request(
                "identity/read", IdentityReadParams(session_id=self._state.session_id)
            ),
        )
        self._current = response.identity
        return response.identity


class RuntimeResource:
    _NOTIFICATION_METHODS = frozenset({"runtime/updated"})

    @classmethod
    def notification_methods(cls) -> frozenset[str]:
        return cls._NOTIFICATION_METHODS

    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self._connection = connection
        self._state = state
        self._session_last_init_duration_ms: int | None = None

    @property
    def session_init_duration_ms(self) -> int | None:
        return self._session_last_init_duration_ms

    @property
    def skills(self) -> list[SkillSummary]:
        return self._state.skills

    @property
    def tools(self) -> list[ToolSummary]:
        return self._state.tools

    @property
    def stats(self) -> AgentStatsSnapshot:
        return self._state.stats

    @property
    def context_window(self) -> int:
        return self._state.context_window

    @property
    def issues(self) -> list[ConfigIssue]:
        return self._state.issues

    @property
    def hooks_count(self) -> int:
        return self._state.hooks_count

    @property
    def mcp(self) -> MCPState:
        return self._state.mcp

    @property
    def session_log(self) -> SessionLogSummary:
        return self._state.session_log

    @property
    def ready(self) -> bool:
        return self._state.ready

    @property
    def custom_skills_count(self) -> int:
        return self._state.custom_skills_count

    def get_skill(self, name: str) -> SkillSummary | None:
        return self._state.get_skill(name)

    def has_tool(self, name: str) -> bool:
        return self._state.has_tool(name)

    async def read_logs(self, *, limit: int = 100, offset: int = 0) -> DebugLogPage:
        client = await self._connection.connect()
        response = validate_wire(
            DiagnosticsLogsReadResponse,
            await client.request(
                "diagnostics/logs/read",
                DiagnosticsLogsReadParams(
                    session_id=self._state.session_id, limit=limit, offset=offset
                ),
            ),
        )
        return response.logs

    async def wait_until_ready(self) -> None:
        client = await self._connection.connect()
        response = validate_wire(
            SessionReadyWaitResponse,
            await client.request(
                "session/ready/wait",
                SessionReadyWaitParams(session_id=self._state.session_id),
            ),
        )
        self._state.ready = True
        self._session_last_init_duration_ms = response.init_duration_ms
        await self.refresh()

    async def refresh(self) -> None:
        client = await self._connection.connect()
        response = validate_wire(
            RuntimeReadResponse,
            await client.request(
                "runtime/read", RuntimeReadParams(session_id=self._state.session_id)
            ),
        )
        self._state.apply_runtime_read(response)

    async def consume_notification(self, notification: Notification) -> bool:
        if notification.method not in self.notification_methods():
            return False
        params = validate_wire(RuntimeUpdatedParams, notification.params)
        if params.session_id != self._state.session_id:
            logger.debug(
                "Discarding runtime/updated for session_id=%s; current session_id=%s",
                params.session_id,
                self._state.session_id,
            )
            # The initialization of a fresh session can finish after an in-place
            # resume has adopted another session. It is still a handled runtime
            # notification; letting it fall through to the event projection
            # makes the client treat the old-session update as an unknown event.
            return True
        self._state.apply_runtime(params.runtime)
        return True
