from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import signal
import sys
from typing import Any, cast, override
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    Agent as AcpAgent,
    Client,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
)
from acp.helpers import ContentBlock
from acp.schema import (
    AcceptElicitationResponse,
    AcpMcpServer,
    AgentCapabilities,
    AuthenticateResponse,
    ClientCapabilities,
    CloseSessionResponse,
    ConfigOptionUpdate,
    Cost,
    ElicitationBooleanPropertySchema,
    ElicitationFormSessionMode,
    ElicitationIntegerPropertySchema,
    ElicitationMultiSelectPropertySchema,
    ElicitationNumberPropertySchema,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    EnumOption,
    EnvVarAuthMethod,
    ForkSessionResponse,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    McpCapabilities,
    McpServerStdio,
    PromptCapabilities,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionForkCapabilities,
    SessionInfo,
    SessionInfoUpdate,
    SessionListCapabilities,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TerminalAuthMethod,
    TitledMultiSelectItems,
    Usage,
    UsageUpdate,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chartreux import __version__
from chartreux.acp.acp_logger import acp_message_observer
from chartreux.acp.auth import AcpAuthController, OnboardingContextLoader
from chartreux.acp.commands import (
    AcpCommandController,
    AcpCommandRegistry,
    InjectedPrompt,
)
from chartreux.acp.content import ProjectedPrompt, project_prompt
from chartreux.acp.exceptions import (
    ConfigurationError,
    InternalError,
    InvalidRequestError,
    NotImplementedMethodError,
    SessionNotFoundError,
    UnauthenticatedError,
    from_public_error,
)
from chartreux.acp.models import ConfigSchemaResponse
from chartreux.acp.session import AcpSession
from chartreux.acp.session_updates import (
    replay_session_updates,
    session_updates_for_event,
)
from chartreux.acp.tool_io import AcpClientToolHandler
from chartreux.acp.user_display_content import (
    USER_DISPLAY_CONTENT_META_KEY,
    parse_user_display_content_metadata,
)
from chartreux.acp.utils import (
    build_model_config,
    is_jetbrains_client,
    make_thinking_response,
)
from chartreux.app_server.events import (
    AppServerEvent,
    CallbackRequested,
    StatsUpdated,
    TurnCompleted,
    TurnRetrying,
)
from chartreux.app_server.host import AppServerHost
from chartreux.app_server.local import (
    ClientDescriptor,
    LocalHarnessHost,
    LocalHarnessOptions,
    NewSessionIntent,
    ResumeSessionIntent,
)
from chartreux.app_server.models import (
    ImageAttachment,
    MentionStats,
    PublicCallbackEntry,
    PublicRetryCategory,
    PublicTurnStatus,
    PublicTurnStopReason,
    TurnErrorCode,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    CallbackKind,
    CallbackResultError,
    ClientCapabilities as AppServerClientCapabilities,
    ClientInfo,
    ClientToolCapability,
    ProtocolErrorCode,
    ReviewBaselineParams,
    ReviewHunksParams,
    ReviewMutationParams,
    ReviewStateParams,
    ReviewTurnDiffParams,
    SessionMCPHttpServer,
    SessionMCPServer,
    SessionMCPStdioServer,
    SessionOptions,
)
from chartreux.app_server.session import AppServerSession, AppServerTurnError
from chartreux.observability.logging import logger
from chartreux.questions import UserAnswer, UserQuestionRequest, UserQuestionResult
from chartreux.user_content import UserDisplayContent, UserResource

NON_INTERACTIVE_DISABLED_TOOLS: tuple[str, ...] = ("ask_user_question",)


INITIAL_AVAILABLE_COMMANDS_DELAY_SECONDS = 0.1
type SessionStarter = Callable[[LocalHarnessOptions], Awaitable[AppServerSession]]


type _ElicitationProperty = (
    ElicitationStringPropertySchema
    | ElicitationNumberPropertySchema
    | ElicitationIntegerPropertySchema
    | ElicitationBooleanPropertySchema
    | ElicitationMultiSelectPropertySchema
)


def _build_elicitation_schema(request: UserQuestionRequest) -> ElicitationSchema:
    properties: dict[str, _ElicitationProperty] = {}
    for index, question in enumerate(request.questions):
        # ACP has no "enum with type your answer option" property
        # We do not enforce enum matching on response so client can add that option everywhere
        # Alternative is adding custom sentinel values
        key = f"q{index}"
        options = [
            EnumOption(const=opt.label, title=opt.label) for opt in question.options
        ]
        if question.multi_select:
            properties[key] = ElicitationMultiSelectPropertySchema(
                type="array",
                title=question.header or None,
                description=question.question,
                items=TitledMultiSelectItems(any_of=options),
            )
        else:
            properties[key] = ElicitationStringPropertySchema(
                type="string",
                title=question.header or None,
                description=question.question,
                one_of=options,
            )
    return ElicitationSchema(
        properties=properties,
        required=list(properties),
        description=request.footer_note,
    )


def _elicit_user_answers(
    schema: ElicitationSchema, request: UserQuestionRequest, content: dict[str, Any]
) -> list[UserAnswer]:
    answers: list[UserAnswer] = []
    for key, question in zip(schema.properties or {}, request.questions, strict=True):
        labels = {opt.label for opt in question.options}
        raw = content.get(key)
        if question.multi_select:
            if not isinstance(raw, list):
                raise InvalidRequestError(
                    f"Elicitation response for {key} must be an array"
                )
            if not raw:
                raise InvalidRequestError(
                    f"Elicitation response for {key} is an empty array"
                )
            joined = ", ".join(str(item) for item in raw)
            is_other = any(str(item) not in labels for item in raw)
            answers.append(
                UserAnswer(question=question.question, answer=joined, is_other=is_other)
            )
        else:
            if not isinstance(raw, str):
                raise InvalidRequestError(
                    f"Elicitation response for {key} must be a string"
                )
            if not raw:
                raise InvalidRequestError(
                    f"Elicitation response for {key} is an empty string"
                )
            answers.append(
                UserAnswer(
                    question=question.question, answer=raw, is_other=raw not in labels
                )
            )
    return answers


@dataclass(frozen=True, slots=True)
class _TurnInput:
    text: str
    client_message_id: str | None = None
    auto_title: str | None = None
    images: list[ImageAttachment] = field(default_factory=list)
    resources: list[UserResource] = field(default_factory=list)
    user_display_content: UserDisplayContent | None = None
    mention_stats: MentionStats | None = None
    injected: bool = False


def _project_acp_mcp_servers(
    servers: list[HttpMcpServer | SseMcpServer | McpServerStdio | AcpMcpServer],
) -> list[SessionMCPServer]:
    projected: list[SessionMCPServer] = []
    for server in servers:
        match server:
            case HttpMcpServer():
                projected.append(
                    SessionMCPHttpServer(
                        transport="streamable-http",
                        name=server.name,
                        url=server.url,
                        headers={
                            header.name: header.value for header in server.headers
                        },
                    )
                )
            case McpServerStdio():
                projected.append(
                    SessionMCPStdioServer(
                        name=server.name,
                        command=server.command,
                        args=server.args,
                        env={variable.name: variable.value for variable in server.env},
                    )
                )
            case SseMcpServer():
                raise ConfigurationError(
                    f"MCP server {server.name!r} uses unsupported SSE transport"
                )
            case AcpMcpServer():
                raise ConfigurationError(
                    f"MCP server {server.name!r} uses unsupported ACP transport"
                )
    return projected


class SessionSetTitleRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    session_id: str = Field(alias="sessionId", min_length=1)
    title: str = Field(min_length=1)


class SessionDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    session_id: str = Field(alias="sessionId", min_length=1)


class SessionIdRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    session_id: str = Field(alias="sessionId", min_length=1)


class LogLevelWriteRequest(SessionIdRequest):
    session_override: str | None = Field(default=None, alias="sessionOverride")
    config_level: str | None = Field(default=None, alias="configLevel")

    @field_validator("session_override", "config_level")
    @classmethod
    def _validate_level(cls, level: str | None) -> str | None:
        if level is None:
            return None
        from chartreux.observability.logging import LOG_LEVELS

        normalized = level.upper()
        if normalized not in LOG_LEVELS:
            raise ValueError(
                f"Invalid log level {level!r}; expected one of {sorted(LOG_LEVELS)}"
            )
        return normalized


class ForkSessionParams(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    message_id: str | None = Field(default=None, alias="messageId")


# Bare name: `Client.ext_notification` prepends the `_` that ACP requires on
# extension methods, so the wire method is `_session/retrying` -- which is what
# the VS Code client subscribes to (see acp-session-retrying.ts). Adding the
# prefix here would emit `__session/retrying` and the retry would never arrive.
RETRYING_EXT_METHOD = "session/retrying"


class SessionRetryingNotification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(serialization_alias="sessionId")
    category: PublicRetryCategory
    detail: str


class ChartreuxAcpAgent(AcpAgent):
    client: Client

    def __init__(
        self,
        *,
        session_starter: SessionStarter | None = None,
        onboarding_context_loader: OnboardingContextLoader | None = None,
        environ_before_dotenv_load: Mapping[str, str] | None = None,
    ) -> None:
        self.sessions: dict[str, AcpSession] = {}
        self.client_capabilities: ClientCapabilities | None = None
        self.client_info: Implementation | None = None
        self._harness_host = LocalHarnessHost()
        self._start_session = session_starter or self._harness_host.start
        self._passive_host: AppServerHost | None = None
        self._passive_host_lock = asyncio.Lock()
        auth_kwargs: dict[str, Any] = {
            "context_loader": onboarding_context_loader,
            "environ_before_dotenv_load": environ_before_dotenv_load,
        }
        self._auth = AcpAuthController(**auth_kwargs)
        self._command_controller = AcpCommandController(
            lambda: self.client, self._send_config_options
        )

    @override
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        del protocol_version, kwargs
        self.client_capabilities = client_capabilities
        self.client_info = client_info
        delegated = bool(
            client_capabilities
            and client_capabilities.field_meta
            and client_capabilities.field_meta.get("browser-auth-delegated") is True
        )
        auth_methods: list[EnvVarAuthMethod | TerminalAuthMethod | Any] = [
            *self._auth.browser_methods(delegated=delegated)
        ]
        supports_terminal = bool(
            client_capabilities
            and client_capabilities.field_meta
            and client_capabilities.field_meta.get("terminal-auth") is True
        )
        if supports_terminal:
            command = sys.executable
            args = (
                ["--setup"]
                if "python" not in Path(command).name
                else [sys.argv[0], "--setup"]
            )
            auth_methods.append(
                TerminalAuthMethod(
                    type="terminal",
                    id="chartreux-setup",
                    name="Register your API Key",
                    description="Register your API Key inside Chartreux",
                    args=args,
                    field_meta={
                        "terminal-auth": {
                            "command": command,
                            "args": args,
                            "label": "Chartreux Setup",
                        }
                    },
                )
            )
        if (
            is_jetbrains_client(client_info)
            and self._auth.status().can_use_active_provider
        ):
            auth_methods = []
        return InitializeResponse(
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(
                    audio=False, embedded_context=True, image=True
                ),
                mcp_capabilities=McpCapabilities(http=True),
                session_capabilities=SessionCapabilities(
                    close=SessionCloseCapabilities(),
                    list=SessionListCapabilities(),
                    fork=SessionForkCapabilities(),
                ),
            ),
            protocol_version=PROTOCOL_VERSION,
            agent_info=Implementation(
                name="chartreux", title="Chartreux", version=__version__
            ),
            auth_methods=cast(Any, auth_methods),
        )

    @override
    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        del session_id, mode_id, kwargs
        raise NotImplementedMethodError("session/set_mode")

    @override
    async def authenticate(
        self, method_id: str, **kwargs: Any
    ) -> AuthenticateResponse | None:
        return await self._auth.authenticate(method_id, kwargs)

    @override
    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio | AcpMcpServer]
        | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del kwargs
        session = await self._create_session(
            Path(cwd),
            NewSessionIntent(),
            workspace_roots=additional_directories,
            mcp_servers=mcp_servers,
        )
        self._send_usage_update(session)
        return NewSessionResponse(
            session_id=session.id,
            config_options=self._config_options(session),
            field_meta=await self._trust_meta(session, cwd),
        )

    @override
    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio | AcpMcpServer]
        | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        del kwargs
        session = await self._create_session(
            Path(cwd),
            ResumeSessionIntent(session_id),
            acp_session_id=session_id,
            workspace_roots=additional_directories,
            mcp_servers=mcp_servers,
        )
        await self._load_complete_history(session)
        for update in replay_session_updates(session.app_server.state):
            await self.client.session_update(session_id=session.id, update=update)
        self._send_usage_update(session)
        return LoadSessionResponse(
            config_options=self._config_options(session),
            field_meta=await self._trust_meta(session, cwd),
        )

    async def _create_session(
        self,
        cwd: Path,
        intent: NewSessionIntent | ResumeSessionIntent,
        *,
        acp_session_id: str | None = None,
        workspace_roots: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio | AcpMcpServer]
        | None = None,
    ) -> AcpSession:
        client_tool_handler = AcpClientToolHandler(self.client)
        try:
            disabled_tools = (
                []
                if self._supports_user_input()
                else list(NON_INTERACTIVE_DISABLED_TOOLS)
            )
            app_server = await self._start_session(
                LocalHarnessOptions(
                    client=self._client_descriptor(),
                    session_options=SessionOptions(
                        cwd=str(cwd),
                        workspace_roots=workspace_roots or [],
                        disabled_tools=disabled_tools,
                        mcp_servers=_project_acp_mcp_servers(mcp_servers or []),
                    ),
                    session=intent,
                    client_tool_handler=client_tool_handler,
                )
            )
        except AppServerResponseError as exc:
            if exc.error.code is ProtocolErrorCode.UNAUTHORIZED:
                data = exc.error.data
                provider = data.get("provider") if isinstance(data, dict) else None
                if isinstance(provider, str):
                    raise UnauthenticatedError.for_provider(provider) from exc
            if exc.error.code is ProtocolErrorCode.NOT_FOUND and isinstance(
                intent, ResumeSessionIntent
            ):
                raise SessionNotFoundError(intent.session_id) from exc
            raise ConfigurationError(exc.error.message) from exc
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        session_id = acp_session_id or app_server.session_id
        client_tool_handler.bind_session(session_id)
        commands = AcpCommandRegistry()
        session = AcpSession(
            session_id=session_id,
            app_server=app_server,
            cwd=cwd.resolve(),
            commands=commands,
        )
        self.sessions[session.id] = session
        session.spawn(self._forward_unsolicited_events(session))
        session.spawn(self._warm_up(session))
        session.spawn(self._send_initial_commands(session))
        return session

    @staticmethod
    async def _load_complete_history(session: AcpSession) -> None:
        while before := session.app_server.resources.sessions.history_before_cursor:
            page = await session.app_server.resources.sessions.load_before(
                before, limit=500
            )
            if not page.data:
                raise RuntimeError("History pagination did not advance")

    def _supports_user_input(self) -> bool:
        capabilities = self.client_capabilities
        return bool(
            capabilities and capabilities.elicitation and capabilities.elicitation.form
        )

    def _client_descriptor(self) -> ClientDescriptor:
        info = self.client_info
        capabilities = self.client_capabilities
        client_tools: list[ClientToolCapability] = []
        if capabilities is not None and capabilities.fs is not None:
            if capabilities.fs.read_text_file:
                client_tools.append("filesystem/read")
            if capabilities.fs.write_text_file:
                client_tools.append("filesystem/write")
        if capabilities is not None and capabilities.terminal:
            client_tools.append("terminal")
        callback_kinds: list[CallbackKind] = []
        if self._supports_user_input():
            callback_kinds.append("user_input")
        return ClientDescriptor(
            info=ClientInfo(
                name=info.name if info is not None else "vibe_acp_client",
                title=info.title if info is not None else None,
                version=info.version if info is not None else "unknown",
                entrypoint="acp",
            ),
            capabilities=AppServerClientCapabilities(
                callback_kinds=callback_kinds, client_tools=client_tools
            ),
        )

    async def _warm_up(self, session: AcpSession) -> None:
        with suppress(Exception):
            await session.app_server.resources.runtime.wait_until_ready()
            await self._notify_mcp_discovery_failures(session)
            await self._notify_mcp_auth(session)

    async def _notify_mcp_discovery_failures(self, session: AcpSession) -> None:
        errors = session.app_server.resources.runtime.mcp.discovery_errors
        if not errors:
            return
        lines = ["The following MCP servers failed to connect:"]
        lines.extend(f"- {name}: {error}" for name, error in sorted(errors.items()))
        await self._command_controller.message(session, "\n".join(lines))

    async def _notify_mcp_auth(self, session: AcpSession) -> None:
        aliases = session.app_server.resources.runtime.mcp.needs_auth
        if not aliases:
            return
        message = (
            "MCP OAuth login is required for: "
            f"{', '.join(aliases)}. Run `/mcp login <alias>` in Chartreux."
        )
        await self._command_controller.message(session, message)

    async def _forward_unsolicited_events(self, session: AcpSession) -> None:
        async for event in session.app_server.events():
            await self._forward_event(session, event)
            if isinstance(event, TurnCompleted):
                self._send_usage_update(session)

    async def _forward_event(self, session: AcpSession, event: AppServerEvent) -> None:
        if isinstance(event, CallbackRequested):
            await self._answer_callback(session, event.callback)
            return
        if isinstance(event, TurnRetrying):
            await self._send_retrying(session, event)
            return
        for update in session_updates_for_event(event):
            await self.client.session_update(session_id=session.id, update=update)
        if isinstance(event, StatsUpdated):
            self._send_usage_update(session)

    async def _send_retrying(self, session: AcpSession, event: TurnRetrying) -> None:
        notification = SessionRetryingNotification(
            session_id=session.id,
            category=event.params.category,
            detail=event.params.detail,
        )
        await self.client.ext_notification(
            RETRYING_EXT_METHOD, notification.model_dump(mode="json", by_alias=True)
        )

    @staticmethod
    async def _prepare_turn_input(
        session: AcpSession,
        content: ProjectedPrompt,
        *,
        message_id: str,
        display: UserDisplayContent | None,
    ) -> _TurnInput:
        prepared = await session.app_server.resources.workspace.prepare_prompt(
            content.text, title_content=content.title_content
        )
        return _TurnInput(
            text=prepared.prompt_text,
            client_message_id=message_id,
            auto_title=prepared.auto_title,
            images=[*prepared.images, *content.images],
            resources=content.resources,
            user_display_content=display,
            mention_stats=prepared.mentions,
        )

    @override
    async def prompt(
        self, session_id: str, prompt: list[ContentBlock], **kwargs: Any
    ) -> PromptResponse:
        session = self._get_session(session_id)
        try:
            display = parse_user_display_content_metadata(
                kwargs.get(USER_DISPLAY_CONTENT_META_KEY)
            )
        except ValidationError as exc:
            raise InvalidRequestError(
                f"Invalid user display content metadata: {exc}"
            ) from exc
        content = project_prompt(prompt)
        text = content.text
        message_id = str(uuid4())
        match await self._command_controller.execute(session, text, message_id):
            case PromptResponse() as response:
                return response
            case InjectedPrompt(text=injected_text):
                turn_input = _TurnInput(text=injected_text, injected=True)
            case None:
                turn_input = await self._prepare_turn_input(
                    session, content, message_id=message_id, display=display
                )

        async def run_turn() -> None:
            async with aclosing(
                session.app_server.act(
                    turn_input.text,
                    client_message_id=turn_input.client_message_id,
                    auto_title=turn_input.auto_title,
                    images=turn_input.images,
                    resources=turn_input.resources,
                    user_display_content=turn_input.user_display_content,
                    mention_stats=turn_input.mention_stats,
                    injected=turn_input.injected,
                )
            ) as events:
                async for event in events:
                    await self._forward_event(session, event)

        prompt_task = asyncio.current_task()
        if prompt_task is None:
            raise RuntimeError("ACP prompt must run in an asyncio task")
        session.set_prompt_task(prompt_task)
        try:
            try:
                await run_turn()
            except asyncio.CancelledError:
                self._send_usage_update(session)
                return PromptResponse(
                    stop_reason="cancelled", usage=self._usage(session)
                )
            except AppServerTurnError as exc:
                if exc.error.code == TurnErrorCode.RESPONSE_TOO_LONG:
                    self._send_usage_update(session)
                    return PromptResponse(
                        stop_reason="max_tokens", usage=self._usage(session)
                    )
                raise from_public_error(exc.error) from exc
            except Exception as exc:
                logger.exception("ACP prompt failed")
                raise InternalError(str(exc)) from exc
        finally:
            session.clear_prompt_task(prompt_task)
        self._send_usage_update(session)
        turn = next(reversed(session.app_server.state.turns or []), None)
        if turn is not None and turn.status is PublicTurnStatus.INTERRUPTED:
            return PromptResponse(stop_reason="cancelled", usage=self._usage(session))
        if turn is not None and turn.stop_reason is PublicTurnStopReason.LIMIT:
            return PromptResponse(
                stop_reason="max_turn_requests", usage=self._usage(session)
            )
        return PromptResponse(stop_reason="end_turn", usage=self._usage(session))

    async def _answer_callback(
        self, session: AcpSession, callback: PublicCallbackEntry
    ) -> None:
        await self._answer_user_input(session, callback, callback.detail)

    async def _answer_user_input(
        self,
        session: AcpSession,
        callback: PublicCallbackEntry,
        detail: UserInputCallbackDetail,
    ) -> None:
        request = detail.request
        schema = _build_elicitation_schema(request)
        response = await self.client.create_elicitation(
            message="User input required",
            mode=ElicitationFormSessionMode(
                session_id=session.id,
                tool_call_id=detail.related_entry_id,
                requested_schema=schema,
            ),
        )
        if not isinstance(response, AcceptElicitationResponse) or not response.content:
            await session.app_server.deny_callback(callback)
            return
        try:
            answers = _elicit_user_answers(schema, request, response.content)
        except InvalidRequestError as exc:
            await session.app_server.reject_callback(
                callback.callback_id, CallbackResultError(message=str(exc))
            )
            return
        await session.app_server.respond_to_callback(
            callback.callback_id,
            UserInputCallbackOutput(result=UserQuestionResult(answers=answers)),
        )

    @override
    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del kwargs
        session = self._get_session(session_id)
        await session.cancel_prompt()

    @override
    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse | None:
        del kwargs
        if not await self._close_session_if_present(session_id):
            raise SessionNotFoundError(session_id)
        return CloseSessionResponse()

    async def _close_session_if_present(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            return False
        await session.close()
        self.sessions.pop(session_id, None)
        return True

    async def close(self) -> None:
        sessions = list(self.sessions.values())
        self.sessions.clear()
        if sessions:
            await asyncio.gather(
                *(session.close() for session in sessions), return_exceptions=True
            )
        if self._passive_host is not None:
            await self._passive_host.close()
            self._passive_host = None
        await self._harness_host.close()

    @override
    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        del cursor, kwargs
        saved = await (await self._host_resources()).list_sessions(cwd=cwd)
        return ListSessionsResponse(
            sessions=[
                SessionInfo(
                    session_id=item.id,
                    cwd=item.cwd or "",
                    # No LLM title yet: fall back to the first-message preview so
                    # the list stays readable instead of showing "untitled".
                    title=item.title or item.preview,
                    updated_at=datetime.fromtimestamp(
                        item.updated_at / 1000, UTC
                    ).isoformat(),
                )
                for item in saved
            ]
        )

    @override
    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        del kwargs
        session = self._get_session(session_id)
        try:
            match config_id, value:
                case "model", str(model) if any(
                    candidate.alias == model
                    for candidate in session.app_server.resources.config.current.models
                ):
                    await session.app_server.resources.config.update(
                        {"active_model": model}, reload_runtime=True
                    )
                case "thinking", str(level) if level in {
                    "off",
                    "low",
                    "medium",
                    "high",
                    "max",
                }:
                    await session.app_server.resources.config.set_thinking(
                        cast(Any, level)
                    )
                case "max_turns", str(raw):
                    await session.app_server.resources.sessions.update_settings(
                        max_turns=int(raw)
                    )
                case "max_tokens", str(raw):
                    await session.app_server.resources.sessions.update_settings(
                        max_tokens=int(raw)
                    )
                case _:
                    raise InvalidRequestError(
                        f"Unsupported config option {config_id}={value!r}"
                    )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        except AppServerResponseError as exc:
            raise InvalidRequestError(exc.error.message) from exc
        return SetSessionConfigOptionResponse(
            config_options=self._config_options(session)
        )

    @override
    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio | AcpMcpServer]
        | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        source = self._get_session(session_id)
        try:
            params = ForkSessionParams.model_validate(kwargs)
            fork = await source.app_server.resources.sessions.fork(
                params.message_id, attach=False
            )
        except ValidationError as exc:
            raise InvalidRequestError(f"Invalid fork parameters: {exc}") from exc
        except AppServerResponseError as exc:
            raise InvalidRequestError(exc.error.message) from exc
        child = await self._create_session(
            Path(cwd),
            ResumeSessionIntent(fork.state.session.id),
            acp_session_id=fork.state.session.id,
            workspace_roots=additional_directories,
            mcp_servers=mcp_servers,
        )
        self._send_usage_update(child)
        return ForkSessionResponse(
            session_id=child.id, config_options=self._config_options(child)
        )

    @override
    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio | AcpMcpServer]
        | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        del kwargs
        if session_id not in self.sessions:
            session = await self._create_session(
                Path(cwd),
                ResumeSessionIntent(session_id),
                acp_session_id=session_id,
                workspace_roots=additional_directories,
                mcp_servers=mcp_servers,
            )
        else:
            session = self._get_session(session_id)
        self._send_usage_update(session)
        return ResumeSessionResponse()

    @override
    async def ext_method(self, method: str, params: dict) -> dict:
        match method:
            case "auth/status":
                state = self._auth.status()
                result = {
                    "authenticated": state.can_use_active_provider,
                    "authState": state.kind.value,
                    "customDomain": self._auth.custom_domain(),
                }
            case "config/schema":
                result = await self._config_schema()
            case "session/set_title":
                try:
                    request = SessionSetTitleRequest.model_validate(params)
                except ValidationError as exc:
                    raise InvalidRequestError(
                        f"Invalid ACP session title request: {exc}"
                    ) from exc
                session = self._find_live_session(request.session_id)
                try:
                    response = (
                        await session.app_server.resources.sessions.rename_with_metadata(
                            request.title
                        )
                        if session is not None
                        else await (await self._host_resources()).rename_session(
                            request.session_id, request.title
                        )
                    )
                except AppServerResponseError as exc:
                    if exc.error.code is ProtocolErrorCode.NOT_FOUND:
                        raise SessionNotFoundError(request.session_id) from exc
                    raise InvalidRequestError(exc.error.message) from exc
                await self.client.session_update(
                    session_id=session.id
                    if session is not None
                    else request.session_id,
                    update=SessionInfoUpdate(
                        session_update="session_info_update",
                        title=response.title,
                        updated_at=response.updated_at,
                    ),
                )
                result = {}
            case "session/delete":
                try:
                    request = SessionDeleteRequest.model_validate(params)
                except ValidationError as exc:
                    raise InvalidRequestError(
                        f"Invalid ACP session delete request: {exc}"
                    ) from exc
                await self._delete_session(request.session_id)
                result = {}
            case "trust/status" | "trust/decision":
                result = await self._trust_extension(method, params)
            case "identity/read":
                result = await self._identity_extension(params)
            case _ if method.startswith("logLevel/"):
                result = await self._log_level_extension(method, params)
            case "rewind/preview" | "rewind/to":
                result = await self._rewind_extension(method, params)
            case (
                "review/state"
                | "review/baseline"
                | "review/turnDiff"
                | "review/hunks"
                | "review/approve"
                | "review/revert"
            ):
                result = await self._review_extension(method, params)
            case _:
                raise NotImplementedMethodError(method)
        return result

    async def _config_schema(self) -> dict[str, Any]:
        response = await (await self._host_resources()).read_config_schema()
        return ConfigSchemaResponse(
            version=response.config_schema_version, schema=response.config_schema
        ).model_dump(mode="json", by_alias=True)

    # -- identity ------------------------------------------------------------

    async def _identity_extension(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            request = SessionIdRequest.model_validate(params)
        except ValidationError as exc:
            raise InvalidRequestError(f"Invalid ACP identity request: {exc}") from exc
        session = self._get_session(request.session_id)
        try:
            identity = await session.app_server.resources.identity.read()
        except AppServerResponseError as exc:
            raise InvalidRequestError(exc.error.message) from exc
        return identity.model_dump(mode="json") if identity else {}

    # -- log level -----------------------------------------------------------

    @staticmethod
    def _log_level_response() -> dict[str, str | None]:
        from chartreux.observability.logging import get_log_level_chain

        chain = get_log_level_chain()
        return {
            "session": chain.session,
            "env": chain.env,
            "config": chain.config,
            "effective": chain.effective,
        }

    async def _log_level_extension(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, str | None]:
        from chartreux.observability.logging import (
            get_session_override,
            set_config_log_level,
            set_session_override,
        )

        match method:
            case "logLevel/read":
                return self._log_level_response()
            case "logLevel/write":
                try:
                    request = LogLevelWriteRequest.model_validate(params)
                except ValidationError as exc:
                    raise InvalidRequestError(
                        f"Invalid ACP log level write request: {exc}"
                    ) from exc
                session = self._get_session(request.session_id)
                previous_override = get_session_override()
                has_session_override = "sessionOverride" in params
                if has_session_override:
                    set_session_override(request.session_override)
                try:
                    if "configLevel" in params:
                        try:
                            await session.app_server.resources.config.update(
                                {"log_level": request.config_level}, reload_runtime=True
                            )
                        except AppServerResponseError as exc:
                            raise InvalidRequestError(exc.error.message) from exc
                        set_config_log_level(request.config_level)
                except BaseException:
                    if (
                        has_session_override
                        and get_session_override() == request.session_override
                    ):
                        set_session_override(previous_override)
                    raise
                return self._log_level_response()
            case _:
                raise NotImplementedMethodError(method)

    async def _delete_session(self, session_id: str) -> None:
        session = self._find_live_session(session_id)
        if session is None:
            await (await self._host_resources()).delete_session(session_id)
            return

        saved_session_id = session.app_server.exit_summary().session_id
        await session.close()
        if saved_session_id is not None:
            await (await self._host_resources()).delete_session(saved_session_id)
        self.sessions.pop(session.id, None)

    @override
    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        raise NotImplementedMethodError(method)

    def on_connect(self, conn: Client) -> None:
        self.client = conn

    async def _host_resources(self) -> AppServerHost:
        if self._passive_host is not None:
            return self._passive_host
        async with self._passive_host_lock:
            if self._passive_host is None:
                self._passive_host = await self._harness_host.connect(
                    LocalHarnessOptions(
                        client=self._client_descriptor(),
                        session_options=SessionOptions(cwd=str(Path.cwd())),
                    )
                )
        return self._passive_host

    async def _trust_extension(self, method: str, params: dict[str, Any]) -> dict:
        requested_session = params.get("sessionId") or params.get("session_id")
        session = (
            self._get_session(requested_session)
            if isinstance(requested_session, str)
            else None
        )
        if method == "trust/status":
            if session is not None:
                response = await session.app_server.resources.workspace.trust_status(
                    params.get("cwd")
                )
            else:
                host = await self._host_resources()
                response = await host.trust_status(params.get("cwd"))
        else:
            if session is None:
                raise InvalidRequestError("Trust decisions require a valid sessionId")
            decision = params.get("decision")
            if not isinstance(decision, str) or decision not in {
                "trust_repo",
                "trust_cwd",
                "decline",
            }:
                raise InvalidRequestError(f"Unknown trust decision: {decision}")
            requested_cwd = params.get("cwd")
            if requested_cwd is not None and not isinstance(requested_cwd, str):
                raise InvalidRequestError("Trust decision cwd must be a string")
            try:
                session_cwd = Path(session.app_server.cwd).expanduser().resolve()
                canonical_cwd = (
                    Path(requested_cwd).expanduser().resolve()
                    if requested_cwd is not None
                    else session_cwd
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise InvalidRequestError("Trust decision cwd is invalid") from exc
            if canonical_cwd != session_cwd:
                raise InvalidRequestError(
                    "Trust decision cwd must match the session working directory"
                )
            response = await session.app_server.resources.workspace.decide_trust(
                cast(Any, decision), cwd=str(canonical_cwd)
            )
        return {
            "trust_status": response.status,
            "details": (
                response.details.model_dump(mode="json", by_alias=True)
                if response.details is not None
                else None
            ),
        }

    async def _rewind_extension(self, method: str, params: dict[str, Any]) -> dict:
        session_id = params.get("sessionId") or params.get("session_id")
        entry_id = params.get("messageId") or params.get("message_id")
        if not isinstance(session_id, str) or not isinstance(entry_id, str):
            raise InvalidRequestError("Rewind requires sessionId and messageId")
        session = self._get_session(session_id)
        if method == "rewind/preview":
            try:
                paths = await session.app_server.resources.sessions.rewind_preview(
                    entry_id
                )
            except AppServerResponseError as exc:
                raise InvalidRequestError(exc.error.message) from exc
            return {"paths": paths}
        try:
            response = await session.app_server.resources.sessions.rewind(
                entry_id,
                restore_files=bool(params.get("restoreFiles", True)),
                inplace=True,
            )
        except AppServerResponseError as exc:
            raise InvalidRequestError(exc.error.message) from exc
        return {
            "messageContent": response.message,
            "restoreErrors": list(response.restore_errors),
            "restoredPaths": list(response.restored_paths),
        }

    async def _review_extension(self, method: str, params: dict[str, Any]) -> dict:
        try:
            match method:
                case "review/state":
                    request = ReviewStateParams.model_validate(params)
                    review = self._get_session(
                        request.session_id
                    ).app_server.resources.review
                    response = await review.state()
                case "review/baseline":
                    request = ReviewBaselineParams.model_validate(params)
                    review = self._get_session(
                        request.session_id
                    ).app_server.resources.review
                    response = await review.baseline(request.path)
                case "review/turnDiff":
                    request = ReviewTurnDiffParams.model_validate(params)
                    review = self._get_session(
                        request.session_id
                    ).app_server.resources.review
                    response = await review.turn_diff(request.path, request.owner)
                case "review/hunks":
                    request = ReviewHunksParams.model_validate(params)
                    review = self._get_session(
                        request.session_id
                    ).app_server.resources.review
                    response = await review.hunks(request.path, request.owner)
                case "review/approve" | "review/revert":
                    request = ReviewMutationParams.model_validate(params)
                    review = self._get_session(
                        request.session_id
                    ).app_server.resources.review
                    try:
                        if method == "review/approve":
                            await review.approve(request.target)
                        else:
                            await review.revert(request.target)
                    except AppServerResponseError as exc:
                        raise InvalidRequestError(exc.error.message) from exc
                    return {}
                case _:
                    raise NotImplementedMethodError(method)
        except ValidationError as exc:
            raise InvalidRequestError(f"Invalid ACP {method} request: {exc}") from exc
        return response.model_dump(mode="json", by_alias=True)

    async def _trust_meta(self, session: AcpSession, cwd: str) -> dict[str, Any]:
        response = await session.app_server.resources.workspace.trust_status(cwd)
        return {
            "workspace_trust": {
                "status": response.status,
                "details": (
                    response.details.model_dump(mode="json", by_alias=True)
                    if response.details is not None
                    else None
                ),
            }
        }

    def _config_options(
        self, session: AcpSession
    ) -> list[SessionConfigOptionSelect | SessionConfigOptionBoolean]:
        config = session.app_server.resources.config.current
        return [build_model_config(config), make_thinking_response(config)]

    def _usage(self, session: AcpSession) -> Usage:
        usage = session.app_server.resources.runtime.stats.token_usage
        return Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    def _send_usage_update(self, session: AcpSession) -> None:
        async def send() -> None:
            runtime = session.app_server.resources.runtime
            stats = runtime.stats
            session_cost = stats.session_cost
            cost = (
                Cost(amount=session_cost, currency="USD")
                if session_cost is not None and session_cost > 0
                else None
            )
            await self.client.session_update(
                session_id=session.id,
                update=UsageUpdate(
                    session_update="usage_update",
                    used=stats.context_tokens,
                    size=runtime.context_window,
                    cost=cost,
                    **{
                        "_meta": {
                            "steps": stats.steps,
                            "promptTokens": stats.session_prompt_tokens,
                            "completionTokens": stats.session_completion_tokens,
                            "cachedTokens": stats.session_cached_tokens,
                            "totalTokens": stats.session_total_llm_tokens,
                            "tokensPerSecond": stats.tokens_per_second,
                            "lastTurnDuration": stats.last_turn_duration,
                            "lastTurnTotalTokens": stats.last_turn_total_tokens,
                        }
                    },
                ),
            )

        session.spawn(send())

    async def _send_initial_commands(self, session: AcpSession) -> None:
        await asyncio.sleep(INITIAL_AVAILABLE_COMMANDS_DELAY_SECONDS)
        await self._command_controller.send_commands(session)

    async def _send_config_options(self, session: AcpSession) -> None:
        await self.client.session_update(
            session_id=session.id,
            update=ConfigOptionUpdate(
                session_update="config_option_update",
                config_options=self._config_options(session),
            ),
        )

    def _get_session(self, session_id: str) -> AcpSession:
        if session := self.sessions.get(session_id):
            return session
        raise SessionNotFoundError(session_id)

    def _find_live_session(self, session_id: str) -> AcpSession | None:
        return self.sessions.get(session_id) or next(
            (
                session
                for session in self.sessions.values()
                if session.app_server.session_id == session_id
            ),
            None,
        )


async def _serve_acp_agent(agent: ChartreuxAcpAgent) -> None:
    try:
        await run_agent(
            agent=agent, use_unstable_protocol=True, observers=[acp_message_observer]
        )
    finally:
        await agent.close()


def run_acp_server(
    *, environ_before_dotenv_load: Mapping[str, str] | None = None
) -> None:
    agent = ChartreuxAcpAgent(environ_before_dotenv_load=environ_before_dotenv_load)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        asyncio.run(_serve_acp_agent(agent))
    except KeyboardInterrupt:
        with suppress(Exception):
            asyncio.run(asyncio.wait_for(agent.close(), timeout=1.0))
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
