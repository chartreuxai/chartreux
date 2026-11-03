from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StrictInt

from chartreux.app_server._model import ProtocolModel
from chartreux.utils import AgentEntrypoint
from chartreux.utils.remote_exec import (
    REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY,
    REMOTE_EXEC_SSH_TRUST_CAPABILITY,
    RemoteExecAttestation,
)
from chartreux.utils.terminal import TerminalEmulator


class ClientInfo(ProtocolModel):
    name: str
    version: str
    title: str | None = None
    entrypoint: AgentEntrypoint = "unknown"
    terminal_emulator: TerminalEmulator = TerminalEmulator.UNKNOWN


type CallbackKind = Literal["user_input"]
# v0.7.1 S4b1b: the client_tools channel also carries the execution-bound
# remote-exec attestation entries (CC — this row is the first lander, so it
# creates the entries). The wire change is forward-compatible only: an older
# app server rejects these entries from a newer client (the closed Literal
# fails validation), and an older strict client rejects any new response key.
# The two remote-exec entries repeat the wire spellings of the shared
# constants (a name is not a valid Literal argument); remote_exec_attestation
# below maps incoming entries using those constants.
type ClientToolCapability = Literal[
    "filesystem/read",
    "filesystem/write",
    "terminal",
    "remote_exec/network_locality",
    "remote_exec/ssh_trust_context",
]
type TransportKind = Literal["in_process", "stdio"]


class ClientToolMethod(StrEnum):
    READ_TEXT_FILE = "clientTool/readTextFile"
    WRITE_TEXT_FILE = "clientTool/writeTextFile"
    TERMINAL_CREATE = "clientTool/terminal/create"
    TERMINAL_WAIT = "clientTool/terminal/wait"
    TERMINAL_OUTPUT = "clientTool/terminal/output"
    TERMINAL_KILL = "clientTool/terminal/kill"
    TERMINAL_RELEASE = "clientTool/terminal/release"


class ClientCapabilities(ProtocolModel):
    callback_kinds: list[CallbackKind] = Field(default_factory=list)
    client_tools: list[ClientToolCapability] = Field(default_factory=list)
    disabled_notifications: list[str] = Field(default_factory=list)


def remote_exec_attestation(capabilities: ClientCapabilities) -> RemoteExecAttestation:
    """Map the wire capability entries onto the shared attestation object.

    The attestation is CLIENT-SELF-REPORTED (an honest limitation: it closes
    accidental misrouting, not a lying client). Only an execution-bound
    statement — network-namespace locality AND the filesystem/SSH trust
    context — admits remote execution; an entry attesting locality ONLY
    hard-denies transfers.
    """
    tools = set(capabilities.client_tools)
    return RemoteExecAttestation(
        network_locality=REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY in tools,
        ssh_trust_context=REMOTE_EXEC_SSH_TRUST_CAPABILITY in tools,
    )


class InitializeParams(ProtocolModel):
    client_info: ClientInfo
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)


class ServerInfo(ProtocolModel):
    name: str
    version: str


class ClientToolReadTextFileParams(ProtocolModel):
    session_id: str
    path: str
    line: Annotated[StrictInt, Field(ge=1)] | None = None
    limit: Annotated[StrictInt, Field(ge=1)] | None = None


class ClientToolReadTextFileResponse(ProtocolModel):
    content: str


class ClientToolWriteTextFileParams(ProtocolModel):
    session_id: str
    path: str
    content: str


class ClientToolTerminalCreateParams(ProtocolModel):
    session_id: str
    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None
    # Credential variable names the client should remove from the terminal's
    # inherited environment. The client owns the spawn, so cooperation is
    # advisory; chartreux forwards the list through ACP's ``_meta``
    # extensibility and still redacts secret values from returned output.
    env_scrub: list[str] | None = None
    cwd: str
    output_byte_limit: Annotated[StrictInt, Field(gt=0)]
    tool_call_id: str | None = None


class ClientToolTerminalCreateResponse(ProtocolModel):
    terminal_id: str


class ClientToolTerminalParams(ProtocolModel):
    session_id: str
    terminal_id: str


class ClientToolTerminalWaitResponse(ProtocolModel):
    exit_code: StrictInt | None = None
    signal: str | None = None


class ClientToolTerminalOutputResponse(ProtocolModel):
    output: str
    truncated: bool


class InitializeResponse(ProtocolModel):
    server_info: ServerInfo
