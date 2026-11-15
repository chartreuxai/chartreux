from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import JsonValue

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.protocol import ProtocolErrorCode


@dataclass(frozen=True, slots=True)
class DispatchResult:
    response: ProtocolModel
    after_response: Callable[[], None] | None = None
    on_response_abandoned: Callable[[], None] | None = None
    runtime_updated: bool = False
    session_attached: bool = False


class RequestFailure(Exception):
    def __init__(
        self, code: ProtocolErrorCode, message: str, data: JsonValue = None
    ) -> None:
        self.code = code
        self.data = data
        super().__init__(message)


def method_not_found(method: str) -> RequestFailure:
    return RequestFailure(
        ProtocolErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}"
    )


#: The retained alias for a manual session title. The wire method that carries
#: the operation is ``SESSION_RENAME_METHOD``; the alias stays in the catalogue
#: so a client that still sends it is told the operation was renamed rather
#: than that the method vanished.
SESSION_TITLE_UPDATE_ALIAS_METHOD = "session/title/update"
SESSION_RENAME_METHOD = "session/rename"


def unsupported_title_alias() -> RequestFailure:
    """The typed refusal for the retained ``session/title/update`` alias.

    No topology carries the alias: the operation lives under ``session/rename``
    on both. A bare unknown-method error would read as a removed feature, and
    an uncarried-operation refusal would name a transport selector that does
    not carry it either, so the answer names the method that does.
    """
    return RequestFailure(
        ProtocolErrorCode.NOT_IMPLEMENTED,
        f"{SESSION_TITLE_UPDATE_ALIAS_METHOD} is not a live method: a manual "
        f"session title is set with {SESSION_RENAME_METHOD}. The request was "
        "not applied.",
        data={
            "alias": SESSION_TITLE_UPDATE_ALIAS_METHOD,
            "method": SESSION_RENAME_METHOD,
        },
    )
