"""JSON-RPC envelope models and envelope validation."""

from __future__ import annotations

from functools import cache
from typing import Any, Literal

from pydantic import JsonValue, StrictInt, StrictStr, TypeAdapter

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.protocol.errors import ProtocolError, _render_protocol_error


class Notification(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, JsonValue]


class ServerRequest(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: StrictInt | StrictStr
    method: str
    params: dict[str, JsonValue]


class AppServerResponseError(RuntimeError):
    def __init__(self, error: ProtocolError) -> None:
        self.error = error
        super().__init__(_render_protocol_error(error))


class JsonRpcProtocolError(RuntimeError):
    pass


class JsonRpcSuccessResponse(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: StrictInt | StrictStr
    result: dict[str, JsonValue]


class JsonRpcErrorResponse(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: StrictInt | StrictStr
    error: ProtocolError


type JsonRpcEnvelope = (
    Notification | ServerRequest | JsonRpcSuccessResponse | JsonRpcErrorResponse
)


@cache
def _json_rpc_envelope_adapter() -> TypeAdapter[JsonRpcEnvelope]:
    return TypeAdapter(JsonRpcEnvelope)


def validate_json_rpc_envelope(value: object) -> JsonRpcEnvelope:
    return _json_rpc_envelope_adapter().validate_python(
        value, by_alias=True, by_name=False
    )


def protocol_value(value: ProtocolModel | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ProtocolModel):
        return value.model_dump(mode="json", by_alias=True)
    return value
