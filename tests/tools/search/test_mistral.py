from __future__ import annotations

import asyncio
import builtins
import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from chartreux.core.tools.search.models import MAX_UPSTREAM_JSON_BYTES
from chartreux.core.tools.search.provider import SearchProviderError


class _Response:
    def __init__(self, outputs: list[object], *, serialized: str | None = None) -> None:
        self.outputs = outputs
        self._serialized = serialized

    def model_dump_json(self) -> str:
        return self._serialized or "{}"


class _Client:
    def __init__(self, response: object | BaseException) -> None:
        self.response = response
        self.sdk_configuration = SimpleNamespace()
        self.calls: list[dict[str, Any]] = []
        self.constructor_args: dict[str, object] = {}
        self.closed = False
        self.beta = SimpleNamespace(
            conversations=SimpleNamespace(start_async=self._start_async)
        )

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True

    async def _start_async(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _provider():
    from chartreux.core.tools.search.mistral import MistralSearchProvider

    return MistralSearchProvider(
        "test-key", "https://mistral.example", model="mistral-small", timeout=2.5
    )


def _set_response(
    monkeypatch: pytest.MonkeyPatch, response: object | BaseException
) -> list[_Client]:
    clients: list[_Client] = []

    def create_client(**kwargs: object) -> _Client:
        client = _Client(response)
        client.constructor_args = kwargs
        clients.append(client)
        return client

    module = types.ModuleType("mistralai.client")
    module.Mistral = create_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mistralai.client", module)
    return clients


@pytest.mark.asyncio
async def test_search_passes_sdk_arguments_disables_telemetry_and_parses_chunks(
    monkeypatch,
):
    response = _Response([
        SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Hello "),
                SimpleNamespace(
                    type="tool_reference", title="One", url="https://one.example"
                ),
                SimpleNamespace(type="text", text="world"),
                SimpleNamespace(
                    type="tool_reference", title="Duplicate", url="https://one.example"
                ),
                SimpleNamespace(
                    type="tool_reference", title="Bad", url="ftp://bad.example"
                ),
            ]
        )
    ])
    clients = _set_response(monkeypatch, response)

    result = await _provider().search("query", max_results=10)

    client = clients[0]
    assert client.constructor_args == {
        "api_key": "test-key",
        "server_url": "https://mistral.example",
    }  # type: ignore[attr-defined]
    assert client.sdk_configuration.__dict__["telemetry"] is False
    assert client.calls == [
        {
            "model": "mistral-small",
            "inputs": "query",
            "tools": [{"type": "web_search"}],
            "store": False,
            "timeout_ms": 2500,
        }
    ]
    assert client.closed
    assert result.answer == "Hello world"
    assert [(source.title, source.url) for source in result.sources] == [
        ("One", "https://one.example")
    ]


@pytest.mark.asyncio
async def test_search_parses_string_only_answer(monkeypatch):
    _set_response(monkeypatch, _Response([SimpleNamespace(content=" answer ")]))
    result = await _provider().search("query", max_results=1)
    assert result.answer == "answer"
    assert result.sources == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_Response([]), "empty search response"),
        (
            _Response([], serialized='"' + "x" * (MAX_UPSTREAM_JSON_BYTES + 1) + '"'),
            "size limit",
        ),
        (TimeoutError("secret timeout"), "request timed out"),
        (RuntimeError("secret auth body"), "request failed"),
    ],
)
async def test_search_errors_are_sanitized(monkeypatch, response, message):
    clients = _set_response(monkeypatch, response)
    with pytest.raises(SearchProviderError, match=message) as caught:
        await _provider().search("query", max_results=1)
    assert "secret" not in str(caught.value)
    assert clients[0].closed


@pytest.mark.asyncio
async def test_search_propagates_cancellation_and_closes_client(monkeypatch):
    clients = _set_response(monkeypatch, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _provider().search("query", max_results=1)
    assert clients[0].closed


def test_rejects_missing_server_url():
    from chartreux.core.tools.search.mistral import MistralSearchProvider

    with pytest.raises(SearchProviderError, match="configured server URL"):
        MistralSearchProvider("key", None, model="model")


def test_mistral_import_is_lazy(monkeypatch):
    module_name = "chartreux.core.tools.search.mistral"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("mistralai"):
            pytest.fail("mistralai must not be imported during provider discovery")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module(module_name)
