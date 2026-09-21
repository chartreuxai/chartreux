from __future__ import annotations

import asyncio
import builtins
import importlib
import sys
import threading

import ddgs
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
import pytest

from chartreux.core.tools.search.duckduckgo import DuckDuckGoSearchProvider
from chartreux.core.tools.search.provider import SearchProviderError


@pytest.mark.asyncio
async def test_search_passes_duckduckgo_backend_and_normalizes_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, str, int, str]] = []

    class Client:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        def text(
            self, query: str, *, max_results: int, backend: str
        ) -> list[dict[str, str]]:
            calls.append((self.timeout, query, max_results, backend))
            return [{"title": "Result", "href": "https://example.com", "body": "Text"}]

    monkeypatch.setattr(ddgs, "DDGS", Client)

    result = await DuckDuckGoSearchProvider(timeout=7.5).search(
        "test query", max_results=3
    )

    assert calls == [(7.5, "test query", 3, "duckduckgo")]
    assert result.answer is None
    assert result.sources[0].model_dump() == {
        "title": "Result",
        "url": "https://example.com",
        "snippet": "Text",
    }


@pytest.mark.asyncio
async def test_search_allows_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def text(self, query: str, *, max_results: int, backend: str) -> list[object]:
            return []

    monkeypatch.setattr(ddgs, "DDGS", Client)

    result = await DuckDuckGoSearchProvider().search("test query", max_results=3)

    assert result.sources == []
    assert result.was_truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            RatelimitException("secret rate-limit detail"),
            "DuckDuckGo search was rate limited",
        ),
        (TimeoutException("secret timeout detail"), "DuckDuckGo search timed out"),
        (DDGSException("secret request detail"), "DuckDuckGo search request failed"),
    ],
)
async def test_ddgs_exceptions_are_sanitized(
    monkeypatch: pytest.MonkeyPatch, error: Exception, message: str
) -> None:
    class Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def text(self, query: str, *, max_results: int, backend: str) -> None:
            raise error

    monkeypatch.setattr(ddgs, "DDGS", Client)

    with pytest.raises(SearchProviderError) as caught:
        await DuckDuckGoSearchProvider().search("test query", max_results=3)

    assert str(caught.value) == message
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_unexpected_ddgs_exception_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "upstream request body detail"

    class Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def text(self, query: str, *, max_results: int, backend: str) -> None:
            raise RuntimeError(secret)

    monkeypatch.setattr(ddgs, "DDGS", Client)

    with pytest.raises(SearchProviderError) as caught:
        await DuckDuckGoSearchProvider().search("test query", max_results=3)

    assert str(caught.value) == "DuckDuckGo search request failed"
    assert secret not in str(caught.value)


@pytest.mark.asyncio
async def test_ddgs_cancellation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def text(self, query: str, *, max_results: int, backend: str) -> None:
            raise asyncio.CancelledError

    monkeypatch.setattr(ddgs, "DDGS", Client)

    with pytest.raises(asyncio.CancelledError):
        await DuckDuckGoSearchProvider().search("test query", max_results=3)


@pytest.mark.asyncio
async def test_ddgs_deadline_abandons_blocking_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def text(self, query: str, *, max_results: int, backend: str) -> list[object]:
            started.set()
            release.wait(timeout=1)
            return []

    monkeypatch.setattr(ddgs, "DDGS", Client)

    try:
        with pytest.raises(SearchProviderError, match="timed out"):
            await DuckDuckGoSearchProvider(timeout=0.01).search(
                "test query", max_results=3
            )
        assert started.is_set()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_ddgs_cancellation_abandons_blocking_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def text(self, query: str, *, max_results: int, backend: str) -> list[object]:
            started.set()
            release.wait(timeout=1)
            return []

    monkeypatch.setattr(ddgs, "DDGS", Client)
    task = asyncio.create_task(
        DuckDuckGoSearchProvider(timeout=1).search("test query", max_results=3)
    )
    try:
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.1)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_blocking_ddgs_client_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class Client:
        def __init__(self, *, timeout: float) -> None:
            pass

        def text(self, query: str, *, max_results: int, backend: str) -> list[object]:
            started.set()
            release.wait(timeout=1)
            return []

    monkeypatch.setattr(ddgs, "DDGS", Client)

    async def concurrent_task() -> str:
        await asyncio.sleep(0)
        return "responsive"

    search_task = asyncio.create_task(
        DuckDuckGoSearchProvider().search("test query", max_results=3)
    )
    try:
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        assert await asyncio.wait_for(concurrent_task(), timeout=0.1) == "responsive"
    finally:
        release.set()
        await search_task


def test_ddgs_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "chartreux.core.tools.search.duckduckgo"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("ddgs"):
            pytest.fail("ddgs must not be imported during provider discovery")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module(module_name)
