"""Bounded child-process interception for the Chartreux delivery contracts.

HTTP is simulated, never allowlisted through to the network. The audit hook
records lower-level attempts before raising; OS descendant denial is supplied
separately by the acceptance runner, not by this Python-only hook.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import sys
from typing import Any

import httpx
import respx

PROVIDER_KEY = "synthetic-provider-credential"
MCP_KEY = "synthetic-mcp-credential"
UNRELATED_KEY = "synthetic-parent-only"
MCP_URL = "https://direct-mcp.invalid/mcp"


def _assert_credentials_absent(request: httpx.Request, *credentials: str) -> None:
    headers = b"\n".join(name + b"=" + value for name, value in request.headers.raw)
    for credential in credentials:
        needle = credential.encode()
        assert needle not in headers, (
            f"credential leaked in {request.method} {request.url} headers: {credential}"
        )
        assert needle not in request.content, (
            f"credential leaked in {request.method} {request.url} body: {credential}"
        )


def record(path: Path, kind: str, operation: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, "operation": operation}) + "\n")


def install_socket_capture(path: Path) -> None:
    """Install only in disposable interpreters: audit hooks cannot be removed."""

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
            "socket.getnameinfo",
        } or (
            event in {"socket.connect", "socket.sendto", "socket.sendmsg"}
            and args[0].family in {socket.AF_INET, socket.AF_INET6}
        ):
            record(path, "denied", event)
            raise OSError("network denied by recorded contract")

    sys.addaudithook(audit)


@contextmanager
def fake_delivery_network(path: Path, provider_url: str) -> Iterator[None]:
    install_socket_capture(path)

    def respond(request: httpx.Request) -> httpx.Response:
        # Exact method + URL matching: sharing a provider's domain does not
        # authorize account, telemetry, catalog or other product-service routes.
        if request.method == "POST" and str(request.url) == provider_url:
            assert request.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
            _assert_credentials_absent(request, MCP_KEY, UNRELATED_KEY)
            assert PROVIDER_KEY.encode() not in request.content
            record(path, "provider", "chat/completions")
            chunk = {
                "id": "contract",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "contract-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Recorded answer."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            )
        if request.method == "POST" and str(request.url) == MCP_URL:
            assert request.headers["authorization"] == f"Bearer {MCP_KEY}"
            _assert_credentials_absent(request, PROVIDER_KEY, UNRELATED_KEY)
            payload = json.loads(request.content)
            method = payload["method"]
            record(path, "mcp", method)
            if "id" not in payload:
                return httpx.Response(202)
            if method == "initialize":
                result = {
                    "protocolVersion": payload["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "contract-mcp", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": []}
            else:
                raise AssertionError(f"unexpected MCP method: {method}")
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result}
            )
        record(path, "denied", f"HTTP {request.method} {request.url}")
        raise OSError("HTTP denied by recorded contract")

    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=respond)
        yield
