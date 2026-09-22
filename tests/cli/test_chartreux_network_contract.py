from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomli_w

from tests.app_server.test_chartreux_local_contract import (
    _isolated_environment,
    _run_child,
)
from tests.stubs.fake_delivery_network import (
    MCP_KEY,
    MCP_URL,
    PROVIDER_KEY,
    UNRELATED_KEY,
)


@pytest.mark.parametrize("backend", ["generic", "mistral"])
@pytest.mark.timeout(45)
def test_cli_real_provider_and_direct_mcp_without_product_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.setenv("OTHER_PROVIDER_API_KEY", "synthetic-parent-only")
    env = _isolated_environment(tmp_path)
    key_variable = "MISTRAL_API_KEY" if backend == "mistral" else "LOCAL_API_KEY"
    env[key_variable] = PROVIDER_KEY
    home = Path(env["HOME"]) / ".chartreux"
    home.mkdir(parents=True)
    api_base = (
        "https://api.mistral.ai/v1"
        if backend == "mistral"
        else "https://provider.invalid/v1"
    )
    (home / "models.toml").write_text(
        tomli_w.dumps({
            "providers": {
                "contract/default": {
                    "api_base": api_base,
                    "api_key_env_var": key_variable,
                    "api_style": "openai",
                    "backend": backend,
                }
            },
            "models": {
                "contract-model": {
                    "deployments": [
                        {"provider": "contract/default", "name": "contract-model"}
                    ]
                }
            },
            "roles": {},
        }),
        encoding="utf-8",
    )
    (home / "config.toml").write_text(
        tomli_w.dumps({
            "active_model": "contract-model",
            "mcp_servers": [
                {
                    "name": "direct",
                    "transport": "streamable-http",
                    "url": MCP_URL,
                    "auth": {
                        "type": "static",
                        "headers": {"Authorization": f"Bearer {MCP_KEY}"},
                    },
                }
            ],
            "session_logging": {"enabled": True, "save_dir": str(home / "sessions")},
        }),
        encoding="utf-8",
    )
    attempts = tmp_path / "attempts.jsonl"
    code = f"""
import os
import sys
from pathlib import Path
from tests.stubs.fake_delivery_network import fake_delivery_network
assert "OTHER_PROVIDER_API_KEY" not in os.environ
with fake_delivery_network(Path({str(attempts)!r}), {api_base + "/chat/completions"!r}):
    from chartreux.cli.entrypoint import main
    sys.argv = ["chartreux", "--trust", "--prompt", "Respond briefly", "--output", "streaming"]
    main()
"""
    for resume in (False, True):
        invocation = code
        if resume:
            invocation = code.replace(
                "    main()", '    sys.argv.append("--continue")\n    main()'
            )
        result = _run_child(
            code=invocation, env=env, cwd=tmp_path, input_data=b"", timeout=20
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert b"Recorded answer." in result.stdout
        entries = [json.loads(line) for line in result.stdout.splitlines()]
        assert all(isinstance(entry, dict) for entry in entries)
        assert sum(entry.get("role") == "assistant" for entry in entries) == (
            2 if resume else 1
        )
        assert PROVIDER_KEY.encode() not in result.stdout + result.stderr
        assert MCP_KEY.encode() not in result.stdout + result.stderr
    records = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert not [item for item in records if item["kind"] == "denied"], records
    assert sum(item["kind"] == "provider" for item in records) == 2
    assert {item["operation"] for item in records if item["kind"] == "mcp"} >= {
        "initialize",
        "tools/list",
    }
    logs = list((home / "logs").glob("*.log"))
    transcripts = list((home / "sessions").rglob("*.jsonl"))
    assert logs and transcripts
    for path in [*logs, *transcripts]:
        contents = path.read_bytes()
        assert PROVIDER_KEY.encode() not in contents
        assert MCP_KEY.encode() not in contents
        assert UNRELATED_KEY.encode() not in contents


def test_fake_delivery_network_rejects_unrelated_credential_header(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts.jsonl"
    code = f"""
import httpx
from pathlib import Path
from tests.stubs.fake_delivery_network import fake_delivery_network
with fake_delivery_network(Path({str(attempts)!r}), "https://api.mistral.ai/v1/chat/completions"):
    try:
        httpx.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={{
                "Authorization": "Bearer {PROVIDER_KEY}",
                "X-Unrelated": {UNRELATED_KEY!r},
            }},
        )
    except AssertionError as exc:
        assert "headers" in str(exc)
    else:
        raise AssertionError("the fake transport accepted a credential leak")
"""
    result = _run_child(
        code=code, env=_isolated_environment(tmp_path), cwd=tmp_path, input_data=b""
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert not attempts.exists()


@pytest.mark.parametrize(
    "operation",
    ["dns", "tcp", "connect_ex", "udp", "sendmsg", "http", "async_http", "urllib"],
)
def test_recording_detects_caught_intentional_network_probe(
    tmp_path: Path, operation: str
) -> None:
    attempts = tmp_path / "attempts.jsonl"
    code = f"""
import asyncio
import socket
import urllib.request
from pathlib import Path
import httpx
from tests.stubs.fake_delivery_network import fake_delivery_network
with fake_delivery_network(Path({str(attempts)!r}), "https://api.mistral.ai/v1/chat/completions"):
    try:
        operation = {operation!r}
        if operation == "dns":
            socket.getaddrinfo("product.invalid", 443)
        elif operation in {{"tcp", "connect_ex"}}:
            with socket.socket() as sock:
                getattr(sock, "connect" if operation == "tcp" else "connect_ex")(("192.0.2.1", 443))
        elif operation in {{"udp", "sendmsg"}}:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                if operation == "udp":
                    sock.sendto(b"probe", ("192.0.2.1", 53))
                else:
                    sock.sendmsg([b"probe"], [], 0, ("192.0.2.1", 53))
        elif operation == "http":
            httpx.get("https://api.mistral.ai/v1/account")
        elif operation == "async_http":
            async def probe():
                async with httpx.AsyncClient() as client:
                    await client.get("https://api.mistral.ai/v1/account")
            asyncio.run(probe())
        else:
            urllib.request.urlopen("https://product.invalid/account", timeout=1)
    except OSError:
        pass  # Simulate a product service swallowing a connection error.
"""
    result = _run_child(
        code=code, env=_isolated_environment(tmp_path), cwd=tmp_path, input_data=b""
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    records = [json.loads(line) for line in attempts.read_text().splitlines()]
    expected = {
        "dns": "socket.getaddrinfo",
        "tcp": "socket.connect",
        "connect_ex": "socket.connect",
        "udp": "socket.sendto",
        "sendmsg": "socket.sendmsg",
        "http": "HTTP GET https://api.mistral.ai/v1/account",
        "async_http": "HTTP GET https://api.mistral.ai/v1/account",
        "urllib": "socket.getaddrinfo",
    }
    assert records == [{"kind": "denied", "operation": expected[operation]}]
