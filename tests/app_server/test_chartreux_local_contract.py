from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any
import urllib.request

import httpx
import pytest
import tomli_w

from chartreux.app_server.protocol import ClientCapabilities, SessionOptions
from chartreux.core.llm_models import FunctionCall, ToolCall
from tests.app_server.backend_contract.conftest import connect_backend_contract_host
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AttemptRecorder:
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def reject_socket(self, _socket: socket.socket, address: object) -> None:
        self.attempts.append(f"socket.connect:{address!r}")
        raise OSError("network denied by contract test")

    def reject_connect_ex(self, _socket: socket.socket, address: object) -> int:
        self.attempts.append(f"socket.connect_ex:{address!r}")
        return 101

    def reject_dns(self, host: object, *args: object, **kwargs: object) -> object:
        self.attempts.append(f"socket.dns:{host!r}")
        raise OSError("DNS denied by contract test")

    async def reject_async_http(
        self,
        _client: httpx.AsyncClient,
        method: object,
        url: object,
        *args: object,
        **kwargs: object,
    ) -> httpx.Response:
        del args, kwargs
        self.attempts.append(f"http.async:{method} {url}")
        raise OSError("HTTP denied by contract test")

    async def reject_async_send(
        self,
        _client: httpx.AsyncClient,
        request: httpx.Request,
        *args: object,
        **kwargs: object,
    ) -> httpx.Response:
        del args, kwargs
        self.attempts.append(f"http.async.send:{request.method} {request.url}")
        raise OSError("HTTP denied by contract test")

    def reject_sync_http(
        self,
        _client: httpx.Client,
        method: object,
        url: object,
        *args: object,
        **kwargs: object,
    ) -> httpx.Response:
        del args, kwargs
        self.attempts.append(f"http.sync:{method} {url}")
        raise OSError("HTTP denied by contract test")

    def reject_sync_send(
        self,
        _client: httpx.Client,
        request: httpx.Request,
        *args: object,
        **kwargs: object,
    ) -> httpx.Response:
        del args, kwargs
        self.attempts.append(f"http.sync.send:{request.method} {request.url}")
        raise OSError("HTTP denied by contract test")

    def reject_urlopen(self, url: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.attempts.append(f"urllib:{url!r}")
        raise OSError("HTTP denied by contract test")


def _bind_denied_service_capture(
    monkeypatch: pytest.MonkeyPatch, recorder: _AttemptRecorder
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", recorder.reject_dns)
    monkeypatch.setattr(socket, "gethostbyname", recorder.reject_dns)
    monkeypatch.setattr(socket, "gethostbyaddr", recorder.reject_dns)
    monkeypatch.setattr(socket, "getnameinfo", recorder.reject_dns)
    monkeypatch.setattr(socket.socket, "connect", recorder.reject_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", recorder.reject_connect_ex)
    monkeypatch.setattr(httpx.AsyncClient, "request", recorder.reject_async_http)
    monkeypatch.setattr(httpx.AsyncClient, "send", recorder.reject_async_send)
    monkeypatch.setattr(httpx.Client, "request", recorder.reject_sync_http)
    monkeypatch.setattr(httpx.Client, "send", recorder.reject_sync_send)
    monkeypatch.setattr(urllib.request, "urlopen", recorder.reject_urlopen)


def _isolated_environment(root: Path) -> dict[str, str]:
    env = {"PATH": os.defpath}

    home = root / "home"
    env.update({
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_STATE_HOME": str(home / "state"),
        "PYTHONPATH": str(_PROJECT_ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        "CHARTREUX_TEST_DISABLE_KEYRING": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "LANG": "C.UTF-8",
    })
    return env


def test_stdio_child_environment_does_not_inherit_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "OTHER_PROVIDER_API_KEY",
        "HTTPS_PROXY",
        "CHARTREUX_HOME",
        "DEBUG_MODE",
    ):
        monkeypatch.setenv(name, "synthetic-parent-value")

    env = _isolated_environment(tmp_path)

    assert "synthetic-parent-value" not in env.values()
    assert env["HOME"] == str(tmp_path / "home")
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.fail.Keyring"


def _run_child(
    *, code: str, env: dict[str, str], cwd: Path, input_data: bytes, timeout: float = 15
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        raise AssertionError(
            f"fresh app-server interpreter did not exit within {timeout}s; "
            f"stdout={stdout!r}, stderr={stderr!r}"
        ) from exc
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    return subprocess.CompletedProcess(
        process.args, process.returncode, stdout=stdout, stderr=stderr
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "backend_name"), [("local", "generic"), ("mistral", "mistral")]
)
async def test_lifecycle_makes_no_product_service_attempts_with_fake_inference(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    backend_name: str,
    tmp_path: Path,
) -> None:
    """Exercise retained lifecycle edges while distinguishing inference from services.

    The fake backend is installed before the harness is constructed, and all
    network seams are bound before startup.  A denied call is recorded even if
    the product catches the resulting exception; the final assertion therefore
    does not mistake blocked egress for proof of no attempt.
    """
    session_root = tmp_path / "sessions"
    model_alias = f"{provider_name}-model"
    provider = {
        "name": provider_name,
        "api_base": (
            "https://api.mistral.ai/v1"
            if provider_name == "mistral"
            else "http://127.0.0.1:9/v1"
        ),
        "api_key_env_var": (
            "MISTRAL_API_KEY" if provider_name == "mistral" else "LOCAL_API_KEY"
        ),
        "backend": backend_name,
    }
    (config_dir / "config.toml").write_text(
        tomli_w.dumps({
            "active_model": model_alias,
            "session_logging": {"enabled": True, "save_dir": str(session_root)},
        }),
        encoding="utf-8",
    )
    provider_id = f"{provider_name}/default"
    (config_dir / "models.toml").write_text(
        tomli_w.dumps({
            "providers": {
                provider_id: {
                    key: value for key, value in provider.items() if key != "name"
                }
            },
            "models": {
                model_alias: {
                    "deployments": [{"provider": provider_id, "name": model_alias}]
                }
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv(provider["api_key_env_var"], "synthetic-provider-credential")

    target = tmp_path / "tool-input.txt"
    target.write_text("tool result", encoding="utf-8")
    tool_call = ToolCall(
        id="read-1",
        index=0,
        function=FunctionCall(
            name="read_file", arguments=json.dumps({"file_path": str(target)})
        ),
    )
    fake_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[tool_call])],
        [mock_llm_chunk(content="streamed answer")],
    ])

    # These bindings deliberately precede host construction/startup.
    recorder = _AttemptRecorder()
    _bind_denied_service_capture(monkeypatch, recorder)

    def fake_create_backend(*args: object, **kwargs: object) -> FakeBackend:
        del args, kwargs
        return fake_backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", fake_create_backend
    )

    connection = await connect_backend_contract_host(
        session_options=SessionOptions(cwd=str(tmp_path), trust_workspace=True),
        capabilities=ClientCapabilities(),
    )
    session = await connection.host.open_session()
    session_id = session.session_id

    def assert_no_service_attempts(stage: str) -> None:
        assert recorder.attempts == [], f"{stage}: {recorder.attempts!r}"

    try:
        assert session.resources.config.current.active_model.alias == model_alias
        assert_no_service_attempts("startup")

        events = [event async for event in session.act("read the file")]
        assert events
        assert fake_backend.requests_messages
        assert_no_service_attempts("streamed tool turn")

        await session.resources.config.reload(reload_runtime=True)
        assert_no_service_attempts("reload")

        await session.resources.loops.create("30s", "idle background check")
        assert await session.resources.loops.list()
        assert_no_service_attempts("idle/background")

        fork = await session.resources.sessions.fork(attach=False)
        assert fork.source_session_id == session_id
        assert_no_service_attempts("child creation")

        await session.close()
        resumed_connection = await connect_backend_contract_host(
            session_options=SessionOptions(cwd=str(tmp_path), trust_workspace=True),
            capabilities=ClientCapabilities(),
        )
        resumed = await resumed_connection.host.resume_session(session_id)
        try:
            assert resumed.session_id == session_id
            assert_no_service_attempts("resume")
        finally:
            await resumed.close()
            await resumed_connection.host.close()
    finally:
        await connection.host.close()

    assert_no_service_attempts("shutdown")


@pytest.mark.timeout(30)
def test_stdio_startup_uses_fresh_synthetic_home_and_closes_on_eof(
    tmp_path: Path,
) -> None:
    prior_home = tmp_path / "prior-home"
    prior_vibe = prior_home / ".chartreux"
    prior_vibe.mkdir(parents=True)
    sentinel = prior_vibe / "sentinel.txt"
    sentinel.write_bytes(b"do not modify\n")
    prior_snapshot = {
        path.relative_to(prior_home): path.read_bytes()
        for path in prior_home.rglob("*")
        if path.is_file()
    }

    root = tmp_path / "synthetic"
    root.mkdir()
    network_guard = root / "network-attempt"
    env = _isolated_environment(root)
    env["CHARTREUX_TEST_NETWORK_GUARD"] = str(network_guard)
    env["CHARTREUX_TEST_PRIOR_HOME"] = str(prior_home)

    code = """
import os
import socket
import sys
from pathlib import Path

network_guard = Path(os.environ["CHARTREUX_TEST_NETWORK_GUARD"])
prior_home = Path(os.environ["CHARTREUX_TEST_PRIOR_HOME"])
assert Path.home() == Path(os.environ["HOME"])
assert Path.home() != prior_home
assert "MISTRAL_API_KEY" not in os.environ


def deny_network(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"} or (
        event in {"socket.connect", "socket.sendto", "socket.sendmsg"}
        and args[0].family in {socket.AF_INET, socket.AF_INET6}
    ):
        network_guard.write_text(event, encoding="utf-8")
        raise AssertionError(f"unexpected network operation: {event}")


sys.addaudithook(deny_network)

from chartreux.app_server.entrypoint import main

main()
"""
    initialize = {
        "jsonrpc": "2.0",
        "id": "initialize",
        "method": "initialize",
        "params": {
            "clientInfo": {"name": "chartreux-test", "version": "0"},
            "capabilities": {},
        },
    }
    result = _run_child(
        code=code,
        env=env,
        cwd=root,
        input_data=(json.dumps(initialize) + "\n").encode(),
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert not network_guard.exists(), result.stderr.decode(errors="replace")
    lines = result.stdout.splitlines()
    assert len(lines) == 1, result.stdout.decode(errors="replace")
    response: dict[str, Any] = json.loads(lines[0])
    assert response["id"] == "initialize"
    assert response["result"]["serverInfo"]["name"] == "chartreux-app-server"
    assert result.stderr == b"" or b"MISTRAL_API_KEY" not in result.stderr

    assert {
        path.relative_to(prior_home): path.read_bytes()
        for path in prior_home.rglob("*")
        if path.is_file()
    } == prior_snapshot
    assert (root / "home" / ".chartreux" / "logs" / "chartreux.log").is_file()
