from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import json
import os
from pathlib import Path
import sys

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.schema import ClientCapabilities, Implementation, TextContentBlock
import pytest

from tests.acp.test_acp_entrypoint_smoke import _AcpSmokeClient
from tests.stubs.fake_delivery_network import MCP_KEY, PROVIDER_KEY, UNRELATED_KEY

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _RecordingAcpClient(_AcpSmokeClient):
    def __init__(self) -> None:
        self.notifications: list[tuple[str, object]] = []

    async def session_update(
        self, session_id: str, update: object, **kwargs: object
    ) -> None:
        del kwargs
        self.notifications.append((session_id, update))


def _isolated_environment(root: Path) -> dict[str, str]:
    env = {"PATH": os.defpath}

    home = root / "home"
    vibe_home = home / ".chartreux"
    vibe_home.mkdir(parents=True)
    (vibe_home / "config.toml").write_text("", encoding="utf-8")
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
        "MISTRAL_API_KEY": "synthetic-provider-credential",
    })
    return env


def test_acp_child_environment_does_not_inherit_credentials(
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
    assert env["MISTRAL_API_KEY"] == "synthetic-provider-credential"
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.fail.Keyring"


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        if proc.stdin is not None and not proc.stdin.is_closing():
            proc.stdin.close()
        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(proc.wait(), timeout=10)
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=5)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_acp_initialize_new_session_and_eof_cleanup_in_fresh_process(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic"
    root.mkdir()
    network_guard = root / "network-attempt"
    env = _isolated_environment(root)
    env["TEST_ACP_NETWORK_GUARD"] = str(network_guard)
    home = Path(env["HOME"]) / ".chartreux"
    session_dir = root / "sessions"
    (home / "config.toml").write_text(
        "[session_logging]\n"
        "enabled = true\n"
        f"save_dir = {json.dumps(str(session_dir))}\n",
        encoding="utf-8",
    )

    code = """
import os
from pathlib import Path
from tests.stubs.fake_delivery_network import fake_delivery_network

network_guard = Path(os.environ["TEST_ACP_NETWORK_GUARD"])
assert "OTHER_PROVIDER_API_KEY" not in os.environ

with fake_delivery_network(network_guard, "https://api.mistral.ai/v1/chat/completions"):
    from chartreux.acp.entrypoint import main

    main()
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        cwd=root,
        env=env,
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
        stderr=aio_subprocess.PIPE,
    )
    conn = None
    client = _RecordingAcpClient()
    stderr = b""
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        conn = connect_to_agent(client, proc.stdin, proc.stdout)

        initialize = await asyncio.wait_for(
            conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(
                    name="chartreux-test", title="Chartreux Test", version="0"
                ),
            ),
            timeout=10,
        )
        assert initialize.protocol_version == PROTOCOL_VERSION
        assert initialize.agent_info is not None
        assert initialize.agent_info.name == "chartreux"

        session = await asyncio.wait_for(
            conn.new_session(cwd=str(root), mcp_servers=[]), timeout=10
        )
        assert session.session_id
        assert proc.returncode is None
        assert not network_guard.exists(), network_guard.read_text()
        turn = await asyncio.wait_for(
            conn.prompt(
                session_id=session.session_id,
                prompt=[TextContentBlock(type="text", text="Respond briefly")],
            ),
            timeout=10,
        )
        assert turn.stop_reason == "end_turn"
        updates = [
            update
            for session_id, update in client.notifications
            if session_id == session.session_id
        ]
        assert any(
            getattr(update, "session_update", None) == "agent_message_chunk"
            and getattr(getattr(update, "content", None), "text", None)
            == "Recorded answer."
            for update in updates
        )
        notification_text = repr(client.notifications)
        for sentinel in (PROVIDER_KEY, MCP_KEY, UNRELATED_KEY):
            assert sentinel not in notification_text

        proc.stdin.close()
        await asyncio.wait_for(proc.wait(), timeout=10)
    finally:
        await _terminate_process(proc)
        if conn is not None:
            await asyncio.wait_for(conn.close(), timeout=5)
        if proc.stderr is not None:
            stderr = await asyncio.wait_for(proc.stderr.read(), timeout=5)

    assert proc.returncode == 0, stderr.decode(errors="replace")
    records = [json.loads(line) for line in network_guard.read_text().splitlines()]
    assert records == [{"kind": "provider", "operation": "chat/completions"}]
    assert b"synthetic-provider-credential" not in stderr
    logs = list((home / "logs").glob("*.log"))
    transcripts = list(session_dir.rglob("*.jsonl"))
    assert logs and transcripts
    persisted = b"".join(path.read_bytes() for path in [*logs, *transcripts])
    assert b"Recorded answer." in persisted
    for sentinel in (PROVIDER_KEY, MCP_KEY, UNRELATED_KEY):
        assert sentinel.encode() not in persisted
