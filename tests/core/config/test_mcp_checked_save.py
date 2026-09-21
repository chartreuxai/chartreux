from __future__ import annotations

import os
from pathlib import Path
import stat
import tomllib
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from chartreux.core.config import ChartreuxConfigSchema, MCPStdio
from chartreux.core.config._mcp_save import MCPSaveError
from chartreux.core.config._restrictions import ConfigCandidate
from chartreux.core.config.layers import _base
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.mcp_servers import (
    PersistedMCPServerResult,
    persist_oauth_mcp_server,
    persist_stdio_mcp_server,
    remove_mcp_server,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.tools.mcp_settings import persist_mcp_toggle

type Operation = Literal["add", "oauth", "remove", "toggle", "tool"]


async def _setup(path: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    path.write_text(
        '[[mcp_servers]]\nname="existing"\ntransport="stdio"\ncommand="unused"\n'
    )
    user = UserConfigLayer(path=path)
    session = OverridesLayer(data={"theme": "dark"})
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[user, session],
        default_layer_resolver=lambda: session,
    )


async def _change(orchestrator, operation: Operation, *, preflight=None, apply=None):
    match operation:
        case "add":
            return await persist_stdio_mcp_server(
                orchestrator,
                MCPStdio(name="added", transport="stdio", command="unused"),
                preflight=preflight,
                apply=apply,
            )
        case "oauth":
            return await persist_oauth_mcp_server(
                orchestrator,
                url="https://example.invalid/mcp",
                scopes=["read"],
                preflight=preflight,
                apply=apply,
            )
        case "remove":
            return await remove_mcp_server(
                orchestrator, "existing", preflight=preflight, apply=apply
            )
        case "toggle" | "tool":
            return await persist_mcp_toggle(
                orchestrator,
                name="existing",
                disabled=True,
                tool_name="read" if operation == "tool" else None,
                candidate_preflight=preflight,
                apply=apply,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "oauth", "remove", "toggle", "tool"])
@pytest.mark.parametrize(
    "failure", ["external", "race", "replace", "preflight", "fsync", "apply"]
)
async def test_checked_mcp_save_boundaries(tmp_path, monkeypatch, operation, failure):
    path = tmp_path / "config.toml"
    orchestrator = await _setup(path)
    before = orchestrator.config
    token = orchestrator.accepted_token
    caches = [(layer.cached_data, layer.fingerprint) for layer in orchestrator.layers]
    disk = path.read_bytes()
    external = disk + b"\n# external edit\n"
    calls = []

    async def prepare(candidate: ConfigCandidate[ChartreuxConfigSchema]):
        calls.append("prepare")
        assert candidate.config is not before
        assert orchestrator.config is before
        if failure == "race":
            path.write_bytes(external)
        if failure == "preflight":
            raise ValueError("private-preflight-value")

    def apply(candidate: ConfigCandidate[ChartreuxConfigSchema]):
        calls.append("apply")
        assert candidate.config is not before
        assert orchestrator.config is before
        if failure == "apply":
            raise ValueError("private-application-value")

    if failure == "external":
        path.write_bytes(external)
    if failure == "replace":

        def reject_replace(*args):
            raise OSError("private-write-value")

        monkeypatch.setattr(_base.os, "replace", reject_replace)
    if failure == "fsync":
        original_fsync = os.fsync

        def reject_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("private-durability-value")
            original_fsync(fd)

        monkeypatch.setattr(_base.os, "fsync", reject_directory)

    with pytest.raises(MCPSaveError) as error:
        await _change(orchestrator, operation, preflight=prepare, apply=apply)
    result = error.value.save_result
    assert result is not None
    assert "private-" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    if failure in {"fsync", "apply"}:
        assert path.read_bytes() != disk
        assert result.revision
        assert result.persistence == (
            "durability_uncertain" if failure == "fsync" else "saved"
        )
        assert result.application == ("applied" if failure == "fsync" else "failed")
        assert calls == ["prepare", "apply"]
    else:
        assert result.persistence == "not_saved"
        assert result.application == "unchanged"
        assert (
            result.error
            == {
                "external": "conflict",
                "race": "conflict",
                "replace": "write",
                "preflight": "validation",
            }[failure]
        )
        assert path.read_bytes() == (
            external if failure in {"external", "race"} else disk
        )
        assert "apply" not in calls
    if failure != "fsync":
        assert orchestrator.config is before
        assert orchestrator.accepted_token is token
        assert [
            (layer.cached_data, layer.fingerprint) for layer in orchestrator.layers
        ] == caches
    else:
        assert orchestrator.config is not before
    assert orchestrator.layers[1].cached_data == caches[1][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "oauth", "remove", "toggle", "tool"])
async def test_success_metadata_and_candidate_callbacks(tmp_path, operation):
    path = tmp_path / "config.toml"
    orchestrator = await _setup(path)
    calls = []

    async def prepare(candidate):
        calls.append("prepare")

    def apply(candidate):
        calls.append("apply")

    value = await _change(orchestrator, operation, preflight=prepare, apply=apply)
    assert value is not None
    result = value if isinstance(value, ConfigSaveResult) else value.save_result
    assert result is not None
    assert result.persistence == "saved"
    assert result.application == "applied"
    assert result.error is None
    assert result.revision == orchestrator.layers[0].fingerprint
    assert calls == ["prepare", "apply"]
    raw = tomllib.loads(path.read_text())["mcp_servers"]
    if operation == "oauth":
        assert raw[-1]["auth"] == {"type": "oauth", "scopes": ["read"]}
    if operation == "tool":
        assert raw[0]["disabled_tools"] == ["read"]
        await _change(orchestrator, operation)
        assert tomllib.loads(path.read_text())["mcp_servers"][0]["disabled_tools"] == [
            "read"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "oauth", "remove", "toggle", "tool"])
@pytest.mark.parametrize("source", ["missing", "lookalike", "unloaded", "child"])
async def test_no_invented_user_authority(tmp_path, monkeypatch, operation, source):
    path = tmp_path / "config.toml"
    orchestrator = await _setup(path)
    if source == "missing":
        orchestrator.remove_layer(0)
    elif source == "lookalike":
        replacement = OverridesLayer(data={})
        orchestrator.replace_or_append_layer("user-toml", replacement)
    elif source == "unloaded":
        orchestrator.replace_or_append_layer("user-toml", UserConfigLayer(path=path))
    else:
        orchestrator = orchestrator._copy_for_child()
    before = path.read_bytes()
    with pytest.raises(MCPSaveError) as error:
        await _change(orchestrator, operation)
    assert error.value.save_result == ConfigSaveResult(
        "user", "not_saved", "unchanged", error="validation"
    )
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_absent_file_is_an_accepted_revision(tmp_path):
    path = tmp_path / "new.toml"
    user = UserConfigLayer(path=path)
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema, layers=[user], default_layer_resolver=lambda: user
    )
    value = await _change(orchestrator, "add")
    assert isinstance(value, PersistedMCPServerResult)
    assert value.created
    assert path.is_file()


@pytest.mark.asyncio
async def test_absent_entries_are_noops(tmp_path, monkeypatch):
    orchestrator = await _setup(tmp_path / "config.toml")
    save = AsyncMock(side_effect=AssertionError("no save expected"))
    monkeypatch.setattr(orchestrator, "save", save)
    assert await persist_mcp_toggle(orchestrator, name="absent", disabled=True) is None
    removed = await remove_mcp_server(orchestrator, "absent")
    assert not removed.removed
    assert removed.save_result is None
    save.assert_not_called()
