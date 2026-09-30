from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import tomllib

import pytest

from chartreux.app_server._web_search_settings import project_web_search_settings
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientInfo,
    ConfigWriteOpWire,
    ConfigWriteParams,
    SettingsReadParams,
    SettingsReadResponse,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@asynccontextmanager
async def opened(
    path: Path, *, inherited: dict | None = None
) -> AsyncIterator[tuple[AgentLoop, AppServerClient]]:
    user = UserConfigLayer(path=path, name="user-test")
    session = OverridesLayer(data={}, name="session-test")
    layers: list[ConfigLayer[RawConfig]] = [
        DefaultConfigLayer(schema=ChartreuxConfigSchema)
    ]
    if inherited is not None:
        layers.append(OverridesLayer(data=inherited, name="inherited-test"))
    layers.extend((user, session))
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=layers,
        default_layer_resolver=lambda: session,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    a, b = memory_transport_pair()
    server = build_test_app_server(loop, b)
    client = AppServerClient(a, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="search-settings-test", version="1"))
        await client.notify("initialized")
        from chartreux.app_server.protocol import SessionStartParams

        await client.request("session/start", SessionStartParams())
        yield loop, client
    finally:
        await a.close()
        await b.close()
        await loop.aclose()


async def read(loop: AgentLoop, client: AppServerClient) -> SettingsReadResponse:
    return SettingsReadResponse.model_validate(
        await client.request(
            "config/settings/read", SettingsReadParams(session_id=loop.session_id)
        )
    )


async def write(
    loop: AgentLoop, client: AppServerClient, revision: str, *ops: ConfigWriteOpWire
) -> dict:
    return await client.request(
        "config/write",
        ConfigWriteParams(
            session_id=loop.session_id,
            target="user",
            expected_revision=revision,
            ops=list(ops),
        ),
    )


@pytest.mark.asyncio
async def test_safe_read_pairs_supported_leaves_with_user_revision_and_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[tools.web_search]\nprovider = "exa"\nmax_results = 7\n')
    monkeypatch.setattr(
        "chartreux.core.tools.builtins.web_search.resolve_api_key",
        lambda name: "key-secret" if name == "EXA_API_KEY" else None,
    )
    async with opened(path, inherited={"tools": {"web_search": {"timeout": 42}}}) as (
        loop,
        client,
    ):
        snapshot = await read(loop, client)
        search = snapshot.web_search
        assert search is not None
        assert search.readiness == "ready"
        assert search.credential_env_var == "EXA_API_KEY"
        assert search.default_credential_env_vars["brave"] == "BRAVE_SEARCH_API_KEY"
        assert snapshot.user_revision
        fields = {field.path: field for field in search.fields}
        assert set(fields) == {
            f"tools.web_search.{name}"
            for name in (
                "permission",
                "provider",
                "api_key_env_var",
                "base_url",
                "timeout",
                "max_results",
                "model",
            )
        }
        assert fields["tools.web_search.provider"].origin == "user-test"
        assert fields["tools.web_search.provider"].saved_value == "exa"
        assert fields["tools.web_search.timeout"].origin == "inherited-test"
        assert not fields["tools.web_search.timeout"].saved_explicit
        assert fields["tools.web_search.permission"].origin == "default"
        assert "key-secret" not in snapshot.model_dump_json()


@pytest.mark.asyncio
async def test_malformed_nested_search_values_and_url_credentials_are_not_reflected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[tools.web_search]\nprovider = { token = "nested-secret" }\n'
        'api_key_env_var = { token = "env-secret" }\n'
        'base_url = "https://user:url-secret@example.com/search?key=query-secret"\n'
    )
    async with opened(path) as (loop, client):
        snapshot = await read(loop, client)
        search = snapshot.web_search
        assert search is not None
        assert search.readiness == "invalid"
        assert set(search.invalid_fields) == {
            "tools.web_search.provider",
            "tools.web_search.api_key_env_var",
            "tools.web_search.base_url",
        }
        fields = {field.path: field for field in search.fields}
        assert fields["tools.web_search.provider"].effective_value == "[invalid]"
        assert fields["tools.web_search.provider"].saved_value == "[invalid]"
        assert fields["tools.web_search.api_key_env_var"].saved_value == "[invalid]"
        assert fields["tools.web_search.base_url"].saved_value == "[redacted]"
        wire = snapshot.model_dump_json()
        for secret in ("nested-secret", "env-secret", "url-secret", "query-secret"):
            assert secret not in wire


def test_hidden_invalid_lower_layer_does_not_require_repair() -> None:
    config = ChartreuxConfigSchema(
        tools={"web_search": {"provider": "duckduckgo", "base_url": ""}}
    )
    search = project_web_search_settings(
        config,
        [
            (
                "inherited",
                {"tools": {"web_search": {"base_url": "https://u:secret@example.com"}}},
            ),
            ("user", {"tools": {"web_search": {"base_url": ""}}}),
        ],
        user_layer="user",
        user_unavailable=False,
        fallback=False,
    )
    assert search.invalid_fields == []
    assert "secret" not in search.model_dump_json()


@pytest.mark.parametrize(
    ("field", "value", "label"),
    [("max_results", 99, "maximum results"), ("timeout", 0, "timeout")],
)
def test_invalid_scalar_is_marked_with_safe_field_reason(
    field: str, value: int, label: str
) -> None:
    config = ChartreuxConfigSchema(tools={"web_search": {field: value}})
    search = project_web_search_settings(
        config,
        [("user", {"tools": {"web_search": {field: value}}})],
        user_layer="user",
        user_unavailable=False,
        fallback=False,
    )
    assert search.readiness == "invalid"
    assert search.invalid_fields == [f"tools.web_search.{field}"]
    assert search.readiness_message == (
        f"Invalid web search {label}; review that setting."
    )


@pytest.mark.asyncio
async def test_leaf_write_validates_atomically_and_preserves_other_tools(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[tools.web_search]\nprovider = "exa"\npermission = "always"\n'
        '[tools.bash]\npermission = "never"\n'
    )
    async with opened(path) as (loop, client):
        revision = (await read(loop, client)).user_revision
        assert revision
        bad = await write(
            loop,
            client,
            revision,
            ConfigWriteOpWire(op="set", path="/tools/web_search/provider", value="bad"),
            ConfigWriteOpWire(op="set", path="/tools/web_search/max_results", value=10),
        )
        assert bad["persistence"] == "not_saved"
        assert bad["rejected"]
        assert tomllib.loads(path.read_text())["tools"]["web_search"] == {
            "provider": "exa",
            "permission": "always",
        }
        saved = await write(
            loop,
            client,
            revision,
            ConfigWriteOpWire(
                op="set", path="/tools/web_search/provider", value="brave"
            ),
            ConfigWriteOpWire(op="set", path="/tools/web_search/max_results", value=10),
        )
        assert saved["persistence"] == "saved"
        assert saved["application"] == "applied"
        persisted = tomllib.loads(path.read_text())["tools"]
        assert persisted["web_search"] == {
            "provider": "brave",
            "permission": "always",
            "max_results": 10,
        }
        assert persisted["bash"] == {"permission": "never"}
        stale = await write(
            loop,
            client,
            revision,
            ConfigWriteOpWire(op="set", path="/tools/web_search/provider", value="exa"),
        )
        assert stale["persistence"] == "not_saved"
        assert stale["failures"] == ["conflict"]


@pytest.mark.asyncio
async def test_empty_string_reset_shadows_inherited_credentials_and_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[tools.web_search]\nprovider = "exa"\n')
    monkeypatch.setattr(
        "chartreux.core.tools.builtins.web_search.resolve_api_key",
        lambda name: "brave-secret" if name == "BRAVE_SEARCH_API_KEY" else None,
    )
    inherited = {
        "tools": {
            "web_search": {
                "api_key_env_var": "OLD_EXA_KEY",
                "base_url": "https://old.example",
            }
        }
    }
    async with opened(path, inherited=inherited) as (loop, client):
        before = await read(loop, client)
        assert before.web_search is not None
        assert before.web_search.credential_env_var == "OLD_EXA_KEY"
        assert before.user_revision
        saved = await write(
            loop,
            client,
            before.user_revision,
            ConfigWriteOpWire(
                op="set", path="/tools/web_search/provider", value="brave"
            ),
            ConfigWriteOpWire(
                op="set", path="/tools/web_search/api_key_env_var", value=""
            ),
            ConfigWriteOpWire(op="set", path="/tools/web_search/base_url", value=""),
        )
        assert saved["persistence"] == "saved"
        assert saved["application"] == "applied"
        after = await read(loop, client)
        assert after.web_search is not None
        assert after.web_search.credential_env_var == "BRAVE_SEARCH_API_KEY"
        assert after.web_search.readiness == "ready"
        fields = {field.path: field for field in after.web_search.fields}
        for name in ("api_key_env_var", "base_url"):
            field = fields[f"tools.web_search.{name}"]
            assert field.effective_value == ""
            assert field.saved_explicit and field.saved_value == ""
            assert field.origin == "user-test"
        persisted = tomllib.loads(path.read_text())["tools"]["web_search"]
        assert persisted["api_key_env_var"] == ""
        assert persisted["base_url"] == ""
        assert "brave-secret" not in after.model_dump_json()


@pytest.mark.asyncio
async def test_saved_source_and_failed_runtime_application_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[tools.web_search]\nprovider = "duckduckgo"\n')
    async with opened(path) as (loop, client):
        before = await read(loop, client)
        assert before.user_revision

        def fail_application(*_args: object) -> None:
            raise RuntimeError("runtime application failed")

        monkeypatch.setattr(loop, "_commit_reload", fail_application)
        result = await write(
            loop,
            client,
            before.user_revision,
            ConfigWriteOpWire(op="set", path="/tools/web_search/max_results", value=8),
        )
        assert result["persistence"] == "saved"
        assert result["application"] == "failed"
        assert result["revision"] != before.user_revision
        assert (
            tomllib.loads(path.read_text())["tools"]["web_search"]["max_results"] == 8
        )
        after = await read(loop, client)
        assert after.user_revision == result["revision"]
        assert after.web_search is not None
        field = next(
            item
            for item in after.web_search.fields
            if item.path == "tools.web_search.max_results"
        )
        assert field.saved_explicit and field.saved_value == 8
        assert field.effective_value == 8
