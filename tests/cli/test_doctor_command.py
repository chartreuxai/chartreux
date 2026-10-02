from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import httpx
from mcp import ClientSession
import pytest

from chartreux.cli import doctor_command as doctor
from chartreux.core.config.chartreux_schema import (
    load_dotenv_values as _original_load_dotenv,
)
from chartreux.core.config.models import MCPHttp, MCPOAuth, MCPStaticAuth, MCPStdio
from chartreux.core.llm.provider_smoke import (
    CapabilityResult,
    ProviderSmokeResult,
    SmokeStatus,
)
from chartreux.core.model_catalog.contracts import DiscoveryError
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import BaseModelDefinition, ModelCatalog
from chartreux.core.tools.mcp.tools import list_tools_http as _original_mcp_listing


def _ambiguous_catalog(base: str, definition: BaseModelDefinition) -> ModelCatalog:
    first = definition.deployments[0].model_dump()
    second = {**first, "provider": "second-provider"}
    return ModelCatalog.model_validate({
        "providers": {
            first["provider"]: SHIPPED_CATALOG.providers[
                first["provider"]
            ].model_dump(),
            "second-provider": SHIPPED_CATALOG.providers[
                first["provider"]
            ].model_dump(),
        },
        "models": {base: {**definition.model_dump(), "deployments": [first, second]}},
    })


def invoke(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict]:
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli([*args, "--json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    assert isinstance(exc.value.code, int)
    return exc.value.code, json.loads(captured.out)


@pytest.fixture(autouse=True)
def forbid_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("chartreux.utils.api_keys.get_api_key_from_keyring", forbidden)
    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.load_dotenv_values", forbidden
    )
    monkeypatch.setattr("chartreux.core.tools.mcp.tools.list_tools_stdio", forbidden)
    monkeypatch.setattr("chartreux.core.tools.mcp.tools.list_tools_http", forbidden)
    monkeypatch.setattr(
        "chartreux.core.llm.provider_smoke.probe_provider_smoke", forbidden
    )
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("keyring.get_password", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


@pytest.fixture
def allow_dotenv_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.load_dotenv_values", Mock()
    )


@pytest.mark.parametrize("mode", ["bare", "live", "smoke"])
def test_dotenv_loading_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    from chartreux.utils.api_keys import resolve_api_key

    env = "DOCTOR_DOTENV_KEY"
    token = "synthetic-dotenv-secret"
    # Use a synthetic file, never the user's actual app dotenv file.
    env_path = tmp_path / "dotenv.txt"
    env_path.write_text(f"{env}={token}\n")
    env_path.chmod(0o644)
    monkeypatch.delenv(env, raising=False)
    config, checks = asyncio.run(doctor._load_config())
    snapshot = config.catalog_snapshot
    base, definition = next(iter(snapshot.catalog.models.items()))
    deployment = definition.deployments[0]
    provider = deployment.provider
    catalog = snapshot.catalog.model_copy(
        update={
            "providers": {
                **snapshot.catalog.providers,
                provider: snapshot.catalog.providers[provider].model_copy(
                    update={"api_key_env_var": env}
                ),
            }
        }
    )
    config.attach_catalog_snapshot(
        replace(snapshot, catalog=catalog, overlaid_providers=frozenset({provider}))
    )
    server = MCPHttp(
        transport="streamable-http",
        name="dotenv-static",
        url="https://example.invalid",
        auth=MCPStaticAuth(api_key_env=env),
    )
    config = config.model_copy(update={"mcp_servers": [server]})
    monkeypatch.setattr(
        doctor, "_load_config", AsyncMock(return_value=(config, checks))
    )
    load = Mock(side_effect=lambda: _original_load_dotenv(env_path=env_path))
    if mode != "bare":
        monkeypatch.setattr(
            "chartreux.core.config.chartreux_schema.load_dotenv_values", load
        )

    async def listing(*_args: object) -> doctor.Check:
        assert resolve_api_key(env) == token
        return doctor.Check("listing", "pass", "listed")

    async def probe(**_kwargs: object) -> ProviderSmokeResult:
        assert resolve_api_key(env) == token
        return ProviderSmokeResult(
            provider,
            deployment.name,
            base,
            CapabilityResult("pass", "ok"),
            CapabilityResult("unsupported", "no-thinking"),
            CapabilityResult("unsupported", "no-image"),
        )

    provider_listing = AsyncMock(side_effect=listing)
    smoke = AsyncMock(side_effect=probe)
    mcp_listing = AsyncMock(return_value=[])
    monkeypatch.setattr(doctor, "_listing", provider_listing)
    monkeypatch.setattr("chartreux.core.llm.provider_smoke.probe_provider_smoke", smoke)
    monkeypatch.setattr("chartreux.core.tools.mcp.tools.list_tools_http", mcp_listing)
    args = (
        ["--live"]
        if mode == "live"
        else ["--smoke", "--model", base, "--provider", provider]
        if mode == "smoke"
        else []
    )
    code, result = invoke(capsys, *args)
    assert code == 0
    credential = next(
        c for c in result["checks"] if c["name"] == f"credential {provider}"
    )
    assert token not in json.dumps(result)
    if mode == "bare":
        load.assert_not_called()
        assert credential["status"] == "unverified"
        assert "dotenv-configured, not inspected" in credential["reason"]
        assert env_path.stat().st_mode & 0o777 == 0o644
    else:
        load.assert_called_once_with()
        assert credential["status"] == "pass"
        assert credential["reason"] == "process-env credential present"
        assert env_path.stat().st_mode & 0o777 == 0o600
    if mode == "live":
        provider_listing.assert_awaited()
        mcp_listing.assert_awaited_once()
        assert mcp_listing.call_args.kwargs["headers"] == server.http_headers()
    else:
        provider_listing.assert_not_awaited()
        mcp_listing.assert_not_awaited()
    if mode == "smoke":
        smoke.assert_awaited_once()
    else:
        smoke.assert_not_awaited()


def test_loaded_dotenv_missing_credential_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCTOR_TEST_KEY", raising=False)
    check = doctor._credential("test", "DOCTOR_TEST_KEY", dotenv_loaded=True)
    assert check.status == "unverified"
    assert "dotenv loaded" in check.reason
    assert "dotenv-configured, not inspected" not in check.reason


def test_bare_local_checks_and_json_purity(capsys: pytest.CaptureFixture[str]) -> None:
    code, result = invoke(capsys)
    assert code == 0
    checks = {c["name"]: c for c in result["checks"]}
    for name in ("config", "catalog", "default model", "active model"):
        assert checks[name]["status"] == "pass"
    assert "project config" not in checks
    assert any(c["status"] == "skipped" for c in checks.values())


def test_credential_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCTOR_TEST_KEY", raising=False)
    missing = doctor._credential("test", "DOCTOR_TEST_KEY")
    assert missing.status == "unverified"
    assert "keyring-configured, not inspected" in missing.reason
    assert "dotenv-configured, not inspected" in missing.reason
    monkeypatch.setenv("DOCTOR_TEST_KEY", "secret-never-rendered")
    present = doctor._credential("test", "DOCTOR_TEST_KEY")
    assert present.status == "pass"
    assert "secret-never-rendered" not in repr(present)
    assert doctor._credential("local", "").status == "pass"


@pytest.mark.parametrize("file", ["config.toml", "models.toml"])
@pytest.mark.parametrize("contents", ["[broken", 'mcp_servers = "SECRET_BAD_VALUE"'])
def test_invalid_sources_sanitized(
    config_dir: Path, capsys: pytest.CaptureFixture[str], file: str, contents: str
) -> None:
    (config_dir / file).write_text(contents)
    code, result = invoke(capsys)
    assert code == 1
    assert any(c["status"] == "fail" for c in result["checks"])
    assert "SECRET_BAD_VALUE" not in json.dumps(result)
    assert "Traceback" not in json.dumps(result)


def test_untrusted_project_ignored(
    tmp_working_directory: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_working_directory / ".chartreux"
    directory.mkdir()
    (directory / "config.toml").write_text("[malformed")
    code, result = invoke(capsys)
    assert code == 0
    assert any("untrusted" in c["reason"] for c in result["checks"])


@pytest.mark.asyncio
async def test_model_resolution_failure_does_not_hide_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checks = await doctor._load_config()
    broken = config.model_copy(
        update={
            "active_model": "does-not-exist",
            "mcp_servers": [
                MCPStdio(transport="stdio", name="local", command="never-launch")
            ],
        }
    )
    monkeypatch.setattr(
        doctor, "_load_config", AsyncMock(return_value=(broken, checks))
    )
    results = await doctor._diagnose(argparse.Namespace(smoke=False, live=False))
    assert any(c.name == "active model" and c.status == "fail" for c in results)
    assert any(c.name == "MCP local" and c.status == "unverified" for c in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        MCPStdio(transport="stdio", name="local", command="never-launch"),
        MCPHttp(
            transport="streamable-http",
            name="remote",
            url="https://example.invalid",
            auth=MCPStaticAuth(),
        ),
        MCPHttp(
            transport="streamable-http",
            name="oauth",
            url="https://example.invalid",
            auth=MCPOAuth(type="oauth", scopes=[]),
        ),
    ],
)
async def test_bare_mcp_config_only(server) -> None:
    result = await doctor._mcp(server, live=False)
    assert result.status == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("stdio", [True, False])
async def test_empty_mcp_listing_healthy(
    monkeypatch: pytest.MonkeyPatch, stdio: bool
) -> None:
    listing = AsyncMock(return_value=[])
    server = (
        MCPStdio(transport="stdio", name="local", command="never-launch")
        if stdio
        else MCPHttp(
            transport="streamable-http",
            name="remote",
            url="https://example.invalid",
            auth=MCPStaticAuth(),
        )
    )
    monkeypatch.setattr(
        f"chartreux.core.tools.mcp.tools.list_tools_{'stdio' if stdio else 'http'}",
        listing,
    )
    result = await doctor._mcp(server, live=True)
    assert result.status == "pass"
    assert "0 tools" in result.reason
    listing.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, expected",
    [
        (200, "pass"),
        (404, "unverified"),
        (500, "fail"),
        (503, "fail"),
        (401, "fail"),
        (403, "fail"),
        (429, "fail"),
    ],
)
async def test_live_listing_classification(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: str
) -> None:
    from chartreux.utils.http import ChartreuxAsyncHTTPClient

    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json={"data": [{"id": "test"}]})
    )
    monkeypatch.setattr("chartreux.utils.api_keys.resolve_api_key", lambda _: None)
    # Mock transport exercises discovery's actual lossy error classification and
    # doctor's response hook without a socket or provider request.
    monkeypatch.setattr(
        "chartreux.utils.http.ChartreuxAsyncHTTPClient",
        lambda **kwargs: ChartreuxAsyncHTTPClient(transport=transport, **kwargs),
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", _original_send)
    provider = next(iter(SHIPPED_CATALOG.providers.values()))
    result = await doctor._listing("test", provider, False)
    assert result.status == expected


_original_send = httpx.AsyncClient.send


@pytest.mark.asyncio
async def test_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chartreux.utils.api_keys.resolve_api_key", lambda _: None)
    monkeypatch.setattr(
        "chartreux.core.model_catalog.discovery.discover_models",
        AsyncMock(return_value=DiscoveryError("connection", "secret-body")),
    )
    result = await doctor._listing(
        "test", next(iter(SHIPPED_CATALOG.providers.values())), False
    )
    assert result.status == "fail"
    assert "secret-body" not in result.reason


@pytest.mark.parametrize(
    "args",
    [
        ["--smoke"],
        ["--provider", "x"],
        ["--smoke", "--provider", "unknown"],
        ["--smoke", "--model", "unknown"],
    ],
)
@pytest.mark.usefixtures("allow_dotenv_load")
def test_invalid_smoke_selectors(
    capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli(args)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "status, code", [("pass", 0), ("fail", 1), ("unverified", 0), ("unsupported", 0)]
)
@pytest.mark.usefixtures("allow_dotenv_load")
def test_smoke_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: SmokeStatus,
    code: int,
) -> None:
    base, definition = next(iter(SHIPPED_CATALOG.models.items()))
    deployment = next(d for d in definition.deployments if not d.disabled)
    probe = AsyncMock(
        return_value=ProviderSmokeResult(
            deployment.provider,
            deployment.name,
            base,
            CapabilityResult(status, "safe-code"),
            CapabilityResult("unverified", "opaque"),
            CapabilityResult("unsupported", "no-image"),
        )
    )
    monkeypatch.setattr("chartreux.core.llm.provider_smoke.probe_provider_smoke", probe)
    actual, result = invoke(
        capsys, "--smoke", "--model", base, "--provider", deployment.provider
    )
    assert actual == code
    probe.assert_awaited_once()
    assert probe.call_args.kwargs["model"].provider == deployment.provider
    assert len([c for c in result["checks"] if c["name"].startswith("smoke")]) == 3


@pytest.mark.asyncio
async def test_ambiguous_model_lists_candidates() -> None:
    config, _ = await doctor._load_config()
    base, definition = next(iter(SHIPPED_CATALOG.models.items()))
    catalog = _ambiguous_catalog(base, definition)
    config.attach_catalog_snapshot(CatalogSnapshot(catalog, "test"))
    with pytest.raises(doctor.InvalidTarget, match="candidates:"):
        doctor._target(config, argparse.Namespace(provider=None, model=base))


@pytest.mark.asyncio
async def test_live_failure_isolated_and_unused_shipped_providers_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checks = await doctor._load_config()
    active = config.get_active_model().provider
    other = "other"
    catalog = config.catalog_snapshot.catalog.model_copy(
        update={
            "providers": {
                **config.catalog_snapshot.catalog.providers,
                other: config.catalog_snapshot.catalog.providers[active].model_copy(),
            }
        }
    )
    config.attach_catalog_snapshot(
        replace(
            config.catalog_snapshot,
            catalog=catalog,
            overlaid_providers=frozenset({other}),
        )
    )
    config = config.model_copy(
        update={
            "mcp_servers": [
                MCPStdio(transport="stdio", name="bad", command="never-launch"),
                MCPStdio(transport="stdio", name="good", command="never-launch"),
            ]
        }
    )
    monkeypatch.setattr(
        doctor, "_load_config", AsyncMock(return_value=(config, checks))
    )
    listing = AsyncMock(
        side_effect=[RuntimeError("secret"), doctor.Check("other", "pass", "listed")]
    )
    monkeypatch.setattr(doctor, "_listing", listing)
    monkeypatch.setattr(
        doctor,
        "_mcp",
        AsyncMock(
            side_effect=[
                RuntimeError("secret"),
                doctor.Check("MCP good", "pass", "0 tools"),
            ]
        ),
    )
    results = await doctor._diagnose(argparse.Namespace(smoke=False, live=True))
    assert listing.await_count == 2
    assert any(
        c.name == f"provider {active} listing" and c.status == "fail" for c in results
    )
    assert any(c.name == "MCP bad" and c.status == "fail" for c in results)
    assert any(c.name == "MCP good" and c.status == "pass" for c in results)
    assert "secret" not in repr(results)


def test_help_does_not_load_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    load = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(doctor, "_load_config", load)
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli(["--help"])
    assert exc.value.code == 0
    assert "billable" in capsys.readouterr().out
    load.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expired, matching, expected",
    [(False, True, "unverified"), (True, True, "fail"), (False, False, "fail")],
)
async def test_oauth_inspection_never_refreshes(
    monkeypatch: pytest.MonkeyPatch, expired: bool, matching: bool, expected: str
) -> None:
    from chartreux.core.auth.mcp_oauth import Fingerprint

    server = MCPHttp(
        transport="streamable-http",
        name="oauth",
        url="https://example.invalid",
        auth=MCPOAuth(type="oauth", scopes=[]),
    )
    monkeypatch.setattr(
        Fingerprint,
        "load",
        AsyncMock(return_value=Fingerprint.compute(server) if matching else None),
    )
    storage = Mock(token_expiry_time=0 if expired else None)
    storage.get_tokens = AsyncMock(return_value=object())
    monkeypatch.setattr(
        "chartreux.core.auth.mcp_oauth.KeyringTokenStorage", Mock(return_value=storage)
    )
    result = await doctor._mcp(server, live=True)
    assert result.status == expected
    storage.get_tokens.assert_awaited_once()
    assert storage.method_calls == [call.get_tokens()]


def test_text_output_honestly_renders_nonpasses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        doctor,
        "_diagnose",
        AsyncMock(
            return_value=[
                doctor.Check("listing", "skipped", "requires --live"),
                doctor.Check("thinking", "unverified", "opaque"),
            ]
        ),
    )
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli([])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "SKIPPED: listing" in output
    assert "UNVERIFIED: thinking" in output


@pytest.mark.usefixtures("allow_dotenv_load")
def test_ambiguous_cli_exits_two_with_candidates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    config, checks = asyncio.run(doctor._load_config())
    base, definition = next(iter(config.catalog_snapshot.catalog.models.items()))
    catalog = _ambiguous_catalog(base, definition)
    config.attach_catalog_snapshot(replace(config.catalog_snapshot, catalog=catalog))
    monkeypatch.setattr(
        doctor, "_load_config", AsyncMock(return_value=(config, checks))
    )
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli(["--smoke", "--model", base])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "candidates:" in captured.err
    assert base in captured.err
    assert "second-provider" in captured.err
    assert captured.out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("contents", [None, "[malformed", 'trusted = "invalid"'])
async def test_trust_store_load_is_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str | None
) -> None:
    from chartreux.core.paths import TRUSTED_FOLDERS_FILE

    path = tmp_path / "trusted_folders.toml"
    if contents is not None:
        path.write_text(contents)
    monkeypatch.setattr(type(TRUSTED_FOLDERS_FILE), "path", property(lambda _: path))
    _, checks = await doctor._load_config()
    assert any(c.name == "trust store" and c.status == "unverified" for c in checks)
    if contents is None:
        assert not path.exists()
    else:
        assert path.read_text() == contents


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "resolved-secret"])
async def test_mcp_explicit_header_wins_case_insensitively(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    server = MCPHttp(
        transport="streamable-http",
        name="explicit",
        url="https://example.invalid",
        auth=MCPStaticAuth(
            headers={"authorization": "explicit-secret"}, api_key_env="TEST_KEY"
        ),
    )
    resolve = Mock(return_value=token)
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr("chartreux.utils.api_keys.resolve_api_key", resolve)
    monkeypatch.setattr("chartreux.core.tools.mcp.tools.list_tools_http", listing)
    result = await doctor._mcp(server, live=True)
    assert result.status == "pass"
    assert listing.call_args.kwargs["headers"] == server.http_headers()
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_listing_unsupported_uses_structured_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chartreux.utils.api_keys.resolve_api_key", lambda _: None)
    monkeypatch.setattr(
        "chartreux.core.model_catalog.discovery.discover_models",
        AsyncMock(
            return_value=DiscoveryError(
                "unsupported_listing",
                "Different safe wording",
                listing_unsupported=True,
            )
        ),
    )
    result = await doctor._listing(
        "test", next(iter(SHIPPED_CATALOG.providers.values())), False
    )
    assert result.status == "unverified"
    assert result.reason == "model listing unsupported"


@pytest.mark.asyncio
async def test_smoke_uses_configured_system_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checks = await doctor._load_config()
    config = config.model_copy(update={"enable_system_trust_store": True})
    base, definition = next(iter(config.catalog_snapshot.catalog.models.items()))
    deployment = definition.deployments[0]
    monkeypatch.setattr(
        doctor, "_load_config", AsyncMock(return_value=(config, checks))
    )
    probe = AsyncMock(
        return_value=ProviderSmokeResult(
            deployment.provider,
            deployment.name,
            base,
            CapabilityResult("pass", "ok"),
            CapabilityResult("unsupported", "no-thinking"),
            CapabilityResult("unsupported", "no-image"),
        )
    )
    monkeypatch.setattr("chartreux.core.llm.provider_smoke.probe_provider_smoke", probe)
    await doctor._diagnose(
        argparse.Namespace(
            smoke=True, live=False, model=base, provider=deployment.provider
        )
    )
    assert probe.call_args.kwargs["enable_system_trust_store"] is True


@pytest.mark.parametrize("invalid", [False, True])
def test_bare_validates_runtime_provider_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], invalid: bool
) -> None:
    config, checks = asyncio.run(doctor._load_config())
    if invalid:
        snapshot = config.catalog_snapshot
        provider = config.get_active_model().provider
        catalog = snapshot.catalog.model_copy(
            update={
                "providers": {
                    **snapshot.catalog.providers,
                    provider: snapshot.catalog.providers[provider].model_copy(
                        update={"backend": "unsupported-backend"}
                    ),
                }
            }
        )
        config.attach_catalog_snapshot(replace(snapshot, catalog=catalog))
        with pytest.raises(ValueError):
            config.get_active_provider()
    monkeypatch.setattr(
        doctor, "_load_config", AsyncMock(return_value=(config, checks))
    )
    code, result = invoke(capsys)
    assert code == int(invalid)
    model_checks = {
        c["name"]: c["status"]
        for c in result["checks"]
        if c["name"] in {"default model", "active model"}
    }
    assert model_checks == dict.fromkeys(
        ("default model", "active model"), "fail" if invalid else "pass"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("env_token", [None, "env-secret"])
async def test_mcp_static_auth_matches_runtime_env_only(
    monkeypatch: pytest.MonkeyPatch, env_token: str | None
) -> None:
    keyring = Mock(return_value="keyring-only-secret")
    monkeypatch.setattr("chartreux.utils.api_keys.get_api_key_from_keyring", keyring)
    monkeypatch.delenv("DOCTOR_MCP_TOKEN", raising=False)
    if env_token:
        monkeypatch.setenv("DOCTOR_MCP_TOKEN", env_token)
    server = MCPHttp(
        transport="streamable-http",
        name="static",
        url="https://example.invalid",
        auth=MCPStaticAuth(api_key_env="DOCTOR_MCP_TOKEN"),
    )
    listing = AsyncMock(return_value=[])
    monkeypatch.setattr("chartreux.core.tools.mcp.tools.list_tools_http", listing)
    result = await doctor._mcp(server, live=True)
    assert result.status == ("pass" if env_token else "fail")
    if env_token:
        assert listing.call_args.kwargs["headers"] == server.http_headers()
    else:
        assert "Authorization" not in server.http_headers()
        listing.assert_not_called()
    keyring.assert_not_called()


@pytest.mark.asyncio
async def test_live_mcp_suppresses_actual_sdk_raw_result_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from chartreux.core.tools.mcp import tools as mcp_tools

    credential = "synthetic-doctor-credential"

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["method"] == "initialize"
        # An invalid initialization response takes the SDK's WARNING Raw result
        # path before ClientSession rejects it. Echo the transmitted credential.
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"Authorization": request.headers["Authorization"]},
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx.AsyncClient, "send", _original_send)
    monkeypatch.setattr(
        mcp_tools,
        "create_vibe_mcp_http_client",
        lambda headers, **kwargs: httpx.AsyncClient(
            headers=headers, transport=transport
        ),
    )
    monkeypatch.setattr(mcp_tools, "list_tools_http", _original_mcp_listing)
    monkeypatch.setattr(mcp_tools, "ClientSession", ClientSession)
    server = MCPHttp(
        transport="streamable-http",
        name="echo",
        url="https://example.invalid/mcp",
        auth=MCPStaticAuth(headers={"Authorization": f"Bearer {credential}"}),
    )
    caplog.set_level(logging.WARNING)
    with pytest.raises(ExceptionGroup):
        await doctor._mcp_readiness(server, live=True)
    assert "Raw result:" in caplog.text
    assert credential in caplog.text
    caplog.clear()
    state = (
        logging.root.manager.disable,
        logging.root.level,
        logging.root.handlers.copy(),
    )
    with pytest.raises(ExceptionGroup):
        await doctor._mcp(server, live=True)
    assert credential not in caplog.text
    assert "Raw result:" not in caplog.text
    assert state == (
        logging.root.manager.disable,
        logging.root.level,
        logging.root.handlers,
    )
