from __future__ import annotations

import os
import traceback
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from chartreux.core.config._source_validation import validate_source
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import LayerImplementationError
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator

_SOURCE = "environment"
_SECRET = "synthetic-secret-sentinel"


def _rendered_error(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment_name", "field"),
    [
        ("CHARTREUX_SYNTHETIC_UNKNOWN", "SYNTHETIC_UNKNOWN"),
        ("CHARTREUX_PROFILE", "PROFILE"),
        ("CHARTREUX_TEST_NETWORK_GUARD", "TEST_NETWORK_GUARD"),
        ("CHARTREUX_MANAGED_SHELL_TOOLS_ENABLED", "MANAGED_SHELL_TOOLS_ENABLED"),
        ("CHARTREUX_SESSION_LOGGING__SYNTHETIC_UNKNOWN", "synthetic_unknown"),
        ("CHARTREUX_SESSION_LOGGING", "session_logging"),
    ],
)
async def test_unknown_environment_fields_fail_before_settings_filtering(
    environment_name: str, field: str
) -> None:
    value = (
        '{"enabled": true, "synthetic_extra": "' + _SECRET + '"}'
        if environment_name.endswith("SESSION_LOGGING")
        else _SECRET
    )
    with patch.dict(os.environ, {environment_name: value}, clear=True):
        with pytest.raises(LayerImplementationError) as caught:
            await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    rendered = _rendered_error(caught.value)
    assert _SOURCE in rendered
    assert field in rendered
    assert _SECRET not in rendered


def test_environment_source_errors_keep_source_and_field_without_input() -> None:
    with pytest.raises(ValidationError) as caught:
        validate_source(
            ChartreuxConfigSchema,
            {"session_logging": {"enabled": _SECRET}},
            source="environment (synthetic-locator)",
        )

    rendered = _rendered_error(caught.value)
    assert "environment (synthetic-locator)" in rendered
    assert "session_logging" in rendered
    assert _SECRET not in rendered
    assert all(error["input"] is None for error in caught.value.errors())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment_name",
    [
        "CHARTREUX_DISABLE_WELCOME_BANNER_ANIMATION",
        "CHARTREUX_API_TIMEOUT",
        "CHARTREUX_ENABLED_AGENTS",
    ],
)
async def test_empty_invalid_scalar_and_structured_values_are_not_ignored(
    environment_name: str,
) -> None:
    with patch.dict(os.environ, {environment_name: ""}, clear=True):
        with pytest.raises(LayerImplementationError) as caught:
            await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    rendered = _rendered_error(caught.value)
    assert "environment" in rendered
    assert environment_name.removeprefix("CHARTREUX_").lower() in rendered
    assert "''" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env",
    [
        {"CHARTREUX_API_TIMEOUT__TYPO": _SECRET},
        {"CHARTREUX_API_TIMEOUT": "0.12", "CHARTREUX_API_TIMEOUT__TYPO": _SECRET},
    ],
)
async def test_nested_scalar_environment_names_fail_before_decoding(
    env: dict[str, str],
) -> None:
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(LayerImplementationError) as caught:
            await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    rendered = _rendered_error(caught.value)
    assert _SOURCE in rendered
    assert "api_timeout__typo" in rendered
    assert _SECRET not in rendered


@pytest.mark.asyncio
async def test_deeply_nested_scalar_environment_name_is_redacted() -> None:
    env = {"CHARTREUX_SESSION_LOGGING__ENABLED__TYPO": _SECRET}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(LayerImplementationError) as caught:
            await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    rendered = _rendered_error(caught.value)
    assert _SOURCE in rendered
    assert "session_logging__enabled__typo" in rendered
    assert _SECRET not in rendered


@pytest.mark.asyncio
async def test_valid_nested_dict_environment_values_are_decoded() -> None:
    env = {
        "CHARTREUX_TOOLS__bash__DENYLIST": '["synthetic-deny"]',
        "CHARTREUX_THINKING_OVERRIDES__local": "high",
    }
    with patch.dict(os.environ, env, clear=True):
        data = await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    assert data.model_dump() == {
        "tools": {"bash": {"denylist": ["synthetic-deny"]}},
        "thinking_overrides": {"local": "high"},
    }


@pytest.mark.asyncio
async def test_unknown_nested_model_field_is_rejected_without_raw_value() -> None:
    env = {
        "CHARTREUX_MODELS__custom__NAME": "custom-model",
        "CHARTREUX_MODELS__custom__PROVIDER": "custom-provider",
        "CHARTREUX_MODELS__custom__SYNTHETIC_UNKNOWN": _SECRET,
    }
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(LayerImplementationError) as caught:
            await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    rendered = _rendered_error(caught.value)
    assert _SOURCE in rendered
    assert "models" in rendered
    assert "catalog_scope" in rendered
    assert _SECRET not in rendered


@pytest.mark.asyncio
async def test_valid_empty_environment_values_are_preserved() -> None:
    env = {
        "CHARTREUX_ACTIVE_MODEL": "",
        "CHARTREUX_ENABLED_AGENTS": "[]",
        "CHARTREUX_SESSION_LOGGING__SESSION_PREFIX": "",
    }
    with patch.dict(os.environ, env, clear=True):
        data = await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    assert data.model_dump() == {
        "active_model": "",
        "enabled_agents": [],
        "session_logging": {"session_prefix": ""},
    }


@pytest.mark.asyncio
async def test_nonconfig_controls_are_exempt_and_provider_environment_is_untouched() -> (
    None
):
    env = {
        "CHARTREUX_HOME": "/synthetic/home",
        "CHARTREUX_TYPING_GRACE_PERIOD_MS": "17",
        "CHARTREUX_ACP_LOGGING_ENABLED": "true",
        "CHARTREUX_TEST_DISABLE_KEYRING": "1",
        "CHARTREUX_TEST_DISABLE_AUTO_TITLE": "1",
        "MISTRAL_API_KEY": _SECRET,
        "ACTIVE_MODEL": _SECRET,
    }
    with patch.dict(os.environ, env, clear=True):
        data = await EnvironmentLayer(schema=ChartreuxConfigSchema).load()
        assert os.environ["MISTRAL_API_KEY"] == _SECRET

    assert data.model_dump() == {}


@pytest.mark.asyncio
async def test_failed_reload_retains_accepted_config_and_policy() -> None:
    env = {
        "CHARTREUX_ACTIVE_MODEL": "glm-5-2",
        "CHARTREUX_TOOLS": '{"bash": {"denylist": ["synthetic-deny"]}}',
    }
    with patch.dict(os.environ, env, clear=True):
        environment = EnvironmentLayer(schema=ChartreuxConfigSchema)
        orchestrator = await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), environment],
            default_layer_resolver=lambda: environment,
        )
        before_config = orchestrator.config.model_dump(mode="json")
        before_restrictions = orchestrator.restrictions
        before_token = orchestrator.accepted_token

        os.environ["CHARTREUX_API_TIMEOUT"] = _SECRET
        with pytest.raises(LayerImplementationError):
            await orchestrator.reload()

    assert orchestrator.config.model_dump(mode="json") == before_config
    assert orchestrator.restrictions == before_restrictions
    assert orchestrator.accepted_token is before_token
    assert environment.cached_data is not None
    assert environment.cached_data.model_dump()["active_model"] == "glm-5-2"
