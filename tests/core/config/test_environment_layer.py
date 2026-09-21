from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.environment import EnvironmentLayer


@pytest.mark.asyncio
async def test_reads_env_vars() -> None:
    env = {
        "MISTRAL_API_KEY": "test-key",
        "CHARTREUX_ACTIVE_MODEL": "mistral-large",
        "CHARTREUX_DISABLE_WELCOME_BANNER_ANIMATION": "true",
        "CHARTREUX_SESSION_LOGGING__ENABLED": "false",
        "CHARTREUX_SESSION_LOGGING__SESSION_PREFIX": "mysession",
        "CHARTREUX_SESSION_LOGGING__GENERATE_TITLES": "false",
        "CHARTREUX_API_TIMEOUT": ".12",
    }
    with patch.dict(os.environ, env, clear=True):
        layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
        data = await layer.load()

    assert data.model_dump() == {
        "active_model": "mistral-large",
        "disable_welcome_banner_animation": True,
        "session_logging": {
            "enabled": False,
            "session_prefix": "mysession",
            "generate_titles": False,
        },
        "api_timeout": 0.12,
    }

    assert layer.name == "environment"


@pytest.mark.asyncio
async def test_no_vars_set_returns_empty() -> None:
    with patch.dict(os.environ, {}, clear=True):
        data = await EnvironmentLayer(schema=ChartreuxConfigSchema).load()
    assert data.model_dump() == {}


@pytest.mark.asyncio
async def test_fingerprint_changes_when_env_changes() -> None:
    with patch.dict(os.environ, {"CHARTREUX_ACTIVE_MODEL": "first-model"}, clear=True):
        layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
        data1 = await layer.load()
        fp1 = layer.fingerprint

        os.environ["CHARTREUX_ACTIVE_MODEL"] = "second-model"
        data2 = await layer.load(force=True)
        fp2 = layer.fingerprint

    assert data1.model_dump() == {"active_model": "first-model"}
    assert data2.model_dump() == {"active_model": "second-model"}
    assert isinstance(fp1, str)
    assert fp1
    assert isinstance(fp2, str)
    assert fp2
    assert fp1 != fp2


@pytest.mark.asyncio
async def test_always_trusted() -> None:
    assert await EnvironmentLayer(schema=ChartreuxConfigSchema).resolve_trust() is True
