from __future__ import annotations

import asyncio
import os
import tomllib

from dotenv import dotenv_values
import pytest

from chartreux.core.config import ProviderConfig
from chartreux.core.llm_models import Backend
from chartreux.core.paths import GLOBAL_ENV_FILE
from chartreux.setup.auth.api_key_persistence import (
    apply_provider_to_config,
    persist_api_key,
    persist_provider_to_config,
)


def provider() -> ProviderConfig:
    return ProviderConfig(
        name="custom",
        api_base="https://custom.example/v1",
        api_key_env_var="CUSTOM_API_KEY",
        backend=Backend.GENERIC,
    )


def test_provider_persistence_writes_models_toml_not_config(config_dir) -> None:
    assert persist_provider_to_config(provider())
    catalog = tomllib.loads((config_dir / "models.toml").read_text())
    assert (
        catalog["providers"]["custom/default"]["api_base"]
        == "https://custom.example/v1"
    )
    config = tomllib.loads((config_dir / "config.toml").read_text())
    assert "providers" not in config


def test_provider_persistence_upserts_catalog_entry(config_dir) -> None:
    assert persist_provider_to_config(provider())
    updated = provider().model_copy(update={"api_base": "https://updated.example/v1"})
    assert asyncio.run(apply_provider_to_config(None, updated))
    catalog = tomllib.loads((config_dir / "models.toml").read_text())
    assert (
        catalog["providers"]["custom/default"]["api_base"]
        == "https://updated.example/v1"
    )


def test_persist_writes_api_key_to_env_file_and_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    assert persist_api_key(provider(), "new-key") == "completed"
    assert os.environ["CUSTOM_API_KEY"] == "new-key"
    assert dotenv_values(GLOBAL_ENV_FILE.path)["CUSTOM_API_KEY"] == "new-key"
