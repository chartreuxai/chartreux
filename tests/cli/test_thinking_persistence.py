from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from chartreux.cli.textual_ui.widgets.thinking_picker import ThinkingPickerApp
from chartreux.core.config import ModelConfig
from tests.conftest import build_test_chartreux_app, build_test_vibe_config_schema


def _persisted_config(config_dir: Path) -> dict[str, object]:
    with (config_dir / "config.toml").open("rb") as file:
        return tomllib.load(file)


@pytest.mark.asyncio
async def test_thinking_change_survives_later_config_change(config_dir: Path) -> None:
    config = build_test_vibe_config_schema(
        active_model="glm-5-2",
        models=[ModelConfig(name="glm-5-2", provider="mistral", alias="glm-5-2")],
    )
    app = build_test_chartreux_app(config=config)

    async with app.run_test():
        await app.on_thinking_picker_app_thinking_selected(
            ThinkingPickerApp.ThinkingSelected("high")
        )
        await app._persist_config_changes({"autocopy_to_clipboard": False})
        assert app.app_server.resources.config.current.active_model.thinking == "high"

    persisted = _persisted_config(config_dir)
    assert "thinking_overrides" not in persisted
    assert "autocopy_to_clipboard" not in persisted
