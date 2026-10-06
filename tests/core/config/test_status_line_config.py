from __future__ import annotations

from pathlib import Path
import tomllib

from pydantic import ValidationError
import pytest

from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import StatusLineConfig
from chartreux.core.config.settings_catalog import (
    EDITABLE_BY_PATH,
    STATUS_LINE_PATHS,
    VISIBLE_SETTINGS,
    render_initial_user_config,
)


def test_status_line_composite_visibility_is_separate_from_backing_validation() -> None:
    visible = {item.path: item for item in VISIBLE_SETTINGS}
    assert visible["status_line"].control == "status_line"
    assert "status_line" not in EDITABLE_BY_PATH
    assert len(STATUS_LINE_PATHS) == 4
    assert set(STATUS_LINE_PATHS).isdisjoint(visible)
    assert all(path in EDITABLE_BY_PATH for path in STATUS_LINE_PATHS)
    defaults = StatusLineConfig().model_dump(mode="json")
    for path in STATUS_LINE_PATHS:
        EDITABLE_BY_PATH[path].validate(defaults[path.split(".")[1]])
    assert tomllib.loads(render_initial_user_config())["status_line"] == defaults


def test_status_line_defaults_and_partial_config() -> None:
    config = ChartreuxConfigSchema.model_validate({
        "status_line": {"separator": "space"}
    })
    assert config.show_message_timestamps is True
    assert config.status_line.model_dump() == {
        "segments": ["directory", "pid", "context"],
        "directory_style": "name",
        "context_style": "tokens-percent",
        "separator": "space",
    }
    assert ChartreuxConfigSchema().status_line.separator == "pipe"


@pytest.mark.asyncio
async def test_background_jobs_config_round_trip(tmp_path: Path) -> None:
    segments = ["background-jobs", "context", "directory"]
    path = tmp_path / "config.toml"
    path.write_text(
        '[status_line]\nsegments = ["background-jobs", "context", "directory"]\n'
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([UserConfigLayer(path=path, name="user")])
    config = await builder.build()
    assert config.status_line.segments == segments
    assert (
        StatusLineConfig.model_validate_json(
            config.status_line.model_dump_json()
        ).segments
        == segments
    )


@pytest.mark.parametrize(
    "segments",
    [
        [],
        ["directory"],
        ["context"],
        ["directory", "context", "pid", "pid"],
        ["directory", "context", "unknown"],
        ["directory", "context", False],
    ],
)
def test_invalid_segments_rejected_on_load_and_catalog_validation(
    segments: list[object],
) -> None:
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({"status_line": {"segments": segments}})
    with pytest.raises(ValueError):
        EDITABLE_BY_PATH["status_line.segments"].validate(segments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("key", "value"),
    [("directory_style", "full"), ("context_style", "percent"), ("separator", "/")],
)
def test_status_line_rejects_unknown_styles(key: str, value: str) -> None:
    with pytest.raises(ValidationError):
        StatusLineConfig.model_validate({key: value})
    with pytest.raises(ValueError):
        EDITABLE_BY_PATH[f"status_line.{key}"].validate(value)


def test_all_segments_selectable_and_recorded_global_spend_described() -> None:
    segments = [
        "context",
        "directory",
        "pid",
        "model",
        "git-branch",
        "spend-today",
        "spend-week",
        "spend-month",
        "background-jobs",
    ]
    assert StatusLineConfig.model_validate({"segments": segments}).segments == segments
    item = EDITABLE_BY_PATH["status_line.segments"]
    item.validate(list(segments))
    assert item.kind == "list" and item.control is None and item.group == "Interface"
    for spend in ("spend-today", "spend-week", "spend-month"):
        assert spend in item.description
    composite = next(item for item in VISIBLE_SETTINGS if item.path == "status_line")
    for description in (item.description, composite.description):
        assert "recorded USD spend across all projects" in description
        assert "calendar day, Monday-start week" in description
        assert "+" in description and "Unknown" in description
        assert "unavailable, not zero spend" in description
        assert "placeholders" not in description
        assert "require the usage ledger" not in description
    bootstrap = tomllib.loads(render_initial_user_config())
    assert bootstrap["status_line"] == StatusLineConfig().model_dump()
    assert bootstrap["show_message_timestamps"] is True


@pytest.mark.asyncio
async def test_sparse_status_line_layers_preserve_defaults_and_replace_order(
    tmp_path: Path,
) -> None:
    lower, higher = tmp_path / "lower.toml", tmp_path / "higher.toml"
    lower.write_text(
        '[status_line]\ndirectory_style = "path"\nsegments = ["directory", "model", "context"]\n'
    )
    higher.write_text(
        '[status_line]\nseparator = "space"\nsegments = ["context", "directory"]\n'
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        UserConfigLayer(path=lower, name="lower"),
        UserConfigLayer(path=higher, name="higher"),
    ])
    config = await builder.build()
    assert config.status_line.model_dump() == {
        "segments": ["context", "directory"],
        "directory_style": "path",
        "context_style": "tokens-percent",
        "separator": "space",
    }
