"""Upgrade boundary from main's persisted config and session metadata."""

from __future__ import annotations

import json
from pathlib import Path

from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.model_catalog.migration import apply_migration, plan_migration
from chartreux.core.model_catalog.resolver import ModelResolver

# Captured from main's create_default_config(), serialized by its config writer.
_MAIN_WRITTEN_CONFIG = """active_model = ""
compaction_model = ""
[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
api_key_env_var = "MISTRAL_API_KEY"
api_style = "openai"
backend = "mistral"
reasoning_field_name = "reasoning_content"
emits_finish_reason = true
project_id = ""
region = ""
[providers.extra_headers]
[[models]]
name = "glm-5-2"
provider = "mistral"
alias = "glm-5-2"
temperature = 0.2
input_price = 1.4
output_price = 4.4
cached_input_price = 0.14
thinking = "high"
supports_images = false
auto_compact_threshold = 400000
"""

# Captured shape of main's SessionLogger metadata writer with its pinned config.
_MAIN_WRITTEN_SESSION = json.dumps({
    "session_id": "main-session",
    "start_time": "2026-01-01T00:00:00+00:00",
    "end_time": None,
    "git_commit": None,
    "git_branch": None,
    "environment": {},
    "username": "chartreux",
    "config": {"active_model": "glm-5-2"},
})


def test_m7_upgrade_from_main_config_and_session_preserves_effective_model(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    catalog = tmp_path / "models.toml"
    session = tmp_path / "metadata.json"
    config.write_text(_MAIN_WRITTEN_CONFIG)
    session.write_text(_MAIN_WRITTEN_SESSION)

    apply_migration(plan_migration(config, catalog))

    persisted_session = json.loads(session.read_text())
    resolved = ModelResolver(load_catalog(catalog)).resolve(
        persisted_session["config"]["active_model"]
    )
    effective = resolved.materialize(auto_compact_threshold=200_000)
    assert (effective.name, effective.provider, effective.temperature) == (
        "glm-5-2",
        "mistral/default",
        0.2,
    )
    assert effective.input_price == 1.4 and effective.auto_compact_threshold == 400_000
