from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
from pathlib import Path

import pytest

from chartreux.app_server.local import ClientDescriptor, LocalHarnessOptions
from chartreux.app_server.protocol import ClientInfo, SessionOptions
from chartreux.cli.programmatic import (
    OutputFormat,
    ProgrammaticLimitError,
    ProgrammaticOutput,
    run_programmatic,
)
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm_models import Backend
from tests.conftest import build_test_vibe_config
from tests.mock.mock_backend_factory import mock_backend_factory
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_CATALOG_FIELDS = {"models", "providers"}
_ORDINARY_OVERRIDE_EXCLUDED_FIELDS = {*_CATALOG_FIELDS, "authorized_roots_by_project"}


def _options() -> LocalHarnessOptions:
    return LocalHarnessOptions(
        client=ClientDescriptor(
            info=ClientInfo(
                name="vibe_programmatic", version="test", entrypoint="programmatic"
            )
        ),
        session_options=SessionOptions(
            disabled_tools=["ask_user_question"], headless=True
        ),
    )


def _use_runtime_config(
    monkeypatch: pytest.MonkeyPatch, config: ChartreuxConfigSchema
) -> None:
    async def build_orchestrator(
        data: dict[str, object] | None = None, *, harness_files: HarnessFilesManager
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        del harness_files
        base = OverridesLayer(
            data=config.model_dump(
                mode="json", exclude=_ORDINARY_OVERRIDE_EXCLUDED_FIELDS
            ),
            name="base",
        )
        default = DefaultConfigLayer(schema=ChartreuxConfigSchema)
        session = OverridesLayer(data=data or {})
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[default, base, session],
            default_layer_resolver=lambda: base,
        )

    monkeypatch.setattr(
        "chartreux.app_server._runtime.build_default_orchestrator", build_orchestrator
    )


def test_streaming_output_uses_public_history_entries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config)

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend(
            mock_llm_chunk(content="Decorators wrap functions.")
        ),
    ):
        result = run_programmatic(
            harness_options=_options(),
            prompt="Explain decorators",
            output_format=OutputFormat.STREAMING,
        )

    assert result is None
    entries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    messages = [entry for entry in entries if entry["type"] == "message"]
    assert [(entry["role"], entry["content"][0]["text"]) for entry in messages] == [
        ("user", "Explain decorators"),
        ("assistant", "Decorators wrap functions."),
    ]
    assert not any(entry["role"] == "system" for entry in messages)


def test_text_output_returns_last_assistant_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config)

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([mock_llm_chunk(content="Understood.")]),
    ):
        result = run_programmatic(
            harness_options=_options(),
            prompt="Continue",
            output_format=OutputFormat.TEXT,
        )

    assert result == "Understood."


def test_json_output_is_the_public_history_list() -> None:
    stream = StringIO()
    output = ProgrammaticOutput(OutputFormat.JSON, stream)
    output.finalize([])

    assert json.loads(stream.getvalue()) == []


def test_untrusted_workspace_warning_comes_from_app_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("project instructions", encoding="utf-8")
    monkeypatch.chdir(project)
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config)

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([mock_llm_chunk(content="Done.")]),
    ):
        run_programmatic(harness_options=_options(), prompt="Continue")

    warning = capsys.readouterr().err
    assert str(project) in warning
    assert "AGENTS.md" in warning
    assert "--trust" in warning


def test_conversation_limits_cross_the_public_turn_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config)
    options = _options()
    options = replace(
        options,
        session_options=options.session_options.model_copy(update={"max_turns": 0}),
    )

    with mock_backend_factory(
        Backend.MISTRAL, lambda provider, **kwargs: FakeBackend()
    ):
        with pytest.raises(ProgrammaticLimitError, match="Turn limit"):
            run_programmatic(harness_options=options, prompt="Continue")


def test_plain_prompt_runs_normal_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config)

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([
            mock_llm_chunk(content="Normal response.")
        ]),
    ):
        result = run_programmatic(
            harness_options=_options(), prompt="Hello", output_format=OutputFormat.TEXT
        )

    assert result == "Normal response."
