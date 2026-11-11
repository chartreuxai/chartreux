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
    ProgrammaticPersistenceError,
    run_programmatic,
)
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm_models import Backend
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import ArchiveUnavailableError, SessionLogger
from tests.conftest import build_test_vibe_config
from tests.mock.mock_backend_factory import mock_backend_factory
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_CATALOG_FIELDS = {"models", "providers"}
_ORDINARY_OVERRIDE_EXCLUDED_FIELDS = {
    *_CATALOG_FIELDS,
    "authorized_roots_by_project",
    "dangerous_command_grants_by_project",
    "remote_exec_grants_by_project",
}


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
    monkeypatch: pytest.MonkeyPatch, config: ChartreuxConfigSchema, tmp_path: Path
) -> None:
    async def build_orchestrator(
        data: dict[str, object] | None = None, *, harness_files: HarnessFilesManager
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        del harness_files
        user_path = tmp_path / "user.toml"
        user_path.write_text(
            "credential_env_passthrough = "
            f"{json.dumps(config.credential_env_passthrough)}\n",
            encoding="utf-8",
        )
        user = UserConfigLayer(path=user_path)
        base = OverridesLayer(
            data=config.model_dump(
                mode="json",
                exclude={
                    *_ORDINARY_OVERRIDE_EXCLUDED_FIELDS,
                    "credential_env_passthrough",
                },
            ),
            name="base",
        )
        default = DefaultConfigLayer(schema=ChartreuxConfigSchema)
        session = OverridesLayer(data=data or {})
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[default, user, base, session],
            default_layer_resolver=lambda: base,
        )

    monkeypatch.setattr(
        "chartreux.app_server._runtime.build_default_orchestrator", build_orchestrator
    )


def test_streaming_output_uses_public_history_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)

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
    _use_runtime_config(monkeypatch, config, tmp_path)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)
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


def test_plain_prompt_runs_normal_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)

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


def _logging_config(tmp_path: Path) -> ChartreuxConfigSchema:
    return build_test_vibe_config(
        include_model_info=False,
        include_commit_signature=False,
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions"), generate_titles=False
        ),
    )


def test_programmatic_persists_the_session_before_returning_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP3 [3D] save-before-response: once run_programmatic returns the final
    response, the session transcript on disk already carries it.
    """
    _use_runtime_config(monkeypatch, _logging_config(tmp_path), tmp_path)

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([
            mock_llm_chunk(content="Persisted answer.")
        ]),
    ):
        result = run_programmatic(
            harness_options=_options(), prompt="Hello", output_format=OutputFormat.TEXT
        )

    assert result == "Persisted answer."
    sessions = [
        path for path in (tmp_path / "sessions").glob("session_*") if path.is_dir()
    ]
    assert len(sessions) == 1
    # The root writer's fence lockfile lives inside the session dir.
    assert (sessions[0] / ".writer-fence.lock").is_file()
    messages, _metadata = SessionLoader.load_session(sessions[0])
    contents = [
        message.content for message in messages if isinstance(message.content, str)
    ]
    assert "Hello" in contents
    assert "Persisted answer." in contents


def test_programmatic_fails_closed_when_the_final_save_does_not_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP3 [3D] save-before-response: a save that fails closed (recorded
    diagnostic, nothing persisted) means no final response is emitted.
    """
    _use_runtime_config(monkeypatch, _logging_config(tmp_path), tmp_path)

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise ArchiveUnavailableError(
            Path("messages.jsonl"), "injected save failure", recoverable_by_restart=True
        )

    monkeypatch.setattr(SessionLogger, "_save_interaction_sync", fail_save)

    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([
            mock_llm_chunk(content="Lost answer.")
        ]),
    ):
        with pytest.raises(ProgrammaticPersistenceError, match="persisted"):
            run_programmatic(
                harness_options=_options(),
                prompt="Hello",
                output_format=OutputFormat.TEXT,
            )
