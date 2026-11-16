from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server.local import (
    ContinueSessionIntent,
    LocalHarnessHost,
    LocalHarnessOptions,
    ResumeSessionIntent,
)
from chartreux.app_server.protocol import AppServerResponseError, SessionOptions
import chartreux.cli.entrypoint as entrypoint
from chartreux.cli.entrypoint import parse_arguments
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.harness_files import (
    get_harness_files_manager,
    reset_harness_files_manager,
)
from chartreux.core.git.worktree import naming_model
from chartreux.core.llm import utility_completion
from chartreux.core.usage import AsyncUsageWriter, UsagePurpose, UsageReader
from chartreux.core.usage_project import resolve_project_key
from tests.app_server.test_usage_attribution import AttemptBackend
from tests.conftest import build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


def _parse(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    monkeypatch.setattr("sys.argv", ["chartreux", *argv])
    return parse_arguments()


def test_removed_updater_flags_are_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for argument in (["--check-upgrade"], ["update"]):
        monkeypatch.setattr("sys.argv", ["chartreux", *argument])
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        assert exc_info.value.code == 2

    assert "not available" in capsys.readouterr().err

    args = _parse(monkeypatch, [])
    assert args.disabled_tools is None


def test_disabled_tools_appends_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["--disabled-tools", "bash", "--disabled-tools", "web*"])
    assert args.disabled_tools == ["bash", "web*"]


def test_enabled_and_disabled_tools_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, ["--enabled-tools", "read", "--disabled-tools", "bash"])
    assert args.enabled_tools == ["read"]
    assert args.disabled_tools == ["bash"]


def test_worktree_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse(monkeypatch, []).worktree is None


def test_bare_worktree_requests_auto_naming(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse(monkeypatch, ["--worktree"]).worktree is True


def test_worktree_with_a_value_keeps_the_name(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse(monkeypatch, ["--worktree", "feat-x"]).worktree == "feat-x"


def test_worktree_after_the_prompt_auto_names(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["Fix the login bug", "--worktree"])

    assert args.worktree is True
    assert args.initial_prompt == "Fix the login bug"


# Documents an argparse trap shared with --resume: an optional-value flag eats
# the next token, so the prompt must precede --worktree or follow a "--".
def test_worktree_before_the_prompt_consumes_it_as_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, ["--worktree", "Fix the login bug"])

    assert args.worktree == "Fix the login bug"
    assert args.initial_prompt is None


def test_double_dash_separates_the_prompt_from_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, ["--worktree", "--", "Fix the login bug"])

    assert args.worktree is True
    assert args.initial_prompt == "Fix the login bug"


def test_suggest_worktree_name_reads_the_dotenv_before_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def suggest(prompt: str | None, **_kwargs: object) -> str:
        calls.append(f"suggest:{prompt}")
        return "repair-oauth-redirect"

    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.load_dotenv_values",
        lambda *_args, **_kwargs: calls.append("dotenv"),
    )
    monkeypatch.setattr(
        "chartreux.core.git.worktree.naming_model.suggest_worktree_name", suggest
    )
    # The autouse fixture leaves the singleton initialised, which is not the
    # state a real CLI start is in here. Without this reset the entrypoint's own
    # init is a no-op and dropping it would go unnoticed.
    reset_harness_files_manager()

    suggestion, context = entrypoint._suggest_worktree_name("Fix the login bug")
    assert suggestion == "repair-oauth-redirect"
    assert context is not None
    # The worktree is prepared before run_cli loads ~/.chartreux/.env, so a key that
    # lives only there has to be in os.environ before the provider is checked.
    assert calls == ["dotenv", "suggest:Fix the login bug"]
    # Raises unless the entrypoint initialised it. Resolving config prompts goes
    # through this singleton, and main() does not set it up until much later.
    get_harness_files_manager()


def test_doctor_dispatches_before_interactive_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("sys.argv", ["chartreux", "doctor", "--json"])
    monkeypatch.setattr("chartreux.cli.doctor_command.run_doctor_cli", calls.append)
    monkeypatch.setattr(
        entrypoint, "parse_arguments", lambda: pytest.fail("interactive startup")
    )
    entrypoint.main()
    assert calls == [["--json"]]


def test_suggest_worktree_name_skips_the_model_without_a_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no prompt means nothing to name from")

    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.load_dotenv_values", explode
    )

    assert entrypoint._suggest_worktree_name(None) == (None, None)


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "empty-answer",
        "failure",
        "timeout",
        "no-key",
        "resume",
        "continue",
        "startup-failure",
    ],
)
def test_cli_naming_drains_early_loop_and_preserves_root_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_dir: Path, scenario: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.load_dotenv_values", lambda: None
    )
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=scenario in {"resume", "continue"},
            save_dir=str(tmp_path / "sessions"),
            generate_titles=False,
        )
    )
    orchestrator = FakeConfigOrchestrator(config)
    monkeypatch.setattr(
        naming_model, "build_default_orchestrator", AsyncMock(return_value=orchestrator)
    )
    backend = AttemptBackend([
        mock_llm_chunk(content="" if scenario == "empty-answer" else "fix-bug")
    ])
    if scenario == "failure":
        from chartreux.core.llm.backend.generic import notify_request_started

        async def fail(**_kwargs: Any) -> Any:
            notify_request_started()
            raise RuntimeError("naming failed")

        monkeypatch.setattr(backend, "complete", fail)
    elif scenario == "timeout":
        from chartreux.core.llm.backend.generic import notify_request_started

        async def slow(**_kwargs: Any) -> Any:
            notify_request_started()
            await asyncio.sleep(10)

        monkeypatch.setattr(backend, "complete", slow)
        monkeypatch.setattr(naming_model, "_TOTAL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(utility_completion, "create_backend", lambda **_: backend)
    monkeypatch.setattr(
        utility_completion,
        "resolve_api_key",
        lambda _: None if scenario == "no-key" else "test-key",
    )
    facades: list[AsyncUsageWriter] = []
    real_close = AsyncUsageWriter.aclose

    async def close(writer: AsyncUsageWriter) -> None:
        facades.append(writer)
        await real_close(writer)

    monkeypatch.setattr(AsyncUsageWriter, "aclose", close)
    suggestion, context = entrypoint._suggest_worktree_name("Fix the bug")
    assert context is not None
    assert suggestion == (
        None
        if scenario in {"empty-answer", "failure", "timeout", "no-key"}
        else "fix-bug"
    )
    assert len(facades) == 1
    assert facades[0]._closed
    assert not facades[0]._pending
    assert not facades[0]._callbacks
    assert context.identity.project_key == resolve_project_key(tmp_path)
    early_records = UsageReader(config_dir / "usage").reconcile().records
    assert len(early_records) == (0 if scenario == "no-key" else 1)
    assert len(context.early_settlements) == len(early_records)
    assert all(
        record.root_session_id == context.identity.root_session_id
        for record in early_records
    )

    monkeypatch.setattr(
        runtime,
        "build_default_orchestrator",
        AsyncMock(side_effect=ValueError("bad config"))
        if scenario == "startup-failure"
        else AsyncMock(return_value=orchestrator),
    )
    monkeypatch.setattr(
        runtime.HarnessProcess,
        "_build_mcp_registry_impl",
        AsyncMock(return_value=FakeMCPRegistry()),
    )
    real_loop = runtime.AgentLoop

    def build_loop(**kwargs: Any) -> Any:
        return real_loop(
            backend=AttemptBackend([mock_llm_chunk(content="answer")]), **kwargs
        )

    monkeypatch.setattr(runtime, "AgentLoop", build_loop)

    async def launch() -> None:
        harness = LocalHarnessHost()
        intent: runtime.LocalSessionIntent = runtime.NewSessionIntent()
        saved_id: str | None = None
        if scenario in {"resume", "continue"}:
            saved = await harness.start(
                LocalHarnessOptions(
                    session_options=SessionOptions(cwd=str(tmp_path), headless=True)
                )
            )
            saved_id = saved.session_id
            async for _ in saved.act("first turn"):
                pass
            await saved.close()
        if scenario == "resume":
            assert saved_id is not None
            intent = ResumeSessionIntent(saved_id)
        elif scenario == "continue":
            intent = ContinueSessionIntent()
        host = await harness.connect(
            LocalHarnessOptions(
                session_options=SessionOptions(cwd=str(tmp_path), headless=True),
                session=intent,
                startup_accounting=context,
            )
        )
        try:
            if scenario == "startup-failure":
                with pytest.raises(AppServerResponseError):
                    await host.open_session()
                assert context.state == "abandoned"
                assert (
                    UsageReader(config_dir / "usage").reconcile().records
                    == early_records
                )
            elif scenario in {"resume", "continue"}:
                session = await host.open_session()
                assert session.session_id == saved_id
                assert session.session_id != context.identity.root_session_id
                assert context.state == "abandoned"
                async for _ in session.act("resumed turn"):
                    pass
                records = UsageReader(config_dir / "usage").reconcile().records
                assert (
                    tuple(
                        record
                        for record in records
                        if record.root_session_id == context.identity.root_session_id
                    )
                    == early_records
                )
                assert all(
                    record.purpose == UsagePurpose.CONVERSATION
                    for record in records
                    if record.root_session_id == saved_id
                )
                await session.close()
            else:
                session = await host.open_session()
                assert session.session_id == context.identity.root_session_id
                assert context.state == "adopted"
                async for _ in session.act("hello"):
                    pass
                records = UsageReader(config_dir / "usage").reconcile().records
                assert [record.purpose for record in records] == [
                    *([UsagePurpose.WORKTREE_NAMING] if early_records else []),
                    UsagePurpose.CONVERSATION,
                ]
                assert all(
                    record.root_session_id == session.session_id for record in records
                )
                await session.close()
        finally:
            await host.close()
            await harness.close()

    asyncio.run(launch())


def test_doctor_help_stays_light_in_fresh_process() -> None:
    import subprocess
    import sys

    code = """
import sys
from chartreux.cli.entrypoint import main
sys.argv = ["chartreux", "doctor", "--help"]
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
assert not any(n == "pydantic" or n.startswith("chartreux.core.config") for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "billable" in result.stdout
