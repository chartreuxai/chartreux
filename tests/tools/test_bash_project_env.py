"""Project-env injection and redaction-backstop tests (P2-b).

The bash tool loads exactly two dotenv files — the trusted folder's ``.env``
and ``$CHARTREUX_HOME/.env`` — into its subprocess environment when the
call's working folder is inside a trusted folder, and the session redaction
set masks the known values in tool output. Trust decisions come from an
in-memory stub manager and the global file is steered through a
``CHARTREUX_HOME`` inside ``tmp_path``, so no test touches the real trust
store, the real Chartreux home, or any live provider: every command runs
locally (``printenv``/``test``) and every assertion is on local state.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path

import pytest

from chartreux.core import env
from chartreux.core.background_jobs import (
    BackgroundJobRegistry,
    BashReadArgs,
    BashStartArgs,
)
from chartreux.core.tools import secret_redaction as sr
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.io_port import ShellCommandRequest, ShellCommandResult
from chartreux.core.tools.secret_redaction import REDACTED_PLACEHOLDER
from chartreux.core.trusted_folders import TrustedFoldersManager
from chartreux.core.utils.shell import _shell_environment
from chartreux.utils.io import BoundedReadResult, ReadSafeResult
from tests.conftest import build_test_agent_loop
from tests.mock.utils import collect_result

PROJECT_SECRET = "sk-project-0123456789abcdef"
GLOBAL_SECRET = "sk-global-9876543210fedcba"
ROTATED_SECRET = "sk-project-rotated-0123456789"


class StubTrustManager(TrustedFoldersManager):
    """Trust decisions served from memory, never touching the real trust store."""

    def __init__(self, roots: dict[Path, Path]) -> None:
        # Maps a working folder to its trust root; unmapped folders are untrusted.
        self._roots = roots

    def find_trust_root(self, path: Path) -> Path | None:
        return self._roots.get(path)


class SessionOwner:
    """A weak-referenceable stand-in for a session owner."""


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    env.reset_cache()
    sr.reset_cache()
    yield
    env.reset_cache()
    sr.reset_cache()


@pytest.fixture
def trusted_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A trusted project folder with a project .env and a stubbed global home."""
    home = tmp_path / "chartreux-home"
    home.mkdir()
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    (home / ".env").write_text(
        f"GLOBAL_TEST_KEY={GLOBAL_SECRET}\nSHARED_TEST_KEY=from-global\n",
        encoding="utf-8",
    )
    project = (tmp_path / "project").resolve()
    project.mkdir()
    (project / ".env").write_text(
        "PROJECT_TEST_KEY="
        f"{PROJECT_SECRET}\nSHARED_TEST_KEY=from-project\nEXPLICIT_TEST_KEY=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        env, "trusted_folders_manager", StubTrustManager({project: project})
    )
    return project


@pytest.fixture
def registered_project_env(trusted_project: Path) -> Iterator[SessionOwner]:
    """The session-level registration the agent loop performs at construction."""
    owner = SessionOwner()
    sr.register_project_env(owner, env.project_env_file(trusted_project))
    yield owner
    sr.unregister_project_env(owner)


def _bash_tool(cwd: Path) -> Bash:
    return Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)


@pytest.mark.asyncio
async def test_bash_injects_the_project_env_into_the_subprocess(
    trusted_project: Path,
) -> None:
    tool = _bash_tool(trusted_project)

    result = await collect_result(
        tool.run(BashArgs(command="printenv PROJECT_TEST_KEY"))
    )
    assert result.stdout.strip() == PROJECT_SECRET

    # $CHARTREUX_HOME/.env loads too (its key is a scrub name, so only the
    # injected file value can provide it).
    result = await collect_result(
        tool.run(BashArgs(command="printenv GLOBAL_TEST_KEY"))
    )
    assert result.stdout.strip() == GLOBAL_SECRET

    # The project .env wins over $CHARTREUX_HOME/.env per key.
    result = await collect_result(
        tool.run(BashArgs(command="printenv SHARED_TEST_KEY"))
    )
    assert result.stdout.strip() == "from-project"


@pytest.mark.asyncio
async def test_bash_child_scripts_find_the_key_via_the_environment(
    trusted_project: Path,
) -> None:
    # The dogfood shape: code reads the environment, the model never sees the
    # value.
    tool = _bash_tool(trusted_project)

    result = await collect_result(
        tool.run(
            BashArgs(
                command='test -n "$PROJECT_TEST_KEY" && echo present || echo absent'
            )
        )
    )

    assert result.stdout.strip() == "present"


@pytest.mark.asyncio
async def test_bash_does_not_inject_outside_trusted_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "chartreux-home"
    home.mkdir()
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    project = (tmp_path / "project").resolve()
    project.mkdir()
    (project / ".env").write_text(
        f"PROJECT_TEST_KEY={PROJECT_SECRET}\n", encoding="utf-8"
    )
    monkeypatch.setattr(env, "trusted_folders_manager", StubTrustManager({}))
    tool = _bash_tool(project)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command="printenv PROJECT_TEST_KEY")))

    assert "Return code: 1" in str(err.value)


@pytest.mark.asyncio
async def test_bash_re_reads_the_project_env_on_rotation(trusted_project: Path) -> None:
    tool = _bash_tool(trusted_project)
    result = await collect_result(
        tool.run(BashArgs(command="printenv PROJECT_TEST_KEY"))
    )
    assert result.stdout.strip() == PROJECT_SECRET

    env_file = trusted_project / ".env"
    env_file.write_text(
        f"PROJECT_TEST_KEY={ROTATED_SECRET}\nSHARED_TEST_KEY=from-project\n",
        encoding="utf-8",
    )
    os.utime(env_file, ns=(0, 2_000_000_000_000_000_000))

    result = await collect_result(
        tool.run(BashArgs(command="printenv PROJECT_TEST_KEY"))
    )
    assert result.stdout.strip() == ROTATED_SECRET


@pytest.mark.asyncio
async def test_explicit_process_value_wins_over_the_env_files(
    trusted_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # EXPLICIT_TEST_KEY is not a chartreux credential name (absent from
    # $CHARTREUX_HOME/.env), so the inherited process value survives the scrub
    # and wins over the .env files.
    monkeypatch.setenv("EXPLICIT_TEST_KEY", "from-process")
    tool = _bash_tool(trusted_project)

    result = await collect_result(
        tool.run(BashArgs(command="printenv EXPLICIT_TEST_KEY"))
    )

    assert result.stdout.strip() == "from-process"


@pytest.mark.asyncio
async def test_printenv_of_a_known_secret_returns_masked_output(
    trusted_project: Path, registered_project_env: SessionOwner
) -> None:
    # The acceptance pin: presence verification without disclosure. The
    # injected value reaches the child (raw stdout), the redaction backstop
    # masks it in the tool result, and a nonzero exit tells absence.
    tool = _bash_tool(trusted_project)

    result = await collect_result(
        tool.run(BashArgs(command="printenv PROJECT_TEST_KEY"))
    )
    assert result.stdout.strip() == PROJECT_SECRET

    assert PROJECT_SECRET in sr.known_secret_values()
    masked = sr.redact_model(result)
    assert masked.stdout == f"{REDACTED_PLACEHOLDER}\n"
    assert PROJECT_SECRET not in masked.stdout
    assert sr.redact(result.stdout) == f"{REDACTED_PLACEHOLDER}\n"

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command="printenv ABSENT_TEST_KEY")))
    assert "Return code: 1" in str(err.value)
    assert PROJECT_SECRET not in str(err.value)


@pytest.mark.asyncio
async def test_inherited_process_value_is_redacted_alongside_the_env_value(
    trusted_project: Path,
    registered_project_env: SessionOwner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # R6: EXPLICIT_TEST_KEY exists in the project .env (from-file) and in
    # the inherited process environment (from-process); the inherited value
    # wins in the subprocess, so printenv emits it — and the redaction set
    # masks BOTH values, not only the .env one.
    monkeypatch.setenv("EXPLICIT_TEST_KEY", "from-process")
    tool = _bash_tool(trusted_project)

    result = await collect_result(
        tool.run(BashArgs(command="printenv EXPLICIT_TEST_KEY"))
    )
    assert result.stdout.strip() == "from-process"

    assert "from-file" in sr.known_secret_values()
    assert "from-process" in sr.known_secret_values()
    masked = sr.redact_model(result)
    assert masked.stdout == f"{REDACTED_PLACEHOLDER}\n"
    assert "from-process" not in masked.stdout
    assert sr.redact("env=from-file ambient=from-process") == (
        f"env={REDACTED_PLACEHOLDER} ambient={REDACTED_PLACEHOLDER}"
    )


@pytest.mark.asyncio
async def test_bash_terminal_path_carries_the_project_env(
    trusted_project: Path,
) -> None:
    class CapturingToolIO:
        supports_read = False
        supports_write = False
        supports_terminal = True

        def __init__(self) -> None:
            self.shell_requests: list[ShellCommandRequest] = []

        async def read_lines(
            self, path: Path, *, start_line: int, limit: int, max_bytes: int
        ) -> BoundedReadResult:
            raise NotImplementedError

        async def read_text(self, path: Path) -> ReadSafeResult:
            raise NotImplementedError

        async def write_text(
            self,
            path: Path,
            content: str,
            *,
            encoding: str = "utf-8",
            newline: str | None = None,
        ) -> None:
            raise NotImplementedError

        async def run_shell(self, request: ShellCommandRequest) -> ShellCommandResult:
            self.shell_requests.append(request)
            return ShellCommandResult(stdout="", stderr="", returncode=0)

    tool_io = CapturingToolIO()
    tool = _bash_tool(trusted_project)

    await collect_result(
        tool.run(
            BashArgs(command="echo hi"),
            InvokeContext(
                tool_call_id="call-1", session_id="session-1", tool_io=tool_io
            ),
        )
    )

    request = tool_io.shell_requests[0]
    assert request.env == {
        "GLOBAL_TEST_KEY": GLOBAL_SECRET,
        "PROJECT_TEST_KEY": PROJECT_SECRET,
        "SHARED_TEST_KEY": "from-project",
        "EXPLICIT_TEST_KEY": "from-file",
    }


@pytest.mark.asyncio
async def test_bash_start_injects_and_redacts_the_project_env(
    trusted_project: Path, registered_project_env: SessionOwner
) -> None:
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    try:
        result = await registry.start(
            port,
            # The trailing short word gives the incremental redactor a safe
            # boundary, so the masked line is captured while the job runs.
            BashStartArgs(command="printenv PROJECT_TEST_KEY; echo done; sleep 30"),
            cwd=trusted_project,
        )
        page = await port.read(BashReadArgs(job_id=result.job.job_id, wait_seconds=2))
        text = "".join(record.text for record in page.records)
        # The masked placeholder proves the injected value reached the
        # background child and was redacted on capture.
        assert REDACTED_PLACEHOLDER in text
        assert PROJECT_SECRET not in text
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_bash_start_without_a_cwd_injects_no_project_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The trust-gated injection keys to the caller's cwd — the same root the
    # session registered its redaction set for. A missing cwd must not fall
    # back to the harness process's cwd: that root's .env values would
    # reach the child unmasked, desynchronized from the session redaction
    # set. Here the process cwd is (stub-)trusted to a project whose .env
    # holds a secret, so the old fallback would leak it into the job output.
    home = tmp_path / "chartreux-home"
    home.mkdir()
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    project = (tmp_path / "project").resolve()
    project.mkdir()
    (project / ".env").write_text(f"NO_CWD_KEY={PROJECT_SECRET}\n", encoding="utf-8")
    monkeypatch.setattr(
        env,
        "trusted_folders_manager",
        StubTrustManager({Path.cwd(): project, project: project}),
    )
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    try:
        result = await registry.start(
            port, BashStartArgs(command="printenv NO_CWD_KEY; echo done"), cwd=None
        )
        page = await port.read(BashReadArgs(job_id=result.job.job_id, wait_seconds=2))
        text = "".join(record.text for record in page.records)
        assert PROJECT_SECRET not in text
        assert "done" in text
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_agent_loop_registers_the_project_env_for_redaction(
    trusted_project: Path,
) -> None:
    loop = build_test_agent_loop(cwd=trusted_project)
    try:
        assert PROJECT_SECRET in sr.known_secret_values()
    finally:
        await loop.aclose()
    assert PROJECT_SECRET not in sr.known_secret_values()


def test_register_project_env_extends_and_releases_the_redaction_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "chartreux-home"
    home.mkdir()
    monkeypatch.setenv("CHARTREUX_HOME", str(home))
    env_file = tmp_path / "project.env"
    env_file.write_text(f"UNIT_TEST_KEY={PROJECT_SECRET}\n", encoding="utf-8")
    owner = SessionOwner()

    assert PROJECT_SECRET not in sr.known_secret_values()
    sr.register_project_env(owner, env_file)
    try:
        assert PROJECT_SECRET in sr.known_secret_values()
        assert sr.redact(f"value: {PROJECT_SECRET}") == f"value: {REDACTED_PLACEHOLDER}"
    finally:
        sr.unregister_project_env(owner)
    assert PROJECT_SECRET not in sr.known_secret_values()


def test_shell_environment_extra_env_fills_names_the_harness_did_not_set() -> None:
    shell_env = _shell_environment({"EXTRA_TEST_KEY": "from-extra"})

    assert shell_env["EXTRA_TEST_KEY"] == "from-extra"
    assert shell_env["TERM"] == "dumb"


def test_shell_environment_prefers_inherited_values_over_extra_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INHERITED_TEST_KEY", "from-process")

    shell_env = _shell_environment({
        "INHERITED_TEST_KEY": "from-extra",
        "TERM": "from-extra",
        "CI": "from-extra",
    })

    assert shell_env["INHERITED_TEST_KEY"] == "from-process"
    assert shell_env["TERM"] == "dumb"
    assert shell_env["CI"] == "true"
