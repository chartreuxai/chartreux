from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from chartreux.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from chartreux.core.events import ToolResultEvent
from chartreux.core.file_revision import full_revision
from chartreux.core.tools.base import BaseToolState, InvokeContext
from chartreux.core.tools.builtins.write_file import (
    WriteFile,
    WriteFileArgs,
    WriteFileConfig,
    WriteFileResult,
)
from chartreux.core.tools.io_port import ShellCommandRequest, ShellCommandResult
from chartreux.core.trusted_folders import trusted_folders_manager
from chartreux.utils.io import BoundedReadResult, ReadSafeResult
from tests.mock.utils import collect_result


class _FakeToolIO:
    """Minimal tool I/O port: client-mediated writes only (no host files)."""

    supports_read = False
    supports_write = True
    supports_terminal = False

    def __init__(self) -> None:
        self.files: dict[Path, str] = {}

    async def read_lines(
        self, path: Path, *, start_line: int, limit: int, max_bytes: int
    ) -> BoundedReadResult:
        raise AssertionError("write_file never reads through the port")

    async def read_text(self, path: Path) -> ReadSafeResult:
        raise AssertionError("write_file never reads through the port")

    async def write_text(
        self,
        path: Path,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        del encoding, newline
        self.files[path] = content

    async def run_shell(self, request: ShellCommandRequest) -> ShellCommandResult:
        raise AssertionError("write_file never runs shell through the port")


@pytest.fixture()
def _setup_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trusted_folders_manager, "is_trusted", lambda _: True)
    monkeypatch.setattr(
        trusted_folders_manager, "find_trust_root", lambda _: tmp_path.resolve()
    )
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    yield
    reset_harness_files_manager()


@pytest.mark.usefixtures("_setup_manager")
def test_format_call_display_relative_to_cwd(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "config.py"

    args = WriteFileArgs(file_path=str(target), content="x")
    display = WriteFile.format_call_display(args)

    assert display.summary == "Writing pkg/config.py"
    assert display.message == "pkg/config.py"
    assert display.settled_message == "pkg/config.py"


@pytest.mark.usefixtures("_setup_manager")
def test_get_result_display_relative_to_cwd(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "config.py"

    result = WriteFileResult(file_path=str(target), bytes_written=1, content="x")
    event = ToolResultEvent(
        tool_call_id="test", tool_name="write_file", tool_class=None, result=result
    )
    display = WriteFile.get_result_display(event)

    assert display.success is True
    assert display.message == "pkg/config.py"
    assert display.message != "config.py"


@pytest.mark.asyncio
async def test_host_write_publishes_the_full_revision(tmp_path: Path) -> None:
    # W6 regression: a host write knows the exact bytes it wrote, so the
    # result carries the full content revision (the content is the complete
    # written file).
    target = tmp_path / "created.txt"
    write = WriteFile(lambda: WriteFileConfig(), BaseToolState())

    result = await collect_result(
        write.run(WriteFileArgs(file_path=str(target), content="created"), None)
    )

    assert result.revision == full_revision(b"created")
    assert target.read_bytes() == b"created"


@pytest.mark.asyncio
async def test_client_mediated_write_publishes_no_revision(tmp_path: Path) -> None:
    # W6 regression: a write through the tool I/O port cannot be attested by
    # the host (the client owns the file — reads and edits degrade to
    # revision=None in exactly this case), so the result carries no revision
    # and the observation path must not fabricate a full-byte revision for it.
    target = tmp_path / "created.txt"
    tool_io = _FakeToolIO()
    write = WriteFile(lambda: WriteFileConfig(), BaseToolState())

    result = await collect_result(
        write.run(
            WriteFileArgs(file_path=str(target), content="created"),
            InvokeContext(tool_call_id="call", tool_io=tool_io),
        )
    )

    assert result.revision is None
    assert tool_io.files[target] == "created"
    assert not target.exists()  # the host never touched the file
