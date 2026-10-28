"""Every shell syntax denial names the compliant alternative inline (P2-e).

The recognition-contract treatment: a worker denied five or eight times for
the same construct was given no instruction, so each audit category's denial
must carry its specific remediation in the rendered message. One pin per
audit category asserts that the actual denial message for a representative
command contains the compliant alternative.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig

# One remediation pin per audit category: (representative command, the
# denial's recognized marker, the compliant alternative the message must
# name). The glob command denies through the expansion inspection's
# unsupported-syntax failure, the one path-glob shape the resolver classifies
# as a syntax denial; caller-side rejections (an option-like word, a match
# outside the workspace) stay policy denials and name no alternative.
_PINS: tuple[tuple[str, str, str], ...] = (
    (
        "echo $HOME",
        "variable expansion",
        "write the literal value directly as the argument",
    ),
    (
        "grep src/**/x.py target.py",
        "path glob candidate",
        "list the directory with ls to see the matching names, or write "
        "the exact paths instead of the glob",
    ),
    (
        "echo =ls",
        "Zsh equals expansion",
        "drop the leading equals sign and invoke the command by its "
        "name or absolute path",
    ),
    (
        "cat <<EOF\nhi\nEOF",
        "heredocs are unsupported in v0.1",
        "use write_file to create the content",
    ),
    (
        "for i in 1 2; do echo $i; done",
        "for loop",
        "split the work into separate simple commands",
    ),
)


def _denied_reason(cwd: Path, command: str) -> str:
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    assert result.permission == ToolPermission.NEVER, (command, result)
    return result.reason or ""


@pytest.mark.parametrize(("command", "marker", "alternative"), _PINS, ids=range(5))
def test_syntax_denial_names_the_compliant_alternative(
    tmp_path: Path, command: str, marker: str, alternative: str
) -> None:
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x = 2\n", encoding="utf-8")
    reason = _denied_reason(tmp_path, command)
    assert marker in reason, reason
    assert "Instead," in reason, reason
    assert alternative in reason, reason
