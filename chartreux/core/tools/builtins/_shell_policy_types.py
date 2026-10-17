"""The shell-command policy result type shared across the policy family.

``ShellCommandPolicy`` is the frozen result every per-command policy function
returns: the approval requirement, denial reason, optional diagnostic, and the
path-inspection and write-operand flags the permission resolver and path
assembly consume. It is extracted ahead of the W3b git-family split because
``_shell_git_policy`` constructs it while the facade imports ``_git_policy``
for ``_COMMAND_POLICIES["git"]`` -- leaving the type in the facade would be a
hard import cycle. ``_shell_command_policy`` (the facade) re-exports it, so
import paths and object identity are unchanged by the split.
"""

from __future__ import annotations

from dataclasses import dataclass

from chartreux.core.tools.builtins._shell_diagnostics import Diagnostic


@dataclass(frozen=True)
class ShellCommandPolicy:
    requires_approval: bool = False
    denial_reason: str | None = None
    diagnostic: Diagnostic | None = None

    def denial_diagnostic(self, command_part: str) -> Diagnostic:
        if self.diagnostic is not None:
            return Diagnostic(
                self.diagnostic.code,
                self.diagnostic.offending_token,
                command_part,
                self.diagnostic.detail,
            )
        return Diagnostic(
            "policy",
            command_part=command_part,
            detail=self.denial_reason
            or "unsafe or unmodeled command options are not permitted",
        )

    inspect_positional_paths: bool = False
    # Test-only classification contract: runtime scopes all sed operands alike,
    # including in-place writes; no reader exemption depends on this flag.
    mutates_files: bool = False
    # The command is a Git reader whose repository-owned config may execute
    # code; the permission resolver inspects that config before permitting it.
    inspect_git_repository: bool = False
    option_path_values: tuple[str, ...] = ()
    # Path operands that name files the command WRITES itself (git archive
    # --output, a writing git config --file, git apply --directory, an
    # init/worktree-add/move or submodule-add destination). Unlike
    # option_path_values, these stay checked against protected Git metadata
    # even when the command is git: git's -C/--git-dir/--work-tree
    # selectors locate the repository, but these operands write it.
    write_path_values: tuple[str, ...] = ()
    positional_values: tuple[str, ...] | None = None
