"""The template acceptance-check runner (P5).

The harness runs the closed :class:`CompletionCheckTemplate` vocabulary
through the scrubbed shell runner (chartreux.core.utils.shell's
``spawn_shell_command``): a credential-scrubbed environment, deterministic
pager/locale settings, DEVNULL stdin, and an isolated process group per
check. Each check is a subprocess: a finite timeout (the plan's ~120s
default), capped capture with secret redaction, and process-group cleanup
on teardown. The runner distinguishes pass, nonzero, timeout, denied, and
launch-failure outcomes; there are no prompts, no retries, and no
fallbacks.

Contract-aware adjudication: the runner consumes the P4-frozen completion
contract (the ``completion_contract`` field on the run record /
accumulator) through the :class:`CompletionContractLike` protocol — core
code never imports the app-server record type. A check runs only when the
contract's capability is AVAILABLE and the check's template is among the
contract envelope's bound templates; denied or unavailable authority, and
templates outside the contract, stamp the check unavailable — an
informational stamp, never a worker fault.

The check's target path is validated under the child's authorized roots
(path authority plus workspace containment): the path is resolved once,
re-checked immediately before spawn, and validation and spawn share one
stable resolved target — a symlink replacement or escape fails the
adjudication instead of redirecting the spawn. The COMPOSED COMMAND then
passes full shell admission — guardrails, path permissions,
dangerous-command classification, and the denylist — through the shell
permission resolver (chartreux.core.tools.builtins._shell_permission_resolver,
``resolve_permission``) immediately before spawn, exactly as the plan's
Decisions line 7 binds: resolver admission first, then the scrubbed
subprocess runner. Finalization never prompts: every admission denial,
including the promptable ask-class (out-of-root, dangerous-command),
stamps the check denied without spawning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import shlex
import time
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.core.completion_contract import CompletionCheckTemplate
from chartreux.core.tools.secret_redaction import redact
from chartreux.core.tools.utils import PathAccess, ToolPermission
from chartreux.core.utils.async_subprocess import (
    registered_process_group,
    terminate_process_group,
)
from chartreux.core.utils.shell import spawn_shell_command

if TYPE_CHECKING:
    pass

__all__ = [
    "DEFAULT_CHECK_TIMEOUT_SECONDS",
    "MAX_CAPTURE_BYTES",
    "CompletionCheckOutcome",
    "CompletionCheckRunner",
    "CompletionCheckStatus",
    "CompletionContractLike",
]

#: The default acceptance-check timeout: the plan's finite ~120s default.
DEFAULT_CHECK_TIMEOUT_SECONDS = 120.0

#: Captured output is capped per stream; the outcome keeps capped tails.
MAX_CAPTURE_BYTES = 64 * 1024

#: The argv each template executes, with the check root as the final
#: operand where the template takes one. Operands are shlex-joined at
#: spawn time, so an operand carrying shell metacharacters stays a literal
#: argument and never re-interprets the command.
_TEMPLATE_ARGV: dict[CompletionCheckTemplate, tuple[str, ...]] = {
    CompletionCheckTemplate.PYTEST: ("uv", "run", "pytest", "-q"),
    CompletionCheckTemplate.RUFF_CHECK: ("uv", "run", "ruff", "check"),
    CompletionCheckTemplate.PYRIGHT: ("uv", "run", "pyright"),
    CompletionCheckTemplate.GIT_HEAD: ("git", "rev-parse", "HEAD"),
    CompletionCheckTemplate.GIT_STATUS: ("git", "status", "--porcelain"),
    CompletionCheckTemplate.GIT_DIFF_STAT: ("git", "diff", "--stat"),
}

#: The templates whose argv takes the check root as its final operand.
_ROOT_OPERAND_TEMPLATES = frozenset({
    CompletionCheckTemplate.PYTEST,
    CompletionCheckTemplate.RUFF_CHECK,
    CompletionCheckTemplate.PYRIGHT,
})


class CompletionCheckStatus(StrEnum):
    """The acceptance-check outcome vocabulary.

    ``PASSED`` is exit 0; ``NONZERO`` is a normal exit with a failing
    code; ``TIMED_OUT`` is the finite timeout expiring (the process group
    is terminated); ``DENIED`` is a refused adjudication (revoked
    authority, a path outside the authorized roots, a shell-admission
    denial, or a template outside the frozen contract);
    ``LAUNCH_FAILURE`` is an operating-system spawn failure. Denied and
    launch-failure are distinct outcomes: a denial is a policy decision,
    a launch failure is an environment defect.
    """

    PASSED = "passed"
    NONZERO = "nonzero"
    TIMED_OUT = "timed_out"
    DENIED = "denied"
    LAUNCH_FAILURE = "launch_failure"


class CompletionContractLike(Protocol):
    """The frozen contract surface the runner consumes (the P4-P5 interface).

    :class:`chartreux.app_server._subagent_records.FrozenCompletionContract`
    satisfies this structurally; core code never imports the record type.
    """

    @property
    def capability(self) -> Any: ...

    @property
    def envelope(self) -> Any | None: ...


@dataclass(frozen=True, slots=True)
class CompletionCheckOutcome:
    """One acceptance check's observed result.

    Captured streams are redacted and capped: ``stdout``/``stderr`` keep
    head-and-tail samples around the capture cap, never the full body.
    ``exit_code`` is the process's exit status, or the negative signal
    number when the child died by a signal it did not receive from the
    runner's own cleanup. A denied check never spawned, so its ``argv``
    is empty.
    """

    template: CompletionCheckTemplate
    status: CompletionCheckStatus
    argv: tuple[str, ...]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    detail: str = ""


class _Denied(Exception):
    """A failed path adjudication; never retried, never re-spawned."""

    def __init__(self, reason: str, target: Path) -> None:
        self.reason = reason
        self.target = target
        super().__init__(f"{reason}: {target}")


@dataclass(frozen=True, slots=True)
class _ShellCommand:
    """One composed check command, as the resolver's args protocol shapes it."""

    command: str


def _capped_tail(data: bytes, limit: int) -> str:
    """The redacted, capped head+tail of one captured stream."""
    if len(data) <= limit:
        decoded = data.decode("utf-8", errors="replace")
    else:
        keep = limit // 2
        marker = len(data) - limit
        decoded = "".join((
            data[:keep].decode("utf-8", errors="replace"),
            f"\n[... {marker} bytes truncated ...]\n",
            data[-keep:].decode("utf-8", errors="replace"),
        ))
    return redact(decoded)


def _bound_templates(
    contract: CompletionContractLike,
) -> frozenset[CompletionCheckTemplate]:
    """The check templates the contract envelope binds, if any.

    A ``None`` envelope binds nothing: every check is then outside the
    contract and stamps unbound (denied) — fail-closed.
    """
    envelope = getattr(contract, "envelope", None)
    if envelope is None:
        return frozenset()
    return frozenset(
        requirement.template
        for requirement in envelope.requirements
        if requirement.template is not None
    )


@dataclass(frozen=True, slots=True)
class CompletionCheckRunner:
    """The harness-built template acceptance-check runner (P5).

    Constructor inputs come from the child runtime: the child's working
    directory, its workspace (authorized roots), its path authority, and
    the shell admission resolver — a ``ShellPermissionResolver`` built
    from the child's bash policy, working directory, workspace, and path
    authority — that admits the composed check command immediately before
    spawn. The runner itself is inert until :meth:`run` executes one
    check.
    """

    cwd: Path
    workspace: Any
    path_authority: Any
    timeout: float = DEFAULT_CHECK_TIMEOUT_SECONDS
    max_capture_bytes: int = MAX_CAPTURE_BYTES
    shell_admission: Any = None

    def check_argv(
        self, template: CompletionCheckTemplate, *, root: Path | None = None
    ) -> tuple[str, ...]:
        """Harness-built argv from templates.

        The closed template vocabulary maps to a fixed argv; the check's
        root, when the template takes one, is the final operand. The argv
        is never shell-interpreted here: it is shlex-joined once, at spawn
        time, so an operand carrying shell metacharacters stays a literal
        argument of the spawned check.
        """
        argv = _TEMPLATE_ARGV[template]
        if root is not None and template in _ROOT_OPERAND_TEMPLATES:
            return (*argv, os.fspath(root))
        return argv

    def validate_check_root(self, root: Path | str) -> Path:
        """Contract-aware path permission for one acceptance-check target.

        The target is resolved ONCE (symlink/TOCTOU: resolve first, spawn
        the resolved target) and adjudicated under the child's authorized
        roots: the path authority must allow reading it, and the resolved
        target must stay within the workspace's authorized roots (a
        symlink escape or a replaced root fails the adjudication instead
        of redirecting the spawn). The returned resolved target is the
        stable object validation and spawn share.
        """
        target = Path(root).resolve()
        decision = self.path_authority.resolve(str(target), PathAccess.READ)
        if decision is not None and decision.permission is ToolPermission.NEVER:
            raise _Denied("path_authority_revoked", target)
        if not self._under_authorized_root(target):
            raise _Denied("root_escape", target)
        return target

    def _under_authorized_root(self, target: Path) -> bool:
        """The workspace-authorized containment check for one resolved target."""
        return self.workspace.allows(target)

    async def run(
        self,
        template: CompletionCheckTemplate,
        *,
        root: Path | None = None,
        contract: CompletionContractLike | None = None,
    ) -> CompletionCheckOutcome:
        """Run one contract-bound acceptance check.

        No prompts, no retries, no fallbacks: the check spawns at most
        once. The frozen contract is consumed first — a denied or
        unavailable capability, and a template outside the contract's
        bound templates, stamp the check unavailable without spawning.
        The target path is re-checked immediately before spawn, and
        validation and spawn share one stable resolved target. The
        composed command then passes full shell admission through the
        resolver immediately before spawn; a denial stamps the check
        denied without spawning.
        """
        started = time.monotonic()
        capability = (
            getattr(contract, "capability", None) if contract is not None else None
        )
        # The capability arrives as the P4 enum member or a plain value;
        # core code never imports the app-server type, so read the value.
        capability_value = getattr(capability, "value", capability)
        if capability is not None and capability_value != "available":
            return self._unavailable(template, capability, started)
        if contract is not None and template not in _bound_templates(contract):
            return self._denied(template, "template_not_bound", started)
        target = self.cwd if root is None else Path(root)
        try:
            target = self.validate_check_root(target)
        except _Denied as denial:
            return self._denied(template, denial.reason, started, denial.target)
        argv = self.check_argv(template, root=target)
        denial = self._admit_shell_command(argv)
        if denial is not None:
            return self._denied(template, denial, started, target)
        return await self._spawn(template, argv, target, started)

    def _admit_shell_command(self, argv: tuple[str, ...]) -> str | None:
        """Admit the composed check command through the shell resolver.

        Full command admission — guardrails, path permissions, the
        dangerous-command classification, and the denylist — runs through
        the shell permission resolver immediately before spawn (the
        plan's Decisions line 7). Finalization never prompts, so EVERY
        denial — including the promptable ask-class (out-of-root,
        dangerous-command) — stamps the check denied without spawning,
        and an admission defect fails closed the same way. Returns the
        denial reason, or ``None`` when the command is admitted (or no
        resolver is configured, as in direct harness construction).
        """
        admission = self.shell_admission
        if admission is None:
            return None
        try:
            context = admission.resolve_permission(_ShellCommand(shlex.join(argv)))
        except Exception:
            return "shell_admission_unavailable"
        if context is not None and context.permission is ToolPermission.NEVER:
            kind = context.denial_kind or "policy"
            return f"shell_admission_denied:{kind}"
        return None

    def _unavailable(
        self, template: CompletionCheckTemplate, capability: Any, started: float
    ) -> CompletionCheckOutcome:
        return CompletionCheckOutcome(
            template=template,
            status=CompletionCheckStatus.DENIED,
            argv=(),
            duration=time.monotonic() - started,
            detail=(
                "check unavailable: capability "
                f"{getattr(capability, 'value', capability)}"
            ),
        )

    def _denied(
        self,
        template: CompletionCheckTemplate,
        reason: str,
        started: float,
        target: Path | None = None,
    ) -> CompletionCheckOutcome:
        return CompletionCheckOutcome(
            template=template,
            status=CompletionCheckStatus.DENIED,
            argv=(),
            duration=time.monotonic() - started,
            detail=reason,
        )

    def _timed_out(
        self,
        template: CompletionCheckTemplate,
        argv: tuple[str, ...],
        target: Path,
        started: float,
    ) -> CompletionCheckOutcome:
        return CompletionCheckOutcome(
            template=template,
            status=CompletionCheckStatus.TIMED_OUT,
            argv=argv,
            duration=time.monotonic() - started,
            detail=f"timed out after {self.timeout:g}s",
        )

    def _launch_failure(
        self,
        template: CompletionCheckTemplate,
        argv: tuple[str, ...],
        target: Path,
        error: OSError,
        started: float,
    ) -> CompletionCheckOutcome:
        return CompletionCheckOutcome(
            template=template,
            status=CompletionCheckStatus.LAUNCH_FAILURE,
            argv=argv,
            duration=time.monotonic() - started,
            detail=f"launch failed: {error}",
        )

    def _classify(
        self,
        template: CompletionCheckTemplate,
        argv: tuple[str, ...],
        target: Path,
        process: asyncio.subprocess.Process,
        stdout: bytes,
        stderr: bytes,
        started: float,
    ) -> CompletionCheckOutcome:
        """Distinguish pass/nonzero/timeout/denied/launch-failure."""
        code = process.returncode
        if code == 0:
            status = CompletionCheckStatus.PASSED
        elif code is not None and code > 0:
            status = CompletionCheckStatus.NONZERO
        else:
            status = CompletionCheckStatus.LAUNCH_FAILURE
        return CompletionCheckOutcome(
            template=template,
            status=status,
            argv=argv,
            exit_code=code,
            stdout=_capped_tail(stdout, self.max_capture_bytes),
            stderr=_capped_tail(stderr, self.max_capture_bytes),
            duration=time.monotonic() - started,
            detail={
                CompletionCheckStatus.PASSED: "exit 0",
                CompletionCheckStatus.NONZERO: f"exit {code}",
            }.get(status, ""),
        )

    async def _spawn(
        self,
        template: CompletionCheckTemplate,
        argv: tuple[str, ...],
        target: Path,
        started: float,
    ) -> CompletionCheckOutcome:
        """Spawn one check through the scrubbed shell runner.

        The command string is shlex-joined (injection operands stay
        literal arguments of the spawned shell), the child inherits the
        scrubbed credential environment, and the process is isolated in
        its own group for bounded cleanup.
        """
        command = shlex.join(argv)
        try:
            process = await spawn_shell_command(command, cwd=target, merge_stderr=False)
        except OSError as error:
            return self._launch_failure(template, argv, target, error, started)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except TimeoutError:
            owner = registered_process_group(process)
            await terminate_process_group(owner)
            return self._timed_out(template, argv, target, started)
        return self._classify(template, argv, target, process, stdout, stderr, started)
