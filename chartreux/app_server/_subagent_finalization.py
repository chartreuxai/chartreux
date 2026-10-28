"""P6a: the once-only normal-path finalization of background subagent runs.

The background monitor finalizes one run EXACTLY ONCE, awaited immediately
before each ``build_result`` call site in the normal monitor path
(``_subagent_execution.py``): the first await performs the finalization and
caches it; a repeated build (a result-build failure rebuilt through the
monitor's exception path) is a cache hit and runs no checks. The
finalization result lives in the run-keyed finalization cache on
``SubagentRunAccumulator`` — the field P3 declared (round 6) and P6a
consumes WITHOUT editing ``core/subagents.py``; ``build_result`` reads the
cache and runs no checks (P10 wires the read; P6a populates the cache, and
P6a's acceptance demonstrates cache creation and reuse without P10).

The finalization itself:

- runs the frozen contract's bound acceptance checks through the P5 runner
  (``CompletionCheckRunner``), constructed from the CHILD runtime's cwd,
  workspace, and path authority — contract-adjudicated exactly as P5
  specifies (capability, template binding, path validation) — with the
  runner's shell admission resolver (a ``ShellPermissionResolver`` built
  from the child's bash policy, cwd, workspace, and path authority) so the
  composed check command passes FULL command admission (guardrails, path
  permissions, dangerous-command classification, denylist) immediately
  before spawn; finalization never prompts, so every admission denial
  stamps the check denied without spawning;
- consumes the admission HEAD baseline captured PRE-EXECUTION at admission
  (P4) — only runs whose contract mandates a commit artifact carry one;
- captures the publication HEAD snapshot HERE, at finalization, so the
  admission-vs-publication comparison reflects what the run left behind
  (again only when the contract mandates a commit artifact — runs without
  a commit mandate perform no HEAD probes, at admission or here);
- runs WITHOUT the registry lock: no lock is held across the check
  subprocesses.

Honesty economics (binding): harness-side check failures (denied, timeout,
unavailable, unrunnable) are informational stamps carried in
``RunFinalization.runtime_failure`` — never worker misconduct, never
replacement-worthy; a nonzero check is real evidence about the run, not a
verifier failure. The finalization never raises into the monitor: a
harness-side finalization defect stamps the failure word and leaves the
checks empty. Cancellation propagates untouched (P6b owns the
cancellation/exception dispositions: a cancelled finalization leaves the
cache empty, and the cancellation path drains an already-started
verification).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chartreux.core.completion_checks import (
    CompletionCheckOutcome,
    CompletionCheckRunner,
    CompletionCheckStatus,
)
from chartreux.core.completion_contract import CompletionCheckTemplate
from chartreux.core.git.errors import GitError
from chartreux.core.git.repo import GitRepo
from chartreux.core.subagents import SubagentRunAccumulator
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
)
from chartreux.core.tools.builtins.bash import BashToolConfig
from chartreux.core.tools.utils import PathAuthority, ToolPermission
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime
    from chartreux.app_server._subagent_records import FrozenCompletionContract

__all__ = ["RunFinalization", "finalize_background_run"]

#: The harness-side failure word each non-evidence check status maps to
#: (honesty economics: informational stamps, never worker misconduct). A
#: PASSED or NONZERO check is real evidence about the run, not a verifier
#: failure, and never maps.
_RUNTIME_FAILURE_BY_STATUS: dict[CompletionCheckStatus, str] = {
    CompletionCheckStatus.DENIED: "denied",
    CompletionCheckStatus.TIMED_OUT: "timeout",
    CompletionCheckStatus.LAUNCH_FAILURE: "unrunnable",
}


@dataclass(frozen=True, slots=True)
class RunFinalization:
    """The once-only finalization evidence of one background run (P6a).

    ``checks`` carries one outcome per contract-bound check template, in
    contract order. ``admission_head`` is the P4 baseline consumed
    unchanged (captured PRE-EXECUTION at admission, only when the contract
    mandates a commit artifact). ``publication_head`` is the snapshot
    captured at finalization — ``None`` when no commit artifact is mandated,
    or when the HEAD probe is unavailable (an evidence-availability fact,
    never a finalization failure). ``runtime_failure`` names the primary
    harness-side failure the finalization observed (``denied``,
    ``timeout``, ``unavailable``, or ``unrunnable``), or ``None`` when the
    verifier stayed healthy — a nonzero check is evidence, not a runtime
    failure.
    """

    run_id: str
    checks: tuple[CompletionCheckOutcome, ...] = ()
    admission_head: str | None = None
    publication_head: str | None = None
    runtime_failure: str | None = None


def _bound_check_templates(
    contract: FrozenCompletionContract | None,
) -> tuple[CompletionCheckTemplate, ...]:
    """The check templates the frozen contract's envelope binds, in order."""
    if contract is None or contract.envelope is None:
        return ()
    templates: list[CompletionCheckTemplate] = []
    for requirement in contract.envelope.requirements:
        template = requirement.template
        if template is not None and template not in templates:
            templates.append(template)
    return tuple(templates)


def _capability_failure(contract: FrozenCompletionContract) -> str | None:
    """The harness-side failure word of a non-available frozen capability."""
    capability = getattr(contract, "capability", None)
    value = getattr(capability, "value", capability)
    if value in {"denied", "unavailable"}:
        return str(value)
    return None


def _shell_admission(
    loop: Any, workspace: Any, path_authority: Any
) -> ShellPermissionResolver:
    """Build the shell admission resolver from the child runtime (P5).

    The resolver's constructor inputs are the child runtime's: its bash
    policy config, its working directory, its workspace (the authorized
    roots), and its bash path authority — the same inputs the bash tool
    itself resolves through. A runtime that cannot report its bash policy
    falls back to the shipped default policy, and a reported authority
    without the full ``PathAuthority`` surface is normalized to a
    permissive workspace-bound snapshot (the resolver's path layer needs
    the complete authority): the admission gate itself is never skipped.
    """
    try:
        config = loop.tool_manager.get_tool_config("bash")
    except Exception:
        config = None
    if not isinstance(config, BashToolConfig):
        config = BashToolConfig()
    if not isinstance(path_authority, PathAuthority):
        path_authority = PathAuthority(
            tool_name="bash",
            permission=ToolPermission.ALWAYS,
            allowlist=(),
            denylist=(),
            sensitive=(),
            workspace=workspace,
        )
    return ShellPermissionResolver(config, loop.cwd, workspace, path_authority)


def _check_runner(runtime: SessionRuntime) -> CompletionCheckRunner | None:
    """Build the P5 acceptance-check runner from the child runtime.

    Constructor inputs come from the child runtime: its working directory,
    its workspace (the authorized roots), its bash path authority, and the
    shell admission resolver built from the same runtime. A child whose
    runtime cannot report them leaves the runner unavailable — a
    harness-side stamp, never a worker fault.
    """
    try:
        loop = runtime.agent_loop
        workspace = loop.tool_manager.workspace
        path_authority = loop.tool_manager.path_authority("bash")
        return CompletionCheckRunner(
            cwd=loop.cwd,
            workspace=workspace,
            path_authority=path_authority,
            shell_admission=_shell_admission(loop, workspace, path_authority),
        )
    except Exception:
        return None


def _capture_publication_head(cwd: Path) -> str | None:
    """Capture the publication HEAD snapshot at finalization (P6a).

    The counterpart of the admission baseline (P4, captured PRE-EXECUTION):
    the publication snapshot is captured HERE so the admission-vs-publication
    comparison reflects what the run left behind. A probe failure (not a
    repository, unborn HEAD, git unavailable) is informational: the snapshot
    is ``None`` — an unavailable evidence fact, never a finalization
    failure.
    """
    try:
        return GitRepo.head_commit_at(cwd)
    except GitError:
        return None


async def _build_finalization(
    *,
    run_id: str,
    contract: FrozenCompletionContract | None,
    admission_head: str | None,
    runtime: SessionRuntime,
) -> RunFinalization:
    """Run the finalization body: the bound checks plus the HEAD snapshot."""
    if contract is None:
        # A record constructed outside an admission carries no mandate
        # surface: no checks, no HEAD probes.
        return RunFinalization(run_id=run_id)
    publication_head = (
        _capture_publication_head(Path(contract.cwd))
        if contract.requires_commit_artifact
        else None
    )
    templates = _bound_check_templates(contract)
    if not templates:
        return RunFinalization(
            run_id=run_id,
            admission_head=admission_head,
            publication_head=publication_head,
        )
    runner = _check_runner(runtime)
    if runner is None:
        return RunFinalization(
            run_id=run_id,
            admission_head=admission_head,
            publication_head=publication_head,
            runtime_failure="unavailable",
        )
    checks: list[CompletionCheckOutcome] = []
    runtime_failure = _capability_failure(contract)
    for template in templates:
        outcome = await runner.run(template, contract=contract)
        checks.append(outcome)
        if runtime_failure is None:
            runtime_failure = _RUNTIME_FAILURE_BY_STATUS.get(outcome.status)
    return RunFinalization(
        run_id=run_id,
        checks=tuple(checks),
        admission_head=admission_head,
        publication_head=publication_head,
        runtime_failure=runtime_failure,
    )


async def finalize_background_run(
    *,
    run_id: str,
    accumulator: SubagentRunAccumulator,
    contract: FrozenCompletionContract | None,
    admission_head: str | None,
    runtime: SessionRuntime,
) -> RunFinalization:
    """Finalize one background run exactly once (P6a).

    ONCE-ONLY: the run-keyed finalization cache on the accumulator — the
    field P3 declared (round 6) and P6a consumes here, without editing
    ``core/subagents.py`` — is checked first. A cached finalization returns
    unchanged and NO check re-runs: the background monitor awaits this
    immediately before EACH ``build_result`` call site, and a repeated build
    reads the same evidence. The first await performs the finalization and
    caches it only once it settles, so a cancellation mid-finalization
    leaves the cache empty (P6b drains the already-started verification).

    The finalization runs WITHOUT the registry lock — no lock is held
    across the check subprocesses — and never raises into the monitor: a
    harness-side finalization defect stamps ``runtime_failure`` and leaves
    the checks empty (an informational stamp, never a worker fault).
    Cancellation propagates untouched.
    """
    cache = accumulator._finalization_cache
    cached = cache.get(run_id)
    if isinstance(cached, RunFinalization):
        return cached
    try:
        finalization = await _build_finalization(
            run_id=run_id,
            contract=contract,
            admission_head=admission_head,
            runtime=runtime,
        )
    except Exception:
        # A harness-side finalization defect is an informational stamp,
        # never a worker fault and never a monitor failure: the result
        # build proceeds with an empty, unavailable-stamped finalization.
        logger.warning("Subagent run finalization failed", exc_info=True)
        finalization = RunFinalization(
            run_id=run_id, admission_head=admission_head, runtime_failure="unrunnable"
        )
    cache[run_id] = finalization
    return finalization
