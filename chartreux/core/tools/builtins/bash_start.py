from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

from pydantic import JsonValue

from chartreux.core.background_jobs import BashStartArgs, StartResult
from chartreux.core.env import project_env_for_cwd
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
)
from chartreux.core.tools.builtins._remote_exec import RemoteExecDenial
from chartreux.core.tools.builtins._shell_permission_resolver import (
    GlobPlanDenial,
    QuotedWordPlanDenial,
    ShellPermissionResolver,
    glob_execution_plan_for_command,
    glob_plan_denial_text,
    quoted_word_execution_plan_for_command,
    quoted_word_plan_denial_text,
    remote_exec_authority_admits,
    remote_exec_plan_for_command,
    shell_command_checkout_no_overwrite_ignore,
    shell_command_mutation_roots,
    shell_command_write_paths,
)
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashToolConfig,
    CancelledShellCommand,
)
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolResultDisplay, ToolUIData
from chartreux.core.tools.utils import effective_scratchpad_roots
from chartreux.core.utils.shell import UnknownShellExecutorError


class FailedJobStart(ToolError):
    """ToolError from a background job whose admission failed.

    A failed admission may still have spawned the command (partial
    execution), so it carries the command's write-classified paths and its
    unknown-write-surface flag; the tool executor copies them onto the error
    result event, and the submission scan collects the footprint exactly as
    a successful start's would be.
    """

    def __init__(
        self,
        message: str,
        *,
        write_paths: list[str] | None = None,
        write_paths_unknown: bool = False,
        mutation_roots: list[str] | None = None,
        mutation_roots_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.write_paths = list(write_paths or [])
        self.write_paths_unknown = write_paths_unknown
        self.mutation_roots = list(mutation_roots or [])
        self.mutation_roots_unknown = mutation_roots_unknown


class BashStart(
    BaseTool[BashStartArgs, StartResult, BashToolConfig, BaseToolState],
    ToolUIData[BashStartArgs, StartResult],
):
    @classmethod
    def path_sensitive_patterns(cls, config: BaseToolConfig) -> tuple[str, ...]:
        return Bash.path_sensitive_patterns(config)

    def resolve_permission(self, args: BashStartArgs) -> PermissionContext | None:
        return ShellPermissionResolver(
            self.config,
            self.cwd,
            self.workspace,
            self.path_authority,
            self.dangerous_command_authorities_getter,
            # The same trust-gated environment the background job's
            # subprocess receives; the egress rule consults it for injected
            # non-loopback proxies (v0.5.9 R1).
            injected_env=project_env_for_cwd(self.cwd),
        ).resolve_permission(args)

    async def run(
        self, args: BashStartArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[StartResult, None]:
        if ctx is None or ctx.background_jobs is None:
            raise ToolError("Background jobs unavailable in this context")
        # A2: the same executed-string rewriting as Bash.run(): a
        # whole-call-eligible exempt checkout executes with
        # --no-overwrite-ignore so an ignored-file collision fails closed.
        # The rewritten string is the executed command (execute_command=...
        # below); args.command stays original, so the stored job summary
        # keeps the original command string.
        command = shell_command_checkout_no_overwrite_ignore(
            args.command, workspace=self.workspace, authority=self.path_authority
        )
        if command is None:
            command = args.command
        # R7: surface the started command's write-classified paths (redirect
        # targets, write-option values, write-classified command operands)
        # on the result, so the submission secret scan covers a started
        # background job's write surface exactly as a foreground run's. The
        # footprint is computed before the admission attempt: a failed or
        # cancelled start may still have spawned the command, and its write
        # surface must reach the submission scan regardless of the outcome.
        # A1: the mutation-roots footprint (the resolved work-tree roots of
        # the command's checkout/switch invocations) is collected alongside
        # it — a mutation-scope footprint that never enters the submission
        # scan's path list.
        write_paths, write_paths_unknown = shell_command_write_paths(
            command,
            cwd=self.cwd,
            workspace=self.workspace,
            authority=self.path_authority,
        )
        mutation_roots, mutation_roots_unknown = shell_command_mutation_roots(
            command,
            cwd=self.cwd,
            workspace=self.workspace,
            authority=self.path_authority,
        )
        # F1pkg1(c): the background transport cannot consume a literal argv
        # plan, so at execution it re-derives the bound plan (the S2
        # precedent: a mutation between the permission check and execution is
        # seen) and hard-denies when one binds — it never falls back to
        # running the original shell text in the background shell.
        try:
            background_plan = glob_execution_plan_for_command(
                args.command,
                cwd=self.cwd,
                workspace=self.workspace,
                authority=self.path_authority,
                scratchpad_roots=effective_scratchpad_roots(self.path_authority),
                workspace_roots=frozenset(self.workspace.authorized_roots),
            )
        except GlobPlanDenial as exc:
            raise FailedJobStart(
                glob_plan_denial_text(exc, args.command),
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        if background_plan is not None:
            raise FailedJobStart(
                "Command refused: it binds a literal argv plan, which the "
                "background shell transport cannot consume; run it with the "
                "foreground bash tool",
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            )
        # F1pkg3: an admitted quoted-variable word binds to a literal argv plan
        # for the same reason; the background shell cannot consume one, so it
        # hard-denies rather than run the shell text the admission replaced.
        try:
            quoted_word_plan = quoted_word_execution_plan_for_command(
                args.command,
                scratchpad_roots=effective_scratchpad_roots(self.path_authority),
                workspace_roots=frozenset(self.workspace.authorized_roots),
            )
        except QuotedWordPlanDenial as exc:
            raise FailedJobStart(
                quoted_word_plan_denial_text(exc, args.command),
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        if quoted_word_plan is not None:
            raise FailedJobStart(
                "Command refused: it binds a literal argv plan, which the "
                "background shell transport cannot consume; run it with the "
                "foreground bash tool",
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            )
        # S4b1b + S4b2a: the background transport is this process's own
        # executor (attested by construction), so a standalone ssh call with
        # a positive destination (literal IP, or a hostname resolved uniquely
        # non-local) runs with the pinned plan as its executed command: the
        # resolved address + fixed host-key identity ride into the spawned
        # argv, and the grant must coincide with the pin. A denied ssh form
        # hard-denies here, before the reservation starts.
        try:
            remote_plan = remote_exec_plan_for_command(
                args.command,
                cwd=self.cwd,
                workspace=self.workspace,
                authority=self.path_authority,
                scratchpad_roots=effective_scratchpad_roots(self.path_authority),
                workspace_roots=frozenset(self.workspace.authorized_roots),
            )
        except RemoteExecDenial as exc:
            raise FailedJobStart(
                f"Command refused: {exc}",
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        if remote_plan is not None:
            try:
                authorities = self.dangerous_command_authorities_getter()
            except Exception:
                authorities = ()
            if not remote_exec_authority_admits(
                authorities, self.cwd, remote_plan.destination
            ):
                raise FailedJobStart(
                    "Command refused: no network.remote-exec grant covers the "
                    f"destination {remote_plan.destination.key} in this repository",
                    write_paths=write_paths,
                    write_paths_unknown=write_paths_unknown,
                    mutation_roots=mutation_roots,
                    mutation_roots_unknown=mutation_roots_unknown,
                )
            command = remote_plan.command
        try:
            result = await ctx.background_jobs.start(
                args, cwd=self.cwd, execute_command=command
            )
        except asyncio.CancelledError as exc:
            # The registry finishes its pre-commit recovery before
            # re-raising; the executor's cancellation handler copies the
            # write footprint onto the cancelled result event.
            raise CancelledShellCommand(
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        except UnknownShellExecutorError as exc:
            # F1pkg1(c): the background spawn site consumed the pinned
            # executor contract at execution and hard-denied an unknown
            # executor; the reservation rolled back, nothing ran.
            raise FailedJobStart(
                f"Unable to start background job: {exc}",
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        except (ValueError, OSError) as exc:
            raise FailedJobStart(
                "Unable to start background job",
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        yield result.model_copy(
            update={
                "write_paths": list(write_paths),
                "write_paths_unknown": write_paths_unknown,
                "mutation_roots": list(mutation_roots),
                "mutation_roots_unknown": mutation_roots_unknown,
            }
        )

    @classmethod
    def format_result_display(cls, result: StartResult) -> ToolResultDisplay:
        return ToolResultDisplay(
            success=True, message=f"{result.job.job_id}: {result.job.state}"
        )

    @classmethod
    def project_result(cls, result: StartResult) -> JsonValue:
        return cast(JsonValue, result.model_dump(mode="json"))

    @classmethod
    def get_status_text(cls) -> str:
        return "Starting background job"
