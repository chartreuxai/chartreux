from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast

from pydantic import JsonValue

from chartreux.core.background_jobs import BashStartArgs, StartResult
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
)
from chartreux.core.tools.builtins.bash import Bash, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolResultDisplay, ToolUIData


class BashStart(
    BaseTool[BashStartArgs, StartResult, BashToolConfig, BaseToolState],
    ToolUIData[BashStartArgs, StartResult],
):
    @classmethod
    def path_sensitive_patterns(cls, config: BaseToolConfig) -> tuple[str, ...]:
        return Bash.path_sensitive_patterns(config)

    def resolve_permission(self, args: BashStartArgs) -> PermissionContext | None:
        return ShellPermissionResolver(
            self.config, self.cwd, self.workspace, self.path_authority
        ).resolve_permission(args)

    async def run(
        self, args: BashStartArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[StartResult, None]:
        if ctx is None or ctx.background_jobs is None:
            raise ToolError("Background jobs unavailable in this context")
        try:
            yield await ctx.background_jobs.start(args, cwd=self.cwd)
        except (ValueError, OSError) as exc:
            raise ToolError("Unable to start background job") from exc

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
