from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast

from pydantic import JsonValue

from chartreux.core.background_jobs import BashStopArgs, StopResult
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
)
from chartreux.core.tools.ui import ToolResultDisplay, ToolUIData


class BashStop(
    BaseTool[BashStopArgs, StopResult, BaseToolConfig, BaseToolState],
    ToolUIData[BashStopArgs, StopResult],
):
    async def run(
        self, args: BashStopArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[StopResult, None]:
        if ctx is None or ctx.background_jobs is None:
            raise ToolError("Background jobs unavailable in this context")
        try:
            yield await ctx.background_jobs.stop(args)
        except (ValueError, OSError) as exc:
            raise ToolError("Unable to stop background job") from exc

    @classmethod
    def format_result_display(cls, result: StopResult) -> ToolResultDisplay:
        return ToolResultDisplay(
            success=True, message=f"{result.job.job_id}: {result.job.state}"
        )

    @classmethod
    def project_result(cls, result: StopResult) -> JsonValue:
        return cast(JsonValue, result.model_dump(mode="json"))

    @classmethod
    def get_status_text(cls) -> str:
        return "Stopping background job"
