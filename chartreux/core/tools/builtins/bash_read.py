from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast

from pydantic import JsonValue

from chartreux.core.background_jobs import BashReadArgs, ReadResult
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
)
from chartreux.core.tools.ui import ToolResultDisplay, ToolUIData


class BashRead(
    BaseTool[BashReadArgs, ReadResult, BaseToolConfig, BaseToolState],
    ToolUIData[BashReadArgs, ReadResult],
):
    async def run(
        self, args: BashReadArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ReadResult, None]:
        if ctx is None or ctx.background_jobs is None:
            raise ToolError("Background jobs unavailable in this context")
        try:
            yield await ctx.background_jobs.read(args)
        except (ValueError, OSError) as exc:
            raise ToolError(
                "Unable to read background job; check the job ID and cursor"
            ) from exc

    @classmethod
    def format_result_display(cls, result: ReadResult) -> ToolResultDisplay:
        return ToolResultDisplay(
            success=True, message=f"{result.job.job_id}: cursor {result.next_cursor}"
        )

    @classmethod
    def project_result(cls, result: ReadResult) -> JsonValue:
        return cast(JsonValue, result.model_dump(mode="json"))

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading background job"
