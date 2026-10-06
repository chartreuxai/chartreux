from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast

from pydantic import JsonValue

from chartreux.core.background_jobs import BashListArgs, ListResult
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
)
from chartreux.core.tools.ui import ToolResultDisplay, ToolUIData


class BashList(
    BaseTool[BashListArgs, ListResult, BaseToolConfig, BaseToolState],
    ToolUIData[BashListArgs, ListResult],
):
    async def run(
        self, args: BashListArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ListResult, None]:
        if ctx is None or ctx.background_jobs is None:
            raise ToolError("Background jobs unavailable in this context")
        try:
            yield ctx.background_jobs.list(args)
        except (ValueError, OSError) as exc:
            raise ToolError("Unable to list background jobs") from exc

    @classmethod
    def format_result_display(cls, result: ListResult) -> ToolResultDisplay:
        return ToolResultDisplay(success=True, message=f"{len(result.jobs)} jobs")

    @classmethod
    def project_result(cls, result: ListResult) -> JsonValue:
        return cast(JsonValue, result.model_dump(mode="json"))

    @classmethod
    def get_status_text(cls) -> str:
        return "Listing background jobs"
