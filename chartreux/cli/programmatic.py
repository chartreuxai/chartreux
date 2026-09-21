from __future__ import annotations

import asyncio
from contextlib import aclosing
from enum import StrEnum, auto
import json
import sys
from typing import TextIO

from pydantic import BaseModel

from chartreux.app_server.events import (
    AppServerEvent,
    CallbackRequested,
    HistoryEntryAdded,
    HistoryEntryUpdated,
)
from chartreux.app_server.local import LocalHarness, LocalHarnessOptions
from chartreux.app_server.models import (
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicTurnStopReason,
)
from chartreux.app_server.session import AppServerSession
from chartreux.observability.logging import logger


class OutputFormat(StrEnum):
    TEXT = auto()
    JSON = auto()
    STREAMING = auto()


class ProgrammaticLimitError(RuntimeError):
    pass


class ProgrammaticOutput:
    def __init__(
        self, output_format: OutputFormat, stream: TextIO | None = None
    ) -> None:
        self._format = output_format
        self._stream = stream or sys.stdout
        self._emitted: set[str] = set()

    def start(self, history: list[PublicHistoryEntry]) -> None:
        if self._format is not OutputFormat.STREAMING:
            return
        for entry in history:
            self._emit_completed(entry)

    def consume(self, event: AppServerEvent) -> None:
        if self._format is not OutputFormat.STREAMING:
            return
        match event:
            case HistoryEntryAdded(entry=entry) | HistoryEntryUpdated(entry=entry):
                self._emit_completed(entry)
            case _:
                pass

    def finalize(self, history: list[PublicHistoryEntry]) -> str | None:
        if self._format is OutputFormat.STREAMING:
            return None
        if self._format is OutputFormat.JSON:
            history_json = [
                entry.model_dump(mode="json", by_alias=True) for entry in history
            ]
            json.dump(history_json, self._stream, indent=2, ensure_ascii=False)
            self._stream.write("\n")
            self._stream.flush()
            return None
        return _last_assistant_text(history)

    def _emit_completed(self, entry: PublicHistoryEntry) -> None:
        if (
            entry.id in self._emitted
            or entry.generation_status is not PublicEntryGenerationStatus.COMPLETED
        ):
            return
        self._emitted.add(entry.id)
        self._write_json(entry)

    def _write_json(self, value: BaseModel) -> None:
        json.dump(
            value.model_dump(mode="json", by_alias=True),
            self._stream,
            ensure_ascii=False,
        )
        self._stream.write("\n")
        self._stream.flush()

    def _print(self, text: str) -> None:
        print(text, file=self._stream)


def run_programmatic(
    *,
    harness_options: LocalHarnessOptions,
    prompt: str,
    output_format: OutputFormat = OutputFormat.TEXT,
) -> str | None:
    logger.info("USER: %s", prompt)
    output = ProgrammaticOutput(output_format)

    async def run() -> str | None:
        session = await LocalHarness(harness_options).start()
        try:
            await session.resources.runtime.wait_until_ready()
            await _warn_if_workspace_untrusted(session)
            output.start(session.history)
            async with aclosing(session.act(prompt)) as events:
                async for event in events:
                    output.consume(event)
                    if isinstance(event, CallbackRequested):
                        await session.deny_callback(event.callback)

            turn = next(reversed(session.state.turns or []), None)
            if turn is not None and turn.stop_reason is PublicTurnStopReason.LIMIT:
                raise ProgrammaticLimitError(
                    _last_assistant_text(session.history)
                    or "The configured conversation limit was reached"
                )
            return output.finalize(session.history)
        finally:
            await session.close()

    return asyncio.run(run())


async def _warn_if_workspace_untrusted(session: AppServerSession) -> None:
    trust = await session.resources.workspace.trust_status()
    details = trust.details
    if trust.status != "untrusted" or details is None:
        return
    detected_files = list(
        dict.fromkeys([*details.detected_files, *details.repo_detected_files])
    )
    if not detected_files:
        return
    print(
        f"Warning: {details.cwd} is not trusted; project configuration "
        f"({', '.join(detected_files)}) will be ignored. Re-run with --trust "
        "to trust this folder temporarily.",
        file=sys.stderr,
    )


def _last_assistant_text(history: list[PublicHistoryEntry]) -> str | None:
    return next(
        (
            entry.text
            for entry in reversed(history)
            if isinstance(entry, PublicMessageEntry)
            and entry.role == "assistant"
            and entry.text
        ),
        None,
    )
