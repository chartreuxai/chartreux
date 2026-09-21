from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from chartreux.core.agent_loop._title_cadence import TitleCadence, TitleGenTicket
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.events import (
    BackgroundWorkEvent,
    BaseEvent,
    SessionTitleUpdatedEvent,
)
from chartreux.core.llm_models import LLMMessage
from chartreux.core.session.title_policy import DEFAULT_TITLE_POLICY, TitlePolicy
from chartreux.observability.logging import logger


@dataclass(frozen=True, slots=True)
class TitleScheduleInputs:
    messages: tuple[LLMMessage, ...]
    session_id: str
    turn_completing: bool
    enabled: bool
    disabled_by_test_switch: bool
    logging_enabled: bool
    title_is_manual: bool
    periodic: bool


@dataclass(frozen=True, slots=True)
class TitleGateInputs:
    config: ChartreuxConfigSchema
    previous_title: str | None


class TitleEventSink(Protocol):
    def publish(self, event: BaseEvent) -> None: ...


class AutoTitleWriter(Protocol):
    async def refresh_auto_title(
        self, title: str, *, expected_session_id: str
    ) -> bool: ...


class ReadTitleAtGate(Protocol):
    def __call__(self) -> TitleGateInputs: ...


class ScheduledTitle(Protocol):
    @property
    def started_event(self) -> BackgroundWorkEvent: ...

    @property
    def start_gate(self) -> asyncio.Event: ...

    def release(self) -> None: ...

    def abort_delivery(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _ScheduledTitle:
    started_event: BackgroundWorkEvent
    start_gate: asyncio.Event
    ticket: TitleGenTicket
    controller: TitleController
    task: asyncio.Task[None]

    def release(self) -> None:
        self.start_gate.set()

    def abort_delivery(self) -> None:
        self.controller._abort_delivery(self)


class TitleController:
    """Own background session-title cadence and generation lifecycle."""

    def __init__(
        self,
        *,
        read_at_gate: ReadTitleAtGate,
        writer: AutoTitleWriter,
        event_sink: TitleEventSink,
        policy: TitlePolicy = DEFAULT_TITLE_POLICY,
    ) -> None:
        self._read_at_gate = read_at_gate
        self._writer = writer
        self._event_sink = event_sink
        self._policy = policy
        self._cadence = TitleCadence(
            refresh_every=policy.refresh_every_steps,
            capped_max_generations=policy.capped_max_generations,
            initial_max_steps=policy.initial_max_steps,
        )
        self._task: asyncio.Task[None] | None = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def cadence(self) -> TitleCadence:
        return self._cadence

    @cadence.setter
    def cadence(self, cadence: TitleCadence) -> None:
        self._cadence = cadence

    def schedule(self, inputs: TitleScheduleInputs) -> ScheduledTitle | None:
        if self._generation_is_blocked(inputs):
            return None
        ticket = self._cadence.begin_if_due(
            periodic=inputs.periodic, turn_completing=inputs.turn_completing
        )
        if ticket is None:
            return None

        work_id = str(uuid4())
        start_gate = asyncio.Event()
        task = asyncio.create_task(
            self._generate_title_task(
                inputs, ticket=ticket, work_id=work_id, start_gate=start_gate
            ),
            name="vibe-session-title",
        )
        self._task = task
        return _ScheduledTitle(
            started_event=BackgroundWorkEvent(
                work_id=work_id,
                kind="session_title",
                phase="started",
                session_id=inputs.session_id,
            ),
            start_gate=start_gate,
            ticket=ticket,
            controller=self,
            task=task,
        )

    def cancel(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()

    def reset(self) -> None:
        self.cancel()
        self._cadence.reset()

    def mark_compaction(self) -> None:
        self._cadence.mark_compaction()

    async def aclose(self) -> None:
        task = self._task
        self.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _generation_is_blocked(self, inputs: TitleScheduleInputs) -> bool:
        if (
            inputs.disabled_by_test_switch
            or not inputs.enabled
            or not inputs.logging_enabled
            or inputs.title_is_manual
        ):
            return True
        return self._task is not None and not self._task.done()

    def _abort_delivery(self, scheduled: _ScheduledTitle) -> None:
        if self._task is scheduled.task:
            self._cadence.restore(scheduled.ticket)
            self.cancel()

    async def _generate_title_task(
        self,
        inputs: TitleScheduleInputs,
        *,
        ticket: TitleGenTicket,
        work_id: str,
        start_gate: asyncio.Event,
    ) -> None:
        from chartreux.core.session.title_model import generate_session_title

        try:
            await start_gate.wait()
            gate_inputs = self._read_at_gate()
            title = await generate_session_title(
                inputs.messages,
                config=gate_inputs.config,
                previous_title=gate_inputs.previous_title,
                policy=self._policy,
            )
            if title is None:
                self._cadence.restore(ticket)
                return
            changed = await self._writer.refresh_auto_title(
                title, expected_session_id=inputs.session_id
            )
            if not changed:
                return
            self._event_sink.publish(
                SessionTitleUpdatedEvent(title=title, session_id=inputs.session_id)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._cadence.restore(ticket)
            logger.warning("Background session title update failed", exc_info=True)
        finally:
            self._event_sink.publish(
                BackgroundWorkEvent(
                    work_id=work_id,
                    kind="session_title",
                    phase="finished",
                    session_id=inputs.session_id,
                )
            )
