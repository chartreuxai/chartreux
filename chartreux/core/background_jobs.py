"""Root-lifetime managed Linux jobs; no tool or AgentLoop integration.

Reservation, commit, leader observation, capture completion and ownership
settlement are separate transitions. Pending reservations and committed jobs
hold capacity until rollback/settlement. A nonzero leader exit is ``exited``;
``failed`` denotes infrastructure/capture failure, not a command's exit code.
Public opaque IDs are never reused. Creator identities are runtime incarnation
objects, compared by identity, never transcript/session IDs.

One lock owns admission, retention, revision and lifecycle. Producers perform
UTF-8 decoding/redaction outside it, with an in-flight marker preventing a
second producer or premature settlement. No lock spans a wait or OS operation.
"""

from __future__ import annotations

import asyncio
import codecs
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Literal
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
)

from chartreux.core.events import BackgroundJobsChangedEvent
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.secret_redaction import (
    REDACTED_PLACEHOLDER,
    IncrementalRedactor,
    ScrubPolicy,
    bind_policy,
    current_policy,
    known_secret_values,
    redact,
    redact_shell_source,
)
from chartreux.core.utils.async_subprocess import (
    ProcessGroupIdentityChanged,
    ProcessGroupOwnership,
    registered_process_group,
    release_process_group,
    terminate_process_group,
)
from chartreux.core.utils.shell import spawn_shell_command

MAX_ALLOCATIONS = 8
MAX_JOBS = 32
MAX_RECORDS = 64
MAX_RECORD_BYTES = 4096
MAX_OUTPUT_BYTES = 256 * 1024
MAX_COMMAND_BYTES = 64 * 1024  # Existing shell-analysis source budget.
JobState = Literal["starting", "running", "stopping", "exited", "failed", "stopped"]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BashStartArgs(_Args):
    command: str
    label: str | None = Field(default=None, max_length=64)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > MAX_COMMAND_BYTES:
            raise ValueError(
                "Command must be nonempty and within the shell byte budget"
            )
        return value

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str | None) -> str | None:
        if value is not None and any(c in value for c in "\r\n\v\f\x85\u2028\u2029"):
            raise ValueError("Label must be single-line")
        return value


class BashReadArgs(_Args):
    job_id: str = Field(min_length=1, max_length=128)
    cursor: StrictInt = Field(default=0, ge=0)
    max_bytes: StrictInt = Field(default=4096, ge=4096, le=64000)
    wait_seconds: float = Field(default=0, ge=0, le=30, allow_inf_nan=False)


class BashStopArgs(_Args):
    job_id: str = Field(min_length=1, max_length=128)


class BashListArgs(_Args):
    include_finished: StrictBool = False


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OutputRecord(_Result):
    sequence: int = Field(ge=0)
    text: str

    @field_validator("text")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_RECORD_BYTES:
            raise ValueError("Record exceeds UTF-8 byte bound")
        return value


class JobSummary(_Result):
    job_id: str = Field(max_length=128)
    command: str = Field(max_length=1024)
    label: str | None = Field(default=None, max_length=64)
    state: JobState
    exit_code: int | None = None
    output_complete: bool = False
    output_incomplete: bool = False
    error: str | None = Field(default=None, max_length=256)


class StartResult(_Result):
    job: JobSummary
    next_cursor: Literal[0] = 0


class ReadResult(_Result):
    job: JobSummary
    records: tuple[OutputRecord, ...] = Field(default=(), max_length=MAX_RECORDS)
    first_cursor: int = Field(ge=0)
    next_cursor: int = Field(ge=0)
    end_cursor: int = Field(ge=0)
    lost_records: int = Field(default=0, ge=0)


class StopResult(_Result):
    job: JobSummary
    already_finished: bool


class ListResult(_Result):
    jobs: tuple[JobSummary, ...] = Field(max_length=MAX_JOBS)


@dataclass
class BackgroundJob:
    """Registry-private state. A single capture producer owns decoder state."""

    job_id: str
    creator: object = field(repr=False)
    command: str = field(repr=False)
    label: str | None = field(repr=False)
    policy: ScrubPolicy = field(repr=False)
    redactor: IncrementalRedactor = field(repr=False)
    decoder: codecs.IncrementalDecoder = field(repr=False)
    state: JobState = "starting"
    committed: bool = False
    allocated: bool = True
    leader_observed: bool = False
    exit_code: int | None = None
    output_complete: bool = False
    output_incomplete: bool = False
    error: str | None = None
    producing: bool = False
    records: deque[OutputRecord] = field(default_factory=deque, repr=False)
    retained_bytes: int = 0
    end_cursor: int = 0
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    owner: ProcessGroupOwnership | None = field(default=None, repr=False)
    supervisor: asyncio.Task[None] | None = field(default=None, repr=False)
    capture_task: asyncio.Task[None] | None = field(default=None, repr=False)
    stop_task: asyncio.Task[None] | None = field(default=None, repr=False)
    stop_requested: bool = False
    admission: asyncio.Task[StartResult] | None = field(default=None, repr=False)
    mask_cache: tuple[int, int, frozenset[str], frozenset[str], bool] | None = field(
        default=None, repr=False
    )

    @property
    def first_cursor(self) -> int:
        return self.records[0].sequence if self.records else self.end_cursor


def _chunks(text: str) -> list[str]:
    """Split only at UTF-8 codepoint boundaries, never truncate a record."""
    chunks: list[str] = []
    start = size = 0
    for index, char in enumerate(text):
        width = len(char.encode("utf-8"))
        if size + width > MAX_RECORD_BYTES:
            chunks.append(text[start:index])
            start, size = index, 0
        size += width
    if start < len(text):
        chunks.append(text[start:])
    return chunks


def _metadata(text: str, limit: int, *, shell: bool = False) -> str:
    # Sanitize complete source BEFORE truncation, which could hide quoting.
    safe = redact_shell_source(text) if shell else text
    if safe is None:
        safe = REDACTED_PLACEHOLDER
    safe = redact(safe)
    return safe[:limit]


class BackgroundJobRegistry:
    """Synchronous state boundary and registry-owned asynchronous supervision.

    Only settled records may be evicted. Escaped sessions are not adopted;
    capture and transport cleanup remain bounded even when they retain pipes.
    """

    def __init__(
        self,
        *,
        generation: int = 0,
        event_sink: Callable[[BackgroundJobsChangedEvent], None] | None = None,
    ) -> None:
        self.generation = generation
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._jobs: dict[str, BackgroundJob] = {}
        self._revision = 0
        self._closed = False
        self._changed = asyncio.Event()
        self._root_creator = object()
        self._identity = uuid.uuid4().hex
        self._next_id = 0
        self._admissions: set[asyncio.Task[StartResult]] = set()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(job.allocated for job in self._jobs.values())

    @property
    def lifetime_id(self) -> str:
        return self._identity

    @property
    def committed_active_count(self) -> int:
        with self._lock:
            return sum(job.committed and job.allocated for job in self._jobs.values())

    def _publish_state(self) -> None:
        # Called under the registry boundary after commit/settlement only. The
        # root's nonblocking sink remains owned by this registry after child close.
        if self.event_sink is not None:
            self.event_sink(
                BackgroundJobsChangedEvent(
                    root_lifetime_id=self._identity,
                    generation=self.generation,
                    revision=self._revision,
                    active_count=self.committed_active_count,
                )
            )

    def retire(self) -> None:
        """Invalidate an empty lifetime synchronously before an identity commit."""
        with self._lock:
            if self.active_count:
                raise ValueError("Background jobs still own live or pending processes")
            self._closed = True
            self._jobs.clear()
            self._notify(revision=True)

    def root_port(self) -> BackgroundJobsPort:
        return BackgroundJobsPort(self, self._root_creator, root=True)

    def creator_port(self, incarnation: object) -> BackgroundJobsPort:
        return BackgroundJobsPort(self, incarnation, root=False)

    def _notify(self, *, revision: bool = False) -> None:
        if revision:
            self._revision += 1
        self._changed.set()
        self._changed = asyncio.Event()

    def _authorized(self, port: BackgroundJobsPort, job_id: str) -> BackgroundJob:
        job = self._jobs.get(job_id)
        if (
            port._registry is not self
            or self._closed
            or job is None
            or (not port._root and job.creator is not port._creator)
        ):
            # Do not echo guessed IDs or reveal sibling existence.
            raise ValueError("Background job unavailable")
        return job

    def reserve(self, port: BackgroundJobsPort, args: BashStartArgs) -> str:
        policy = current_policy().capture_for_redaction()
        with bind_policy(policy):
            command = _metadata(args.command, 1024, shell=True)
            label = _metadata(args.label, 64) if args.label is not None else None
        job = BackgroundJob(
            "",
            port._creator,
            command,
            label,
            policy,
            IncrementalRedactor(policy),
            codecs.getincrementaldecoder("utf-8")("replace"),
        )
        with self._lock:
            if self._closed or port._registry is not self:
                raise ValueError("Background jobs closed or foreign port")
            if sum(j.allocated for j in self._jobs.values()) >= MAX_ALLOCATIONS:
                raise ValueError("Background job capacity reached")
            if len(self._jobs) >= MAX_JOBS:
                oldest = next(
                    (key for key, j in self._jobs.items() if not j.allocated), None
                )
                if oldest is None:
                    raise ValueError("Background job retention capacity reached")
                del self._jobs[oldest]
            job_id = f"{self._identity}-{self._next_id:x}"
            self._next_id += 1
            job.job_id = job_id
            self._jobs[job_id] = job
            self._notify(revision=True)
        return job_id

    async def start(
        self, port: BackgroundJobsPort, args: BashStartArgs, *, cwd: Path | None = None
    ) -> StartResult:
        job_id = self.reserve(port, args)
        task = asyncio.create_task(self._launch(job_id, args.command, cwd))
        with self._lock:
            self._jobs[job_id].admission = task
            self._admissions.add(task)
        task.add_done_callback(lambda done: self._admission_done(job_id, done))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            # Pre-commit recovery remains owned even under repeated cancellation.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    def _admission_done(self, job_id: str, task: asyncio.Task[StartResult]) -> None:
        with self._lock:
            self._admissions.discard(task)
            if task.cancelled():
                job = self._jobs.get(job_id)
                if (
                    job is not None
                    and job.allocated
                    and not job.committed
                    and job.process is None
                ):
                    self.rollback(job_id)
            else:
                task.exception()

    async def _launch(self, job_id: str, command: str, cwd: Path | None) -> StartResult:
        spawn = asyncio.create_task(
            spawn_shell_command(command, cwd=cwd, merge_stderr=True)
        )
        try:
            proc = await asyncio.shield(spawn)
        except BaseException:
            recovery = asyncio.create_task(self._recover_launch(job_id, spawn))
            while not recovery.done():
                try:
                    await asyncio.shield(recovery)
                except asyncio.CancelledError:
                    continue
            recovery.result()
            raise
        # No await between handle acquisition and registry/task commit. Close
        # cannot interleave here, and caller cancellation cannot own the tasks.
        job = self._jobs[job_id]
        job.process = proc
        job.owner = registered_process_group(proc)
        if self._closed:
            # Close may have sealed admission while the spawn was completing.
            # Retain the allocation until this recovered handle is reclaimed.
            cleanup = asyncio.create_task(self._reclaim_launch(job))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise ValueError("Background jobs closed")
        result = self.commit(job_id)
        job.capture_task = asyncio.create_task(self._capture_process(job))
        self._supervise(job)
        return result

    async def _recover_launch(
        self, job_id: str, spawn: asyncio.Task[asyncio.subprocess.Process]
    ) -> None:
        try:
            proc = await spawn
        except Exception:
            self.rollback(job_id)
            return
        job = self._jobs[job_id]
        job.process = proc
        job.owner = registered_process_group(proc)
        await self._reclaim_launch(job)

    async def _reclaim_launch(self, job: BackgroundJob) -> None:
        assert job.owner is not None and job.process is not None

        # Drain during termination even for an admission that never committed.
        async def drain() -> None:
            assert job.process is not None and job.process.stdout is not None
            while await job.process.stdout.read(64 * 1024):
                pass

        drain_task = asyncio.create_task(drain())
        try:
            await terminate_process_group(job.owner)
            job.process._transport.close()  # type: ignore[attr-defined]
            await asyncio.wait_for(job.process.wait(), 0.5)
            release_process_group(job.process)
            self.rollback(job.job_id)
        finally:
            drain_task.cancel()
            await asyncio.gather(drain_task, return_exceptions=True)

    def _supervise(self, job: BackgroundJob) -> asyncio.Task[None]:
        task = asyncio.create_task(self._supervisor(job))
        job.supervisor = task
        # Keep failed handles for retry, but consume background task exceptions.
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        return task

    async def _capture_process(self, job: BackgroundJob) -> None:
        assert job.process is not None and job.process.stdout is not None
        abnormal = False
        try:
            while data := await job.process.stdout.read(64 * 1024):
                self.capture(job.job_id, data)
        except asyncio.CancelledError:
            abnormal = True
        except Exception:
            abnormal = True
        finally:
            try:
                self.capture(job.job_id, b"", final=True, abnormal=abnormal)
            except Exception:
                # Decoder/redactor failure must never flush unresolved raw text.
                with self._lock:
                    job.producing = False
                    job.output_complete = job.output_incomplete = True
                    self._notify()

    def _group_stop(self, job: BackgroundJob) -> asyncio.Task[None]:
        if job.stop_task is None or (
            job.stop_task.done() and job.stop_task.exception() is not None
        ):
            assert job.owner is not None
            job.stop_task = asyncio.create_task(terminate_process_group(job.owner))
        return job.stop_task

    async def _supervisor(self, job: BackgroundJob) -> None:
        assert job.process is not None and job.capture_task is not None
        proc = job.process
        try:
            # returncode is updated by the child watcher independently of EOF.
            leader_deadline: float | None = None
            loop = asyncio.get_running_loop()
            while proc.returncode is None:
                assert job.owner is not None
                if job.stop_task is None:
                    await asyncio.to_thread(job.owner.members)
                if job.stop_task is not None and job.stop_task.done():
                    job.stop_task.result()
                    if leader_deadline is None:
                        leader_deadline = loop.time() + 0.5
                    if loop.time() >= leader_deadline:
                        raise RuntimeError("Process leader did not settle")
                await asyncio.sleep(0.01 if job.stop_task is not None else 0.25)
            if not job.leader_observed:
                self.observe_leader(job.job_id, proc.returncode)
            await asyncio.shield(self._group_stop(job))
            try:
                await asyncio.wait_for(asyncio.shield(job.capture_task), 0.5)
            except TimeoutError:
                job.capture_task.cancel()
                await job.capture_task
            # Escaped setsid descendants are outside ownership. Closing our
            # transport bounds local FD/task lifetime, not their lifetime.
            proc._transport.close()  # type: ignore[attr-defined]
            await asyncio.wait_for(proc.wait(), 0.5)
            release_process_group(proc)
            self.settle(job.job_id, stopped=job.stop_requested)
        except ProcessGroupIdentityChanged:
            # Historical ownership is no longer retryable. Abandon signaling,
            # close local capture/transport, and release the allocation as failed.
            assert job.owner is not None
            job.owner.valid = False
            job.capture_task.cancel()
            await asyncio.gather(job.capture_task, return_exceptions=True)
            if not job.output_complete:
                self.capture(job.job_id, b"", final=True, abnormal=True)
            proc._transport.close()  # type: ignore[attr-defined]
            release_process_group(proc)
            with self._lock:
                job.leader_observed = True
                job.exit_code = proc.returncode
            self.settle(job.job_id, error="Process group identity changed")
        except Exception:
            with self._lock:
                job.error = "Background job cleanup failed"
                self._notify()
            raise

    async def stop(self, port: BackgroundJobsPort, args: BashStopArgs) -> StopResult:
        with self._lock:
            job = self._authorized(port, args.job_id)
            pending = not job.committed
            admission = job.admission
            if pending:
                job.stop_requested = True
                if admission is not None:
                    admission.cancel()
                elif job.process is None:
                    self.rollback(job.job_id)
        if pending:
            if admission is not None:
                await asyncio.shield(asyncio.gather(admission, return_exceptions=True))
            if job.allocated:
                raise RuntimeError("Background job admission cleanup failed")
            return StopResult(
                job=self._summary(job, current_policy()), already_finished=False
            )
        with self._lock:
            already_finished = not job.allocated or (
                job.process is not None and job.process.returncode is not None
            )
            if job.allocated:
                if not already_finished and not job.stop_requested:
                    job.stop_requested = True
                    job.state = "stopping"
                    self._notify()
                self._group_stop(job)
                if job.supervisor is None or job.supervisor.done():
                    self._supervise(job)
            supervisor = job.supervisor
        if job.allocated and supervisor is not None:
            await asyncio.shield(supervisor)
        return StopResult(
            job=self._summary(job, current_policy()), already_finished=already_finished
        )

    async def aclose(self) -> None:
        with self._lock:
            self._closed = True
            self._notify()
            if self._close_task is None or (
                self._close_task.done() and self._close_task.exception() is not None
            ):
                self._close_task = asyncio.create_task(self._close())
                self._close_task.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
            task = self._close_task
        await asyncio.shield(task)

    async def _close(self) -> None:
        admissions = tuple(self._admissions)
        for admission in admissions:
            admission.cancel()
        await asyncio.gather(*admissions, return_exceptions=True)
        supervisors: list[asyncio.Task[None]] = []
        for job in tuple(self._jobs.values()):
            if not job.committed:
                if job.process is not None:
                    supervisors.append(asyncio.create_task(self._reclaim_launch(job)))
                else:
                    self.rollback(job.job_id)
            elif job.allocated:
                assert job.process is not None
                if job.process.returncode is None:
                    job.stop_requested = True
                    job.state = "stopping"
                self._group_stop(job)
                if job.supervisor is None or job.supervisor.done():
                    self._supervise(job)
                assert job.supervisor is not None
                supervisors.append(job.supervisor)
        results = await asyncio.gather(*supervisors, return_exceptions=True)
        if any(isinstance(result, BaseException) for result in results):
            raise RuntimeError("Background jobs cleanup failed; retry close")
        with self._lock:
            for job in self._jobs.values():
                job.records.clear()
                job.retained_bytes = 0
            self._jobs.clear()
            self._notify(revision=True)

    def rollback(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.committed or job.producing:
                raise ValueError("Cannot roll back a committed launch")
            if job.stop_requested:
                job.committed = True
                job.state = "stopped"
                job.allocated = False
                job.output_complete = True
            else:
                del self._jobs[job_id]
            self._notify(revision=True)

    def commit(self, job_id: str) -> StartResult:
        with self._lock:
            job = self._jobs[job_id]
            if self._closed or job.committed:
                raise ValueError("Launch cannot commit")
            job.committed = True
            job.state = "running"
            self._notify(revision=True)
            self._publish_state()
        return StartResult(job=self._summary(job, current_policy()))

    def observe_leader(self, job_id: str, exit_code: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if not job.committed or job.leader_observed or not job.allocated:
                raise ValueError("Invalid leader observation")
            job.leader_observed = True
            job.exit_code = exit_code
            self._notify()

    def capture(
        self, job_id: str, data: bytes, *, final: bool = False, abnormal: bool = False
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if not job.committed or job.output_complete or job.producing:
                raise ValueError("Invalid capture transition")
            job.producing = True
        try:
            # Bound transient decoded/redacted/record-building buffers even if
            # an adapter supplies a giant write. Retention is enforced per batch.
            for offset in range(0, max(1, len(data)), 64 * 1024):
                last = offset + 64 * 1024 >= len(data)
                decoded = job.decoder.decode(
                    data[offset : offset + 64 * 1024], final=final and last
                )
                safe = job.redactor.feed(
                    decoded, final=final and last, abnormal=abnormal
                )
                records = _chunks(safe)
                with self._lock:
                    for text in records:
                        record = OutputRecord(sequence=job.end_cursor, text=text)
                        job.end_cursor += 1
                        job.records.append(record)
                        job.retained_bytes += len(text.encode("utf-8"))
                        while (
                            len(job.records) > MAX_RECORDS
                            or job.retained_bytes > MAX_OUTPUT_BYTES
                        ):
                            job.retained_bytes -= len(
                                job.records.popleft().text.encode("utf-8")
                            )
                    if records:
                        self._notify()
        except BaseException:
            with self._lock:
                job.producing = False
                job.output_incomplete = True
                self._notify()
            raise
        with self._lock:
            job.producing = False
            job.output_complete = final
            job.output_incomplete |= abnormal
            if final:
                self._notify()

    def settle(
        self, job_id: str, *, error: str | None = None, stopped: bool = False
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            policy = job.policy
        with bind_policy(ScrubPolicy.for_redaction((policy,), current_policy())):
            safe_error = _metadata(error, 256) if error is not None else None
        with self._lock:
            if not job.allocated:
                return  # Exactly-once ownership release.
            if not job.leader_observed or not job.output_complete or job.producing:
                raise ValueError("Owned cleanup has not settled")
            job.error = safe_error
            job.state = (
                "failed"
                if error is not None or job.output_incomplete
                else "stopped"
                if stopped
                else "exited"
            )
            job.allocated = False
            self._notify(revision=True)
            self._publish_state()

    @staticmethod
    def _snapshot(job: BackgroundJob) -> JobSummary:
        return JobSummary(
            job_id=job.job_id,
            command=job.command,
            label=job.label,
            state=job.state,
            exit_code=job.exit_code,
            output_complete=job.output_complete,
            output_incomplete=job.output_incomplete,
            error=job.error,
        )

    @staticmethod
    def _sanitize_summary(
        summary: JobSummary, policy: ScrubPolicy, live: ScrubPolicy
    ) -> JobSummary:
        with bind_policy(ScrubPolicy.for_redaction((policy,), live)):
            return summary.model_copy(
                update={
                    "command": _metadata(summary.command, 1024, shell=True),
                    "label": _metadata(summary.label, 64)
                    if summary.label is not None
                    else None,
                    "error": _metadata(summary.error, 256)
                    if summary.error is not None
                    else None,
                }
            )

    def _summary(self, job: BackgroundJob, live: ScrubPolicy) -> JobSummary:
        with self._lock:
            summary, policy = self._snapshot(job), job.policy
        return self._sanitize_summary(summary, policy, live)

    def _page(  # noqa: PLR0914 - immutable boundary and outward-mask cache snapshots
        self, port: BackgroundJobsPort, args: BashReadArgs
    ) -> tuple[ReadResult, asyncio.Event]:
        live = current_policy()
        with self._lock:
            job = self._authorized(port, args.job_id)
            if not job.committed:
                raise ValueError("Background job unavailable")
            first, end = job.first_cursor, job.end_cursor
            if args.cursor > end:
                raise ValueError("Cursor exceeds output end")
            retained = tuple(job.records)
            summary = self._snapshot(job)
            policy = job.policy
            event = self._changed
            revision = self._revision
            cached = job.mask_cache
        # Recheck the ENTIRE retained boundary context, not just this page.
        # If any new match crosses records, mask the retained window; mapping a
        # transformed substring back to individual immutable cursors is unsafe.
        # Missing predecessor context also fails closed. Expansion is accounted
        # before page selection, and every outward record still fits 4096 bytes.
        with bind_policy(ScrubPolicy.for_redaction((job.policy,), live)):
            values = known_secret_values()
            key = (revision, end, values, secret_redaction.credential_env_var_names())
            if cached is not None and cached[:4] == key:
                mask = cached[4]
            else:
                text = "".join(record.text for record in retained)
                mask = (
                    first > 0
                    or IncrementalRedactor(current_policy()).feed(text, final=True)
                    != text
                )
                with self._lock:
                    job.mask_cache = (*key, mask)
        cursor = max(args.cursor, first)
        selected: list[OutputRecord] = []
        size = 0
        for record in retained:
            if record.sequence < cursor:
                continue
            outward = (
                OutputRecord(sequence=record.sequence, text=REDACTED_PLACEHOLDER)
                if mask
                else record
            )
            width = len(outward.text.encode("utf-8"))
            if size + width > args.max_bytes:
                break
            selected.append(outward)
            size += width
            cursor = record.sequence + 1
        return ReadResult(
            job=self._sanitize_summary(summary, policy, live),
            records=tuple(selected),
            first_cursor=first,
            next_cursor=cursor,
            end_cursor=end,
            lost_records=max(0, first - args.cursor),
        ), event

    async def read(self, port: BackgroundJobsPort, args: BashReadArgs) -> ReadResult:
        deadline = asyncio.get_running_loop().time() + args.wait_seconds
        while True:
            page, event = self._page(port, args)
            remaining = deadline - asyncio.get_running_loop().time()
            if page.records or page.job.output_complete or remaining <= 0:
                return page
            try:
                await asyncio.wait_for(event.wait(), remaining)
            except TimeoutError:
                return self._page(port, args)[0]

    def list(self, port: BackgroundJobsPort, args: BashListArgs) -> ListResult:
        with self._lock:
            if self._closed or port._registry is not self:
                raise ValueError("Background jobs closed or foreign port")
            jobs = tuple(
                job
                for job in self._jobs.values()
                if job.committed
                and (port._root or job.creator is port._creator)
                and (args.include_finished or job.allocated)
            )
        return ListResult(
            jobs=tuple(self._summary(job, current_policy()) for job in jobs)
        )


class BackgroundJobsPort:
    """Trusted runtime capability; root sees all, creator sees its incarnation.

    Constructors/identities are not model arguments. A borrowed port cannot
    close the registry; only its root-lifetime owner holds that operation.
    """

    def __init__(
        self, registry: BackgroundJobRegistry, creator: object, *, root: bool
    ) -> None:
        self._registry = registry
        self._creator = creator
        self._root = root

    @property
    def active_count(self) -> int:
        """Root-wide unsettled allocations, including evicted creators' jobs."""
        return self._registry.active_count

    @property
    def lifetime_id(self) -> str:
        return self._registry.lifetime_id

    @property
    def generation(self) -> int:
        return self._registry.generation

    @property
    def committed_active_count(self) -> int:
        return self._registry.committed_active_count

    def borrow(self) -> BackgroundJobsPort:
        """Create a fresh trusted creator incarnation, never a stored-session alias."""
        with self._registry._lock:
            if self._registry._closed:
                raise ValueError("Background jobs closed")
            return self._registry.creator_port(object())

    async def start(
        self, args: BashStartArgs, *, cwd: Path | None = None
    ) -> StartResult:
        return await self._registry.start(self, args, cwd=cwd)

    async def stop(self, args: BashStopArgs) -> StopResult:
        return await self._registry.stop(self, args)

    def reserve(self, args: BashStartArgs) -> str:
        return self._registry.reserve(self, args)

    async def read(self, args: BashReadArgs) -> ReadResult:
        return await self._registry.read(self, args)

    def list(self, args: BashListArgs | None = None) -> ListResult:
        return self._registry.list(self, args or BashListArgs())

    def authorize(self, job_id: str) -> None:
        with self._registry._lock:
            self._registry._authorized(self, job_id)
