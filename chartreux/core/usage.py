"""Immutable local accounting contracts and component-wise call pricing.

Accounting counts preserve reporting presence: None means unreported, while 0
means explicitly reported zero. Adapters must retain that distinction across
stream chunks; bookkeeping chunks do not report usage. These contracts do not
change LLMUsage's legacy numeric defaults or AgentStats' conversation semantics.
No request/response content or arbitrary metadata belongs in a ledger record.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from enum import StrEnum
import logging
from pathlib import Path
import stat
from threading import Lock
from typing import Annotated, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from chartreux.core._usage_calendar import (
    CalendarWindow,
    CalendarWindows,
    calendar_windows,
    resolve_local_timezone,
    utc_now,
)
from chartreux.core.config import ModelConfig

TokenCount = Annotated[int, Field(ge=0, strict=True)]
UsdAmount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class UsagePurpose(StrEnum):
    CONVERSATION = "conversation"
    COMPACTION = "compaction"
    TITLE = "title"
    WORKTREE_NAMING = "worktree-naming"


class UsageOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    REFUSED = "refused"


class UsageState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class SnapshotState(StrEnum):
    LOADING = "loading"
    UNAVAILABLE = "unavailable"
    READY = "ready"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UsagePrices(_ImmutableModel):
    """Call-time USD rates per million tokens; None is unknown, 0 is free."""

    input: UsdAmount | None = None
    output: UsdAmount | None = None
    cached_input: UsdAmount | None = None


class UsageTokens(_ImmutableModel):
    """Reported component counts, without fabricating absent fields as zero.

    Input includes cached input. Complete presence requires all three counts;
    partial presence has at least one reported count; missing has none. A
    terminated stream may still have partial usage even with all counts present
    (its final total was not received). CallAccountingOutcome carries that state.
    """

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    cached_input_tokens: TokenCount | None = None

    @property
    def presence_state(self) -> UsageState:
        counts = (self.input_tokens, self.output_tokens, self.cached_input_tokens)
        if all(count is None for count in counts):
            return UsageState.MISSING
        if any(count is None for count in counts):
            return UsageState.PARTIAL
        return UsageState.COMPLETE


class UsageAttribution(_ImmutableModel):
    """Invocation-local identity, frozen before a request can be attempted."""

    root_session_id: str
    session_id: str
    parent_session_id: str | None = None
    agent_role: str
    agent_profile: str | None = None
    purpose: UsagePurpose = UsagePurpose.CONVERSATION
    model: str
    provider: str
    wire_name: str
    project_key: str | None


class CallAccountingOutcome(UsageTokens):
    """Content-free finalization, including partial usage on failure/interruption."""

    outcome: UsageOutcome
    usage_state: UsageState

    @model_validator(mode="after")
    def _validate_presence(self) -> CallAccountingOutcome:
        presence = self.presence_state
        if self.usage_state == UsageState.MISSING and presence != UsageState.MISSING:
            raise ValueError("Missing usage cannot contain reported counts")
        if self.usage_state == UsageState.COMPLETE and presence != UsageState.COMPLETE:
            raise ValueError("Complete usage requires all component counts")
        if self.usage_state == UsageState.PARTIAL and presence == UsageState.MISSING:
            raise ValueError("Partial usage requires a reported count")
        return self


class UsageRecord(UsageAttribution, CallAccountingOutcome):
    schema_version: Literal[1] = 1
    record_id: str
    occurred_at: datetime
    prices_usd_per_million: UsagePrices
    known_cost_usd: UsdAmount
    has_unknown_cost: bool
    currency: Literal["USD"] = "USD"

    @field_validator("occurred_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("occurred_at must be timezone-aware UTC")
        return value.astimezone(UTC)


class ComponentPricing(_ImmutableModel):
    tokens: TokenCount | None
    price_known: bool
    known_cost_usd: UsdAmount
    has_unknown_cost: bool


class UsagePricing(_ImmutableModel):
    input: ComponentPricing
    cached_input: ComponentPricing
    output: ComponentPricing
    known_cost_usd: UsdAmount
    has_unknown_cost: bool


def capture_prices(model: ModelConfig) -> UsagePrices:
    """Freeze configured rates before request dispatch, never when appending.

    An absent cached rate is unknown, not a fallback to the input rate.
    """
    return UsagePrices(
        input=model.input_price if model.input_price_known else None,
        output=model.output_price if model.output_price_known else None,
        cached_input=(
            model.cached_input_price if model.cached_input_price_known else None
        ),
    )


def price_usage(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None,
    prices: UsagePrices,
) -> UsagePricing:
    """Price known components, retaining an unknown remainder.

    Uncached input is max(0, input - cached), without clamping cached to input.
    If either input count is unreported, uncached input cannot be established.
    Reported zero costs zero even with an unknown price; unreported counts have
    unknown cost even at a free rate (usage coverage is still incomplete).
    """
    counts = UsageTokens(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )
    uncached = (
        max(0, counts.input_tokens - counts.cached_input_tokens)
        if counts.input_tokens is not None and counts.cached_input_tokens is not None
        else None
    )

    def component(tokens: int | None, price: float | None) -> ComponentPricing:
        return ComponentPricing(
            tokens=tokens,
            price_known=price is not None,
            known_cost_usd=(tokens * price / 1_000_000)
            if tokens is not None and price is not None
            else 0.0,
            has_unknown_cost=tokens is None or (tokens != 0 and price is None),
        )

    input_component = component(uncached, prices.input)
    cached_component = component(counts.cached_input_tokens, prices.cached_input)
    output_component = component(counts.output_tokens, prices.output)
    # Sum before dividing, preserving the gateway's existing arithmetic order.
    known_cost = 0.0
    for tokens, price in (
        (uncached, prices.input),
        (counts.cached_input_tokens, prices.cached_input),
        (counts.output_tokens, prices.output),
    ):
        if tokens and price is not None:
            known_cost += tokens * price
    return UsagePricing(
        input=input_component,
        cached_input=cached_component,
        output=output_component,
        known_cost_usd=known_cost / 1_000_000,
        has_unknown_cost=any(
            item.has_unknown_cost
            for item in (input_component, cached_component, output_component)
        ),
    )


class CoverageWarningCode(StrEnum):
    WRITE_FAILED = "write-failed"
    UNREADABLE = "unreadable"
    MALFORMED_RECORD = "malformed-record"
    TORN_TAIL = "torn-tail"
    UNSUPPORTED_SCHEMA = "unsupported-schema"


class CoverageWarning(_ImmutableModel):
    """Structured coverage degradation, never raw exception text."""

    code: CoverageWarningCode
    root_session_id: str | None = None
    record_id: str | None = None


class UsageComponentSnapshot(_ImmutableModel):
    """Known token/cost lower bounds plus independent completeness flags.

    has_known_cost means at least one contribution is priced, including a free
    contribution. It must not be inferred from known_cost_usd > 0.
    """

    tokens: TokenCount = 0
    has_unknown_tokens: bool = False
    known_cost_usd: UsdAmount = 0.0
    has_known_cost: bool = False
    has_unknown_cost: bool = False


class UsageAggregateSnapshot(_ImmutableModel):
    """READY + request_count=0 is valid empty, not loading or unavailable.

    For nonempty snapshots: no known cost + unknown cost is entirely unknown;
    both flags indicate a known lower bound plus unknown remainder; known cost
    without unknown cost and a zero amount is genuinely priced zero. Warnings
    describe missing ledger coverage independently of unknown recorded costs.
    """

    state: SnapshotState = SnapshotState.READY
    request_count: TokenCount = 0
    input_tokens: TokenCount = 0
    output_tokens: TokenCount = 0
    cached_input_tokens: TokenCount = 0
    has_unknown_tokens: bool = False
    known_cost_usd: UsdAmount = 0.0
    has_known_cost: bool = False
    has_unknown_cost: bool = False
    input: UsageComponentSnapshot = Field(default_factory=UsageComponentSnapshot)
    cached_input: UsageComponentSnapshot = Field(default_factory=UsageComponentSnapshot)
    output: UsageComponentSnapshot = Field(default_factory=UsageComponentSnapshot)
    warnings: tuple[CoverageWarning, ...] = ()
    currency: Literal["USD"] = "USD"


class UsageModelSnapshot(UsageAggregateSnapshot):
    """Deployment row grouped by (model, provider, wire_name), not alias alone."""

    model: str
    provider: str
    wire_name: str


class AccountingSink(Protocol):
    async def __call__(self, record: UsageRecord, /) -> None:
        """Settle one immutable record without raising into inference control flow."""


# Blocking writer core. The async sink/lifecycle layer runs append off-loop and
# owns cancellation shielding, draining, and callbacks after settlement.
class UsageWriteDisposition(StrEnum):
    APPENDED = "appended"
    ALREADY_PRESENT = "already-present"
    FAILED = "failed"


@dataclass(frozen=True)
class UsageWriteResult:
    disposition: UsageWriteDisposition
    warning: CoverageWarning | None = None


class UsageWriter:
    """Thread-safe durable writer, independent of transcript/session logging.

    Record IDs come from the immutable record, generated once by its caller.
    append is deliberately synchronous: the async layer must not run it on the
    event loop. A failed result is uncertain, not proof that bytes were absent;
    retrying the same record reconciles under the filesystem lock.
    """

    def __init__(self, usage_dir: Path | None = None) -> None:
        self._usage_dir = usage_dir
        self._append_lock = Lock()

    def append(self, record: UsageRecord) -> UsageWriteResult:
        from chartreux.core._usage_io import append_usage_record
        from chartreux.core.paths import USAGE_DIR

        with self._append_lock:
            try:
                content = (record.model_dump_json() + "\n").encode("utf-8")
                usage_dir = (
                    self._usage_dir if self._usage_dir is not None else USAGE_DIR.path
                )
                # A bounded retry reconciles failures before/after publication.
                # The record and its identity are never regenerated on retry.
                for attempt in range(2):
                    try:
                        appended = append_usage_record(
                            usage_dir, record.root_session_id, record.record_id, content
                        )
                    except Exception:
                        if attempt == 1:
                            raise
                    else:
                        return UsageWriteResult(
                            UsageWriteDisposition.APPENDED
                            if appended
                            else UsageWriteDisposition.ALREADY_PRESENT
                        )
            except Exception:
                warning = CoverageWarning(
                    code=CoverageWarningCode.WRITE_FAILED,
                    root_session_id=record.root_session_id,
                    record_id=record.record_id,
                )
                # Never log exception strings (they can contain arbitrary data).
                logging.getLogger(__name__).warning(
                    "Usage ledger write failed; recorded usage coverage is degraded"
                )
                return UsageWriteResult(UsageWriteDisposition.FAILED, warning)
        raise AssertionError("Unreachable writer disposition")


async def _wait_for_usage_settlement[T](task: asyncio.Task[T]) -> T:
    """Delay even repeated caller cancellation until the retained task settles."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)
        # Retrieve any exception without replacing the caller's cancellation.
        if not task.cancelled():
            task.exception()
        raise


type UsageWriteCallback = Callable[[UsageRecord, UsageWriteResult], None]


class AsyncUsageWriter:
    """Event-loop-owned facade over the blocking, deduplicating UsageWriter.

    Subscribers run synchronously on the event loop after filesystem settlement,
    with no writer/filesystem locks held. They receive successful invalidations
    (including reconciled records) and failed dispositions with coverage warnings.
    Callback failures cannot escape into inference control flow.

    drain is a write barrier, not a producer barrier. Owners must stop producers
    before closing; aclose rejects new writes and settles all accepted writes.
    """

    def __init__(
        self, usage_dir: Path | None = None, *, writer: UsageWriter | None = None
    ) -> None:
        if writer is not None and usage_dir is not None:
            raise ValueError("An existing writer cannot be combined with usage_dir")
        self._writer = writer if writer is not None else UsageWriter(usage_dir)
        self._pending: set[asyncio.Task[UsageWriteResult]] = set()
        self._callbacks: list[UsageWriteCallback] = []
        self._closed = False

    def subscribe(self, callback: UsageWriteCallback) -> Callable[[], None]:
        self._callbacks.append(callback)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._callbacks.remove(callback)

        return unsubscribe

    def _publish(self, record: UsageRecord, result: UsageWriteResult) -> None:
        for callback in tuple(self._callbacks):
            try:
                callback(record, result)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Usage ledger settlement callback failed"
                )

    @staticmethod
    def _failure(record: UsageRecord) -> UsageWriteResult:
        return UsageWriteResult(
            UsageWriteDisposition.FAILED,
            CoverageWarning(
                code=CoverageWarningCode.WRITE_FAILED,
                root_session_id=record.root_session_id,
                record_id=record.record_id,
            ),
        )

    async def _append(self, record: UsageRecord) -> UsageWriteResult:
        try:
            # The sync core retries uncertain appends using this exact record ID
            # under flock, including all required durability barriers.
            result = await asyncio.to_thread(self._writer.append, record)
        except Exception:
            # Also contain executor submission/facade failures, not just IO errors.
            logging.getLogger(__name__).warning(
                "Usage ledger write failed; recorded usage coverage is degraded"
            )
            result = self._failure(record)
        self._publish(record, result)
        return result

    async def append(self, record: UsageRecord) -> UsageWriteResult:
        if self._closed:
            result = self._failure(record)
            self._publish(record, result)
            return result
        task = asyncio.create_task(self._append(record))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return await _wait_for_usage_settlement(task)

    async def __call__(self, record: UsageRecord, /) -> None:
        """AccountingSink adapter; append exposes the disposition when needed."""
        await self.append(record)

    async def _drain(self) -> None:
        while self._pending:
            await asyncio.gather(*tuple(self._pending))

    async def drain(self) -> None:
        await _wait_for_usage_settlement(asyncio.create_task(self._drain()))

    async def aclose(self) -> None:
        self._closed = True
        await self.drain()


@dataclass(frozen=True)
class UsageReadSnapshot:
    """Unfiltered, deduplicated stored records for deterministic aggregation.

    LOADING precedes the first reconciliation. A missing store is valid empty;
    UNAVAILABLE means directory enumeration failed. File-level damage leaves a
    READY partial result with warnings. No historical costs are recalculated.
    """

    records: tuple[UsageRecord, ...] = ()
    warnings: tuple[CoverageWarning, ...] = ()
    state: SnapshotState = SnapshotState.LOADING


@dataclass(frozen=True)
class _UsageFileEntry:
    # Device/inode also detect replacement with identical size and mtime.
    fingerprint: tuple[int, int, int, int] | None
    records: tuple[UsageRecord, ...] = ()
    warnings: tuple[CoverageWarning, ...] = ()


class UsageReader:
    """Blocking stat-based cache, independent of transcript validity and dates.

    Run reconcile/refresh_invalidated in a worker thread. snapshot, invalidate,
    and on_settlement are cheap thread-safe seams for a future service. Wire
    ``writer.subscribe(reader.on_settlement)`` for writers sharing this store,
    then refresh_invalidated to update only locally affected paths (no scan).

    Scans serialize, but never hold the state lock during IO. An invalidation
    advances an epoch: a pass started earlier cannot commit stale entries or
    clear dirty paths. Its returned snapshot may still be old; the next refresh
    consumes the retained invalidation. Background scheduling belongs to the
    service, not this reader.
    """

    def __init__(self, usage_dir: Path | None = None) -> None:
        self._usage_dir = usage_dir
        self._state_lock = Lock()
        self._scan_lock = Lock()
        self._entries: dict[Path, _UsageFileEntry] = {}
        self._dirty: set[Path] = set()
        self._epoch = 0
        self._snapshot = UsageReadSnapshot()
        self._store_warnings: tuple[CoverageWarning, ...] = ()
        self._write_warnings: dict[str, CoverageWarning] = {}

    @property
    def usage_dir(self) -> Path:
        from chartreux.core.paths import USAGE_DIR

        return self._usage_dir if self._usage_dir is not None else USAGE_DIR.path

    @property
    def snapshot(self) -> UsageReadSnapshot:
        with self._state_lock:
            return self._snapshot

    def invalidate(self, path: Path) -> None:
        """Mark one ledger dirty without listing the home or performing IO."""
        with self._state_lock:
            self._epoch += 1
            self._dirty.add(path)

    def on_settlement(self, record: UsageRecord, result: UsageWriteResult) -> None:
        # Failed writes are uncertain too: some or all bytes may be present.
        path = self.usage_dir / record.root_session_id / "usage.jsonl"
        with self._state_lock:
            self._epoch += 1
            self._dirty.add(path)
            if result.warning is not None:
                self._write_warnings[record.record_id] = result.warning
            elif result.disposition != UsageWriteDisposition.FAILED:
                self._write_warnings.pop(record.record_id, None)

    @staticmethod
    def _read_entry(
        path: Path, cached: _UsageFileEntry | None
    ) -> _UsageFileEntry | None:
        from chartreux.core._usage_io import parse_usage_jsonl, read_usage_file

        def warning(code: CoverageWarningCode) -> CoverageWarning:
            return CoverageWarning(code=code, root_session_id=path.parent.name)

        try:
            info = path.stat()
            fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if cached is not None and cached.fingerprint == fingerprint:
                return cached
            parsed = parse_usage_jsonl(read_usage_file(path))
        except FileNotFoundError:
            return None
        except OSError:
            # Do not cache failures by stat: retry permission/transient errors.
            return _UsageFileEntry(
                None, warnings=(warning(CoverageWarningCode.UNREADABLE),)
            )

        warnings: list[CoverageWarning] = []
        if parsed.malformed_lines:
            warnings.append(warning(CoverageWarningCode.MALFORMED_RECORD))
        if parsed.torn_tail:
            warnings.append(warning(CoverageWarningCode.TORN_TAIL))
        records: list[UsageRecord] = []
        for value in parsed.records:
            if value.get("schema_version", 1) != 1:
                warnings.append(warning(CoverageWarningCode.UNSUPPORTED_SCHEMA))
                continue
            try:
                records.append(UsageRecord.model_validate(value))
            except ValidationError:
                warnings.append(warning(CoverageWarningCode.MALFORMED_RECORD))
        return _UsageFileEntry(
            fingerprint, tuple(records), tuple(dict.fromkeys(warnings))
        )

    def reconcile(self) -> UsageReadSnapshot:
        """Enumerate every root, rereading only changed or invalidated files."""
        return self._refresh(scan=True)

    def refresh_invalidated(self) -> UsageReadSnapshot:
        """Consume local invalidations without enumerating the store."""
        return self._refresh(scan=False)

    def _refresh(self, *, scan: bool) -> UsageReadSnapshot:
        with self._scan_lock:
            with self._state_lock:
                epoch = self._epoch
                dirty = self._dirty.copy()
                entries = self._entries.copy()
                state = self._snapshot.state
                store_warnings = self._store_warnings
                write_warnings = tuple(self._write_warnings.values())
            if scan:
                paths: set[Path] = set()
                warnings: list[CoverageWarning] = []
                state = SnapshotState.READY
                try:
                    children = list(self.usage_dir.iterdir())
                except FileNotFoundError:
                    children = []
                except OSError:
                    children = []
                    state = SnapshotState.UNAVAILABLE
                    warnings.append(
                        CoverageWarning(code=CoverageWarningCode.UNREADABLE)
                    )
                for child in children:
                    try:
                        if stat.S_ISDIR(child.stat().st_mode):
                            paths.add(child / "usage.jsonl")
                    except OSError:
                        warnings.append(
                            CoverageWarning(
                                code=CoverageWarningCode.UNREADABLE,
                                root_session_id=child.name,
                            )
                        )
                store_warnings = tuple(warnings)
                entries = {
                    path: entry for path, entry in entries.items() if path in paths
                }
            else:
                paths = dirty
            for path in sorted(paths):
                entry = self._read_entry(
                    path, None if path in dirty else entries.get(path)
                )
                if entry is None:
                    entries.pop(path, None)
                else:
                    entries[path] = entry
            records: dict[str, UsageRecord] = {}
            all_warnings = list(store_warnings) + list(write_warnings)
            # Stable first-wins dedup, independent of enumeration order. Build
            # off-lock so settlement callbacks never wait on a large record set.
            for path in sorted(entries):
                entry = entries[path]
                all_warnings.extend(entry.warnings)
                for record in entry.records:
                    records.setdefault(record.record_id, record)
            snapshot = UsageReadSnapshot(
                records=tuple(records.values()),
                warnings=tuple(dict.fromkeys(all_warnings)),
                state=state,
            )
            with self._state_lock:
                if self._epoch != epoch:
                    return self._snapshot
                self._entries = entries
                self._dirty.difference_update(dirty)
                self._store_warnings = store_warnings
                self._snapshot = snapshot
                return self._snapshot


# Deterministic aggregation. Scheduling, reconciliation and revisions are owned
# by the lifecycle layer; this section never opens files or consults a catalog.
type UsageWindow = Literal["day", "week", "month"]


@dataclass(frozen=True)
class UsageWindowSnapshot:
    boundaries: CalendarWindow
    totals: UsageAggregateSnapshot


@dataclass(frozen=True)
class UsageWindowSummaries:
    day: UsageWindowSnapshot
    week: UsageWindowSnapshot
    month: UsageWindowSnapshot


@dataclass(frozen=True)
class UsageServiceSnapshot:
    """Projection input with service revision; the host supplies project identity.

    Component snapshots describe token breakdown, not apportioned spending:
    records store a total cost, not component costs. Nonzero component costs are
    therefore unknown rather than recalculated from historical or current rates.
    """

    calendar: CalendarWindows
    summaries: UsageWindowSummaries
    window: UsageWindow
    selected: UsageAggregateSnapshot
    models: tuple[UsageModelSnapshot, ...]
    project_filter: str | None
    warnings: tuple[CoverageWarning, ...]
    revision: int = 0

    @property
    def as_of(self) -> datetime:
        return self.calendar.as_of


class _UsageAccumulator:
    def __init__(self) -> None:
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.uncached_tokens = 0
        self.unknown_input = False
        self.unknown_output = False
        self.unknown_cached = False
        self.unknown_uncached = False
        self.unknown_tokens = False
        self.cost = 0.0
        self.known_cost = False
        self.unknown_cost = False

    def add(self, record: UsageRecord) -> None:
        self.requests += 1
        self.input_tokens += record.input_tokens or 0
        self.output_tokens += record.output_tokens or 0
        self.cached_tokens += record.cached_input_tokens or 0
        self.unknown_input |= record.input_tokens is None
        self.unknown_output |= record.output_tokens is None
        self.unknown_cached |= record.cached_input_tokens is None
        partial = record.usage_state != UsageState.COMPLETE
        self.unknown_tokens |= partial
        if record.input_tokens is None or record.cached_input_tokens is None:
            self.unknown_uncached = True
        else:
            self.uncached_tokens += max(
                0, record.input_tokens - record.cached_input_tokens
            )
        # A partial stream's observed counts remain lower bounds even when all
        # three fields were reported. Do not silently mark them complete.
        self.unknown_input |= partial
        self.unknown_output |= partial
        self.unknown_cached |= partial
        self.unknown_uncached |= partial
        self.cost += record.known_cost_usd
        self.unknown_cost |= record.has_unknown_cost
        # Known zero is not inferred from an amount > 0. A partially priced
        # record can have a priced (even free) contribution and an unknown rest.
        uncached = (
            max(0, record.input_tokens - record.cached_input_tokens)
            if record.input_tokens is not None
            and record.cached_input_tokens is not None
            else None
        )
        self.known_cost |= (
            not record.has_unknown_cost
            or record.known_cost_usd > 0
            or any(
                count is not None and rate is not None
                for count, rate in (
                    (uncached, record.prices_usd_per_million.input),
                    (
                        record.cached_input_tokens,
                        record.prices_usd_per_million.cached_input,
                    ),
                    (record.output_tokens, record.prices_usd_per_million.output),
                )
            )
        )

    def snapshot(
        self, state: SnapshotState, warnings: tuple[CoverageWarning, ...]
    ) -> UsageAggregateSnapshot:
        def component(tokens: int, unknown: bool) -> UsageComponentSnapshot:
            return UsageComponentSnapshot(
                tokens=tokens,
                has_unknown_tokens=unknown,
                # Only a reported, complete zero establishes component cost
                # without inventing an allocation of the stored record total.
                has_known_cost=self.requests > 0 and tokens == 0 and not unknown,
                has_unknown_cost=tokens > 0 or unknown,
            )

        return UsageAggregateSnapshot(
            state=state,
            request_count=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_tokens,
            has_unknown_tokens=self.unknown_tokens,
            known_cost_usd=self.cost,
            has_known_cost=self.known_cost,
            has_unknown_cost=self.unknown_cost,
            input=component(self.uncached_tokens, self.unknown_uncached),
            cached_input=component(self.cached_tokens, self.unknown_cached),
            output=component(self.output_tokens, self.unknown_output),
            warnings=warnings,
        )


def aggregate_usage(
    records: Iterable[UsageRecord],
    *,
    as_of: datetime,
    timezone: tzinfo | str,
    window: UsageWindow = "day",
    project_key: str | None = None,
    state: SnapshotState = SnapshotState.READY,
    warnings: tuple[CoverageWarning, ...] = (),
) -> UsageServiceSnapshot:
    """Pure single-pass aggregation of a deduplicated reader snapshot.

    Completion-time attribution uses half-open UTC calendar boundaries. as_of
    selects those windows, not an additional truncation boundary. Call this in
    a worker thread for large inputs; no reader or lifecycle state is accessed.
    """
    if window not in {"day", "week", "month"}:
        raise ValueError(f"Unknown usage window: {window}")
    calendar = calendar_windows(as_of, timezone)
    boundaries = (calendar.day, calendar.week, calendar.month)
    accumulators = {item.kind: _UsageAccumulator() for item in boundaries}
    deployments: dict[tuple[str, str, str], _UsageAccumulator] = {}
    for record in records:
        if project_key is not None and record.project_key != project_key:
            continue
        for boundary in boundaries:
            if boundary.start_utc <= record.occurred_at < boundary.end_utc:
                accumulators[boundary.kind].add(record)
                if boundary.kind == window:
                    key = (record.model, record.provider, record.wire_name)
                    deployments.setdefault(key, _UsageAccumulator()).add(record)
    totals = {
        kind: accumulator.snapshot(state, warnings)
        for kind, accumulator in accumulators.items()
    }
    models = tuple(
        UsageModelSnapshot(
            **accumulator.snapshot(state, warnings).model_dump(),
            model=key[0],
            provider=key[1],
            wire_name=key[2],
        )
        for key, accumulator in sorted(deployments.items())
    )
    return UsageServiceSnapshot(
        calendar=calendar,
        summaries=UsageWindowSummaries(
            day=UsageWindowSnapshot(calendar.day, totals["day"]),
            week=UsageWindowSnapshot(calendar.week, totals["week"]),
            month=UsageWindowSnapshot(calendar.month, totals["month"]),
        ),
        window=window,
        selected=totals[window],
        models=models,
        project_filter=project_key,
        warnings=warnings,
    )


class UsageScheduledCall(Protocol):
    def cancel(self) -> None: ...


type UsageScheduler = Callable[[float, Callable[[], None]], UsageScheduledCall]
type UsageUpdateCallback = Callable[[UsageServiceSnapshot], None]


class UsageService:
    """Cached, revisioned projections with an opt-in background lifecycle.

    start() schedules the cold scan and returns immediately. Once started, read
    is a cheap cached global-day read; use aread for other filters/windows and
    fresh rollover results. aread does not scan: transports explicitly reconcile
    on screen open/refresh and their client-owned polling cadence. Without start,
    read retains the synchronous, IO-free projection seam for supplied snapshots.

    Subscriptions receive global day snapshots, monotonically revisioned. Local
    settlements dirty only their file and share a fixed 250ms publication window.
    Calendar changes are checked on reads/reconciliation (no idle polling timer).
    Owners must stop producers before aclose: publishing stops first, then attached
    writers drain. Writers are not closed, since their ownership is external.
    """

    def __init__(
        self,
        source: UsageReader | UsageReadSnapshot,
        *,
        clock: Callable[[], datetime] = utc_now,
        timezone_resolver: Callable[[], tzinfo] = resolve_local_timezone,
        scheduler: UsageScheduler | None = None,
    ) -> None:
        self._source = source
        self._clock = clock
        self._timezone_resolver = timezone_resolver
        self._scheduler = scheduler
        self._started = False
        self._closed = False
        self._revision = 0
        self._generation = 0
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._initial: asyncio.Task[None] | None = None
        self._debounce: UsageScheduledCall | None = None
        self._callbacks: list[UsageUpdateCallback] = []
        self._writers: dict[AsyncUsageWriter, Callable[[], None]] = {}
        self._published_source: UsageReadSnapshot | None = None
        self._published: UsageServiceSnapshot | None = None

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def state(self) -> SnapshotState:
        return (
            self._published.selected.state
            if self._published is not None
            else SnapshotState.LOADING
        )

    def _snapshot(self) -> UsageReadSnapshot:
        return (
            self._source.snapshot
            if isinstance(self._source, UsageReader)
            else self._source
        )

    def _project(
        self,
        snapshot: UsageReadSnapshot,
        calendar: CalendarWindows,
        window: UsageWindow = "day",
        project_key: str | None = None,
    ) -> UsageServiceSnapshot:
        return aggregate_usage(
            snapshot.records,
            as_of=calendar.as_of,
            timezone=calendar.timezone_fingerprint.zone,
            window=window,
            project_key=project_key,
            state=snapshot.state,
            warnings=snapshot.warnings,
        )

    def _calendar(self) -> CalendarWindows:
        return calendar_windows(self._clock(), self._timezone_resolver())

    def _calendar_changed(self, calendar: CalendarWindows) -> bool:
        previous = self._published
        return previous is None or any((
            previous.calendar.timezone_fingerprint != calendar.timezone_fingerprint,
            previous.calendar.day != calendar.day,
            previous.calendar.week != calendar.week,
            previous.calendar.month != calendar.month,
        ))

    def _changed(self, snapshot: UsageReadSnapshot, calendar: CalendarWindows) -> bool:
        return snapshot != self._published_source or self._calendar_changed(calendar)

    def _spawn(self, scan: bool = False) -> asyncio.Task[None]:
        task = asyncio.create_task(self._refresh(scan=scan))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def start(self) -> asyncio.Task[None]:
        """Nonblocking and idempotent; the returned task is the readiness barrier."""
        if self._closed:
            raise RuntimeError("Usage service is closed")
        if self._initial is None:
            self._started = True
            self._initial = self._spawn(scan=True)
        return self._initial

    async def wait_ready(self) -> None:
        await asyncio.shield(self.start())

    def subscribe(self, callback: UsageUpdateCallback) -> Callable[[], None]:
        if self._closed:
            raise RuntimeError("Usage service is closed")
        self._callbacks.append(callback)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._callbacks.remove(callback)

        return unsubscribe

    def attach_writer(self, writer: AsyncUsageWriter) -> None:
        """Subscribe once to each writer sharing the reader's usage directory."""
        if self._closed:
            raise RuntimeError("Usage service is closed")
        if not isinstance(self._source, UsageReader):
            raise TypeError("Writer subscriptions require a UsageReader")
        if writer not in self._writers:
            self._writers[writer] = writer.subscribe(self._on_settlement)

    def detach_writer(self, writer: AsyncUsageWriter) -> None:
        """Release a retired writer's subscription after its final settlement."""
        unsubscribe = self._writers.pop(writer, None)
        if unsubscribe is not None:
            unsubscribe()

    def _on_settlement(self, record: UsageRecord, result: UsageWriteResult) -> None:
        if self._closed:
            return
        assert isinstance(self._source, UsageReader)
        self._source.on_settlement(record, result)
        self._generation += 1
        if self._started and self._debounce is None:
            schedule = self._scheduler or asyncio.get_running_loop().call_later
            self._debounce = schedule(0.25, self._flush)

    def _flush(self) -> None:
        self._debounce = None
        if not self._closed:
            self._spawn()

    async def _refresh(self, *, scan: bool) -> None:
        async with self._lock:
            while not self._closed:
                generation = self._generation
                if isinstance(self._source, UsageReader):
                    operation = (
                        self._source.reconcile
                        if scan
                        else self._source.refresh_invalidated
                    )
                    await asyncio.to_thread(operation)
                snapshot = self._snapshot()
                calendar = self._calendar()
                changed = await asyncio.to_thread(self._changed, snapshot, calendar)
                if not changed:
                    return
                projected = await asyncio.to_thread(self._project, snapshot, calendar)
                # Reader epochs guard disk commits; this generation also guards
                # an aggregation that was in flight when a settlement arrived.
                if generation != self._generation:
                    continue
                if self._closed:
                    return
                self._revision += 1
                self._published_source = snapshot
                self._published = replace(projected, revision=self._revision)
                for callback in tuple(self._callbacks):
                    try:
                        callback(self._published)
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "Usage service update callback failed"
                        )
                return

    async def reconcile(self) -> UsageServiceSnapshot:
        """Explicit external reconciliation; unchanged files remain stat-cached."""
        await self.wait_ready()
        await self._refresh(scan=True)
        assert self._published is not None
        return self._published

    async def aread(
        self, window: UsageWindow = "day", project_key: str | None = None
    ) -> UsageServiceSnapshot:
        """Read a consistent revision, doing large filtered aggregation off-loop."""
        await self.wait_ready()
        await self._refresh(scan=False)
        async with self._lock:
            assert self._published is not None
            assert self._published_source is not None
            published = self._published
            if window == "day" and project_key is None:
                return published
            result = await asyncio.to_thread(
                self._project,
                self._published_source,
                published.calendar,
                window,
                project_key,
            )
            return replace(result, revision=published.revision)

    def read(
        self, window: UsageWindow = "day", project_key: str | None = None
    ) -> UsageServiceSnapshot:
        if not self._started:
            return self._project(
                self._snapshot(), self._calendar(), window, project_key
            )
        if window != "day" or project_key is not None:
            raise ValueError("Use aread for background-service filtered/window reads")
        calendar = self._calendar()
        if not self._closed and self._calendar_changed(calendar) and not self._tasks:
            self._spawn()
        if self._published is not None:
            return self._published
        return self._project(UsageReadSnapshot(), calendar)

    async def _close(self) -> None:
        self._closed = True
        if self._debounce is not None:
            self._debounce.cancel()
            self._debounce = None
        for unsubscribe in self._writers.values():
            unsubscribe()
        self._callbacks.clear()
        # Never cancel worker-thread scans: settle them before retiring the cache.
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks))
        await asyncio.gather(*(writer.drain() for writer in self._writers))
        self._writers.clear()

    async def aclose(self) -> None:
        await _wait_for_usage_settlement(asyncio.create_task(self._close()))
