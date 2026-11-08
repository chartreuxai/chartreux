"""Deterministic permission round-trip cell for the E2 row (v0.7.5 stage 2, X8d).

E2 gates ``permission_round_trip_p95_ms`` on the ``permission_fake_response``
cell: the wall from a runtime permission request being issued to the matching
disposition being applied, per round trip. The exchange is the real one — the
shell policy classifies a dangerous git command, the tool decision layer asks
through the interaction broker, the app-server turn controller opens a
user-input callback on the live projector, the answer crosses the real
transport pair, and the settled disposition resumes the waiting call — with
one substitution only: a deterministic fake responder takes the human's place,
so no human latency and no network delay enters the measurement.

Measurement window (the cell's committed definition):

* the **request** is issued when the callback entry is published in progress
  (``HistoryEntryAdded`` for the ``PublicCallbackEntry`` the server opened);
* the **disposition is applied** when that same entry is published completed
  (``HistoryEntryUpdated``), which the server emits before it resolves the
  core waiting decision.

Both endpoints are published states of one entry observed on the client side
of the same stream, so the two legs of publish-to-deliver are included
symmetrically and the sample is the prompt/response exchange itself.

The fake response is deterministic and must MATCH the request it answers: the
responder picks the once-grant label from the options the request actually
offers, keyed to the request's own question text. A non-matching answer would
fail closed to a denial in production, so the cell also proves application on
the transcript: every round trip's command actually ran, and the
declined-command feedback never appears. The responder definition (command,
answer label, scripted script shape, warmups, measured count) is committed as
:data:`FAKE_RESPONDER_FIXTURE_HASH`.

Nothing here changes production behaviour: the cell drives the app-server
session with a scripted backend and answers the callbacks it raises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

from chartreux.app_server.events import (
    CallbackRequested,
    HistoryEntryAdded,
    HistoryEntryUpdated,
)
from chartreux.app_server.models import (
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.session import AppServerSession
from chartreux.core.agent_loop._dangerous_command_grants import (
    DANGEROUS_COMMAND_DECLINED_FEEDBACK,
    DANGEROUS_COMMAND_ONCE_LABEL,
)
from chartreux.core.config.harness_files import (
    get_harness_files_manager,
    init_harness_files_manager,
)
from chartreux.core.events import UserInputOrigin
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.perf._runtime_report import (
    CELL_PERMISSION,
    COUNT_PERMISSION_ROUND_TRIPS,
    COVERAGE_PERMISSION_ROUND_TRIPS,
    DEFAULT_WARMUP_OPS,
    METRIC_VERSION,
    MODE_MEASURE,
    PERMISSION_ROUND_TRIP,
    UNIT_MS,
    build_run_record,
)
from tests.perf._runtime_scenario import EgressGuard
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend

PERMISSION_CAMPAIGN_ID = "E2"
PERMISSION_CELL = CELL_PERMISSION
PERMISSION_METRIC = PERMISSION_ROUND_TRIP
PERMISSION_UNIT = UNIT_MS
PERMISSION_COVERAGE_UNIT = COVERAGE_PERMISSION_ROUND_TRIPS
PERMISSION_COUNT_KEY = COUNT_PERMISSION_ROUND_TRIPS

#: Appendix B E2 floor: >= 100 measured round trips after >= 10 warmups.
PERMISSION_WARMUP_ROUND_TRIPS = DEFAULT_WARMUP_OPS
PERMISSION_MEASURED_ROUND_TRIPS = 100

#: The in-suite shape (plan §5): shortened, and never campaign data.
PERMISSION_CONTRACT_WARMUPS = 1
PERMISSION_CONTRACT_MEASURED = 2

#: The deterministic responder: one dangerous git command per round trip. The
#: command must require runtime approval and must be harmless when granted —
#: this one fails immediately in a repository with no stash entries.
PERMISSION_COMMAND = "git stash drop"
PERMISSION_TOOL_NAME = "bash"
PERMISSION_ANSWER_LABEL = DANGEROUS_COMMAND_ONCE_LABEL
PERMISSION_ORIGIN = UserInputOrigin.RUNTIME_APPROVAL
PERMISSION_PROMPT = "Exercise a permission round trip."
PERMISSION_TERMINAL_CONTENT = "permission round trips complete"
PERMISSION_TOOL_CALL_ID = "permission-round-trip-{index:04d}"

#: The cell's own catalog entry: no API-key environment variable, so the
#: credential-stripping egress guard never blocks the scripted backend.
PERMISSION_PROVIDER = "perf-permission"
PERMISSION_MODEL = "perf-permission-model"

PERMISSION_SCHEMA = "chartreux.runtime_permission_fixture.v1"
#: Committed digest of :func:`fake_responder_fixture`; drift is fixture drift.
FAKE_RESPONDER_FIXTURE_HASH = (
    "5d544d92cb78509653cd1daaf7848b3b6490ce55c4941949cecd246ebd6b389f"
)

#: The measurement window this cell's samples are taken over.
PERMISSION_WINDOW = "callback_entry_added->callback_entry_completed"

_APPLIED_STATUS = PublicEntryGenerationStatus.COMPLETED
_P50_FRACTION = 0.50
_P95_FRACTION = 0.95
_SAMPLE_PRECISION = 4


class PermissionCellError(RuntimeError):
    """A permission round trip did not behave as the cell's contract requires."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"permission cell rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


def fake_responder_fixture() -> dict[str, Any]:
    """Return the committed fake-responder definition (the fixture itself).

    The scripted script is part of the definition: one stream per round trip
    carrying one tool call, and one terminal stream, so a responder that
    answered the wrong request would change the digest as well.
    """
    return {
        "schema": PERMISSION_SCHEMA,
        "campaign_id": PERMISSION_CAMPAIGN_ID,
        "cell": PERMISSION_CELL,
        "metric": PERMISSION_METRIC,
        "unit": PERMISSION_UNIT,
        "command": PERMISSION_COMMAND,
        "tool": PERMISSION_TOOL_NAME,
        "tool_call_id_template": PERMISSION_TOOL_CALL_ID,
        "answer_label": PERMISSION_ANSWER_LABEL,
        "origin": str(PERMISSION_ORIGIN),
        "prompt": PERMISSION_PROMPT,
        "terminal_content": PERMISSION_TERMINAL_CONTENT,
        "response_shape": {
            "streams_per_round_trip": 1,
            "chunks_per_stream": 1,
            "tool_calls_per_stream": 1,
            "terminal_streams": 1,
            "streaming": False,
        },
        "path_policy": {
            "start_point": "git_init_repository",
            "scratchpad_roots": "none",
        },
        "warmup_round_trips": PERMISSION_WARMUP_ROUND_TRIPS,
        "measured_round_trips": PERMISSION_MEASURED_ROUND_TRIPS,
        "contract_warmups": PERMISSION_CONTRACT_WARMUPS,
        "contract_measured": PERMISSION_CONTRACT_MEASURED,
        "measurement_window": PERMISSION_WINDOW,
    }


def fake_responder_fixture_digest(fixture: Mapping[str, Any] | None = None) -> str:
    """Return the sha256 digest of the fake-responder fixture definition."""
    document = fake_responder_fixture() if fixture is None else dict(fixture)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PermissionRoundTrip:
    """One measured permission round trip: request issued, disposition applied."""

    index: int
    callback_id: str
    question: str
    offered_labels: tuple[str, ...]
    answer_label: str
    elapsed_ms: float
    applied_status: str

    @property
    def applied(self) -> bool:
        """True when the disposition was published as applied for this trip."""
        return self.applied_status == str(_APPLIED_STATUS)

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "callback_id": self.callback_id,
            "question": self.question,
            "offered_labels": list(self.offered_labels),
            "answer_label": self.answer_label,
            "elapsed_ms": round(self.elapsed_ms, _SAMPLE_PRECISION),
            "applied_status": self.applied_status,
            "applied": self.applied,
        }


@dataclass(frozen=True, slots=True)
class PermissionCellResult:
    """One cell run: measured round-trip samples plus their coverage counts."""

    fixture_hash: str
    warmup_ops: int
    measured_ops: int
    round_trips: tuple[PermissionRoundTrip, ...]
    executed_commands: int
    declined_responses: int
    setup_wall_ms: float
    turn_wall_ms: float

    @property
    def samples_ms(self) -> tuple[float, ...]:
        """Return the measured round-trip samples, warmups discarded."""
        return tuple(trip.elapsed_ms for trip in self.round_trips[self.warmup_ops :])

    @property
    def total_ops(self) -> int:
        return self.warmup_ops + self.measured_ops

    def applied_round_trips(self) -> int:
        return sum(1 for trip in self.round_trips if trip.applied)

    def unit_counts(self) -> dict[str, int]:
        """Return the row's chunks-coverage unit counts (permission trips)."""
        return {PERMISSION_COUNT_KEY: self.measured_ops}

    def chunks_coverage(self) -> dict[str, Any]:
        """Return the run's chunks-coverage block (plan §3, WP6 field set)."""
        missing: list[str] = []
        if self.measured_ops <= 0:
            missing.append(PERMISSION_COUNT_KEY)
        return {
            "unit": PERMISSION_COVERAGE_UNIT,
            "measured_ops": self.measured_ops,
            "warmup_ops": self.warmup_ops,
            "unit_counts": self.unit_counts(),
            "complete": not missing,
            "missing": missing,
        }

    def summary(self) -> dict[str, Any]:
        """Summarize the gated series: count, p50, p95 (nearest rank) and max."""
        samples = self.samples_ms
        if not samples:
            return {
                "count": 0,
                "p50": None,
                "p95": None,
                "max": None,
                "null_reason": "no_gated_samples",
            }
        return {
            "count": len(samples),
            "p50": nearest_rank(samples, fraction=_P50_FRACTION),
            "p95": nearest_rank(samples, fraction=_P95_FRACTION),
            "max": max(samples),
        }

    def run_record(self) -> dict[str, Any]:
        """Return the cell's own run record (cell, coverage and per-trip data)."""
        return {
            "schema": PERMISSION_SCHEMA,
            "metric_version": METRIC_VERSION,
            "campaign_id": PERMISSION_CAMPAIGN_ID,
            "cell": PERMISSION_CELL,
            "fixture_hash": self.fixture_hash,
            "measurement_window": PERMISSION_WINDOW,
            "chunks_coverage": self.chunks_coverage(),
            "summary": self.summary(),
            "executed_commands": self.executed_commands,
            "declined_responses": self.declined_responses,
            "setup_wall_ms": round(self.setup_wall_ms, _SAMPLE_PRECISION),
            "turn_wall_ms": round(self.turn_wall_ms, _SAMPLE_PRECISION),
            "round_trips": [trip.record() for trip in self.round_trips],
        }


def nearest_rank(samples: Sequence[float], *, fraction: float) -> float:
    """Return the determinants' nearest-rank percentile of one sample series."""
    if not samples:
        raise ValueError("nearest_rank requires at least one sample")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def prepare_permission_project(root: Path | str) -> Path:
    """Create the disposable git repository one cell run measures in.

    The repository starts clean and remote-less: the command's approval
    classification needs a git repository, and a granted command must have no
    destination it could reach.
    """
    project = Path(root)
    project.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "init", "-q", str(project)], check=True, timeout=60)
    except FileNotFoundError as error:
        raise PermissionCellError(
            "git_unavailable", "the permission cell needs the git executable"
        ) from error
    except subprocess.CalledProcessError as error:
        raise PermissionCellError("git_init_failed", str(error)) from error
    return project


def permission_cell_guard(root: Path | str) -> EgressGuard:
    """Install and assert the fail-closed egress policy for one cell run."""
    guard = EgressGuard(Path(root))
    guard.install()
    guard.assert_closed()
    return guard


def _ensure_harness_files() -> None:
    """Initialise the harness-file manager a bare-script runner lacks.

    pytest supplies this through an autouse fixture; a per-cell subprocess does
    not, and the agent loop requires it at construction. Initialisation is
    idempotent and reads only the redirected config home the egress guard owns.
    """
    try:
        get_harness_files_manager()
    except RuntimeError:
        init_harness_files_manager("user", "project")


def permission_catalog_snapshot() -> CatalogSnapshot:
    """Return the cell's key-less catalog: the scripted backend serves the model.

    The measurement process carries no provider credential (the egress guard
    strips every one of them), so the cell's provider declares no API-key
    environment variable at all; nothing here can reach a provider.
    """
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                PERMISSION_PROVIDER: {
                    "api_base": "https://permission.invalid",
                    "backend": "generic",
                }
            },
            "models": {
                PERMISSION_MODEL: {
                    "deployments": [
                        {
                            "provider": PERMISSION_PROVIDER,
                            "name": f"{PERMISSION_MODEL}-wire",
                        }
                    ]
                }
            },
            "roles": {
                "main": {
                    "description": "Permission round-trip cell main preset",
                    "model": PERMISSION_MODEL,
                    "thinking": "high",
                }
            },
        }),
        "perf-permission",
    )


def _tool_call(index: int) -> ToolCall:
    return ToolCall(
        id=PERMISSION_TOOL_CALL_ID.format(index=index),
        index=0,
        function=FunctionCall(
            name=PERMISSION_TOOL_NAME,
            arguments=json.dumps({"command": PERMISSION_COMMAND}),
        ),
    )


@dataclass(slots=True)
class _PendingRoundTrip:
    """One callback entry the server opened, awaiting its disposition."""

    callback_id: str
    issued_at: float
    question: str
    offered_labels: tuple[str, ...]


class PermissionCell:
    """Drive one E2 run: scripted requests, deterministic fake responses."""

    def __init__(
        self,
        project: Path | str,
        *,
        warmups: int = PERMISSION_WARMUP_ROUND_TRIPS,
        measured: int = PERMISSION_MEASURED_ROUND_TRIPS,
    ) -> None:
        if warmups < 0 or measured <= 0:
            raise PermissionCellError(
                "round_trip_budget_invalid", f"warmups={warmups}, measured={measured}"
            )
        self.project = Path(project)
        self.warmups = warmups
        self.measured = measured

    @property
    def total_round_trips(self) -> int:
        return self.warmups + self.measured

    def scripted_responses(self) -> list[list[Any]]:
        """Return the deterministic backend script: one approval call per trip."""
        responses = [
            [mock_llm_chunk(content="", tool_calls=[_tool_call(index)])]
            for index in range(self.total_round_trips)
        ]
        responses.append([mock_llm_chunk(content=PERMISSION_TERMINAL_CONTENT)])
        return responses

    def responder_output(self, question: str) -> UserInputCallbackOutput:
        """Return the matching deterministic response for one request."""
        return UserInputCallbackOutput(
            result=UserQuestionResult(
                answers=[UserAnswer(question=question, answer=PERMISSION_ANSWER_LABEL)]
            )
        )

    async def run(self) -> PermissionCellResult:
        """Execute the cell and return its measured round trips."""
        _ensure_harness_files()
        setup_started = perf_counter()
        backend = FakeBackend(self.scripted_responses())
        agent_loop = build_test_agent_loop(
            config=build_test_vibe_config().attach_catalog_snapshot(
                permission_catalog_snapshot()
            ),
            backend=backend,
            enable_streaming=False,
            cwd=self.project,
            user_input_capability=True,
        )
        session = await create_test_app_server_session(agent_loop)
        setup_wall_ms = (perf_counter() - setup_started) * 1000
        pending: dict[str, _PendingRoundTrip] = {}
        round_trips: list[PermissionRoundTrip] = []
        turn_started = perf_counter()
        try:
            async for event in session.act(PERMISSION_PROMPT):
                now = perf_counter()
                if isinstance(event, HistoryEntryAdded):
                    entry = event.entry
                    if isinstance(entry, PublicCallbackEntry):
                        pending.setdefault(entry.callback_id, self._opening(entry, now))
                elif isinstance(event, HistoryEntryUpdated):
                    trip = self._close(
                        entry=event.entry,
                        now=now,
                        pending=pending,
                        index=len(round_trips),
                    )
                    if trip is not None:
                        round_trips.append(trip)
                elif isinstance(event, CallbackRequested):
                    await self._answer(
                        session, event.callback, pending=pending, now=now
                    )
        finally:
            turn_wall_ms = (perf_counter() - turn_started) * 1000
            await session.close()
            executed = _executed_command_responses(agent_loop.messages)
            declined = _declined_responses(agent_loop.messages)
            await agent_loop.aclose()
        result = PermissionCellResult(
            fixture_hash=FAKE_RESPONDER_FIXTURE_HASH,
            warmup_ops=self.warmups,
            measured_ops=self.measured,
            round_trips=tuple(round_trips),
            executed_commands=executed,
            declined_responses=declined,
            setup_wall_ms=setup_wall_ms,
            turn_wall_ms=turn_wall_ms,
        )
        self.assert_contract(result)
        return result

    def _opening(
        self, entry: PublicCallbackEntry, issued_at: float
    ) -> _PendingRoundTrip:
        """Record the request as issued, rejecting a non-approval prompt."""
        detail = entry.detail
        origin = getattr(detail, "origin", None)
        if origin != str(PERMISSION_ORIGIN):
            raise PermissionCellError(
                "request_origin_mismatch", f"{origin!r} is not a runtime approval"
            )
        request = detail.request
        if not request.questions:
            raise PermissionCellError("request_without_question", entry.callback_id)
        question = request.questions[0]
        offered = tuple(choice.label for choice in question.options)
        if PERMISSION_ANSWER_LABEL not in offered:
            raise PermissionCellError(
                "answer_label_not_offered",
                f"{PERMISSION_ANSWER_LABEL!r} not in {offered}",
            )
        if PERMISSION_COMMAND not in question.question:
            raise PermissionCellError(
                "request_names_another_command", question.question[:120]
            )
        return _PendingRoundTrip(
            callback_id=entry.callback_id,
            issued_at=issued_at,
            question=question.question,
            offered_labels=offered,
        )

    def _close(
        self,
        *,
        entry: object,
        now: float,
        pending: dict[str, _PendingRoundTrip],
        index: int,
    ) -> PermissionRoundTrip | None:
        """Record a disposition applied at the published completed update."""
        if not isinstance(entry, PublicCallbackEntry):
            return None
        if entry.generation_status != _APPLIED_STATUS:
            return None
        opening = pending.pop(entry.callback_id, None)
        if opening is None:
            raise PermissionCellError("disposition_without_request", entry.callback_id)
        return PermissionRoundTrip(
            index=index,
            callback_id=opening.callback_id,
            question=opening.question,
            offered_labels=opening.offered_labels,
            answer_label=PERMISSION_ANSWER_LABEL,
            elapsed_ms=(now - opening.issued_at) * 1000,
            applied_status=str(entry.generation_status),
        )

    async def _answer(
        self,
        session: AppServerSession,
        callback: PublicCallbackEntry,
        *,
        pending: dict[str, _PendingRoundTrip],
        now: float,
    ) -> None:
        """Apply the matching deterministic response to the emitted request."""
        opening = pending.get(callback.callback_id)
        if opening is None:
            opening = self._opening(callback, now)
            pending[callback.callback_id] = opening
        await session.respond_to_callback(
            opening.callback_id, self.responder_output(opening.question)
        )

    def assert_contract(self, result: PermissionCellResult) -> None:
        """Fail closed unless every round trip behaved as the cell requires."""
        if len(result.round_trips) != self.total_round_trips:
            raise PermissionCellError(
                "round_trip_count_mismatch",
                f"{len(result.round_trips)} of {self.total_round_trips} settled",
            )
        if result.applied_round_trips() != self.total_round_trips:
            raise PermissionCellError(
                "disposition_not_applied",
                f"{result.applied_round_trips()} of {self.total_round_trips} applied",
            )
        if result.declined_responses:
            raise PermissionCellError(
                "fake_response_did_not_match", f"{result.declined_responses} declined"
            )
        if result.executed_commands != self.total_round_trips:
            raise PermissionCellError(
                "disposition_not_reached_the_tool",
                f"{result.executed_commands} of {self.total_round_trips} executed",
            )


def _executed_command_responses(messages: Sequence[Any]) -> int:
    """Count tool responses proving the granted command actually ran."""
    marker = f"Command failed: '{PERMISSION_COMMAND}'"
    return sum(
        1
        for message in messages
        if getattr(message, "role", None) == "tool"
        and isinstance(getattr(message, "content", None), str)
        and marker in message.content
    )


def _declined_responses(messages: Sequence[Any]) -> int:
    """Count tool responses carrying the production declined-command feedback."""
    return sum(
        1
        for message in messages
        if getattr(message, "role", None) == "tool"
        and isinstance(getattr(message, "content", None), str)
        and DANGEROUS_COMMAND_DECLINED_FEEDBACK in message.content
    )


async def run_permission_cell(
    project: Path | str,
    *,
    warmups: int = PERMISSION_WARMUP_ROUND_TRIPS,
    measured: int = PERMISSION_MEASURED_ROUND_TRIPS,
) -> PermissionCellResult:
    """Run one permission cell over a prepared project directory."""
    cell = PermissionCell(project, warmups=warmups, measured=measured)
    return await cell.run()


def build_permission_run_record(
    result: PermissionCellResult,
    *,
    run_id: str,
    run_start_utc: str,
    mode: str = MODE_MEASURE,
    closure_status: str | None = None,
    closure_note: str | None = None,
    machine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate the WP6 E2 run record for one measured cell run."""
    return build_run_record(
        PERMISSION_CAMPAIGN_ID,
        run_id=run_id,
        run_start_utc=run_start_utc,
        measured_ops=result.measured_ops,
        warmup_ops=result.warmup_ops,
        samples=list(result.samples_ms),
        unit_counts=result.unit_counts(),
        mode=mode,
        closure_status=closure_status,
        closure_note=(
            "run record is not a campaign artifact until the campaign closes"
            if closure_note is None
            else closure_note
        ),
        fixture_hash=result.fixture_hash,
        machine=machine,
    )


__all__ = [
    "FAKE_RESPONDER_FIXTURE_HASH",
    "PERMISSION_ANSWER_LABEL",
    "PERMISSION_CAMPAIGN_ID",
    "PERMISSION_CELL",
    "PERMISSION_COMMAND",
    "PERMISSION_CONTRACT_MEASURED",
    "PERMISSION_CONTRACT_WARMUPS",
    "PERMISSION_COUNT_KEY",
    "PERMISSION_COVERAGE_UNIT",
    "PERMISSION_MEASURED_ROUND_TRIPS",
    "PERMISSION_METRIC",
    "PERMISSION_ORIGIN",
    "PERMISSION_UNIT",
    "PERMISSION_WARMUP_ROUND_TRIPS",
    "PERMISSION_WINDOW",
    "PermissionCell",
    "PermissionCellError",
    "PermissionCellResult",
    "PermissionRoundTrip",
    "build_permission_run_record",
    "fake_responder_fixture",
    "fake_responder_fixture_digest",
    "nearest_rank",
    "permission_cell_guard",
    "prepare_permission_project",
    "run_permission_cell",
]
