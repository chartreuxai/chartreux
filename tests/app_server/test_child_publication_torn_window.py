"""The child publication transaction's torn persist-to-link window.

The published transaction persists the empty child session BEFORE it records
the parent link. A crash — or a lost delete during rollback — between those
two writes leaves a scan-visible child transcript with no parent link. The
attempt id written at the FIRST durable write is what attributes that window
to its attempt, and the tombstone is what retracts it so the record can never
resurrect as an adoptable child.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError
import pytest

from chartreux.app_server import _subagent_launch
from chartreux.app_server._subagent_recovery import ChildTranscriptReconciler
from chartreux.app_server._subagent_transcript import (
    CHILD_TOMBSTONE_FILENAME,
    ChildResultRecord,
    ChildStartMarker,
    ChildTombstoneRecord,
    ChildTranscriptOutcome,
    ChildTranscriptStore,
    child_result_digest,
    persisted_child_attempt_id,
    read_child_transcript,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.session_types import ChildSessionLink, SessionMetadata
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.stubs.fake_backend import FakeBackend

_ROOT_SESSION_ID = "root-torn-session"
_ATTEMPT_ID = "attempt-0123456789abcdef"


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "sessions" / _ROOT_SESSION_ID
    root.mkdir(parents=True)
    return root


def _store(
    root: Path, name: str = "coder-py_20260101_000000_abcdef12"
) -> ChildTranscriptStore:
    child_dir = root / "agents" / name
    child_dir.mkdir(parents=True, exist_ok=True)
    return ChildTranscriptStore(child_dir, root_session_dir=root)


def _start(attempt_id: str | None = None) -> ChildStartMarker:
    return ChildStartMarker(
        agent_id="coder:py",
        session_id="child-session-1",
        run_id="turn-1",
        started_at="2026-01-01T00:00:00+00:00",
        attempt_id=attempt_id,
    )


def _result() -> ChildResultRecord:
    return ChildResultRecord(
        agent_id="coder:py",
        session_id="child-session-1",
        run_id="turn-1",
        result_id="coder:py:turn-1",
        digest=child_result_digest(
            agent_id="coder:py",
            session_id="child-session-1",
            run_id="turn-1",
            outcome="completed",
            error=None,
        ),
        outcome="completed",
        completed_at="2026-01-01T00:00:05+00:00",
    )


def _tombstone(
    *, attempt_id: str | None = _ATTEMPT_ID, session_id: str = "child-session-1"
) -> ChildTombstoneRecord:
    return ChildTombstoneRecord(
        session_id=session_id,
        reason="publication-rejected",
        created_at="2026-01-01T00:00:06+00:00",
        attempt_id=attempt_id,
        agent_id="coder:py",
    )


# ---------------------------------------------------------------------------
# The torn window the tombstone exists for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_without_link_tombstone_gates_the_child_transcript_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist lands, the link never does, and the delete is lost."""
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent = build_test_agent_loop(config=_config(logging), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    root.turns._projector = MagicMock()
    captured: dict[str, Any] = {}
    original_create = registry._runtime_factory.create_child

    async def capture_created_child(
        loop: AgentLoop,
        agent_name: str,
        *,
        session_id: str | None = None,
        session_dir: Path | None = None,
    ) -> AgentLoop:
        child = await original_create(
            loop, agent_name, session_id=session_id, session_dir=session_dir
        )
        captured["child_dir"] = child.session_logger.session_dir
        captured["session_id"] = child.session_id
        return child

    monkeypatch.setattr(
        registry._runtime_factory, "create_child", capture_created_child
    )

    async def torn_link(
        loop: AgentLoop,
        child: AgentLoop,
        tool_call_id: str,
        agent_name: str,
        *,
        attempt_id: str | None = None,
    ) -> None:
        # The torn window, observed exactly where it happens: the empty child
        # session is already durable, the parent link is not.
        child_dir = child.session_logger.session_dir
        assert child_dir is not None
        captured["attempt_id"] = attempt_id
        captured["persisted_attempt_id"] = persisted_child_attempt_id(child_dir)
        captured["meta_text"] = (child_dir / "meta.json").read_text()
        raise RuntimeError("torn persist-to-link window")

    monkeypatch.setattr(AgentLoop, "record_child_session", torn_link)

    def lost_delete(*_args: object, **_kwargs: object) -> None:
        raise OSError("delete lost")

    monkeypatch.setattr(_subagent_launch, "delete_saved_session", lost_delete)
    try:
        with pytest.raises(RuntimeError, match="torn persist-to-link window"):
            await _background_result(
                registry,
                TaskArgs(task="torn", identity="coder", background=True),
                InvokeContext(tool_call_id="torn", session_id=parent.session_id),
            )
        assert registry._children == {}
        assert registry._child_links == {}

        attempt_id = captured["attempt_id"]
        assert isinstance(attempt_id, str)
        assert attempt_id.startswith("attempt-")
        # The FIRST durable write carried it; the parent link never existed.
        assert captured["persisted_attempt_id"] == attempt_id
        assert json.loads(captured["meta_text"])["attempt_id"] == attempt_id

        child_dir = captured["child_dir"]
        assert isinstance(child_dir, Path)
        root_session_dir = parent.session_logger.session_dir
        assert root_session_dir is not None
        store = ChildTranscriptStore(child_dir, root_session_dir=root_session_dir)
        tombstone = store.read_tombstone()
        assert tombstone is not None
        assert tombstone.attempt_id == attempt_id
        assert tombstone.session_id == captured["session_id"]
        assert tombstone.reason == _subagent_launch.PUBLICATION_ROLLBACK_REASON

        reconciler = ChildTranscriptReconciler(root_session_dir)
        reconciliation = reconciler.reconcile(child_dir)
        assert reconciliation.outcome is ChildTranscriptOutcome.TOMBSTONED
        assert reconciliation.tombstoned is True
        assert reconciliation.attempt_id == attempt_id
        assert reconciliation.result_id is None
        assert reconciliation.has_unconsumed_result is False
        assert reconciliation.denial is not None
        assert "retracted" in reconciliation.denial
        assert "replayed" in reconciliation.denial
        # The retraction gates the scan: nothing is handed to the root.
        assert reconciler.scan() == []

        # Idempotence: retracting the same attempt again is a no-op.
        before = (child_dir / CHILD_TOMBSTONE_FILENAME).read_bytes()
        store.record_tombstone(tombstone)
        assert (child_dir / CHILD_TOMBSTONE_FILENAME).read_bytes() == before
        assert reconciler.scan() == []
    finally:
        await parent.aclose()


# ---------------------------------------------------------------------------
# A retracted transcript never hands a result over
# ---------------------------------------------------------------------------


def test_a_tombstone_retracts_a_completed_looking_transcript(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start(_ATTEMPT_ID))
    store.record_result(_result())
    assert store.adjudicate().outcome is ChildTranscriptOutcome.COMPLETED

    store.record_tombstone(_tombstone())
    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.TOMBSTONED
    assert adjudication.attempt_id == _ATTEMPT_ID
    assert adjudication.result is None

    reconciliation = ChildTranscriptReconciler(root).reconcile(store.child_dir)
    assert reconciliation.outcome is ChildTranscriptOutcome.TOMBSTONED
    assert reconciliation.result_id is None
    assert reconciliation.has_unconsumed_result is False
    assert ChildTranscriptReconciler(root).scan() == []


def test_a_legacy_tombstone_retracts_by_session_id(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start())
    store.record_tombstone(_tombstone(attempt_id=None))

    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.TOMBSTONED
    assert adjudication.attempt_id is None
    assert ChildTranscriptReconciler(root).scan() == []


def test_a_retraction_by_session_id_upgrades_once_the_attempt_is_known(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_tombstone(_tombstone(attempt_id=None))
    store.record_tombstone(_tombstone())
    tombstone = store.read_tombstone()
    assert tombstone is not None
    assert tombstone.attempt_id == _ATTEMPT_ID


# ---------------------------------------------------------------------------
# Records: attempt-id tolerance and legacy loads
# ---------------------------------------------------------------------------


def test_the_runtime_start_marker_carries_the_attempt_id_only_when_known(
    tmp_path: Path,
) -> None:
    marker = _start()
    assert "attempt_id" not in marker.to_payload()
    assert ChildStartMarker.from_payload(marker.to_payload()).attempt_id is None

    tagged = _start(_ATTEMPT_ID)
    assert tagged.to_payload()["attempt_id"] == _ATTEMPT_ID
    assert ChildStartMarker.from_payload(tagged.to_payload()).attempt_id == _ATTEMPT_ID
    # Tolerant reader: an unknown value never fails the whole record.
    tolerant = {**marker.to_payload(), "attempt_id": 17}
    assert ChildStartMarker.from_payload(tolerant).attempt_id is None

    # The run-time marker keeps its semantics and bridges the attempt.
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(tagged)
    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
    assert adjudication.attempt_id == _ATTEMPT_ID


def test_the_persisted_first_write_bridges_a_markerless_record(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _store(root)
    metadata = SessionMetadata(
        session_id="child-session-1",
        start_time="2026-01-01T00:00:00+00:00",
        end_time=None,
        git_commit=None,
        git_branch=None,
        environment={},
        username="tester",
        attempt_id=_ATTEMPT_ID,
    )
    (store.child_dir / "meta.json").write_text(
        metadata.model_dump_json(), encoding="utf-8"
    )

    assert persisted_child_attempt_id(store.child_dir) == _ATTEMPT_ID
    adjudication = read_child_transcript(store.child_dir, root_session_dir=root)
    assert adjudication.outcome is ChildTranscriptOutcome.NEVER_EXECUTED
    assert adjudication.attempt_id == _ATTEMPT_ID

    reconciliation = ChildTranscriptReconciler(root).reconcile(store.child_dir)
    assert reconciliation.attempt_id == _ATTEMPT_ID
    params = reconciliation.coordinator_message()["params"]
    assert isinstance(params, dict)
    assert params["attempt_id"] == _ATTEMPT_ID


def test_child_session_link_attempt_id_is_optional_and_strict() -> None:
    legacy = {
        "session_id": "child-session-1",
        "tool_call_id": "call-1",
        "agent": "coder:py",
        "relative_path": "agents/coder-py_20260101_000000_abcdef12",
    }
    link = ChildSessionLink.model_validate(legacy)
    assert link.attempt_id is None
    assert ChildSessionLink.model_validate(link.model_dump()) == link

    tagged = ChildSessionLink.model_validate({**legacy, "attempt_id": _ATTEMPT_ID})
    assert tagged.attempt_id == _ATTEMPT_ID
    assert ChildSessionLink.model_validate(tagged.model_dump()) == tagged
    with pytest.raises(ValidationError):
        ChildSessionLink.model_validate({**legacy, "unexpected": "field"})


def test_session_metadata_attempt_id_loads_legacy_records() -> None:
    legacy = {
        "session_id": "session-1",
        "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": None,
        "git_commit": None,
        "git_branch": None,
        "environment": {},
        "username": "tester",
    }
    metadata = SessionMetadata.model_validate(legacy)
    assert metadata.attempt_id is None

    tagged = metadata.model_copy(update={"attempt_id": _ATTEMPT_ID})
    reloaded = SessionMetadata.model_validate(tagged.model_dump(mode="json"))
    assert reloaded.attempt_id == _ATTEMPT_ID


def test_persist_empty_session_takes_the_attempt_id_as_an_optional_keyword() -> None:
    parameters = inspect.signature(AgentLoop.persist_empty_session).parameters
    assert list(parameters) == ["self", "attempt_id"]
    attempt_parameter = parameters["attempt_id"]
    assert attempt_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert attempt_parameter.default is None
