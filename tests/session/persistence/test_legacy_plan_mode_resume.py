"""Resume-notice detection for legacy plan-mode sessions (plan-mode removal).

Old sessions whose raw ``meta.json`` carries a ``plan_mode`` key resume as
ordinary sessions: the typed ``SessionMetadata`` drops the retired key
(``extra="ignore"``), and detection reads the RAW dict at the loader choke
point only. These tests pin both halves of the resume notice's detection:

- ``SessionLoader.load_session_bounded`` sets ``plan_mode_present`` from the
  raw metadata dict;
- ``legacy_plan_mode_path`` extracts the plan document path from the same
  raw dict, never from a typed record.
"""

from __future__ import annotations

import json
from pathlib import Path

from chartreux.core.session.session_loader import (
    BoundedSessionLoadState,
    SessionLoader,
    legacy_plan_mode_path,
)


def _write_session(session_dir: Path, metadata: dict) -> Path:
    session_dir.mkdir(parents=True)
    (session_dir / "messages.jsonl").write_text(
        json.dumps({"role": "user", "content": "hello"}) + "\n", encoding="utf-8"
    )
    (session_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return session_dir


def test_bounded_load_flags_a_raw_plan_mode_key(tmp_path: Path) -> None:
    session_dir = _write_session(
        tmp_path / "sessions" / "s1",
        {
            "session_id": "s1",
            "total_messages": 1,
            "plan_mode": {"plan_path": "/tmp/workspaces/tasks/plan-x/plan.md"},
        },
    )

    load = SessionLoader.load_session_bounded(session_dir)

    assert load.state is BoundedSessionLoadState.AVAILABLE
    assert load.plan_mode_present is True
    assert load.metadata is not None
    assert legacy_plan_mode_path(load.metadata) == (
        "/tmp/workspaces/tasks/plan-x/plan.md"
    )


def test_bounded_load_without_the_key_stays_silent(tmp_path: Path) -> None:
    session_dir = _write_session(
        tmp_path / "sessions" / "s2", {"session_id": "s2", "total_messages": 1}
    )

    load = SessionLoader.load_session_bounded(session_dir)

    assert load.state is BoundedSessionLoadState.AVAILABLE
    assert load.plan_mode_present is False
    assert load.metadata is not None
    assert legacy_plan_mode_path(load.metadata) is None


def test_load_session_keeps_the_raw_key_available_for_detection(tmp_path: Path) -> None:
    session_dir = _write_session(
        tmp_path / "sessions" / "s3",
        {
            "session_id": "s3",
            "total_messages": 1,
            "plan_mode": {"plan_path": "/tmp/plan.md", "state": "drafting"},
        },
    )

    _messages, raw_metadata = SessionLoader.load_session(session_dir)

    # The raw dict still carries the retired key, so the choke point can see
    # it; the typed record never does.
    assert "plan_mode" in raw_metadata
    assert legacy_plan_mode_path(raw_metadata) == "/tmp/plan.md"


def test_legacy_plan_mode_path_ignores_malformed_values() -> None:
    assert legacy_plan_mode_path({}) is None
    assert legacy_plan_mode_path({"plan_mode": None}) is None
    assert legacy_plan_mode_path({"plan_mode": "drafting"}) is None
    assert legacy_plan_mode_path({"plan_mode": {"plan_path": ""}}) is None
    assert legacy_plan_mode_path({"plan_mode": {"plan_path": 42}}) is None
