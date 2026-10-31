"""Unit tests for the OSC 7501 encoder, message pipeline, and emitter.

Section ownership: WP2 owns the encoder, pipeline, redaction pins, security
battery, and emitter sections. App-level state mapping is appended below the
emitter section by WP4a/WP4b/WP5a/WP5b — keep this block self-contained.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from chartreux.app_server.events import HistoryEntryAdded, TurnCompleted, TurnStarted
from chartreux.app_server.models import (
    CancelledEffectState,
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    FailedEffectState,
    GenericEffectDetail,
    OpenCallbackState,
    PendingEffectState,
    PublicCallbackEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicError,
    PublicTurn,
    PublicTurnStatus,
    RunningEffectState,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    RootRecoveryAdmitResponse,
    RootRecoveryScheduleResponse,
)
from chartreux.app_server.session import AppServerSession, AppServerTurnError
from chartreux.cli.textual_ui import program_status
from chartreux.cli.textual_ui.app import (
    _OWNERSHIP_MISSING,
    ChartreuxApp,
    TurnOwnership,
    _BashSettlement,
    _OwnershipMissingSentinel,
)
from chartreux.cli.textual_ui.notifications import NotificationContext
from chartreux.cli.textual_ui.program_status import (
    ProgramStatusEmitter,
    encode_clear,
    encode_report,
    prepare_message,
)
from chartreux.cli.textual_ui.turn_supervisor import TurnOutcome
from chartreux.cli.textual_ui.widgets.compact import CompactMessage
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage
from chartreux.core.events import UserMessageEvent
from chartreux.core.tools import secret_redaction as sr
from chartreux.questions import QuestionChoice, UserQuestion, UserQuestionRequest
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend

# Synthetic values, built by concatenation so no credential-shaped literal is
# ever committed. Registration happens inside each security test via
# ``bind_policy``; the redactor matches by value, so an unregistered token
# would make these tests pass vacuously.
_FIXED_SECRET = "sk-" + "live-" + "0123456789"
_LONG_SECRET = "sk-osc7501-" + "0123456789abcdef" * 71 + "01234"

_UNSET = object()


@contextmanager
def _registered(*secrets: str) -> Iterator[None]:
    policy = sr.ScrubPolicy(
        redaction_credentials=tuple(
            (f"SYNTHETIC_OSC_{index}", secret) for index, secret in enumerate(secrets)
        )
    )
    with sr.bind_policy(policy):
        yield


def _fake_app(
    *,
    is_headless: bool = False,
    is_running: bool = True,
    shutdown_started: bool = False,
    driver: object = _UNSET,
) -> MagicMock:
    app = MagicMock()
    type(app).is_headless = PropertyMock(return_value=is_headless)
    app.is_running = is_running
    app._shutdown_started = shutdown_started
    app._driver = MagicMock() if driver is _UNSET else driver
    return app


def _decoded_msg(report: str) -> str:
    payload = report.split("msg=", 1)[1].removesuffix("\x07")
    return base64.b64decode(payload).decode("utf-8")


# --- encoder -------------------------------------------------------------


class TestEncodeReport:
    @pytest.mark.parametrize("state", ["idle", "working", "done", "blocked", "error"])
    def test_state_is_byte_exact(self, state: str) -> None:
        assert encode_report(state) == f"\x1b]7501;state={state}:app=chartreux\x07"

    @pytest.mark.parametrize("kind", ["question", "permission"])
    def test_blocked_kind_is_byte_exact(self, kind: str) -> None:
        assert (
            encode_report("blocked", kind=kind)
            == f"\x1b]7501;state=blocked:kind={kind}:app=chartreux\x07"
        )

    def test_message_pair_is_fixed_last(self) -> None:
        assert (
            encode_report("done", msg="All good")
            == "\x1b]7501;state=done:app=chartreux:msg=QWxsIGdvb2Q=\x07"
        )

    def test_message_round_trips_through_base64(self) -> None:
        text = "héllo wörld\nsecond line"
        report = encode_report("error", msg=text)
        assert _decoded_msg(report) == text
        assert "\n" not in report

    def test_bel_terminates_the_report(self) -> None:
        assert encode_report("blocked", kind="question").endswith("\x07")

    def test_clear_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unsupported program-status state"):
            encode_report("clear")

    @pytest.mark.parametrize("state", ["", "cleared", "IDLE", "blocked ", "done\n"])
    def test_unknown_state_raises(self, state: str) -> None:
        with pytest.raises(ValueError, match="unsupported program-status state"):
            encode_report(state)

    @pytest.mark.parametrize("kind", ["", "auth", "QUESTION", "permission "])
    def test_unknown_kind_raises(self, kind: str) -> None:
        with pytest.raises(ValueError, match="unsupported program-status kind"):
            encode_report("blocked", kind=kind)

    @pytest.mark.parametrize("state", ["idle", "working", "done", "error"])
    def test_kind_requires_blocked(self, state: str) -> None:
        with pytest.raises(ValueError, match="requires state blocked"):
            encode_report(state, kind="question")

    @pytest.mark.parametrize(
        "app", ["", "chartreux app", "chartreux:state=idle", "chartreux/app", "x" * 33]
    )
    def test_app_grammar_rejects_invalid(self, app: str) -> None:
        with pytest.raises(ValueError, match="invalid program-status app name"):
            encode_report("idle", app=app)

    @pytest.mark.parametrize("app", ["chartreux", "my.app-1_x+", "A" * 32])
    def test_app_grammar_accepts_valid(self, app: str) -> None:
        assert encode_report("idle", app=app) == f"\x1b]7501;state=idle:app={app}\x07"

    @pytest.mark.parametrize("state", ["idle", "working", "done", "error"])
    def test_reserved_keys_are_not_emitted(self, state: str) -> None:
        report = encode_report(state)
        assert "id=" not in report
        assert "progress=" not in report
        assert "title=" not in report

    def test_blocked_report_has_no_reserved_keys(self) -> None:
        report = encode_report("blocked", kind="permission", msg="Allow?")
        assert "id=" not in report
        assert "progress=" not in report
        assert "title=" not in report

    def test_over_limit_message_raises(self) -> None:
        with pytest.raises(ValueError, match="maximum size"):
            encode_report("done", msg="a" * 2049)

    def test_message_at_the_limit_is_accepted(self) -> None:
        report = encode_report("done", msg="a" * 2048)
        assert len(_decoded_msg(report)) == 2048
        payload = report.split("msg=", 1)[1].removesuffix("\x07")
        assert len(payload) == 2732

    def test_worst_case_blocked_permission_report_is_2788_bytes(self) -> None:
        report = encode_report("blocked", kind="permission", msg="a" * 2048)
        assert len(report) == len(report.encode("ascii")) == 2788
        assert len(report) < 4096


class TestEncodeClear:
    def test_clear_is_byte_exact(self) -> None:
        assert encode_clear() == "\x1b]7501;state=clear\x07"

    def test_clear_carries_no_kind_app_or_message(self) -> None:
        report = encode_clear()
        assert report.startswith("\x1b]7501;state=clear")
        assert "kind=" not in report
        assert "app=" not in report
        assert "msg=" not in report

    def test_clear_is_still_refused_as_a_state(self) -> None:
        with pytest.raises(ValueError, match="unsupported program-status state"):
            encode_report("clear")


# --- message pipeline ----------------------------------------------------


class TestPrepareMessage:
    def test_none_is_omitted(self) -> None:
        assert prepare_message(None) is None

    @pytest.mark.parametrize("text", ["", "   ", "\x00\x01", "\u200b", "\n\nsecond"])
    def test_empty_sanitized_text_is_omitted(self, text: str) -> None:
        assert prepare_message(text) is None

    def test_control_characters_become_single_spaces(self) -> None:
        assert prepare_message("hello\x00\x01world\t\x7f done") == "hello world done"

    def test_first_line_only(self) -> None:
        assert prepare_message("first line\nsecond line") == "first line"
        assert prepare_message("first\r\nsecond") == "first"

    def test_invisible_formatting_is_stripped(self) -> None:
        text = "a\u200bb\u200e\u200f\u202ac\u202cd\ufeff\u2060e\u061c"
        assert prepare_message(text) == "abcde"

    def test_joiners_are_kept(self) -> None:
        text = "ab\u200dcd\u200cef"
        assert prepare_message(text) == text

    def test_truncation_is_codepoint_safe(self) -> None:
        prepared = prepare_message("é" * 2000)
        assert prepared is not None
        assert prepared == "é" * 1024
        assert len(prepared.encode("utf-8")) <= 2048

    def test_truncation_backs_off_a_split_codepoint(self) -> None:
        prepared = prepare_message("a" * 2047 + "é" * 3)
        assert prepared is not None
        assert prepared == "a" * 2047
        assert len(prepared.encode("utf-8")) == 2047

    def test_post_final_pass_overflow_omits_instead_of_retruncating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"count": 0}

        def growing(text: str) -> str:
            calls["count"] += 1
            return text if calls["count"] < 3 else text + "x" * 3000

        monkeypatch.setattr(sr, "redact", growing)
        assert prepare_message("short") is None
        assert calls["count"] == 3

    @pytest.mark.parametrize("failing_pass", [1, 2, 3])
    def test_redaction_raise_in_any_pass_omits(
        self, monkeypatch: pytest.MonkeyPatch, failing_pass: int
    ) -> None:
        calls = {"count": 0}

        def failing(text: str) -> str:
            calls["count"] += 1
            if calls["count"] == failing_pass:
                raise RuntimeError("redaction failed")
            return text

        monkeypatch.setattr(sr, "redact", failing)
        assert prepare_message("hello") is None
        assert calls["count"] == failing_pass

    def test_scan_failure_omits(
        self, monkeypatch: pytest.MonkeyPatch, registered_secret: str
    ) -> None:
        def failing_scan(text: str) -> list[tuple[int, int, str]]:
            raise RuntimeError("scan failed")

        monkeypatch.setattr(sr, "_encoded_candidates", failing_scan)
        assert prepare_message("hello") is None

    def test_question_text_shape_collapses_to_first_line(self) -> None:
        raw = "Which database should I use?\n\nPostgres or SQLite?"
        assert prepare_message(raw) == "Which database should I use?"

    def test_permission_summary_keeps_the_command(self) -> None:
        # The emit site joins the first line and the next non-empty line with
        # one space; the pipeline must keep the command in the payload.
        summary = "Allow this command? " + "git push origin " + "main"
        prepared = prepare_message(summary)
        assert prepared is not None
        assert "git push origin main" in prepared

    def test_raw_multiline_permission_summary_would_drop_the_command(self) -> None:
        raw = "Allow this command?\n\n" + "git push origin " + "main"
        assert prepare_message(raw) == "Allow this command?"

    def test_raw_input_past_the_bound_is_omitted_before_pass_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"count": 0}

        def counting(text: str) -> str:
            calls["count"] += 1
            return text

        monkeypatch.setattr(sr, "redact", counting)
        oversized = "a" * (program_status._MAX_RAW_INPUT_CHARS + 1)
        assert prepare_message(oversized) is None
        # The oversize check precedes the raw pass, which is the cost bound.
        assert calls["count"] == 0

    def test_raw_input_at_the_bound_is_processed_normally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"count": 0}

        def counting(text: str) -> str:
            calls["count"] += 1
            return text

        monkeypatch.setattr(sr, "redact", counting)
        prepared = prepare_message("a" * program_status._MAX_RAW_INPUT_CHARS)
        assert prepared == "a" * program_status._MAX_MSG_BYTES
        assert calls["count"] > 0


@pytest.fixture
def registered_secret() -> Iterator[str]:
    with _registered(_FIXED_SECRET):
        yield _FIXED_SECRET


# --- redaction pins ------------------------------------------------------


class TestRedactionPins:
    def test_encoded_candidate_scanner_exists(self) -> None:
        assert callable(sr._encoded_candidates)

    def test_encoded_candidate_budget_exists(self) -> None:
        assert isinstance(sr._MAX_ENCODED_CANDIDATES, int)
        assert sr._MAX_ENCODED_CANDIDATES > 0

    def test_pipeline_uses_the_redactor_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A run of at least _MIN_ENCODED_RUN alphabet characters is required
        # before the scan produces candidates at all.
        with _registered(_FIXED_SECRET):
            monkeypatch.setattr(sr, "_MAX_ENCODED_CANDIDATES", 1)
            assert prepare_message("abcdefgh bravo charlie") is None


# --- security battery ----------------------------------------------------


class TestSecurityBattery:
    def test_u200b_split_secret_is_caught_after_the_strip(self) -> None:
        secret = _FIXED_SECRET
        split = secret[:10] + "\u200b" + secret[10:]
        with _registered(secret):
            # Non-vacuity: pass 1 cannot see the value across the split.
            assert sr.redact(split) == split
            prepared = prepare_message(f"Run with {split} now")
        assert prepared is not None
        assert secret not in prepared
        assert sr.REDACTED_PLACEHOLDER in prepared

    def test_mod4_1_two_layer_carrier_is_caught_only_after_truncation(self) -> None:
        secret = _LONG_SECRET
        layer_one = base64.b64encode(secret.encode()).decode()
        layer_two = base64.b64encode(layer_one.encode()).decode()
        carrier = layer_two + "A"
        with _registered(secret):
            # Fixture pin: 1152-byte secret -> 2048-char two-layer carrier, and
            # the one extra trailing char makes it 2049 (len % 4 == 1), which
            # the decode scan skips. The 2048-byte truncation strips that char
            # and rebuilds a decodable carrier, so only pass 3 can catch it.
            assert len(secret.encode()) == 1152
            assert len(layer_two) == 2048
            assert len(carrier) == 2049
            assert len(carrier) % 4 == 1
            # Mandatory non-vacuity: neither the fast path nor the scan sees it.
            assert carrier not in sr.known_secret_values()
            assert sr.redact(carrier) == carrier
            prepared = prepare_message(carrier)
        assert prepared == sr.REDACTED_PLACEHOLDER
        assert secret not in prepared
        assert layer_one not in prepared

    def test_candidate_budget_starvation_omits_the_message(self) -> None:
        secret = _FIXED_SECRET
        witness = ("abcdefgh x " * 64) + "!!! " + secret.encode().hex()
        with _registered(secret):
            assert len(sr._encoded_candidates(witness)) >= sr._MAX_ENCODED_CANDIDATES
            # The leak this guards against: the carrier survives every pass.
            assert sr.redact(witness) == witness
            assert prepare_message(witness) is None

    def test_hex_carrier_alone_is_masked_and_present(self) -> None:
        secret = _FIXED_SECRET
        with _registered(secret):
            prepared = prepare_message(secret.encode().hex())
        assert prepared is not None
        assert prepared == sr.REDACTED_PLACEHOLDER
        assert secret not in prepared

    def test_accounting_is_skipped_without_known_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sr, "known_secret_values", lambda: frozenset())
        witness = ("abcdefgh x " * 64) + "!!! " + "a" * 400
        assert prepare_message(witness) is not None


# --- emitter -------------------------------------------------------------


class TestProgramStatusEmitter:
    def test_emits_through_the_driver(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        app._driver.write.assert_called_once_with(
            "\x1b]7501;state=working:app=chartreux\x07"
        )

    def test_silent_when_headless(self) -> None:
        app = _fake_app(is_headless=True)
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("idle")
        app._driver.write.assert_not_called()

    def test_silent_without_a_driver(self) -> None:
        app = _fake_app(driver=None)
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("idle")
        assert app._driver is None

    def test_silent_when_the_config_gate_is_off(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: False)
        emitter.emit("idle")
        app._driver.write.assert_not_called()

    def test_silent_when_the_app_is_not_running(self) -> None:
        app = _fake_app(is_running=False)
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("idle")
        app._driver.write.assert_not_called()

    def test_shutdown_check_runs_inside_emit(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("idle")
        app._shutdown_started = True
        emitter.emit("done")
        app._driver.write.assert_called_once()

    def test_dedup_skips_identical_reports(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        emitter.emit("working")
        app._driver.write.assert_called_once()

    def test_dedup_reemits_a_second_blocked_message(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("blocked", kind="question", msg="first question")
        emitter.emit("blocked", kind="question", msg="second question")
        assert app._driver.write.call_count == 2
        assert _decoded_msg(app._driver.write.call_args.args[0]) == "second question"

    def test_republish_bypasses_dedup(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("blocked", kind="permission", msg="Allow this command?")
        emitter.emit("blocked", kind="permission", msg="Allow this command?")
        assert app._driver.write.call_count == 1
        emitter.republish()
        assert app._driver.write.call_count == 2
        assert app._driver.write.call_args.args[0] == (
            "\x1b]7501;state=blocked:kind=permission:app=chartreux:msg="
            + base64.b64encode(b"Allow this command?").decode("ascii")
            + "\x07"
        )

    def test_republish_without_a_prior_report_writes_nothing(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.republish()
        app._driver.write.assert_not_called()

    def test_republish_respects_the_guards(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        app._shutdown_started = True
        emitter.republish()
        app._driver.write.assert_called_once()

    def test_blocked_prompt_is_encoded_with_kind_and_pipeline_output(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("blocked", kind="question", msg="Use it?\n\nDetails")
        report = app._driver.write.call_args.args[0]
        assert report.startswith(
            "\x1b]7501;state=blocked:kind=question:app=chartreux:msg="
        )
        assert _decoded_msg(report) == "Use it?"

    def test_unencodable_message_is_sent_state_only(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("blocked", kind="question", msg="bad \ud800 text")
        app._driver.write.assert_called_once_with(
            "\x1b]7501;state=blocked:kind=question:app=chartreux\x07"
        )

    def test_redaction_raise_is_sent_state_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(text: str) -> str:
            raise RuntimeError("redaction failed")

        monkeypatch.setattr(sr, "redact", failing)
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("blocked", kind="permission", msg="Allow?")
        app._driver.write.assert_called_once_with(
            "\x1b]7501;state=blocked:kind=permission:app=chartreux\x07"
        )

    def test_refused_state_never_raises(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("clear")
        app._driver.write.assert_not_called()

    def test_over_limit_pipeline_output_never_writes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(program_status, "prepare_message", lambda _text: "a" * 2049)
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("done", msg="whatever")
        app._driver.write.assert_not_called()

    def test_raw_input_past_the_bound_still_reports_the_state(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit(
            "blocked",
            kind="question",
            msg="a" * (program_status._MAX_RAW_INPUT_CHARS + 1),
        )
        app._driver.write.assert_called_once_with(
            "\x1b]7501;state=blocked:kind=question:app=chartreux\x07"
        )

    def test_write_failure_is_swallowed_and_not_deduped(self) -> None:
        app = _fake_app()
        app._driver.write.side_effect = RuntimeError("driver closed")
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        emitter.emit("working")
        assert app._driver.write.call_count == 2

    def test_gate_exception_is_swallowed(self) -> None:
        def exploding_gate() -> bool:
            raise RuntimeError("config unavailable")

        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=exploding_gate)
        emitter.emit("idle")
        app._driver.write.assert_not_called()

    def test_clear_record_writes_while_shutdown_started_and_not_running(self) -> None:
        # THE DISCRIMINATING BYPASS PIN: an emit()-routed implementation is
        # suppressed by _can_write() here, and _write() would leave the clear
        # as _last_report; both fail this node, so never weaken it.
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        app._shutdown_started = True
        app.is_running = False
        emitter.clear_record()
        assert app._driver.write.call_count == 2
        assert app._driver.write.call_args.args[0] == encode_clear()

    def test_clear_record_writes_with_the_config_gate_off(self) -> None:
        gate = True
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: gate)
        emitter.emit("working")
        gate = False
        emitter.clear_record()
        assert app._driver.write.call_count == 2
        assert app._driver.write.call_args.args[0] == encode_clear()

    def test_clear_record_without_a_prior_report_writes_nothing(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.clear_record()
        app._driver.write.assert_not_called()

    def test_clear_record_writes_nothing_when_headless(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        # is_headless is a property on the mock type (see _fake_app), so the
        # flip after the seed reinstalls it on the type.
        type(app).is_headless = PropertyMock(return_value=True)
        emitter.clear_record()
        app._driver.write.assert_called_once()

    def test_clear_record_writes_nothing_without_a_driver(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        driver = app._driver
        app._driver = None
        emitter.clear_record()
        assert app._driver is None
        driver.write.assert_called_once()

    def test_clear_record_does_not_bump_generation(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        generation = emitter.generation
        emitter.clear_record()
        assert emitter.generation == generation
        app._driver.write.assert_called_with(encode_clear())

    def test_republish_after_clear_record_writes_nothing(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("working")
        emitter.clear_record()
        assert app._driver.write.call_count == 2
        emitter.republish()
        assert app._driver.write.call_count == 2

    def test_clear_record_writes_the_encoded_clear_bytes(self) -> None:
        app = _fake_app()
        emitter = ProgramStatusEmitter(app, get_enabled=lambda: True)
        emitter.emit("idle")
        emitter.clear_record()
        app._driver.write.assert_called_with(encode_clear())
        assert app._driver.write.call_args.args[0] == ("\x1b]7501;state=clear\x07")


# --- app-level emission (WP4a) -------------------------------------------
#
# The app owns the emitter and the ownership epochs. These sections pin the
# lifecycle hooks, the production gate (config -> ConfigView -> gate), and the
# binding-helper contract that WP4b/WP5a call from their start sites.


def _app_with_mock_emitter() -> tuple[ChartreuxApp, MagicMock]:
    emitter = MagicMock(spec=ProgramStatusEmitter)
    return build_test_chartreux_app(program_status=emitter), emitter


def _turn_started(app: ChartreuxApp, turn_id: str) -> TurnStarted:
    return TurnStarted(
        PublicTurn(
            id=turn_id,
            session_id=app.app_server.session_id,
            status=PublicTurnStatus.IN_PROGRESS,
            started_at=1,
        )
    )


def _accept_at_the_start_site(app: ChartreuxApp, *, accepted: bool) -> None:
    """The contract every accepted-start site follows (WP4b/WP5a call sites).

    The monotonic count is read before the enqueue await; the bump and the slot
    stash happen only when the enqueue reported acceptance.
    """
    started_count = app._started_count_snapshot()
    if accepted:
        app._record_accepted_start(started_count)


def _emitted_states(emitter: MagicMock) -> list[Any]:
    return [call_.args[0] for call_ in emitter.emit.call_args_list]


class TestAppEmissionLifecycle:
    @pytest.mark.asyncio
    async def test_startup_reports_idle_state_only(self) -> None:
        fake = _fake_app()
        emitter = ProgramStatusEmitter(fake, get_enabled=lambda: True)
        app = build_test_chartreux_app(program_status=emitter)
        async with app.run_test():
            await app._session_ready.wait()
            # Several readiness points run at startup; dedup keeps exactly one
            # state-only report on the wire.
            assert fake._driver.write.call_count == 1
            assert (
                fake._driver.write.call_args.args[0]
                == "\x1b]7501;state=idle:app=chartreux\x07"
            )

    @pytest.mark.asyncio
    async def test_session_ready_never_bumps_the_epoch(self) -> None:
        fake = _fake_app()
        emitter = ProgramStatusEmitter(fake, get_enabled=lambda: True)
        app = build_test_chartreux_app(program_status=emitter)
        async with app.run_test():
            assert fake._driver.write.call_count == 1
            app._mark_session_ready()
            app._mark_session_ready()
            # Dedup collapses the extra readiness points to the same report.
            assert fake._driver.write.call_count == 1
            assert app._record_epoch == 0
            assert app._pending_start_epoch is None

    @pytest.mark.asyncio
    async def test_clear_history_reports_idle_and_bumps_the_epoch(self) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            app.app_server.clear_history = AsyncMock()
            app._reset_message_widgets = AsyncMock()
            app._mount_and_scroll = AsyncMock()
            emitter.reset_mock()
            epoch = app._record_epoch
            await app._clear_history()
            assert _emitted_states(emitter) == ["idle"]
            assert app._record_epoch == epoch + 1
            assert app._pending_start_epoch is None

    @pytest.mark.asyncio
    async def test_compact_success_reports_idle_and_bumps_the_epoch(self) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            app.app_server.compact = AsyncMock()
            emitter.reset_mock()
            epoch = app._record_epoch
            await app._run_compact(CompactMessage())
            assert _emitted_states(emitter) == ["idle"]
            assert app._record_epoch == epoch + 1

    @pytest.mark.asyncio
    async def test_context_cleared_reports_idle_and_bumps_the_epoch(self) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            app._reset_message_widgets = AsyncMock()
            app._mount_and_scroll = AsyncMock()
            emitter.reset_mock()
            epoch = app._record_epoch
            await app._on_context_cleared()
            assert _emitted_states(emitter) == ["idle"]
            assert app._record_epoch == epoch + 1

    @pytest.mark.asyncio
    async def test_resume_bumps_the_epoch_and_its_idle_collapses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _fake_app()
        emitter = ProgramStatusEmitter(fake, get_enabled=lambda: True)
        app = build_test_chartreux_app(program_status=emitter)
        async with app.run_test():
            idle_report = "\x1b]7501;state=idle:app=chartreux\x07"
            assert fake._driver.write.call_args.args[0] == idle_report
            writes = fake._driver.write.call_count
            epoch = app._record_epoch
            monkeypatch.setattr(app._session_flow, "resume_local_session", AsyncMock())
            await app._resume_local_session("abcd1234")
            assert app._record_epoch == epoch + 1
            assert app._pending_start_epoch is None
            # The resume flow's own return-to-input point; same state, no write.
            app._mark_session_ready()
            assert fake._driver.write.call_count == writes

    @pytest.mark.asyncio
    async def test_resume_signal_republishes_bypassing_dedup(self) -> None:
        fake = _fake_app()
        emitter = ProgramStatusEmitter(fake, get_enabled=lambda: True)
        app = build_test_chartreux_app(program_status=emitter)
        async with app.run_test():
            assert fake._driver.write.call_count == 1
            first_report = fake._driver.write.call_args.args[0]
            event: Any = MagicMock()
            app._on_driver_signal_resume(event)
            assert fake._driver.write.call_count == 2
            assert fake._driver.write.call_args.args[0] == first_report


class TestProductionGate:
    @pytest.mark.asyncio
    async def test_gate_follows_the_config_view(self) -> None:
        app = build_test_chartreux_app()
        async with app.run_test():
            assert app.config.emit_program_status is True
            assert app._program_status._get_enabled() is True

    @pytest.mark.asyncio
    async def test_gate_blocks_a_disabled_config_key(self) -> None:
        config = build_test_vibe_config(emit_program_status=False)
        app = build_test_chartreux_app(config=config)
        async with app.run_test():
            assert app.config.emit_program_status is False
            assert app._program_status._get_enabled() is False
            assert app._program_status._can_write() is False

    def test_gate_blocks_without_an_attached_session(self) -> None:
        app = build_test_chartreux_app()
        assert app._app_server is None
        assert app._program_status._get_enabled() is False

    def test_injected_emitter_replaces_the_default(self) -> None:
        emitter = MagicMock(spec=ProgramStatusEmitter)
        app = build_test_chartreux_app(program_status=emitter)
        assert app._program_status is emitter


class TestOwnershipEpochs:
    @pytest.mark.asyncio
    async def test_consumed_turn_started_binds_and_bumps_once(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            epoch = app._record_epoch
            await app._handle_turn_event(_turn_started(app, "turn-1"))
            assert app.turn_epoch_map == {"turn-1": epoch + 1}
            assert app._record_epoch == epoch + 1
            assert app._turn_started_count == 1
            assert app._active_turn_id == "turn-1"
            assert app._pending_start_epoch is None

    @pytest.mark.asyncio
    async def test_unsolicited_listener_binds_before_its_awaits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            epoch = app._record_epoch
            observed = asyncio.Event()

            async def events() -> Any:
                yield _turn_started(app, "turn-1")
                observed.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(app.app_server, "events", events)
            listener = asyncio.create_task(app._listen_app_server_events())
            try:
                await asyncio.wait_for(observed.wait(), timeout=5)
            finally:
                listener.cancel()
                with suppress(asyncio.CancelledError):
                    await listener
            assert app.turn_epoch_map == {"turn-1": epoch + 1}
            assert app._record_epoch == epoch + 1
            assert app._turn_started_count == 1

    @pytest.mark.asyncio
    async def test_already_mapped_turn_is_skipped(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            state = (
                app._record_epoch,
                app._turn_started_count,
                dict(app.turn_epoch_map),
            )
            app._record_turn_owner("turn-1")
            assert (
                app._record_epoch,
                app._turn_started_count,
                dict(app.turn_epoch_map),
            ) == state

    @pytest.mark.asyncio
    async def test_acceptance_slot_binds_without_a_second_bump(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            _accept_at_the_start_site(app, accepted=True)
            accepted_epoch = app._record_epoch
            assert app._pending_start_epoch == accepted_epoch
            app._record_turn_owner("turn-1")
            assert app.turn_epoch_map == {"turn-1": accepted_epoch}
            assert app._record_epoch == accepted_epoch
            assert app._pending_start_epoch is None
            assert app._turn_started_count == 1
            assert app._active_turn_id == "turn-1"

    @pytest.mark.asyncio
    async def test_event_first_start_is_not_double_bumped(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            started_count = app._started_count_snapshot()
            app._record_turn_owner("turn-1")
            epoch = app._record_epoch
            app._record_accepted_start(started_count)
            assert app._record_epoch == epoch
            assert app._pending_start_epoch is None
            assert app._turn_started_count == 1

    @pytest.mark.asyncio
    async def test_rejected_enqueue_neither_bumps_nor_sets_the_slot(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            _accept_at_the_start_site(app, accepted=False)
            assert app._record_epoch == 0
            assert app._pending_start_epoch is None
            assert app._turn_started_count == 0
            # The accepted variant is the discriminator.
            _accept_at_the_start_site(app, accepted=True)
            assert app._record_epoch == 1
            assert app._pending_start_epoch == 1

    @pytest.mark.asyncio
    async def test_non_acceptance_bump_discards_the_slot(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            _accept_at_the_start_site(app, accepted=True)
            assert app._pending_start_epoch is not None
            # An interrupt (or any other lifecycle bump) between acceptance and
            # TurnStarted invalidates the correlation.
            app._advance_record_epoch()
            assert app._pending_start_epoch is None
            epoch = app._record_epoch
            app._record_turn_owner("turn-1")
            assert app.turn_epoch_map == {"turn-1": epoch + 1}

    @pytest.mark.asyncio
    async def test_second_acceptance_invalidates_the_first_slot(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            _accept_at_the_start_site(app, accepted=True)
            first = app._record_epoch
            _accept_at_the_start_site(app, accepted=True)
            second = app._record_epoch
            assert second == first + 1
            app._record_turn_owner("turn-1")
            # Epoch-only correlation: the late start binds the newest slot.
            assert app.turn_epoch_map == {"turn-1": second}

    @pytest.mark.asyncio
    async def test_supersede_retires_the_predecessor_but_not_the_store(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            token = app._ownership_token_for("turn-1", None)
            assert isinstance(token, TurnOwnership)
            app._failure_tokens["turn-1"] = token
            app._record_turn_owner("turn-2")
            assert "turn-1" not in app.turn_epoch_map
            assert app.turn_epoch_map == {"turn-2": app._record_epoch}
            assert app._active_turn_id == "turn-2"
            # Retirement never touches the keyed failure-token store.
            assert app._failure_tokens == {"turn-1": token}


class TestOwnershipTokenCapture:
    @pytest.mark.asyncio
    async def test_typed_id_in_the_map_freezes_that_epoch(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            token = app._ownership_token_for("turn-1", None)
            assert isinstance(token, TurnOwnership)
            assert token.session_id == app.app_server.session_id
            assert token.record_epoch == app.turn_epoch_map["turn-1"]
            assert token.terminal_turn_id == "turn-1"

    @pytest.mark.asyncio
    async def test_unmapped_typed_id_withholds_without_a_live_fallback(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            token = app._ownership_token_for("turn-2", None)
            assert token is app._OWNERSHIP_MISSING
            assert token is _OWNERSHIP_MISSING
            assert isinstance(token, _OwnershipMissingSentinel)

    @pytest.mark.asyncio
    async def test_no_id_prefers_a_complete_frozen_pair(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            frozen = ("turn-1", app.turn_epoch_map["turn-1"])
            # A successor changes the record; the invocation-local pair does not.
            app._advance_record_epoch()
            token = app._ownership_token_for(None, frozen)
            assert isinstance(token, TurnOwnership)
            assert token.record_epoch == frozen[1]
            assert token.terminal_turn_id == "turn-1"
            assert token.record_epoch != app._record_epoch

    @pytest.mark.asyncio
    async def test_epoch_none_pair_falls_through_to_the_slot(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            _accept_at_the_start_site(app, accepted=True)
            token = app._ownership_token_for(None, ("turn-1", None))
            assert isinstance(token, TurnOwnership)
            assert token.record_epoch == app._record_epoch
            assert token.terminal_turn_id is None

    @pytest.mark.asyncio
    async def test_no_id_without_frozen_context_or_slot_withholds(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            assert app._ownership_token_for(None, None) is app._OWNERSHIP_MISSING
            _accept_at_the_start_site(app, accepted=True)
            app._advance_record_epoch()
            assert app._ownership_token_for(None, None) is app._OWNERSHIP_MISSING

    @pytest.mark.asyncio
    async def test_raising_session_id_read_withholds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")

            def raising(_session: AppServerSession) -> str:
                raise RuntimeError("closing")

            # Scoped: the property must be back before the app's teardown
            # awaits run its workers, which otherwise read a raising session id.
            with monkeypatch.context() as scoped:
                scoped.setattr(AppServerSession, "session_id", property(raising))
                assert (
                    app._ownership_token_for("turn-1", None) is app._OWNERSHIP_MISSING
                )

    def test_missing_session_withholds(self) -> None:
        app = build_test_chartreux_app()
        _accept_at_the_start_site(app, accepted=True)
        assert app._ownership_token_for(None, None) is app._OWNERSHIP_MISSING


class TestDelayedScheduleOwnership:
    @pytest.mark.asyncio
    async def test_frozen_token_reports_its_own_epoch_after_a_successor(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            token = app._ownership_token_for("turn-1", None)
            assert isinstance(token, TurnOwnership)
            frozen = token.record_epoch
            app._record_turn_owner("turn-2")
            # A cancellation idle's pinned predicate is a live-epoch equality,
            # so a frozen token from an older epoch must not match it.
            assert token.record_epoch == frozen
            assert token.record_epoch != app._record_epoch
            assert token.session_id == app.app_server.session_id

    @pytest.mark.asyncio
    async def test_a_retired_turn_captures_the_sentinel(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            app._record_turn_owner("turn-2")
            assert app._ownership_token_for("turn-1", None) is app._OWNERSHIP_MISSING
            # The sentinel carries no record_epoch: the schedule stores None and
            # the pinned predicate is False.
            assert not isinstance(app._OWNERSHIP_MISSING, TurnOwnership)


class TestShutdownReportsNothing:
    @pytest.mark.asyncio
    async def test_shutdown_cleanup_writes_no_report(self) -> None:
        app, emitter = _app_with_mock_emitter()
        await app.prepare()
        emitter.reset_mock()
        app._shutdown_started = True
        await asyncio.wait_for(app.shutdown_cleanup(), timeout=5)
        emitter.emit.assert_not_called()
        emitter.republish.assert_not_called()
        # The exit clear belongs to ``_shutdown`` alone: the post-loop cleanup
        # still writes nothing.
        emitter.clear_record.assert_not_called()


# --- app-level emission (WP4b) -------------------------------------------
#
# The ``!bash`` path and the solicited turn start own the record: one settlement
# holder per shell invocation, one accepted-start bump before the turn's
# ``working``. These sections pin the terminal mapping, the settlement ordering,
# and the start-site contracts.


def _effect_entry(entry_id: str, state: Any) -> PublicEffectEntry:
    return PublicEffectEntry(
        id=entry_id,
        session_id="session-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="shell",
        detail=GenericEffectDetail(
            tool_name="shell",
            display=EffectCallDisplay(summary="shell", status_text="Running shell"),
        ),
        state=state,
    )


def _added(entry: PublicEffectEntry) -> HistoryEntryAdded:
    return HistoryEntryAdded(entry=entry)


def _completed_state() -> CompletedEffectState:
    return CompletedEffectState(display=EffectResultDisplay(success=True, message="ok"))


def _failed_state(message: str, *, output_text: str = "") -> FailedEffectState:
    return FailedEffectState(
        error=PublicError(message=message),
        output_text=output_text,
        display=EffectResultDisplay(success=False, message=message),
    )


class _RecordingHandler:
    """Stand-in for the event handler: records, or blocks once entered.

    Carries the attribute surface the app reads outside the forwarding call
    (startup registry refresh and shutdown stream finalization).
    """

    _retry_presentation: Any = None
    current_compact: Any = None

    def __init__(self, *, entered: asyncio.Event | None = None) -> None:
        self.events: list[Any] = []
        self._entered = entered

    async def handle_event(self, event: Any, *, loading_widget: Any = None) -> None:
        self.events.append(event)
        if self._entered is not None:
            self._entered.set()
            await asyncio.Event().wait()

    async def finalize_streaming(self) -> None:
        return None

    def stop_current_tool_call(self, *args: Any, **kwargs: Any) -> None:
        return None


def _install_shell(
    app: ChartreuxApp,
    monkeypatch: pytest.MonkeyPatch,
    events: Sequence[Any],
    *,
    error: Exception | None = None,
    gate: asyncio.Event | None = None,
) -> None:
    """Replace the app server's shell stream with a scripted one."""

    async def run(command: str) -> Any:
        for event in events:
            yield event
        if error is not None:
            raise error
        if gate is not None:
            await gate.wait()

    monkeypatch.setattr(app.app_server.resources.shell, "run", run)


def _emitted_calls(emitter: MagicMock) -> list[tuple[str, Any]]:
    return [
        (call.args[0], call.kwargs.get("msg")) for call in emitter.emit.call_args_list
    ]


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.005)


class TestBashSettlementReports:
    @pytest.mark.asyncio
    async def test_completed_effect_reports_done_state_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            _install_shell(
                app, monkeypatch, [_added(_effect_entry("op-1", _completed_state()))]
            )
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            await app._handle_bash_command("true")
            # Exit 0: `done` carries no msg, exactly like `working`.
            assert _emitted_calls(emitter) == [("working", None), ("done", None)]

    @pytest.mark.asyncio
    async def test_failed_effect_prefers_the_error_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            entry = _effect_entry(
                "op-1", _failed_state("exit status 2", output_text="boom")
            )
            _install_shell(app, monkeypatch, [_added(entry)])
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            await app._handle_bash_command("false")
            assert _emitted_calls(emitter) == [
                ("working", None),
                ("error", "exit status 2"),
            ]

    @pytest.mark.asyncio
    async def test_failed_effect_falls_back_to_the_output_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            entry = _effect_entry("op-1", _failed_state("", output_text="not found"))
            _install_shell(app, monkeypatch, [_added(entry)])
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            await app._handle_bash_command("nope")
            assert _emitted_calls(emitter) == [
                ("working", None),
                ("error", "not found"),
            ]

    @pytest.mark.asyncio
    async def test_cancelled_effect_reports_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            entry = _effect_entry("op-1", CancelledEffectState(reason="interrupted"))
            _install_shell(app, monkeypatch, [_added(entry)])
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            await app._handle_bash_command("sleep 100")
            assert _emitted_calls(emitter) == [("working", None), ("idle", None)]

    @pytest.mark.asyncio
    async def test_pre_effect_failure_reports_error_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            # The `_workspace_cwd` RequestFailure class: the request fails
            # before `start_effect`, so no effect event is ever forwarded.
            _install_shell(
                app, monkeypatch, [], error=RuntimeError("workspace unavailable")
            )
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            await app._handle_bash_command("pwd")
            calls = _emitted_calls(emitter)
            assert calls == [
                ("working", None),
                ("error", "Command failed: workspace unavailable"),
            ]
            assert sum(1 for state, _ in calls if state == "error") == 1

    @pytest.mark.asyncio
    async def test_entry_id_comes_from_the_first_forwarded_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            settlement = _BashSettlement()
            events = [
                _added(_effect_entry("op-1", PendingEffectState())),
                # A second id never rewrites the captured one.
                _added(_effect_entry("op-2", RunningEffectState())),
                _added(_effect_entry("op-1", _completed_state())),
            ]
            _install_shell(app, monkeypatch, events)
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            await app._handle_bash_command_inner("printf hi", settlement)
            assert settlement.entry_id == "op-1"
            assert settlement.settled is True
            assert _emitted_calls(emitter) == [("working", None), ("done", None)]


class TestBashSettlementOrdering:
    @pytest.mark.asyncio
    async def test_escape_after_a_settled_error_preserves_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            gate = asyncio.Event()
            entry = _effect_entry("op-1", _failed_state("exit status 1"))
            _install_shell(app, monkeypatch, [_added(entry)], gate=gate)
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            task = asyncio.create_task(app._handle_bash_command("false"))
            await _wait_until(lambda: len(_emitted_calls(emitter)) >= 2)
            # Escape cancels the shell task after the failure already landed.
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert _emitted_calls(emitter) == [
                ("working", None),
                ("error", "exit status 1"),
            ]

    @pytest.mark.asyncio
    async def test_cancel_before_any_event_reports_idle_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            _install_shell(app, monkeypatch, [], gate=asyncio.Event())
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            task = asyncio.create_task(app._handle_bash_command("sleep 100"))
            await _wait_until(lambda: len(_emitted_calls(emitter)) >= 1)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert _emitted_calls(emitter) == [("working", None), ("idle", None)]

    @pytest.mark.asyncio
    async def test_cancel_during_cleanup_after_a_settled_outcome_adds_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            cleanup_started = asyncio.Event()

            async def hanging_remove() -> None:
                cleanup_started.set()
                await asyncio.Event().wait()

            _install_shell(
                app, monkeypatch, [_added(_effect_entry("op-1", _completed_state()))]
            )
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            monkeypatch.setattr(app, "_ensure_loading_widget", AsyncMock())
            monkeypatch.setattr(app, "_remove_loading_widget", hanging_remove)
            task = asyncio.create_task(app._handle_bash_command("true"))
            await asyncio.wait_for(cleanup_started.wait(), timeout=5)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            # The settled `done` stands; the backstop reads the holder, not a map.
            assert _emitted_calls(emitter) == [("working", None), ("done", None)]

    @pytest.mark.asyncio
    async def test_cancel_inside_a_slow_handler_keeps_the_settled_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            handler_entered = asyncio.Event()
            entry = _effect_entry("op-1", _failed_state("exit status 3"))
            _install_shell(app, monkeypatch, [_added(entry)])
            monkeypatch.setattr(
                app, "event_handler", _RecordingHandler(entered=handler_entered)
            )
            task = asyncio.create_task(app._handle_bash_command("false"))
            await asyncio.wait_for(handler_entered.wait(), timeout=5)
            # Cancelled inside the forwarding await, not between iterations:
            # the terminal report was settled before that await began.
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert _emitted_calls(emitter) == [
                ("working", None),
                ("error", "exit status 3"),
            ]

    @pytest.mark.asyncio
    async def test_startup_cancel_reports_one_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            entered = asyncio.Event()

            async def hanging_ensure(*args: Any, **kwargs: Any) -> None:
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(app, "_ensure_loading_widget", hanging_ensure)
            _install_shell(app, monkeypatch, [], gate=asyncio.Event())
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            task = asyncio.create_task(app._handle_bash_command("sleep 5"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            # Cancelled inside the loading-widget await: never `working`, one idle.
            assert _emitted_calls(emitter) == [("idle", None)]

    @pytest.mark.asyncio
    async def test_a_shell_run_owns_the_record_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            _install_shell(app, monkeypatch, [], gate=asyncio.Event())
            monkeypatch.setattr(app, "event_handler", _RecordingHandler())
            turn_start = AsyncMock()
            monkeypatch.setattr(app, "_handle_user_message", turn_start)
            epoch = app._record_epoch
            await app._dispatch_idle_input("!sleep 30")
            assert app._is_busy() is True
            await _wait_until(lambda: _emitted_calls(emitter) == [("working", None)])
            # A prompt submitted now takes the busy branch: no turn start, no
            # second bump, no report from the turn path.
            await app._dispatch_submitted_value("hello")
            turn_start.assert_not_called()
            assert app._record_epoch == epoch + 1
            assert app.turn_epoch_map == {}
            assert app._pending_start_epoch is None
            task = app._bash_task
            assert task is not None
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert _emitted_calls(emitter) == [("working", None), ("idle", None)]


class TestSolicitedTurnStart:
    @pytest.mark.asyncio
    async def test_accepted_start_bumps_then_reports_working(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            epoch = app._record_epoch
            accepted: list[str] = []

            async def accept(message: str, **kwargs: Any) -> bool:
                accepted.append(message)
                return True

            monkeypatch.setattr(app, "_enqueue_prompt_with_resources", accept)
            await app._handle_user_message("hello")
            assert accepted == ["hello"]
            assert app._record_epoch == epoch + 1
            assert app._pending_start_epoch == app._record_epoch
            assert app._turn_started_count == 0
            # `working` carries no msg and is emitted with the bump, after the
            # protected await.
            assert _emitted_calls(emitter) == [("working", None)]

    @pytest.mark.asyncio
    async def test_rejected_enqueue_reports_idle_without_bumping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            epoch = app._record_epoch

            async def reject(message: str, **kwargs: Any) -> bool:
                return False

            monkeypatch.setattr(app, "_enqueue_prompt_with_resources", reject)
            await app._handle_user_message("hello")
            assert _emitted_calls(emitter) == [("idle", None)]
            assert app._record_epoch == epoch
            assert app._pending_start_epoch is None
            assert app._turn_started_count == 0
            assert app.turn_epoch_map == {}
            assert app._turn_supervisor.pending_turn is False

    @pytest.mark.asyncio
    async def test_startup_cancel_before_the_enqueue_reports_idle_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            emitter.reset_mock()
            entered = asyncio.Event()

            async def hanging_ensure(*args: Any, **kwargs: Any) -> None:
                entered.set()
                await asyncio.Event().wait()

            async def never(message: str, **kwargs: Any) -> bool:
                raise AssertionError("the enqueue must not be reached")

            monkeypatch.setattr(app, "_ensure_loading_widget", hanging_ensure)
            monkeypatch.setattr(app, "_enqueue_prompt_with_resources", never)
            epoch = app._record_epoch
            task = asyncio.create_task(app._handle_user_message("hello"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            # The pre-try await now sits inside the protected try, so the
            # re-raise arm still leaves the record idle.
            assert _emitted_calls(emitter) == [("idle", None)]
            assert app._record_epoch == epoch
            assert app._pending_start_epoch is None
            assert app._turn_supervisor.pending_turn is False


# --- turn-supervisor outcomes (WP5a) -------------------------------------
#
# The supervisor owns outcome threading, ownership capture, retirement, and the
# blocked/working return-to-input reports. These tests drive its real methods
# with a scripted client turn stream and a mock emitter.


def _turn_completed(
    app: ChartreuxApp,
    turn_id: str,
    status: PublicTurnStatus,
    *,
    next_turn_id: str | None = None,
    error: PublicError | None = None,
) -> TurnCompleted:
    return TurnCompleted(
        PublicTurn(
            id=turn_id,
            session_id=app.app_server.session_id,
            status=status,
            started_at=1,
            completed_at=2,
            error=error,
            next_turn_id=next_turn_id,
        )
    )


def _stub_supervisor_ui(app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the supervisor's UI collaborators with no-op recorders."""
    monkeypatch.setattr(app._turn_supervisor, "ensure_runtime_ready", AsyncMock())
    monkeypatch.setattr(app, "event_handler", None)
    monkeypatch.setattr(app, "_remove_loading_widget", AsyncMock())
    monkeypatch.setattr(app, "_ensure_loading_widget", AsyncMock())
    monkeypatch.setattr(app, "_mount_and_scroll", AsyncMock())
    monkeypatch.setattr(app, "_refresh_windowing_from_history", AsyncMock())
    monkeypatch.setattr(
        app._root_recovery, "observe_terminal", AsyncMock(return_value=False)
    )


def _install_client_turn(
    app: ChartreuxApp,
    monkeypatch: pytest.MonkeyPatch,
    events: Sequence[Any],
    *,
    error: Exception | None = None,
    tracker: tuple[str, PublicTurnStatus] | None = None,
    bind: bool = True,
    gate: asyncio.Event | None = None,
) -> None:
    """Drive ``handle_turn`` through a scripted ``act`` stream.

    ``tracker`` mirrors the real generator's terminal-status tracker: the fake
    generator writes it after its last event, like a normal completion.
    """

    async def fake_handle_event(event: Any) -> None:
        if bind and isinstance(event, TurnStarted):
            app._record_turn_owner(event.turn.id)

    def act(*args: Any, **kwargs: Any) -> Any:
        async def stream() -> Any:
            for event in events:
                yield event
            if error is not None:
                raise error
            if tracker is not None:
                app.app_server.last_turn_status = tracker
            if gate is not None:
                await gate.wait()

        return stream()

    monkeypatch.setattr(app, "_handle_turn_event", fake_handle_event)
    monkeypatch.setattr(app.app_server, "act", act)


def _record_mounts(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    mounts: list[dict[str, Any]] = []

    async def mount(
        error: Exception, message: str, *, ownership: object | None = None
    ) -> None:
        mounts.append({"error": error, "message": message, "ownership": ownership})

    monkeypatch.setattr(app._turn_supervisor, "_mount_turn_error", mount)
    return mounts


def _probe_store_at_error(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """Record the keyed store as it stands when ``_handle_turn_error`` starts."""
    seen: list[dict[str, Any]] = []

    async def probe(*, cancelled: bool = False) -> None:
        seen.append(dict(app._failure_tokens))

    monkeypatch.setattr(app._turn_supervisor, "_handle_turn_error", probe)
    return seen


def _spy_finalize(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    original = app._turn_supervisor.finalize_turn_ui

    async def spy(**kwargs: Any) -> None:
        calls.append(dict(kwargs))
        await original(**kwargs)

    monkeypatch.setattr(app._turn_supervisor, "finalize_turn_ui", spy)
    return calls


class TestFinalizeOutcomeDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            (TurnOutcome.COMPLETED, ["done"]),
            (TurnOutcome.INTERRUPTED, ["idle"]),
            (TurnOutcome.FAILED, []),
            (TurnOutcome.NO_REPORT, []),
        ],
    )
    async def test_outcome_rows_dispatch(
        self, outcome: TurnOutcome, expected: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor.finalize_turn_ui(
                outcome=outcome, turn_id="turn-1"
            )
            assert _emitted_states(emitter) == expected

    @pytest.mark.asyncio
    async def test_stale_generation_neither_reports_nor_retires(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            app._turn_supervisor.turn_ui_generation = 2
            emitter.reset_mock()
            await app._turn_supervisor.finalize_turn_ui(
                outcome=TurnOutcome.COMPLETED, turn_ui_generation=1, turn_id="turn-1"
            )
            assert _emitted_states(emitter) == []
            assert "turn-1" in app.turn_epoch_map

    @pytest.mark.asyncio
    async def test_notify_complete_false_skips_the_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor.finalize_turn_ui(
                notify_complete=False, outcome=TurnOutcome.COMPLETED, turn_id="turn-1"
            )
            assert _emitted_states(emitter) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", [TurnOutcome.FAILED, TurnOutcome.INTERRUPTED])
    async def test_complete_notification_still_fires(
        self, outcome: TurnOutcome, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            notifier = MagicMock()
            monkeypatch.setattr(app._terminal_notifier, "notify", notifier)
            app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor.finalize_turn_ui(
                outcome=outcome, turn_id="turn-1"
            )
            notifier.assert_any_call(NotificationContext.COMPLETE)

    @pytest.mark.asyncio
    async def test_app_side_finalize_forwards_the_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            supervisor = app._turn_supervisor
            forwarded = AsyncMock()

            async def record(**kwargs: Any) -> None:
                await forwarded(**kwargs)

            monkeypatch.setattr(supervisor, "finalize_turn_ui", record)
            await app._finalize_turn_ui(outcome=TurnOutcome.INTERRUPTED)
            assert forwarded.await_args is not None
            assert forwarded.await_args.kwargs["outcome"] is TurnOutcome.INTERRUPTED
            await app._finalize_turn_ui()
            assert forwarded.await_args.kwargs["outcome"] is TurnOutcome.NO_REPORT

    @pytest.mark.asyncio
    async def test_terminal_report_precedes_the_teardown_await(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successor accepted during the finalizer's teardown await must not
        be overwritten by the predecessor's terminal report.

        The report is emitted before the first await, so the wire reads
        predecessor-terminal THEN successor-working — never successor-working
        followed by the stale predecessor outcome.
        """
        app, host = _wire_app()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            app._program_status.emit("working")
            # A distinct predecessor state just before finalization: without it
            # the successor's `working` would dedup against the predecessor's
            # and mask the ordering this test pins.
            app._program_status.emit("blocked", kind="question", msg="Pick one")
            entered = asyncio.Event()
            released = asyncio.Event()

            class _ParkedLoading:
                parent: object = object()

                async def remove(self) -> None:
                    entered.set()
                    await released.wait()

            monkeypatch.setattr(app, "_loading_widget", _ParkedLoading())
            finalizer = asyncio.create_task(
                app._turn_supervisor.finalize_turn_ui(
                    outcome=TurnOutcome.COMPLETED, turn_id="turn-1"
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            # The successor is accepted and surfaces inside the teardown await.
            _accept_at_the_start_site(app, accepted=True)
            app._program_status.emit("working")
            app._record_turn_owner("turn-2")
            released.set()
            await finalizer
            assert _wire_states(host) == [
                "idle",
                "working",
                "blocked",
                "done",
                "working",
            ]
            assert app.turn_epoch_map == {"turn-2": app._record_epoch}
            assert app._active_turn_id == "turn-2"


class TestRetirement:
    @pytest.mark.asyncio
    async def test_finalize_retires_the_active_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor.finalize_turn_ui(turn_id="turn-1")
            assert app.turn_epoch_map == {}
            assert _emitted_states(emitter) == []

    @pytest.mark.asyncio
    async def test_late_finalizer_cannot_retire_a_successor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            app._record_turn_owner("turn-2")
            emitter.reset_mock()
            await app._turn_supervisor.finalize_turn_ui(turn_id="turn-1")
            assert "turn-2" in app.turn_epoch_map
            assert app._active_turn_id == "turn-2"

    @pytest.mark.asyncio
    async def test_repeated_turns_stay_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            for index in range(3):
                turn_id = f"turn-{index}"
                app._record_turn_owner(turn_id)
                await app._turn_supervisor.finalize_turn_ui(turn_id=turn_id)
                assert app.turn_epoch_map == {}

    @pytest.mark.asyncio
    async def test_repeated_chain_stays_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            for index in range(3):
                app._record_turn_owner(f"turn-{index}")
                # Supersede-time retirement keeps exactly one live entry.
                assert len(app.turn_epoch_map) == 1
            assert app._active_turn_id == "turn-2"
            await app._turn_supervisor.finalize_turn_ui(turn_id="turn-2")
            assert app.turn_epoch_map == {}

    @pytest.mark.asyncio
    async def test_interrupt_settles_and_retires_before_the_cancel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_agent_job_active", lambda: True)
            app._record_turn_owner("turn-1")
            epoch = app._record_epoch
            emitter.reset_mock()
            task = asyncio.create_task(asyncio.Event().wait())
            app._turn_supervisor.agent_task = task
            try:
                await app._turn_supervisor.interrupt_turn()
            finally:
                with suppress(asyncio.CancelledError):
                    await task
            assert task.cancelled()
            assert app.turn_epoch_map == {}
            assert app._active_turn_id is None
            assert app._record_epoch == epoch + 1
            assert app._pending_start_epoch is None
            assert _emitted_calls(emitter) == [("idle", None)]


class TestSolicitedOutcomes:
    @pytest.mark.asyncio
    async def test_completed_turn_reports_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [
                    _turn_started(app, "turn-1"),
                    _turn_completed(app, "turn-1", PublicTurnStatus.COMPLETED),
                ],
                tracker=("turn-1", PublicTurnStatus.COMPLETED),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert _emitted_calls(emitter) == [("done", None)]
            assert app.turn_epoch_map == {}

    @pytest.mark.asyncio
    async def test_interrupted_tracker_reports_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [
                    _turn_started(app, "turn-1"),
                    _turn_completed(app, "turn-1", PublicTurnStatus.INTERRUPTED),
                ],
                tracker=("turn-1", PublicTurnStatus.INTERRUPTED),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert _emitted_calls(emitter) == [("idle", None)]

    @pytest.mark.asyncio
    async def test_generator_unwind_never_reads_the_tracker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            # A previous turn's COMPLETED is still the session-wide value.
            app.app_server.last_turn_status = ("turn-0", PublicTurnStatus.COMPLETED)
            _install_client_turn(
                app,
                monkeypatch,
                [_turn_started(app, "turn-1")],
                error=RuntimeError("stream closed"),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            # The failure path reports one error and never a durable `done`.
            calls = _emitted_calls(emitter)
            assert [state for state, _ in calls] == ["error"]
            assert calls[0][1]
            assert app.app_server.last_turn_status == (
                "turn-0",
                PublicTurnStatus.COMPLETED,
            )

    @pytest.mark.asyncio
    async def test_recovery_pending_passes_no_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(
                app._root_recovery, "observe_terminal", AsyncMock(return_value=True)
            )
            calls = _spy_finalize(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [_turn_started(app, "turn-1")],
                error=RuntimeError("stream closed"),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert _emitted_calls(emitter) == []
            assert calls[-1]["outcome"] is TurnOutcome.NO_REPORT
            assert calls[-1]["notify_complete"] is False

    @pytest.mark.asyncio
    async def test_chained_turns_stay_working_across_the_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            calls = _spy_finalize(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [
                    _turn_started(app, "turn-1"),
                    _turn_completed(
                        app, "turn-1", PublicTurnStatus.COMPLETED, next_turn_id="turn-2"
                    ),
                    _turn_started(app, "turn-2"),
                    _turn_completed(app, "turn-2", PublicTurnStatus.COMPLETED),
                ],
                tracker=("turn-2", PublicTurnStatus.COMPLETED),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            # Only the terminal segment settles the record.
            assert _emitted_calls(emitter) == [("done", None)]
            assert calls[-1]["turn_id"] == "turn-2"


class TestUnsolicitedOutcomes:
    @pytest.mark.asyncio
    async def test_begin_reports_working(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            bound_epoch = app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor._begin_unsolicited_turn(bound_epoch=bound_epoch)
            assert _emitted_calls(emitter) == [("working", None)]

    @pytest.mark.asyncio
    async def test_completed_reports_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor._complete_unsolicited_turn(
                _turn_completed(app, "turn-1", PublicTurnStatus.COMPLETED)
            )
            assert _emitted_calls(emitter) == [("done", None)]

    @pytest.mark.asyncio
    async def test_interrupted_reports_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor._complete_unsolicited_turn(
                _turn_completed(app, "turn-1", PublicTurnStatus.INTERRUPTED)
            )
            assert _emitted_calls(emitter) == [("idle", None)]

    @pytest.mark.asyncio
    async def test_chained_completion_does_not_finalize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            calls = _spy_finalize(app, monkeypatch)
            emitter.reset_mock()
            await app._turn_supervisor._complete_unsolicited_turn(
                _turn_completed(
                    app, "turn-1", PublicTurnStatus.COMPLETED, next_turn_id="turn-2"
                )
            )
            assert _emitted_calls(emitter) == []
            assert calls == []
            # The chain keeps its live ownership entry for the successor.
            assert "turn-1" in app.turn_epoch_map

    @pytest.mark.asyncio
    async def test_failed_ignores_the_tracker_and_reports_one_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            app.app_server.last_turn_status = ("turn-1", PublicTurnStatus.COMPLETED)
            emitter.reset_mock()
            await app._turn_supervisor._complete_unsolicited_turn(
                _turn_completed(
                    app,
                    "turn-1",
                    PublicTurnStatus.FAILED,
                    error=PublicError(message="boom"),
                )
            )
            calls = _emitted_calls(emitter)
            assert [state for state, _ in calls] == ["error"]
            assert calls[0][1]
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_failed_with_recovery_pending_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(
                app._root_recovery, "observe_terminal", AsyncMock(return_value=True)
            )
            app._record_turn_owner("turn-1")
            emitter.reset_mock()
            await app._turn_supervisor._complete_unsolicited_turn(
                _turn_completed(
                    app,
                    "turn-1",
                    PublicTurnStatus.FAILED,
                    error=PublicError(message="boom"),
                )
            )
            assert _emitted_calls(emitter) == []
            assert app._failure_tokens == {}


class TestObservedFailureOwnership:
    @pytest.mark.asyncio
    async def test_typed_failure_writes_the_key_before_the_error_await(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            seen = _probe_store_at_error(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            app._record_turn_owner("turn-1")
            _install_client_turn(
                app,
                monkeypatch,
                [],
                error=AppServerTurnError(PublicError(message="boom"), turn_id="turn-1"),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert len(seen) == 1
            assert set(seen[0]) == {"turn-1"}
            token = seen[0]["turn-1"]
            assert isinstance(token, TurnOwnership)
            assert mounts[0]["ownership"] is token
            # The finally pop bounds the store before the finalize await.
            assert app._failure_tokens == {}
            assert _emitted_states(emitter) == []

    @pytest.mark.asyncio
    async def test_unmapped_typed_id_stores_the_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            seen = _probe_store_at_error(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [],
                error=AppServerTurnError(PublicError(message="boom"), turn_id="turn-9"),
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert set(seen[0]) == {"turn-9"}
            assert seen[0]["turn-9"] is app._OWNERSHIP_MISSING
            assert mounts[0]["ownership"] is app._OWNERSHIP_MISSING
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_non_typed_failure_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            seen = _probe_store_at_error(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            _install_client_turn(
                app, monkeypatch, [], error=RuntimeError("stream closed")
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert seen == [{}]
            assert mounts[0]["ownership"] is app._OWNERSHIP_MISSING
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_no_id_uses_the_invocation_frozen_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [_turn_started(app, "turn-1")],
                error=RuntimeError("stream closed"),
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            token = mounts[0]["ownership"]
            assert isinstance(token, TurnOwnership)
            assert token.terminal_turn_id == "turn-1"
            assert token.record_epoch == app._record_epoch
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_no_id_uses_the_accepted_start_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            _accept_at_the_start_site(app, accepted=True)
            _install_client_turn(
                app, monkeypatch, [], error=RuntimeError("startup failure")
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            token = mounts[0]["ownership"]
            assert isinstance(token, TurnOwnership)
            assert token.record_epoch == app._record_epoch
            assert token.terminal_turn_id is None

    @pytest.mark.asyncio
    async def test_no_id_with_a_cleared_slot_withholds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            _accept_at_the_start_site(app, accepted=True)
            app._advance_record_epoch()
            _install_client_turn(
                app, monkeypatch, [], error=RuntimeError("startup failure")
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert mounts[0]["ownership"] is app._OWNERSHIP_MISSING

    @pytest.mark.asyncio
    async def test_epoch_none_pair_falls_through_to_the_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            _accept_at_the_start_site(app, accepted=True)
            # The TurnStarted arrives without binding the turn's own epoch.
            _install_client_turn(
                app,
                monkeypatch,
                [_turn_started(app, "turn-1")],
                error=RuntimeError("stream closed"),
                bind=False,
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            token = mounts[0]["ownership"]
            assert isinstance(token, TurnOwnership)
            assert token.record_epoch == app._record_epoch
            assert token.terminal_turn_id is None

    @pytest.mark.asyncio
    async def test_absent_map_entry_freezes_no_context_without_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [_turn_started(app, "turn-1")],
                error=RuntimeError("stream closed"),
                bind=False,
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert mounts[0]["ownership"] is app._OWNERSHIP_MISSING

    @pytest.mark.asyncio
    async def test_frozen_context_failure_reports_exactly_one_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _install_client_turn(
                app,
                monkeypatch,
                [_turn_started(app, "turn-1")],
                error=RuntimeError("stream closed"),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert [state for state, _ in _emitted_calls(emitter)] == ["error"]
            assert app._mount_and_scroll.await_count == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_successor_epoch_mismatch_suppresses_the_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            entry = _effect_entry("op-1", _completed_state())

            async def handler(event: Any) -> None:
                if isinstance(event, TurnStarted):
                    app._record_turn_owner("turn-1")
                if isinstance(event, HistoryEntryAdded):
                    # A concurrent successor binds the record mid-stream.
                    app._record_turn_owner("turn-2")

            monkeypatch.setattr(app, "_handle_turn_event", handler)
            _install_client_turn(
                app,
                monkeypatch,
                [_turn_started(app, "turn-1"), _added(entry)],
                error=RuntimeError("predecessor failed"),
            )
            # Re-assert the handler (the installer replaced it above).
            monkeypatch.setattr(app, "_handle_turn_event", handler)
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert _emitted_calls(emitter) == []
            # Suppression affects the OSC report only.
            assert app._mount_and_scroll.await_count == 1  # type: ignore[attr-defined]
            assert app._failure_tokens == {}


class TestStoreCleanup:
    @pytest.mark.asyncio
    async def test_fatal_init_reports_idle_and_leaves_the_store_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            mounts = _record_mounts(app, monkeypatch)
            app._fatal_init_error = True
            app._record_turn_owner("turn-1")
            _install_client_turn(
                app,
                monkeypatch,
                [],
                error=AppServerTurnError(PublicError(message="boom"), turn_id="turn-1"),
            )
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert _emitted_calls(emitter) == [("idle", None)]
            assert mounts == []
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_cancellation_during_the_error_await_still_pops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)

            async def cancelling(*, cancelled: bool = False) -> None:
                raise asyncio.CancelledError

            monkeypatch.setattr(app._turn_supervisor, "_handle_turn_error", cancelling)
            app._record_turn_owner("turn-1")
            _install_client_turn(
                app,
                monkeypatch,
                [],
                error=AppServerTurnError(PublicError(message="boom"), turn_id="turn-1"),
            )
            with pytest.raises(asyncio.CancelledError):
                await app._turn_supervisor.handle_turn("hi", injected=True)
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_unsolicited_cancellation_during_the_error_await_pops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)

            async def cancelling(*, cancelled: bool = False) -> None:
                raise asyncio.CancelledError

            monkeypatch.setattr(app._turn_supervisor, "_handle_turn_error", cancelling)
            app._record_turn_owner("turn-1")
            with pytest.raises(asyncio.CancelledError):
                await app._turn_supervisor._complete_unsolicited_turn(
                    _turn_completed(
                        app,
                        "turn-1",
                        PublicTurnStatus.FAILED,
                        error=PublicError(message="boom"),
                    )
                )
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_solicited_finally_leaves_a_same_id_successor_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            successor = app._ownership_token_for("turn-1", None)
            assert isinstance(successor, TurnOwnership)

            async def consume_then_successor(*args: Any, **kwargs: Any) -> bool:
                # What observe_terminal does, then a same-id successor records
                # its own write before this invocation's finally runs.
                consumed = app._failure_tokens.pop("turn-1", None)
                assert consumed is not successor
                app._failure_tokens["turn-1"] = successor
                return False

            monkeypatch.setattr(
                app._root_recovery, "observe_terminal", consume_then_successor
            )
            _install_client_turn(
                app,
                monkeypatch,
                [],
                error=AppServerTurnError(PublicError(message="boom"), turn_id="turn-1"),
            )
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert app._failure_tokens == {"turn-1": successor}

    @pytest.mark.asyncio
    async def test_unsolicited_finally_leaves_a_same_id_successor_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            app._record_turn_owner("turn-1")
            successor = app._ownership_token_for("turn-1", None)
            assert isinstance(successor, TurnOwnership)

            async def consume_then_successor(*args: Any, **kwargs: Any) -> bool:
                consumed = app._failure_tokens.pop("turn-1", None)
                assert consumed is not successor
                app._failure_tokens["turn-1"] = successor
                return False

            monkeypatch.setattr(
                app._root_recovery, "observe_terminal", consume_then_successor
            )
            await app._turn_supervisor._complete_unsolicited_turn(
                _turn_completed(
                    app,
                    "turn-1",
                    PublicTurnStatus.FAILED,
                    error=PublicError(message="boom"),
                )
            )
            # B's token survives A's finally and is B's own write.
            assert app._failure_tokens == {"turn-1": successor}

    @pytest.mark.asyncio
    async def test_pre_start_generic_failure_settles_one_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(
                app._turn_supervisor,
                "ensure_runtime_ready",
                AsyncMock(side_effect=RuntimeError("init failed")),
            )
            _accept_at_the_start_site(app, accepted=True)
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            calls = _emitted_calls(emitter)
            assert [state for state, _ in calls] == ["error"]
            assert calls[0][1]
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_real_fatal_init_reports_one_idle_never_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(
                app._turn_supervisor,
                "ensure_runtime_ready",
                AsyncMock(side_effect=RuntimeError("init failed")),
            )
            app._fatal_init_error = True
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            assert _emitted_calls(emitter) == [("idle", None)]
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_mapped_skip_never_increments_the_started_count(self) -> None:
        app, _ = _app_with_mock_emitter()
        async with app.run_test():
            app._record_turn_owner("turn-1")
            app._record_turn_owner("turn-1")
            assert app._turn_started_count == 1


class _RetryHandler:
    """Minimal event-handler stand-in for the retry admission gate."""

    _retry_presentation: Any = None

    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[Any] = []

    def begin_retry(self, command_message: Any = None) -> bool:
        self.calls.append(command_message)
        return self.allowed

    def cancel_retry_presentation(self) -> None:
        self._retry_presentation = None


class TestRetryEmissions:
    @pytest.mark.asyncio
    async def test_retry_reports_working_after_the_accepted_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_agent_job_active", lambda: False)
            monkeypatch.setattr(app, "event_handler", _RetryHandler())
            monkeypatch.setattr(app, "_handle_turn", AsyncMock())
            epoch = app._record_epoch
            emitter.reset_mock()
            await app._turn_supervisor.retry("more")
            task = app._turn_supervisor.agent_task
            assert task is not None
            await task
            assert _emitted_calls(emitter) == [("working", None)]
            assert app._record_epoch == epoch + 1
            assert app._pending_start_epoch == epoch + 1
            assert app._turn_started_count == 0
            app._turn_supervisor.agent_task = None

    @pytest.mark.asyncio
    async def test_rejected_retry_never_bumps_or_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_agent_job_active", lambda: True)
            epoch = app._record_epoch
            emitter.reset_mock()
            await app._turn_supervisor.retry("more")
            assert _emitted_calls(emitter) == []
            assert app._record_epoch == epoch
            assert app._pending_start_epoch is None
            assert app._turn_supervisor.agent_task is None

    @pytest.mark.asyncio
    async def test_begin_retry_declining_never_bumps_or_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_agent_job_active", lambda: False)
            monkeypatch.setattr(app, "event_handler", _RetryHandler(False))
            epoch = app._record_epoch
            emitter.reset_mock()
            await app._turn_supervisor.retry("more")
            assert _emitted_calls(emitter) == []
            assert app._record_epoch == epoch
            assert app._pending_start_epoch is None

    @pytest.mark.asyncio
    async def test_auto_retry_reports_working_after_the_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_await_interrupt_settled", AsyncMock())
            monkeypatch.setattr(app, "event_handler", _RetryHandler())
            monkeypatch.setattr(app, "_handle_turn", AsyncMock())
            epoch = app._record_epoch
            emitter.reset_mock()
            await app._turn_supervisor.auto_retry_incomplete_stream(0)
            assert _emitted_calls(emitter) == [("working", None)]
            assert app._record_epoch == epoch + 1
            assert app._pending_start_epoch == epoch + 1
            app._turn_supervisor.agent_task = None

    @pytest.mark.asyncio
    async def test_auto_retry_rejection_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_await_interrupt_settled", AsyncMock())
            monkeypatch.setattr(app, "event_handler", _RetryHandler(False))
            epoch = app._record_epoch
            emitter.reset_mock()
            await app._turn_supervisor.auto_retry_incomplete_stream(0)
            assert _emitted_calls(emitter) == []
            assert app._record_epoch == epoch
            assert app._pending_start_epoch is None

    @pytest.mark.asyncio
    async def test_retry_path_startup_cancel_reports_one_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            entered = asyncio.Event()

            async def hanging_remove() -> None:
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(app, "_remove_loading_widget", hanging_remove)
            _install_client_turn(app, monkeypatch, [])
            emitter.reset_mock()
            task = asyncio.create_task(
                app._turn_supervisor.handle_turn("hi", injected=True)
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            # The pre-try await now sits inside the protected block.
            assert _emitted_calls(emitter) == [("idle", None)]

    @pytest.mark.asyncio
    async def test_aborted_prompt_prep_reports_one_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(
                app, "_prepare_prompt_or_abort", AsyncMock(return_value=None)
            )
            calls = _spy_finalize(app, monkeypatch)
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi")
            assert _emitted_calls(emitter) == [("idle", None)]
            assert calls[-1]["outcome"] is TurnOutcome.NO_REPORT


class _FakeLoading:
    parent: object = object()

    def pause_timer(self) -> None:
        return None

    def resume_timer(self) -> None:
        return None

    def begin_action_required(self, title: str) -> None:
        return None

    def end_action_required(self) -> None:
        return None

    def set_status(self, status: str) -> None:
        return None


class _QuestionStub:
    def __init__(self, callback_id: str) -> None:
        self._callback_id = callback_id
        self.other_texts: dict[int, str] = {}
        self.closed = False
        self.disabled = False
        self.statuses: list[str] = []

    def close_for_expiry(self) -> None:
        self.closed = True

    def set_submission_status(self, status: str) -> None:
        self.statuses.append(status)


def _question(text: str) -> UserQuestion:
    return UserQuestion(
        question=text,
        options=[QuestionChoice(label="one"), QuestionChoice(label="two")],
    )


def _callback(
    callback_id: str = "cb-1",
    *,
    origin: str | None = None,
    question: str = "Pick one",
    questions: list[UserQuestion] | None = None,
    title: str = "Input required",
) -> PublicCallbackEntry:
    request = UserQuestionRequest(questions=questions or [_question(question)])
    return PublicCallbackEntry(
        id=f"callback:{callback_id}",
        session_id="session-1",
        turn_id="turn-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        callback_id=callback_id,
        title=title,
        detail=UserInputCallbackDetail(request=request, origin=origin),
        state=OpenCallbackState(),
    )


def _stub_callback_ui(app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_loading_widget", _FakeLoading())
    monkeypatch.setattr(app, "_wait_for_typing_pause", AsyncMock())
    monkeypatch.setattr(app, "_secondary_surface_active", lambda: False)
    monkeypatch.setattr(app, "_switch_to_question_app", AsyncMock())
    monkeypatch.setattr(app, "_switch_to_input_app", AsyncMock())
    monkeypatch.setattr(app, "_show_callback", AsyncMock())
    monkeypatch.setattr(app, "_terminal_notifier", MagicMock())
    monkeypatch.setattr(app, "query_one", lambda *args, **kwargs: MagicMock())


def _emitted_blocked(emitter: MagicMock) -> list[tuple[Any, Any, Any]]:
    return [
        (call_.args[0], call_.kwargs.get("kind"), call_.kwargs.get("msg"))
        for call_ in emitter.emit.call_args_list
    ]


class TestBlockedEmission:
    @pytest.mark.asyncio
    async def test_model_question_reports_blocked_question(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            emitter.reset_mock()
            await app._turn_supervisor.show_callback(
                _callback(origin="model_question", question="Pick one")
            )
            assert _emitted_blocked(emitter) == [("blocked", "question", "Pick one")]

    @pytest.mark.asyncio
    async def test_runtime_approval_keeps_the_command_in_the_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            emitter.reset_mock()
            question = "Allow this command?\n\nrm -rf /tmp/x\n\nAlways allow?"
            await app._turn_supervisor.show_callback(
                _callback(origin="runtime_approval", question=question)
            )
            assert _emitted_blocked(emitter) == [
                ("blocked", "permission", "Allow this command? rm -rf /tmp/x")
            ]
            assert "rm -rf /tmp/x" in _emitted_blocked(emitter)[0][2]

    @pytest.mark.asyncio
    async def test_two_question_permission_uses_questions_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            emitter.reset_mock()
            await app._turn_supervisor.show_callback(
                _callback(
                    origin="runtime_approval",
                    questions=[
                        _question("Allow this one?\n\ncmd-one"),
                        _question("Also this?\n\ncmd-two"),
                    ],
                )
            )
            msg = _emitted_blocked(emitter)[0][2]
            assert "cmd-one" in msg
            assert "cmd-two" not in msg

    @pytest.mark.asyncio
    async def test_unknown_origin_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            emitter.reset_mock()
            await app._turn_supervisor.show_callback(_callback(origin=None))
            assert _emitted_calls(emitter) == []

    @pytest.mark.asyncio
    async def test_deferred_callback_does_not_overwrite_the_visible_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            emitter.reset_mock()
            await app._turn_supervisor.show_callback(
                _callback("cb-a", origin="model_question", question="A?")
            )
            await app._turn_supervisor.show_callback(
                _callback("cb-b", origin="model_question", question="B?")
            )
            assert _emitted_blocked(emitter) == [("blocked", "question", "A?")]
            assert [
                pending.callback_id
                for pending in app._turn_supervisor.pending_callbacks
            ] == ["cb-b"]

    @pytest.mark.asyncio
    async def test_duplicate_callback_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            emitter.reset_mock()
            callback = _callback(origin="model_question")
            await app._turn_supervisor.show_callback(callback)
            await app._turn_supervisor.show_callback(callback)
            assert _emitted_blocked(emitter) == [("blocked", "question", "Pick one")]


class TestCallbackReturnEmissions:
    @pytest.mark.asyncio
    async def test_respond_reports_working_when_a_turn_is_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())
            monkeypatch.setattr(app._turn_supervisor, "agent_job_active", lambda: True)
            app._turn_supervisor.active_callback = _callback()
            emitter.reset_mock()
            await app._turn_supervisor.respond_to_active_callback(
                UserInputCallbackOutput(
                    result=UserQuestionResult(answers=[], cancelled=True)
                )
            )
            assert _emitted_calls(emitter) == [("working", None)]

    @pytest.mark.asyncio
    async def test_respond_reports_idle_without_an_active_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())
            monkeypatch.setattr(app._turn_supervisor, "agent_job_active", lambda: False)
            app._turn_supervisor.active_callback = _callback()
            emitter.reset_mock()
            await app._turn_supervisor.respond_to_active_callback(
                UserInputCallbackOutput(
                    result=UserQuestionResult(answers=[], cancelled=True)
                )
            )
            assert _emitted_calls(emitter) == [("idle", None)]

    @pytest.mark.asyncio
    async def test_showing_the_next_prompt_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())
            monkeypatch.setattr(app._turn_supervisor, "agent_job_active", lambda: True)
            app._turn_supervisor.active_callback = _callback()
            app._turn_supervisor.pending_callbacks.append(_callback("cb-2"))
            emitter.reset_mock()
            await app._turn_supervisor.respond_to_active_callback(
                UserInputCallbackOutput(
                    result=UserQuestionResult(answers=[], cancelled=True)
                )
            )
            assert _emitted_calls(emitter) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("active", "expected"), [(True, "working"), (False, "idle")]
    )
    async def test_question_expired_reports_the_live_record(
        self, active: bool, expected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            monkeypatch.setattr(
                app._turn_supervisor, "agent_job_active", lambda: active
            )
            entry = _callback("cb-1")
            app._turn_supervisor.active_callback = entry
            monkeypatch.setattr(
                app, "query_one", lambda *args, **kwargs: _QuestionStub("cb-1")
            )
            emitter.reset_mock()
            await app._turn_supervisor.question_expired(entry)
            assert _emitted_calls(emitter) == [(expected, None)]

    @pytest.mark.asyncio
    async def test_queued_question_expiry_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            entry = _callback("cb-1")
            app._turn_supervisor.active_callback = None
            app._turn_supervisor.pending_callbacks.append(entry)
            emitter.reset_mock()
            await app._turn_supervisor.question_expired(entry)
            assert _emitted_calls(emitter) == []


class TestLocalQuestionsEmitNothing:
    @pytest.mark.asyncio
    async def test_local_question_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            request = UserQuestionRequest(questions=[_question("Pick")])
            emitter.reset_mock()
            task = asyncio.create_task(
                app._turn_supervisor.request_local_user_input(request)
            )
            await _wait_until(
                lambda: app._turn_supervisor.pending_local_question is not None
            )
            future = app._turn_supervisor.pending_local_question
            assert future is not None
            future.set_result(UserQuestionResult(answers=[], cancelled=True))
            result = await task
            assert result.cancelled is True
            emitter.emit.assert_not_called()


# --- session tracker (WP5a, chartreux/app_server/session.py) --------------


class TestSessionTurnStatusTracker:
    @pytest.mark.asyncio
    async def test_completed_status_is_keyed_by_turn_id(self) -> None:
        backend = FakeBackend([[mock_llm_chunk(content="done")]])
        agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
        session = await create_test_app_server_session(agent_loop)
        try:
            events = [event async for event in session.act("hello")]
            started = next(event for event in events if isinstance(event, TurnStarted))
            assert session.last_turn_status == (
                started.turn.id,
                PublicTurnStatus.COMPLETED,
            )
        finally:
            await session.close()
            await agent_loop.aclose()

    @pytest.mark.asyncio
    async def test_completed_then_interrupted_sequence_tracks_each_turn(self) -> None:
        backend = FakeBackend([[mock_llm_chunk(content="first")]])
        agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
        session = await create_test_app_server_session(agent_loop)
        started = asyncio.Event()

        async def blocking_act(msg: str, **_kwargs: Any) -> Any:
            started.set()
            await asyncio.Event().wait()
            yield UserMessageEvent(content=msg, message_id="active-user")

        try:
            first = [event async for event in session.act("first")]
            first_terminal = next(
                event for event in first if isinstance(event, TurnStarted)
            )
            assert session.last_turn_status == (
                first_terminal.turn.id,
                PublicTurnStatus.COMPLETED,
            )

            agent_loop.act = blocking_act
            seen: list[object] = []

            async def drive() -> None:
                async for event in session.act("second"):
                    seen.append(event)

            task = asyncio.create_task(drive())
            await asyncio.wait_for(started.wait(), timeout=5)
            await session.interrupt()
            await asyncio.wait_for(task, timeout=5)
            second_terminal = next(
                event for event in seen if isinstance(event, TurnStarted)
            )
            assert second_terminal.turn.id != first_terminal.turn.id
            assert session.last_turn_status == (
                second_terminal.turn.id,
                PublicTurnStatus.INTERRUPTED,
            )
        finally:
            await session.close()
            await agent_loop.aclose()

    @pytest.mark.asyncio
    async def test_interrupt_unwind_does_not_inherit_completed(self) -> None:
        backend = FakeBackend([[mock_llm_chunk(content="first")]])
        agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
        session = await create_test_app_server_session(agent_loop)
        started = asyncio.Event()

        async def blocking_act(msg: str, **_kwargs: Any) -> Any:
            started.set()
            await asyncio.Event().wait()
            yield UserMessageEvent(content=msg, message_id="active-user")

        try:
            first = [event async for event in session.act("first")]
            first_terminal = next(
                event for event in first if isinstance(event, TurnStarted)
            )
            agent_loop.act = blocking_act
            seen: list[object] = []

            async def drive() -> None:
                async for event in session.act("second"):
                    seen.append(event)

            task = asyncio.create_task(drive())
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            second_terminal = next(
                event for event in seen if isinstance(event, TurnStarted)
            )
            assert second_terminal.turn.id != first_terminal.turn.id
            status = session.last_turn_status
            assert status is not None
            assert status[1] is PublicTurnStatus.INTERRUPTED
            assert status[0] == second_terminal.turn.id
        finally:
            await session.close()
            await agent_loop.aclose()


# --- ownership settlement + recovery hooks (WP5b) -------------------------
#
# The recovery controller consumes the failed turn's KEYED ownership at
# observation time and freezes it into the armed schedule; the DETACHED manual
# fallback then reports only while that ownership still owns the record. These
# tests drive the real controller, the real supervisor mount, and the real
# emitter over a recording fake driver, so every claim here is a wire report:
# a suppressed `error` writes nothing at all.

_DETACHED_ERROR = "predecessor failure"


def _wire_app() -> tuple[ChartreuxApp, MagicMock]:
    """The real app over a real emitter writing to a recording fake driver."""
    host = _fake_app()
    emitter = ProgramStatusEmitter(host, get_enabled=lambda: True)
    return build_test_chartreux_app(program_status=emitter), host


def _wire_states(host: MagicMock) -> list[str]:
    return [
        call.args[0].split("state=", 1)[1].split(":", 1)[0].removesuffix("\x07")
        for call in host._driver.write.call_args_list
    ]


def _root_error(turn_id: str = "turn-1") -> AppServerTurnError:
    return AppServerTurnError(PublicError(message=_DETACHED_ERROR), turn_id=turn_id)


def _scheduled_response() -> RootRecoveryScheduleResponse:
    return RootRecoveryScheduleResponse(
        decision="scheduled",
        episode_id="episode-1",
        idempotency_key="key-1",
        attempt=1,
        limit=2,
        delay_seconds=30.0,
    )


def _declined_admission() -> RootRecoveryAdmitResponse:
    return RootRecoveryAdmitResponse(decision="declined", attempt=1, limit=2)


async def _blocked_sleep(_delay: float) -> None:
    await asyncio.Event().wait()


async def _drain_tasks(rounds: int = 16) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


def _stub_recovery_ui(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch, *, parked: bool = False
) -> tuple[AsyncMock, asyncio.Event]:
    """Stub the UI collaborators the detached mount touches.

    ``parked`` makes an ERROR mount wait for the returned gate, so a test can
    change ownership while the fallback is mid-mount: the ownership check runs
    after that await, never before it.
    """
    monkeypatch.setattr(app, "event_handler", None)
    monkeypatch.setattr(app, "_refresh_windowing_from_history", AsyncMock())
    monkeypatch.setattr(app, "_remove_loading_widget", AsyncMock())
    monkeypatch.setattr(app, "_ensure_loading_widget", AsyncMock())
    gate = asyncio.Event()
    mount = AsyncMock()

    async def parked_mount(widget: Any) -> None:
        if isinstance(widget, ErrorMessage):
            await gate.wait()

    mount.side_effect = parked_mount if parked else None
    monkeypatch.setattr(app, "_mount_and_scroll", mount)
    return mount, gate


def _error_mounts(mount: AsyncMock) -> int:
    """The error widgets the detached fallback actually mounted."""
    return sum(
        1 for call_ in mount.await_args_list if isinstance(call_.args[0], ErrorMessage)
    )


def _install_recovery_wire(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch
) -> AsyncMock:
    """Mock the recovery wire surface and park the local timer.

    Returns the schedule mock, the only wire call the battery counts.
    """
    schedule = AsyncMock(return_value=_scheduled_response())
    monkeypatch.setattr(app.app_server, "schedule_root_recovery", schedule)
    monkeypatch.setattr(
        app.app_server,
        "admit_root_recovery",
        AsyncMock(return_value=_declined_admission()),
    )
    monkeypatch.setattr(
        app.app_server, "cancel_root_recovery", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(app._root_recovery, "_sleep", _blocked_sleep, raising=False)
    return schedule


def _arm(app: ChartreuxApp, *, turn_id: str = "turn-1", with_token: bool = True) -> Any:
    """Reproduce the supervisor's first-observation capture and the arm start.

    The failed turn's own ``working`` report precedes its failure on the wire,
    exactly as it does in a real turn.
    """
    app._record_turn_owner(turn_id)
    app._program_status.emit("working")
    token: Any = None
    if with_token:
        token = app._ownership_token_for(turn_id, None)
        app._failure_tokens[turn_id] = token
    return token


async def _observe(app: ChartreuxApp, *, turn_id: str = "turn-1") -> bool:
    return await app._root_recovery.observe_terminal(
        _root_error(turn_id), solicited=True
    )


async def _fire(app: ChartreuxApp, *, drain: bool = True) -> None:
    """Fire the armed timer: admission declines and the fallback detaches."""
    await app._root_recovery._on_wake()
    if drain:
        await _drain_tasks()


async def _execute_rewind(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch, *, inplace: bool
) -> list[int]:
    """Run the real rewind hook; return the input generation at the wire call."""
    flow = app._session_flow
    observed_generations: list[int] = []

    async def rewind(_entry_id: str, *, restore_files: bool, inplace: bool) -> Any:
        observed_generations.append(app._input_generation)
        return MagicMock(message="rewound prompt", restore_errors=[])

    monkeypatch.setattr(app.app_server.resources.sessions, "rewind", rewind)
    monkeypatch.setattr(app.app_server.resources, "refresh", AsyncMock())
    monkeypatch.setattr(app._queue, "clear_server_queue", AsyncMock())
    monkeypatch.setattr(app, "_request_agent_transcript_close", AsyncMock())
    monkeypatch.setattr(app, "_switch_to_input_app", AsyncMock())
    monkeypatch.setattr(app, "_reset_message_widgets", AsyncMock())
    monkeypatch.setattr(app, "_resume_history_from_messages", AsyncMock())
    monkeypatch.setattr(flow, "rewind_mode", True)
    monkeypatch.setattr(
        flow, "rewind_highlighted_widget", MagicMock(history_entry_id="entry-1")
    )
    await flow.execute_rewind(restore_files=False, inplace=inplace)
    return observed_generations


async def _run_hook(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch, hook: str
) -> None:
    """Run one record-owner lifecycle hook (each ends with a single ``idle``)."""
    if hook == "clear":
        monkeypatch.setattr(app.app_server, "clear_history", AsyncMock())
        monkeypatch.setattr(app, "_reset_message_widgets", AsyncMock())
        await app._clear_history()
    elif hook == "compact":
        monkeypatch.setattr(app.app_server, "compact", AsyncMock())
        await app._run_compact(CompactMessage())
    elif hook == "context-cleared":
        monkeypatch.setattr(app, "_reset_message_widgets", AsyncMock())
        await app._on_context_cleared()
    elif hook == "session-change":
        monkeypatch.setattr(app._session_flow, "resume_local_session", AsyncMock())
        await app._resume_local_session("abcd1234")
        # The resume flow's own return-to-input point (stubbed above).
        app._mark_session_ready()
    else:
        raise AssertionError(f"unknown hook {hook!r}")


class TestArmConsumption:
    @pytest.mark.asyncio
    async def test_key_present_freezes_the_real_token_into_the_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            schedule = _install_recovery_wire(app, monkeypatch)
            token = _arm(app)
            assert isinstance(token, TurnOwnership)
            assert await _observe(app) is True
            pending = app._root_recovery.pending
            assert pending is not None
            assert pending.ownership is token
            assert pending.record_epoch == token.record_epoch
            # The keyed entry is CONSUMED, so a replay cannot re-freeze it.
            assert app._failure_tokens == {}
            assert await _observe(app) is True
            assert app._failure_tokens == {}
            assert app._root_recovery.pending is pending
            assert schedule.await_count == 1

    @pytest.mark.asyncio
    async def test_key_absent_freezes_the_sentinel_by_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app, with_token=False)
            assert await _observe(app) is True
            pending = app._root_recovery.pending
            assert pending is not None
            assert pending.ownership is app._OWNERSHIP_MISSING
            assert pending.record_epoch is None
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_duplicate_observation_still_consumes_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            schedule = _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            # A replay writes the key again; the duplicate return is AFTER the
            # pop, so the store never keeps an unconsumed entry.
            app._failure_tokens["turn-1"] = app._ownership_token_for("turn-1", None)
            assert await _observe(app) is True
            assert app._failure_tokens == {}
            assert schedule.await_count == 1

    @pytest.mark.asyncio
    async def test_queued_work_return_still_consumes_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            schedule = _install_recovery_wire(app, monkeypatch)
            _arm(app)
            monkeypatch.setattr(
                type(app._queue), "has_server_work", property(lambda _queue: True)
            )
            assert await _observe(app) is False
            assert app._failure_tokens == {}
            assert app._root_recovery.pending is None
            assert schedule.await_count == 0

    @pytest.mark.asyncio
    async def test_non_typed_failure_never_reaches_the_pop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            schedule = _install_recovery_wire(app, monkeypatch)
            app._record_turn_owner("turn-1")
            token = app._ownership_token_for("turn-1", None)
            app._failure_tokens["turn-1"] = token
            observed = await app._root_recovery.observe_terminal(
                RuntimeError("stream closed"), solicited=True
            )
            assert observed is False
            # The guard returns before the pop: the entry is untouched.
            assert app._failure_tokens == {"turn-1": token}
            assert schedule.await_count == 0

    @pytest.mark.asyncio
    async def test_the_pop_precedes_the_session_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)

            def raising(_session: AppServerSession) -> str:
                raise RuntimeError("closing")

            # Scoped: the property must be back before the app's teardown
            # awaits run its workers, which otherwise read a raising session id.
            with monkeypatch.context() as scoped:
                scoped.setattr(AppServerSession, "session_id", property(raising))
                assert await _observe(app) is False
            # Consumed before the key build and the session read.
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_last_write_wins_between_producers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            token_a = app._failure_tokens["turn-1"]
            assert isinstance(token_a, TurnOwnership)
            assert await _observe(app) is True
            first = app._root_recovery.pending
            # A second producer: another terminal turn writes and pops its OWN
            # token, and its arm replaces the earlier schedule.
            app._advance_record_epoch()
            app._record_turn_owner("turn-2")
            token_b = app._ownership_token_for("turn-2", None)
            assert isinstance(token_b, TurnOwnership)
            app._failure_tokens["turn-2"] = token_b
            assert await _observe(app, turn_id="turn-2") is True
            second = app._root_recovery.pending
            assert first is not None
            assert first.ownership is token_a
            assert second is not None
            assert second.ownership is token_b
            assert second.record_epoch == token_b.record_epoch
            assert second.record_epoch != token_a.record_epoch
            assert app._failure_tokens == {}


class TestSentinelArm:
    @pytest.mark.asyncio
    async def test_withheld_ownership_suppresses_the_detached_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app, with_token=False)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            await _fire(app)
            # The transcript still renders; only the OSC report is withheld.
            assert mount.await_count == 1
            assert _wire_states(host) == []

    @pytest.mark.asyncio
    async def test_withheld_ownership_never_reports_a_cancellation_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app, with_token=False)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            app._invalidate_user_input()
            await _drain_tasks()
            # record_epoch is None, so the pinned equality is False.
            assert _wire_states(host) == []
            assert app._root_recovery.pending is None


class TestDetachedFallbackBattery:
    @pytest.mark.asyncio
    async def test_live_successor_owns_the_record_so_no_error_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            # The successor started and is live when the fallback fires.
            app._record_turn_owner("turn-2")
            app._program_status.emit("working")
            host._driver.write.reset_mock()
            await _fire(app)
            assert mount.await_count == 1
            assert _wire_states(host) == []

    @pytest.mark.asyncio
    async def test_finished_successor_keeps_its_durable_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            # The successor started AND finished before the fallback fires.
            app._record_turn_owner("turn-2")
            await app._turn_supervisor.finalize_turn_ui(
                outcome=TurnOutcome.COMPLETED, turn_id="turn-2"
            )
            host._driver.write.reset_mock()
            await _fire(app)
            assert mount.await_count == 1
            assert _wire_states(host) == []

    @pytest.mark.asyncio
    async def test_declined_admission_surfaces_the_error_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            await _fire(app)
            assert mount.await_count == 1
            assert _wire_states(host) == ["error"]
            assert _decoded_msg(host._driver.write.call_args.args[0]) == _DETACHED_ERROR
            # Normal consumption leaves no keyed entry behind.
            assert app._failure_tokens == {}

    @pytest.mark.asyncio
    async def test_draft_keystroke_after_the_decline_still_reports_the_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            await _fire(app, drain=False)
            # One draft keystroke, no successor: the fallback is already
            # detached, and the draft only moves the input generation.
            app._invalidate_user_input()
            await _drain_tasks()
            assert mount.await_count == 1
            assert _wire_states(host) == ["error"]
            assert _wire_states(host)[-1] != "working"

    @pytest.mark.asyncio
    async def test_contended_finalizer_mutex_still_reports_the_error_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            async with app._turn_ui_lock():
                await _fire(app)
            assert mount.await_count == 1
            assert _wire_states(host) == ["error"]
            await _drain_tasks()
            assert _wire_states(host) == ["error"]

    @pytest.mark.asyncio
    async def test_successor_starting_during_the_schedule_await_mismatches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            entered = asyncio.Event()
            release = asyncio.Event()

            async def schedule(*_args: Any, **_kwargs: Any) -> Any:
                entered.set()
                await release.wait()
                return _scheduled_response()

            monkeypatch.setattr(app.app_server, "schedule_root_recovery", schedule)
            token = _arm(app)
            assert isinstance(token, TurnOwnership)
            frozen_epoch = token.record_epoch
            observation = asyncio.create_task(_observe(app))
            await asyncio.wait_for(entered.wait(), timeout=5)
            # The successor starts while the schedule request is still open.
            app._record_turn_owner("turn-2")
            app._program_status.emit("working")
            release.set()
            assert await observation is True
            pending = app._root_recovery.pending
            assert pending is not None
            assert pending.record_epoch == frozen_epoch
            assert pending.record_epoch != app._record_epoch
            host._driver.write.reset_mock()
            await _fire(app)
            assert mount.await_count == 1
            assert _wire_states(host) == []

    @pytest.mark.asyncio
    async def test_rejected_manual_retry_keeps_the_frozen_ownership(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            frozen_epoch = app._record_epoch
            host._driver.write.reset_mock()
            await _fire(app, drain=False)
            monkeypatch.setattr(app, "_agent_job_active", lambda: True)
            await app._turn_supervisor.retry("more")
            await _drain_tasks()
            # A rejected retry never bumps, so the failure still speaks.
            assert app._record_epoch == frozen_epoch
            assert mount.await_count == 1
            assert _wire_states(host) == ["error"]

    @pytest.mark.asyncio
    async def test_rejected_enqueue_keeps_the_frozen_ownership(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, gate = _stub_recovery_ui(app, monkeypatch, parked=True)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            frozen_epoch = app._record_epoch
            host._driver.write.reset_mock()
            await _fire(app, drain=False)
            await _drain_tasks()
            # The fallback is detached and parked inside its mount await.
            assert mount.await_count == 1
            assert _wire_states(host) == []

            async def reject(_message: str, **_kwargs: Any) -> bool:
                # The durable error lands while the prompt is still enqueuing;
                # only then is the enqueue refused.
                gate.set()
                await _wait_until(lambda: _wire_states(host) == ["error"])
                return False

            monkeypatch.setattr(app, "_enqueue_prompt_with_resources", reject)
            await app._handle_user_message("hello")
            await _drain_tasks()
            # The rejection cleanup must not overwrite the durable error.
            assert _wire_states(host) == ["error"]
            assert _decoded_msg(host._driver.write.call_args.args[0]) == _DETACHED_ERROR
            assert app._record_epoch == frozen_epoch
            assert mount.await_count == 1

    @pytest.mark.asyncio
    async def test_raising_enqueue_keeps_the_frozen_ownership(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, gate = _stub_recovery_ui(app, monkeypatch, parked=True)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            frozen_epoch = app._record_epoch
            host._driver.write.reset_mock()
            await _fire(app, drain=False)
            await _drain_tasks()
            assert mount.await_count == 1
            assert _wire_states(host) == []

            async def explode(_message: str, **_kwargs: Any) -> bool:
                gate.set()
                await _wait_until(lambda: _wire_states(host) == ["error"])
                raise RuntimeError("prompt prep failed")

            monkeypatch.setattr(app, "_enqueue_prompt_with_resources", explode)
            with pytest.raises(RuntimeError):
                await app._handle_user_message("hello")
            await _drain_tasks()
            # The re-raising cleanup must not overwrite the durable error.
            assert _wire_states(host) == ["error"]
            assert app._record_epoch == frozen_epoch
            assert mount.await_count == 1

    @pytest.mark.asyncio
    async def test_rejected_enqueue_without_an_intervening_report_still_reports_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            # A live record so the rejection's idle is a real transition.
            app._program_status.emit("working")
            assert _wire_states(host) == ["idle", "working"]

            async def reject(_message: str, **_kwargs: Any) -> bool:
                return False

            monkeypatch.setattr(app, "_enqueue_prompt_with_resources", reject)
            await app._handle_user_message("hello")
            # No report intervened, so the prompt really is back: one idle.
            assert _wire_states(host) == ["idle", "working", "idle"]

    @pytest.mark.asyncio
    async def test_independent_live_successor_gets_no_cancellation_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            # An INDEPENDENT successor owns the record now; its own `working`
            # is byte-identical to the predecessor's and collapses on dedup.
            app._record_turn_owner("turn-2")
            app._program_status.emit("working")
            app._invalidate_user_input()
            await _drain_tasks()
            assert _wire_states(host) == []
            assert app._root_recovery.pending is None

    @pytest.mark.asyncio
    async def test_independent_finished_successor_keeps_its_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            app._record_turn_owner("turn-2")
            await app._turn_supervisor.finalize_turn_ui(
                outcome=TurnOutcome.COMPLETED, turn_id="turn-2"
            )
            before = _wire_states(host)
            assert before[-1] == "done"
            app._invalidate_user_input()
            await _drain_tasks()
            assert _wire_states(host) == before
            assert app._root_recovery.pending is None

    @pytest.mark.asyncio
    async def test_the_keyed_store_is_empty_after_normal_consumption(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, _host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            app._record_turn_owner("turn-2")
            app._failure_tokens["turn-2"] = app._ownership_token_for("turn-2", None)
            assert await _observe(app, turn_id="turn-2") is True
            assert app._failure_tokens == {}
            await _fire(app)
            assert app._failure_tokens == {}


class TestRecoveryHooks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("inplace", [True, False])
    async def test_rewind_settles_the_owner_and_reports_one_idle(
        self, inplace: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            frozen_epoch = app._record_epoch
            generation = app._input_generation
            host._driver.write.reset_mock()
            observed = await _execute_rewind(app, monkeypatch, inplace=inplace)
            # The invalidation lands BEFORE the rewind request goes out.
            assert observed == [generation + 1]
            # An in-place rewind keeps the session id, so the epoch bump is
            # what retires the old owner.
            assert app._record_epoch == frozen_epoch + 1
            assert app._root_recovery.pending is None
            await _fire(app)
            assert _wire_states(host) == ["idle"]
            assert _error_mounts(mount) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "hook", ["clear", "compact", "context-cleared", "session-change"]
    )
    async def test_owner_change_between_schedule_and_fire(
        self, hook: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, _gate = _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            frozen_epoch = app._record_epoch
            host._driver.write.reset_mock()
            await _run_hook(app, monkeypatch, hook)
            assert app._record_epoch == frozen_epoch + 1
            await _fire(app)
            # Exactly one idle from the hook; never a predecessor error.
            assert _wire_states(host) == ["idle"]
            # /compact and the context handoff do not cancel the schedule, so
            # their fallback fires and is suppressed after the mount await.
            expected_errors = 1 if hook in {"compact", "context-cleared"} else 0
            assert _error_mounts(mount) == expected_errors

    @pytest.mark.asyncio
    @pytest.mark.parametrize("hook", ["clear", "compact"])
    async def test_already_detached_fallback_is_settled_by_the_hook(
        self, hook: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, gate = _stub_recovery_ui(app, monkeypatch, parked=True)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            await _fire(app, drain=False)
            # The fallback is detached and parked mid-mount.
            await _drain_tasks()
            assert mount.await_count == 1
            assert _wire_states(host) == []
            await _run_hook(app, monkeypatch, hook)
            gate.set()
            await _drain_tasks()
            assert _wire_states(host) == ["idle"]

    @pytest.mark.asyncio
    async def test_interrupt_settlement_wins_over_the_detached_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            mount, gate = _stub_recovery_ui(app, monkeypatch, parked=True)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            await _fire(app, drain=False)
            await _drain_tasks()
            assert mount.await_count == 1
            monkeypatch.setattr(app, "_agent_job_active", lambda: True)
            task = asyncio.create_task(asyncio.Event().wait())
            app._turn_supervisor.agent_task = task
            try:
                # The predecessor's finalizer is still contending for the
                # mutex; the ACCEPTED interruption retires ownership anyway.
                async with app._turn_ui_lock():
                    await app._turn_supervisor.interrupt_turn()
            finally:
                with suppress(asyncio.CancelledError):
                    await task
                app._turn_supervisor.agent_task = None
            gate.set()
            await _drain_tasks()
            assert _wire_states(host) == ["idle"]
            assert _error_mounts(mount) == 1

    @pytest.mark.asyncio
    async def test_draft_keystroke_cancels_the_schedule_with_one_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            app._invalidate_user_input()
            await _drain_tasks()
            # The record returns to the prompt instead of staying `working`.
            assert _wire_states(host) == ["idle"]
            assert app._root_recovery.pending is None

    @pytest.mark.asyncio
    async def test_typing_during_a_live_working_turn_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            app._record_turn_owner("turn-1")
            app._program_status.emit("working")
            host._driver.write.reset_mock()
            app._invalidate_user_input()
            await _drain_tasks()
            assert _wire_states(host) == []
            assert app._root_recovery.pending is None

    @pytest.mark.asyncio
    async def test_shutdown_path_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            app.on_unmount()
            await _drain_tasks()
            assert _wire_states(host) == []
            assert app._root_recovery.pending is None

    @pytest.mark.asyncio
    async def test_manual_retry_pair_is_idle_then_working(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_recovery_ui(app, monkeypatch)
            _install_recovery_wire(app, monkeypatch)
            _arm(app)
            assert await _observe(app) is True
            host._driver.write.reset_mock()
            app._invalidate_user_input()
            # An ACCEPTED manual retry starts its own turn.
            monkeypatch.setattr(app, "_agent_job_active", lambda: False)
            monkeypatch.setattr(app, "event_handler", _RetryHandler())
            monkeypatch.setattr(app, "_handle_turn", AsyncMock())
            await app._turn_supervisor.retry("more")
            retry_task = app._turn_supervisor.agent_task
            assert retry_task is not None
            await retry_task
            app._turn_supervisor.agent_task = None
            await _drain_tasks()
            assert _wire_states(host) == ["idle", "working"]


# --- WP7 review-deep fixes ------------------------------------------------
#
# The wire grammar is parsed here from the design's own shape
# (`ESC ] 7501 ; pair (":" pair)* BEL`), never from the encoder's pair
# construction, so a wrong separator cannot pass by agreeing with itself.


def _parse_osc7501(report: str) -> tuple[dict[str, str], list[str]]:
    prefix = "\x1b]7501;"
    assert report.startswith(prefix), report
    assert report.endswith("\x07"), report
    body = report[len(prefix) : -1]
    pairs: dict[str, str] = {}
    order: list[str] = []
    for part in body.split(":"):
        match = re.fullmatch(r"([a-z]+)=([A-Za-z0-9_.,+/=-]*)", part)
        assert match is not None, f"not a key=value pair: {part!r}"
        key, value = match.groups()
        assert key in {"state", "kind", "app", "msg"}, key
        assert key not in pairs, key
        order.append(key)
        pairs[key] = value
    return pairs, order


class TestWireGrammar:
    def test_conforming_parse_yields_the_expected_pairs(self) -> None:
        report = encode_report("blocked", kind="permission", msg="Allow?")
        pairs, order = _parse_osc7501(report)
        assert order == ["state", "kind", "app", "msg"]
        assert pairs["state"] == "blocked"
        assert pairs["kind"] == "permission"
        assert pairs["app"] == "chartreux"
        assert base64.b64decode(pairs["msg"]).decode("utf-8") == "Allow?"

    def test_state_only_report_parses_as_state_then_app(self) -> None:
        pairs, order = _parse_osc7501(encode_report("working"))
        assert order == ["state", "app"]
        assert pairs == {"state": "working", "app": "chartreux"}

    def test_no_semicolon_survives_inside_the_body(self) -> None:
        report = encode_report("blocked", kind="question", msg="Pick one")
        body = report[len("\x1b]7501;") : -1]
        assert ";" not in body
        assert body.count(":") == 3

    def test_the_old_semicolon_join_is_rejected(self) -> None:
        old = "\x1b]7501;state=blocked;kind=permission;app=chartreux\x07"
        with pytest.raises(AssertionError):
            _parse_osc7501(old)


def _saturating_scan(target: str) -> tuple[Callable[[str], list[Any]], list[str]]:
    """Count every scanned input; saturate the accounting for *target* only.

    The saturated list holds well-formed ``(start, end, body)`` candidates. The
    redactor's own decode pass consumes whatever this stub returns, so an
    ill-formed placeholder would raise inside ``redact`` — a raise
    ``prepare_message`` swallows into the same ``None`` these tests assert,
    leaving the accounting check untested.
    """
    real = sr._encoded_candidates
    seen: list[str] = []

    def scan(text: str) -> list[Any]:
        seen.append(text)
        if text == target:
            return [(0, 8, "abcdefgh")] * sr._MAX_ENCODED_CANDIDATES
        return real(text)

    return scan, seen


def _accounting_disabled_for(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    """Disable the accounting check for *target* only (mutation control)."""
    real = program_status._scan_saturated

    def scan(redaction: Any, text: str) -> bool:
        if text == target:
            return False
        return real(redaction, text)

    monkeypatch.setattr(program_status, "_scan_saturated", scan)


class TestCandidateAccountingPasses:
    def test_post_sanitize_saturation_omits_the_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = _FIXED_SECRET
        carrier = secret.encode().hex()
        # Long enough that the post-truncation form differs from this one, so
        # the only accounting check this input can trip is the post-sanitize
        # one.
        filler = "c" * 2100
        raw = "carrier " + carrier[:10] + "\u200b" + carrier[10:] + " " + filler
        # The carrier is whole only once the strip removes the zero-width space.
        sanitized = "carrier " + carrier + " " + filler
        with _registered(secret):
            # Non-vacuity: pass 1 sees the split value and cannot saturate.
            assert sr.redact(raw) == raw
            assert len(sr._encoded_candidates(raw)) < sr._MAX_ENCODED_CANDIDATES
            scan, seen = _saturating_scan(sanitized)
            monkeypatch.setattr(sr, "_encoded_candidates", scan)
            assert prepare_message(raw) is None
            # Mutation control: with this one check disabled the same input
            # survives, so the None above is that check doing its job.
            _accounting_disabled_for(monkeypatch, sanitized)
            assert prepare_message(raw) is not None
        # Only the post-sanitize form saturates, so the None above is that
        # check doing its job; the raw form was scanned first and did not.
        assert seen.index(sanitized) > seen.index(raw)

    def test_post_truncation_saturation_omits_the_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = "a" * 2048 + "b" * 10
        truncated = "a" * 2048
        with _registered(_FIXED_SECRET):
            scan, seen = _saturating_scan(truncated)
            monkeypatch.setattr(sr, "_encoded_candidates", scan)
            assert prepare_message(raw) is None
            # Mutation control: with this one check disabled the same input
            # survives, so the None above is that check doing its job.
            _accounting_disabled_for(monkeypatch, truncated)
            assert prepare_message(raw) is not None
        # Only the 2048-byte form saturates, so the None above is the
        # post-truncation check doing its job; the longer input was scanned
        # first and did not.
        assert seen.index(truncated) > seen.index(raw)


class TestCallbackReturnGeneration:
    @pytest.mark.asyncio
    async def test_turn_completing_during_the_switch_keeps_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())
            monkeypatch.setattr(app._turn_supervisor, "agent_job_active", lambda: False)
            app._turn_supervisor.active_callback = _callback()

            async def switch_with_completion() -> None:
                # The turn finalizes while the switch is in flight.
                app._program_status.emit("done")

            monkeypatch.setattr(app, "_switch_to_input_app", switch_with_completion)
            host._driver.write.reset_mock()
            await app._turn_supervisor.respond_to_active_callback(
                UserInputCallbackOutput(
                    result=UserQuestionResult(answers=[], cancelled=True)
                )
            )
            assert _wire_states(host) == ["done"]

    @pytest.mark.asyncio
    async def test_plain_return_still_reports_the_live_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())
            monkeypatch.setattr(app._turn_supervisor, "agent_job_active", lambda: True)
            app._turn_supervisor.active_callback = _callback()
            host._driver.write.reset_mock()
            await app._turn_supervisor.respond_to_active_callback(
                UserInputCallbackOutput(
                    result=UserQuestionResult(answers=[], cancelled=True)
                )
            )
            assert _wire_states(host) == ["working"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("durable_state", "expected"), [("done", ["done"]), ("error", ["error"])]
    )
    async def test_turn_finishing_during_the_response_keeps_the_durable_outcome(
        self, durable_state: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)

            async def respond_while_finishing(*args: Any, **kwargs: Any) -> None:
                # The turn finalizes while the answer is being delivered: the
                # return emission must not overwrite it with idle/working.
                app._program_status.emit(durable_state, msg="turn failed")

            monkeypatch.setattr(
                app.app_server, "respond_to_callback", respond_while_finishing
            )
            monkeypatch.setattr(app._turn_supervisor, "agent_job_active", lambda: False)
            app._turn_supervisor.active_callback = _callback()
            host._driver.write.reset_mock()
            await app._turn_supervisor.respond_to_active_callback(
                UserInputCallbackOutput(
                    result=UserQuestionResult(answers=[], cancelled=True)
                )
            )
            assert _wire_states(host) == expected

    @pytest.mark.asyncio
    async def test_expiry_completing_during_the_switch_keeps_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _stub_callback_ui(app, monkeypatch)
            monkeypatch.setattr(app._turn_supervisor, "agent_job_active", lambda: True)
            entry = _callback("cb-1")
            app._turn_supervisor.active_callback = entry
            monkeypatch.setattr(
                app, "query_one", lambda *args, **kwargs: _QuestionStub("cb-1")
            )

            async def switch_with_completion() -> None:
                app._program_status.emit("done")

            monkeypatch.setattr(app, "_switch_to_input_app", switch_with_completion)
            host._driver.write.reset_mock()
            await app._turn_supervisor.question_expired(entry)
            assert _wire_states(host) == ["done"]


class TestPostBindingHandlerFailure:
    @pytest.mark.asyncio
    async def test_binding_then_raising_still_reports_one_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            _install_client_turn(app, monkeypatch, [_turn_started(app, "turn-1")])

            async def binding_then_raising(event: Any) -> None:
                if isinstance(event, TurnStarted):
                    # The handler binds synchronously, then the awaited
                    # presentation work fails.
                    app._record_turn_owner(event.turn.id)
                    raise RuntimeError("presentation failed")

            monkeypatch.setattr(app, "_handle_turn_event", binding_then_raising)
            emitter.reset_mock()
            await app._turn_supervisor.handle_turn("hi", injected=True)
            calls = _emitted_calls(emitter)
            assert [state for state, _ in calls] == ["error"]
            assert calls[0][1]
            assert app._failure_tokens == {}


class TestUnsolicitedStartOwnership:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "instruction", ["interrupt_settlement", "interrupt_requested", "ownership_bump"]
    )
    async def test_interrupt_during_the_pre_working_awaits_keeps_idle(
        self, instruction: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            bound_epoch = app._record_turn_owner("turn-1")

            async def remove_loading() -> None:
                if instruction == "interrupt_settlement":
                    # What interrupt_turn settles synchronously.
                    app._advance_record_epoch()
                    app._program_status.emit("idle")
                elif instruction == "ownership_bump":
                    # A non-interrupt bump (/clear, /compact, context cleared):
                    # the same settlement without any interrupt involved.
                    app._advance_record_epoch()
                    app._program_status.emit("idle")
                else:
                    app._turn_supervisor.interrupt_requested = True

            monkeypatch.setattr(app, "_remove_loading_widget", remove_loading)
            emitter.reset_mock()
            await app._turn_supervisor._begin_unsolicited_turn(bound_epoch=bound_epoch)
            if instruction == "interrupt_requested":
                assert _emitted_calls(emitter) == []
            else:
                assert _emitted_calls(emitter) == [("idle", None)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bump", ["ownership_bump", "interrupt"])
    @pytest.mark.parametrize("queue_item_id", [None, "queue-1"])
    async def test_ownership_bump_during_the_queue_await_keeps_the_settlement(
        self, bump: str, queue_item_id: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, emitter = _app_with_mock_emitter()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_handle_turn_event", AsyncMock())
            surfaced = asyncio.Event()

            async def queue_turn_started(item_id: str | None) -> None:
                if bump == "ownership_bump":
                    # /clear, /compact, or a context reset settles idle here,
                    # after the listener's synchronous bind.
                    app._advance_record_epoch()
                else:
                    app._turn_supervisor.interrupt_requested = True
                app._program_status.emit("idle")
                await asyncio.sleep(0)

            monkeypatch.setattr(app._queue, "turn_started", queue_turn_started)

            async def events() -> Any:
                yield _turn_started(app, "turn-1")
                surfaced.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(app.app_server, "events", events)
            emitter.reset_mock()
            listener = asyncio.create_task(app._listen_app_server_events())
            try:
                await asyncio.wait_for(surfaced.wait(), timeout=5)
            finally:
                listener.cancel()
                with suppress(asyncio.CancelledError):
                    await listener
            # The settlement survives: only its own `idle`, never `working`.
            assert _emitted_calls(emitter) == [("idle", None)]

    @pytest.mark.asyncio
    async def test_retired_turn_is_not_reauthorized_by_the_duplicate_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The listener binds at TurnStarted, awaits the queue, then handles the
        SAME event again (a second bind). An interrupt settling inside that
        await retires the turn; the second bind must not resurrect it, and its
        stale failure must not speak over the settlement.
        """
        app, host = _wire_app()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_agent_job_active", lambda: True)
            parked = asyncio.Event()
            released = asyncio.Event()

            async def queue_turn_started(item_id: str | None) -> None:
                parked.set()
                await released.wait()

            monkeypatch.setattr(app._queue, "turn_started", queue_turn_started)
            handled = asyncio.Event()

            async def events() -> Any:
                yield _turn_started(app, "retired-turn")
                yield _turn_completed(
                    app,
                    "retired-turn",
                    PublicTurnStatus.FAILED,
                    error=PublicError(message=_DETACHED_ERROR),
                )
                handled.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(app.app_server, "events", events)
            # The record already reported the turn's `working` (the startup
            # idle dedups against the settlement otherwise).
            app._program_status.emit("working")
            assert _wire_states(host) == ["idle", "working"]
            listener = asyncio.create_task(app._listen_app_server_events())
            try:
                await asyncio.wait_for(parked.wait(), timeout=5)
                # The real interrupt settlement: retire, bump, emit idle.
                task = asyncio.create_task(asyncio.Event().wait())
                app._turn_supervisor.agent_task = task
                try:
                    await app._turn_supervisor.interrupt_turn()
                finally:
                    with suppress(asyncio.CancelledError):
                        await task
                    app._turn_supervisor.agent_task = None
                assert app.turn_epoch_map == {}
                assert app._active_turn_id is None
                released.set()
                await asyncio.wait_for(handled.wait(), timeout=5)
                await _drain_tasks()
            finally:
                listener.cancel()
                with suppress(asyncio.CancelledError):
                    await listener
            # The duplicate bind neither re-bound the turn nor counted a start.
            assert app.turn_epoch_map == {}
            # Its typed failure resolves to the sentinel, so the wire keeps the
            # interrupt's settlement and nothing else.
            assert _wire_states(host) == ["idle", "working", "idle"]
            assert app._active_turn_id is None
            assert "retired-turn" in app._retired_turn_ids
            assert app._turn_started_count == 1

    @pytest.mark.asyncio
    async def test_accepted_successor_keeps_the_retired_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An accepted successor between retirement and the duplicate bind must
        not clear the retired set.

        The old listener's second ``_record_turn_owner`` lands after its queue
        await. If the successor's accepted start cleared ``_retired_turn_ids``,
        that bind would re-authorize the retired turn with a fresh epoch and its
        stale failure would overwrite the successor's durable ``done``.
        """
        app, host = _wire_app()
        async with app.run_test():
            _stub_supervisor_ui(app, monkeypatch)
            monkeypatch.setattr(app, "_agent_job_active", lambda: True)
            parked = asyncio.Event()
            released = asyncio.Event()

            async def queue_turn_started(item_id: str | None) -> None:
                parked.set()
                await released.wait()

            monkeypatch.setattr(app._queue, "turn_started", queue_turn_started)
            handled = asyncio.Event()

            async def events() -> Any:
                yield _turn_started(app, "retired-turn")
                yield _turn_completed(
                    app,
                    "retired-turn",
                    PublicTurnStatus.FAILED,
                    error=PublicError(message=_DETACHED_ERROR),
                )
                handled.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(app.app_server, "events", events)
            app._program_status.emit("working")
            assert _wire_states(host) == ["idle", "working"]
            listener = asyncio.create_task(app._listen_app_server_events())
            try:
                await asyncio.wait_for(parked.wait(), timeout=5)
                # The real interrupt settlement: retire, bump, emit idle.
                task = asyncio.create_task(asyncio.Event().wait())
                app._turn_supervisor.agent_task = task
                try:
                    await app._turn_supervisor.interrupt_turn()
                finally:
                    with suppress(asyncio.CancelledError):
                        await task
                    app._turn_supervisor.agent_task = None
                assert "retired-turn" in app._retired_turn_ids
                # The successor's accepted start arrives while the old listener
                # still sits in its queue await, then the successor owns the
                # record and finishes durably.
                _accept_at_the_start_site(app, accepted=True)
                app._program_status.emit("working")
                app._record_turn_owner("successor-turn")
                await app._turn_supervisor.finalize_turn_ui(
                    outcome=TurnOutcome.COMPLETED, turn_id="successor-turn"
                )
                assert app.turn_epoch_map == {}
                released.set()
                await asyncio.wait_for(handled.wait(), timeout=5)
                await _drain_tasks()
            finally:
                listener.cancel()
                with suppress(asyncio.CancelledError):
                    await listener
            # The retired turn's stale failure never overwrote the successor's
            # durable `done`.
            assert _wire_states(host) == ["idle", "working", "idle", "working", "done"]
            # The duplicate bind neither re-bound the retired turn ...
            assert app.turn_epoch_map == {}
            assert app._active_turn_id == "successor-turn"
            assert "retired-turn" in app._retired_turn_ids
            assert app._turn_started_count == 2


# --- exit clear (E3) -------------------------------------------------------
#
# The clear is the literal first statement of ``_shutdown``. These nodes observe
# it on the wire ``_wire_app``'s real emitter writes to; that recording host's
# guards are static, so the position assertion is relative to the session's
# reports, never an "absolute last write" claim.


class TestExitClear:
    @pytest.mark.asyncio
    async def test_exit_clear_follows_the_last_report_and_is_byte_exact(self) -> None:
        app, host = _wire_app()
        async with app.run_test():
            # Startup idle already spoke; the explicit report makes the position
            # pin independent of that readiness path.
            app._program_status.emit("working")
        writes = [call.args[0] for call in host._driver.write.call_args_list]
        assert writes.count(encode_clear()) == 1
        clear_index = writes.index(encode_clear())
        assert writes[clear_index] == encode_clear()
        # Positional: the clear follows the last report the session emitted. A
        # teardown emit can still land behind it on this static-guard host.
        assert clear_index > writes.index(encode_report("working"))
        pre_clear_states = _wire_states(host)[:clear_index]
        assert pre_clear_states
        assert "clear" not in pre_clear_states

    @pytest.mark.asyncio
    async def test_exit_clear_precedes_a_raising_event_listener_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        landed: list[list[str]] = []

        async def boom() -> None:
            landed.append(_wire_states(host))
            raise RuntimeError("listener stop boom")

        monkeypatch.setattr(app, "_stop_app_server_event_listener", boom)
        with pytest.raises(RuntimeError, match="listener stop boom"):
            async with app.run_test():
                app._program_status.emit("working")
        # As observed AT the raise: the clear was already on the wire.
        assert landed
        assert landed[0][-1] == "clear"
        assert landed[0].count("clear") == 1
        # The raise aborted run_test's teardown, so the message loop is still
        # live; stop it explicitly.
        app.exit()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_exit_clear_precedes_a_raising_transcript_reset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        landed: list[list[str]] = []

        def boom_reset() -> None:
            landed.append(_wire_states(host))
            raise RuntimeError("transcript reset boom")

        with pytest.raises(RuntimeError, match="transcript reset boom"):
            async with app.run_test():
                app._program_status.emit("working")
                monkeypatch.setattr(app._transcript, "reset", boom_reset)
        assert landed
        assert landed[0][-1] == "clear"
        assert landed[0].count("clear") == 1
        app.exit()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_force_quit_path_clears_the_record(self) -> None:
        app, host = _wire_app()
        async with app.run_test():
            app._program_status.emit("working")
            app._force_quit()
            force_quit_task = app._force_quit_task
            assert force_quit_task is not None
            await asyncio.wait_for(force_quit_task, timeout=5)
        states = _wire_states(host)
        assert states.count("clear") == 1
        assert states.index("clear") > states.index("working")
