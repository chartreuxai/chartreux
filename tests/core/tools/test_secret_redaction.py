from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.core.tools import secret_redaction as sr
from chartreux.core.tools.mcp.tools import build_stdio_params
from chartreux.core.utils.shell import _shell_environment

FAKE_NAME = "CHARTREUX_TEST_CREDENTIAL_KEY"
FAKE_VALUE = "sk-test-0123456789abcdefghijklmnop"
FAKE_B64 = "c2stdGVzdC0wMTIzNDU2Nzg5YWJjZGVmZ2hpamtsbW5vcA=="


@pytest.mark.parametrize("fallback_fails", [False, True])
def test_persisted_result_failure_logs_never_include_exception_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fallback_fails: bool,
) -> None:
    from chartreux.core.llm_models import PersistedToolResult

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(f"Invalid credential: {FAKE_VALUE}")

    result = PersistedToolResult(output={"text": FAKE_VALUE}, duration=0)
    monkeypatch.setattr(sr, "redact_model", fail)
    if fallback_fails:
        monkeypatch.setattr(sr, "redact_json_value", fail)
    with (
        sr.bind_policy(
            sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, FAKE_VALUE),))
        ),
        caplog.at_level("DEBUG", logger=sr.logger.name),
    ):
        cleaned = sr.redact_persisted_result(result)
    assert FAKE_VALUE not in str(cleaned.output)
    assert "reconstruction failed" in caplog.text
    assert ("fallback failed" in caplog.text) is fallback_fails
    assert FAKE_VALUE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.fixture(autouse=True)
def _clean_module_state() -> Any:
    sr.set_env_passthrough(())
    sr.set_mcp_static_auth_env_names(())
    sr.reset_cache()
    yield
    sr.set_env_passthrough(())
    sr.set_mcp_static_auth_env_names(())
    sr.reset_cache()


@pytest.fixture
def fake_credential(monkeypatch: pytest.MonkeyPatch) -> dict[str, str | None]:
    """One known credential name with a loaded value and a stored .env value."""
    stored = "sk-stored-9876543210zyxwvutsrqp"
    monkeypatch.setenv(FAKE_NAME, FAKE_VALUE)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: stored})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.reset_cache()
    return {FAKE_NAME: stored}


def test_shell_feedback_redacts_registered_secret_before_preview() -> None:
    from chartreux.core.tools.builtins._shell_diagnostics import (
        Diagnostic,
        render_diagnostic,
    )

    secret = "synthetic-diagnostic-[secret]\n" + "x" * 300
    with sr.bind_policy(sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, secret),))):
        feedback = render_diagnostic(Diagnostic("denylist", secret, "printf " + secret))
    assert "synthetic-diagnostic" not in feedback
    assert "REDACTED" in feedback
    assert "\n" not in feedback


@pytest.mark.parametrize("secret", [1234567890123456, 1234567890.125])
def test_registered_numeric_json_secrets_are_redacted_without_mutation(
    secret: int | float,
) -> None:
    payload = {"secret": secret, "nested": [secret, 42, 1.5, True, None]}
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, str(secret)),))
    ):
        cleaned = sr.redact_json_value(payload)
    assert cleaned == {
        "secret": sr.REDACTED_PLACEHOLDER,
        "nested": [sr.REDACTED_PLACEHOLDER, 42, 1.5, True, None],
    }
    assert payload == {"secret": secret, "nested": [secret, 42, 1.5, True, None]}


@pytest.mark.parametrize(
    "number", [1234567890123456.0, 91234567890123456, 1.234567890123456e20]
)
def test_numeric_substring_and_float_forms(number: int | float) -> None:
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, "1234567890123456"),))
    ):
        assert sr.redact_json_value({"nested": [number, True, 42]}) == {
            "nested": [sr.REDACTED_PLACEHOLDER, True, 42]
        }


def test_presentation_redacts_values_before_repr() -> None:
    from pydantic import BaseModel

    from chartreux.core.events import ToolCallEvent
    from chartreux.core.tools.builtins.bash import Bash
    from chartreux.core.tools.ui import ToolUIDataAdapter

    class Args(BaseModel):
        text: str

    secret = "synthetic-back\\slash-secret"
    event = ToolCallEvent(
        tool_name="fixture",
        tool_class=Bash,
        args=Args(text=secret),
        tool_call_id="fixture",
    )
    with sr.bind_policy(sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, secret),))):
        presentation = ToolUIDataAdapter(None).get_call_presentation(event)
    assert "synthetic-back" not in presentation.model_dump_json()
    assert "REDACTED" in presentation.display.summary
    assert event.args == Args(text=secret)


@pytest.mark.parametrize("command", ["printf 'unclosed", "bash -c 'unclosed"])
def test_unrecoverable_tool_call_source_omits_payload(command: str) -> None:
    import json

    from chartreux.core.events import ToolCallEvent
    from chartreux.core.llm_models import FunctionCall, ToolCall
    from chartreux.core.tools.builtins.bash import Bash, BashArgs

    call = ToolCall(
        function=FunctionCall(name="bash", arguments=json.dumps({"command": command}))
    )
    event = ToolCallEvent(
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command=command),
        tool_call_id="fixture",
    )
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, FAKE_VALUE),))
    ):
        assert sr.sanitize_recorded_tool_call(call).function.arguments is None
        assert sr.sanitize_recorded_tool_call(event).args is None
    assert event.args == BashArgs(command=command)


@pytest.mark.parametrize(
    "command",
    [
        "printf synthetic-se\\\ncret-0123456789",
        r"printf $'synthetic-se\x63ret-0123456789'",
        r"printf $'synthetic-se\143ret-0123456789'",
        r"printf $'synthetic-se\u0063ret-0123456789'",
    ],
)
def test_shell_escape_recovery_sanitizes_recorded_arguments_and_presentation(
    command: str,
) -> None:
    import json

    from chartreux.core.events import ToolCallEvent
    from chartreux.core.llm_models import FunctionCall, ToolCall
    from chartreux.core.tools.builtins.bash import Bash, BashArgs
    from chartreux.utils.tool_presentation import (
        EffectCallDisplay,
        ToolCallPresentation,
        ToolEffectKind,
    )

    secret = "synthetic-secret-0123456789"
    presentation = ToolCallPresentation(
        kind=ToolEffectKind.SHELL,
        display=EffectCallDisplay(summary=command, status_text="Running"),
    )
    call = ToolCall(
        function=FunctionCall(name="bash", arguments=json.dumps({"command": command})),
        presentation=presentation,
    )
    event = ToolCallEvent(
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command=command),
        tool_call_id="fixture",
        presentation=presentation,
    )
    with sr.bind_policy(sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, secret),))):
        for original in (call, event):
            recorded = sr.sanitize_recorded_tool_call(original)
            serialized = recorded.model_dump_json(exclude={"tool_class"})
            assert secret not in serialized
            assert "synthetic-se" not in serialized
            assert "cret-0123456789" not in serialized
            assert "REDACTED" in serialized
    assert event.args == BashArgs(command=command)
    assert call.presentation == presentation
    assert json.loads(call.function.arguments or "{}") == {"command": command}


@pytest.mark.parametrize(
    "command",
    [
        r"printf $'unknown\cX'",
        r"printf $'invalid\xZZ'",
        r"printf $'invalid\xff'",
        r"printf $'nul\0'",
        "printf $'unclosed",
    ],
)
def test_uncertain_shell_escapes_omit_recorded_values(command: str) -> None:
    import json

    from chartreux.core.events import ToolCallEvent
    from chartreux.core.llm_models import FunctionCall, ToolCall
    from chartreux.core.tools.builtins.bash import Bash, BashArgs

    call = ToolCall(
        function=FunctionCall(name="bash", arguments=json.dumps({"command": command}))
    )
    event = ToolCallEvent(
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command=command),
        tool_call_id="fixture",
    )
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, FAKE_VALUE),))
    ):
        assert sr.redact_shell_source(command) is None
        assert sr.sanitize_recorded_tool_call(call).function.arguments is None
        assert sr.sanitize_recorded_tool_call(event).args is None


@pytest.mark.parametrize(
    "command",
    [
        '  printf  "hello"  ',
        r"printf $'hello\n'",
        "printf syn\\\nthtoken",
        r"printf hello\ world",
    ],
)
def test_clean_shell_recordings_are_byte_identical(command: str) -> None:
    import json

    from chartreux.core.llm_models import FunctionCall, ToolCall

    arguments = json.dumps({"command": command}, indent=2)
    call = ToolCall(function=FunctionCall(name="bash", arguments=arguments))
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, FAKE_VALUE),))
    ):
        assert sr.redact_shell_source(command) == command
        assert sr.sanitize_recorded_tool_call(call).function.arguments == arguments


@pytest.mark.parametrize(
    "command",
    ['echo "don\'t"', 'git commit -m "don\'t break"', r"printf $'hello\x22world'"],
)
def test_literal_quotes_preserve_recorded_arguments_events_and_presentation(
    command: str,
) -> None:
    import json

    from chartreux.core.events import ToolCallEvent
    from chartreux.core.llm_models import FunctionCall, ToolCall
    from chartreux.core.tools.builtins.bash import Bash, BashArgs
    from chartreux.utils.tool_presentation import (
        EffectCallDisplay,
        ToolCallPresentation,
        ToolEffectKind,
    )

    expected = command
    arguments = json.dumps({"command": command}, indent=2)
    presentation = ToolCallPresentation(
        kind=ToolEffectKind.SHELL,
        display=EffectCallDisplay(summary=command, status_text=FAKE_VALUE),
    )
    call = ToolCall(
        function=FunctionCall(name="bash", arguments=arguments),
        presentation=presentation,
    )
    event = ToolCallEvent(
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command=command),
        tool_call_id="fixture",
        presentation=presentation,
    )
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, FAKE_VALUE),))
    ):
        assert sr.redact_shell_source(command) == expected
        recorded_call = sr.sanitize_recorded_tool_call(call)
        recorded_event = sr.sanitize_recorded_tool_call(event)
    assert recorded_call.function.arguments is not None
    assert json.loads(recorded_call.function.arguments) == {"command": expected}
    assert recorded_call.function.arguments == arguments
    assert recorded_event.args == BashArgs(command=expected)
    for recorded in (recorded_call, recorded_event):
        assert recorded.presentation is not None
        assert recorded.presentation.display.summary == expected
        assert recorded.presentation.display.status_text == sr.REDACTED_PLACEHOLDER
        assert FAKE_VALUE not in recorded.model_dump_json(exclude={"tool_class"})
    assert call.function.arguments == arguments
    assert event.args == BashArgs(command=command)
    assert call.presentation == event.presentation == presentation


def test_recorded_tool_call_numeric_substrings() -> None:
    import json

    from chartreux.core.llm_models import FunctionCall, ToolCall

    payload = {"nested": [1234567890123456.0, 91234567890123456, True, 42]}
    call = ToolCall(
        function=FunctionCall(name="fixture", arguments=json.dumps(payload))
    )
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=((FAKE_NAME, "1234567890123456"),))
    ):
        recorded = sr.sanitize_recorded_tool_call(call)
    assert json.loads(recorded.function.arguments) == {
        "nested": [sr.REDACTED_PLACEHOLDER, sr.REDACTED_PLACEHOLDER, True, 42]
    }
    assert json.loads(call.function.arguments or "{}") == payload


def test_redact_replaces_known_secret_value(
    fake_credential: dict[str, str | None],
) -> None:
    text = f"the key is {FAKE_VALUE} and the old one was {fake_credential[FAKE_NAME]}"
    redacted = sr.redact(text)
    assert FAKE_VALUE not in redacted
    assert "sk-stored" not in redacted
    assert redacted.count(sr.REDACTED_PLACEHOLDER) == 2


def test_redact_catches_base64_encoded_secret(
    fake_credential: dict[str, str | None],
) -> None:
    assert sr.redact(f"exfil via {FAKE_B64}") == f"exfil via {sr.REDACTED_PLACEHOLDER}"


def test_unpadded_base64url_with_non_aligned_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "credential-abcdefghijklmno"  # 26 characters
    monkeypatch.setenv(FAKE_NAME, value)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None})
    sr.reset_cache()
    encoded = base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")
    assert len(encoded) % 4 != 0
    assert encoded not in sr.redact(encoded)


def test_single_percent_encoded_character_is_redacted(
    fake_credential: dict[str, str | None],
) -> None:
    encoded = "%73" + FAKE_VALUE[1:]
    assert encoded not in sr.redact(encoded)
    mixed = "%73" + FAKE_VALUE[1:8] + "%30" + FAKE_VALUE[9:]
    assert mixed not in sr.redact(mixed)


def test_live_sessions_union_credential_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {})
    a, b = object_owner(), object_owner()
    name = "MCP_SESSION_A_TOKEN"
    sr.register_session_policy(a, sr.ScrubPolicy(credential_names=frozenset({name})))
    sr.register_session_policy(b, sr.ScrubPolicy())
    try:
        with sr.bind_policy(sr.ScrubPolicy()):
            assert name not in sr.scrub_child_env({name: "private"})
    finally:
        sr.unregister_session_policy(a)
        sr.unregister_session_policy(b)


@pytest.mark.parametrize(
    "snapshot", [None, SimpleNamespace(), SimpleNamespace(catalog=None)]
)
def test_scrub_policy_accepts_snapshot_without_catalog(snapshot: Any) -> None:
    config = SimpleNamespace(catalog_snapshot=snapshot, credential_env_passthrough=[])

    assert sr.ScrubPolicy.from_config(config).credential_names == frozenset()


@pytest.mark.parametrize("refresh", ["rediscovery", "reset_cache"])
def test_accepted_provider_names_survive_disk_removal(
    monkeypatch: pytest.MonkeyPatch, refresh: str
) -> None:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.model_catalog.loader import CatalogSnapshot
    from chartreux.core.model_catalog.schema import ModelCatalog, ProviderDefinition

    catalog = ModelCatalog(
        providers={
            "test": ProviderDefinition(
                api_base="https://test.invalid", api_key_env_var=FAKE_NAME
            ),
            "keyless": ProviderDefinition(api_base="https://keyless.invalid"),
        },
        models={},
    )
    config = ChartreuxConfigSchema.model_construct().attach_catalog_snapshot(
        CatalogSnapshot(catalog, "accepted")
    )
    policy = sr.ScrubPolicy.from_config(config)
    passthrough_policy = sr.ScrubPolicy.from_config(
        config.model_copy(update={"credential_env_passthrough": [FAKE_NAME]})
    )
    assert policy.credential_names == {FAKE_NAME}
    assert policy.redaction_credentials == ()
    assert policy.redaction_oauth_values == frozenset()

    disk_names = {FAKE_NAME}
    identity = [1]
    monkeypatch.setenv(FAKE_NAME, FAKE_VALUE)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset(disk_names))
    monkeypatch.setattr(sr, "_env_file_token", lambda: identity[0])
    sr.reset_cache()
    assert FAKE_NAME in sr.credential_env_var_names()
    disk_names.clear()
    if refresh == "rediscovery":
        identity[0] += 1
    else:
        sr.reset_cache()
    assert FAKE_NAME not in sr.credential_env_var_names()

    with sr.bind_policy(policy):
        assert FAKE_NAME not in _shell_environment()
        assert sr.redact(FAKE_VALUE) == sr.REDACTED_PLACEHOLDER
    assert sr.scrub_child_env({FAKE_NAME: FAKE_VALUE}, policy) == {}

    owner = object_owner()
    sr.register_session_policy(owner, passthrough_policy)
    try:
        assert FAKE_NAME not in _shell_environment()
        assert sr.redact(FAKE_VALUE) == sr.REDACTED_PLACEHOLDER
        with sr.bind_policy(passthrough_policy):
            assert _shell_environment()[FAKE_NAME] == FAKE_VALUE
            assert sr.redact(FAKE_VALUE) == sr.REDACTED_PLACEHOLDER
        assert sr.scrub_child_env({FAKE_NAME: FAKE_VALUE}, policy) == {}
    finally:
        sr.unregister_session_policy(owner)
    assert FAKE_NAME not in sr.credential_env_var_names()


def object_owner() -> Any:
    return type("Session", (), {})()


def test_redact_catches_hex_encoded_secret(
    fake_credential: dict[str, str | None],
) -> None:
    hex_value = FAKE_VALUE.encode("utf-8").hex()
    assert sr.redact(f"exfil via {hex_value}") == (
        f"exfil via {sr.REDACTED_PLACEHOLDER}"
    )


def test_redact_catches_base32_encoded_secret(
    fake_credential: dict[str, str | None],
) -> None:
    b32_value = base64.b32encode(FAKE_VALUE.encode("utf-8")).decode("ascii")
    assert sr.redact(f"exfil via {b32_value}") == (
        f"exfil via {sr.REDACTED_PLACEHOLDER}"
    )


def test_redact_catches_reversed_secret(fake_credential: dict[str, str | None]) -> None:
    # `rev` on the value (or the whole .env file) before or without encoding.
    reversed_value = FAKE_VALUE[::-1]
    assert sr.redact(f"exfil via {reversed_value}") == (
        f"exfil via {sr.REDACTED_PLACEHOLDER}"
    )
    reversed_line = f"{FAKE_NAME}={FAKE_VALUE}\n"[::-1]
    encoded = base64.b64encode(reversed_line.encode("utf-8")).decode("ascii")
    assert sr.redact(f"exfil via {encoded}") == (f"exfil via {sr.REDACTED_PLACEHOLDER}")


def test_redact_catches_openssl_wrapped_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    # `openssl base64` wraps at 64 columns, so the encoded value spans lines
    # and never matches the single-line fast-path variants.
    value = "sk-wrapped-" + "0123456789abcdef" * 4
    monkeypatch.setenv(FAKE_NAME, value)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.reset_cache()
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    assert len(encoded) > 64
    wrapped = "\n".join(encoded[i : i + 64] for i in range(0, len(encoded), 64))
    redacted = sr.redact(f"$ openssl base64\n{wrapped}\ndone")
    assert value not in redacted
    assert encoded not in redacted
    assert sr.REDACTED_PLACEHOLDER in redacted


def test_redact_catches_whole_file_base64_with_multiple_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A two-key .env base64'd whole: the per-line fast-path variants only
    # match line-offset-aligned encodings, so the second key survives them.
    # The decode-and-scan backstop decodes the whole dump and redacts it.
    other_name = "CHARTREUX_TEST_OTHER_KEY"
    other_value = "sk-other-0123456789abcdefghij"
    monkeypatch.setenv(FAKE_NAME, FAKE_VALUE)
    monkeypatch.setattr(
        sr,
        "_read_dotenv_entries",
        lambda: {FAKE_NAME: FAKE_VALUE, other_name: other_value},
    )
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.reset_cache()
    env_file = f"{FAKE_NAME}={FAKE_VALUE}\n{other_name}={other_value}\n"
    encoded = base64.b64encode(env_file.encode("utf-8")).decode("ascii")
    wrapped = "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))
    redacted = sr.redact(f"$ base64 chartreux.env\n{wrapped}")
    assert FAKE_VALUE not in redacted
    assert other_value not in redacted
    assert encoded not in redacted
    assert sr.REDACTED_PLACEHOLDER in redacted


def test_redact_leaves_ordinary_encoded_looking_text_alone(
    fake_credential: dict[str, str | None],
) -> None:
    # Long hex/base64-looking runs that decode to nothing secret stay intact.
    text = f"sha256: {'a1b2c3d4e5f60718293a4b5c6d7e8f90' * 2} ok"
    assert sr.redact(text) == text


def test_redact_is_noop_without_known_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.reset_cache()
    assert sr.redact("nothing to see here") == "nothing to see here"


def test_redact_skips_values_below_minimum_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FAKE_NAME, "shorter")
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.reset_cache()
    assert sr.redact("a shorter appears") == "a shorter appears"


@pytest.mark.parametrize("length", [7, 8, 15, 16])
def test_direct_known_secret_length_threshold(
    monkeypatch: pytest.MonkeyPatch, length: int
) -> None:
    value = "z" * length
    monkeypatch.setenv(FAKE_NAME, value)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None})
    sr.reset_cache()
    expected = sr.REDACTED_PLACEHOLDER if length >= 8 else value
    assert sr.redact(f"a {value} appears") == f"a {expected} appears"


def test_short_encoded_secrets_match_direct_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "aB!9xyZ2"  # 8 characters; two escapes make a 12-character carrier.
    monkeypatch.setenv(FAKE_NAME, value)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None})
    sr.reset_cache()
    carrier = "aB%21%39xyZ2"
    assert len(carrier) == 12
    assert sr.redact(carrier) == sr.REDACTED_PLACEHOLDER
    assert sr.redact("ordinary 12345678 and 0123456789abcdef") == (
        "ordinary 12345678 and 0123456789abcdef"
    )
    value = "short-secret-15"  # 15 characters.
    monkeypatch.setenv(FAKE_NAME, value)
    assert sr.redact(("prefix " + value + " suffix").encode().hex()) == (
        sr.REDACTED_PLACEHOLDER
    )


def test_search_provider_env_is_scrubbed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name, value = "EXA_API_KEY", "exa-sensitive-0123456789"
    monkeypatch.setenv(name, value)
    assert name in sr.credential_env_scrub_names()
    assert name not in sr.scrub_child_env({name: value})
    assert sr.redact(value) == sr.REDACTED_PLACEHOLDER


def test_rotated_dotenv_value_first_query(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = {FAKE_NAME: "first-secret-0123456789"}
    identity = [1]
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: entries.copy())
    monkeypatch.setattr(sr, "_env_file_token", lambda: (identity[0], 32))
    sr.reset_cache()
    assert sr.redact(entries[FAKE_NAME]) == sr.REDACTED_PLACEHOLDER
    entries[FAKE_NAME] = "other-secret-0123456789"  # same length
    identity[0] = 2
    assert sr.redact(entries[FAKE_NAME]) == sr.REDACTED_PLACEHOLDER


def test_keyring_miss_set_invalidation_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.utils import keyring as kr

    record: dict[str, str] = {}
    monkeypatch.setattr(
        kr, "_set_password", lambda service, name, value: record.update({name: value})
    )
    monkeypatch.setattr(kr, "get_api_key_from_keyring", lambda name: record.get(name))
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None})
    sr.reset_cache()
    assert sr.redact(FAKE_VALUE) == FAKE_VALUE
    monkeypatch.setattr(kr, "_is_keyring_disabled", lambda: False)
    kr.set_api_key_in_keyring(FAKE_NAME, FAKE_VALUE)
    assert sr.redact(FAKE_VALUE) == sr.REDACTED_PLACEHOLDER
    sr.invalidate_keyring_credential(FAKE_NAME)
    record.clear()
    clock = [100.0]
    monkeypatch.setattr(sr.time, "monotonic", lambda: clock[0])
    assert sr.redact(FAKE_VALUE) == FAKE_VALUE
    record[FAKE_NAME] = FAKE_VALUE
    assert sr.redact(FAKE_VALUE) == FAKE_VALUE
    clock[0] += sr._NEGATIVE_KEYRING_TTL + 1
    assert sr.redact(FAKE_VALUE) == sr.REDACTED_PLACEHOLDER


@pytest.mark.parametrize(
    "carrier",
    [
        lambda value: base64.urlsafe_b64encode((value + "\xff").encode()).decode(),
        lambda value: "".join(f"%{byte:02X}" for byte in value.encode()),
        lambda value: base64.b64encode(base64.b64encode(value.encode())).decode(),
        lambda value: "\n".join(
            f"{offset:07o} "
            + " ".join(f"{byte:02x}" for byte in value.encode()[offset : offset + 16])
            for offset in range(0, len(value.encode()), 16)
        ),
    ],
)
def test_encoded_carriers(fake_credential: dict[str, str | None], carrier: Any) -> None:
    encoded = carrier(FAKE_VALUE)
    assert encoded not in sr.redact(f"carrier:\n{encoded}\n")


def test_indented_od_no_offset_rows(fake_credential: dict[str, str | None]) -> None:
    data = FAKE_VALUE.encode() + b"x" * (-len(FAKE_VALUE.encode()) % 16)
    rows = "\n".join(
        " " + " ".join(f"{byte:02x}" for byte in data[i : i + 16])
        for i in range(0, len(data), 16)
    )
    assert sr.redact(f"$ od -An -tx1\n{rows}\ndone") == (
        f"$ od -An -tx1\n{sr.REDACTED_PLACEHOLDER}\ndone"
    )


def test_long_carrier_window_boundary(fake_credential: dict[str, str | None]) -> None:
    carrier = base64.b64encode(("x" * 5300 + FAKE_VALUE + "y" * 2300).encode()).decode()
    assert len(carrier) > 8192
    assert carrier not in sr.redact(carrier)


def test_budget_exhaustion_is_bounded(fake_credential: dict[str, str | None]) -> None:
    carrier = base64.b64encode(
        ("x" * (sr._MAX_ENCODED_CANDIDATES * 5500) + FAKE_VALUE).encode()
    ).decode()
    # Encoding outside the bounded decode budget can remain visible; callers
    # must never treat this as an exhaustive exfiltration control.
    assert sr.redact(carrier) == carrier


def test_budget_exhaustion_still_redacts_direct_value(
    fake_credential: dict[str, str | None],
) -> None:
    # Distinct punctuated candidates fill the 128-window budget before tail.
    decoys = ":".join(f"{i:016x}" for i in range(sr._MAX_ENCODED_CANDIDATES))
    tail = base64.b64encode(("prefix " + FAKE_VALUE + " suffix").encode()).decode()
    text = f"{decoys}:{tail}:{FAKE_VALUE}"
    redacted = sr.redact(text)
    assert tail in redacted  # Accepted residual: later encoded carrier passes.
    assert redacted.endswith(f":{sr.REDACTED_PLACEHOLDER}")


def test_prose_collateral_is_zero(fake_credential: dict[str, str | None]) -> None:
    prose = (
        "The documentation describes authorization, evaluation, portability, and compatibility. "
        * 25
    )
    assert sr.redact(prose) == prose


def test_redaction_per_event_latency_record(
    fake_credential: dict[str, str | None],
) -> None:
    for label, text in (
        ("typical", "normal output " * 40),
        ("large", "normal output " * 1500),
    ):
        start = time.perf_counter()
        sr.redact(text)
        elapsed = time.perf_counter() - start
        print(f"redact {label}: {elapsed:.6f}s")
        assert elapsed < 2.0


def test_known_mcp_record_consulted_without_keyring_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.utils import keyring as kr

    username = "mcp-oauth:known:tokens"
    token = "known-access-token-0123456789"
    queries: list[str] = []

    def lookup(name: str) -> str | None:
        queries.append(name)
        return '{"access_token":"' + token + '"}' if name == username else None

    monkeypatch.setattr(kr, "get_api_key_from_keyring", lookup)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {})
    policy = sr.ScrubPolicy(oauth_records=frozenset({username}))
    with sr.bind_policy(policy):
        assert sr.redact(token) == sr.REDACTED_PLACEHOLDER
    assert username in queries
    assert all(
        name == username or name in sr.credential_env_var_names() for name in queries
    )


@pytest.mark.asyncio
async def test_oauth_tokens_registered_on_login_refresh_and_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.core.auth import mcp_oauth as oauth

    records: dict[str, str] = {}
    monkeypatch.setattr(
        oauth, "set_api_key_in_keyring", lambda name, raw: records.update({name: raw})
    )
    monkeypatch.setattr(
        oauth, "get_api_key_from_keyring_uncached", lambda name: records.get(name)
    )
    name = "mcp-oauth:example:tokens"
    first = '{"access_token":"oauth-first-0123456789","refresh_token":"refresh-first-0123456789"}'
    second = '{"access_token":"oauth-next-0123456789","refresh_token":"refresh-next-0123456789"}'
    await oauth._kr_set(name, first)
    assert (
        sr.redact("oauth-first-0123456789 refresh-first-0123456789")
        == "[REDACTED] [REDACTED]"
    )
    await oauth._kr_set(name, second)
    assert (
        sr.redact("oauth-next-0123456789 refresh-next-0123456789")
        == "[REDACTED] [REDACTED]"
    )
    sr.reset_cache()
    assert await oauth._kr_get(name) == second
    assert sr.redact("oauth-next-0123456789") == sr.REDACTED_PLACEHOLDER
    monkeypatch.setattr(
        oauth, "delete_api_key_from_keyring", lambda name: records.pop(name)
    )
    await oauth._kr_delete(name)
    assert sr.redact("oauth-next-0123456789") == "oauth-next-0123456789"


def test_scrub_child_env_removes_credential_vars(
    fake_credential: dict[str, str | None],
) -> None:
    env = sr.scrub_child_env({FAKE_NAME: FAKE_VALUE, "PATH": "/usr/bin"})
    assert FAKE_NAME not in env
    assert env["PATH"] == "/usr/bin"


def test_passthrough_restores_scrubbed_var(
    fake_credential: dict[str, str | None],
) -> None:
    assert FAKE_NAME in sr.credential_env_scrub_names()
    sr.set_env_passthrough([FAKE_NAME])
    assert FAKE_NAME not in sr.credential_env_scrub_names()
    env = sr.scrub_child_env({FAKE_NAME: FAKE_VALUE, "PATH": "/usr/bin"})
    assert env[FAKE_NAME] == FAKE_VALUE


def test_shell_environment_is_scrubbed(fake_credential: dict[str, str | None]) -> None:
    env = _shell_environment()
    assert FAKE_NAME not in env
    assert env["CI"] == "true"
    assert "PATH" in env


def test_stdio_params_use_explicit_scrubbed_environment(
    fake_credential: dict[str, str | None],
) -> None:
    params = build_stdio_params(["/usr/bin/env"])
    assert params.env is not None
    assert FAKE_NAME not in params.env
    # The SDK's safe inherited defaults survive the scrub.
    assert "PATH" in params.env


def test_stdio_params_per_server_env_acts_as_passthrough(
    fake_credential: dict[str, str | None],
) -> None:
    params = build_stdio_params(["/usr/bin/env"], env={FAKE_NAME: FAKE_VALUE})
    assert params.env is not None
    assert params.env[FAKE_NAME] == FAKE_VALUE


def test_git_project_context_subprocess_gets_scrubbed_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from chartreux.core.config import ProjectContextConfig
    from chartreux.core.system_prompt import ProjectContextProvider

    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        captured["args"] = args
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "chartreux.core.system_prompt.resolve_git_executable",
        lambda cwd=None: "/usr/bin/git",
    )
    provider = ProjectContextProvider(ProjectContextConfig(), root_path=tmp_path)
    provider._run_git(["status"], timeout=1.0)
    env = captured["env"]
    assert isinstance(env, dict)
    assert FAKE_NAME not in env
    assert "PATH" in env


@pytest.mark.asyncio
async def test_grep_child_environment_is_scrubbed(
    fake_credential: dict[str, str | None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from chartreux.core.tools.builtins.grep import Grep

    captured: dict[str, Any] = {}

    async def fake_spawn(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(returncode=1, communicate=reply)

    async def reply() -> tuple[bytes, bytes]:
        return b"", b""

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    fake = cast(
        Grep, SimpleNamespace(cwd=tmp_path, config=SimpleNamespace(default_timeout=2))
    )
    assert await Grep._execute_search(fake, ["grep", "something"]) == ""
    assert FAKE_NAME not in captured["env"]


def test_git_metadata_and_index_children_are_scrubbed(
    fake_credential: dict[str, str | None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from chartreux.cli.autocompletion.file_indexer.store import FileIndexStore
    from chartreux.core.session.session_logger import SessionLogger

    captured: list[dict[str, Any]] = []

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="")

    class FakeProcess:
        returncode = 1

        def communicate(self, timeout: float) -> tuple[bytes, bytes]:
            return b"", b""

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        captured.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "chartreux.core.session.session_logger.resolve_git_executable",
        lambda cwd: "/usr/bin/git",
    )
    monkeypatch.setattr(
        "chartreux.cli.autocompletion.file_indexer.store.resolve_git_executable",
        lambda cwd: "/usr/bin/git",
    )
    assert SessionLogger._fetch_git_metadata(
        cast(SessionLogger, SimpleNamespace(cwd=tmp_path))
    ) == (None, None)
    assert (
        FileIndexStore._list_git_entries(
            cast(FileIndexStore, SimpleNamespace()), tmp_path, None
        )
        is None
    )
    assert len(captured) == 2
    assert FAKE_NAME not in captured[0]["env"]
    assert "env" not in captured[1]


@pytest.mark.asyncio
async def test_client_terminal_request_carries_scrub_list(
    fake_credential: dict[str, str | None],
) -> None:
    from chartreux.app_server._tool_io import ClientToolIO
    from chartreux.app_server.protocol import (
        ClientCapabilities,
        ClientToolTerminalCreateResponse,
        ClientToolTerminalOutputResponse,
        ClientToolTerminalWaitResponse,
    )
    from chartreux.core.tools.io_port import ShellCommandRequest

    class Bridge:
        def client_capabilities(self) -> ClientCapabilities:
            return ClientCapabilities(client_tools=["terminal"])

        def current_session_id(self) -> str:
            return "root"

        async def request_client_result(self, method, params, response_type):
            if method == "clientTool/terminal/create":
                self.created = params
                return response_type.model_validate(
                    ClientToolTerminalCreateResponse(terminal_id="t1").model_dump(
                        mode="json"
                    )
                )
            if method == "clientTool/terminal/wait":
                return response_type.model_validate(
                    ClientToolTerminalWaitResponse(exit_code=0).model_dump(mode="json")
                )
            return response_type.model_validate(
                ClientToolTerminalOutputResponse(
                    output="ok", truncated=False
                ).model_dump(mode="json")
            )

    bridge = Bridge()
    result = await ClientToolIO(bridge).run_shell(
        ShellCommandRequest(
            session_id="root",
            tool_call_id="call-1",
            command="/bin/bash",
            cwd=Path("/workspace"),
            timeout=1.0,
            max_output_bytes=100,
        )
    )
    assert result.stdout == "ok"
    assert FAKE_NAME in bridge.created.env_scrub


@pytest.mark.asyncio
async def test_acp_handler_forwards_scrub_list_in_meta(
    fake_credential: dict[str, str | None],
) -> None:
    from chartreux.acp.tool_io import AcpClientToolHandler
    from chartreux.app_server.protocol import ClientToolTerminalCreateParams

    class FakeAcpClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create_terminal(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)

            class _Response:
                terminal_id = "terminal-1"

            return _Response()

        async def session_update(self, **kwargs: Any) -> None:
            return None

    client = FakeAcpClient()
    handler = AcpClientToolHandler(client)  # type: ignore[arg-type]
    handler.bind_session("acp-session")

    await handler.create_terminal(
        ClientToolTerminalCreateParams(
            session_id="session-1",
            command="/bin/bash",
            args=["-c", "echo hi"],
            env_scrub=[FAKE_NAME],
            cwd="/workspace",
            output_byte_limit=100,
        )
    )
    (call,) = client.calls
    assert call["env_scrub"] == [FAKE_NAME]

    # Without a scrub list, no meta key is forwarded.
    await handler.create_terminal(
        ClientToolTerminalCreateParams(
            session_id="session-1",
            command="/bin/bash",
            cwd="/workspace",
            output_byte_limit=100,
        )
    )
    assert "env_scrub" not in client.calls[1]


def test_policy_isolation_and_contextless_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.conftest import build_test_agent_loop, build_test_vibe_config

    other = "CHARTREUX_TEST_OTHER_CREDENTIAL"
    monkeypatch.setattr(
        sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None, other: None}
    )
    sr.reset_cache()
    a = build_test_agent_loop(
        config=build_test_vibe_config(credential_env_passthrough=[FAKE_NAME])
    )
    b = build_test_agent_loop(
        config=build_test_vibe_config(credential_env_passthrough=[other])
    )
    env = {FAKE_NAME: FAKE_VALUE, other: FAKE_VALUE}
    assert sr.scrub_child_env(env, a.scrub_policy) == {FAKE_NAME: FAKE_VALUE}
    assert sr.scrub_child_env(env, b.scrub_policy) == {other: FAKE_VALUE}
    assert sr.scrub_child_env(env) == {}


@pytest.mark.asyncio
async def test_concurrent_session_shell_children_do_not_share_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.conftest import build_test_agent_loop, build_test_vibe_config

    other = "CHARTREUX_TEST_SECOND_KEY"
    monkeypatch.setenv(FAKE_NAME, FAKE_VALUE)
    monkeypatch.setenv(other, "other-value")
    monkeypatch.setattr(
        sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None, other: None}
    )
    sr.reset_cache()
    a = build_test_agent_loop(
        config=build_test_vibe_config(credential_env_passthrough=[FAKE_NAME])
    )
    b = build_test_agent_loop(
        config=build_test_vibe_config(credential_env_passthrough=[other])
    )

    async def shell_child(policy: sr.ScrubPolicy) -> tuple[bool, bool]:
        with sr.bind_policy(policy):
            await asyncio.sleep(0)
            env = _shell_environment()
            return FAKE_NAME in env, other in env

    assert await asyncio.gather(
        shell_child(a.scrub_policy), shell_child(b.scrub_policy)
    ) == [(True, False), (False, True)]


@pytest.mark.asyncio
async def test_policy_change_retires_persistent_mcp_connections() -> None:
    from tests.conftest import build_test_agent_loop

    loop = build_test_agent_loop()
    old_pool = loop._mcp_pool
    assert old_pool is not None
    loop.scrub_policy = sr.ScrubPolicy(frozenset({FAKE_NAME}))
    loop._retire_mcp_pool()
    replacement = loop._create_mcp_pool()
    try:
        assert old_pool._closed
        assert replacement is not old_pool
        assert replacement._policy == loop.scrub_policy
    finally:
        await replacement.aclose()
        await loop.aclose()


@pytest.mark.asyncio
async def test_policy_admission_snapshot_and_cancellation_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {FAKE_NAME: None})
    sr.reset_cache()
    started = asyncio.Event()
    resume = asyncio.Event()
    first = sr.ScrubPolicy(frozenset({FAKE_NAME}))
    second = sr.ScrubPolicy()

    async def admitted() -> dict[str, str]:
        started.set()
        await resume.wait()
        return sr.scrub_child_env({FAKE_NAME: FAKE_VALUE})

    with sr.bind_policy(first):
        task = asyncio.create_task(admitted())
    await started.wait()
    with sr.bind_policy(second):
        resume.set()
        assert await task == {FAKE_NAME: FAKE_VALUE}
        assert sr.scrub_child_env({FAKE_NAME: FAKE_VALUE}) == {}
    assert sr.current_policy() == sr.ScrubPolicy()

    with sr.bind_policy(first):
        cancelled = asyncio.create_task(asyncio.sleep(10))
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert sr.current_policy() == sr.ScrubPolicy()


def test_mcp_names_are_policy_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    name_a, name_b = "TEST_MCP_AUTH_A", "TEST_MCP_AUTH_B"
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.reset_cache()
    a = sr.ScrubPolicy(credential_names=frozenset({name_a}))
    b = sr.ScrubPolicy(credential_names=frozenset({name_b}))
    assert sr.scrub_child_env({name_a: "a", name_b: "b"}, a) == {name_b: "b"}
    assert sr.scrub_child_env({name_a: "a", name_b: "b"}, b) == {name_a: "a"}
    assert {name_a, name_b}.isdisjoint(sr.credential_env_var_names())


def test_agent_loop_syncs_passthrough_from_config() -> None:
    from tests.conftest import build_test_agent_loop, build_test_vibe_config

    config = build_test_vibe_config(credential_env_passthrough=[FAKE_NAME])
    loop = build_test_agent_loop(config=config)
    assert loop.scrub_policy.passthrough == {FAKE_NAME}
    assert sr.env_passthrough() == frozenset()


def test_mcp_static_auth_env_names_collects_static_auth_only() -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(
        mcp_servers=[
            SimpleNamespace(auth=SimpleNamespace(api_key_env="MCP_HTTP_TOKEN")),
            SimpleNamespace(),  # stdio server: no auth at all
            SimpleNamespace(auth=SimpleNamespace()),  # oauth: no api_key_env
            SimpleNamespace(auth=SimpleNamespace(api_key_env="")),
        ]
    )
    assert sr.mcp_static_auth_env_names(config) == {"MCP_HTTP_TOKEN"}
    assert sr.mcp_static_auth_env_names(SimpleNamespace()) == frozenset()


def test_mcp_static_auth_env_name_secret_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "CHARTREUX_TEST_MCP_TOKEN"
    value = "tok-mcp-0123456789abcdefghij"
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.set_mcp_static_auth_env_names([name])
    sr.reset_cache()
    assert name in sr.credential_env_var_names()
    assert sr.redact(f"leaked header token: {value}") == (
        f"leaked header token: {sr.REDACTED_PLACEHOLDER}"
    )


def test_agent_loop_syncs_mcp_static_auth_env_names() -> None:
    from chartreux.core.config.models import MCPHttp, MCPStaticAuth
    from tests.conftest import build_test_agent_loop, build_test_vibe_config

    config = build_test_vibe_config(
        mcp_servers=[
            MCPHttp(
                name="remote",
                transport="streamable-http",
                url="https://example.invalid/mcp",
                auth=MCPStaticAuth(api_key_env=FAKE_NAME),
            )
        ]
    )
    loop = build_test_agent_loop(config=config)
    assert FAKE_NAME in sr.credential_env_var_names(loop.scrub_policy)
    assert FAKE_NAME not in sr.current_policy().credential_names


def test_structured_payload_sanitizes_encoded_values_and_keys(
    fake_credential: dict[str, str | None],
) -> None:
    encoded = FAKE_VALUE.encode().hex()
    payload = {
        f"key:{FAKE_VALUE}": "first",
        f"other:{FAKE_VALUE}": "second",
        "encoded": encoded,
        "ordinary": "ordinary content",
    }
    cleaned = sr.redact_json_value(payload)
    assert FAKE_VALUE not in str(cleaned)
    assert encoded not in str(cleaned)
    assert len(cleaned) == len(payload)
    assert set(cleaned.values()) == {
        "first",
        "second",
        "ordinary content",
        sr.REDACTED_PLACEHOLDER,
    }
    assert cleaned["ordinary"] == "ordinary content"
    collision = sr.redact_json_value({FAKE_VALUE: 1, FAKE_VALUE[::-1]: 2})
    assert list(collision) == [sr.REDACTED_PLACEHOLDER, f"{sr.REDACTED_PLACEHOLDER}#2"]
    assert list(collision.values()) == [1, 2]


def test_deep_json_subtree_is_omitted_without_losing_siblings(
    fake_credential: dict[str, str | None],
) -> None:
    payload: dict[str, Any] = {FAKE_VALUE: [FAKE_VALUE]}
    for _ in range(1100):
        payload = {"nested": payload}
    cleaned = sr.redact_json_value({"ok": "visible", "deep": payload})
    assert cleaned["ok"] == "visible"
    node = cleaned["deep"]
    for _ in range(sr._MAX_JSON_REDACTION_DEPTH - 1):
        node = node["nested"]
    assert node == sr.REDACTED_PLACEHOLDER


def test_persisted_result_reconstruction_failure_is_safe(
    fake_credential: dict[str, str | None],
) -> None:
    from chartreux.core.llm_models import PersistedToolResult

    class BrokenResult(PersistedToolResult):
        def model_dump(self, **kwargs: Any) -> Any:
            raise ValueError("unavailable")

    # Construct without validation to simulate a corrupt persisted model.
    result = BrokenResult.model_construct(duration=1.25, cancelled=True)
    cleaned = sr.redact_persisted_result(result)
    assert cleaned.output == {"error": "Tool result unavailable"}
    assert cleaned.duration == 1.25
    assert cleaned.cancelled is True


def test_redact_persisted_result_fallback_redacts_presentation(
    fake_credential: dict[str, str | None],
) -> None:
    from chartreux.core.llm_models import PersistedToolResult
    from chartreux.utils.tool_presentation import (
        EffectResultDisplay,
        ToolEffectKind,
        ToolResultPresentation,
    )

    class UnvalidatableResult(PersistedToolResult):
        @classmethod
        def model_validate(cls, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("revalidation disabled for test")

    result = UnvalidatableResult(
        duration=0.0,
        cancelled=True,
        output={"stdout": f"leaked {FAKE_VALUE}"},
        presentation=ToolResultPresentation(
            kind=ToolEffectKind.TOOL,
            display=EffectResultDisplay(
                success=True, verb="Ran", message=f"leaked {FAKE_VALUE}"
            ),
        ),
    )

    redacted = sr.redact_persisted_result(result)

    assert redacted.duration == 0.0
    assert redacted.cancelled is True
    assert FAKE_VALUE not in redacted.output["stdout"]
    assert FAKE_VALUE not in redacted.presentation.display.message
    assert sr.REDACTED_PLACEHOLDER in redacted.output["stdout"]
    assert sr.REDACTED_PLACEHOLDER in redacted.presentation.display.message
