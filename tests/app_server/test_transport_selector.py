from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    TRANSPORT_ENV_VAR,
    MalformedTransportSelectorError,
    TransportSelection,
    UnsupportedTransportSelectorError,
    parse_transport_selector,
    resolve_transport_selection,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.environment import EnvironmentLayer


def _environ(**values: str) -> dict[str, str]:
    return dict(values)


@pytest.mark.parametrize("value", ["in_process", "IN_PROCESS", " in_process "])
def test_in_process_is_the_supported_selector(value: str) -> None:
    assert parse_transport_selector(value) == IN_PROCESS_TRANSPORT


@pytest.mark.parametrize("value", ["", "carrier_pigeon", "in process", "0"])
def test_malformed_selectors_are_rejected_without_echoing_the_value(value: str) -> None:
    with pytest.raises(MalformedTransportSelectorError) as caught:
        parse_transport_selector(value)

    message = str(caught.value)
    assert TRANSPORT_ENV_VAR in message
    assert "'in_process'" in message
    if value.strip():
        assert value.strip() not in message


def test_multi_process_is_recognized_but_not_wired() -> None:
    with pytest.raises(UnsupportedTransportSelectorError, match="multi-process"):
        parse_transport_selector("multi_process")


def test_default_selection_is_in_process() -> None:
    selection = resolve_transport_selection(environ=_environ())

    assert selection == TransportSelection(
        kind=IN_PROCESS_TRANSPORT, selector=IN_PROCESS_TRANSPORT, source="default"
    )


@pytest.mark.parametrize("value", ["in_process", "IN_PROCESS"])
def test_environment_selector_selects_in_process_from_any_case(value: str) -> None:
    selection = resolve_transport_selection(
        environ=_environ(**{TRANSPORT_ENV_VAR: value})
    )

    assert selection.kind == IN_PROCESS_TRANSPORT
    assert selection.source == "environment"


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_environment_selector_stays_at_the_default(value: str) -> None:
    selection = resolve_transport_selection(
        environ=_environ(**{TRANSPORT_ENV_VAR: value})
    )

    assert selection.source == "default"


def test_config_escape_key_selects_in_process() -> None:
    selection = resolve_transport_selection(multi_process=False, environ=_environ())

    assert selection.kind == IN_PROCESS_TRANSPORT
    assert selection.source == "config"


def test_explicit_environment_selector_outranks_the_config_escape() -> None:
    selection = resolve_transport_selection(
        multi_process=False, environ=_environ(**{TRANSPORT_ENV_VAR: "in_process"})
    )

    assert selection.source == "environment"


def test_a_malformed_environment_selector_does_not_fall_back_to_config() -> None:
    with pytest.raises(MalformedTransportSelectorError):
        resolve_transport_selection(
            multi_process=False,
            environ=_environ(**{TRANSPORT_ENV_VAR: "carrier_pigeon"}),
        )


def test_default_selection_ignores_an_enabled_multi_process_flag() -> None:
    selection = resolve_transport_selection(multi_process=True, environ=_environ())

    assert selection.source == "default"


def test_the_config_escape_key_is_a_schema_field() -> None:
    assert ChartreuxConfigSchema().multi_process is True
    assert ChartreuxConfigSchema(multi_process=False).multi_process is False


@pytest.mark.asyncio
async def test_transport_controls_are_exempt_from_environment_config_validation() -> (
    None
):
    # Regression: the WP1 checkpoint runs the suite with the selector (and
    # spawned children carry the bootstrap marker) in the environment. Both are
    # process controls, so config loading must not treat them as unknown settings.
    env = {
        TRANSPORT_ENV_VAR: "in_process",
        "CHARTREUX_PROCESS_BOOTSTRAP": "1",
        "CHARTREUX_MULTI_PROCESS": "false",
    }
    with patch.dict(os.environ, env, clear=True):
        data = await EnvironmentLayer(schema=ChartreuxConfigSchema).load()

    assert data.model_dump() == {"multi_process": False}


class _StopConnect(Exception):
    """Ends ``connect`` once the harness factory has recorded its arguments."""


@pytest.mark.asyncio
async def test_the_local_harness_connects_with_the_selected_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.app_server import local as local_mod

    captured: dict[str, object] = {}

    async def fake_create(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise _StopConnect

    monkeypatch.setattr(local_mod, "create_harness_server", fake_create)
    monkeypatch.setenv(TRANSPORT_ENV_VAR, "in_process")

    with pytest.raises(_StopConnect):
        await local_mod.LocalHarnessHost().connect(local_mod.LocalHarnessOptions())

    assert captured["transport_kind"] == IN_PROCESS_TRANSPORT


@pytest.mark.asyncio
async def test_the_local_harness_refuses_a_malformed_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.app_server import local as local_mod

    monkeypatch.setenv(TRANSPORT_ENV_VAR, "carrier_pigeon")

    with pytest.raises(MalformedTransportSelectorError):
        await local_mod.LocalHarnessHost().connect(local_mod.LocalHarnessOptions())
