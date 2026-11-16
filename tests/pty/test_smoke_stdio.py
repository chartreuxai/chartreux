"""Smoke flows for the spawned TUI on the stdio transport (design §6.5).

Every node here drives ONE spawned CLI through ONE surface. The TUI keeps a
screen stack, so a single long session makes a later failure unattributable and
lets one stuck overlay turn every following step red for the wrong reason; one
session per flow keeps both cost and diagnosis bounded.

What these flows prove is that the surface renders over the stdio topology with
the deterministic in-gate provider and no wire refusal: the harness spawns a
real `chartreux` process, the mock provider is the only model endpoint on the
gate's loopback, and each node asserts the rendered marker it observed.

Not every §6.5 flow is open here: root-death recovery and the transcript
Load-more affordance landed in ``test_smoke_stdio_flows.py`` (the sibling
that owns the multi-process and scripted-provider scenarios), and so did the
spawned-child composed-OSC leg, the agent-browser mount and -- since the F10
owner-side ``agents/update`` emission arm landed -- the browser's row
lifecycle with the transcript view, release and cancel interactions, all
documented at their nodes there.
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group(name="pty_gate")]

#: The prompt footer, rendered whenever no overlay is open: a reliable
#: "we are back at the prompt" marker.
MAIN_SCREEN = "F1 Help · /help"

#: The command palette claims the first Enter to complete the command name, so
#: a slash command needs a submit Enter after it.
PALETTE_SETTLE_S = 0.6


def _settle_to_prompt(driver) -> None:
    driver.wait_for(MAIN_SCREEN, 40)


def _submit(driver, text: str) -> int:
    """Type a line, submit it, and return the transcript mark to inspect from."""
    mark = driver.mark()
    driver.send(text)
    time.sleep(0.4)
    driver.send("\r")
    return mark


def _open_command(driver, command: str, marker: str) -> int:
    """Open a slash command's screen and wait for its rendered marker."""
    mark = driver.mark()
    driver.send(command)
    time.sleep(PALETTE_SETTLE_S)
    driver.send("\r")
    time.sleep(PALETTE_SETTLE_S)
    driver.send("\r")
    driver.wait_for(marker, 40)
    return mark


def _assert_no_wire_errors(driver, mark: int) -> None:
    errors = driver.errors_since(mark)
    assert not errors, "\n".join(errors) + "\n\n" + driver.tail(mark, 1500)


def test_prompt_submit_streams_a_reply_and_quits_cleanly(
    pty_driver, local_mock_provider
) -> None:
    """A submitted prompt reaches the in-gate provider and streams its reply."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _submit(driver, "hello from the pty gate")
    driver.wait_for("from mock server", 60)
    assert len(local_mock_provider.requests) >= 1
    _assert_no_wire_errors(driver, mark)

    # session/stop over the stdio topology: Ctrl+C twice leaves cleanly.
    driver.send("\x03")
    driver.wait_for("Press Ctrl+C again to quit", 20)
    driver.send("\x03")
    driver.wait_for_exit(30)


def test_model_picker_lists_the_configured_model(pty_driver) -> None:
    """`/model` renders the picker with the isolated catalog's model."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/model", "Model for this session")
    assert "mock-provider/mock-model" in driver.text()
    _assert_no_wire_errors(driver, mark)


def test_settings_screen_reads_the_live_config(pty_driver) -> None:
    """`/settings` reads the live configuration over the stdio transport."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/settings", "PROMPTS & COMPACTION")
    _assert_no_wire_errors(driver, mark)


def test_settings_filter_opens_the_shell_permissions_panel(pty_driver) -> None:
    """The settings filter narrows to `Shell Permissions` and Enter opens it.

    The §6.5 shell-permissions flow: the option list's query feed narrows the
    catalog to the descriptor row, Enter activates it, and the panel mounts
    over the stdio topology with the session/project grant sections the
    service inspects -- no wire refusal on the open or the inspection.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/settings", "PROMPTS & COMPACTION")
    time.sleep(1.0)

    driver.send("permission")
    driver.wait_for("Inspect and revoke dangerous-command", 30)
    driver.send("\r")
    driver.wait_for("Live session grants", 30)
    _assert_no_wire_errors(driver, mark)


def test_mcp_screen_reports_an_empty_catalog(pty_driver) -> None:
    """`/mcp` reads the MCP catalog over the stdio transport."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/mcp", "No MCP servers configured")
    _assert_no_wire_errors(driver, mark)


def test_loop_screen_reports_no_scheduled_loops(pty_driver) -> None:
    """`/loop` reaches the loop owner over the stdio transport."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/loop", "No scheduled loops.")
    _assert_no_wire_errors(driver, mark)


def test_debug_console_reports_the_live_session(pty_driver) -> None:
    """`/debug` renders the console the agent's diagnostics feed."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/debug", "Debug console")
    _assert_no_wire_errors(driver, mark)


def test_proxy_setup_screen_renders(pty_driver) -> None:
    """`/proxy-setup` reads and renders the proxy settings."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/proxy-setup", "Proxy URL for HTTP requests")
    _assert_no_wire_errors(driver, mark)


def test_branch_refusal_names_the_in_process_route(pty_driver) -> None:
    """`/branch` refuses on stdio and names the route that carries it."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _open_command(driver, "/branch", "is not carried over the stdio transport")
    assert "CHARTREUX_TRANSPORT=in_process" in driver.text()[mark:]


def test_shell_command_runs_in_the_spawned_tree(pty_driver) -> None:
    """A `!` shell command runs in the spawned process tree and renders."""
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = _submit(driver, "!echo shell-probe-pty")
    driver.wait_for("shell-probe-pty", 60)
    _assert_no_wire_errors(driver, mark)


def test_the_spawned_root_agent_proves_the_stdio_topology(pty_driver) -> None:
    """The TUI's tree holds a separate root-agent process, not an in-process one."""
    driver = pty_driver
    _settle_to_prompt(driver)
    driver.wait_for_root_agent(30)


#: The banner a refused image send renders. The harness config pins the model to
#: `supports_images = false`, so preparing such a prompt is refused over the wire
#: before a turn starts: the deterministic failed send these nodes replay.
_FAILED_SEND_BANNER = "does not support images"


def _write_attachment(driver) -> str:
    """Write an attachable image into the spawned cwd; return its mention token.

    A relative token keeps the drafted line short enough to render whole: the
    input row scrolls horizontally to the caret, and an absolute path under the
    gate's basetemp is longer than the row.
    """
    name = "draft-shot.png"
    (driver.workdir / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    return name


def _rendered_since(driver, mark: int, limit: int = 4000) -> str:
    """Rendered text since ``mark`` with whitespace removed.

    Soft wrap breaks a long drafted line at the terminal width, so the order pin
    must not depend on where the wrap fell; the rendered order itself does.
    """
    return "".join(driver.tail(mark, limit).split())


def _refusal_restored(driver, mark: int, attachment: str) -> bool:
    """Whether a frame since ``mark`` shows a refusal banner above the draft.

    The banner and the input row render in one frame, banner first, and the
    draft is in that frame only once the refusal restored it (a submit clears
    the input first). That pair distinguishes a NEW refusal from the previous
    one's banner being repainted, which a plain "the banner appeared" check
    cannot do.
    """
    window = _rendered_since(driver, mark, 8000)
    banner = window.rfind(_FAILED_SEND_BANNER.replace(" ", ""))
    return banner >= 0 and window.rfind(attachment) > banner


def test_failed_send_restoration_keeps_the_draft_distinct(
    pty_driver, local_mock_provider
) -> None:
    """A refused send comes back as a draft the next keystrokes follow.

    Replays the artifact-derived sequence of ``pty-probe-archive/out/stdio-p0b``
    (design §11): a send is refused, its text and its attachment token are
    restored, the user types, and the merged draft goes straight into a SECOND
    refusal. The artifact's cause was the restoration's caret -- it sat at the
    START of the restored draft, so the next typing landed in FRONT of it and
    the send carried a prompt nobody had written (``probe-2: say hiprobe-1: say
    hi``). The pin is the rendered order, draft first and typing after it, on
    consecutive refusals.

    The draft keeps prose after the mention so the typing does not fuse into the
    attachment's path: a fused token is not a mention at all, and the send would
    be accepted instead of refused.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    attachment = _write_attachment(driver)
    draft = f"restore-probe see @{attachment} tail-sentinel"
    requests_before = len(local_mock_provider.requests)

    mark = _submit(driver, draft)
    driver.wait_until(
        lambda: _refusal_restored(driver, mark, attachment),
        60,
        "the refused send to restore the draft",
    )
    # The refused send is the draft again, attachment token included, and the
    # refusal never reached the model.
    restored = _rendered_since(driver, mark)
    assert "restore-probesee@" in restored
    assert "draft-shot.png" in restored
    assert len(local_mock_provider.requests) == requests_before

    # Typing after the restoration follows the draft instead of prepending.
    driver.send("after-typing")
    driver.wait_until(
        lambda: "tail-sentinelafter-typing" in _rendered_since(driver, mark),
        30,
        "the typing to follow the restored draft",
    )
    assert "after-typingrestore-probe" not in _rendered_since(driver, mark)

    # Consecutive refusal: the merged draft goes out again, comes back again,
    # and again the next typing follows it rather than prepending.
    driver.drain(idle=0.2)
    second = driver.mark()
    driver.send("\r")
    # The submit clears the input; give that frame time to land so the wait
    # cannot be satisfied by the previous screen's banner-over-draft frame.
    time.sleep(0.3)
    driver.wait_until(
        lambda: _refusal_restored(driver, second, attachment),
        60,
        "the second refusal to restore the draft",
    )
    driver.send("last-typing")
    driver.wait_until(
        lambda: "after-typinglast-typing" in _rendered_since(driver, mark),
        30,
        "the second draft to take the next typing after it",
    )
    assert "last-typingrestore-probe" not in _rendered_since(driver, mark)
    assert len(local_mock_provider.requests) == requests_before
    _assert_no_wire_errors(driver, mark)


def test_safety_dangerous_opens_the_ceremony_warning_screen(pty_driver) -> None:
    """`/safety dangerous` opens the non-dismissible warning screen (v0.8.0 A-P1).

    The ceremony renders over the stdio topology with the typed-phrase gate
    visible, no wire refusal fires, and no DANGEROUS banner appears (the
    transition needs the full ceremony, which the pilot tests in
    tests/cli/test_safety_command.py drive through the phrase and confirm
    screens).
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    mark = driver.mark()
    driver.send("/safety dangerous")
    time.sleep(PALETTE_SETTLE_S)
    driver.send("\r")
    time.sleep(PALETTE_SETTLE_S)
    driver.send("\r")
    driver.wait_for("suspends every Chartreux guard", 40)
    driver.wait_for("ENABLE DANGEROUS MODE", 40)
    assert "DANGEROUS MODE ACTIVE" not in driver.text()
    _assert_no_wire_errors(driver, mark)
