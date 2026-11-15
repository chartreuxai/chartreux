"""CI gate for the capability manifest: every registered method, classified.

These nodes are the structural "never again" check from the stdio-parity
design (§6.1): the manifest is keyed by ``SERVER_METHODS`` membership -- the
same tuple dispatch consults -- so a newly registered method fails here until
it is classified and bound to an executable test node, and a renamed or moved
test fails here instead of silently orphaning an entry.

The binding asserted here is *resolvability*: each manifest node id must name a
real test definition in this tree. C5b layers the per-transport executed-receipt
check on top of these ids.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    SERVER_METHODS,
    AppServerResponseError,
    ProtocolErrorCode,
)
from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    STDIO_TRANSPORT,
    memory_transport_pair,
)
from chartreux.core.config.harness_files import init_harness_files_manager
from tests.app_server import executed_receipts
from tests.app_server.capability_manifest import (
    ACP_EXTENSIONS,
    CARRIED,
    CLASSIFICATIONS,
    DELIBERATE_REFUSAL,
    MANIFEST,
    UNSUPPORTED_ON_BOTH,
    MethodEntry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The design's documented deliberate refusals (§4): typed on stdio, alive
#: in-process. A classification change here is a design change, not a tweak.
DOCUMENTED_REFUSALS = {
    "policy/roots/replace",
    "session/fork",
    "session/relocate",
    "session/rewind",
    "session/rewind/read",
}

#: The methods this release documents as having no handler in either topology.
DOCUMENTED_UNSUPPORTED = {
    "narration/summarize",
    "session/title/update",
    "shell/interrupt",
    "shell/run",
}


def _entry(method: str):
    matches = [entry for entry in MANIFEST if entry.method == method]
    assert len(matches) == 1, (
        f"{method}: expected exactly one entry, got {len(matches)}"
    )
    return matches[0]


def _split_test_node(node_id: str) -> tuple[Path, tuple[str, ...]]:
    """Split ``tests/path.py::Class::test_name[params]`` into path and names.

    The parameter suffix is dropped: the base node id names the collected test
    function, and selecting it runs every parameterization.
    """
    path_text, sep, remainder = node_id.partition("::")
    assert sep, f"{node_id}: not a pytest node id"
    names = tuple(part.split("[", 1)[0] for part in remainder.split("::"))
    assert names and all(names), f"{node_id}: empty test name"
    return REPO_ROOT / path_text, names


def _defines_test(path: Path, names: tuple[str, ...]) -> bool:
    """Whether the file defines the named test, at module level or one class deep."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False

    def _defines(body: list[ast.stmt], wanted: tuple[str, ...]) -> bool:
        name = wanted[0]
        rest = wanted[1:]
        for node in body:
            if isinstance(node, ast.ClassDef) and rest:
                if node.name == name and _defines(node.body, rest):
                    return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name and not rest:
                    return True
        return False

    return _defines(tree.body, names)


def _assert_resolvable(node_id: str) -> None:
    path, names = _split_test_node(node_id)
    assert path.is_file(), f"{node_id}: test file does not exist"
    assert path.parts[-1].startswith("test_"), f"{node_id}: not a test module"
    assert names[-1].startswith("test_"), f"{node_id}: not a test function"
    assert _defines_test(path, names), f"{node_id}: test definition not found"


# -- keying and classification ------------------------------------------------


def test_manifest_covers_every_server_method_exactly() -> None:
    """The manifest is keyed by the dispatch tuple itself: no drift either way."""
    manifest_methods = [entry.method for entry in MANIFEST]
    registered = set(SERVER_METHODS)

    missing = sorted(registered - set(manifest_methods))
    unknown = sorted(set(manifest_methods) - registered)
    assert not missing, f"registered methods with no manifest entry: {missing}"
    assert not unknown, f"manifest entries for unregistered methods: {unknown}"


def test_manifest_has_no_duplicate_entries() -> None:
    """A duplicate entry would let one classification quietly shadow another."""
    methods = [entry.method for entry in MANIFEST]
    duplicates = sorted({method for method in methods if methods.count(method) > 1})
    assert not duplicates, f"duplicate manifest entries: {duplicates}"


def test_every_entry_is_classified() -> None:
    """Only the design's four classifications are valid; nothing is left unclassified."""
    assert {entry.kind for entry in MANIFEST} <= set(CLASSIFICATIONS)
    unclassified = [entry.method for entry in MANIFEST if not entry.kind]
    assert not unclassified, f"unclassified entries: {unclassified}"
    assert len(MANIFEST) == len(SERVER_METHODS)


def test_every_entry_documents_an_owner_and_a_rationale() -> None:
    """A classification without an owner and a reason is not a reviewable record."""
    for entry in MANIFEST:
        assert entry.owner.strip(), f"{entry.method}: empty owner"
        assert entry.rationale.strip(), f"{entry.method}: empty rationale"
        assert len(entry.rationale) >= 40, f"{entry.method}: rationale is a label"


def test_refusal_and_unsupported_sets_are_the_documented_ones() -> None:
    """Classification changes here are design changes, so they are pinned."""
    refusals = {entry.method for entry in MANIFEST if entry.kind == DELIBERATE_REFUSAL}
    unsupported = {
        entry.method for entry in MANIFEST if entry.kind == UNSUPPORTED_ON_BOTH
    }
    assert refusals == DOCUMENTED_REFUSALS
    assert unsupported == DOCUMENTED_UNSUPPORTED
    assert not (refusals & unsupported)


def test_narration_summarize_is_classified_explicitly() -> None:
    """The ACP-only method is recorded as such, never counted as parity."""
    entry = _entry("narration/summarize")
    assert entry.kind == UNSUPPORTED_ON_BOTH
    assert entry.acp_only is True
    assert [other.method for other in MANIFEST if other.acp_only] == [
        "narration/summarize"
    ]
    assert "ACP-only" in entry.rationale


# -- manifest-to-executed-node binding ----------------------------------------


def test_every_entry_cites_a_resolvable_test_id() -> None:
    """Every entry names at least one real node; each cited node must resolve."""
    for entry in MANIFEST:
        assert entry.tests, f"{entry.method}: no test id"
    unresolved = [
        node_id
        for entry in MANIFEST
        for node_id in entry.tests
        if not _resolves(node_id)
    ]
    assert not unresolved, f"unresolvable manifest test ids: {unresolved}"


def _resolves(node_id: str) -> bool:
    try:
        _assert_resolvable(node_id)
    except AssertionError:
        return False
    return True


# -- ACP mapping-pin drift ----------------------------------------------------


def test_acp_extensions_match_the_mapping_pin_tables() -> None:
    """The ACP surface is a mapping; the manifest cites the pins that record it."""
    from tests.acp.test_agent_methods import (
        REWIND_APP_SERVER_METHODS,
        REWIND_MAPPING_TEST_IDS,
    )
    from tests.acp.test_log_level import (
        LOG_LEVEL_APP_SERVER_METHODS,
        LOG_LEVEL_MAPPING_TEST_IDS,
    )

    expected_methods = {**LOG_LEVEL_APP_SERVER_METHODS, **REWIND_APP_SERVER_METHODS}
    expected_tests = {**LOG_LEVEL_MAPPING_TEST_IDS, **REWIND_MAPPING_TEST_IDS}

    assert {extension.name for extension in ACP_EXTENSIONS} == set(expected_methods)
    for extension in ACP_EXTENSIONS:
        assert extension.app_server_method == expected_methods[extension.name]
        assert extension.tests == expected_tests[extension.name]
        assert extension.owner.strip() and extension.rationale.strip()


def test_acp_extension_targets_are_registered_and_classified() -> None:
    """A mapping must land on a registered method with the expected standing."""
    registered = set(SERVER_METHODS)
    for extension in ACP_EXTENSIONS:
        target = extension.app_server_method
        if target is None:
            continue
        assert target in registered, f"{extension.name} maps to unregistered {target}"
        kind = _entry(target).kind
        if target.startswith("session/rewind"):
            assert kind == DELIBERATE_REFUSAL
        else:
            assert kind == CARRIED
        _assert_resolvable(extension.tests[0])


# -- explicit unsupported pins ------------------------------------------------


async def _assert_in_process_unsupported(method: str) -> None:
    from tests.conftest import build_test_agent_loop
    from tests.stubs.app_server import (
        attach_test_app_server_session,
        start_test_app_server,
    )

    init_harness_files_manager("user", "project")
    loop = build_test_agent_loop()
    client = start_test_app_server(loop)
    session = await attach_test_app_server_session(client)
    try:
        with pytest.raises(AppServerResponseError) as caught:
            await client.request(method, {"sessionId": session.session_id})
    finally:
        await session.close()
        await client.close()
        await loop.aclose()
    assert caught.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND


async def _assert_stdio_unsupported(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server._runtime import create_harness_server
    from tests.app_server.backend_contract.conftest import install_memory_root_agent
    from tests.conftest import build_test_agent_loop
    from tests.stubs.app_server import attach_test_app_server_session

    init_harness_files_manager("user", "project")
    loop = build_test_agent_loop()
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(
        server_transport, transport_kind=STDIO_TRANSPORT
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    try:
        with pytest.raises(AppServerResponseError) as caught:
            await client.request(method, {"sessionId": session.session_id})
    finally:
        await session.close()
        await client.close()
        await harness.close()
        await loop.aclose()
    assert caught.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED


@pytest.mark.asyncio
async def test_narration_summarize_is_unsupported_on_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACP-only narration answers unknown in-process and uncarried on stdio."""
    await _assert_in_process_unsupported("narration/summarize")
    await _assert_stdio_unsupported("narration/summarize", monkeypatch)


@pytest.mark.asyncio
async def test_shell_run_is_unsupported_on_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No handler exists in either topology, so neither may report success."""
    await _assert_in_process_unsupported("shell/run")
    await _assert_stdio_unsupported("shell/run", monkeypatch)


@pytest.mark.asyncio
async def test_shell_interrupt_is_unsupported_on_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interrupt shares the absent shell carrier on both transports."""
    await _assert_in_process_unsupported("shell/interrupt")
    await _assert_stdio_unsupported("shell/interrupt", monkeypatch)


# -- transport constants stay the two topologies ------------------------------


def test_acp_transports_are_the_two_topologies() -> None:
    """The manifest's ACP ids are built from the shipped transport constants."""
    from tests.app_server.capability_manifest import ACP_TRANSPORTS

    assert ACP_TRANSPORTS == (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)


# -- per-transport executed receipts (C5b) ------------------------------------
#
# The nodes above bind a classification to a resolvable test id. These bind it
# to an executed receipt: a `carried` entry is satisfied only by a node that
# actually ran and passed on the transport in question. The sample entry and
# journal below make the rule independent of what this particular run happens
# to select; the last nodes drive the live machinery and the recorded journal.

_SAMPLE_NODE = "tests/app_server/test_capability_manifest.py::test_sample_carried_node"
_CARRIED_SAMPLE = MethodEntry(
    method="sample/carried",
    kind=CARRIED,
    owner="receipt machinery (test sample)",
    rationale="A synthetic carried entry that exists only to exercise the receipt rule.",
    tests=(_SAMPLE_NODE,),
)

#: A real carried entry, used to prove the gate's scoping against the manifest.
_AGENTS_CANCEL_NODE = (
    "tests/app_server/test_canonical_methods.py::"
    "test_agents_cancel_lifecycle_allow_policy"
)


def _receipt(
    outcome: str,
    transport: str | None,
    *,
    node_id: str = _SAMPLE_NODE,
    shared: bool = False,
) -> executed_receipts.Receipt:
    """One synthetic receipt; ``stamped`` is the seam's own label."""
    return executed_receipts.Receipt(
        node_id=node_id,
        outcome=outcome,
        transport=transport,
        stamped=executed_receipts.SEAM_STAMPED,
        shared=shared,
    )


def _deselected(node_id: str) -> executed_receipts.Receipt:
    return executed_receipts.Receipt(
        node_id=node_id,
        outcome=executed_receipts.DESELECTED,
        transport=None,
        stamped=executed_receipts.UNSTAMPED,
    )


def test_a_passed_receipt_satisfies_only_its_own_transport() -> None:
    """The stamp decides which leg is proven; one receipt is never both."""
    receipts = executed_receipts.ReceiptJournal([
        _receipt(executed_receipts.PASSED, STDIO_TRANSPORT)
    ])
    assert receipts.satisfies(_CARRIED_SAMPLE.tests, STDIO_TRANSPORT)
    assert not receipts.satisfies(_CARRIED_SAMPLE.tests, IN_PROCESS_TRANSPORT)


def test_a_skipped_node_never_satisfies_a_carried_classification() -> None:
    """A skip proves nothing, whatever transport it was collected under."""
    receipts = executed_receipts.ReceiptJournal([
        _receipt(executed_receipts.SKIPPED, STDIO_TRANSPORT)
    ])
    assert not receipts.satisfies(_CARRIED_SAMPLE.tests, STDIO_TRANSPORT)


def test_a_deselected_node_never_satisfies_a_carried_classification() -> None:
    """A deselected leg was collected but never executed, so it is no receipt."""
    receipts = executed_receipts.ReceiptJournal([_deselected(_SAMPLE_NODE)])
    for transport in executed_receipts.RECEIPT_TRANSPORTS:
        assert not receipts.satisfies(_CARRIED_SAMPLE.tests, transport)


def test_a_shared_node_never_satisfies_either_transport() -> None:
    """One node exercising both transports is not evidence for either."""
    receipts = executed_receipts.ReceiptJournal([
        _receipt(executed_receipts.PASSED, None, shared=True)
    ])
    for transport in executed_receipts.RECEIPT_TRANSPORTS:
        assert not receipts.satisfies(_CARRIED_SAMPLE.tests, transport)


def test_an_in_process_only_run_never_satisfies_the_stdio_leg() -> None:
    """The in-process baseline is evidence for in_process and nothing more."""
    receipts = executed_receipts.ReceiptJournal([
        _receipt(executed_receipts.PASSED, IN_PROCESS_TRANSPORT)
    ])
    assert receipts.satisfies(_CARRIED_SAMPLE.tests, IN_PROCESS_TRANSPORT)
    assert not receipts.satisfies(_CARRIED_SAMPLE.tests, STDIO_TRANSPORT)


def test_a_receipt_for_an_uncited_node_never_satisfies_an_entry() -> None:
    """A passing node the entry does not cite is somebody else's evidence."""
    receipts = executed_receipts.ReceiptJournal([
        _receipt(
            executed_receipts.PASSED,
            STDIO_TRANSPORT,
            node_id=(
                "tests/app_server/test_capability_manifest.py::"
                "test_some_other_executed_node"
            ),
        )
    ])
    assert not receipts.satisfies(_CARRIED_SAMPLE.tests, STDIO_TRANSPORT)


def test_parameterized_legs_are_distinct_receipts_for_one_entry() -> None:
    """A transport-parameterized citation matches per leg, never as a pair."""
    tests = (
        "tests/app_server/backend_contract/test_loops.py::"
        "test_loop_carrier_answers_from_the_process_that_holds_the_schedule",
    )
    base = tests[0]
    receipts = executed_receipts.ReceiptJournal([
        _receipt(
            executed_receipts.PASSED,
            IN_PROCESS_TRANSPORT,
            node_id=f"{base}[in_process]",
        ),
        _receipt(executed_receipts.SKIPPED, STDIO_TRANSPORT, node_id=f"{base}[stdio]"),
    ])
    assert receipts.satisfies(tests, IN_PROCESS_TRANSPORT)
    assert not receipts.satisfies(tests, STDIO_TRANSPORT)


def test_the_gate_reconciles_only_the_entries_this_run_executed() -> None:
    """Scope is the run's executed surface: a deselection proves nothing either way."""
    passed = executed_receipts.ReceiptJournal([
        _receipt(executed_receipts.PASSED, STDIO_TRANSPORT, node_id=_AGENTS_CANCEL_NODE)
    ])
    assert executed_receipts.reconcile(passed, transport=STDIO_TRANSPORT) == ()

    skipped = executed_receipts.ReceiptJournal([
        _receipt(
            executed_receipts.SKIPPED, STDIO_TRANSPORT, node_id=_AGENTS_CANCEL_NODE
        )
    ])
    failures = executed_receipts.reconcile(skipped, transport=STDIO_TRANSPORT)
    assert any(failure.startswith("agents/cancel:") for failure in failures)

    deselected = executed_receipts.ReceiptJournal([_deselected(_AGENTS_CANCEL_NODE)])
    assert executed_receipts.reconcile(deselected, transport=STDIO_TRANSPORT) == ()

    assert (
        executed_receipts.reconcile(
            executed_receipts.ReceiptJournal(), transport=STDIO_TRANSPORT
        )
        == ()
    )


def test_the_journal_round_trips_through_json_lines(tmp_path: Path) -> None:
    """What a gate driver reads back is what the worker recorded."""
    path = tmp_path / "receipts.jsonl"
    receipt = _receipt(executed_receipts.PASSED, STDIO_TRANSPORT)
    path.write_text(json.dumps(receipt.as_dict()) + "\n", encoding="utf-8")
    rebuilt = executed_receipts.read_journal(path)
    assert rebuilt.receipts == (receipt,)


def test_an_ordinary_node_carries_the_ambient_in_process_stamp() -> None:
    """The suite autouse pins in_process; a node that stamps nothing runs there."""
    assert executed_receipts.current_transport() == IN_PROCESS_TRANSPORT


@pytest.mark.asyncio
async def test_the_stdio_seam_stamps_the_running_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installing the memory-wire root is the seam that makes a node stdio."""
    from tests.app_server.backend_contract.conftest import install_memory_root_agent
    from tests.conftest import build_test_agent_loop

    loop = build_test_agent_loop()
    try:
        install_memory_root_agent(monkeypatch, loop)
        assert executed_receipts.current_transport() == STDIO_TRANSPORT
    finally:
        await loop.aclose()


def test_acp_extension_pins_name_one_literal_id_per_transport() -> None:
    """An ACP mapping is pinned per transport, not by one shared node."""
    for extension in ACP_EXTENSIONS:
        stamped = [
            transport
            for transport in executed_receipts.RECEIPT_TRANSPORTS
            if any(
                node_id.endswith(f"[{extension.name}-{transport}]")
                for node_id in extension.tests
            )
        ]
        assert sorted(stamped) == sorted(executed_receipts.RECEIPT_TRANSPORTS), (
            extension.name
        )


def test_an_acp_pin_that_stopped_executing_fails_its_manifest_binding() -> None:
    """A mapping pin is evidence only while executed and transport-stamped."""
    extension = next(item for item in ACP_EXTENSIONS if item.name == "logLevel/write")
    healthy = executed_receipts.ReceiptJournal([
        _receipt(
            executed_receipts.PASSED, IN_PROCESS_TRANSPORT, node_id=extension.tests[0]
        ),
        _receipt(executed_receipts.PASSED, STDIO_TRANSPORT, node_id=extension.tests[1]),
    ])
    assert executed_receipts.reconcile(healthy, transport=STDIO_TRANSPORT) == ()

    drifted = executed_receipts.ReceiptJournal([
        _receipt(
            executed_receipts.PASSED, IN_PROCESS_TRANSPORT, node_id=extension.tests[0]
        ),
        _receipt(
            executed_receipts.SKIPPED, STDIO_TRANSPORT, node_id=extension.tests[1]
        ),
    ])
    failures = executed_receipts.reconcile(drifted, transport=STDIO_TRANSPORT)
    assert any(failure.startswith("acp:logLevel/write:") for failure in failures)


def test_a_recorded_receipt_always_carries_a_transport_stamp() -> None:
    """The recovery promise: every journal line says which transport it proves."""
    path = executed_receipts.journal_path()
    if path is None:
        pytest.skip("no receipt journal was driven for this run")
    recorded = executed_receipts.read_journal(path)
    assert recorded.receipts, f"{path}: the journal recorded no receipts"
    for receipt in recorded.receipts:
        if receipt.shared or receipt.outcome == executed_receipts.DESELECTED:
            assert receipt.transport is None, receipt
            continue
        assert receipt.transport in executed_receipts.RECEIPT_TRANSPORTS, receipt
        assert receipt.stamped in {
            executed_receipts.SEAM_STAMPED,
            executed_receipts.AMBIENT_STAMPED,
        }, receipt


def test_a_driven_gate_reconciles_the_recorded_journal() -> None:
    """The gate's verdict over a produced journal, for a driver that wants it."""
    path = executed_receipts.journal_path()
    gate = executed_receipts.gate_transport()
    if path is None or gate is None:
        pytest.skip("no receipt gate was driven for this run")
    assert gate in executed_receipts.RECEIPT_TRANSPORTS, gate
    recorded = executed_receipts.read_journal(path)
    assert recorded.receipts, f"{path}: the journal recorded no receipts"
    failures = executed_receipts.gate_failures()
    assert not failures, "\n".join(failures)
