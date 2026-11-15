"""Per-transport executed receipts for the capability manifest.

``tests/app_server/capability_manifest.py`` names the node ids that keep each
classification honest, and ``test_capability_manifest.py`` proves every id
resolves to a real test definition. Resolution is not execution: a node that
never ran, ran on the other transport, was skipped, was deselected, or
exercised both transports inside one node is not evidence that a ``carried``
operation crosses to the process that owns its state.

This module records one receipt per executed node -- the node id, its outcome,
and the transport the node actually executed on -- and decides whether a
manifest entry is satisfied. The stamp is taken at the seam that genuinely
establishes a transport (the backend-contract transport fixture, the
memory-root agent that stands in for a spawned stdio child, the ACP transport
launcher). A node that establishes no transport carries the suite's ambient
``in_process`` stamp, which is the route the suite-wide autouse pins.

The decision rule is deliberately strict:

* only a ``passed`` receipt satisfies an entry;
* a ``skipped`` or ``deselected`` receipt never does;
* a node stamped with more than one transport is ``shared`` and satisfies
  neither -- one node is not two receipts;
* a receipt satisfies only the transport it is stamped with, so an
  ``in_process``-only run cannot satisfy the ``stdio`` leg.

Receipts live in memory per worker and, when ``PYTEST_CHARTREUX_RECEIPT_JOURNAL``
names a path, are appended to it as JSON lines so a gate driver can reconcile a
whole xdist run. A gate run exports ``PYTEST_CHARTREUX_RECEIPT_GATE`` with the
required transport; ``pytest_sessionfinish`` then fails the run when an executed
``carried`` entry has no passed receipt on that transport. The names avoid the
``CHARTREUX_`` prefix deliberately: that prefix is the runtime's configuration
namespace, and an unknown ``CHARTREUX_*`` name fails every config read.

Scope is the run's executed surface, not the whole manifest: a node the run
never executed (deselected by a ``-k`` filter, or never collected) is not
judged, so the driver's selection defines coverage -- a gate over a module with
no stdio leg cannot claim it. A journal is scoped to one driver run: the driver
owns journal freshness, and an empty journal under a gate fails closed instead
of passing vacuously.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading

from chartreux.app_server.transport import IN_PROCESS_TRANSPORT, STDIO_TRANSPORT

RECEIPT_TRANSPORTS = (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)

PASSED = "passed"
FAILED = "failed"
SKIPPED = "skipped"
DESELECTED = "deselected"

SEAM_STAMPED = "seam"
AMBIENT_STAMPED = "ambient"
UNSTAMPED = "none"

JOURNAL_ENV_VAR = "PYTEST_CHARTREUX_RECEIPT_JOURNAL"
GATE_ENV_VAR = "PYTEST_CHARTREUX_RECEIPT_GATE"


def base_node_id(node_id: str) -> str:
    """The node id without its parameter suffix, so a leg matches its entry."""
    return node_id.partition("[")[0]


@dataclass(frozen=True, slots=True)
class Receipt:
    """One executed node and the transport it actually ran on."""

    node_id: str
    outcome: str
    transport: str | None
    stamped: str
    shared: bool = False

    @property
    def base(self) -> str:
        """The unparameterized node id this receipt belongs to."""
        return base_node_id(self.node_id)

    def satisfies(self, tests: Iterable[str], transport: str) -> bool:
        """Whether this receipt is evidence for ``transport`` of ``tests``."""
        if self.shared or self.outcome != PASSED or self.transport != transport:
            return False
        return self.base in {base_node_id(node) for node in tests}

    def as_dict(self) -> dict[str, object]:
        """The JSON-lines shape written to the journal."""
        return {
            "node_id": self.node_id,
            "outcome": self.outcome,
            "transport": self.transport,
            "stamped": self.stamped,
            "shared": self.shared,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Receipt:
        """Rebuild a receipt from its journal line."""
        transport = payload.get("transport")
        return cls(
            node_id=str(payload["node_id"]),
            outcome=str(payload["outcome"]),
            transport=transport if isinstance(transport, str) else None,
            stamped=str(payload.get("stamped", UNSTAMPED)),
            shared=bool(payload.get("shared", False)),
        )


class ReceiptJournal:
    """The receipts one run recorded, and the decision made over them."""

    def __init__(self, receipts: Iterable[Receipt] = ()) -> None:
        self._receipts = tuple(receipts)

    def __iter__(self):
        return iter(self._receipts)

    def __len__(self) -> int:
        return len(self._receipts)

    @property
    def receipts(self) -> tuple[Receipt, ...]:
        """Every receipt in the journal, in arrival order."""
        return self._receipts

    def executed_nodes(self) -> frozenset[str]:
        """Every base node this run actually executed (a deselection never ran)."""
        return frozenset(
            receipt.base for receipt in self._receipts if receipt.outcome != DESELECTED
        )

    def satisfying(self, tests: Iterable[str], transport: str) -> tuple[Receipt, ...]:
        """The receipts that are evidence for ``transport`` of ``tests``."""
        return tuple(
            receipt for receipt in self._receipts if receipt.satisfies(tests, transport)
        )

    def satisfies(self, tests: Iterable[str], transport: str) -> bool:
        """Whether any recorded receipt is evidence for ``transport``."""
        return bool(self.satisfying(tests, transport))


class ReceiptRecorder:
    """Accumulates per-node transport stamps and finalizes executed receipts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: str | None = None
        self._stamps: dict[str, list[str]] = {}
        self._phases: dict[str, dict[str, str]] = {}
        self._receipts: list[Receipt] = []

    def begin_item(self, node_id: str) -> None:
        """Start recording one item: its stamps begin empty, its phases unset."""
        with self._lock:
            self._current = node_id
            self._stamps[node_id] = []
            self._phases[node_id] = {}

    def stamp_transport(self, transport: str) -> None:
        """Record that the running node established ``transport`` at its seam."""
        if transport not in RECEIPT_TRANSPORTS:
            raise ValueError(f"unknown transport stamp: {transport!r}")
        with self._lock:
            node_id = self._current
            if node_id is None:
                return
            stamps = self._stamps.setdefault(node_id, [])
            if transport not in stamps:
                stamps.append(transport)

    def record_phase(self, node_id: str, when: str, outcome: str) -> None:
        """Record one report phase, finalizing the receipt at teardown.

        Only the process that ran the item records it: under xdist the
        controller also receives every worker's reports, and a receipt minted
        there would stamp a node this process never executed.
        """
        with self._lock:
            phases = self._phases.get(node_id)
            if phases is None:
                return
            phases[when] = outcome
            if when != "teardown":
                return
            receipt = self._finalize_locked(node_id)
            if receipt is None:
                return
            self._receipts.append(receipt)
            self._append_locked(receipt)
            if self._current == node_id:
                self._current = None

    def record_deselected(self, node_id: str) -> None:
        """Record a collected-but-deselected node: never evidence."""
        receipt = Receipt(
            node_id=node_id, outcome=DESELECTED, transport=None, stamped=UNSTAMPED
        )
        with self._lock:
            self._receipts.append(receipt)
            self._append_locked(receipt)

    def current_stamps(self) -> tuple[str, ...]:
        """The transports the running node has stamped so far."""
        with self._lock:
            node_id = self._current
            if node_id is None:
                return ()
            return tuple(self._stamps.get(node_id, ()))

    def current_transport(self) -> str | None:
        """The transport the running node is stamped with (None when shared)."""
        transport, shared = _resolve_stamp(self.current_stamps())
        return None if shared else transport

    @property
    def receipts(self) -> tuple[Receipt, ...]:
        """Every receipt finalized so far in this process."""
        with self._lock:
            return tuple(self._receipts)

    def journal(self) -> ReceiptJournal:
        """A journal over the receipts finalized so far."""
        return ReceiptJournal(self.receipts)

    def _finalize_locked(self, node_id: str) -> Receipt | None:
        stamps = self._stamps.pop(node_id, [])
        phases = self._phases.pop(node_id, {})
        outcome = _resolve_outcome(phases)
        if outcome is None:
            return None
        transport, shared = _resolve_stamp(stamps)
        stamped = SEAM_STAMPED if stamps else AMBIENT_STAMPED
        return Receipt(
            node_id=node_id,
            outcome=outcome,
            transport=transport,
            stamped=stamped,
            shared=shared,
        )

    def _append_locked(self, receipt: Receipt) -> None:
        path = journal_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(receipt.as_dict(), sort_keys=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def _resolve_outcome(phases: Mapping[str, str]) -> str | None:
    if not phases:
        return None
    if FAILED in phases.values():
        return FAILED
    if phases.get("setup") == SKIPPED or phases.get("call") == SKIPPED:
        return SKIPPED
    if phases.get("call") == PASSED:
        return PASSED
    return SKIPPED


def _resolve_stamp(stamps: Sequence[str]) -> tuple[str | None, bool]:
    """The single transport a node ran on, or a shared node that ran on several."""
    unique = tuple(dict.fromkeys(stamps))
    if len(unique) > 1:
        return (None, True)
    if unique:
        return (unique[0], False)
    return (IN_PROCESS_TRANSPORT, False)


RECORDER = ReceiptRecorder()


def begin_item(node_id: str) -> None:
    """Hook entry point: a new item starts recording."""
    RECORDER.begin_item(node_id)


def stamp_transport(transport: str) -> None:
    """Hook entry point: the running item establishes ``transport``."""
    RECORDER.stamp_transport(transport)


def record_phase(node_id: str, when: str, outcome: str) -> None:
    """Hook entry point: one report phase arrived for an item."""
    RECORDER.record_phase(node_id, when, outcome)


def record_deselected(node_id: str) -> None:
    """Hook entry point: an item was collected and then deselected."""
    RECORDER.record_deselected(node_id)


def current_transport() -> str | None:
    """The transport stamp of the test that is running right now."""
    return RECORDER.current_transport()


def current_stamps() -> tuple[str, ...]:
    """The transport stamps the running test has recorded so far."""
    return RECORDER.current_stamps()


def journal() -> ReceiptJournal:
    """A journal over this process's finalized receipts."""
    return RECORDER.journal()


def journal_path() -> Path | None:
    """The journal destination the driver named, if any."""
    value = os.environ.get(JOURNAL_ENV_VAR)
    return Path(value) if value else None


def gate_transport() -> str | None:
    """The transport a gate run requires, if the driver named one."""
    return os.environ.get(GATE_ENV_VAR) or None


def read_journal(path: Path) -> ReceiptJournal:
    """Read a journal written by any number of workers."""
    if not path.is_file():
        return ReceiptJournal()
    receipts = [
        Receipt.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return ReceiptJournal(receipts)


def receipt_requirements() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Every entry that needs executed receipts: carried entries and ACP pins."""
    from tests.app_server.capability_manifest import ACP_EXTENSIONS, CARRIED, MANIFEST

    requirements = [
        (entry.method, entry.tests) for entry in MANIFEST if entry.kind == CARRIED
    ]
    requirements.extend(
        (f"acp:{extension.name}", extension.tests) for extension in ACP_EXTENSIONS
    )
    return tuple(requirements)


def reconcile(journal_: ReceiptJournal, *, transport: str) -> tuple[str, ...]:
    """Failures for one gate transport, scoped to the nodes this run executed.

    An entry whose cited test was never executed in this run is out of scope:
    the gate's selection, not this function, decides which surface is proven.
    An entry that was executed must carry a passed, non-shared receipt for
    ``transport``; a skip, a failure, a shared node or an other-transport
    execution is not evidence.
    """
    executed = journal_.executed_nodes()
    failures: list[str] = []
    for label, tests in receipt_requirements():
        cited = {base_node_id(node) for node in tests}
        if not cited & executed:
            continue
        if journal_.satisfies(tests, transport):
            continue
        recorded = [receipt.outcome for receipt in journal_ if receipt.base in cited]
        failures.append(
            f"{label}: no passed {transport} receipt among {sorted(cited)} "
            f"(recorded: {sorted(set(recorded)) or ['nothing']})"
        )
    return tuple(failures)


def gate_failures() -> tuple[str, ...]:
    """The receipt gate's verdict for this run, fail-closed on a broken setup."""
    transport = gate_transport()
    if transport is None:
        return ()
    if transport not in RECEIPT_TRANSPORTS:
        return (f"{GATE_ENV_VAR}={transport!r} is not one of {RECEIPT_TRANSPORTS}",)
    path = journal_path()
    if path is None:
        return (
            f"{GATE_ENV_VAR} is set but {JOURNAL_ENV_VAR} names no journal; "
            "an unrecorded gate cannot be reconciled",
        )
    recorded = read_journal(path)
    failures = list(reconcile(recorded, transport=transport))
    cited = frozenset(
        base_node_id(node) for _, tests in receipt_requirements() for node in tests
    )
    if not recorded.satisfies(cited, transport):
        failures.insert(
            0,
            f"receipt journal {path} holds no passed {transport} receipt for a "
            "manifest-cited node; the gate cannot pass vacuously",
        )
    return tuple(failures)
