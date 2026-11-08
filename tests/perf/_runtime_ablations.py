"""Ablation harness for the runtime cells (v0.7.5 stage 2, WP5a + WP5b).

Five ablations are landed here — ``transport_object`` and ``integrity_stub``
(WP5a), plus ``markdown_no_render``, ``gc_disabled`` and ``source_coalesce``
(WP5b) — applied on top of the WP3b headless scenario machinery, the WP4
adapter definitions and the production streaming widget. r12 pins the ablation
family as opt-in, ONE AT A TIME, proof-of-activation and non-production: an
ablation exists to price a component's contribution to a cell, never to become
a cell of its own. This module makes that shape mechanical instead of
documentary:

* **Declaration.** :data:`ABLATION_SPECS` carries each ablation's pinned
  interpretation and equivalence standing. ``transport_object`` is a
  ``compound_upper_bound``, ``integrity_stub`` runs on ``disposable_data``,
  ``markdown_no_render`` removes the renderer, ``gc_disabled`` removes
  automatic collection and ``source_coalesce`` coalesces the streaming source
  deltas. Only ``source_coalesce`` asserts equivalence — final text and turn
  ordering, the property S1-E(a) has to preserve before delta coalescing may
  re-enter production — and the declarations are asserted
  (:func:`assert_ablation_declaration`) rather than assumed, so a drifted
  declaration fails the contract instead of silently gating a number.
* **One at a time.** :data:`ABLATION_COMBINATIONS` is ``None`` (r12
  ``combinations: null``) and :func:`assert_single_ablation` refuses an empty or
  multi-ablation selection.
* **Proof of activation.** Each bracket counts what it actually removed: the
  ``transport_object`` run reports the envelopes the degenerate object carried
  and the serializations the real transport never performed; the
  ``integrity_stub`` run reports the disposable messages it installed, the
  digest that proves they are not the committed fixture, and the integrity
  boundary that would have refused them; ``markdown_no_render`` reports the
  text that reached the write path and the zero characters the widget
  rendered; ``gc_disabled`` reports the collections that ran inside its
  window; ``source_coalesce`` reports the deltas it received against the
  writes it emitted.
* **Neutrality where it is real.** The transport ablation preserves the
  serialized event stream — same envelopes, same order — because it removes
  cost, not events. The integrity stub never touches the committed fixture.
  The coalescing ablation preserves the final text of every turn and the order
  the turns land in; that is the one equivalence an ablation here may assert.
  Those are the properties asserted; the cost difference is recorded as an
  observation and is never gated.
* **Restoration.** Every bracket restores the exact prior objects on exit and
  states it in the record, because an ablation that leaked into the next cell
  would poison every later run.

Nothing here changes production behaviour: the brackets swap test-visible
objects or the class attribute on the path under observation and put the
originals back, and the fixture data the ``integrity_stub`` ablation installs
is disposable by construction.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
import gc
from types import ModuleType
from typing import Any, cast

from chartreux.app_server import transport as transport_module
from chartreux.cli.textual_ui.widgets.messages import StreamingMessageBase
from chartreux.core.llm_models import LLMMessage, Role
from tests.perf._runtime_fixture import (
    FIXTURE_HASH,
    build_fixture,
    fixture_digest,
    validate_fixture,
)
from tests.perf._runtime_scenario import HeadlessScenario, _CollectingClient

ABLATION_TRANSPORT_OBJECT = "transport_object"
ABLATION_INTEGRITY_STUB = "integrity_stub"
ABLATION_MARKDOWN_NO_RENDER = "markdown_no_render"
ABLATION_GC_DISABLED = "gc_disabled"
ABLATION_SOURCE_COALESCE = "source_coalesce"
ABLATION_IDS: tuple[str, ...] = (
    ABLATION_TRANSPORT_OBJECT,
    ABLATION_INTEGRITY_STUB,
    ABLATION_MARKDOWN_NO_RENDER,
    ABLATION_GC_DISABLED,
    ABLATION_SOURCE_COALESCE,
)
ABLATION_SCHEMA = "chartreux.runtime_ablation.v1"

#: r12: ablations are one at a time, and the report's ``combinations`` field is
#: pinned to null — a compound row would split an unknown share between two
#: components and could never be attributed.
ABLATION_APPLICATION_POLICY = "one_at_a_time"
ABLATION_COMBINATIONS: None = None

#: The equivalence standing an ablation carries when an ablated run is not a
#: valid stand-in for a gated cell: no claim is made.
NOT_ASSERTED = "not_asserted"

#: The one equivalence an ablation MAY assert (plan §3 WP5b): the coalesced run
#: reproduces the per-delta run's final text, in the same order.
FINAL_TEXT_ORDERING = "final_text_ordering"

#: ``integrity_stub`` geometry: small, seeded, and cheap enough that the cell's
#: cost stays dominated by the code path under observation.
INTEGRITY_STUB_MESSAGES = 6
INTEGRITY_STUB_CHARS = 120

_SAMPLE_PRECISION = 3


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One ablation's pinned declaration, carried into every record it produces."""

    id: str
    interpretation: str
    equivalence: str
    equivalence_reason: str
    declaration: str
    observed_only: bool
    production: bool

    def record(self) -> dict[str, Any]:
        return {
            "schema": ABLATION_SCHEMA,
            "ablation": self.id,
            "interpretation": self.interpretation,
            "equivalence": self.equivalence,
            "equivalence_reason": self.equivalence_reason,
            "declaration": self.declaration,
            "observed_only": self.observed_only,
            "production": self.production,
            "application": ABLATION_APPLICATION_POLICY,
            "combinations": ABLATION_COMBINATIONS,
        }


ABLATION_SPECS: dict[str, AblationSpec] = {
    ABLATION_TRANSPORT_OBJECT: AblationSpec(
        id=ABLATION_TRANSPORT_OBJECT,
        interpretation="compound_upper_bound",
        equivalence=NOT_ASSERTED,
        equivalence_reason="compound_upper_bound",
        declaration=(
            "transport_object (interpretation: compound_upper_bound;"
            " equivalence NOT asserted)"
        ),
        observed_only=True,
        production=False,
    ),
    ABLATION_INTEGRITY_STUB: AblationSpec(
        id=ABLATION_INTEGRITY_STUB,
        interpretation="disposable_data",
        equivalence=NOT_ASSERTED,
        equivalence_reason="disposable_data",
        declaration="integrity_stub (disposable data; equivalence not asserted)",
        observed_only=True,
        production=False,
    ),
    ABLATION_MARKDOWN_NO_RENDER: AblationSpec(
        id=ABLATION_MARKDOWN_NO_RENDER,
        interpretation="render_component",
        equivalence=NOT_ASSERTED,
        equivalence_reason="rendering_removed",
        declaration=(
            "markdown_no_render (rendered output removed; equivalence NOT asserted)"
        ),
        observed_only=True,
        production=False,
    ),
    ABLATION_GC_DISABLED: AblationSpec(
        id=ABLATION_GC_DISABLED,
        interpretation="collector_disabled",
        equivalence=NOT_ASSERTED,
        equivalence_reason="process_state",
        declaration=(
            "gc_disabled (automatic collection disabled; equivalence NOT asserted)"
        ),
        observed_only=True,
        production=False,
    ),
    ABLATION_SOURCE_COALESCE: AblationSpec(
        id=ABLATION_SOURCE_COALESCE,
        interpretation="delta_coalescing",
        equivalence=FINAL_TEXT_ORDERING,
        equivalence_reason="final_text_ordering",
        declaration=(
            "source_coalesce (delta coalescing; final-text/ordering equivalence"
            " ASSERTED)"
        ),
        observed_only=True,
        production=False,
    ),
}

#: The equivalence claim each ablation is PINNED to. Only ``source_coalesce``
#: may assert one — every other bracket removes a component without reproducing
#: its output, so a claim there would turn a cost estimate into a false
#: equivalence, and a weaker claim on the coalescing bracket would understate
#: what S1-E(a) has to preserve.
ABLATION_EQUIVALENCE: dict[str, str] = {
    ABLATION_TRANSPORT_OBJECT: NOT_ASSERTED,
    ABLATION_INTEGRITY_STUB: NOT_ASSERTED,
    ABLATION_MARKDOWN_NO_RENDER: NOT_ASSERTED,
    ABLATION_GC_DISABLED: NOT_ASSERTED,
    ABLATION_SOURCE_COALESCE: FINAL_TEXT_ORDERING,
}


def ablation_spec(ablation: str) -> AblationSpec:
    """Return one ablation's pinned declaration."""
    try:
        return ABLATION_SPECS[ablation]
    except KeyError:
        raise ValueError(f"unknown ablation: {ablation}") from None


def assert_ablation_declaration(ablation: AblationSpec | str) -> AblationSpec:
    """Assert one ablation's declaration is the pinned one, and return it.

    An ablation that claimed a gated quantity, production status or an
    equivalence it is not pinned to would quietly turn into a measurement cell;
    refusing the declaration keeps that from happening through a well-meant
    edit.
    """
    spec = ablation_spec(ablation) if isinstance(ablation, str) else ablation
    if ABLATION_SPECS.get(spec.id) != spec:
        raise AssertionError(f"{spec.id} declaration drifted from the pinned one")
    if spec.production:
        raise AssertionError(f"{spec.id} claims to change production code")
    if not spec.observed_only:
        raise AssertionError(f"{spec.id} claims to gate a quantity")
    pinned_equivalence = ABLATION_EQUIVALENCE[spec.id]
    if spec.equivalence != pinned_equivalence:
        raise AssertionError(
            f"{spec.id} claims equivalence {spec.equivalence!r};"
            f" the pinned claim is {pinned_equivalence!r}"
        )
    return spec


def assert_single_ablation(ablations: Sequence[str]) -> str:
    """Assert a one-at-a-time ablation selection and return the selected id."""
    selected = tuple(ablations)
    if len(selected) != 1:
        raise AssertionError(
            f"ablations are {ABLATION_APPLICATION_POLICY} and never combined;"
            f" got {list(selected)}"
        )
    return assert_ablation_declaration(selected[0]).id


# --------------------------------------------------------------------------
# transport_object
# --------------------------------------------------------------------------


class StubJsonRpcTransport:
    """A degenerate, protocol-compatible transport end that does no transport work.

    ``send`` hands the envelope the notification sink already built straight to
    the peer's inbox: no JSON encoding, no queued put through a task, no paired
    close-waiter. It is the object the ``transport_object`` ablation
    substitutes, and it is why that ablation's wall difference is a compound
    UPPER BOUND on the transport component rather than one component's cost.
    """

    def __init__(
        self,
        incoming: asyncio.Queue[dict[str, Any] | None],
        outgoing: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        self._incoming = incoming
        self._outgoing = outgoing
        self.sent: list[dict[str, Any]] = []
        self._closed = False

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("JSON-RPC transport is closed")
        self.sent.append(message)
        self._outgoing.put_nowait(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            message = await self._incoming.get()
            if message is None:
                return
            yield message

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._outgoing.put_nowait(None)


def stub_transport_pair() -> tuple[StubJsonRpcTransport, StubJsonRpcTransport]:
    """Return an ablated pair with the shape of ``memory_transport_pair``."""
    client_inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    server_inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    return (
        StubJsonRpcTransport(client_inbox, server_inbox),
        StubJsonRpcTransport(server_inbox, client_inbox),
    )


class _JsonProxy:
    """The transport module's own ``json`` reference, counting ``dumps`` calls."""

    __slots__ = ("_counter", "_module")

    def __init__(self, module: ModuleType, counter: SerializationCounter) -> None:
        self._module = module
        self._counter = counter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def dumps(self, *args: Any, **kwargs: Any) -> Any:
        self._counter.dumps_calls += 1
        return self._module.dumps(*args, **kwargs)


class SerializationCounter:
    """Counts the JSON serializations the app-server transport module performs.

    The notification sink already serializes each envelope once to record the
    wire bytes, so the transport's own encoding is the SECOND dump r12's "no
    second JSON dump" rule forbids measuring twice. Counting it here is what
    lets a cell state, rather than assume, that the ablated path performed none.
    """

    def __init__(self, module: ModuleType = transport_module) -> None:
        # Typed ``Any`` so the module's own ``json`` reference can be swapped
        # and put back without a cast at every site.
        self._module: Any = module
        self.dumps_calls = 0

    @contextmanager
    def install(self) -> Iterator[SerializationCounter]:
        """Count this module's ``json.dumps`` calls for the duration of the block."""
        original = self._module.json
        self._module.json = _JsonProxy(original, self)
        try:
            yield self
        finally:
            self._module.json = original


@dataclass(slots=True)
class TransportObjectAblation:
    """The ``transport_object`` bracket's live state and its run record."""

    cell: str
    adapter: str
    stub: StubJsonRpcTransport
    client: _CollectingClient
    observations: dict[str, Any] = field(default_factory=dict)
    restored: bool = False

    @property
    def stub_sends(self) -> int:
        return len(self.stub.sent)

    def delivered_methods(self) -> list[str]:
        """Return the method sequence the ablated client end received."""
        return [
            str(envelope["method"])
            for envelope in self.client.delivered
            if isinstance(envelope.get("method"), str)
        ]

    async def settle(self, *, timeout: float = 5.0) -> None:
        """Wait until the ablated client end has received everything sent."""
        await self.client.settle(self.stub_sends, timeout=timeout)

    def record(self) -> dict[str, Any]:
        payload = assert_ablation_declaration(ABLATION_TRANSPORT_OBJECT).record()
        payload.update({
            "cell": self.cell,
            "adapter": self.adapter,
            "activated": self.stub_sends > 0,
            "restored": self.restored,
            "activation": {
                "stub_sends": self.stub_sends,
                "delivered_notifications": len(self.client.delivered),
            },
            "observations": dict(self.observations),
        })
        return payload


@asynccontextmanager
async def ablate_transport_object(
    scenario: HeadlessScenario,
) -> AsyncIterator[TransportObjectAblation]:
    """Substitute ``scenario``'s transport object with a degenerate one.

    The sink keeps building the same serialized envelope and the client end
    keeps receiving it, so the turn still completes and still publishes what a
    real client would see. What disappears is the transport object's own work —
    the JSON encoding, the queued put and its paired close-waiter — which is why
    the wall difference this bracket records is a compound UPPER BOUND (r12) and
    never a gated quantity. Both swapped references are restored on exit.
    """
    sink = scenario.sink
    original_transport = sink._transport
    original_client_end = scenario._client_end
    server_stub, client_stub = stub_transport_pair()
    collector = _CollectingClient(cast(Any, client_stub))
    ablation = TransportObjectAblation(
        cell=scenario.cell,
        adapter=scenario.spec.adapter,
        stub=server_stub,
        client=collector,
    )
    await original_client_end.stop()
    scenario._client_end = collector
    collector.start()
    sink._transport = server_stub  # type: ignore[assignment]
    try:
        yield ablation
    finally:
        sink._transport = original_transport
        await collector.stop()
        await server_stub.close()
        await client_stub.close()
        scenario._client_end = original_client_end
        original_client_end.start()
        ablation.restored = (
            sink._transport is original_transport
            and scenario._client_end is original_client_end
        )


# --------------------------------------------------------------------------
# integrity_stub
# --------------------------------------------------------------------------


def _stub_text(prefix: str, char_count: int) -> str:
    if char_count <= 0:
        raise ValueError("char_count must be positive")
    return (prefix * (char_count // len(prefix) + 1))[:char_count]


def build_integrity_stub(
    message_count: int = INTEGRITY_STUB_MESSAGES,
) -> list[LLMMessage]:
    """Build the disposable history the ``integrity_stub`` ablation installs.

    The data is cheap, seeded and generated locally: it carries NO committed
    fixture lineage, so a run measured on it can never satisfy the eligibility
    rule that binds a run to its cell's fixture hash — which is what makes it
    disposable and its measurements non-equivalent. The shape is structurally
    plain on purpose: the ablation removes the integrity-valid fixture
    requirement, not the transcript's admissibility, and data the turn boundary
    would refuse could not be measured at all.
    """
    if message_count < 2:
        raise ValueError("the stub needs at least one turn")
    return [
        LLMMessage(
            role=Role.user if index % 2 == 0 else Role.assistant,
            content=_stub_text(f"ablation-stub-{index:02d} ", INTEGRITY_STUB_CHARS),
        )
        for index in range(message_count)
    ]


def integrity_verdict(messages: Sequence[LLMMessage]) -> dict[str, Any]:
    """Run the committed fixture's own save-boundary check over ``messages``.

    This is the production boundary (``SessionLogger._check_transcript_persistence``)
    reached through the WP3a entry point, so the verdict is what a real save
    would do with this data.
    """
    try:
        validate_fixture(messages)
    except Exception as error:
        return {"valid": False, "error": type(error).__name__}
    return {"valid": True, "error": None}


@dataclass(slots=True)
class IntegrityStubAblation:
    """The ``integrity_stub`` bracket's live state and its run record."""

    cell: str
    adapter: str
    stub_messages: int
    stub_digest: str
    validation: dict[str, Any]
    observations: dict[str, Any] = field(default_factory=dict)
    installed: bool = False
    restored: bool = False

    def record(self) -> dict[str, Any]:
        payload = assert_ablation_declaration(ABLATION_INTEGRITY_STUB).record()
        payload.update({
            "cell": self.cell,
            "adapter": self.adapter,
            "activated": self.installed and self.stub_messages > 0,
            "restored": self.restored,
            "activation": {
                "stub_messages": self.stub_messages,
                "stub_digest": self.stub_digest,
                "data_disposable": True,
                "fixture_lineage": False,
                "integrity_boundary_skipped": 1,
                "save_boundary_verdict": dict(self.validation),
            },
            "observations": dict(self.observations),
        })
        return payload


@contextmanager
def ablate_integrity_stub(
    scenario: HeadlessScenario, *, messages: Sequence[LLMMessage] | None = None
) -> Iterator[IntegrityStubAblation]:
    """Install disposable history in place of the integrity-valid fixture.

    A real cell runs against the committed 200k fixture and its data passes the
    save-boundary integrity check; this ablation runs against cheap disposable
    data and skips that boundary, which is exactly why its measurements are
    declared non-equivalent. The committed fixture is never touched: only this
    scenario's live transcript is replaced, and its prior contents are restored
    on exit.
    """
    loop = scenario.root_loop
    original = list(loop.messages)
    stub = list(build_integrity_stub() if messages is None else messages)
    ablation = IntegrityStubAblation(
        cell=scenario.cell,
        adapter=scenario.spec.adapter,
        stub_messages=len(stub),
        stub_digest=fixture_digest(stub),
        validation=integrity_verdict(stub),
    )
    loop.messages.reset(stub)
    ablation.installed = True
    try:
        yield ablation
    finally:
        loop.messages.reset(original)
        ablation.restored = [id(message) for message in loop.messages] == [
            id(message) for message in original
        ]


# --------------------------------------------------------------------------
# markdown_no_render
# --------------------------------------------------------------------------


def rendered_markdown(widget: StreamingMessageBase) -> str:
    """Return the markdown source the streaming widget has rendered so far.

    This reads the live ``Markdown`` widget's own source, so it reports what the
    renderer was actually handed rather than what any accumulator buffered.
    """
    markdown = widget._markdown
    return "" if markdown is None else markdown._markdown


class NoRenderStream:
    """A streaming target that accumulates text and renders nothing.

    The production target parses every write as markdown into the live widget;
    this one only appends. That is what makes ``markdown_no_render`` a
    rendering-cost ablation rather than a text one: the delivered text is
    unchanged, the rendered transcript is not.
    """

    def __init__(self) -> None:
        self.writes = 0
        self.chars = 0
        self.text = ""

    async def write(self, text: str) -> None:
        if not text:
            return
        self.writes += 1
        self.chars += len(text)
        self.text += text


@dataclass(slots=True)
class MarkdownNoRenderAblation:
    """The ``markdown_no_render`` bracket's live state and its run record."""

    cell: str
    adapter: str
    stream: NoRenderStream
    rendered_chars: int = 0
    observations: dict[str, Any] = field(default_factory=dict)
    restored: bool = False

    @property
    def writes(self) -> int:
        return self.stream.writes

    def record(self) -> dict[str, Any]:
        payload = assert_ablation_declaration(ABLATION_MARKDOWN_NO_RENDER).record()
        payload.update({
            "cell": self.cell,
            "adapter": self.adapter,
            "activated": self.stream.writes > 0,
            "restored": self.restored,
            "activation": {
                "stream_writes": self.stream.writes,
                "stream_chars": self.stream.chars,
                "markdown_rendered_chars": self.rendered_chars,
            },
            "observations": dict(self.observations),
        })
        return payload


@contextmanager
def ablate_markdown_no_render(
    widget: StreamingMessageBase, *, cell: str, adapter: str
) -> Iterator[MarkdownNoRenderAblation]:
    """Route the widget's streamed text past the markdown renderer.

    Production's write path parses each buffered frame into the live ``Markdown``
    widget; this bracket replaces that routing with a plain accumulator, so the
    frame cadence, the write buffer and the delivered text stay production's
    while the markdown parse and the widget update are gone. The widget under
    observation is passed in so the record can state what it rendered — nothing
    — instead of assuming it. Production's ``_write_chunk`` is restored on exit.
    """
    original = StreamingMessageBase._write_chunk
    stream = NoRenderStream()
    ablation = MarkdownNoRenderAblation(cell=cell, adapter=adapter, stream=stream)

    async def write_without_rendering(
        _message: StreamingMessageBase, text: str
    ) -> None:
        await stream.write(text)

    StreamingMessageBase._write_chunk = write_without_rendering  # type: ignore[method-assign]
    try:
        yield ablation
    finally:
        StreamingMessageBase._write_chunk = original
        ablation.rendered_chars = len(rendered_markdown(widget))
        ablation.restored = StreamingMessageBase._write_chunk is original


# --------------------------------------------------------------------------
# source_coalesce
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SourceCoalesceAblation:
    """The ``source_coalesce`` bracket's live state and its run record."""

    cell: str
    adapter: str
    widget: StreamingMessageBase
    append: Callable[[StreamingMessageBase, str], Awaitable[None]]
    turns: list[str] = field(default_factory=list)
    deltas: int = 0
    writes: int = 0
    buffer: str = ""
    observations: dict[str, Any] = field(default_factory=dict)
    restored: bool = False

    @property
    def coalesced(self) -> bool:
        """Whether the bracket delivered fewer writes than it received deltas."""
        return self.deltas > self.writes

    async def flush(self) -> None:
        """Deliver the buffered deltas as ONE write through production's path.

        This is the ablation's whole mechanism: a turn's deltas reach the render
        path as a single write instead of one per delta. What must not change is
        the text the path ends up with, nor the order the turns land in — the
        bracket records both so a caller can assert that equivalence.
        """
        text, self.buffer = self.buffer, ""
        if not text:
            return
        self.writes += 1
        await self.append(self.widget, text)
        self.turns.append(text)

    def record(self) -> dict[str, Any]:
        payload = assert_ablation_declaration(ABLATION_SOURCE_COALESCE).record()
        payload.update({
            "cell": self.cell,
            "adapter": self.adapter,
            "activated": self.writes > 0 and self.coalesced,
            "restored": self.restored,
            "activation": {
                "deltas": self.deltas,
                "writes": self.writes,
                "coalesced": self.coalesced,
                "turns": len(self.turns),
            },
            "observations": dict(self.observations),
        })
        return payload


@contextmanager
def ablate_source_coalesce(
    widget: StreamingMessageBase, *, cell: str, adapter: str
) -> Iterator[SourceCoalesceAblation]:
    """Coalesce the widget's streamed source deltas into one write per turn.

    S1-E(a) delta coalescing is the change this ablation prices, and the plan
    keeps it out of production until something shows it changes nothing that
    matters. The bracket buffers ``append_content`` calls and hands the buffer
    to production's own ``append_content`` on :meth:`SourceCoalesceAblation.flush`,
    so the render path, the frame pacing and the delivered text stay
    production's; what changes is how many writes the source costs. The bracket
    follows the class-wide write path, so a run that mounts one widget per turn
    points ``widget`` at the current turn's widget before flushing.
    """
    original = StreamingMessageBase.append_content
    ablation = SourceCoalesceAblation(
        cell=cell, adapter=adapter, widget=widget, append=original
    )

    async def coalescing_append(_message: StreamingMessageBase, content: str) -> None:
        if not content:
            return
        ablation.deltas += 1
        ablation.buffer += content

    StreamingMessageBase.append_content = coalescing_append  # type: ignore[method-assign]
    try:
        yield ablation
    finally:
        StreamingMessageBase.append_content = original
        ablation.restored = StreamingMessageBase.append_content is original


# --------------------------------------------------------------------------
# gc_disabled
# --------------------------------------------------------------------------


class CollectionCounter:
    """Counts completed garbage-collection cycles while installed."""

    def __init__(self) -> None:
        self.collections = 0

    def __call__(self, phase: str, info: dict[str, int]) -> None:
        del info
        if phase == "stop":
            self.collections += 1

    @contextmanager
    def install(self) -> Iterator[CollectionCounter]:
        """Count collections for the duration of the block."""
        gc.callbacks.append(self)
        try:
            yield self
        finally:
            gc.callbacks.remove(self)


@dataclass(slots=True)
class GcDisabledAblation:
    """The ``gc_disabled`` bracket's live state and its run record."""

    cell: str
    adapter: str
    was_enabled: bool
    freeze_count: int
    counter: CollectionCounter
    observations: dict[str, Any] = field(default_factory=dict)
    activated: bool = False
    restored: bool = False

    def record(self) -> dict[str, Any]:
        payload = assert_ablation_declaration(ABLATION_GC_DISABLED).record()
        payload.update({
            "cell": self.cell,
            "adapter": self.adapter,
            "activated": self.activated,
            "restored": self.restored,
            "activation": {
                "collector_was_enabled": self.was_enabled,
                "collector_disabled": self.activated,
                "collections_during": self.counter.collections,
                "freeze_count": self.freeze_count,
            },
            "observations": dict(self.observations),
        })
        return payload


@contextmanager
def ablate_gc_disabled(*, cell: str, adapter: str) -> Iterator[GcDisabledAblation]:
    """Run the bracketed workload with automatic garbage collection off.

    GC pauses are one of the long-session pathologies worth pricing, and the
    collection share is only visible when it is removed: the bracket disables
    the collector for its window, counts the collections that actually ran
    inside it, and puts the collector back exactly as it found it — including
    the freeze count, which a wrongly restored bracket would silently change.
    """
    was_enabled = gc.isenabled()
    freeze_count = gc.get_freeze_count()
    counter = CollectionCounter()
    ablation = GcDisabledAblation(
        cell=cell,
        adapter=adapter,
        was_enabled=was_enabled,
        freeze_count=freeze_count,
        counter=counter,
    )
    gc.collect()  # settle the heap first so nothing is mid-cycle in the window
    gc.disable()
    with counter.install():
        ablation.activated = not gc.isenabled()
        try:
            yield ablation
        finally:
            if was_enabled:
                gc.enable()
            ablation.restored = (
                gc.isenabled() == was_enabled and gc.get_freeze_count() == freeze_count
            )


def ablation_observation(
    *, ablation: str, cell: str, **observations: Any
) -> dict[str, Any]:
    """Render one ablation's recorded observation, never a gated comparison.

    The cost difference a bracket measures is reported with its precision and
    its ``gated: False`` marker, so no reader can mistake an upper bound for a
    threshold result.
    """
    spec = assert_ablation_declaration(ablation)
    payload = spec.record()
    payload.update({
        "cell": cell,
        "observed_only": True,
        "gated": False,
        "observations": {
            key: (
                round(value, _SAMPLE_PRECISION) if isinstance(value, float) else value
            )
            for key, value in observations.items()
        },
    })
    return payload


def ablation_wall_observation(
    *, cell: str, real_ms: float, ablated_ms: float
) -> dict[str, Any]:
    """Record the paired wall difference of a bracket as an upper bound."""
    return ablation_observation(
        ablation=ABLATION_TRANSPORT_OBJECT,
        cell=cell,
        real_wall_ms=real_ms,
        ablated_wall_ms=ablated_ms,
        upper_bound_ms=real_ms - ablated_ms,
        samples=2,
    )


def fixture_immutability_probe() -> dict[str, Any]:
    """Report the committed fixture's digest and integrity verdict, unchanged."""
    messages = build_fixture()
    return {
        "fixture_digest": fixture_digest(messages),
        "fixture_hash": FIXTURE_HASH,
        "integrity": integrity_verdict(messages),
    }


__all__ = [
    "ABLATION_APPLICATION_POLICY",
    "ABLATION_COMBINATIONS",
    "ABLATION_EQUIVALENCE",
    "ABLATION_GC_DISABLED",
    "ABLATION_IDS",
    "ABLATION_INTEGRITY_STUB",
    "ABLATION_MARKDOWN_NO_RENDER",
    "ABLATION_SCHEMA",
    "ABLATION_SOURCE_COALESCE",
    "ABLATION_SPECS",
    "ABLATION_TRANSPORT_OBJECT",
    "FINAL_TEXT_ORDERING",
    "INTEGRITY_STUB_CHARS",
    "INTEGRITY_STUB_MESSAGES",
    "NOT_ASSERTED",
    "AblationSpec",
    "CollectionCounter",
    "GcDisabledAblation",
    "IntegrityStubAblation",
    "MarkdownNoRenderAblation",
    "NoRenderStream",
    "SerializationCounter",
    "SourceCoalesceAblation",
    "StubJsonRpcTransport",
    "TransportObjectAblation",
    "ablate_gc_disabled",
    "ablate_integrity_stub",
    "ablate_markdown_no_render",
    "ablate_source_coalesce",
    "ablate_transport_object",
    "ablation_observation",
    "ablation_spec",
    "ablation_wall_observation",
    "assert_ablation_declaration",
    "assert_single_ablation",
    "build_integrity_stub",
    "fixture_immutability_probe",
    "integrity_verdict",
    "rendered_markdown",
    "stub_transport_pair",
]
