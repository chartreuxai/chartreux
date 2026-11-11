"""Seeded replay/normalization skeleton and the common component adapter (WP6 [6A]).

Both topologies host the same component: :func:`run_component_loop` talks over a
:class:`Channel`, and the in-process adapter swaps the child's JSON-lines pipes
for an in-memory queue pair. Component code is therefore byte-identical between
topologies, which is the property the CI escape gate depends on
(design.md:604-606); [6C] later substitutes the real presentation components
behind the same contract.

The seeded script and :func:`normalize_frames` are the replay skeleton: a
trace is a seed plus a script, run through whichever topology, normalized to
drop process-volatile fields, and compared by digest. This module lands the
skeleton only — it carries no presentation trace.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time
from typing import Protocol, TextIO, cast

from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    parse_transport_selector,
)
from tests.differential import egress

REPLAY_ROOT = Path(__file__).resolve().parents[2]

_REPLAY_CHILD_MODULE = "tests.differential.replay_child"
_TURN = "turn"
_DELTA = "delta"
_TURN_END = "turn_end"
_VOLATILE_FIELDS = frozenset({"clock"})
_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
)


class ReplayError(RuntimeError):
    """A replay transport or component contract violation."""


COMPONENT_ECHO = "echo"
COMPONENT_ROOT_ORACLE = "root_oracle"
COMPONENT_PRESENTATION = "presentation"
COMPONENT_CHILD_CRASH = "child_crash"
COMPONENT_LIFECYCLE = "lifecycle"
COMPONENT_PROBES = "probes"

#: The two selectors the both-mode gate runs every trace under (WP6 [6D]).
TRANSPORT_MODES = (STDIO_TRANSPORT, IN_PROCESS_TRANSPORT)


def component_entry(name: str) -> Callable[[Channel], None]:
    """Resolve a hostable component by name, identically in both topologies.

    The name travels to the spawned child as a single argv element, so the
    component a trace exercises is the one the driver asked for, never a
    build-time default.
    """
    if name == COMPONENT_ECHO:
        return run_component_loop
    if name == COMPONENT_ROOT_ORACLE:
        # Deferred: the root oracle loads the agent-loop and config layers,
        # which the presentation skeleton must not pull into every trace.
        from tests.differential import root_oracle

        return root_oracle.run_root_oracle_loop
    if name == COMPONENT_PRESENTATION:
        # Deferred: presentation traces load the event/projection models.
        from tests.differential import presentation

        return presentation.run_presentation_loop
    if name == COMPONENT_CHILD_CRASH:
        # Deferred: the crash oracle spawns its own marker-writing children.
        from tests.differential import crash_oracle

        return crash_oracle.run_child_crash_loop
    if name == COMPONENT_LIFECYCLE:
        # Deferred: the shutdown/lease nodes build a session runtime registry.
        from tests.differential import lifecycle

        return lifecycle.run_lifecycle_loop
    if name == COMPONENT_PROBES:
        # Deferred: the probe reports this process's transport and egress state.
        from tests.differential import probes

        return probes.run_probe_loop
    raise ReplayError(f"unknown replay component: {name!r}")


@contextmanager
def transport_mode(mode: str) -> Iterator[None]:
    """Run the surrounding trace with ``mode`` as the selected transport (WP6 [6D]).

    The selector is published in the environment, so the spawned topology's
    child observes the same selection the in-process component does: the mode
    is part of the trace's run configuration, not of its topology. An unusable
    selector fails the trace here rather than inside a component.
    """
    parse_transport_selector(mode)
    previous = os.environ.get(TRANSPORT_ENV_VAR)
    os.environ[TRANSPORT_ENV_VAR] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TRANSPORT_ENV_VAR, None)
        else:
            os.environ[TRANSPORT_ENV_VAR] = previous


@dataclass(frozen=True)
class ReplayScript:
    """A seeded, deterministic user-turn sequence."""

    seed: int
    turns: tuple[str, ...]


def _word(seed: int, index: int) -> str:
    # A local mixed-linear generator keeps seeded scripts independent of any
    # third-party RNG implementation, so a library bump cannot change a trace.
    mixed = (seed + 1) * 6_364_136_223_846_793_005 + (index + 1) * 1_442_695_040
    return _WORDS[(mixed >> 33) % len(_WORDS)]


def build_replay_script(seed: int, *, turns: int = 3, words: int = 3) -> ReplayScript:
    """Build the deterministic script for ``seed``."""
    prompts = tuple(
        " ".join(_word(seed, turn * words + offset) for offset in range(words))
        for turn in range(turns)
    )
    return ReplayScript(seed=seed, turns=prompts)


class Channel(Protocol):
    """The message contract both topologies satisfy."""

    def send(self, message: Mapping[str, object]) -> None: ...

    def receive(self) -> dict[str, object] | None:
        """Return the next message, or ``None`` at end of stream."""
        ...

    def close(self) -> None: ...


class InProcessChannel:
    """One end of an in-memory channel pair."""

    def __init__(
        self,
        inbox: Queue[dict[str, object] | None],
        outbox: Queue[dict[str, object] | None],
        *,
        timeout: float,
    ) -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._timeout = timeout

    def send(self, message: Mapping[str, object]) -> None:
        self._outbox.put(dict(message))

    def receive(self) -> dict[str, object] | None:
        try:
            return self._inbox.get(timeout=self._timeout)
        except Empty as exc:
            raise ReplayError("timed out waiting for the in-process component") from exc

    def close(self) -> None:
        self._outbox.put(None)


def in_process_channel_pair(
    *, timeout: float = 5.0
) -> tuple[InProcessChannel, InProcessChannel]:
    """Two ends of one in-memory channel; the first is the driver end."""
    driver_inbox: Queue[dict[str, object] | None] = Queue()
    component_inbox: Queue[dict[str, object] | None] = Queue()
    return (
        InProcessChannel(inbox=driver_inbox, outbox=component_inbox, timeout=timeout),
        InProcessChannel(inbox=component_inbox, outbox=driver_inbox, timeout=timeout),
    )


class JsonLineChannel:
    """A channel over the JSON-lines pipes of a spawned component process."""

    def __init__(self, reader: TextIO, writer: TextIO) -> None:
        self._reader = reader
        self._writer = writer

    def send(self, message: Mapping[str, object]) -> None:
        self._writer.write(json.dumps(dict(message), sort_keys=True) + "\n")
        self._writer.flush()

    def receive(self) -> dict[str, object] | None:
        line = self._reader.readline()
        if not line:
            return None
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ReplayError(f"replay frame is not an object: {message!r}")
        return message

    def close(self) -> None:
        self._writer.close()


def run_component_loop(channel: Channel) -> None:
    """The shared component: echo each turn as ordered delta frames.

    The same function runs in-process and inside the spawned replay child.
    """
    run = f"run-{os.getpid()}"
    sequence = 0
    while (message := channel.receive()) is not None:
        if message.get("type") != _TURN:
            raise ReplayError(f"unknown replay message: {message!r}")
        text = str(message.get("text", ""))
        words = tuple(word for word in text.split(" ") if word)
        for index, word in enumerate(words):
            channel.send({
                "type": _DELTA,
                "run": run,
                "index": index,
                "text": word,
                "clock": round(time.monotonic(), 6),
            })
        channel.send({
            "type": _TURN_END,
            "run": run,
            "sequence": sequence,
            "text": text,
            "clock": round(time.monotonic(), 6),
        })
        sequence += 1


@dataclass(frozen=True)
class ReplayOutcome:
    frames: tuple[dict[str, object], ...]
    texts: tuple[str, ...]


class ReplayDriver:
    """Drive a scripted turn sequence over any channel."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel

    def run(self, script: ReplayScript) -> ReplayOutcome:
        frames: list[dict[str, object]] = []
        texts: list[str] = []
        for prompt in script.turns:
            self._channel.send({"type": _TURN, "text": prompt})
            while True:
                frame = self._channel.receive()
                if frame is None:
                    raise ReplayError("component channel closed mid-turn")
                frames.append(frame)
                if frame.get("type") == _TURN_END:
                    texts.append(str(frame.get("text", "")))
                    break
        return ReplayOutcome(frames=tuple(frames), texts=tuple(texts))


def normalize_frames(
    frames: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Drop process-volatile fields and canonicalize run identities and ordering."""
    run_ids: dict[str, str] = {}
    normalized: list[dict[str, object]] = []
    for frame in frames:
        values: dict[str, object] = {}
        for key in sorted(frame):
            if key in _VOLATILE_FIELDS:
                continue
            value = frame[key]
            if key == "run":
                value = run_ids.setdefault(str(value), f"run-{len(run_ids)}")
            values[key] = value
        normalized.append(values)
    return tuple(normalized)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(frames: Iterable[Mapping[str, object]]) -> str:
    """Stable sha256 of an already-normalized frame tuple."""
    return hashlib.sha256(canonical_json(list(frames)).encode("utf-8")).hexdigest()


@contextmanager
def in_process_component(
    *, timeout: float = 5.0, component: str = COMPONENT_ECHO
) -> Iterator[Channel]:
    """Host the component with the process boundary removed."""
    entry = component_entry(component)
    driver_channel, component_channel = in_process_channel_pair(timeout=timeout)
    worker = threading.Thread(
        target=entry, args=(component_channel,), name="replay-component", daemon=True
    )
    worker.start()
    try:
        yield driver_channel
    finally:
        driver_channel.close()
        worker.join(timeout=timeout)
        if worker.is_alive():
            raise ReplayError("in-process component did not stop")


@contextmanager
def subprocess_component(
    *,
    mode: str = egress.MODE_ALL,
    timeout: float = 5.0,
    component: str = COMPONENT_ECHO,
) -> Iterator[Channel]:
    """Host the same component in a spawned, egress-blocked child process."""
    # Resolve in the parent too: an unknown component fails the trace here
    # instead of as a child exit code.
    component_entry(component)
    environment = os.environ.copy()
    environment.update(egress.child_environment(mode))
    with egress.bootstrap_directory() as guard_dir:
        environment["PYTHONPATH"] = os.pathsep.join([
            str(guard_dir),
            str(REPLAY_ROOT),
            *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []),
        ])
        process = subprocess.Popen(
            [sys.executable, "-m", _REPLAY_CHILD_MODULE, component],
            cwd=str(REPLAY_ROOT),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdin is None or process.stdout is None:
            raise ReplayError("replay child did not expose its stdio pipes")
        channel = JsonLineChannel(
            cast("TextIO", process.stdout), cast("TextIO", process.stdin)
        )
        try:
            yield channel
        finally:
            _stop_child(process, channel, timeout=timeout)


def _stop_child(
    process: subprocess.Popen[str], channel: Channel, *, timeout: float
) -> None:
    channel.close()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
    stderr = process.stderr.read() if process.stderr is not None else ""
    if process.returncode != 0:
        raise ReplayError(f"replay child exited {process.returncode}: {stderr.strip()}")
