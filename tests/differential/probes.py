"""WP6 [6D]: the per-process probe the both-mode and egress gates drive.

The probe answers two questions about the process that hosts it:

``report``
    Which transport selection this process resolves and which egress mode its
    transport guard installed — proof that the [6D] mode axis and the [6A]
    spawn-env marker actually reached the process the trace ran in, rather
    than only the driver's environment.
``connect``
    What a transport connect to one address does here — the egress trace uses
    it to show that a spawned component reaches neither the fake provider's
    loopback endpoint nor an independent non-loopback destination.

This component is deliberately not digest-compared across topologies: its
whole purpose is to differ between a process with a boundary and one without.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

from chartreux.app_server.transport import resolve_transport_selection
from tests.differential import egress
from tests.differential.replay import Channel, ReplayError

_SELECTION_FIELDS = ("kind", "selector", "source")


class ProbeHost:
    """Answer probe requests for one process."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel

    def serve(self) -> None:
        while (message := self._channel.receive()) is not None:
            kind = message.get("type")
            if kind == "report":
                self._channel.send({"type": "report", **self._report()})
            elif kind == "connect":
                self._channel.send({
                    "type": "connect",
                    "address": message.get("address"),
                    **self._connect(message),
                })
            else:
                raise ReplayError(f"unknown probe message: {message!r}")

    @staticmethod
    def _report() -> dict[str, Any]:
        selection = resolve_transport_selection()
        return {
            "egress_mode": egress.installed_mode(),
            "egress_marker": bool(
                os.environ.get(egress.EGRESS_BLOCK_ENV) == egress.installed_mode()
            ),
            "selection": {
                field: getattr(selection, field) for field in _SELECTION_FIELDS
            },
        }

    @staticmethod
    def _connect(message: Mapping[str, Any]) -> dict[str, Any]:
        address = message.get("address")
        if not isinstance(address, list) or len(address) != 2:
            raise ReplayError(f"connect probe carries no [host, port]: {message!r}")
        host, port = address
        return egress.probe(str(host), int(port))


def run_probe_loop(channel: Channel) -> None:
    """Component entry point shared by both topologies."""
    ProbeHost(channel).serve()
