"""Child entry point for the subprocess topology of the replay skeleton (WP6 [6A]).

The egress guard is installed by the generated ``sitecustomize`` from the
spawn-env marker before this module is imported. The child refuses to run when
the marker did not reach the transport, so a differential trace can never
execute in a silently unblocked process.
"""

from __future__ import annotations

import os
import sys

from tests.differential import egress
from tests.differential.replay import COMPONENT_ECHO, JsonLineChannel, component_entry


def main() -> int:
    expected = os.environ.get(egress.EGRESS_BLOCK_ENV, "")
    installed = egress.installed_mode()
    if not expected or installed != expected:
        raise SystemExit(
            "replay child requires the egress marker to be applied before import "
            f"({egress.EGRESS_BLOCK_ENV}={expected!r}, installed={installed!r})"
        )
    name = sys.argv[1] if len(sys.argv) > 1 else COMPONENT_ECHO
    component_entry(name)(JsonLineChannel(sys.stdin, sys.stdout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
