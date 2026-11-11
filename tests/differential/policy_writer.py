"""A real competing root-config writer for the [6B] competing-writer oracle.

Runs in a separate process against the same user-layer store the oracle is
using, so the revision check arbitrates two independent root writers exactly as
it does in production: the first writer's revision wins and the second reports
a conflict. The caller passes the revision it captured, and the outcome is
printed as JSON for the trace to compare.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

from tests.differential.policy_store import write_project_root

USAGE = (
    "usage: python -m tests.differential.policy_writer "
    "CONFIG PROJECT ROOT EXPECTED_REVISION"
)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit(USAGE)
    config, project, root, expected_revision = argv
    result = asyncio.run(
        write_project_root(
            config=Path(config),
            project=Path(project),
            root=Path(root),
            expected_revision=expected_revision,
        )
    )
    print(
        json.dumps(
            {
                "target": result.target,
                "persistence": result.persistence,
                "application": result.application,
                "error": result.error,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
