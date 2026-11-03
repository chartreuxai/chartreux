"""Sessionless inspection and removal of stored workspace trust decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tomli_w

AUTHORITY_NOTE = (
    "Trust authorizes a folder's project configuration, harness resources, and "
    "project shell environment. It does not grant workspace roots "
    "(authorized_roots_by_project) or tool authority."
)

RESTART_NOTICE = "Restart any running Chartreux session for this change to take effect."

_REMOVE_CONSEQUENCE = (
    "Its project configuration, harness resources, and project shell environment "
    "stop loading, every path below it inherits the next closest decision, and a "
    "running session keeps the old decision until it is restarted."
)


class UnusableTrustStore(RuntimeError):
    """The stored trust decisions cannot be read or written safely."""


def run_trust_cli(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.trust_command is None:
        parser.print_help()
        return
    match args.trust_command:
        case "list":
            raise SystemExit(_list(args))
        case "remove":
            raise SystemExit(_remove(args))
        case _:
            parser.error(f"Unsupported trust command {args.trust_command!r}.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chartreux trust",
        description=(
            "Inspect and remove stored workspace trust decisions. Removal requires "
            "an interactive confirmation and has no force flag."
        ),
    )
    subparsers = parser.add_subparsers(dest="trust_command")
    list_parser = subparsers.add_parser(
        "list",
        help="List stored trust decisions.",
        description=(
            "List every stored trust decision. With PATH, report the decision that "
            "governs that path: the closest recorded decision wins, so a child path "
            "inherits its nearest recorded ancestor and never the entry above it. "
            "Exit codes: 0 listed, 2 the trust store cannot be read."
        ),
    )
    list_parser.add_argument(
        "path",
        nargs="?",
        metavar="PATH",
        help="Report the decision governing this path instead of listing entries.",
    )
    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove one stored trust decision.",
        description=(
            "Remove the one stored decision recorded for PATH. Sibling and descendant "
            "entries and every decision recorded for another path are untouched. "
            "Confirmation is required and defaults to no; without an interactive "
            "stdin the removal is denied, and no --yes/force flag exists. Exit codes: "
            "0 removed, 1 nothing removed, 2 the trust store cannot be read or written."
        ),
    )
    remove_parser.add_argument(
        "path", metavar="PATH", help="The recorded path whose decision to remove."
    )
    return parser


def _list(args: argparse.Namespace) -> int:
    try:
        store_path, decisions = _read_decisions()
    except UnusableTrustStore as exc:
        print(f"chartreux trust list: {exc}", file=sys.stderr)
        return 2
    print(f"Trust store: {store_path}")
    print(AUTHORITY_NOTE)
    print()
    if args.path is not None:
        return _print_resolution(args.path, decisions)
    entries = [
        (kind, entry) for kind in ("trusted", "untrusted") for entry in decisions[kind]
    ]
    if not entries:
        print("No stored trust decisions.")
        return 0
    print(
        "Stored trust decisions (the closest recorded decision governs a path and "
        "everything below it):"
    )
    for kind, entry in entries:
        ancestor = _ancestor_decision(entry, decisions)
        note = f"; overrides {ancestor[0]} {ancestor[1]}" if ancestor else ""
        print(f"  {kind:<9} {entry}  [stored{note}]")
    return 0


def _print_resolution(raw: str, decisions: dict[str, list[str]]) -> int:
    target = _normalized(raw)
    kind, entry = _resolve_decision(target, decisions)
    if entry is None:
        print(
            f"{target}: undecided (no recorded decision at this path or any ancestor)"
        )
        return 0
    origin = "recorded at this path" if entry == target else f"inherited from {entry}"
    print(f"{target}: {kind} ({origin})")
    return 0


def _remove(args: argparse.Namespace) -> int:
    try:
        store_path, decisions = _read_decisions()
    except UnusableTrustStore as exc:
        print(f"chartreux trust remove: {exc}; nothing was removed", file=sys.stderr)
        return 2
    keys = _exact_keys(args.path)
    matches = [
        (kind, entry)
        for kind in ("trusted", "untrusted")
        for entry in decisions[kind]
        if entry in keys
    ]
    if not matches:
        print(
            f"chartreux trust remove: no stored decision is recorded for {args.path!r}; "
            "nothing was removed. Run `chartreux trust list` to see the recorded "
            "decisions.",
            file=sys.stderr,
        )
        return 1
    recorded = ", ".join(f"{kind} {entry}" for kind, entry in matches)
    if not _stdin_is_interactive():
        print(
            "chartreux trust remove: refusing to remove without an interactive "
            f"confirmation for {recorded}; run it from a terminal. There is no "
            "force flag.",
            file=sys.stderr,
        )
        return 1
    if not _confirm_removal(
        f"Remove stored decision {recorded}? {_REMOVE_CONSEQUENCE} [y/N] "
    ):
        print("Removal cancelled; the stored decision was kept.")
        return 1
    try:
        remaining = _remove_entries(store_path, keys)
    except UnusableTrustStore as exc:
        print(f"chartreux trust remove: {exc}", file=sys.stderr)
        return 2
    for kind, entry in matches:
        print(f"Removed stored decision {kind} {entry}.")
    target = _normalized(args.path)
    kind, entry = _resolve_decision(target, remaining)
    if entry is None:
        print(
            f"After removal, {target} has no recorded decision at this path or any "
            "ancestor."
        )
    else:
        origin = (
            "recorded at this path" if entry == target else f"inherited from {entry}"
        )
        print(f"After removal, {target} resolves to {kind} ({origin}).")
    print(RESTART_NOTICE)
    return 0


def _read_decisions() -> tuple[Path, dict[str, list[str]]]:
    """Stored decisions, refusing any store that cannot answer (never guessing)."""
    from chartreux.core.paths import TRUSTED_FOLDERS_FILE
    from chartreux.core.trusted_folders import (
        TrustStoreState,
        classify_trust_store,
        parse_trust_store,
    )

    store_path = TRUSTED_FOLDERS_FILE.path
    try:
        data = store_path.read_bytes()
    except FileNotFoundError:
        return store_path, {"trusted": [], "untrusted": []}
    except OSError as exc:
        raise UnusableTrustStore(
            f"trust store unreadable at {store_path}: {exc}"
        ) from exc
    state, detail = classify_trust_store(data)
    if state is not TrustStoreState.READY:
        first_line = detail.splitlines()[0] if detail else "no detail"
        raise UnusableTrustStore(
            f"trust store {state.value} at {store_path}: {first_line}"
        )
    return store_path, parse_trust_store(data)


def _remove_entries(store_path: Path, keys: frozenset[str]) -> dict[str, list[str]]:
    """Remove exactly the *keys* entries through the checked-write primitive."""
    from chartreux.core.config.layers._checked_store import (
        CheckedWriteOutcome,
        checked_write_store,
        store_revision,
    )
    from chartreux.core.trusted_folders import parse_trust_store

    remaining: dict[str, list[str]] = {}

    def edit(current: dict[str, list[str]]) -> bytes:
        candidate = {
            kind: [entry for entry in current.get(kind, []) if entry not in keys]
            for kind in ("trusted", "untrusted")
        }
        remaining.update(candidate)
        return tomli_w.dumps(candidate).encode("utf-8")

    result = checked_write_store(
        store_path,
        edit,
        parse=parse_trust_store,
        missing_value={"trusted": [], "untrusted": []},
        expected_revision=store_revision(store_path),
    )
    if result.outcome is CheckedWriteOutcome.SAVED:
        return remaining
    if result.outcome is CheckedWriteOutcome.DURABILITY_UNCERTAIN:
        raise UnusableTrustStore(
            "the store was replaced but its durability is unconfirmed; verify the "
            "file before relying on the removal"
        )
    raise UnusableTrustStore(
        f"the decision was not removed ({result.outcome.value}): {result.error}"
    )


def _normalized(raw: str | Path) -> str:
    return str(Path(raw).expanduser().resolve())


def _exact_keys(raw: str) -> frozenset[str]:
    """The stored spellings *raw* can match; a stored entry is its own string."""
    path = Path(raw).expanduser()
    keys = {str(path.absolute())}
    try:
        keys.add(str(path.resolve()))
    except OSError:
        pass
    return frozenset(keys)


def _resolve_decision(
    target: str, decisions: dict[str, list[str]]
) -> tuple[str | None, str | None]:
    """Closest recorded decision for *target*; ``(None, None)`` when undecided."""
    current = Path(target)
    while True:
        entry = str(current)
        if entry in decisions["trusted"]:
            return "trusted", entry
        if entry in decisions["untrusted"]:
            return "untrusted", entry
        if current.parent == current:
            return None, None
        current = current.parent


def _ancestor_decision(
    entry: str, decisions: dict[str, list[str]]
) -> tuple[str, str] | None:
    """Closest decision strictly above *entry*, which *entry* may override."""
    parent = Path(entry).parent
    while True:
        candidate = str(parent)
        if candidate in decisions["trusted"]:
            return "trusted", candidate
        if candidate in decisions["untrusted"]:
            return "untrusted", candidate
        if parent.parent == parent:
            return None
        parent = parent.parent


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _confirm_removal(prompt: str) -> bool:
    """One interactive yes/no answer; end of input and every other answer is no."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        answer = sys.stdin.readline()
    except (OSError, KeyboardInterrupt):
        sys.stdout.write("\n")
        return False
    if not answer:
        sys.stdout.write("\n")
        return False
    return answer.strip().lower() in {"y", "yes"}
