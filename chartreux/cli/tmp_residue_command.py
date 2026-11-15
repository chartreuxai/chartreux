"""Sessionless preview and confirmed removal of point-3 temp residue.

``chartreux tmp-residue list`` is ALWAYS read-only and works noninteractively:
it lists every residue tree under the temp root — exact absolute path, entry
count, and class (a managed allocation with its owner identity summary and
gate state, or ``legacy-unmarked`` for unmarked ``chartreux-scratchpad-*`` /
``vibe-scratchpad-*`` trees) — and never deletes anything.

``chartreux tmp-residue remove PATH...`` deletes ONLY after a separate
interactive confirmation that names the exact paths. Without an interactive
stdin the removal is denied (noninteractive operation stays preview-only) and
no ``--yes``/force flag exists. Managed allocations are additionally gated:
a live or indeterminate owner is reported, never deleted. The shared
``pytest-of-<user>`` root is never swept by this command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat as stat_module
import sys
import tempfile

MANAGED_PARENT_PREFIX = "chartreux-managed-"
LEGACY_PREFIXES = ("chartreux-scratchpad-", "vibe-scratchpad-")
PYTEST_ROOT_PREFIX = "pytest-of-"

UNMARKED_LIVENESS_WARNING = (
    "liveness cannot be established for unmarked trees: they carry no ownership "
    "record, so only you can confirm nothing is using them"
)

_REMOVE_CONSEQUENCE = (
    "Deletion is permanent and only ever happens for the exact paths named "
    "below, after your interactive confirmation."
)


class UnusableTempRoot(RuntimeError):
    """The temp root cannot be inspected safely."""


@dataclass(frozen=True, slots=True)
class ResidueEntry:
    """One residue tree in the preview inventory."""

    path: Path
    kind: str  # "managed" | "legacy-unmarked"
    entries: int | None
    detail: str


def run_tmp_residue_cli(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.tmp_residue_command is None:
        parser.print_help()
        return
    match args.tmp_residue_command:
        case "list":
            raise SystemExit(_list(args))
        case "remove":
            raise SystemExit(_remove(args))
        case _:
            parser.error(
                f"Unsupported tmp-residue command {args.tmp_residue_command!r}."
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chartreux tmp-residue",
        description=(
            "Preview temp residue (managed scratchpad/pytest-run roots and "
            "legacy unmarked scratchpad trees). The preview is always "
            "read-only; removal requires an interactive confirmation and has "
            "no force flag."
        ),
    )
    subparsers = parser.add_subparsers(dest="tmp_residue_command")
    list_parser = subparsers.add_parser(
        "list",
        help="List the residue inventory (read-only).",
        description=(
            "List every residue tree under the temp root: exact absolute path, "
            "entry count, and class. Managed allocations also show the owner "
            "identity summary and their reap-gate state. This command never "
            "deletes anything. Exit codes: 0 listed, 2 the temp root cannot "
            "be inspected."
        ),
    )
    list_parser.add_argument(
        "--temp-root",
        metavar="DIR",
        default=None,
        help="Inspect this temp root instead of the system one.",
    )
    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove named residue trees after an interactive confirmation.",
        description=(
            "Remove exactly the residue trees named by PATH arguments. Each path "
            "must appear in the inventory (a managed allocation or a legacy "
            "unmarked scratchpad tree); managed allocations whose gates do not "
            "pass are reported, never deleted. Confirmation is required and "
            "defaults to no; without an interactive stdin the removal is denied "
            "and nothing is deleted. There is no --yes/force flag. Exit codes: "
            "0 removed, 1 nothing removed, 2 the temp root cannot be inspected."
        ),
    )
    remove_parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="An exact residue path to remove (see `chartreux tmp-residue list`).",
    )
    remove_parser.add_argument(
        "--temp-root",
        metavar="DIR",
        default=None,
        help="Resolve inventory paths against this temp root instead of the system one.",
    )
    return parser


def _temp_root(args: argparse.Namespace) -> Path:
    raw = getattr(args, "temp_root", None)
    root = Path(raw).expanduser() if raw else Path(tempfile.gettempdir())
    return Path(os.path.abspath(root))


def _entry_count(path: Path) -> int | None:
    try:
        total = 0
        for _root, dirs, files in os.walk(path):
            total += len(dirs) + len(files)
        return total
    except OSError:
        return None


def _managed_inventory(parent: Path) -> list[ResidueEntry]:
    from chartreux.core import tmp_ownership as to

    entries: list[ResidueEntry] = []
    if not parent.is_dir():
        return entries
    for child in sorted(parent.iterdir()):
        if not to.is_managed_name(child.name):
            continue
        try:
            info = child.lstat()
        except OSError:
            continue
        if not stat_module.S_ISDIR(info.st_mode):
            continue
        alloc = to.allocation_from_path(child)
        record = to.load_ownership_record(parent, child.name)
        if record is None:
            detail = "torn-init (managed name, no ownership record)"
        else:
            verdict = to.adjudicate(record.owner.to_json())
            owner = record.owner
            summary = (
                f"owner pid={owner.pid} pgid={owner.pgid} "
                f"starttime={owner.starttime} boot_id={owner.boot_id[:8]}… "
                f"({verdict.verdict}: {verdict.reason})"
            )
            try:
                with to.parent_lock(parent):
                    gates = to.evaluate_allocation_gates(alloc)
            except OSError as exc:
                detail = f"{summary}; gate evaluation failed ({exc})"
            else:
                state = "reapable" if gates.reapable else ", ".join(gates.reasons)
                detail = f"{summary}; gates: {state}"
        entries.append(
            ResidueEntry(
                path=child, kind="managed", entries=_entry_count(child), detail=detail
            )
        )
    return entries


def _legacy_inventory(root: Path) -> list[ResidueEntry]:
    entries: list[ResidueEntry] = []
    try:
        children = sorted(root.iterdir())
    except OSError as exc:
        raise UnusableTempRoot(f"temp root {root} cannot be listed: {exc}") from exc
    for child in children:
        if not child.name.startswith(LEGACY_PREFIXES):
            continue
        try:
            info = child.lstat()
        except OSError:
            continue
        if not stat_module.S_ISDIR(info.st_mode):
            continue
        entries.append(
            ResidueEntry(
                path=child,
                kind="legacy-unmarked",
                entries=_entry_count(child),
                detail=UNMARKED_LIVENESS_WARNING,
            )
        )
    return entries


def build_inventory(temp_root: Path | str | None = None) -> list[ResidueEntry]:
    """The read-only residue inventory: managed allocations under the per-user
    managed parent plus legacy unmarked scratchpad trees at the temp root.

    The shared ``pytest-of-<user>`` root is never part of the inventory.
    """
    root = Path(temp_root) if temp_root is not None else Path(tempfile.gettempdir())
    root = Path(os.path.abspath(root))
    parent = root / f"{MANAGED_PARENT_PREFIX}{os.getuid()}"
    return [*_managed_inventory(parent), *_legacy_inventory(root)]


def _format_entry(entry: ResidueEntry) -> str:
    count = "?" if entry.entries is None else str(entry.entries)
    return f"  {entry.kind:<15} entries={count:<8} {entry.path}\n      {entry.detail}"


def _list(args: argparse.Namespace) -> int:
    root = _temp_root(args)
    try:
        inventory = build_inventory(root)
    except UnusableTempRoot as exc:
        print(f"chartreux tmp-residue list: {exc}", file=sys.stderr)
        return 2
    print(f"Temp root: {root}")
    print(
        "Preview only — nothing is deleted by this command. Removal happens "
        "only via `chartreux tmp-residue remove <path>...` after an interactive "
        "confirmation."
    )
    print()
    if not inventory:
        print("No residue trees.")
        return 0
    print(f"Residue inventory ({len(inventory)} tree(s)):")
    for entry in inventory:
        print(_format_entry(entry))
    return 0


def _is_pytest_root(path: Path) -> bool:
    return any(part.startswith(PYTEST_ROOT_PREFIX) for part in path.parts)


def _resolve_named(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return Path(os.path.abspath(path))


def _resolve_named_targets(
    raw_paths: list[str], root: Path, inventory: dict[str, ResidueEntry]
) -> list[tuple[ResidueEntry, Path]] | None:
    """Resolve the named PATH arguments to inventory entries.

    Returns None (after printing the refusal) for any path that is not an
    inventoried residue tree, is a symlink, lives under the shared
    ``pytest-of-<user>`` root, or cannot be inspected — nothing is removed.
    """
    targets: list[tuple[ResidueEntry, Path]] = []
    for raw in raw_paths:
        path = _resolve_named(root, raw)
        if _is_pytest_root(path):
            print(
                f"chartreux tmp-residue remove: refusing {path}: the shared "
                f"{PYTEST_ROOT_PREFIX}<user> root is never swept; reap it "
                "manually if you are sure nothing uses it.",
                file=sys.stderr,
            )
            return None
        entry = inventory.get(str(path))
        if entry is None:
            print(
                f"chartreux tmp-residue remove: {path} is not in the residue "
                "inventory; nothing was removed. Run `chartreux tmp-residue "
                "list` to see the inventoried trees.",
                file=sys.stderr,
            )
            return None
        try:
            info = path.lstat()
        except OSError as exc:
            print(
                f"chartreux tmp-residue remove: {path} cannot be inspected "
                f"({exc}); nothing was removed.",
                file=sys.stderr,
            )
            return None
        if stat_module.S_ISLNK(info.st_mode):
            print(
                f"chartreux tmp-residue remove: refusing symlink {path}; "
                "nothing was removed.",
                file=sys.stderr,
            )
            return None
        targets.append((entry, path))
    return targets


def _gate_managed_targets(
    targets: list[tuple[ResidueEntry, Path]],
) -> list[tuple[ResidueEntry, Path]] | None:
    """Gate the managed allocations: a non-reapable tree is reported, never
    deleted, exactly like the harness's own reapers. Returns None on refusal.
    """
    from chartreux.core import tmp_ownership as to

    confirmed: list[tuple[ResidueEntry, Path]] = []
    for entry, path in targets:
        if entry.kind != "managed":
            confirmed.append((entry, path))
            continue
        alloc = to.allocation_from_path(path)
        try:
            with to.parent_lock(alloc.parent):
                gates = to.evaluate_allocation_gates(alloc)
        except OSError as exc:
            print(
                f"chartreux tmp-residue remove: {path} gates could not be "
                f"evaluated ({exc}); retained, nothing was removed.",
                file=sys.stderr,
            )
            return None
        if not gates.reapable:
            print(
                f"chartreux tmp-residue remove: {path} is retained by its "
                f"live-user gates ({', '.join(gates.reasons)}); it is "
                "reported, never deleted.",
                file=sys.stderr,
            )
            return None
        confirmed.append((entry, path))
    return confirmed


def _print_noninteractive_refusal(confirmed: list[tuple[ResidueEntry, Path]]) -> None:
    print(
        "chartreux tmp-residue remove: refusing to remove without an "
        "interactive confirmation; noninteractive operation is "
        "preview-only. The exact paths that would be removed:",
        file=sys.stderr,
    )
    for entry, path in confirmed:
        print(f"  {entry.kind} {path}", file=sys.stderr)
    print("Run it from a terminal. There is no force flag.", file=sys.stderr)


def _confirm_removal(confirmed: list[tuple[ResidueEntry, Path]]) -> bool:
    print("The following exact paths would be removed:")
    for entry, _path in confirmed:
        print(_format_entry(entry))
    print()
    print(UNMARKED_LIVENESS_WARNING + ".")
    print(_REMOVE_CONSEQUENCE)
    prompt = f"Remove these {len(confirmed)} exact path(s)? [y/N] "
    if not _confirm(prompt):
        print("Removal cancelled; every named path was kept.")
        return False
    return True


def _delete_confirmed(confirmed: list[tuple[ResidueEntry, Path]]) -> int:
    from chartreux.core import tmp_ownership as to

    removed = 0
    for entry, path in confirmed:
        if entry.kind == "managed":
            result = to.reap_allocation(to.allocation_from_path(path))
            if not result.reaped:
                print(
                    f"chartreux tmp-residue remove: {path} was retained "
                    f"({', '.join(result.reasons)}); nothing else was touched.",
                    file=sys.stderr,
                )
                continue
        else:
            try:
                shutil.rmtree(path)
            except OSError as exc:
                print(
                    f"chartreux tmp-residue remove: {path} could not be removed "
                    f"({exc}); nothing else was touched.",
                    file=sys.stderr,
                )
                continue
        print(f"Removed {entry.kind} tree {path}.")
        removed += 1
    if removed:
        print(f"Removed {removed} of {len(confirmed)} named path(s).")
    return 0 if removed == len(confirmed) else 1


def _remove(args: argparse.Namespace) -> int:
    root = _temp_root(args)
    try:
        inventory = {str(entry.path): entry for entry in build_inventory(root)}
    except UnusableTempRoot as exc:
        print(
            f"chartreux tmp-residue remove: {exc}; nothing was removed", file=sys.stderr
        )
        return 2
    targets = _resolve_named_targets(args.paths, root, inventory)
    if targets is None:
        return 1
    confirmed = _gate_managed_targets(targets)
    if confirmed is None:
        return 1
    if not _stdin_is_interactive():
        _print_noninteractive_refusal(confirmed)
        return 1
    if not _confirm_removal(confirmed):
        return 1
    return _delete_confirmed(confirmed)


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _confirm(prompt: str) -> bool:
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
