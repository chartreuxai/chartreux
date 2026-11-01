"""Read-only inspection of user skill copies against the shipped copies."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from chartreux.core.skills import manager as skills_manager
from chartreux.core.skills.manager import ShadowEntry, ShadowState

if TYPE_CHECKING:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.config.harness_files import HarnessFilesManager


def run_skills_cli(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.skills_command is None:
        parser.print_help()
        return
    match args.skills_command:
        case "diff":
            try:
                code = asyncio.run(_diff(args, parser))
            except Exception as exc:
                print(f"chartreux skills diff: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
            raise SystemExit(code)
        case _:
            parser.error(f"Unsupported skills command {args.skills_command!r}.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chartreux skills", description="Inspect the skill surface."
    )
    subparsers = parser.add_subparsers(dest="skills_command")
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare user skill copies with the shipped copies.",
        description=(
            "Compare user skill copies in ~/.chartreux/skills and ~/.agents/skills "
            "with the shipped copy of the same name. The shipped copy is the diff's "
            "old side. Exit codes: 0 nothing differs, 1 a loaded user copy differs, "
            "2 operational error."
        ),
    )
    diff_parser.add_argument(
        "name", nargs="?", metavar="NAME", help="Restrict the report to one skill name."
    )
    diff_parser.add_argument(
        "--json", action="store_true", help="Emit JSON only on stdout."
    )
    return parser


async def _load_config() -> tuple[ChartreuxConfigSchema, HarnessFilesManager]:
    """Load the trusted configuration the loader itself would discover with."""
    from chartreux.core.config.default_orchestrator import build_default_orchestrator
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.trusted_folders import TrustedFoldersManager

    trust_store = TrustedFoldersManager()
    manager = HarnessFilesManager(
        sources=("user", "project"), cwd=Path.cwd(), trust_store=trust_store
    )
    orchestrator = await build_default_orchestrator(harness_files=manager)
    return orchestrator.config, manager


async def _diff(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config, manager = await _load_config()
    inventory = skills_manager.build_shadow_inventory(
        lambda: config, harness_files=manager
    )
    entries = inventory.entries
    if args.name is not None:
        entries = tuple(entry for entry in entries if entry.name == args.name)
        if not entries:
            parser.error(
                f"unknown skill {args.name!r}: no user copy of that name shadows "
                "or is shadowed by another skill"
            )
    code = _exit_code(entries)
    if args.json:
        print(
            json.dumps({
                "entries": [_entry_payload(entry) for entry in entries],
                "exit_code": code,
            })
        )
    else:
        print(_report(entries, inventory.user_dirs))
    return code


def _exit_code(entries: tuple[ShadowEntry, ...]) -> int:
    """0 nothing differs, 1 a loaded user copy differs, 2 operational error."""
    if any(entry.state is ShadowState.UNVERIFIED for entry in entries):
        return 2
    return int(any(entry.state is ShadowState.DIVERGED for entry in entries))


def _entry_payload(entry: ShadowEntry) -> dict[str, object]:
    return {
        "name": entry.name,
        "path": str(entry.skill_path),
        "user_dir": str(entry.user_dir),
        "state": str(entry.state),
        "shipped_path": str(entry.shipped_path) if entry.shipped_path else None,
        "winner_path": str(entry.winner_path) if entry.winner_path else None,
        "winner_scope": entry.winner_scope,
        "diff": entry.diff,
        "reason": entry.reason,
    }


def _report(entries: tuple[ShadowEntry, ...], user_dirs: tuple[Path, ...]) -> str:
    if not entries:
        if not user_dirs:
            return "no user skill directories exist yet"
        scanned = ", ".join(str(user_dir) for user_dir in user_dirs)
        return f"no shadowed user skills in {scanned}"
    return "\n\n".join(_entry_text(entry) for entry in entries)


def _entry_text(entry: ShadowEntry) -> str:
    match entry.state:
        case ShadowState.IDENTICAL:
            return "\n".join((
                f"identical: {entry.name}",
                f"  user:    {entry.skill_path}",
                f"  shipped: {entry.shipped_path}",
                "  matches the shipped copy; suggest removal after reviewing "
                "support files — never automatic",
            ))
        case ShadowState.DIVERGED:
            return "\n".join((
                f"diverged: {entry.name}",
                f"  shipped: {entry.shipped_path}",
                f"  user:    {entry.skill_path}",
                entry.diff.rstrip("\n"),
            ))
        case ShadowState.INACTIVE:
            winner = (
                f"{entry.winner_path} ({entry.winner_scope})"
                if entry.winner_path is not None
                else f"the {entry.winner_scope} skill"
            )
            return "\n".join((
                f"inactive: {entry.name}",
                f"  user:   {entry.skill_path}",
                f"  loads:  {winner}",
                "  this user copy does not load; edit the winning copy instead",
            ))
        case _:
            return "\n".join((
                f"unverified: {entry.name}",
                f"  user:   {entry.skill_path}",
                f"  error:  {entry.reason}",
            ))


__all__ = ["run_skills_cli"]
