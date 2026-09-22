from __future__ import annotations

import sys

# isort: off
# Capture the process-start monotonic timestamp as early as possible (before
# any heavier imports below) so the vibe.startup metric measures from true
# process start. Re-exported for any consumer that needs the same anchor.
from chartreux.cli.process_start import (
    PROCESS_START_MONOTONIC as PROCESS_START_MONOTONIC,
)
# isort: on

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from chartreux import __version__

# Anything heavier than argparse is imported inside the functions below, after
# argument parsing, so that --help/--version don't pay for the config stack
# (pydantic, textual, rich) at import time.

if TYPE_CHECKING:
    from chartreux.core.git.worktree import PreparedWorktree, WorktreeCleanupState
    from chartreux.core.git.worktree.record import OwnershipToken


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Chartreux interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  mcp            Manage MCP server configuration (chartreux mcp --help).\n"
            "  models         Manage the model catalog (chartreux models --help).\n\n"
            "Environment variables:\n"
            "  CHARTREUX_HOME       Override the Chartreux home directory (default: ~/.chartreux)\n"
            "  LOG_LEVEL       Logging level: DEBUG, INFO, WARNING (default), ERROR, CRITICAL.\n"
            "                  Also set via log_level in config.toml or /log-level at runtime.\n"
            "                  Logs are written to $CHARTREUX_HOME/logs/chartreux.log.\n"
            "  LOG_MAX_BYTES   Max size of chartreux.log before rotation (default: 10485760).\n"
            "  CHARTREUX_*          Override any config field (e.g. CHARTREUX_ACTIVE_MODEL=local)."
        ),
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "initial_prompt",
        nargs="?",
        metavar="PROMPT",
        help="Initial prompt for programmatic mode. A positional prompt, -p/--prompt, "
        "or non-empty piped stdin sends a prompt, outputs the response, and exits.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        nargs="?",
        const="",
        metavar="TEXT",
        help="Run in programmatic mode: send prompt, output response, and exit. "
        "A positional prompt or non-empty piped stdin also selects this mode. "
        "All tool calls are auto-approved.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        metavar="N",
        help="Maximum number of assistant turns (only applies in programmatic mode).",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        metavar="DOLLARS",
        help="Maximum cost in dollars (only applies in programmatic mode). "
        "Session will be interrupted if cost exceeds this limit.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="Maximum total prompt + completion tokens across the session "
        "(only applies in programmatic mode). "
        "Session will be interrupted if usage exceeds this limit.",
    )
    parser.add_argument(
        "--enabled-tools",
        action="append",
        metavar="TOOL",
        help="Enable specific tools. In programmatic mode, this disables all other tools. "
        "Can use exact names, glob patterns (e.g., 'bash*'), or "
        "regex with 're:' prefix. Can be specified multiple times.",
    )
    parser.add_argument(
        "--disabled-tools",
        action="append",
        metavar="TOOL",
        help="Disable specific tools after --enabled-tools filtering. "
        "Can use exact names, glob patterns (e.g., 'bash*'), or "
        "regex with 're:' prefix. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json", "streaming"],
        default="text",
        help="Output format for programmatic mode: 'text' "
        "for human-readable (default), 'json' for all messages at end, "
        "'streaming' for newline-delimited JSON per message.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive setup: theme, providers, API keys, and model selection.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        metavar="DIR",
        help="Change to this directory before running",
    )
    parser.add_argument(
        "--worktree",
        nargs="?",
        const=True,
        default=None,
        metavar="NAME",
        help="Run inside a git worktree under $CHARTREUX_HOME/worktrees. With NAME, "
        "create (or reuse) a worktree and branch named NAME. Without NAME, "
        "create a new one named after the prompt (or a random slug) on a "
        "chartreux/<name> branch. Implicitly trusted for the session. Ignored with "
        "--setup.",
    )
    parser.add_argument(
        "--add-dir",
        action="append",
        metavar="DIR",
        default=[],
        help="Additional working directory for file access and context. "
        "Implicitly trusted for the session (same semantics as --trust). "
        "Can be specified multiple times.",
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help="Trust the working directory for this invocation only (not "
        "persisted to trusted_folders.toml). Skips the trust prompt. "
        "Use this for non-interactive automation.",
    )

    continuation_group = parser.add_mutually_exclusive_group()
    continuation_group.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_session",
        help="Continue from the most recent saved session",
    )
    continuation_group.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION_ID",
        help="Resume a session. Without SESSION_ID, shows an interactive picker.",
    )
    cli_args = sys.argv[1:]
    if cli_args[:1] == ["update"]:
        parser.error("the 'update' command is not available")
    return parser.parse_args(cli_args)


def _enter_worktree(
    args: argparse.Namespace,
) -> tuple[PreparedWorktree, OwnershipToken]:
    """Prepare the requested worktree, transfer its hold to the CLI, and chdir."""
    from rich import print as rprint

    from chartreux.core.git.errors import GitError
    from chartreux.core.git.worktree import ManagedWorktree, WorktreeRepository
    from chartreux.core.git.worktree.record import (
        AcquireResult,
        HolderKind,
        acquire_holder,
        release_holder,
    )

    requested = "" if args.worktree is True else f" {args.worktree!r}"
    rprint(f"[dim]Preparing worktree{requested}...[/]", file=sys.stderr)
    try:
        with WorktreeRepository.open(Path.cwd()) as repository:
            if args.worktree is True:
                prompt = args.prompt or args.initial_prompt
                session = repository.prepare_auto(
                    prompt=prompt, suggested_name=_suggest_worktree_name(prompt)
                )
            else:
                session = repository.prepare(args.worktree)
    except GitError as e:
        rprint(f"[red]Error: {e}[/]")
        sys.exit(1)

    pending_token = session.pending_token
    managed = ManagedWorktree.at(session.root)
    if managed is None or pending_token is None:
        if pending_token is not None:
            release_holder(pending_token)
        rprint("[red]Error: Unable to protect prepared worktree.[/]")
        sys.exit(1)

    acquired = acquire_holder(
        managed.claim,
        f"cli-{os.getpid()}",
        kind=HolderKind.CLI,
        expected_generation=pending_token.claim_generation,
    )
    if acquired.outcome is not AcquireResult.SUCCESS or acquired.token is None:
        release_holder(pending_token)
        rprint("[red]Error: Unable to protect prepared worktree.[/]")
        sys.exit(1)

    cli_token = acquired.token
    try:
        # The pending attachment holder protects preparation. The CLI holder
        # takes over before chdir, so the worktree remains live for all user
        # input and startup work that follows.
        release_holder(pending_token)
        os.chdir(session.path)
    except BaseException:
        release_holder(cli_token)
        raise
    rprint(f"[dim]Using worktree: {session.path}[/]", file=sys.stderr)
    return session, cli_token


def _prompt_remove_worktree(
    worktree: PreparedWorktree, cleanup_state: WorktreeCleanupState
) -> bool:
    from rich import print as rprint

    reasons = ", ".join(cleanup_state.reasons)
    rprint(f"[yellow]Worktree {worktree.name!r} has {reasons}.[/]", file=sys.stderr)
    rprint(
        "[yellow]Remove it and delete its branch? This discards worktree changes, "
        "untracked files, and commits.[/]",
        file=sys.stderr,
    )
    sys.stderr.write("Remove worktree? [y/N] ")
    sys.stderr.flush()
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.stderr.write("\n")
        return False
    return answer in {"y", "yes", "remove"}


def _prompt_delete_attached_branch(worktree: PreparedWorktree) -> bool:
    from rich import print as rprint

    rprint(
        f"[yellow]Branch {worktree.branch!r} existed before this session "
        f"and was attached, not created by Chartreux.[/]",
        file=sys.stderr,
    )
    sys.stderr.write(f"Also delete branch {worktree.branch!r}? [y/N] ")
    sys.stderr.flush()
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.stderr.write("\n")
        return False
    return answer in {"y", "yes", "delete"}


def _cleanup_worktree_on_exit(  # noqa: PLR0911
    worktree: PreparedWorktree, cli_token: OwnershipToken
) -> None:
    from rich import print as rprint

    from chartreux.core.git.errors import GitError
    from chartreux.core.git.worktree import WorktreeReleaseOutcome, remove_if_unheld
    from chartreux.core.git.worktree.record import AcquireResult, inspect_holders

    try:
        cleanup_state = worktree.inspect_for_cleanup()
    except GitError as e:
        rprint(
            f"[yellow]Could not inspect worktree for cleanup: {e}[/]", file=sys.stderr
        )
        return

    if not cleanup_state.is_clean and not _prompt_remove_worktree(
        worktree, cleanup_state
    ):
        rprint(f"[dim]Keeping worktree: {worktree.root}[/]", file=sys.stderr)
        return

    delete_branch = worktree.branch_created or _prompt_delete_attached_branch(worktree)

    # Do not hold the prune lock while prompting. A concurrent attachment may
    # arrive while the user decides, so inspect now for a useful message and
    # let remove_if_unheld re-check under the destructive lock immediately
    # before removal.
    from chartreux.core.git.worktree import ManagedWorktree

    managed = ManagedWorktree.at(worktree.root)
    if managed is None:
        rprint(f"[dim]Keeping worktree: {worktree.root}[/]", file=sys.stderr)
        return
    holders = inspect_holders(managed.claim)
    if holders.outcome is AcquireResult.ERROR:
        rprint(
            f"[yellow]Could not inspect worktree holders: {holders.error}[/]",
            file=sys.stderr,
        )
        return
    if remaining := frozenset(holders.holders) - {cli_token.holder_id}:
        rprint(
            f"[dim]Keeping worktree {worktree.root}: in use by "
            f"{len(remaining)} other session(s)[/]",
            file=sys.stderr,
        )
        return

    try:
        rprint(f"[dim]Removing worktree: {worktree.root}[/]", file=sys.stderr)
        worktree.leave_if_current_directory()
        release = remove_if_unheld(
            worktree.root,
            expected_generation=cli_token.claim_generation,
            retiring_token=cli_token,
            delete_branch=delete_branch,
        )
    except GitError as e:
        rprint(f"[yellow]Could not remove worktree: {e}[/]", file=sys.stderr)
        return

    if release.outcome is WorktreeReleaseOutcome.KEPT_IN_USE:
        rprint(
            f"[dim]Keeping worktree {worktree.root}: in use by another session[/]",
            file=sys.stderr,
        )
        return
    if release.outcome is not WorktreeReleaseOutcome.REMOVED:
        rprint(f"[dim]Keeping worktree: {worktree.root}[/]", file=sys.stderr)
        return

    rprint(f"[dim]Removed worktree: {worktree.root}[/]", file=sys.stderr)
    if not delete_branch:
        rprint(f"[dim]Kept branch: {worktree.branch}[/]", file=sys.stderr)


def _suggest_worktree_name(prompt: str | None) -> str | None:
    # Bare `vibe --worktree` has nothing to name from, so skip the dotenv read
    # and the event loop rather than spinning both up to be told None.
    if not prompt:
        return None

    import asyncio

    from chartreux.core.config.chartreux_schema import load_dotenv_values
    from chartreux.core.config.harness_files import init_harness_files_manager
    from chartreux.core.git.worktree.naming_model import suggest_worktree_name

    # Worktrees are prepared before run_cli, so neither of the things the
    # suggestion needs has happened yet. ~/.chartreux/.env is not in os.environ, so a
    # key that lives only there would read as absent; and loading config
    # resolves prompts through the global harness manager, which is not
    # initialised until later in main(). Both calls are idempotent -- the second
    # init with these same sources returns without replacing the singleton.
    load_dotenv_values()
    init_harness_files_manager("user", "project")
    return asyncio.run(suggest_worktree_name(prompt, cwd=Path.cwd()))


def _set_process_title() -> None:
    # Cosmetic: renames the process from "python3" to "Chartreux CLI" in ps/top and
    # Activity Monitor on Linux/macOS so it can be spotted and killed; concurrent
    # instances are told apart by the process manager's PID column.
    try:
        import setproctitle

        from chartreux.cli._process_title import process_name

        setproctitle.setproctitle(process_name())
    except Exception:
        pass


def main() -> None:
    _set_process_title()

    if sys.argv[1:2] == ["models"]:
        from chartreux.core.model_catalog.migration import run_models_cli

        run_models_cli(sys.argv[2:])
        return

    if sys.argv[1:2] == ["mcp"]:
        from chartreux.cli.mcp_command import run_mcp_cli

        run_mcp_cli(sys.argv[2:])
        return

    args = parse_arguments()
    worktree_session: PreparedWorktree | None = None
    cli_token: OwnershipToken | None = None

    from rich import print as rprint

    from chartreux.core.config.harness_files import init_harness_files_manager
    from chartreux.core.paths import LITERAL_LOG_FILE, LOG_FILE
    from chartreux.observability.logging import init_file_logging

    init_file_logging(LOG_FILE.path, repair_path=LITERAL_LOG_FILE.path)

    if args.workdir:
        workdir = args.workdir.expanduser().resolve()
        if not workdir.is_dir():
            rprint(
                f"[red]Error: --workdir does not exist or is not a directory: {workdir}[/]"
            )
            sys.exit(1)
        os.chdir(workdir)

    # Must run before `cwd` is read and before run_cli so that session lookups
    # (-c / --resume picker) scope to the worktree directory.
    if args.worktree and not args.setup:
        worktree_session, cli_token = _enter_worktree(args)

    try:
        try:
            Path.cwd()
        except FileNotFoundError:
            rprint(
                "[red]Error: Current working directory no longer exists.[/]\n"
                "[yellow]The directory you started Chartreux from has been deleted. "
                "Please change to an existing directory and try again, "
                "or use --workdir to specify a working directory.[/]"
            )
            sys.exit(1)

        additional_dirs: list[Path] = []
        for d in args.add_dir:
            resolved = Path(d).expanduser().resolve()
            if not resolved.is_dir():
                rprint(
                    f"[red]Error: --add-dir path does not exist "
                    f"or is not a directory: {d}[/]"
                )
                sys.exit(1)
            additional_dirs.append(resolved)

        args.add_dir = [str(path) for path in additional_dirs]
        # Capture the final launch directory before startup work.  The session
        # must retain the directory selected by the user (--workdir/worktree or
        # the process cwd), even if a later dependency changes the process cwd.
        args.session_cwd = Path.cwd().resolve()
        init_harness_files_manager("user", "project")

        _run_cli_with_worktree_cleanup(args, worktree_session, cli_token)
    finally:
        # Startup can fail before _run_cli_with_worktree_cleanup() is entered.
        # Release the process-bound CLI token on those paths too.
        if cli_token is not None and not cli_token.consumed:
            from chartreux.core.git.worktree.record import release_holder

            release_holder(cli_token)


def _run_cli_with_worktree_cleanup(
    args: argparse.Namespace,
    worktree_session: PreparedWorktree | None,
    cli_token: OwnershipToken | None,
) -> None:
    from chartreux.cli.cli import run_cli

    session_started = False
    try:
        run_cli(args)
        session_started = True
    except SystemExit as e:
        session_started = e.code in {0, None}
        raise
    finally:
        # Only auto-clean worktrees Chartreux created this run, and only once a
        # session actually ran — a startup failure (bad config, --continue with
        # no sessions) must not delete a reused worktree or its branch.
        if (
            worktree_session is not None
            and worktree_session.created
            and not getattr(args, "is_programmatic", False)
            and session_started
            and cli_token is not None
        ):
            _cleanup_worktree_on_exit(worktree_session, cli_token)
        # Released for every worktree this run held, not only the ones eligible
        # for cleanup above. The token is process-bound and cannot release a
        # holder acquired by another CLI process.
        if cli_token is not None and not cli_token.consumed:
            from chartreux.core.git.worktree.record import release_holder

            release_holder(cli_token)


if __name__ == "__main__":
    main()
