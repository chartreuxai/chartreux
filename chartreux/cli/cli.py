from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from pydantic import ValidationError
from rich import print as rprint

from chartreux import __version__
from chartreux.cli.session_exit import print_session_resume_message
from chartreux.cli.terminal_detect import detect_terminal
from chartreux.core.config import (
    ChartreuxConfigSchema,
    MissingAPIKeyError,
    load_dotenv_values,
)
from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.core.config.layer import ConfigStorageError
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.paths import CHARTREUX_HOME, GLOBAL_ENV_FILE, HISTORY_FILE
from chartreux.utils.private_paths import restrict_private_file

# The TUI app, onboarding, and programmatic runner are each imported at their
# call site: every launch needs at most one of them, and they are too heavy to
# load speculatively at startup.

if TYPE_CHECKING:
    from chartreux.app_server.local import LocalSessionIntent


def has_usable_terminal() -> bool:
    """Return whether the full-screen setup UI can use both terminal streams."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def get_prompt_from_stdin() -> str | None:
    """Read piped input before mode selection without changing stdin."""
    if sys.stdin.isatty():
        return None
    try:
        content = sys.stdin.read().strip()
    except KeyboardInterrupt:
        return None
    return content or None


def restore_interactive_stdin(*, stdin_isatty: bool) -> None:
    """Restore the terminal for an interactive TUI after reading piped stdin."""
    if stdin_isatty:
        return
    try:
        sys.stdin = sys.__stdin__ = open("/dev/tty")
    except OSError:
        pass


def _format_config_validation_error(exc: ValidationError) -> str:
    lines = [f"Invalid configuration ({exc.error_count()} error(s)):"]
    for err in exc.errors(include_url=False):
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_config_orchestrator_or_exit() -> ConfigOrchestrator[ChartreuxConfigSchema]:
    try:
        return asyncio.run(build_default_orchestrator())
    except ValidationError as e:
        rprint(f"[yellow]{_format_config_validation_error(e)}[/]")
        sys.exit(1)
    except ConfigStorageError as e:
        rprint(
            f"[yellow]Cannot {e.operation} the Chartreux config file at {e.path}: "
            f"{e.__cause__}.\nChartreux needs read/write access to it. If it is managed "
            "read-only (e.g. symlinked from the Nix store), make it writable or set "
            "CHARTREUX_HOME to a writable directory.[/]"
        )
        sys.exit(1)
    except ValueError as e:
        rprint(f"[yellow]{e}[/]")
        sys.exit(1)


def require_api_key_or_onboard(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], *, interactive: bool
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    try:
        orchestrator.config.require_active_provider_api_key()
        return orchestrator
    except (MissingAPIKeyError, ValueError) as e:
        if not interactive:
            print(
                f"Error: {e}. Set the environment variable (e.g. in ~/.chartreux/.env "
                "or your shell), or run `chartreux --setup` once interactively.",
                file=sys.stderr,
            )
            sys.exit(1)

        from chartreux.setup.onboarding import run_onboarding

        return run_onboarding(orchestrator=orchestrator)


def bootstrap_config_files() -> None:
    history_file = HISTORY_FILE.path
    if not history_file.exists():
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history_file.write_text("Hello Chartreux!\n", "utf-8")
        except Exception as e:
            rprint(f"[yellow]Could not create history file: {e}[/]")

    config_file = CHARTREUX_HOME.path / "config.toml"
    if not config_file.exists():
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text("", "utf-8")
            rprint(
                f"[dim]Created selections config at {config_file}[/]", file=sys.stderr
            )
        except Exception as e:
            rprint(f"[yellow]Could not create default config file: {e}[/]")

    env_file = GLOBAL_ENV_FILE.path
    if not env_file.exists():
        try:
            env_file.parent.mkdir(parents=True, exist_ok=True)
            env_file.write_text("", "utf-8")
            # The file is created empty but will hold API keys once filled in,
            # so restrict it to owner-only immediately at creation.
            restrict_private_file(env_file)
        except Exception as e:
            rprint(f"[yellow]Could not create .env file: {e}[/]")


def _session_intent(
    args: argparse.Namespace, *, allow_picker: bool
) -> LocalSessionIntent:
    from chartreux.app_server.local import (
        ContinueSessionIntent,
        NewSessionIntent,
        ResumeSessionIntent,
    )

    if args.continue_session:
        return ContinueSessionIntent()
    if args.resume is True:
        if allow_picker:
            return NewSessionIntent()
        raise ValueError("--resume requires a session ID in programmatic mode")
    if isinstance(args.resume, str):
        return ResumeSessionIntent(args.resume)
    return NewSessionIntent()


def _session_cwd(args: argparse.Namespace) -> Path:
    """Return the directory selected by the entrypoint for this session."""
    return Path(getattr(args, "session_cwd", Path.cwd())).resolve()


def _run_programmatic_mode(args: argparse.Namespace, stdin_prompt: str | None) -> None:
    from chartreux.app_server.local import ClientDescriptor, LocalHarnessOptions
    from chartreux.app_server.protocol import (
        AppServerResponseError,
        ClientCapabilities,
        ClientInfo,
        SessionOptions,
    )
    from chartreux.cli.programmatic import (
        OutputFormat,
        ProgrammaticLimitError,
        run_programmatic,
    )

    if args.prompt is not None:
        programmatic_prompt = args.prompt
    elif args.initial_prompt is not None:
        programmatic_prompt = args.initial_prompt
    else:
        programmatic_prompt = stdin_prompt
    if not programmatic_prompt:
        print("Error: No prompt provided for programmatic mode", file=sys.stderr)
        sys.exit(1)
    output_format = OutputFormat(args.output if hasattr(args, "output") else "text")

    try:
        session_intent = _session_intent(args, allow_picker=False)
        final_response = run_programmatic(
            harness_options=LocalHarnessOptions(
                client=ClientDescriptor(
                    info=ClientInfo(
                        name="vibe_programmatic",
                        title="Chartreux programmatic CLI",
                        version=__version__,
                        entrypoint="programmatic",
                        terminal_emulator=detect_terminal(),
                    ),
                    capabilities=ClientCapabilities(callback_kinds=["user_input"]),
                ),
                session_options=SessionOptions(
                    cwd=str(_session_cwd(args)),
                    workspace_roots=list(args.add_dir),
                    enabled_tools=args.enabled_tools,
                    disabled_tools=[*(args.disabled_tools or ()), "ask_user_question"],
                    max_turns=args.max_turns,
                    max_price=args.max_price,
                    max_session_tokens=args.max_tokens,
                    headless=True,
                    trust_workspace=bool(args.trust or args.worktree),
                ),
                session=session_intent,
            ),
            prompt=programmatic_prompt or "",
            output_format=output_format,
        )
        if final_response:
            print(final_response)
        sys.exit(0)
    except ProgrammaticLimitError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except AppServerResponseError as e:
        print(f"Error: {e.error.message}", file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_interactive_mode(args: argparse.Namespace, stdin_prompt: str | None) -> None:
    from chartreux.app_server.local import (
        ClientDescriptor,
        LocalHarness,
        LocalHarnessOptions,
    )
    from chartreux.app_server.protocol import (
        AppServerResponseError,
        ClientCapabilities,
        ClientInfo,
        SessionOptions,
    )
    from chartreux.cli.textual_ui.app import StartupOptions, run_textual_ui

    # --worktree runs in a checkout Chartreux just made from the repo the user
    # launched from, and --trust is the user saying so outright. Both grant the
    # workspace session trust when the session is built, so prompting first
    # would ask about a decision already taken - and for --worktree it would
    # ask again for every new worktree, which is one per session.
    trust_workspace = bool(args.trust or args.worktree)

    harness = LocalHarness(
        LocalHarnessOptions(
            client=ClientDescriptor(
                info=ClientInfo(
                    name="vibe_tui",
                    title="Chartreux Textual",
                    version=__version__,
                    entrypoint="cli",
                    terminal_emulator=detect_terminal(),
                ),
                capabilities=ClientCapabilities(callback_kinds=["user_input"]),
            ),
            session_options=SessionOptions(
                cwd=str(_session_cwd(args)),
                workspace_roots=list(args.add_dir),
                enabled_tools=args.enabled_tools,
                disabled_tools=list(args.disabled_tools or ()),
                trust_workspace=trust_workspace,
            ),
            session=_session_intent(args, allow_picker=True),
        )
    )
    try:
        summary = run_textual_ui(
            start_app_server=harness.connect,
            history_file=HISTORY_FILE.path,
            startup=StartupOptions(
                initial_prompt=args.initial_prompt or stdin_prompt,
                show_resume_picker=args.resume is True,
                is_resuming_session=(
                    args.continue_session or isinstance(args.resume, str)
                ),
                prompt_for_workspace_trust=not trust_workspace,
                resume_session_id=(
                    args.resume if isinstance(args.resume, str) else None
                ),
                continue_latest=bool(args.continue_session),
            ),
        )
    except AppServerResponseError as exc:
        rprint(f"[red]Error:[/] {exc.error.message}")
        sys.exit(1)
    print_session_resume_message(summary)


def run_cli(args: argparse.Namespace) -> None:
    load_dotenv_values()

    if args.setup:
        bootstrap_config_files()
        if not has_usable_terminal():
            print(
                "Interactive setup requires a terminal. Run `chartreux --setup` "
                "from an interactive terminal.",
                file=sys.stderr,
            )
            sys.exit(1)
        from chartreux.setup.onboarding import run_onboarding

        orchestrator = load_config_orchestrator_or_exit()
        run_onboarding(orchestrator=orchestrator)
        sys.exit(0)

    # Read the original stdin before configuration or onboarding can start an
    # interactive TUI and reopen /dev/tty. A prompt explicitly supplied on the
    # command line takes precedence over piped input.
    stdin_isatty = sys.stdin.isatty()
    stdin_prompt = get_prompt_from_stdin()
    is_programmatic = (
        args.prompt is not None
        or args.initial_prompt is not None
        or stdin_prompt is not None
    )
    args.is_programmatic = is_programmatic
    bootstrap_config_files()

    if not is_programmatic:
        restore_interactive_stdin(stdin_isatty=stdin_isatty)

    try:
        require_api_key_or_onboard(
            load_config_orchestrator_or_exit(),
            interactive=not is_programmatic and has_usable_terminal(),
        )
        if is_programmatic:
            _run_programmatic_mode(args=args, stdin_prompt=stdin_prompt)
        else:
            _run_interactive_mode(args=args, stdin_prompt=stdin_prompt)

    except (KeyboardInterrupt, EOFError):
        rprint("\n[dim]Bye![/]")
        sys.exit(0)
