from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from chartreux.core.tools.builtins._shell_argv import ShellToken, split_shell_tokens
from chartreux.core.tools.secret_redaction import scrub_child_env
from chartreux.core.utils.async_subprocess import spawn_registered_process

# Glob metacharacters whose unquoted occurrence a shell pathname-expands.
# Mirrors the permission resolver's _SHELL_GLOB_CHARACTERS.
_GLOB_METACHARACTERS = frozenset("*?[")

_TERMINAL_EXPANSION_REFUSAL = (
    "Command refused: it needs shell pathname expansion, which no terminal "
    "transport binds to a pinned executor contract. Re-run it with the "
    "pattern quoted, or as a literal argv."
)


# --- Immutable executor option contract (v0.7.1 F1pkg1(b)) -------------------
#
# The contract pins, IN THE EXECUTING PROCESS (never a prior probe): the
# executor identity (:func:`shell_executor`, the same binary handed to the
# spawn), every expansion-relevant option family OFF, startup files off via
# the child environment (BASH_ENV/ENV/ZSHENV removed; ZDOTDIR redirected for
# zsh so no user .zshenv loads), and the locale (LC_COLLATE pinned to C so
# bracket ranges collate deterministically). An executor with no contract
# entry is UNKNOWN and fails closed: permission resolution denies
# expansion-bearing commands and the spawn refuses.


class UnknownShellExecutorError(ValueError):
    """The ambient shell has no pinned executor option contract."""


@dataclass(frozen=True)
class ExecutorOptionContract:
    """The pinned expansion behavior of one executing shell binary.

    ``executor`` is the spawned binary path (``None`` = kernel default
    ``/bin/sh``). ``pinned_off_options`` names the expansion-relevant options
    the contract holds OFF; ``startup_env_scrub`` names environment variables
    removed so no startup file can load; ``startup_env_pins``/``locale_pins``
    are (name, value) pins applied to every spawned child. The system-wide
    startup files (/etc/zshenv, /etc/profile) and the shell binary itself stay
    a root-owned trust base, exactly like the interpreter the model runs on.
    """

    executor: str | None
    family: str
    pinned_off_options: frozenset[str]
    startup_env_scrub: frozenset[str]
    startup_env_pins: tuple[tuple[str, str], ...]
    locale_pins: tuple[tuple[str, str], ...]


# bash's expansion options, all default-off and pinned off: extglob changes
# what *(...) parses to, globstar makes ** recursive, nullglob/dotglob/
# failglob/nocaseglob change the match and no-match behavior.
_BASH_PINNED_OFF = frozenset({
    "extglob",
    "globstar",
    "nullglob",
    "dotglob",
    "failglob",
    "nocaseglob",
})
# zsh's expansion options, pinned off. zsh's `**` is recursive by DEFAULT
# (no option), so consecutive stars stay fail-closed in the model itself.
_ZSH_PINNED_OFF = frozenset({
    "extendedglob",
    "globdots",
    "kshglob",
    "nullglob",
    "caseglob",
    "shglob",
})
_POSIX_PINNED_OFF: frozenset[str] = frozenset()

_EXECUTOR_FAMILIES = {
    "bash": ("bash", _BASH_PINNED_OFF),
    "zsh": ("zsh", _ZSH_PINNED_OFF),
    "sh": ("posix", _POSIX_PINNED_OFF),
    "dash": ("posix", _POSIX_PINNED_OFF),
    "ash": ("posix", _POSIX_PINNED_OFF),
    "hush": ("posix", _POSIX_PINNED_OFF),
    "busybox": ("posix", _POSIX_PINNED_OFF),
}

# Environment variables that make a shell load a startup file: bash reads
# $BASH_ENV for every non-interactive shell, POSIX-mode shells read $ENV,
# ksh-style shells read $KSH_ENV; zsh reads $ZDOTDIR/.zshenv always, which
# the zsh pin below redirects away.
_STARTUP_ENV_SCRUB = frozenset({"BASH_ENV", "ENV", "ZSHENV", "KSH_ENV"})
# A path that cannot exist (/dev/null is a file), so zsh finds no .zshenv.
_PINNED_NO_ZDOTDIR = "/dev/null/chartreux-pinned-no-zdotdir"
_LOCALE_PINS = (("LC_CTYPE", "C.UTF-8"), ("LC_COLLATE", "C"))


@lru_cache(maxsize=16)
def executor_option_contract(executor: str | None) -> ExecutorOptionContract | None:
    """The pinned option contract for one executor binary, or None if unknown.

    ``None`` as *executor* means the environment names no shell: the kernel
    default ``/bin/sh`` applies at spawn, so the contract resolves that.
    """
    if executor is None:
        try:
            executor = os.path.realpath("/bin/sh")
        except OSError:
            return None
    name = Path(executor).name
    entry = _EXECUTOR_FAMILIES.get(name)
    if entry is None:
        return None
    family, pinned_off = entry
    pins: tuple[tuple[str, str], ...] = ()
    if family == "zsh":
        pins = (("ZDOTDIR", _PINNED_NO_ZDOTDIR),)
    return ExecutorOptionContract(
        executor=executor,
        family=family,
        pinned_off_options=pinned_off,
        startup_env_scrub=_STARTUP_ENV_SCRUB,
        startup_env_pins=pins,
        locale_pins=_LOCALE_PINS,
    )


def current_executor_contract() -> ExecutorOptionContract | None:
    """The contract of the shell binary shell text is spawned under right now.

    Reads :func:`shell_executor` on every call (the cache is keyed on the
    resolved binary, so a changed ``$SHELL`` is honored) and never a prior
    probe: the identity reported is the one the spawn uses.
    """
    return executor_option_contract(shell_executor())


def apply_executor_contract_env(
    env: dict[str, str], contract: ExecutorOptionContract | None = None
) -> dict[str, str]:
    """Apply the contract's startup and locale pins to a child env.

    With no ``contract`` the current executor's contract is resolved. A spawn
    that already bound its contract (F1pkg1(c)) passes it explicitly, so the
    env pins derive from the same executor read the denial check used.
    """
    if contract is None:
        contract = current_executor_contract()
    if contract is None:
        return env
    env = {
        name: value
        for name, value in env.items()
        if name not in contract.startup_env_scrub
    }
    env.pop("LC_ALL", None)
    for name, value in (*contract.startup_env_pins, *contract.locale_pins):
        env[name] = value
    return env


async def spawn_argv_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> asyncio.subprocess.Process:
    """Run one fully literal argv without any shell in between.

    The same environment and output-stream contract as
    :func:`spawn_shell_command`; nothing in ``argv`` is re-parsed, so a word
    the caller bound cannot be re-expanded into different arguments. This is
    the transport that CONSUMES a bound literal plan (F1pkg1(c)): it never
    falls back to running the original shell text.
    """
    env = _shell_environment(extra_env)
    cwd = cwd or Path.cwd()
    return await spawn_registered_process(
        asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
    )


async def spawn_shell_command(
    command: str,
    *,
    cwd: Path | None = None,
    merge_stderr: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> asyncio.subprocess.Process:
    # F1pkg1(c): the pinned contract is consumed AT EXECUTION, from ONE read
    # of the executor: the binary whose contract is checked is the binary
    # handed to the spawn, and the env pins derive from that same contract, so
    # the approved semantics are exactly the spawned semantics. An unknown
    # executor fails closed before any process starts.
    executor = shell_executor()
    contract = executor_option_contract(executor)
    if contract is None:
        # Fail closed: an unknown executor has no pinned option contract, so
        # its expansion behavior cannot be modeled or pinned.
        raise UnknownShellExecutorError(
            f"no pinned executor option contract for shell {executor!r}; "
            "the command was not run"
        )
    env = _shell_environment(extra_env, contract=contract)
    cwd = cwd or Path.cwd()
    return await spawn_registered_process(
        asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
            if merge_stderr
            else asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=cwd,
            executable=executor,
            start_new_session=True,
        )
    )


def _shell_environment(
    extra_env: Mapping[str, str] | None = None,
    *,
    contract: ExecutorOptionContract | None = None,
) -> dict[str, str]:
    # Chartreux's credential variables never reach shell children unless the
    # user opted specific names back in via credential_env_passthrough.
    env = {
        **scrub_child_env(os.environ),
        "CI": "true",
        "NONINTERACTIVE": "1",
        "NO_TTY": "1",
    }
    # LC_ALL overrides every LC_* category, so a user-set LC_ALL=C (common in
    # CI/containers) would defeat LC_CTYPE below and yield non-UTF-8 output.
    env.pop("LC_ALL", None)
    env = {
        **env,
        "TERM": "dumb",
        "DEBIAN_FRONTEND": "noninteractive",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "LESS": "-FX",
        "LC_CTYPE": "C.UTF-8",
    }
    if extra_env:
        # The project environment (the trusted folder's .env and
        # $CHARTREUX_HOME/.env) only fills names the harness did not set: an
        # explicit process/shell value wins over the .env files.
        env = {**extra_env, **env}
    # The executor option contract's startup and locale pins (F1pkg1(b)) apply
    # LAST, so no project .env can re-introduce a startup-file variable:
    # no BASH_ENV/ENV/ZSHENV/KSH_ENV loads, zsh's ZDOTDIR is redirected away
    # from any user .zshenv, and the collation locale is pinned so bracket
    # ranges are deterministic. A spawn that bound its contract (F1pkg1(c))
    # passes it so the pins come from the same executor read.
    return apply_executor_contract_env(env, contract)


def shell_executor() -> str | None:
    """The shell binary shell-text commands execute under, as spawned.

    Read here and handed to the spawn, so the identity reported is the one the
    executor actually uses and never a separate prior probe. ``None`` means the
    environment names no shell: the kernel default (``/bin/sh``) applies and
    the executor is unnamed rather than pinned.
    """
    return os.environ.get("SHELL")


def terminal_shell_text_refusal(
    command: str, *, args: Sequence[str] | None = None
) -> str | None:
    """The refusal reason for shell text a terminal transport must not expand.

    F1pkg1(a) elected the REFUSAL lane of the design's terminal-executor
    decision (Amendment F). The app-server terminal transports hand shell text
    to an executor this process does not own: the session shell runs under the
    ambient ``$SHELL`` (:func:`shell_executor`) and a client terminal executor
    is spawned by the client, so the permission model's expansion decisions
    describe no executor pinned here. The design's client-capability
    attestation (Amendment X/CC) covers network-namespace locality and transfer
    filesystem/SSH trust context, attests no executor, is client-self-reported,
    and has no entry in the protocol yet (S4b1b creates it), so every terminal
    client is non-attesting by construction. Glob-bearing shell text therefore
    refuses rather than running under an unmodeled expansion contract.

    Only expansion is refused: text with no unquoted glob metacharacter, and a
    literal argv spawned without a shell, are untouched, and a literal argv
    never falls back to shell text. A bare unquoted ``[`` refuses too, even
    where no closing bracket makes it a pattern — the precise pattern grammar
    is the executor option contract's business.
    """
    candidates = [command, *args] if args else [command]
    if any(_carries_unquoted_glob(candidate) for candidate in candidates):
        return _TERMINAL_EXPANSION_REFUSAL
    return None


def _carries_unquoted_glob(text: str) -> bool:
    if not any(character in text for character in _GLOB_METACHARACTERS):
        return False
    try:
        tokens = split_shell_tokens(text)
    except ValueError:
        # Unparseable quoting cannot be proven literal; fail closed.
        return True
    return any(
        isinstance(token, ShellToken) and token.meta_unquoted for token in tokens
    )


__all__ = [
    "ExecutorOptionContract",
    "UnknownShellExecutorError",
    "apply_executor_contract_env",
    "current_executor_contract",
    "executor_option_contract",
    "shell_executor",
    "spawn_argv_command",
    "spawn_shell_command",
    "terminal_shell_text_refusal",
]
