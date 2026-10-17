"""Golden shell-policy corpus for ``_shell_command_policy`` (v0.5.5 stage 3.1).

The corpus is the differential evidence for the W3a/W3b decomposition of
``chartreux/core/tools/builtins/_shell_command_policy.py``: it records, against
the pre-split module, the exact observable behavior of every public entry point
the split moves. Stage 3.1 records it as the FIRST commit on the stage branch;
every later split commit must keep the replay green.

Layout:

- ``shell_policy_corpus/corpus.jsonl`` — one JSON object per line: ``id``,
  ``family``, the call inputs (``argv``, ``cwd``, ...), and ``expect`` (the
  recorded observations). Every observed call is recorded as either
  ``{"value": ...}`` or ``{"exception": {"type", "message"}}``.
- ``shell_policy_corpus/fixtures/`` — the deterministic repository fixture
  tree the I/O entries are recorded against. Git cannot track paths under a
  ``.git`` component, so the committed tree names every ``.git`` entry
  ``git-dir``; :func:`materialize_fixtures` copies the tree to a temp dir and
  renames each ``git-dir`` to ``.git`` before any observation runs.

Families (see :data:`OBSERVERS` for the exact recipe of each):

- ``policy`` — full ``ShellCommandPolicy`` asdict of ``analyze_shell_command_policy``.
- ``paths`` — ``path_candidates`` order (with and without ``inspect_executable``).
- ``git_pure`` — ``canonical_git_command``, ``is_dangerous_git_subcommand``,
  ``git_work_tree_writing_subcommand``, and the full
  ``DangerousCommandClassification`` asdict of ``classify_dangerous_git_command``.
- ``executor`` — ``executor_boundary`` asdict (or exception) per admission mode.
- ``executor_aux`` — ``executor_owned_tokens``, ``is_package_command``,
  ``package_argument_indexes``, ``inline_interpreter_switch``.
- ``argv_helpers`` — ``_command_name``, ``_options_before_double_dash``,
  ``matches_command_prefix`` (the W3a argv normalization surface).
- ``registry`` — the full ``EXECUTOR_REGISTRY`` key list and every grammar asdict.
- ``git_inspection`` — scope resolution, metadata/config discovery, configured
  remotes, work-tree resolution, cwd relocations, and config/hook risk, recorded
  against the materialized fixture tree with paths relative to its root.
- ``push_risk`` — ``git_push_destination_risk`` for literal modeled push calls.
- ``divergence`` — the deliberate divergences pinned end to end: dangling
  ``git -C``/``--git-dir``/``--work-tree``/``-c`` (runtime walker tolerates,
  classifier hard-denies) and ``GIT``-cased executables (canonical form
  case-sensitive, classifier casefolded).

Set-derived outputs are recorded sorted (their iteration order is hash-seed
dependent), and absolute fixture paths are recorded relative to the fixture
root (``<FIXTURES>`` in message text) so the corpus is machine-independent.

Regenerate after an intentional behavior change (never to mask a regression):

    uv run python tests/tools/test_shell_policy_corpus.py --record

Check-only (what CI runs via the round-trip test):

    uv run python tests/tools/test_shell_policy_corpus.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pytest

from chartreux.core.dangerous_commands import GitRepositoryRelocation, ModeledGitCall
from chartreux.core.tools.builtins import (
    _shell_argv,
    _shell_command_policy as policy,
    _shell_executor_grammar,
    _shell_git_classifier,
    _shell_git_inspection,
    _shell_git_policy,
)

CORPUS_DIR = Path(__file__).resolve().parent / "shell_policy_corpus"
CORPUS_PATH = CORPUS_DIR / "corpus.jsonl"
FIXTURES_SOURCE = CORPUS_DIR / "fixtures"
GIT_MARKER = ".git"
PLACEHOLDER = "git-dir"
FIXTURES_TOKEN = "<FIXTURES>"

# ---------------------------------------------------------------------------
# Corpus argv matrices
# ---------------------------------------------------------------------------

GIT_ARGV: list[list[str]] = [
    # Degenerate and bare invocations.
    [],
    ["git"],
    ["git", "-h"],
    ["git", "-v"],
    ["git", "--version"],
    ["git", "version"],
    ["git", "help"],
    # Executable-token normalization.
    ["GIT", "log"],
    ["Git", "status"],
    ["/usr/bin/git", "log"],
    ["/usr/bin/git", "push", "origin", "main"],
    ["git.exe", "log"],
    ["GIT", "push", "origin", "main"],
    # Global-option walk.
    ["git", "-C", "dir", "log"],
    ["git", "-Cdir", "log"],
    ["git", "-C", "a", "-C", "b", "log"],
    ["git", "--git-dir", "g", "log"],
    ["git", "--git-dir=g", "log"],
    ["git", "--work-tree", "w", "log"],
    ["git", "--work-tree=w", "log"],
    ["git", "--git-dir", "g", "--work-tree", "w", "log"],
    ["git", "-c", "k=v", "log"],
    ["git", "-c", "k=v", "-c", "j=w", "log"],
    ["git", "--config-env", "x", "log"],
    ["git", "--exec-path", "p", "log"],
    ["git", "--exec-path=p", "log"],
    ["git", "-p", "log"],
    ["git", "-P", "log"],
    ["git", "--paginate", "log"],
    ["git", "--no-pager", "log"],
    ["git", "--bare", "log"],
    ["git", "--", "log"],
    ["git", "-C", "dir"],
    ["git", "--", "-C", "log"],
    # Dangling and unknown globals.
    ["git", "-C"],
    ["git", "--git-dir"],
    ["git", "--work-tree"],
    ["git", "-c"],
    ["git", "--exec-path"],
    ["git", "--no-such", "log"],
    ["git", "-Q", "log"],
    # Readers.
    ["git", "log"],
    ["git", "log", "--oneline"],
    ["git", "log", "-p"],
    ["git", "status"],
    ["git", "status", "--short"],
    ["git", "diff"],
    ["git", "diff", "HEAD"],
    ["git", "show"],
    ["git", "show", "HEAD:file"],
    ["git", "blame", "f"],
    ["git", "grep", "pat"],
    ["git", "grep", "-r", "pat", "HEAD"],
    ["git", "ls-files"],
    ["git", "branch"],
    ["git", "branch", "-a"],
    ["git", "tag"],
    ["git", "rev-parse", "HEAD"],
    ["git", "rev-list", "HEAD"],
    ["git", "describe"],
    ["git", "cat-file", "-p", "HEAD"],
    ["git", "diff-tree", "HEAD"],
    ["git", "ls-tree", "HEAD"],
    ["git", "merge-base", "a", "b"],
    ["git", "name-rev", "HEAD"],
    ["git", "shortlog"],
    ["git", "show-ref"],
    ["git", "verify-pack", "p"],
    ["git", "whatchanged"],
    ["git", "for-each-ref"],
    ["git", "column"],
    ["git", "count-objects"],
    ["git", "check-attr", "x", "f"],
    ["git", "check-ignore", "f"],
    ["git", "check-mailmap", "a@b"],
    ["git", "check-ref-format", "refs/heads/x"],
    ["git", "diff-files"],
    ["git", "diff-index", "HEAD"],
    ["git", "get-tar-commit-id"],
    ["git", "reflog"],
    ["git", "stash"],
    ["git", "var", "GIT_AUTHOR_IDENT"],
    # push.
    ["git", "push"],
    ["git", "push", "origin"],
    ["git", "push", "origin", "main"],
    ["git", "push", "--force", "origin", "main"],
    ["git", "push", "--force-with-lease", "origin", "main"],
    ["git", "push", "--delete", "origin", "main"],
    ["git", "push", "--mirror", "origin"],
    ["git", "push", "--all", "origin"],
    ["git", "push", "--tags", "origin"],
    ["git", "push", "--follow-tags", "origin"],
    ["git", "push", "-f", "origin", "main"],
    ["git", "push", "-f", "-f", "origin", "main"],
    ["git", "push", "--repo", "https://x/y", "main"],
    ["git", "push", "--repo=https://x/y", "main"],
    ["git", "push", "--receive-pack", "prog", "origin"],
    ["git", "push", "--exec", "prog", "origin"],
    ["git", "push", "-u", "origin"],
    ["git", "push", "--push-option", "k=v", "origin"],
    ["git", "push", "https://example.com/r.git", "main"],
    ["git", "push", "/tmp/dest", "main"],
    ["git", "push", "origin", "main:refs/heads/x"],
    ["git", "push", "origin", ":refs/heads/x"],
    ["git", "push", "origin", "HEAD"],
    ["git", "push", "origin", "refs/heads/main"],
    ["git", "push", "origin", "+refs/heads/main"],
    ["git", "push", "origin", "main", "extra"],
    ["git", "push", "--", "origin"],
    ["git", "push", "--dry-run", "origin"],
    ["git", "push", "--atomic", "origin"],
    ["git", "push", "--no-verify", "origin"],
    ["git", "push", "-n", "origin"],
    ["git", "push", "-q", "origin"],
    ["git", "push", "-v", "origin"],
    # checkout / switch / restore.
    ["git", "checkout"],
    ["git", "checkout", "main"],
    ["git", "checkout", "-b", "new"],
    ["git", "checkout", "-B", "new"],
    ["git", "checkout", "--orphan", "x"],
    ["git", "checkout", "-f", "main"],
    ["git", "checkout", "--force", "main"],
    ["git", "checkout", "--", "path"],
    ["git", "checkout", "main", "--", "path"],
    ["git", "checkout", "--conflict", "merge", "--", "f"],
    ["git", "checkout", "--pathspec-from-file", "f"],
    ["git", "checkout", "--recurse-submodules", "main"],
    ["git", "checkout", "-p", "main"],
    ["git", "switch", "main"],
    ["git", "switch", "-c", "new"],
    ["git", "switch", "-C", "new"],
    ["git", "switch", "--discard-changes", "main"],
    ["git", "switch", "--orphan", "x"],
    ["git", "switch", "-f", "main"],
    ["git", "switch", "--recurse-submodules", "main"],
    ["git", "switch", "--", "path"],
    ["git", "restore", "f"],
    ["git", "restore", "--source", "HEAD", "f"],
    ["git", "restore", "-s", "HEAD", "--", "f"],
    ["git", "restore", "--staged", "f"],
    ["git", "restore", "--pathspec-from-file", "f"],
    ["git", "restore", "--worktree", "f"],
    # stash.
    ["git", "stash", "push"],
    ["git", "stash", "push", "-m", "msg"],
    ["git", "stash", "-m", "msg", "push"],
    ["git", "stash", "pop"],
    ["git", "stash", "apply"],
    ["git", "stash", "drop"],
    ["git", "stash", "clear"],
    ["git", "stash", "list"],
    ["git", "stash", "show"],
    ["git", "stash", "branch", "x"],
    ["git", "stash", "--pathspec-from-file", "f"],
    ["git", "stash", "push", "-u"],
    # reflog.
    ["git", "reflog", "show"],
    ["git", "reflog", "expire", "--expire", "now", "HEAD"],
    ["git", "reflog", "delete", "HEAD@{0}"],
    ["git", "reflog", "expire", "--expire-unreachable", "now", "--all"],
    # reset.
    ["git", "reset"],
    ["git", "reset", "--hard"],
    ["git", "reset", "--soft", "HEAD"],
    ["git", "reset", "--mixed"],
    ["git", "reset", "--keep", "HEAD"],
    ["git", "reset", "--merge", "HEAD"],
    ["git", "reset", "-q", "HEAD"],
    ["git", "reset", "--pathspec-from-file", "f"],
    ["git", "reset", "--", "f"],
    ["git", "reset", "--ha"],
    ["git", "reset", "--hard", "HEAD", "--", "f"],
    ["git", "reset", "HEAD", "--", "f"],
    # clean.
    ["git", "clean", "-f"],
    ["git", "clean", "-fd"],
    ["git", "clean", "-fdx"],
    ["git", "clean", "-n"],
    ["git", "clean", "-f", "-f"],
    ["git", "clean", "-i"],
    ["git", "clean", "-q"],
    ["git", "clean", "-e", "pat", "-fd"],
    ["git", "clean", "--exclude", "pat", "-fd"],
    ["git", "clean", "-X", "dir"],
    ["git", "clean", "-x", "-d"],
    # config.
    ["git", "config", "user.name"],
    ["git", "config", "--get", "user.name"],
    ["git", "config", "--get-all", "x"],
    ["git", "config", "--list"],
    ["git", "config", "-l"],
    ["git", "config", "user.name", "v"],
    ["git", "config", "--add", "remote.origin.push", "refs/heads/*"],
    ["git", "config", "remote.origin.push", "refs/heads/*"],
    ["git", "config", "url.https://x.insteadof", "y"],
    ["git", "config", "branch.main.remote", "origin"],
    ["git", "config", "branch.main.pushremote", "origin"],
    ["git", "config", "remote.origin.mirror", "true"],
    ["git", "config", "core.hookspath", "hooks"],
    ["git", "config", "core.editor", "vim"],
    ["git", "config", "core.sshcommand", "ssh"],
    ["git", "config", "core.askpass", "prog"],
    ["git", "config", "core.gitproxy", "proxy"],
    ["git", "config", "sequence.editor", "vim"],
    ["git", "config", "filter.x.clean", "prog"],
    ["git", "config", "filter.x.process", "prog"],
    ["git", "config", "merge.x.driver", "prog"],
    ["git", "config", "diff.x.command", "prog"],
    ["git", "config", "diff.x.textconv", "prog"],
    ["git", "config", "diff.x.external", "prog"],
    ["git", "config", "man.x.path", "prog"],
    ["git", "config", "man.x.viewer", "less"],
    ["git", "config", "interactive.difffilter", "prog"],
    ["git", "config", "include.path", "f"],
    ["git", "config", "includeif.gitdir:/x.path", "f"],
    ["git", "config", "includeif.onbranch:x.path", "f"],
    ["git", "config", "--unset", "user.name"],
    ["git", "config", "--unset-all", "x"],
    ["git", "config", "--rename-section", "core", "core2"],
    ["git", "config", "--remove-section", "core"],
    ["git", "config", "--rename-section", "filter.x", "core"],
    ["git", "config", "--file", "f", "user.name", "v"],
    ["git", "config", "-f", "f", "user.name", "v"],
    ["git", "config", "--blob", "b", "x", "v"],
    ["git", "config", "set", "user.name", "v"],
    ["git", "config", "get", "user.name"],
    ["git", "config", "edit"],
    ["git", "config", "--type", "int", "x"],
    ["git", "config", "--default", "v", "x"],
    ["git", "config", "--comment", "c", "x", "v"],
    ["git", "config", "-e"],
    ["git", "config", "-z"],
    ["git", "config", "--null", "x"],
    ["git", "config", "--show-origin", "x"],
    # remote.
    ["git", "remote"],
    ["git", "remote", "-v"],
    ["git", "remote", "add", "origin", "url"],
    ["git", "remote", "set-url", "origin", "url"],
    ["git", "remote", "rename", "a", "b"],
    ["git", "remote", "remove", "origin"],
    ["git", "remote", "rm", "origin"],
    ["git", "remote", "prune", "origin"],
    ["git", "remote", "update", "origin"],
    ["git", "remote", "show", "origin"],
    ["git", "remote", "get-url", "origin"],
    # worktree / submodule.
    ["git", "worktree", "list"],
    ["git", "worktree", "add", "path"],
    ["git", "worktree", "add", "-b", "br", "path"],
    ["git", "worktree", "add", "--reason", "r", "path"],
    ["git", "worktree", "add", "--detach", "path"],
    ["git", "worktree", "move", "old", "new"],
    ["git", "worktree", "move", "-f", "old", "new"],
    ["git", "worktree", "remove", "path"],
    ["git", "worktree", "prune"],
    ["git", "worktree", "repair"],
    ["git", "submodule", "add", "url", "path"],
    [
        "git",
        "submodule",
        "add",
        "-b",
        "br",
        "--reference",
        "r",
        "--separate-git-dir",
        "d",
        "url",
        "path",
    ],
    ["git", "submodule", "status"],
    ["git", "submodule", "update", "--init"],
    # archive / apply / init.
    ["git", "archive", "HEAD"],
    ["git", "archive", "--output", "a.tar", "HEAD"],
    ["git", "archive", "-o", "a.tar", "HEAD"],
    ["git", "archive", "--output=a.tar", "HEAD"],
    ["git", "apply", "p.diff"],
    ["git", "apply", "--directory", "d", "p.diff"],
    ["git", "apply", "--unsafe-paths", "p.diff"],
    ["git", "apply", "--3way", "p.diff"],
    ["git", "init"],
    ["git", "init", "dir"],
    ["git", "init", "--bare", "dir"],
    ["git", "init", "--initial-branch", "main", "dir"],
    ["git", "init", "--separate-git-dir", "g", "dir"],
    ["git", "init", "--shared", "dir"],
    ["git", "init", "--shared=group", "dir"],
    # Other non-readers.
    ["git", "add", "f"],
    ["git", "add", "-p", "f"],
    ["git", "add", "-A"],
    ["git", "commit", "-m", "x"],
    ["git", "commit", "--amend"],
    ["git", "rm", "f"],
    ["git", "rm", "-r", "d"],
    ["git", "rm", "--cached", "f"],
    ["git", "mv", "a", "b"],
    ["git", "clone", "url"],
    ["git", "clone", "--bare", "url"],
    ["git", "clone", "-b", "br", "url"],
    ["git", "fetch", "origin"],
    ["git", "fetch", "--all"],
    ["git", "pull"],
    ["git", "pull", "origin", "main"],
    ["git", "merge", "branch"],
    ["git", "cherry-pick", "abc"],
    ["git", "revert", "abc"],
    ["git", "rebase", "main"],
    ["git", "am", "--abort"],
    ["git", "bisect", "start"],
    ["git", "bisect", "reset"],
    ["git", "notes", "add", "-m", "x"],
    ["git", "help", "log"],
    # Unmodeled builtins and unrecognized subcommands.
    ["git", "gc"],
    ["git", "fsck", "--lost-found"],
    ["git", "bundle", "create", "b"],
    ["git", "daemon"],
    ["git", "filter-branch"],
    ["git", "update-ref", "refs/heads/x", "abc"],
    ["git", "receive-pack", "d"],
    ["git", "upload-pack", "d"],
    ["git", "credential", "fill"],
    ["git", "nosuchcmd"],
    ["git", "log", "--no-such-option"],
    ["git", "status", "--no-such"],
]

EXECUTOR_ARGV: list[list[str]] = [
    # uv.
    ["uv", "run", "cmd"],
    ["uv", "run", "cmd", "arg"],
    ["uv", "run", "--directory", "d", "cmd"],
    ["uv", "run", "--directory=d", "cmd"],
    ["uv", "run", "--project", "p", "cmd"],
    ["uv", "run", "--with", "pkg", "cmd"],
    ["uv", "run", "--with-editable", "./e", "cmd"],
    ["uv", "run", "--with-requirements", "r.txt", "cmd"],
    ["uv", "run", "--no-sync", "cmd"],
    ["uv", "run", "-q", "-v", "cmd"],
    ["uv", "run", "-p", "3.12", "cmd"],
    ["uv", "run", "--python", "3.12", "cmd"],
    ["uv", "pip", "install", "pkg"],
    ["uv", "pip", "install", "-r", "req.txt"],
    ["uv", "pip", "install", "./local"],
    ["uv", "pip", "install", "--python", "3.12", "pkg"],
    ["uv", "pip", "list"],
    ["uv", "--version"],
    ["uv", "run", "--no-such", "cmd"],
    ["uv", "run", "--project"],
    ["uv", "run", "--project="],
    ["uv", "run"],
    ["uv", "run", "--", "cmd"],
    # npx.
    ["npx", "pkg"],
    ["npx", "pkg", "arg"],
    ["npx", "-p", "pkg", "cmd"],
    ["npx", "-p", "a", "-p", "b", "cmd"],
    ["npx", "--yes", "pkg"],
    ["npx", "--no", "pkg"],
    ["npx", "--no-install", "pkg"],
    ["npx", "--quiet", "pkg"],
    ["npx", "--package", "a", "cmd"],
    ["npx", "-y", "-n", "pkg"],
    ["npx", "--registry", "r", "pkg"],
    ["npx"],
    # pipx.
    ["pipx", "run", "pkg"],
    ["pipx", "run", "--spec", "s", "pkg"],
    ["pipx", "run", "--python", "3.12", "pkg"],
    ["pipx", "run", "pkg", "arg"],
    ["pipx", "install", "pkg"],
    ["pipx", "run", "--verbose", "pkg"],
    ["pipx"],
    # go.
    ["go", "run", "main.go"],
    ["go", "run", "main.go", "arg"],
    ["go", "run", "./pkg"],
    ["go", "run", "-race", "main.go"],
    ["go", "run", "--tags", "t", "main.go"],
    ["go", "run", "-C", "d", "main.go"],
    ["go", "-C", "d", "run", "main.go"],
    ["go", "-C", "d", "run", "main.go", "arg"],
    ["go", "run", "--mod", "m", "main.go"],
    ["go", "build", "./..."],
    ["go", "run"],
    ["go", "run", "--bogus", "main.go"],
    ["go", "run", "--", "main.go"],
    ["go", "-C", "d"],
    # cargo.
    ["cargo", "run"],
    ["cargo", "run", "--manifest-path", "C.toml"],
    ["cargo", "run", "--manifest-path=C.toml"],
    ["cargo", "run", "--bin", "b"],
    ["cargo", "run", "--example", "e"],
    ["cargo", "run", "--package", "p"],
    ["cargo", "run", "-p", "p"],
    ["cargo", "run", "--release", "--", "--x"],
    ["cargo", "run", "--quiet", "--verbose", "--", "a"],
    ["cargo", "run", "--target", "t", "--", "a"],
    ["cargo", "run", "--", "arg"],
    ["cargo", "run", "arg"],
    ["cargo", "build"],
    ["cargo", "run", "--config", "k=v", "--", "a"],
    ["cargo", "run", "--target-dir", "d", "--", "a"],
    ["cargo", "run", "-j", "4", "--", "a"],
    ["cargo", "run", "-F", "f", "--", "a"],
    # script.
    ["script", "out.log", "cmd"],
    ["script", "out.log", "cmd", "arg"],
    ["script", "-c", "cmd", "out"],
    ["script", "-c", "cmd"],
    ["script", "cmd"],
    ["script"],
    # env.
    ["env", "cmd"],
    ["env", "FOO=1", "cmd"],
    ["env", "FOO=1", "BAR=2", "cmd", "arg"],
    ["env", "-i", "cmd"],
    ["env", "-i", "FOO=1", "cmd"],
    ["env", "--ignore-environment", "cmd"],
    ["env", "-u", "F", "cmd"],
    ["env", "--unset", "F", "cmd"],
    ["env", "-C", "d", "cmd"],
    ["env", "--chdir", "d", "cmd"],
    ["env", "-C", "d", "-C", "e", "cmd"],
    ["env", "-C", "d", "FOO=1", "cmd"],
    ["env", "-", "cmd"],
    ["env", "FOO", "cmd"],
    ["env", "FOO=1"],
    ["env"],
    ["env", "-x", "cmd"],
    # eval / command / builtin / exec.
    ["eval", "echo", "hi"],
    ["eval", "echo hi"],
    ["eval", ""],
    ["eval"],
    ["command", "-v", "python"],
    ["command", "-V", "python"],
    ["command", "python", "-c", "x"],
    ["command", "-p", "python"],
    ["command", "python"],
    ["command"],
    ["builtin", "cd", "d"],
    ["builtin", "echo", "hi"],
    ["builtin"],
    ["exec", "python", "-c", "x"],
    ["exec", "-a", "nm", "cmd"],
    ["exec", "-c", "cmd"],
    ["exec", "-l", "cmd"],
    ["exec", "-cl", "cmd"],
    ["exec", "cmd"],
    ["exec"],
    # busybox / toybox.
    ["busybox", "sh", "-c", "x"],
    ["busybox", "ash", "-c", "x"],
    ["busybox", "bash", "-c", "x"],
    ["busybox", "dash", "-c", "x"],
    ["busybox", "hush", "-c", "x"],
    ["busybox", "ls", "-l"],
    ["busybox", "cp", "a", "b"],
    ["busybox"],
    ["toybox", "sh", "-c", "x"],
    ["toybox", "ash", "-c", "x"],
    ["toybox", "cp", "a", "b"],
    ["toybox", "ls"],
    ["toybox"],
    # timeout / nice / stdbuf.
    ["timeout", "10", "cmd"],
    ["timeout", "10", "cmd", "arg"],
    ["timeout", "--signal", "KILL", "10", "cmd"],
    ["timeout", "--signal=KILL", "10", "cmd"],
    ["timeout", "-s", "KILL", "10", "cmd"],
    ["timeout", "-k", "5", "10", "cmd"],
    ["timeout", "--kill-after", "5", "10", "cmd"],
    ["timeout", "--foreground", "10", "cmd"],
    ["timeout", "--preserve-status", "10", "cmd"],
    ["timeout", "--verbose", "10", "cmd"],
    ["timeout", "-v", "10", "cmd"],
    ["timeout", "cmd"],
    ["timeout", "10"],
    ["timeout"],
    ["nice", "-n", "5", "cmd"],
    ["nice", "--adjustment", "5", "cmd"],
    ["nice", "-n5", "cmd"],
    ["nice", "-5", "cmd"],
    ["nice", "cmd"],
    ["nice"],
    ["stdbuf", "-o", "0", "cmd"],
    ["stdbuf", "-o0", "cmd"],
    ["stdbuf", "-i", "0", "-o", "0", "-e", "0", "cmd"],
    ["stdbuf", "-i0", "-o0", "-e0", "cmd"],
    ["stdbuf", "cmd"],
    ["stdbuf"],
    # flock / setsid / nohup.
    ["flock", "lock", "cmd"],
    ["flock", "-x", "lock", "cmd"],
    ["flock", "-s", "lock", "cmd"],
    ["flock", "--shared", "lock", "cmd"],
    ["flock", "--exclusive", "lock", "cmd"],
    ["flock", "--wait", "5", "lock", "cmd"],
    ["flock", "-w", "5", "lock", "cmd"],
    ["flock", "--conflict-exit-code", "1", "lock", "cmd"],
    ["flock", "-E", "1", "lock", "cmd"],
    ["flock", "lock", "cmd", "arg"],
    ["flock", "lock"],
    ["flock"],
    ["setsid", "cmd"],
    ["setsid", "--fork", "cmd"],
    ["setsid", "--ctty", "cmd"],
    ["setsid", "--wait", "cmd"],
    ["setsid", "-c", "cmd"],
    ["setsid", "-f", "cmd"],
    ["setsid", "-w", "cmd"],
    ["setsid"],
    ["nohup", "cmd"],
    ["nohup", "cmd", "arg"],
    ["nohup"],
    # ionice / taskset / time.
    ["ionice", "-c", "2", "cmd"],
    ["ionice", "-n", "5", "cmd"],
    ["ionice", "-p", "1", "cmd"],
    ["ionice", "-P", "1"],
    ["ionice", "--class", "2", "cmd"],
    ["ionice", "--classdata", "x", "cmd"],
    ["ionice", "--ignore", "cmd"],
    ["ionice", "cmd"],
    ["ionice"],
    ["taskset", "-c", "0", "cmd"],
    ["taskset", "--cpu-list", "0", "cmd"],
    ["taskset", "-a", "cmd"],
    ["taskset", "cmd"],
    ["taskset", "0x1", "cmd"],
    ["taskset"],
    ["time", "cmd"],
    ["time", "-f", "fmt", "cmd"],
    ["time", "--format", "fmt", "cmd"],
    ["time", "-o", "f", "cmd"],
    ["time", "--output", "f", "cmd"],
    ["time", "-a", "-o", "f", "cmd"],
    ["time", "-p", "cmd"],
    ["time", "-v", "cmd"],
    ["time", "-q", "cmd"],
    ["time"],
    # Shells.
    ["bash", "-c", "echo hi"],
    ["bash", "-c", "echo hi", "arg0"],
    ["sh", "-c", "echo hi"],
    ["bash", "-lc", "x"],
    ["bash", "-l", "-c", "x"],
    ["bash", "--rcfile", "f", "-i"],
    ["bash", "--init-file", "f"],
    ["bash", "-i"],
    ["bash", "-s"],
    ["bash", "-r"],
    ["bash", "+o", "history"],
    ["bash", "+O", "vi"],
    ["bash", "-c"],
    ["bash", "--", "-c", "x"],
    ["bash", "script.sh"],
    ["bash"],
    ["sh", "-c", "x", "name"],
    ["zsh", "-c", "x"],
    ["ksh", "-c", "x"],
    ["dash", "-c", "x"],
    ["ash", "-c", "x"],
    ["hush", "-c", "x"],
    # Python family: inline switches and the module-executor path.
    ["python", "-c", "print(1)"],
    ["python", "-c", "print(1)", "arg"],
    ["python", "-c"],
    ["python", "-B", "-c", "x"],
    ["python", "-E", "-S", "-c", "x"],
    ["python", "-I", "-c", "x"],
    ["python", "-u", "-c", "x"],
    ["python", "-q", "-c", "x"],
    ["python", "-v", "-c", "x"],
    ["python", "-W", "ignore", "-c", "x"],
    ["python", "-X", "utf8", "-c", "x"],
    ["python", "-d", "-c", "x"],
    ["python", "-O", "-c", "x"],
    ["python", "-OO", "-c", "x"],
    ["python", "-s", "-c", "x"],
    ["python", "-S", "-c", "x"],
    ["python", "-b", "-c", "x"],
    ["python", "-h"],
    ["python", "--check-hash-based-pycs", "always", "-c", "x"],
    ["python", "script.py"],
    ["python", "script.py", "arg"],
    ["python"],
    ["python", "-m", "http.server"],
    ["python", "-m", "http.server", "8080"],
    ["python", "-m", "pip", "install", "x"],
    ["python", "-m", "pip"],
    ["python", "-m", "uv"],
    ["python", "-m", "uv", "run", "cmd"],
    ["python", "-m", "unknown.mod"],
    ["python", "-m", "unknown.mod", "arg"],
    ["python", "-m", "a.b"],
    ["python", "-m", "123"],
    ["python", "-m", "-x"],
    ["python", "-m"],
    ["python3", "-c", "x"],
    ["python3", "-m", "venv", "v"],
    ["pypy", "-c", "x"],
    ["pypy3", "-m", "pip"],
    ["python", "-c", "x", "-m", "y"],
    ["python", "-m", "os", "-c", "x"],
    # node / perl / ruby.
    ["node", "-e", "code"],
    ["node", "--eval", "code"],
    ["node", "-p", "code"],
    ["node", "--print", "code"],
    ["node", "-r", "mod", "-e", "x"],
    ["node", "--require", "mod", "-e", "x"],
    ["node", "-i", "-e", "x"],
    ["node", "--input-type", "module", "-e", "x"],
    ["node", "--no-warnings", "-e", "x"],
    ["node", "script.js"],
    ["node", "-e"],
    ["node", "-e", "x", "-e", "y"],
    ["perl", "-e", "code"],
    ["perl", "-w", "-e", "x"],
    ["perl", "-n", "-e", "x"],
    ["perl", "-p", "-e", "x"],
    ["perl", "-l", "-e", "x"],
    ["perl", "-T", "-e", "x"],
    ["perl", "-M", "Mod", "-e", "x"],
    ["perl", "-I", "dir", "-e", "x"],
    ["perl", "-F", "pat", "-e", "x"],
    ["perl", "script.pl"],
    ["perl", "-e"],
    ["perl", "-e", "x", "-w"],
    ["ruby", "-e", "code"],
    ["ruby", "-w", "-e", "x"],
    ["ruby", "-n", "-e", "x"],
    ["ruby", "-p", "-e", "x"],
    ["ruby", "-d", "-e", "x"],
    ["ruby", "-v", "-e", "x"],
    ["ruby", "-I", "dir", "-e", "x"],
    ["ruby", "-C", "d", "-e", "x"],
    ["ruby", "-E", "UTF-8", "-e", "x"],
    ["ruby", "-F", "pat", "-e", "x"],
    ["ruby", "-K", "u", "-e", "x"],
    ["ruby", "-T", "-e", "x"],
    ["ruby", "-W", "-e", "x"],
    ["ruby", "-W0", "-e", "x"],
    ["ruby", "--disable", "gems", "-e", "x"],
    ["ruby", "--enable", "gems", "-e", "x"],
    ["ruby", "script.rb"],
    ["ruby", "-e"],
    ["ruby", "-e", "x", "-w"],
    # Nesting and depth.
    ["uv", "run", "python", "-c", "x"],
    ["env", "FOO=1", "uv", "run", "cmd"],
    ["bash", "-c", "uv run x"],
    ["timeout", "10", "npx", "pkg"],
    ["npx", "python", "-c", "x"],
    ["env", "A=1", "env", "B=2", "env", "C=3", "env", "D=4", "env", "E=5", "cmd"],
    [
        "env",
        "A=1",
        "env",
        "B=2",
        "env",
        "C=3",
        "env",
        "D=4",
        "env",
        "E=5",
        "env",
        "F=6",
        "env",
        "G=7",
        "env",
        "H=8",
        "env",
        "I=9",
        "cmd",
    ],
]

TOOL_ARGV: list[list[str]] = [
    # Listing / reading.
    ["ls", "-l", "dir"],
    ["echo", "hi"],
    ["cat", "f"],
    ["grep", "-r", "pat", "dir"],
    ["grep", "-rn", "p", "."],
    ["file", "f"],
    ["du", "-sh", "d"],
    ["wc", "-l", "f"],
    ["diff", "a", "b"],
    ["tree", "d"],
    ["more", "f"],
    # find.
    ["find", ".", "-name", "*.py"],
    ["find", ".", "-exec", "rm", "{}", ";"],
    ["find", ".", "-delete"],
    ["find", "d", "-exec", "cmd", "{}", "+"],
    # rm / chmod / chown / mv / cp / ln / install / truncate.
    ["rm", "-r", "dir"],
    ["rm", "-rf", "/"],
    ["rm", "--no-preserve-root", "/"],
    ["rm", "-f", "f"],
    ["rm", "--", "-f"],
    ["rm", "-d", "d"],
    ["chmod", "+x", "f"],
    ["chmod", "-R", "755", "d"],
    ["chmod", "u+x", "f"],
    ["chmod", "755", "f"],
    ["chmod", "0777", "f"],
    ["chmod", "07777", "f"],
    ["chmod", "-R", "a+x", "d"],
    ["chown", "-R", "root", "f"],
    ["chown", "root", "f"],
    ["mv", "a", "b"],
    ["cp", "a", "b"],
    ["cp", "-r", "a", "b"],
    ["ln", "-s", "a", "b"],
    ["install", "a", "b"],
    ["install", "-D", "a", "b"],
    ["truncate", "-s", "0", "f"],
    # dd / tee.
    ["dd", "if=a", "of=b"],
    ["dd", "if=", "of="],
    ["tee", "f"],
    ["tee", "-a", "f"],
    # sed / less.
    ["sed", "-i", "s/x/y/", "f"],
    ["sed", "-i.bak", "s/x/y/", "f"],
    ["sed", "--in-place", "s/x/y/", "f"],
    ["sed", "-e", "s/x/y/", "f"],
    ["sed", "-f", "script", "f"],
    ["sed", "-n", "p", "f"],
    ["sed", "s/x/y/", "f", "g"],
    ["sed", "--expression", "s/x/y/", "--file", "s", "f"],
    ["less", "+F", "f"],
    ["less", "+/pat", "f"],
    ["less", "-k", "f"],
    ["less", "+g", "f"],
    ["less", "-o", "log", "f"],
    ["less", "+G", "f"],
    # date / sort / uniq / checksums.
    ["date"],
    ["date", "-s", "now"],
    ["date", "-f", "f"],
    ["date", "--set", "now"],
    ["date", "-j", "-f", "%Y", "2020"],
    ["date", "+%Y-%m-%d"],
    ["sort", "-o", "out", "in"],
    ["sort", "--output", "out", "in"],
    ["sort", "-u", "in"],
    ["uniq", "in"],
    ["uniq", "in", "out"],
    ["uniq", "-c", "in"],
    ["md5sum", "--check", "sums"],
    ["md5sum", "-c", "sums"],
    ["sha256sum", "--check", "s"],
    ["shasum", "-c", "s"],
    # archives.
    ["tar", "-xf", "a.tar"],
    ["tar", "-xf", "a.tar", "-C", "d"],
    ["tar", "--directory", "d", "-xf", "a"],
    ["tar", "-cf", "a.tar", "d"],
    ["tar", "-xf", "a.tgz", "--wildcards", "*"],
    ["unzip", "a.zip", "-d", "d"],
    ["zip", "a.zip", "f"],
    ["zip", "-r", "a.zip", "d"],
    ["7z", "x", "a.7z"],
    ["7za", "a", "a.7z", "f"],
    ["7zr", "x", "a.7z"],
    ["unar", "a.zip"],
    ["zstd", "-d", "f.zst"],
    ["zstdcat", "f.zst"],
    ["unzstd", "f.zst"],
    ["zstdmt", "-d", "f"],
    ["bsdtar", "-xf", "a"],
    ["pax", "-w", "f"],
    # split / csplit / cpio / strip / openssl.
    ["split", "-b", "1k", "f", "p"],
    ["split", "--prefix", "p", "f"],
    ["csplit", "--prefix", "p", "f", "/pat/"],
    ["cpio", "-o", ">", "names"],
    ["cpio", "-i", "<", "archive"],
    ["strip", "bin"],
    ["strip", "-o", "out", "in"],
    ["openssl", "enc", "-out", "f"],
    ["openssl", "--out", "f", "enc"],
    # compilers / linkers / archivers.
    ["gcc", "-o", "bin", "src.c"],
    ["gcc", "-o.bin", "src.c"],
    ["cc", "-o", "bin", "s.c"],
    ["clang", "--output", "bin", "s.c"],
    ["g++", "-o", "bin", "s.cpp"],
    ["clang++", "-o", "bin", "s.cpp"],
    ["c++", "-o", "bin", "s.cpp"],
    ["rustc", "-o", "bin", "s.rs"],
    ["as", "-o", "o", "s"],
    ["ld", "-o", "bin", "o"],
    ["ar", "rcs", "a.a", "o"],
    # xargs / gpg / rsync / patch / iconv / xmllint / mktemp / scp.
    ["xargs", "cmd"],
    ["xargs", "-0", "cmd"],
    ["xargs", "-I{}", "cmd", "{}"],
    ["xargs", "-d", "\n", "cmd"],
    ["gpg", "--sign", "f"],
    ["gpg", "-o", "f", "--sign"],
    ["rsync", "a", "b"],
    ["rsync", "-a", "src/", "dst"],
    ["rsync", "--rsync-path", "cmd", "a", "b"],
    ["rsync", "-e", "ssh", "a", "b"],
    ["patch", "f.diff"],
    ["patch", "-d", "d", "f.diff"],
    ["patch", "--dry-run", "f"],
    ["iconv", "-f", "UTF-8", "-t", "LATIN1", "f"],
    ["xmllint", "--output", "f", "x.xml"],
    ["mktemp"],
    ["mktemp", "tpl"],
    ["mktemp", "-d"],
    ["mktemp", "--suffix", "s", "t"],
    ["scp", "a", "b"],
    ["scp", "-r", "a", "b"],
    ["scp", "-P", "22", "a", "b"],
    ["scp", "-o", "Opt=v", "a", "b"],
    # Package managers outside the executor registry path.
    ["npm", "install", "pkg"],
    ["npm", "run", "build"],
    ["pip", "install", "pkg"],
    ["pip", "install", "-r", "r.txt"],
    ["pip3", "install", "--user", "pkg"],
    ["pnpm", "add", "pkg"],
    ["yarn", "add", "pkg"],
    ["bun", "install", "pkg"],
    ["poetry", "add", "pkg"],
    # Degenerate controls.
    ["x"],
    ["ls"],
]


def _dedup(argv_lists: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for argv in argv_lists:
        key = tuple(argv)
        if key not in seen:
            seen.add(key)
            unique.append(argv)
    return unique


ALL_POLICY_ARGV = _dedup([*GIT_ARGV, *EXECUTOR_ARGV, *TOOL_ARGV])
GIT_PURE_ARGV = _dedup([*GIT_ARGV, ["ls", "-l"], ["cat", "f"], ["uv", "run", "cmd"]])
DIVERGENCE_ARGV: list[list[str]] = [
    ["git", "-C"],
    ["git", "--git-dir"],
    ["git", "--work-tree"],
    ["git", "-c"],
    ["GIT", "log"],
    ["GIT", "push", "origin", "main"],
    ["Git", "status"],
]

ARGV_HELPER_CASES: list[dict[str, Any]] = [
    *(
        {"call": "command_name", "token": token}
        for token in [
            "git",
            "GIT",
            "Git",
            "gIt",
            "/usr/bin/git",
            "/usr/bin/GIT",
            "./git",
            "../bin/git",
            "git.exe",
            "~/bin/git",
            '"git"',
            "'git'",
            "git ",
            " git",
            "  git  ",
            "C:\\Tools\\git.exe",
            "sh",
            "SH",
            "Ls",
            "lS",
            "/usr/local/bin/python3",
            "python3.12",
            "python3.12.exe",
            "",
            "-",
            "--",
            "a/b/c/ls",
            "git/",
            "/git",
            "GIT.EXE",
            "Git.Exe",
        ]
    ),
    *(
        {"call": "options_before_double_dash", "args": args}
        for args in [
            [],
            ["--"],
            ["a", "--"],
            ["--", "a"],
            ["a", "b"],
            ["-x", "--", "-y"],
            ["--", "--"],
            ["a", "--", "b", "--", "c"],
            ["--", "a", "--", "b"],
        ]
    ),
    *(
        {"call": "matches_command_prefix", "tokens": tokens, "pattern": pattern}
        for tokens, pattern in [
            (["git", "log"], ["git"]),
            (["GIT", "log"], ["git"]),
            (["/usr/bin/git", "log"], ["git"]),
            (["git"], ["git", "log"]),
            ([], ["git"]),
            (["git", "log"], ["GIT"]),
            (["git", "log"], ["/usr/bin/git"]),
            (["git", "x"], ["git", "y"]),
            (["git", "--", "log"], ["git", "--"]),
            (["ls", "-l"], ["ls", "-l", "-a"]),
            (["git", "log"], ["git", "log"]),
            (["git", "log", "x"], ["git", "log"]),
        ]
    ),
]

GIT_INSPECTION_CASES: list[dict[str, Any]] = [
    {"argv": ["git", "log"], "cwd": "plain-repo"},
    {"argv": ["git", "status"], "cwd": "plain-repo"},
    {"argv": ["git", "diff", "HEAD"], "cwd": "plain-repo"},
    {"argv": ["git", "log", "--oneline"], "cwd": "plain-repo"},
    {"argv": ["git", "--no-pager", "log"], "cwd": "plain-repo"},
    {"argv": ["git", "-P", "log"], "cwd": "plain-repo"},
    {"argv": ["git", "commit", "-m", "x"], "cwd": "plain-repo"},
    {"argv": ["git", "push", "origin", "main"], "cwd": "plain-repo"},
    {"argv": ["git", "-C", "nested", "log"], "cwd": "plain-repo"},
    {"argv": ["git", "log"], "cwd": "plain-repo/nested"},
    {
        "argv": ["git", "log"],
        "cwd": "plain-repo",
        "relocation": {"cwd_paths": ["nested"]},
    },
    {"argv": ["git", "log"], "cwd": "plain-repo", "relocation": {"git_dir": ".git"}},
    {
        "argv": ["git", "log"],
        "cwd": "plain-repo",
        "relocation": {"work_tree": "nested"},
    },
    {
        "argv": ["git", "log"],
        "cwd": "plain-repo",
        "relocation": {"cwd_paths": ["../no-repo"]},
    },
    {
        "argv": ["git", "log"],
        "cwd": "plain-repo",
        "relocation": {"git_dir": "../separate-gitdir"},
    },
    {"argv": ["git", "--git-dir", ".git", "log"], "cwd": "plain-repo"},
    {
        "argv": ["git", "--git-dir", ".git", "--work-tree", "nested", "log"],
        "cwd": "plain-repo",
    },
    {"argv": ["git", "-C", "nested", "-C", "..", "log"], "cwd": "plain-repo"},
    {"argv": ["git", "-C"], "cwd": "plain-repo"},
    {"argv": ["git", "log"], "cwd": "pager-repo"},
    {"argv": ["git", "status"], "cwd": "pager-repo"},
    {"argv": ["git", "diff"], "cwd": "pager-repo"},
    {"argv": ["git", "--no-pager", "log"], "cwd": "pager-repo"},
    {"argv": ["git", "-P", "log"], "cwd": "pager-repo"},
    {"argv": ["git", "--paginate", "log"], "cwd": "pager-repo"},
    {"argv": ["git", "commit", "-m", "x"], "cwd": "pager-repo"},
    {"argv": ["git", "diff"], "cwd": "pager-section-repo"},
    {"argv": ["git", "log"], "cwd": "pager-section-repo"},
    {"argv": ["git", "status"], "cwd": "hook-repo"},
    {"argv": ["git", "log"], "cwd": "hook-repo"},
    {"argv": ["git", "diff", "HEAD"], "cwd": "hook-repo"},
    {"argv": ["git", "commit", "-m", "x"], "cwd": "hook-repo"},
    {"argv": ["git", "status"], "cwd": "hookspath-repo"},
    {"argv": ["git", "log"], "cwd": "hookspath-repo"},
    {"argv": ["git", "log"], "cwd": "unreadable-config-repo"},
    {"argv": ["git", "status"], "cwd": "unreadable-config-repo"},
    {"argv": ["git", "log"], "cwd": "gitfile-repo"},
    {"argv": ["git", "status"], "cwd": "gitfile-repo"},
    {"argv": ["git", "log"], "cwd": "separate-gitdir"},
    {"argv": ["git", "status"], "cwd": "separate-gitdir"},
    {"argv": ["git", "log"], "cwd": "linked-worktree"},
    {"argv": ["git", "status"], "cwd": "linked-worktree"},
    {"argv": ["git", "log"], "cwd": "main-repo"},
    {"argv": ["git", "status"], "cwd": "main-repo"},
    {"argv": ["git", "--no-pager", "log"], "cwd": "main-repo"},
    {"argv": ["git", "log"], "cwd": "bare-repo"},
    {"argv": ["git", "status"], "cwd": "bare-repo"},
    {"argv": ["git", "log"], "cwd": "no-repo"},
    {"argv": ["git", "-C", "../plain-repo", "log"], "cwd": "no-repo"},
    {
        "argv": ["git", "log"],
        "cwd": "no-repo",
        "relocation": {"git_dir": "../plain-repo/.git"},
    },
    {"argv": ["git", "log"], "cwd": "garbage-gitfile-repo"},
    {"argv": ["git", "log"], "cwd": "dangling-gitfile-repo"},
    {"argv": ["git", "status"], "cwd": "dangling-gitfile-repo"},
]

PUSH_RISK_CASES: list[dict[str, Any]] = [
    {
        "argv": ["git", "push", "origin", "main"],
        "cwd": "plain-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["origin", "main"],
        },
    },
    {
        "argv": ["git", "push", "upstream", "main"],
        "cwd": "plain-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["upstream", "main"],
        },
    },
    {
        "argv": ["git", "push", "https://x/y", "main"],
        "cwd": "plain-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["https://x/y", "main"],
        },
    },
    {
        "argv": ["git", "push", "--repo", "url", "main"],
        "cwd": "plain-repo",
        "call": {
            "subcommand": "push",
            "options": ["--repo"],
            "option_values": [["--repo", "url"]],
            "operands": ["main"],
        },
    },
    {
        "argv": ["git", "push"],
        "cwd": "plain-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": [],
        },
    },
    {
        "argv": ["git", "fetch", "origin"],
        "cwd": "plain-repo",
        "call": {
            "subcommand": "fetch",
            "options": [],
            "option_values": [],
            "operands": ["origin"],
        },
    },
    {
        "argv": ["git", "push", "origin", "main"],
        "cwd": "unreadable-config-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["origin", "main"],
        },
    },
    {
        "argv": ["git", "push", "origin", "main"],
        "cwd": "no-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["origin", "main"],
        },
    },
    {
        "argv": ["git", "push", "upstream", "main"],
        "cwd": "bare-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["upstream", "main"],
        },
    },
    {
        "argv": ["git", "push", "origin", "main"],
        "cwd": "bare-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["origin", "main"],
        },
    },
    {
        "argv": ["git", "push", "origin", "main"],
        "cwd": "garbage-gitfile-repo",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["origin", "main"],
        },
    },
    {
        "argv": ["git", "push", "origin", "main"],
        "cwd": "linked-worktree",
        "call": {
            "subcommand": "push",
            "options": [],
            "option_values": [],
            "operands": ["origin", "main"],
        },
    },
]

# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convert a recorded value into a deterministic JSON-compatible structure."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (set, frozenset)):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def call_json(call: Callable[[], Any]) -> dict[str, Any]:
    """Record one call as ``{"value": ...}`` or ``{"exception": ...}``."""
    try:
        return {"value": to_jsonable(call())}
    except Exception as error:
        return {"exception": {"type": type(error).__name__, "message": str(error)}}


def rel_path(path: Path, root: Path) -> str:
    """Render one fixture path relative to the fixture root."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path).replace(str(root), FIXTURES_TOKEN)


def rel_paths(paths: Any, root: Path) -> list[str]:
    """Render a path collection sorted by its relative spelling."""
    return sorted(rel_path(path, root) for path in paths)


def scope_json(scope: Any, root: Path) -> dict[str, Any] | None:
    """Render one RepositoryScope with fixture-relative paths."""
    if scope is None:
        return None
    return {
        "git_dir": rel_path(scope.git_dir, root),
        "work_tree": (
            None if scope.work_tree is None else rel_path(scope.work_tree, root)
        ),
    }


def build_relocation(spec: dict[str, Any] | None) -> GitRepositoryRelocation:
    """Build the relocation one corpus entry resolves its scope with."""
    if not spec:
        return GitRepositoryRelocation()
    return GitRepositoryRelocation(
        cwd_paths=tuple(spec.get("cwd_paths", ())),
        git_dir=spec.get("git_dir"),
        work_tree=spec.get("work_tree"),
    )


def materialize_fixtures(source: Path, dest: Path) -> Path:
    """Copy the committed fixture tree and turn ``git-dir`` into ``.git``.

    Git refuses to track paths under a ``.git`` component, so the committed
    tree stores every ``.git`` directory/file under the placeholder name
    ``git-dir``; the rename below produces the real marker layout the policy
    module inspects. Copy modes are preserved, so executable hooks stay
    executable.
    """
    root = dest / "fixtures"
    shutil.copytree(source, root)
    placeholders = sorted(
        root.rglob(PLACEHOLDER), key=lambda path: len(path.parts), reverse=True
    )
    for path in placeholders:
        path.rename(path.with_name(GIT_MARKER))
    return root.resolve()


@pytest.fixture(scope="session")
def fixtures_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The materialized fixture tree shared by all corpus tests."""
    return materialize_fixtures(FIXTURES_SOURCE, tmp_path_factory.mktemp("corpus"))


# ---------------------------------------------------------------------------
# Family observers (shared verbatim by the recorder and the replay)
# ---------------------------------------------------------------------------


def observe_policy(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Full ``ShellCommandPolicy`` asdict of ``analyze_shell_command_policy``."""
    argv = [str(token) for token in entry["argv"]]
    return {
        "policy": call_json(lambda: policy.analyze_shell_command_policy(list(argv)))
    }


def observe_paths(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """``path_candidates`` order, with and without ``inspect_executable``."""
    argv = [str(token) for token in entry["argv"]]
    return {
        "candidates": call_json(
            lambda: policy.path_candidates(list(argv), inspect_positional_paths=True)
        ),
        "candidates_executable": call_json(
            lambda: policy.path_candidates(
                list(argv), inspect_positional_paths=True, inspect_executable=True
            )
        ),
    }


def observe_git_pure(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Canonical command, routing flags, and the full classification asdict."""
    argv = [str(token) for token in entry["argv"]]
    return {
        "canonical": call_json(lambda: policy.canonical_git_command(list(argv))),
        "is_dangerous": call_json(
            lambda: policy.is_dangerous_git_subcommand(list(argv))
        ),
        "tree_writing": call_json(
            lambda: policy.git_work_tree_writing_subcommand(list(argv))
        ),
        "classification": call_json(
            lambda: policy.classify_dangerous_git_command(list(argv))
        ),
    }


def observe_executor(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """``executor_boundary`` asdict (or exception) for one admission mode."""
    argv = [str(token) for token in entry["argv"]]
    admission = bool(entry["admission"])
    return {
        "boundary": call_json(
            lambda: policy.executor_boundary(list(argv), admission=admission)
        )
    }


def observe_executor_aux(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Executor token projection helpers for one argv."""
    argv = [str(token) for token in entry["argv"]]
    return {
        "owned": call_json(lambda: policy.executor_owned_tokens(list(argv))),
        "is_package": call_json(lambda: policy.is_package_command(list(argv))),
        "package_indexes": call_json(
            lambda: sorted(policy.package_argument_indexes(list(argv)))
        ),
        "inline_switch": call_json(
            lambda: policy.inline_interpreter_switch(list(argv))
        ),
    }


def observe_argv_helpers(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """The W3a argv normalization helpers on one token/argument vector."""
    kind = entry["call"]
    if kind == "command_name":
        return {"command_name": call_json(lambda: policy._command_name(entry["token"]))}
    if kind == "options_before_double_dash":
        return {
            "options": call_json(
                lambda: policy._options_before_double_dash([
                    str(token) for token in entry["args"]
                ])
            )
        }
    if kind == "matches_command_prefix":
        return {
            "matches": call_json(
                lambda: policy.matches_command_prefix(
                    [str(token) for token in entry["tokens"]],
                    [str(token) for token in entry["pattern"]],
                )
            )
        }
    raise AssertionError(f"unknown argv helper call: {kind}")


def observe_registry(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """The full EXECUTOR_REGISTRY: sorted families plus every grammar asdict."""
    return {
        "families": sorted(policy.EXECUTOR_REGISTRY),
        "grammars": {
            name: to_jsonable(policy.EXECUTOR_REGISTRY[name])
            for name in sorted(policy.EXECUTOR_REGISTRY)
        },
    }


def observe_git_inspection(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Repository-inspection outputs for one argv against the fixture tree."""
    argv = [str(token) for token in entry["argv"]]
    cwd = root / str(entry["cwd"])
    relocation = build_relocation(entry.get("relocation"))

    def resolve_scope() -> Any:
        return policy.resolve_repository_scope(cwd, relocation)

    def remotes() -> Any:
        scope = resolve_scope()
        if scope is None:
            return None
        names = policy.git_configured_remote_names(scope)
        return None if names is None else sorted(names)

    return {
        "scope": call_json(lambda: scope_json(resolve_scope(), root)),
        "metadata_paths": call_json(
            lambda: (
                None
                if (paths := policy.git_metadata_paths(argv, cwd=cwd)) is None
                else rel_paths(paths, root)
            )
        ),
        "work_tree": call_json(
            lambda: (
                None
                if (path := policy.resolve_git_work_tree(argv, cwd)) is None
                else rel_path(path, root)
            )
        ),
        "cwd_relocations": call_json(
            lambda: (
                None
                if (paths := policy.git_cwd_relocations(argv, {cwd})) is None
                else rel_paths(paths, root)
            )
        ),
        "config_risk": call_json(
            lambda: policy.git_repository_config_risk(argv, cwd=cwd)
        ),
        "remote_names": call_json(remotes),
    }


def observe_push_risk(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """``git_push_destination_risk`` for one literal modeled push call."""
    cwd = root / str(entry["cwd"])
    relocation = build_relocation(entry.get("relocation"))
    spec = entry["call"]
    call = ModeledGitCall(
        subcommand=str(spec["subcommand"]),
        options=tuple(str(option) for option in spec["options"]),
        option_values=tuple(
            (str(name), str(value)) for name, value in spec["option_values"]
        ),
        operands=tuple(str(operand) for operand in spec["operands"]),
    )

    def run() -> Any:
        scope = policy.resolve_repository_scope(cwd, relocation)
        if scope is None:
            return None
        risk = policy.git_push_destination_risk(call, scope)
        if risk is None:
            return None
        return risk.replace(str(root), FIXTURES_TOKEN)

    return {"push_risk": call_json(run)}


def observe_divergence(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """The pinned deliberate divergences, observed end to end in one entry."""
    argv = [str(token) for token in entry["argv"]]
    return {
        "policy": call_json(lambda: policy.analyze_shell_command_policy(list(argv))),
        "candidates": call_json(
            lambda: policy.path_candidates(list(argv), inspect_positional_paths=True)
        ),
        "canonical": call_json(lambda: policy.canonical_git_command(list(argv))),
        "is_dangerous": call_json(
            lambda: policy.is_dangerous_git_subcommand(list(argv))
        ),
        "tree_writing": call_json(
            lambda: policy.git_work_tree_writing_subcommand(list(argv))
        ),
        "classification": call_json(
            lambda: policy.classify_dangerous_git_command(list(argv))
        ),
    }


OBSERVERS: dict[str, Callable[[dict[str, Any], Path], dict[str, Any]]] = {
    "policy": observe_policy,
    "paths": observe_paths,
    "git_pure": observe_git_pure,
    "executor": observe_executor,
    "executor_aux": observe_executor_aux,
    "argv_helpers": observe_argv_helpers,
    "registry": observe_registry,
    "git_inspection": observe_git_inspection,
    "push_risk": observe_push_risk,
    "divergence": observe_divergence,
}

FAMILY_ORDER = tuple(OBSERVERS)


def observe_entry(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run the recipe one corpus entry was recorded with."""
    observer = OBSERVERS.get(str(entry["family"]))
    assert observer is not None, f"unknown corpus family: {entry['family']}"
    return observer(entry, root)


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def build_family_entries(family: str, root: Path) -> list[dict[str, Any]]:
    """Build every corpus entry of one family, in recorded order."""
    entries: list[dict[str, Any]] = []

    def add(inputs: dict[str, Any], expect: dict[str, Any]) -> None:
        entries.append({
            "id": f"{family}:{len(entries) + 1:04d}",
            "family": family,
            **inputs,
            "expect": expect,
        })

    if family == "policy":
        for argv in ALL_POLICY_ARGV:
            add({"argv": argv}, observe_policy({"argv": argv}, root))
    elif family == "paths":
        for argv in ALL_POLICY_ARGV:
            add({"argv": argv}, observe_paths({"argv": argv}, root))
    elif family == "git_pure":
        for argv in GIT_PURE_ARGV:
            add({"argv": argv}, observe_git_pure({"argv": argv}, root))
    elif family == "executor":
        for argv in EXECUTOR_ARGV:
            for admission in (False, True):
                entry = {"argv": argv, "admission": admission}
                add(entry, observe_executor(entry, root))
    elif family == "executor_aux":
        for argv in EXECUTOR_ARGV:
            add({"argv": argv}, observe_executor_aux({"argv": argv}, root))
    elif family == "argv_helpers":
        for case in ARGV_HELPER_CASES:
            add(case, observe_argv_helpers(case, root))
    elif family == "registry":
        add({}, observe_registry({}, root))
    elif family == "git_inspection":
        for case in GIT_INSPECTION_CASES:
            add(case, observe_git_inspection(case, root))
    elif family == "push_risk":
        for case in PUSH_RISK_CASES:
            add(case, observe_push_risk(case, root))
    elif family == "divergence":
        for argv in DIVERGENCE_ARGV:
            add({"argv": argv}, observe_divergence({"argv": argv}, root))
    else:
        raise AssertionError(f"unknown corpus family: {family}")
    return entries


def build_entries(root: Path) -> list[dict[str, Any]]:
    """Build the whole corpus in its recorded order."""
    return [
        entry for family in FAMILY_ORDER for entry in build_family_entries(family, root)
    ]


def serialize_entries(entries: list[dict[str, Any]]) -> str:
    """Serialize entries as the canonical JSONL text of the corpus."""
    return "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)


def load_corpus() -> list[dict[str, Any]]:
    """Load the committed corpus entries (empty when not yet recorded)."""
    if not CORPUS_PATH.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _difference(
    entry_id: str, observed: Any, expected: Any, path: str = "expect"
) -> str:
    """Locate the first difference between an observation and its record."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed), key=str):
            if key not in observed:
                return f"{entry_id}: missing key {path}.{key}"
            if key not in expected:
                return f"{entry_id}: unexpected key {path}.{key}"
            difference = _difference(
                entry_id, observed[key], expected[key], f"{path}.{key}"
            )
            if difference:
                return difference
        return ""
    if observed != expected:
        return f"{entry_id}: {path}: observed {observed!r} != recorded {expected!r}"
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_corpus_entry_schema() -> None:
    """The committed corpus is well-formed and covers every family."""
    entries = load_corpus()
    assert entries, (
        "corpus.jsonl is missing or empty; regenerate with: "
        "uv run python tests/tools/test_shell_policy_corpus.py --record"
    )
    ids = [str(entry["id"]) for entry in entries]
    assert len(ids) == len(set(ids)), "duplicate corpus entry ids"
    families: set[str] = set()
    for entry in entries:
        family = str(entry["family"])
        families.add(family)
        assert family in OBSERVERS, f"unknown family in {entry['id']}"
        assert entry["id"].startswith(f"{family}:"), entry["id"]
        assert isinstance(entry["expect"], dict) and entry["expect"], entry["id"]
        if family in {"policy", "paths", "git_pure", "executor_aux", "divergence"}:
            assert isinstance(entry["argv"], list), entry["id"]
        if family == "executor":
            assert isinstance(entry["argv"], list), entry["id"]
            assert isinstance(entry["admission"], bool), entry["id"]
    assert families == set(FAMILY_ORDER), (
        f"corpus families {sorted(families)} != expected {sorted(FAMILY_ORDER)}"
    )


@pytest.mark.parametrize("entry", load_corpus(), ids=lambda entry: entry["id"])
def test_corpus_replay(entry: dict[str, Any], fixtures_root: Path) -> None:
    """Replay one corpus entry: the module must reproduce the record exactly."""
    observed = observe_entry(entry, fixtures_root)
    difference = _difference(str(entry["id"]), observed, entry["expect"])
    assert not difference, difference


@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_corpus_recorder_round_trip(family: str, fixtures_root: Path) -> None:
    """The recorder reproduces the committed corpus byte-identically."""
    regenerated = serialize_entries(build_family_entries(family, fixtures_root))
    committed = "".join(
        line + "\n"
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line)["family"] == family
    )
    assert regenerated == committed, (
        f"corpus.jsonl is stale for family {family!r}; regenerate with: "
        "uv run python tests/tools/test_shell_policy_corpus.py --record"
    )


def test_corpus_fixture_tree_materializes(fixtures_root: Path) -> None:
    """The materialized fixture tree has the layout the corpus assumes."""
    assert (fixtures_root / "plain-repo" / ".git" / "config").is_file()
    assert (fixtures_root / "plain-repo" / "nested" / "inner.txt").is_file()
    assert (fixtures_root / "gitfile-repo" / ".git").is_file()
    assert (fixtures_root / "linked-worktree" / ".git").is_file()
    assert (fixtures_root / "bare-repo" / "HEAD").is_file()
    assert (fixtures_root / "bare-repo" / "objects").is_dir()
    assert (fixtures_root / "unreadable-config-repo" / ".git" / "config").is_dir()
    hook = fixtures_root / "hook-repo" / ".git" / "hooks" / "post-index-change"
    assert hook.stat().st_mode & 0o111, "executable hook lost its mode"
    bare_hook = fixtures_root / "bare-repo" / "hooks" / "post-index-change"
    assert bare_hook.stat().st_mode & 0o111, "executable bare hook lost its mode"
    plain_hook = fixtures_root / "plain-repo" / ".git" / "hooks" / "pre-push"
    assert not plain_hook.stat().st_mode & 0o111, "non-executable hook changed"


def test_facade_reexports_preserve_moved_identity() -> None:
    """The 3.2/3.3a/3.3b, 4.3, 4.4, and 4.5 splits: every facade name is the moved object itself."""
    moved = (
        "_COMBINED_SHORT_OPTION_MIN_LENGTH",
        "_OptionTable",
        "_ParsedOptions",
        "_SHORT_OPTION_LENGTH",
        "_argument_matches",
        "_command_name",
        "_has_long",
        "_has_short",
        "_has_short_in_cluster",
        "_is_combined_short_options",
        "_match_long",
        "_options_before_double_dash",
        "_scan_options",
        "matches_command_prefix",
    )
    for name in moved:
        assert getattr(policy, name) is getattr(_shell_argv, name), name
    grammar_moved = (
        "_INLINE_SWITCHES",
        "_MAX_EXECUTOR_DEPTH",
        "_PACKAGE_PATH_COMMANDS",
        "_PACKAGE_PATH_OPTIONS",
        "EXECUTOR_REGISTRY",
        "_executor_option",
        "_executor_package_policy",
        "_module_executor_command",
        "_package_local_path",
        "_package_module_tokens",
        "_package_requirement_path",
        "_short_interpreter_option",
        "ArgumentRange",
        "BoundaryKind",
        "executor_boundary",
        "executor_owned_tokens",
        "ExecutorBoundary",
        "ExecutorGrammar",
        "inline_interpreter_switch",
        "is_package_command",
        "package_argument_indexes",
    )
    for name in grammar_moved:
        assert getattr(policy, name) is getattr(_shell_executor_grammar, name), name

    # 4.3a-i: the classifier's tables, scanners, and per-subcommand verdicts.
    classifier_moved = (
        "_GIT_PUSH_TABLE",
        "_GIT_PUSH_BOOLEAN_SHORTS",
        "_GIT_PUSH_FORCE_OPTIONS",
        "_GIT_PUSH_HARD_GUARD_LONGS",
        "_PUSH_URL_SCHEME",
        "_push_destination_is_explicit",
        "_push_destination_fail_closed_reason",
        "_GIT_SWITCH_TABLE",
        "_GIT_SWITCH_BOOLEAN_SHORTS",
        "_GIT_SWITCH_DISCARD_OPTIONS",
        "_GIT_CHECKOUT_TABLE",
        "_GIT_RESTORE_TABLE",
        "_GIT_WORKTREE_TABLES",
        "_GIT_STASH_TABLE",
        "_GIT_REFLOG_TABLE",
        "_STASH_REFLOG_OPERATION_IDS",
        "_GIT_RESET_TABLE",
        "_GIT_RESET_BOOLEAN_SHORTS",
        "_GIT_CLEAN_TABLE",
        "_GIT_CLEAN_BOOLEAN_SHORTS",
        "_GIT_REMOTE_TABLE",
        "_GIT_REMOTE_BOOLEAN_SHORTS",
        "_GIT_REMOTE_MUTATION_SUBCOMMANDS",
        "_ROUTINE",
        "_hard_denied",
        "_grantable_classification",
        "_fail_closed_on_end_of_options",
        "_refspec_fail_closed_reason",
        "_push_operation_ids",
        "_classify_git_push",
        "_classify_git_worktree_subcommand",
        "_classify_git_switch",
        "_classify_git_stash_or_reflog",
        "_classify_git_hard_guard_subcommand",
        "_classify_git_remaining",
        # 4.3a-ii: the git config reason family and _classify_git_config.
        "_GIT_CONFIG_REMOTE_PUSH_EFFECTING_KEYS",
        "_GIT_CONFIG_BRANCH_PUSH_EFFECTING_KEYS",
        "_GIT_CONFIG_URL_PUSH_EFFECTING_KEYS",
        "_GIT_CONFIG_INCLUDE_SECTIONS",
        "_GIT_CONFIG_CORE_PROGRAM_KEYS",
        "_GIT_CONFIG_CORE_PROGRAM_SELECTION_KEYS",
        "_GIT_CONFIG_SEQUENCE_PROGRAM_SELECTION_KEYS",
        "_GIT_CONFIG_MAN_PROGRAM_SELECTION_KEYS",
        "_GIT_CONFIG_INTERACTIVE_PROGRAM_SELECTION_KEYS",
        "_GIT_CONFIG_FILTER_PROGRAM_SELECTION_KEYS",
        "_GIT_CONFIG_MERGE_PROGRAM_SELECTION_KEYS",
        "_GIT_CONFIG_DIFF_PROGRAM_SELECTION_KEYS",
        "_git_config_key_is_push_effecting",
        "_git_config_section_is_push_effecting",
        "_git_config_key_is_program_selection",
        "_git_config_section_is_program_selection",
        "_git_config_push_effecting_reason",
        "_git_config_program_selection_reason",
        "_git_config_program_section_reason",
        "_git_config_rename_guard_reason",
        "_git_config_section_operation_guard_reason",
        "_git_config_write_guard_reason",
        "_classify_git_config",
        # 4.3b: dispatch and dangerous-subcommand classification.
        "_classify_git_remote",
        "_dispatch_git_subcommand",
        "classify_dangerous_git_command",
        "_DANGEROUS_GIT_SUBCOMMANDS",
        "is_dangerous_git_subcommand",
    )
    for name in classifier_moved:
        assert getattr(policy, name) is getattr(_shell_git_classifier, name), name

    # 4.4: the inspection half (the family's only I/O module).
    inspection_moved = (
        "_FALSE_GIT_CONFIG_VALUES",
        "_GIT_CONFIG_ENTRY",
        "_GIT_CONFIG_SECTION_HEADER",
        "_GIT_TREE_WRITING_SUBCOMMANDS",
        "_INCLUDE_DIRECTIVE_NAMES",
        "_git_config_entries",
        "_git_config_paths",
        "_git_config_remote_url_names",
        "_git_dir_config_paths",
        "_git_index_hook_risk",
        "_git_value_is_active",
        "_parse_git_config_entries",
        "git_configured_remote_names",
        "git_cwd_relocations",
        "git_metadata_paths",
        "git_push_destination_risk",
        "git_work_tree_writing_subcommand",
        "resolve_git_work_tree",
        "resolve_repository_scope",
    )
    for name in inspection_moved:
        assert getattr(policy, name) is getattr(_shell_git_inspection, name), name

    # 4.5: the pure git-policy half (subcommand tables + _git_policy).
    git_policy_moved = (
        "_GIT_APPLY_TABLE",
        "_GIT_APPLY_UNSAFE_PATHS_REASON",
        "_GIT_ARCHIVE_OUTPUT_TABLE",
        "_GIT_INIT_TABLE",
        "_GIT_SUBCOMMAND_TABLE",
        "_GIT_SUBMODULE_ADD_TABLE",
        "_GIT_WORKTREE_ADD_TABLE",
        "_GIT_WORKTREE_MOVE_TABLE",
        "_git_archive_output_values",
        "_git_destination_operands",
        "_git_destructive_policy",
        "_git_policy",
        "_legacy_clean_flags",
    )
    for name in git_policy_moved:
        assert getattr(policy, name) is getattr(_shell_git_policy, name), name
    assert policy._COMMAND_POLICIES["git"] is _shell_git_policy._git_policy

    # The registry is one dict: the facade re-export keeps its identity (a
    # mutation seam — test_shell_command_policy.py imports it from the facade).
    assert policy.EXECUTOR_REGISTRY is _shell_executor_grammar.EXECUTOR_REGISTRY
    assert len(policy.EXECUTOR_REGISTRY) == 36
    assert all(
        isinstance(grammar.options, _shell_argv._OptionTable)
        for grammar in policy.EXECUTOR_REGISTRY.values()
    )


# ---------------------------------------------------------------------------
# Recorder CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Record (--record) or check the corpus against the current module."""
    parser = argparse.ArgumentParser(
        description="Golden shell-policy corpus recorder/replayer."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="regenerate corpus.jsonl against the current module",
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="shell-policy-corpus-") as tmp:
        root = materialize_fixtures(FIXTURES_SOURCE, Path(tmp))
        entries = build_entries(root)
    text = serialize_entries(entries)
    if args.record:
        CORPUS_PATH.write_text(text, encoding="utf-8")
        print(f"recorded {len(entries)} entries to {CORPUS_PATH}")
        return 0
    committed = CORPUS_PATH.read_text(encoding="utf-8") if CORPUS_PATH.is_file() else ""
    if committed == text:
        print(f"corpus round-trips: {len(entries)} entries byte-identical")
        return 0
    print(
        f"corpus is stale: {len(entries)} entries regenerate differently; "
        "run with --record"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
