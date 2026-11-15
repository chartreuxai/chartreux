Chartreux is an independent fork of Mistral Vibe.

## Development cycle

- `main` is the release trunk: one commit per release, tagged `vX.Y.Z`. No direct commits to `main` (the v0.2.0 release commit is the one manual exception).
- `development` is the integration branch: all work happens on feature branches cut from `development` and is squash-merged into `development`.
- Release procedure: on `development`, bump the version in `pyproject.toml` and `chartreux/__init__.py` and finalize the CHANGELOG entry; squash-merge `development` into `main`; tag the release commit on `main`; push `main` and the tag; continue `development` from the merged state. The strict docs build `uv run --with 'mkdocs>=1.6.1' mkdocs build --strict` (timeout=300) is a release step, run and reviewed before the squash-merge.
- The strict docs build is release procedure, not per-change verification: it writes `site/` into the repository and can regenerate tracked documentation pages. It runs only under release-owner authorization, which covers its `site/` and tracked-documentation side effects, and the release owner decides which generated paths are kept, committed, or discarded.
- The project is pre-release alpha: no stability guarantees, breaking changes allowed without migration paths.

## Verification

- Full test suite (parallel) (timeout=900): `uv run pytest`
- Full suite from a git worktree (parallel) (timeout=900): `cd <worktree> && PYTHONPATH=<worktree> uv run --project /home/pav/code/chartreux --no-sync pytest --ignore=tests/cli/test_installed_contracts.py`
- Typecheck (timeout=300): `uv run pyright`
- Lint and format (timeout=180): `uv run ruff check . && uv run ruff format --check .`

Notes:
- Do NOT pass `-o addopts=''`: addopts carries xdist parallelism (`-n auto`); stripping it turns a ~2-minute parallel suite into a ~20-minute serial run.
- For long suites, redirect output to a temp file and read the tail — never rerun the suite to recover truncated output. Use `--lf` or targeted node IDs for failure follow-ups.
- Installed-contract tests (tests/cli/test_installed_contracts.py) require `UV_FIND_LINKS` pointing at a pre-provisioned wheelhouse; skip them locally without it (CI covers them).
- The strict docs build is not a per-change check: it belongs to the release procedure (see Development cycle) and runs under release-owner authorization.
- Prose inspection is not behavioral proof: reading the file, the diff, or a documented command never substitutes for running the check and reading its exit status.

## Worker output boundary

- For runs whose brief forbids repository output, that boundary covers every write inside the repository, including ignored and untracked incidental writes: pytest cache and `--basetemp`, ruff cache, bytecode, uv cache, coverage files, and build outputs such as `site/`. Resolve every destination before running a command.
- Keep those writes outside the tree: `PYTHONDONTWRITEBYTECODE=1`, an external pytest `cache_dir` (or `-p no:cacheprovider`) with an external `--basetemp` while retaining the project addopts, an external `RUFF_CACHE_DIR`, and an external `UV_CACHE_DIR`.
- A project environment must already exist and be used read-only: point at it with `UV_PROJECT_ENVIRONMENT=<pre-provisioned external environment>` or run `uv run --project /home/pav/code/chartreux --no-sync`. Never let uv sync or create an environment implicitly inside the worktree.
- Inspect fixtures and plugins for their destinations before a suite run. If a declared command, test, or plugin writes elsewhere inside the repository, or if safe isolation would change the meaning of a mandated command, stop and report instead of silently rewriting the command or claiming a pass.
- Never delete pre-existing outputs, and capture long-suite output outside the tree.

## Temporary-file placement

- Every verification invocation gets its own fresh, unique invocation directory under the session scratchpad, and points its basetemp, pytest cache dir, uv cache, and ruff cache at it. An explicit destination under the scratchpad takes precedence over the managed run root and is never removed automatically; a worker choosing an arbitrary explicit destination outside the scratchpad is user-owned and stays outside automatic reaping (the backstop covers omission only: no explicit destination means the plugin's managed root).
- Repo-default pytest runs land under the managed pytest run root (`<temproot>/chartreux-managed-<uid>/pytest-run-*` via `pytest_chartreux`), never under the shared `/tmp/pytest-of-<user>` tree; there is no fixed `--basetemp` in addopts. The legacy `/tmp/pytest-of-<user>` tree is never swept automatically.
- The repo default is `tmp_path_retention_policy = "failed"`: only failed tests' call-phase tmp dirs are retained, passed tests' dirs are removed per test. Setup/teardown-phase failure artifacts may not be retained — copy-before-teardown, or rerun with a temporary `-o tmp_path_retention_policy=all` override, for deeper debugging.
- Clean up only your own completed invocation's outputs, after preserving any requested diagnostics; no wildcard cleanup between packages, and never a directory a still-running job is using.

## Secret scan

- Verification artifacts — basetemps, tool caches, and logs — belong under the injected scratchpad, in a fresh, unique directory per verification invocation, for the organization and attribution of the worker's own artifacts. That placement does not extend the submission scan: paths under SHARED scratchpad roots are partitioned out as transient content — one bounded `INFO` log line, never a finding, never a block — so anything that must block submission must live outside shared scratchpad roots. Paths outside a scratchpad stay in the scan: a cache under `~/.cache` is scanned and still blocks, and a worker with no scratchpad stops the affected verification instead of choosing an uninspected location.
- The approved-match manifest `.chartreux-secret-scan.json` is the only suppression surface, and it is policy, not a sandbox. Both gates apply it from the COMMITTED HEAD blob — the submission gate freezes it at admission and fails closed on a mid-run change, and the pre-commit hook reads HEAD, never the index or the working tree — so neither a working-tree edit nor a single commit staging a secret with its approval self-authorizes. An entry is canonical path + rule ID + digest of the rule's value-span bytes; only a rule whose match span covers the complete sensitive material is approvable, and `residual_exclusions` (path + rule, no digest) is reserved for partial-coverage rules and is EMPTY today.
- A new synthetic fixture value follows the approval-first two-commit sequence inside one reviewed change: commit the manifest entry, then the fixture. Review bars entries for values the repository never publishes, and a worker may never approve its own submission.
- This is instruction, not automatic enforcement: a session that breaks these rules still blocks at the fail-closed scan, and a `policy_mismatch` block is parent-retryable only — never auto-retried.

## Git workflow policy

- Unpublished history: rebase, `commit --amend`, and `reset --soft`/`--mixed` on unpublished history are agent-executable. A branch is UNPUBLISHED only when HEAD is not contained in any remote-tracking ref (`git rev-list --count HEAD --not --remotes > 0` and no remote-tracking branch contains HEAD).
- Published history: rewrites of commits contained in remote-tracking refs remain handoff-only and approval-gated. The publication gate checks the operation's ENTIRE affected commit range (every commit the operation could rewrite, e.g. `git rev-list` of the range being rebased) against remote-reachable history; remote-tracking refs count as current only if refreshed in this session (e.g. via `git fetch`) or the affected range is provably local-only.
- Backstop rule: when a branch's publication status is UNCERTAIN, the agent stops and hands the operation to the user — never rewrites on uncertainty.
- Destructive checkout: "destructive checkout" means a checkout that discards uncommitted changes; a clean-tree checkout of an existing local branch is not destructive.
- Hard guards preserved: update-ref, forced and deleting pushes, and history rewrites of published commits keep their existing gates exactly as before this amendment.
