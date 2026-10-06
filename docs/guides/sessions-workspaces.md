# Sessions and workspaces

A session holds the conversation, its selected model identity, and its working
context. With session logging enabled (the default), sessions are saved locally
and can be continued or resumed.

## Continue and resume

Use `-c` or `--continue` to reopen the most recent saved session in the
current directory. `--resume` opens the session picker, and `--resume ID`
opens a specific session; unique partial IDs are accepted. Inside the TUI,
`/resume` and `/continue` open the same picker. Deleting a saved session in
the picker requires pressing `d` twice, and the active session cannot be
deleted there.

```bash
chartreux --continue
chartreux --resume
chartreux --resume abc123
```

Sessions are scoped to their working directory. The continue option and picker
therefore show sessions for the current directory or worktree. To move a
conversation to another worktree, resume it explicitly by ID.

A session commits its base model and provider deployment when assigned. Resume
validates and restores that identity rather than resolving a role again.
`/clear` begins a new conversation using current configuration; `/branch`
creates a separate resumable copy. See the [command reference](../reference/commands.md)
for the complete command surface.

## Managed-job lifetime

Managed shell jobs belong to one live root runtime, including jobs launched by
its children. Normal turns, interruption after launch commit, and compaction
preserve them. `/clear`, a new or replacement root, and resume/continue handoff
stop the departing root's jobs before changing session identity. Ending the root
session or shutting down the server also performs owned-job cleanup.

A fork or clone has a fresh, empty job registry; it does not share or adopt the
source root's jobs. If a session-changing rewind retires the old root, its jobs
are stopped; an in-place transcript rewind does not itself stop them. Saved
transcripts can contain historical job IDs and read results, but those handles
do not authorize access in a resumed or forked runtime. No processes or live
output buffers are adopted across a Chartreux restart.

Actual worktree relocation is rejected while jobs or pending launches exist;
a same-directory no-op is allowed. Stop jobs before relocating or reducing shell,
workspace, or credential authority. Cleanup is bounded for owned process groups,
not a guarantee against escaped descendants or forced-process-exit leakage; see
[Tools and safety](tools-safety.md#managed-shell-jobs).

## Usage storage

Recorded usage is stored independently of transcripts at
`$CHARTREUX_HOME/usage/<root-session-id>/usage.jsonl`. A root session owns its
subagent calls. The ledger also records compaction, automatic title generation,
and worktree-naming calls, including naming calls made before a session starts.
Those records remain even if startup fails.

Disabling session logging or changing its save directory does not disable or
relocate the ledger. Deleting a saved session does not delete its usage records;
rewinding a conversation does not undo recorded spend. Sessions from before the
ledger are not backfilled from transcripts: `/usage` shows **Recorded usage only**.
There is currently no automatic ledger retention or compaction.

Usage project grouping differs from the directory-scoped session picker. Linked
Git worktrees share one project identity through their common Git directory;
separate clones are separate projects even if their remote URL matches. Outside
Git, grouping uses the canonical root workspace path. Subagents inherit the root's
project identity. `/usage` defaults to All projects and can filter to Current
project; status-line spend always covers all projects in the current Chartreux home.

## Checkpoints and rewind

Use `/rewind`, or press Escape twice with an empty input, to return to an earlier
user message. Choose whether to restore files, then whether to keep the rewind
in the current session or fork to a new one. The selected message and all later
messages are removed from the active conversation; the selected prompt returns
to the input for editing. A fork preserves the original conversation as its
parent. An in-place rewind saves the shortened conversation under the same
session ID, without creating a parent copy.

Conversation and file restoration are separate choices. Editing without
restoring files leaves disk contents unchanged. Restoring files applies the
checkpoint states for the selected turn, including removing a captured file
that did not exist at that point. A preserved parent conversation is not a
separate copy of the workspace: both sessions still refer to the same files.

### What is captured

Checkpoints record file bytes or absence for paths supplied by snapshot-aware
file tools. Known paths are re-read at turn boundaries, so later changes to
those paths can be captured even when the tool making them supplies no snapshot.
This is not a scan or backup of the whole workspace. A shell command, hook,
external editor, or child agent can change an untracked path without giving
rewind a file state to restore. File snapshots are memory-only and are cleared
when switching or clearing sessions. After resume, new rewinds can restore only
edits checkpointed in the current process, not files from the saved transcript.

### File restoration and limits

Rewind saves or forks the conversation first, then restores checkpointed files
individually. Transcript generation fencing prevents a stale queued save from
overwriting the retained conversation after rewind, reset, or session rebind.
It does not make file restoration and transcript saving one atomic operation.

Restoration writes are staged beside each target before single-file replacement,
so a failed staging write does not truncate the target. Existing file permissions
are preserved. Read, write, and deletion failures are contained per path and
reported; other paths can still be restored. A failure can therefore leave a
partially restored workspace while the conversation has already been shortened.
There is no transaction-wide byte limit, compensation, or interrupted-transaction
recovery. Stop other writers before restoring files; rewind is not a workspace
backup and does not coordinate with external editors or processes.

## Compaction

`/compact` summarizes older history to reduce the context sent in later model
requests. You may provide extra instructions after the command to guide the
summary. Compaction preserves the session and visible conversation; subsequent
requests use the latest compacted context followed by newer messages.

Automatic compaction follows the selected model's threshold, falling back to
`auto_compact_threshold`. Set `compaction_model` to use another compatible
catalog model, or `compaction_prompt_id` to select a custom compaction prompt.
An empty `compaction_model` uses the current main model, including any
session override. The [configuration
reference](../reference/configuration.md) lists the related settings.

## Session roots and working directories

The working directory is the session's primary root. Start in a different
location with `--workdir`:

```bash
chartreux --workdir /path/to/project
```

Use repeatable `--add-dir` for additional workspace roots:

```bash
chartreux --add-dir /path/to/library --add-dir /path/to/other-project
```

Each additional directory is available for the session, is implicitly trusted,
and receives the same in-root file-tool treatment as the primary working
directory. It contributes local instructions, extension directories for skills,
prompts, tools, and agents, and hooks. Its `config.toml` is not merged: the
project TOML layer is rooted only at the primary working directory. Nested roots
remain distinct, so both a repository and one of its subdirectories may
contribute root-local extensions and instructions. See [instructions and skills](instructions-skills.md)
and [tools and safety](tools-safety.md) for the trust and authority model.

## Worktrees

`--worktree NAME` creates or reuses a Git worktree and starts the session in
that checkout:

```bash
chartreux --worktree my-feature
```

Chartreux stores managed worktrees below `$CHARTREUX_HOME/worktrees/` and
checks out the requested branch there. A named existing worktree is reused only
when it belongs to the same repository and is on the requested branch. If
started from a repository subdirectory, the session enters the corresponding
subdirectory of the worktree.

With no name, `--worktree` chooses one from the prompt or a random slug. Put a
positional prompt before this optional argument, or separate it with `--`:

```bash
chartreux "Fix the login bug" --worktree
chartreux --worktree -- "Fix the login bug"
```

## Ownership and cleanup

Chartreux records ownership only for worktrees it creates, and records a hold
for each session using one. It never removes a worktree without a valid
ownership record. Holds and resumable-session locations prevent cleanup of a
checkout that may still be in use; if the session list cannot be read,
Chartreux conservatively keeps all managed worktrees.

For an interactive session, automatic cleanup considers only a worktree created
for that run after the session has started. A clean, unchanged worktree may be
removed on exit. Changes, untracked files, or commits made since session start
require a choice to keep or remove it. Reused worktrees are left in place, and
programmatic worktree runs are not automatically cleaned up because there is
no exit prompt.

An app-server session closing does not by itself authorize removal. Deleting a
session can reclaim a Chartreux-created worktree only when no other session
holds it and it has no protected work. Worktrees with missing or unreadable
ownership data, or markers left after a terminated process, are retained for
manual review.
