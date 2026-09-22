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

## Compaction

`/compact` summarizes older history to reduce the context sent in later model
requests. You may provide extra instructions after the command to guide the
summary. Compaction preserves the session and visible conversation; subsequent
requests use the latest compacted context followed by newer messages.

Automatic compaction follows the selected model's threshold, falling back to
`auto_compact_threshold`. Set `compaction_model` to use another compatible
catalog model, or `compaction_prompt_id` to select a custom compaction prompt.
An empty `compaction_model` uses the active model. The [configuration
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
