# Command reference

## CLI

`chartreux [PROMPT]` starts the interactive client. A positional prompt,
`--prompt`, or non-empty piped input selects programmatic mode: it sends the
prompt, prints a response, and exits.

| Option | Meaning |
| --- | --- |
| `-h`, `--help` | Show help. |
| `-v`, `--version` | Show the version. |
| `-p [TEXT]`, `--prompt [TEXT]` | Programmatic prompt; tool calls are auto-approved. |
| `--max-turns N` | Maximum assistant turns in programmatic mode. |
| `--max-price DOLLARS` | Maximum session cost in programmatic mode. |
| `--max-tokens N` | Maximum prompt-plus-completion tokens in programmatic mode. |
| `--enabled-tools TOOL` | Repeatable exact, glob, or `re:` tool filter; in programmatic mode it disables other tools. |
| `--disabled-tools TOOL` | Repeatable exact, glob, or `re:` tool filter applied after `--enabled-tools`. |
| `--output {text,json,streaming}` | Programmatic output: human text (default), one JSON result, or newline-delimited JSON. |
| `--setup` | Run interactive setup—provider, credentials, and model selection—then exit. Requires an interactive terminal; otherwise, it prints actionable guidance instead of launching the TUI. |
| `--workdir DIR` | Change to this directory before launch. |
| `--worktree [NAME]` | Run in a managed Git worktree. With a name, create or reuse it; without one, create a name from the prompt or a random slug. Ignored with `--setup`. |
| `--add-dir DIR` | Repeatable additional workspace root; trusted for this session. |
| `--trust` | Trust the working directory for this invocation only. |
| `--identity ID` | Select identity for a new conversation (for example `coder` or `user:name`); frozen for that conversation. |
| `--workflow ID` | Select workflow for a new conversation (for example `minimal` or `ceremony`). |
| `--orchestration-strategy ID` | Select orchestration strategy for a new conversation; `none` disables a strategy body. |
| `-P [THEME]`, `--plan [THEME]` | Start in plan mode, optionally with a heading-only theme. Interactive only: rejected with `-p`/`--prompt`, a positional prompt, piped stdin, and `--setup`. Composes with `-c`/`--resume` as described under [Plan mode](#plan-mode). |
| `-c`, `--continue` | Continue the most recent saved session. Mutually exclusive with `--resume`. |
| `--resume [SESSION_ID]` | Resume a session; without an ID, open the picker. |

### Subcommands

| Command | Options |
| --- | --- |
| `chartreux models migrate [--preview \| --apply]` | Preview (default) or move legacy catalog tables to `models.toml`. |
| `chartreux doctor [--live] [--smoke] [--provider ID] [--model BASE] [--json]` | Read-only diagnostics; see tiers below. |
| `chartreux skills diff [NAME] [--json]` | Compare user skill copies in `~/.chartreux/skills` and `~/.agents/skills` with the shipped copy of the same name. `NAME` limits the report to one skill; an unknown name is an error. The report is read-only: it never removes or edits a skill. `--json` emits `entries` and `exit_code`. Exit codes: `0` nothing differs, `1` a loaded user copy differs, `2` operational error (including invalid UTF-8 or an unknown `NAME`). |
| `chartreux mcp remove NAME` | Remove a user-configured MCP server. |
| `chartreux mcp add NAME` | `--transport {streamable-http,stdio}` (default `streamable-http`), `--url`, `--command`, repeatable `--arg VALUE`, repeatable `--env NAME=VALUE`, repeatable `--header NAME=VALUE`, `--api-key-env VAR` (also `--bearer-token-env-var`), `--api-key-header HEADER`, `--api-key-format FORMAT`, `--no-login`, `--startup-timeout-sec SECONDS`, `--tool-timeout-sec SECONDS`. Remote servers require `--url`; stdio servers require `--command`. |
| `chartreux trust list [PATH]` | List every stored trust decision (`trusted` and `untrusted`) as recorded in `~/.chartreux/trusted_folders.toml`. With `PATH`, report the decision governing it: the closest recorded decision wins, so a child path inherits its nearest recorded ancestor. Exit codes: `0` listed, `2` the trust store cannot be read (it is malformed, invalid, or unreadable; it is never rewritten). |
| `chartreux trust remove PATH` | Remove the one stored decision recorded for `PATH`. Sibling and descendant entries, and every decision recorded for another path, are untouched. Interactive confirmation is required and defaults to no; without an interactive terminal the removal is denied, and no `--yes`/force flag exists. A running session keeps the decision it loaded, so restart Chartreux for the removal to take effect. Exit codes: `0` removed, `1` nothing removed (absent entry, declined, or non-interactive), `2` the trust store cannot be read or written. |

### Doctor

Bare `chartreux doctor` validates local trusted configuration, the catalog,
model selection, credential provenance, and MCP configuration. It makes no
network requests, launches no processes, does not read the app's `.env`, and does
not inspect keyring credentials. Untrusted project configuration is ignored and reported.
Both `--live` and `--smoke` load the app's `.env` into the process environment as
normal startup does, including normalizing its permissions to owner-only; existing
non-empty process environment values take precedence.
`--live` lists provider metadata and initializes/lists tools from configured MCP
servers (including launching stdio servers). It may read keyring credentials;
OAuth checks inspect stored fingerprint/expiry only, never refresh or log in,
and skip runtime listing. Empty MCP tool lists are healthy.
`--smoke` sends billable tool, thinking, and image capability probes to **one**
enabled deployment. Supply `--provider ID` and/or `--model BASE`; these options
require `--smoke`, and ambiguous selections are rejected with candidates.
Explicit `--provider`/`--model` selection
bypasses `config.allowed_models`: diagnostics probe the selected deployment
regardless of the agent-allowed set. `--live` and `--smoke` are independent.
No mode creates or repairs configuration, persists trust, or writes sessions.
Skipped, unsupported, and unverified results are not passes or failures.
`warn` results are advisory — doctor reports one per user skill copy that
shadows a shipped skill, naming the winning copy, and suggests removal only for
an identical copy after you review its support files. No removal is automatic,
and `warn` never changes the exit code: only `fail` does.
The route section reports the `[orchestration.routes]` table the trusted
configuration independently loads — never a running conversation's live table
(that is `/routes`). It lists each target's seat, binding, identity, tier, and
overlay, the model a marker binding resolves to today (a preview, not a launch
guarantee), and the distinct-model requirement beside the resolved count
(`UNKNOWN` when a target cannot resolve). Reload validation and launch-time
diversity are separate facts: the shipped `review.deep` default passes reload
validation (every target defers to launch) while its launch-time diversity
fails, and both are reported.
`--json` emits only JSON on stdout: a `checks` array of objects with `name`,
`status`, and `reason`, plus `exit_code`. Exit codes: `0` no failed checks,
`1` failed checks, `2` invalid invocation. Redirected stdio is supported.
For worked checks and capability verdicts, see [Troubleshooting](../guides/troubleshooting.md#run-diagnostics)
and [Smoke-testing a deployment](../guides/models.md#smoke-testing-a-deployment).

### Other entry points

`chartreux-acp` accepts no argument for stdio operation, plus `-h`/`--help`,
`-v`/`--version`, and `--setup`. `chartreux-app-server` is a stdio JSON-RPC
server and currently accepts only `-h`/`--help`.

## Built-in slash commands

These are the current terminal command registry. Custom user-invocable skills
can add commands.

| Command | Action |
| --- | --- |
| `/help` | Show command and shortcut help. |
| `/model` | Override the main model for this session; the saved `@main` preset remains unchanged. |
| `/thinking` | Override the thinking level for this session; the saved `@main` preset remains unchanged. |
| `/reload` | Refresh AGENTS.md, skills, and catalogs; frozen policy stays bound, edits reported as pending. |
| `/open-config-file` | Open the user config file in your editor; creates the file with a commented template if absent, and reloads after editing if it changed. |
| `/settings` | Browse and edit curated user settings. |
| `/clear`, `/new` | Start a new conversation; optionally provide a seed prompt. |
| `/copy` | Copy the last agent message. |
| `/paste-image` | Paste a clipboard image into the prompt (available on supported systems). |
| `/log` | Show the current interaction-log path. |
| `/log-level` | Change the session log level or persist it. |
| `/debug` | Toggle the debug console. |
| `/agents` | Toggle the docked agent browser above the input: Main plus subagents grouped by lifecycle (Active, Retained Idle, Recent) with Resident/Active/Idle/Recent/All filter tabs, name-first rows with dimmed handles, and active-only status counts. |
| `/compact [instructions]` | Summarize the conversation context. |
| `/plan [THEME]` | Enter plan mode; `/plan status` reports the active mode and plan document, `/plan accept` hands the plan to an executor, `/plan cancel` exits and restores the previous selections. Any other text is the plan's theme. |
| `/exit`, `exit`, `quit`, `:q`, `:quit` | Exit immediately while idle. During consequential active work, opens the exit confirmation dialog. |
| `/status` | Display agent statistics plus the next request's thinking estimate: **Requested**, **Semantic**, and **API effort** (the provider's wire value, or `unknown` when it cannot be resolved). |
| `/usage` | Browse recorded usage and USD cost estimates by calendar window and project; available while a turn runs. |
| `/routes` | Browse this conversation's live `[orchestration.routes]` table: each target's purpose, seat, configured binding, identity, tier, and overlay, the model a marker binding resolves to today (a preview, not a launch guarantee), and `requires_distinct_models` beside the resolved distinct-model count. Reload validation and launch-time diversity are reported separately; a reload that swaps the table updates the open browser. Read-only, available while a turn runs. |
| `/proxy-setup` | Configure proxy and certificate settings. See [networking](../integrations/networking.md). |
| `/providers` | Open Provider Settings to add or manage model providers. |
| `/resume`, `/continue` | Browse, resume, or delete saved sessions. |
| `/rename` | Rename the current session. |
| `/mcp` | Show MCP servers; supports `add`, `status`, `login`, and `logout` operations. |
| `/rewind` | Rewind to a prior message; press Escape twice as a shortcut when input is empty. |
| `/branch` | Fork the current conversation into a resumable session. |
| `/retry [instructions]` | Continue an interrupted model response. |
| `/loop <interval> <prompt>` | Schedule a recurring prompt; `/loop list` and `/loop cancel <id\|all>` manage it. |
| `/theme` | Select `auto`, `light`, or `dark`. |
| `/web-search` | Configure the web-search provider, credential, and search settings. |

## Plan mode

`/plan` starts a planning session. Entry applies the configured planner's model
and thinking from `[plan.planner]` as session overrides, reloads the runtime,
allocates the plan workspace, and posts the planner seed, which queues behind an
active turn.

The seed has the planner orient itself about the repository first — current
branch, recent releases, existing workspace tasks — present a brief orientation,
and then stand by for the plan agenda. It forbids inferring or inventing a task,
proposing agenda items, and beginning design, planning, or implementation on its
own: the user drives. The plan document is written only after its design and the
plan itself have been accepted, as Markdown in the allocated task directory with
a `## Summary` section whose text is the brief handed to the execution session.

`/plan THEME` sets the plan's theme, and any text after `/plan` that is not an
action is a theme. A theme is a heading, never an instruction or an
authorization. `accept`, `cancel`, and `status` are the exact-match actions:

| Command | Action |
| --- | --- |
| `/plan [THEME]` | Enter plan mode, with an optional theme. Entry is refused while the mode is already active. |
| `/plan status` | Report the active mode, its theme, and the plan document path, in the same wording the post-compaction notice uses. |
| `/plan accept` | Read the plan document, snapshot the configured executor selection, and show a consent screen with the plan path, summary, executor, and theme before handing off to a fresh executor session. Nothing is launched if you decline. |
| `/plan cancel` | Leave the mode and restore the pre-entry model and thinking selections. The allocated workspace is removed only when it holds nothing but chartreux's ownership marker; otherwise the plan document stays at its path. |

While the mode is active the prompt area carries an amber tint and a `plan`
border label. After any compaction — `/compact`, proactive auto-compaction, or
the reactive context-too-long path — the compaction boundary opens with a
notice naming the mode and the plan document path, with the theme only when the
mode has one, so the model cannot lose the mode mid-session. `/clear` ends the
mode and keeps every file.

Plan mode is persisted with the session and restored on resume; see
[Plan mode across resume and branch](../guides/sessions-workspaces.md#plan-mode-across-resume-and-branch).

### Starting in plan mode

`-P [THEME]` / `--plan [THEME]` enters plan mode when the interactive interface
opens, with the same heading-only theme semantics as `/plan THEME`. With
`-c`/`--resume` the mode is ensured after the session lands: a restored
plan-mode record keeps its own theme and the flag's theme is ignored, while a
session that was not planning enters plan mode on the resumed conversation —
use a bare `chartreux -P` when you want a clean start.

The flag is rejected for every launch that has no interactive session to land
in: `-p`/`--prompt`, a positional prompt, input piped on stdin, and `--setup`
each exit with an error instead of being silently ignored. A theme beginning
with `-` is read as an unknown option, so keep themes plain text.

Planner model and thinking come from `[plan.planner]`. There is no
`--plan-model` flag; the environment bridge is the override:

```bash
CHARTREUX_PLAN__PLANNER__MODEL=mistral-large-4 chartreux --plan
```

`CHARTREUX_PLAN__PLANNER__THINKING` overrides the level the same way. See
[environment variables](../reference/configuration.md#environment-variables-and-env)
for the naming convention.

## Key shortcuts

| Shortcut | Action |
| --- | --- |
| `Enter` | Submit input. |
| `Ctrl+J` or `Shift+Enter` | Insert a newline. |
| `Escape` | Interrupt an agent or close a dialog; `Esc Esc` opens rewind when input is empty. |
| `Ctrl+C` | Interrupt active work or quit while idle; clears non-empty input before quitting. Idle quit confirmation follows `ask_confirmation_on_exit`. |
| `Ctrl+D` | Delete right, or quit while idle according to `ask_confirmation_on_exit`. |
| `Ctrl+Z` | Suspend with a message. |
| `Ctrl+G` | Open the current plan/input in an external editor. |
| `Ctrl+O` | Toggle tool output. |
| `Ctrl+Y` or `Ctrl+Shift+C` | Copy the current selection. |
| `Shift+Up` / `Shift+Down` | Scroll chat. |
| `Ctrl+\\` | Toggle the debug console. |
| `Ctrl+Shift+A` | Toggle the docked agent browser; use `D` for details or `F1` for local help inside it. |
| `Alt+Left` / `Alt+Right` | Move by word in input. |
| `Ctrl+V` | Paste a clipboard image where that platform feature is available. |

Prefix a line with `!` to run a user-authored shell command directly, or type
`@` followed by a path for path completion.
