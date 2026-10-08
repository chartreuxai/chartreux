# Configuration

## Configuration layers

Chartreux resolves normal settings in this order, from lower to higher precedence:

1. Built-in defaults.
2. User configuration: `$CHARTREUX_HOME/config.toml` (normally `~/.chartreux/config.toml`).
3. Trusted project configuration: `.chartreux/config.toml`, discovered by searching upward from the working directory.
4. `CHARTREUX_*` configuration environment variables.
5. The active agent profile.
6. Runtime overrides.

An untrusted project configuration is not loaded. Project configuration selects catalog entries but cannot define providers or deployments; those belong in the user model catalog. See [Models](models.md).

`CHARTREUX_` setting names are case-insensitive and use `__` for nesting, for example:

```bash
export CHARTREUX_PROJECT_CONTEXT__TIMEOUT_SECONDS=5
```

Configuration environment variables do not discard empty values: a valid empty string or empty list remains a setting, while an invalid empty number, boolean, or structured value is an error. `CHARTREUX_HOME` is a separate location control, not a configuration-field prefix.

## User files and custom home

By default, user-owned files are under `~/.chartreux`:

- `config.toml` holds selections and other general settings.
- `models.toml` is the optional overlay for the shipped model catalog.
- `.env` can hold provider API keys.
- `logs/` holds local logs.

Set `CHARTREUX_HOME` before starting Chartreux to use another home directory:

```bash
export CHARTREUX_HOME="/path/to/chartreux-home"
```

This changes the locations of the files above as well as user prompts, agent profiles, and related user data.

Dispatch policy is owned only by the user catalog overlay in `models.toml`,
not `config.toml` or trusted project configuration. Its `[dispatch]` table
selects a shipped `mode` (`orchestrated` or `standalone`) and may sparsely
replace named `slots`. Slot fields include `profile`, `role`, `purposes`, and
`implements`; omitted fields inherit the selected preset, while list-valued
fields replace the whole list. Invalid dispatch data falls back atomically to
the standalone preset and emits a visible diagnostic; unrelated valid model
catalog entries remain available. This fallback does not repair or overwrite
the invalid file. Absent an explicit selection, fresh sessions use standalone:
direct implementation with delegated verification and independent review, with
all approval gates intact. Select orchestration explicitly with:

```toml
[dispatch]
mode = "orchestrated"
```

Legacy tier entries do not infer a mode. A saved change takes effect in the next session.
See the [configuration reference](../reference/configuration.md#dispatch-overlay).

## API keys and `.env`

A provider declares the name of its credential variable. For example, the shipped Mistral provider uses `MISTRAL_API_KEY`:

```bash
export MISTRAL_API_KEY="your-api-key"
```

Run `chartreux --setup` from an interactive terminal to configure a provider,
credentials, and models. Provider and model saves remain explicit. Setup then
shows an automatically seeded summary: usable existing role selections are
preserved, and unavailable bindings are assigned a ready configured model.
No per-role questions are required. **Finish setup** saves the seeded presets
and completes setup; **Customize role presets** opens the existing model/thinking
editor for optional changes or repairs. Invalid or unavailable bindings block
completion with repair guidance.

The presets screen defaults to **Standalone** (direct implementation) for fresh
users and also offers **Orchestrated**, including orchestration with one canonical
model. Explicit existing selections are preserved. After saving a second usable
canonical model, a live compaction event is the primary reliable need signal
for a passive graduation nudge at an idle boundary. The conservative v1 failure
trigger counts only two deduplicated, budget-exceeded implementation-purpose
attempts on the same task by the same retained agent; other failures and
ambiguous slot identities do not count.
It opens the presets screen, never changes mode automatically, and
can be dismissed permanently; headless sessions do not show it. Saved mode
changes apply next session, not to an already running conversation. Setup uses
the configured theme (`auto` by default), without asking for a theme choice.
Web search is not a required setup stage: saved choices remain unchanged, and
Settings > Web search or `/web-search` remains available later.
Setup saves configured provider keys in `$CHARTREUX_HOME/.env`; if persistence
fails, it reports when a key is available only for the current session.
Without an interactive terminal, setup prints actionable guidance instead of
launching the TUI. You can also create that file yourself:

```dotenv
MISTRAL_API_KEY=your-api-key
```

Credential precedence is precise: a **non-empty** value already present in the process environment wins. If that variable is unset or empty, Chartreux loads a non-empty value from `.env`. At runtime, API-key lookup falls back to the keyring only when no non-empty environment value is available. Setup and the Web Search editor save keys to `.env` when possible, with a session-only outcome if disk persistence fails; they do not save onboarding credentials to the keyring. The keyring is only a runtime lookup fallback. Keep `.env` private and do not commit it.

For web search, use Settings > Web search or `/web-search` to select a provider and manage its credential separately from search settings. A key may be saved or kept for the current session; see [Tools and safety](tools-safety.md) for provider, readiness, and retry behavior.

## Status line and message timing

Open `/settings` and select **Status line** to customize the bottom session row.
`Enter` or `Space` cycles a segment's state; directory and context variants are
part of that cycle, and those two segments cannot be disabled. Use `[` and `]`
to reorder segments. The live example preview reflects pending edits, not live
usage. **Apply changes** saves the batch to the user configuration; leaving with
unsaved edits asks whether to discard them. `Ctrl+R` removes saved status-line
overrides after confirmation, allowing lower-precedence values to apply. `D`
shows details and `F1` opens local help.

The same settings can be written in `config.toml`:

```toml
[status_line]
segments = ["directory", "git-branch", "context", "background-jobs"]
directory_style = "name"
context_style = "tokens-percent"
separator = "pipe"
```

Enable **Background jobs** (`background-jobs`) for `Jobs N`, including `Jobs 0`
when none are active. This optional count covers managed shell jobs in the current
root session and all its children, including jobs that survive child completion;
it does not count background agent runs or finished retained job records. It
refreshes from live session state between turns. The default row still contains
only directory, PID, and context. At narrow widths, PID yields first, then
optional segments from the end, so place Jobs earlier if it should outlast other
optional details.

Context displays as, for example, `135k/400k (34%)`; the denominator is the
effective automatic-compaction threshold, not the model's maximum context window.
Git branch lookup is asynchronous and cached. Enable `spend-today`, `spend-week`,
or `spend-month` to show recorded USD spend across **all projects**, for example
`Today $12.34`, `Week $12.34`, or `Month $12.34`. Windows are local calendar day,
Monday-start week, and month,
not rolling intervals. `—` means loading or unavailable, `$12.34+` is a known
lower bound with some cost unknown, and `Unknown` means nothing is priced.
`$0.00` can mean a valid empty window or recorded zero-cost calls. Open `/usage`
for Details or a Current project filter; that filter never changes status-line scope. See the
[status-line reference](../reference/configuration.md#status-line) for accepted
segments and width handling.

`show_message_timestamps` controls known posting times and settled durations in
main and child transcripts. Headers show local `HH:MM`, with a date for messages
not posted today, without role labels. Tool summaries show per-call durations;
the last assistant message shows the whole-turn total. These values do not tick
while work runs. Turning the setting off hides the displays but does not stop
capture or persistence; older messages without timestamps stay unstamped.

## Logging and diagnostics

Use `chartreux doctor` to validate trusted configuration, catalog entries, and
model selection locally. Bare doctor does not load `.env`, inspect keyring
credentials, start subprocesses, or make network requests. For opt-in live
checks and billable provider smoke probes, see [Troubleshooting](troubleshooting.md#run-diagnostics).

Chartreux writes structured local logs to `$CHARTREUX_HOME/logs/chartreux.log`. Set `log_level` in `config.toml`, use `/log-level` for a session override or persisted setting, or set `LOG_LEVEL`. Valid levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`; the default is `WARNING`.

The `/log-level` picker edits session and saved config levels as drafts. Use
Tab/Shift+Tab to move between scope controls, levels, and **Apply changes**;
Up/Down moves within the levels, and Enter/Space or a badge click toggles the
selected scope's draft level. Only **Apply changes** or `Ctrl+S` applies the
batch. Escape asks before discarding unsaved edits.

The effective level is chosen in this order: session override, `DEBUG_MODE=true` or a valid `LOG_LEVEL`, `log_level` in configuration, then the default. For local data and network-traffic policy, see [Privacy](../project/privacy.md). For the complete setting schema, see the [Configuration reference](../reference/configuration.md).
