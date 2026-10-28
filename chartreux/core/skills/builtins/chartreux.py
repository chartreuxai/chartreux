from __future__ import annotations

from chartreux import __version__
from chartreux.core.skills.models import SkillInfo, SkillSource

_PROMPT_TEMPLATE = """# Chartreux CLI Self-Awareness

You are running inside **Chartreux**, an independently maintained CLI coding agent.
This skill gives you full knowledge of the application internals so you can help
the user understand, configure, and troubleshoot their Chartreux installation.

## Going Deeper

This builtin skill is the standalone reference for the running Chartreux version.
When behavior changes, update this reference and its synchronization tests together.

## CHARTREUX_HOME

The user's Chartreux home directory defaults to `~/.chartreux` but can be overridden via
the `CHARTREUX_HOME` environment variable. All user-level configuration, skills, tools,
prompts, workflows, orchestration strategies, logs, and session data live here.

### Directory Structure

```
~/.chartreux/
  config.toml          # Optional user configuration, created on first saved setting
  hooks.toml           # User-level hook definitions
  .env                 # API keys and credentials (dotenv format)
  chartreuxhistory          # Command history
  trusted_folders.toml # Trust database for project folders
  prompts/             # Custom identity prompts (*.md)
  workflows/            # Custom workflow bodies (*.md)
  orchestration_strategies/ # Custom strategy bodies (*.md)
  skills/               # User-level skills
  tools/               # Custom tools (<name>.py); descriptions & overrides in tools/prompts/<name>.md
  logs/
    chartreux.log           # Main log file
    session/           # Session log files
  plans/               # Session plans

~/.agents/
  skills/              # Additional user-level skills directory
```

### Project-Local Configuration

When in a trusted folder, Chartreux also looks for project-local configuration:
- `.chartreux/config.toml` - Project-specific config (overrides user config)
- `.chartreux/hooks.toml` - Project-specific hooks (requires trusted folder)
- `.chartreux/skills/` - Project-specific skills
- `.chartreux/tools/` - Project-specific tools (`<name>.py`); a `prompts/<name>.md` beside them sets or overrides the description of the tool named `<name>` — builtin, MCP, or custom (e.g. `.chartreux/tools/prompts/bash.md` re-describes `bash`). Same `tools/*.py` + `tools/prompts/*.md` layout as the builtins.
- `.chartreux/agents/` and `.chartreux/prompts/` - Retired project-local resource directories; detected loudly and never consumed
- User prompts, workflows, and orchestration strategies live only in `~/.chartreux/prompts/`, `~/.chartreux/workflows/`, and `~/.chartreux/orchestration_strategies/`
- `.agents/skills/` - Standard agent skills directory

### AGENTS.md Discovery

`AGENTS.md` files provide directory-scoped instructions to the model. At startup,
Chartreux loads `~/.chartreux/AGENTS.md` and every `AGENTS.md` from the project root up
through the trust chain. `AGENTS.md` files in subdirectories are discovered
lazily: when `read_file` reads a file below the project root, any `AGENTS.md`
between the file's parent and the project root is injected into
context.

## Lifecycle: Exit, Update, Version, Resume

### Exit

Chat input (case-insensitive): `/exit`, `exit`, `quit`, `:q`, `:quit`.
When idle, `Ctrl+C` and `Ctrl+D` follow `ask_confirmation_on_exit`; `/exit`
quits immediately. During active work, `Ctrl+C` interrupts, while `/exit` opens
confirmation for consequential work. `Ctrl+D` deletes to the right in the input.
`Ctrl+Z` suspends on POSIX (resume with `fg`).

### Version

`chartreux --version` (or `-v`) prints it and exits. Not shown anywhere in-session.

### Resume

- `chartreux -c` / `--continue`: most recent session in this terminal (TTY-scoped;
  falls back to latest in cwd).
- `chartreux --resume [SESSION_ID]`: specific session; without an id, opens a picker.
- In-session: `/resume` (alias `/continue`).

#### Session titles

Each session has a `title` stored in `meta.json` (with `title_source`: `auto` or
`manual`). A session stays untitled until a background LLM call generates a
concise descriptive title (the `--resume` list shows a message preview until
then). Automatic generation runs only for the interactive CLI; other clients
(ACP, app server, programmatic) keep their own session management and fall back
to the message preview. Title generation runs on the session's active
model/provider. The first title waits for the opening turn to finish (or a few
model steps) so it isn't generated off a thin tool-call preamble. Because title
generation uses the active model, it stays bounded — one title at the start plus
one after a compaction, a couple at most — so the active model isn't re-invoked
every few turns. The refresh keeps the opening intent and the latest exchange in
view and feeds the previous
title back so it refines rather than restarts. `/rename <title>` sets a `manual`
title that auto-generation never overwrites. Set `session_logging.generate_titles
= false` to turn automatic titles off (the `--resume` list and tab then use the
message preview). The current title also drives the terminal tab/window title
(OSC), updated on rename, auto-title changes, and resume; it never blocks a turn.

#### Session storage & folder scoping

Local sessions are written under `~/.chartreux/logs/session/` (override with
`session_logging.save_dir`). Each session records the `cwd` it ran in. The
`/resume` picker, `--continue`, and bare `--resume` (no id) are **scoped to the
current folder**: only sessions whose `cwd` matches where Chartreux is launched are
listed, so the same directory shows its own history and nothing else. Switch
folders to see a different set. The explicit `--resume <SESSION_ID>` form is
**not** folder-scoped: it resolves the session by id regardless of which folder
it ran in.

Each conversation commits its resolved model and concrete provider deployment. Resume validates that binding and restores the frozen identity/workflow/strategy policy bundle; it does not re-resolve policy from current resource files. `/reload` refreshes live context and reports policy edits as pending. `/clear` starts a fresh conversation using current selections. `/model` and `/thinking` change the current session; configure the saved main model in `[roles.main]` in `models.toml`. The retired `@orchestrator` role has no alias.

## Configuration (config.toml)

The configuration file uses TOML format. When it does not exist, Chartreux uses its
built-in defaults. CLI startup creates an empty selections-only `config.toml` and
an empty `.env` when they are absent; the shipped catalog supplies provider and
model defaults. Effective configuration is layered, from lowest to highest precedence:

1. built-in defaults;
2. user TOML (`~/.chartreux/config.toml`);
3. trusted project TOML (`.chartreux/config.toml`);
4. `CHARTREUX_` environment variables; and
5. explicit invocation selections (CLI > environment > config > shipped default for identity, workflow, and orchestration strategy).

An untrusted project config is not loaded. Loading or inspecting persistence is pure: it does not write, publish, or cache a session change. Persistent saves require exactly one explicit `user` or `project` target and that target's expected revision; mixed or implicit persistent targets are rejected. Save results separate persistence (`not_saved`, `saved`, `durability_uncertain`) from application (`unchanged`, `applied`, `failed`), including saved-but-not-applied outcomes. Persistent saves never mirror values into session overrides.

Catalog definitions (`models`, `providers`) come from the shipped catalog plus the
user-owned `models.toml` overlay, not a `config.toml` layer. Project, environment,
and session sources may select canonical catalog models or roles but cannot define
catalogs. `authorized_roots_by_project` is accepted only from an installed user
layer. General `config/write` cannot grant or change roots; persistent roots must
be saved explicitly in the user configuration. In an interactive TUI session, an
out-of-root file-tool request prompts once per canonical root: **Allow this session**
grants it until the session ends (not after resume), while **Always for this project**
saves it to the user layer for future sessions. Saving uses the root orchestrator's
revision-checked policy/roots/save path; child requests never write configuration.
Save conflicts can be retried, and failed, uncertain, or cancelled saves leave the
grant session-only. `/reload` applies saved grants immediately. The prompt explains
that access includes read and write through shell and warns for roots at or above
`$HOME`; Esc or cancel denies. Programmatic and ACP sessions never prompt and point
to `authorized_roots_by_project` in user `config.toml`. Session-only `policy/roots/read`
and `policy/roots/replace` can change in-memory roots only, and replacement
requires an expected revision plus literal `userInitiated: true`. These RPCs never
write disk.

Ordinary higher-precedence settings do not remove independent permission or other source restrictions.

Configuration environment variables use the `CHARTREUX_` prefix. Nested schema fields
use `__`, such as `CHARTREUX_SESSION_LOGGING__ENABLED`. Supplied unknown names or
malformed values fail with an error identifying the `environment` source and field; the
offending value is not echoed. The exact non-config controls `CHARTREUX_HOME`,
`CHARTREUX_TYPING_GRACE_PERIOD_MS`,
`CHARTREUX_ACP_LOGGING_ENABLED`, `CHARTREUX_TEST_DISABLE_KEYRING`, and
`CHARTREUX_TEST_DISABLE_AUTO_TITLE` remain honored by their consumers. Provider
`VIBE_*` variables are not a fallback.

Custom identity prompts resolve only from `~/.chartreux/prompts/`. Workflows and orchestration strategies resolve only from their user directories. Project-local resource directories are detected loudly and never consumed; reserved built-in names cannot be shadowed.

### Key Settings

Agent behavior is selected in `[agent]`:

```toml
[agent]
identity = "coder"                  # Main identity
workflow = "minimal"                # Conversation workflow
orchestration_strategy = "none"     # Delegation packaging strategy
```

Selections resolve CLI > environment > config > shipped default and freeze with resource content for the conversation. `/reload` reports changed selections and resources as pending; they apply to new conversations. `/clear` starts a conversation with current selections, while resume restores the frozen bundle. The saved main model and thinking preset are configured in `[roles.main]` in `models.toml`; purpose routes are configured under `[orchestration.routes.*]`.

`system_prompt_id`, profile settings, dispatch settings, agent path/filter settings, and `CHARTREUX_ENABLED_AGENTS` are retired and rejected with actionable diagnostics. Custom identities come only from user prompts; the strict loader rejects malformed or empty resources, and reserved built-in names are unshadowable.

theme = "auto"  # Follow terminal background, then OS light/dark preference
disable_welcome_banner_animation = false
autocopy_to_clipboard = true  # Enable automatic copying of selected text to clipboard
file_watcher_for_autocomplete = false
ask_confirmation_on_exit = true  # Confirm idle Ctrl+C/Ctrl+D quits
ascii_chrome = false  # Use ASCII equivalents for application chrome glyphs
log_level = "WARNING"  # Optional. DEBUG | INFO | WARNING | ERROR | CRITICAL — log level for ~/.chartreux/logs/chartreux.log
displayed_workdir = ""  # Optional working-directory label shown in the UI
context_warnings = false  # Show context-window warnings
show_thinking_nodes = false  # Show reasoning/thinking nodes in the UI
```

### Copy and Text Selection

- **Copy shortcuts**: `Ctrl+Y` and `Ctrl+Shift+C` both copy the current selection to the clipboard. When autocopy is enabled (default), releasing the mouse over a selection also copies automatically. Each successful copy flashes a brief inline "Copied to clipboard" notice.
- **Multi-click selection**: Double-click selects a word, triple-click selects the current paragraph; dragging extends the selection at the same granularity.

```toml
# Behavior
compaction_model = ""             # Canonical model or @role; empty uses current main model, same provider required
enable_notifications = true
enable_system_trust_store = false  # Use OS trust store for outbound HTTPS
api_timeout = 720.0               # API request timeout in seconds
api_connect_timeout = 10.0        # HTTP connection timeout in seconds
api_write_timeout = 30.0          # HTTP write timeout in seconds
api_pool_timeout = 10.0           # HTTP connection-pool timeout in seconds
raise_on_compaction_failure = false  # Raise instead of continuing after compaction fails
api_retry_max_elapsed_time = 300.0  # Retry budget for retryable API failures in seconds
auto_compact_threshold = 200000   # Fallback for models without their own threshold

# Git commit behavior
include_commit_signature = true   # Add "Co-Authored-By" to commits

# System prompt composition
include_model_info = true         # Include model name in system prompt
include_project_context = true    # Include project context (git info, cwd) in system prompt
include_prompt_detail = true      # Include OS info, tool prompts, skills, and agents in system prompt

[project_context]
default_commit_count = 5           # Recent commits included in project context
timeout_seconds = 2.0              # Project-context collection timeout in seconds
```

`bypass_tool_permissions` is a historical configuration key. It does nothing
and is rejected in v0.1; remove it from configuration files.

### Local Diagnostics

Chartreux does not create product analytics or OpenTelemetry spans, configure telemetry exporters, or send telemetry to a remote service. It retains local diagnostic and session logs plus token and cost accounting. Network traffic includes requests to explicitly configured provider and MCP endpoints, user- or model-requested `web_fetch` URLs, and the configured `web_search` provider. `web_fetch` can retrieve arbitrary URLs; `web_search` contacts its selected provider. This policy does not promise that provider SDK packages or arbitrary user extensions contain no OTel code or perform no separately configured traffic.

### Model Catalog

Providers, models, and roles live in `~/.chartreux/models.toml` (or `$CHARTREUX_HOME/models.toml`), a sparse user overlay on the shipped catalog. Configure the saved main-model default under `[roles.main]`. Purpose routes live under `[orchestration.routes.*]`; legacy `[dispatch]` input is retired and diagnosed with successor guidance. `/model` and `/thinking` override the current session. If a legacy `config.toml` contains `active_model`, move that selection to the `[roles.main]` preset; if it contains `providers` or `models` tables, run `chartreux models migrate` (then `chartreux models migrate --apply` after reviewing the preview). Use `/providers` to manage providers in Provider Settings, or `chartreux --setup` for onboarding.

Generic providers support the `openai`, `openai-responses`, and `anthropic` API
styles. The Mistral backend uses `backend = "mistral"`. Provider definitions may
set `emits_finish_reason = false` for endpoints that do not reliably terminate
streams with a finish reason, and `extra_headers = { "Header-Name" = "value" }`
to attach additional HTTP headers to provider requests.

#### API-style migration

`vertex-anthropic` and `reasoning` API styles are removed. The former reasoning
adapter encoded thinking blocks inside message `content`; this is not equivalent to
`reasoning_field_name`, which only renames a separate reasoning-content field. That
protocol is intentionally deprecated, not migrated.

`supported_thinking_levels` in a catalog model narrows the selected deployment's
encodable levels; omitting it does not disable thinking. A declaration inconsistent
with its effective model entry or provider is rejected as a typed configuration error.
Provider admission describes wire encodability, not a guarantee that an undeclared
model accepts every admitted effort. GLM-5.2 maps both `low` and `medium` to `high`,
so that escalation is a no-op. GLM-5.3 models require a non-`off` configured
thinking default because they cannot disable thinking; `off` is rejected as a typed
configuration error. Anthropic history can force requested `off` to effective
`medium` even when a narrowing declaration excludes `medium`, because declarations
constrain requested levels only.

### Tool Configuration

```toml
# Additional tool search paths
tool_paths = ["/path/to/custom/tools"]

# Enable only specific tools (glob and regex supported)
enabled_tools = ["bash", "read_file", "grep"]

# Disable specific tools after enabled_tools filtering
disabled_tools = ["web_fetch"]

# Per-tool configuration
[tools.bash]
permission = "always"
denylist = ["gdb", "pdb"]

# Web search: provider is auto, mistral, exa, brave, or duckduckgo.
# Keyed providers read their default credential environment variable unless overridden.
[tools.web_search]
provider = "brave"
api_key_env_var = "BRAVE_SEARCH_API_KEY"
timeout = 30
max_results = 5
```

`web_search` defaults to `auto`, which always selects Mistral and never falls
back to another provider. An explicit `api_key_env_var` takes precedence. For
`auto` and `mistral`, the selected variable is the configured Mistral
provider's credential variable, or `MISTRAL_API_KEY` when no Mistral provider
is configured; `exa` and `brave` use `EXA_API_KEY` and `BRAVE_SEARCH_API_KEY` by
default. Only the selected variable is checked. `duckduckgo` requires no
credential. Use Settings > Web search or `/web-search`; `Save API key` is
independent of `Save search settings`, and reports whether the key was saved or
is available only for the current session. Readiness does not verify a live
connection. Standalone Settings shows Mistral once; `auto` remains a supported
configuration alias for Mistral. First-run setup finishes after the role
summary or customized presets without checking or changing web-search settings.
Configure optional search afterward via Settings > Web search or `/web-search`.
`read_image`
is available only when the active deployment in `models.toml` has
`supports_images = true`.

For an `openai-responses` provider that does not accept images in function-call
outputs, set `supports_tool_result_images = false` in its `models.toml`
provider table. Chartreux then projects tool-result images into a synthetic
user turn instead of sending them in the Responses function output.

The built-in shell surface exposes the POSIX `bash` tool. It reads permissions
and denylists from `[tools.bash]`; its resolver's hard guards determine shell
policy. The historical `[tools.bash].allowlist` key is rejected in v0.1; remove
it from configuration files. Recognized dangerous git commands carry typed
policy: `git push` (publication) and the consequential operations (forced,
deleting, or mirrored pushes, `git checkout`, `git restore`,
`git switch --discard-changes`/`-f`, `git stash drop`/`clear`,
`git reflog expire`/`delete`) are denied pending authority — interactive
sessions prompt the user (Deny default; publication grantable once, per session,
or always for the project; consequential once or per session, never always),
and noninteractive sessions deny with remediation naming the actual path:
publication operations point to `dangerous_command_grants_by_project` for
this project in the user config.toml, consequential operations cannot be
pre-authorized and require approval in an interactive session, and a mixed
call covers only the persistable subset. The denial carries the resolver's
reason. A refusal means the user declined: do not retry that
exact command. Hard guards (`git reset --hard`, forced non-dry-run `git clean`,
recursive `rm`) are never grantable — the user runs them. An explicit
`denylist` entry is an absolute veto no grant overrides. Approval is permission
authority, not standing intent to publish. Inspect and revoke dangerous-command
authority in Settings > Shell Permissions; there is no public grant endpoint. Each
model-facing call is finite: it
starts a fresh shell, closes stdin so the command receives EOF, enforces a timeout,
and returns separate stdout and stderr for local execution. Shell state and process
handles do not persist across calls. Managed background sessions, polling, continued
stdin, and cursor-based output access are intentionally not provided.
When an ACP client advertises terminal capability, retained client-terminal delegation
is used instead; that client-produced output may be merged, so the local stream split
does not apply to the delegated path.

**Special case — `find` command:** Chartreux detects `-exec`, `-execdir`,
`-ok`, and `-okdir` predicates and keeps those command forms subject to the
shell safety policy.

#### File Tool Permission Resolution

File-based tools (`read_file`, `read_image`, `grep`, `write_file`, `edit`) enforce runtime authority in this order:

1. **Tool denial, denylist, or sensitive-pattern match** → always denied
2. **Runtime-owned Plan/scratchpad scope** → allowed only for the exact designated Plan file or session scratchpad, including inherited scopes
3. **Authorized workspace roots** → allowed
4. **Otherwise** → denied; configured tool permissions do not grant filesystem authority

Configured allowlists never expand filesystem authority, and no arbitrary home-directory access is implied.

### Skill Configuration

```toml
# Additional skill search paths
skill_paths = ["/path/to/custom/skills"]

# Enable only specific skills
enabled_skills = ["chartreux", "custom-*"]

# Disable specific skills
disabled_skills = ["experimental-*"]
```

### Identity, Workflow, and Strategy

```toml
[agent]
identity = "coder"                  # coder, orchestrator, researcher, reviewer, advisor, or user:<name>
workflow = "minimal"                # minimal or user:<name>
orchestration_strategy = "none"     # none, default, or user:<name>

[orchestration.routes.<purpose>]
# Minimum number of distinct models required
# targets: required, non-empty list of exact model/identity/tier bindings

[subagents]
retain_agents = false
idle_ttl_seconds = 300
max_idle_agents = 4
tombstone_ttl_seconds = 900
max_tombstones = 32
```

The five built-in identities are coder (optional delegation, unrestricted personal tools within grants), orchestrator (required delegation; personally non-mutating), and researcher, reviewer, advisor (read-only, non-mutating personal and delegated tools; the reviewer alone carries a bash execution carve-out for non-mutating verification commands). Custom `user:<name>` identities are prompt-only and carry no enforced posture. Purpose routes bind launch targets, not permissions. Multi-seat purposes require an explicit seat; review execution uses coder with the shipped review overlay, while analysis seats use reviewer. Model diversity is checked by canonical model, not provider deployment.

Selections resolve CLI > environment > config > shipped default. The default model role is `@main`; `@orchestrator` is retired, with no alias. The conversation freezes identity/workflow/strategy content in its policy bundle. Reload reports edits as pending; resume uses the persisted bundle; `/clear` binds current selections. Resource layout is `~/.chartreux/prompts/`, `~/.chartreux/workflows/`, and `~/.chartreux/orchestration_strategies/`; only user resources resolve, with reserved built-in names and no shadowing. A per-launch `retain` overrides the global `[subagents].retain_agents` default; there is no identity-specific retention setting. Idle TTL and capacity are global defaults.

Legacy `system_prompt_id`, agent path/filter settings, dispatch slots, and profile settings are retired. `chartreux doctor` reports actionable retirement errors; rename model role `@orchestrator` to `@main`.

Routine tool execution has no approval dialog. Runtime denials, sensitive-file protections, workspace-root checks, and other scope checks remain enforced.

### MCP Servers

Remote MCP servers can be added non-interactively from the shell:

```bash
chartreux mcp add NAME \\
  --url <server-url> \\
  --transport streamable-http \\
  --api-key-env MISTRAL_API_KEY

chartreux mcp remove NAME
```

The supported transports are exactly `streamable-http` and `stdio`. The local
`http` alias is rejected; configuration loading does not migrate or rewrite old
transport values. Static auth is selected when `--api-key-env` or `--header` is
provided. Otherwise the server uses OAuth and starts its configured login flow
by default. Pass `--no-login` to only persist the OAuth configuration. Run
`chartreux mcp add --help` for all supported authentication and timeout options.
Use `/mcp logout <alias>` for an explicit decision to revoke stored OAuth
credentials; do not treat remove/re-add as an identity migration.

Configured OAuth MCP servers can also be added from inside Chartreux:

```text
/mcp add <server-url>
/mcp add <server-url> --name docs --scope read --transport streamable-http --no-login
```

`/mcp add` is OAuth-only, uses `transport = "streamable-http"`, and rejects
`--transport http`. It writes `auth.type = "oauth"` with optional scopes and
starts login by default.

```toml
[[mcp_servers]]
name = "my-server"
transport = "stdio"
command = "npx"
args = ["-y", "@my/mcp-server"]

[[mcp_servers]]
name = "remote-server"
transport = "streamable-http"
url = "http://localhost:8000"

[mcp_servers.auth]
type = "static"
api_key_env = "MCP_API_KEY"
api_key_header = "Authorization"
api_key_format = "Bearer {token}"

[[mcp_servers]]
name = "oauth-server"
transport = "streamable-http"
url = "http://localhost:8001/mcp"

[mcp_servers.auth]
type = "oauth"
scopes = ["read", "write"]
# Optional: client_id = "pre-registered-public-client"
# Optional: client_metadata_url = "https://example.com/client-metadata.json"
# Optional: redirect_port = 47823
```

HTTP MCP authentication belongs in the nested `auth` block and is either
`static` or `oauth`. Legacy top-level `api_key_env` and `headers` keys are
rejected with source-aware validation; they are not promoted or migrated.
OAuth credentials are bound to the configured server identity. If an alias is
changed to a different identity, use a separate alias; deliberate reuse
requires an explicit logout decision. An old grant is never silently consumed
or relabeled, and automatic grant deletion is not an identity fix.

MCP calls use one-shot HTTP requests and persistent, serialized stdio sessions
within the current session. Ambiguous transport failures are never automatically
replayed; a later explicit call may reconnect, but exactly-once execution and
remote rollback are not promised. Cancellation can prevent queued work and
retire active local transport cleanup, but cannot undo a remote side effect.

MCP results preserve `isError`, structured content, and text. Explicit `null`
tool arguments remain distinct from omitted arguments. Unsupported content blocks
produce type-only omission notices. Discovery follows `nextCursor` pages and
rejects repeated or invalid cursors within a finite page limit.

Descriptor caching is non-authoritative. Anonymous HTTP contexts may persist
descriptors; private contexts are session-only, and **all stdio contexts have no
persistent cache** because inherited environment and working-directory state may
affect their credentials and configuration. Private cache identities are not
published or derived from secrets.

### Session Logging

```toml
[session_logging]
enabled = true
save_dir = ""                     # Defaults to ~/.chartreux/logs/session
session_prefix = "session"
generate_titles = false           # Background LLM session titles; false uses the message preview
```

### Provider Authentication

Onboarding accepts an API key for the active provider and uses the theme already
configured for Chartreux; it does not ask for or save a theme choice. Providers
without an `api_key_env_var` do not require a key. Provider endpoints remain
configured through `api_base`; credentials use `api_key_env_var`.

### Hooks

Hooks let users run commands automatically at lifecycle events. They are always
available — no flag is required; dropping a `hooks.toml` in place is enough.
Commands are parsed with `shlex.split` using POSIX tokenization; they do not
receive shell expansion. Shell syntax such as pipes, redirects, or expansions
requires wrapping the command in `sh -c '...'`.

#### Config and hook types

Hooks live in `hooks.toml` files (separate from `config.toml`), discovered in
this order:

1. `<project>/.chartreux/hooks.toml` — loaded first, only when the folder is
   trusted.
2. `~/.chartreux/hooks.toml` — loaded second.

A duplicate `name` across the two files is reported as a config issue and the
project entry wins. Config-load errors (invalid TOML, missing required
fields) surface in the TUI as warnings and the offending hook is skipped.

```toml
[[hooks]]
name = "lint"                       # Required: unique within the file.
type = "post_agent"                 # Required: post_agent | pre_tool | post_tool.
command = "eslint --quiet ."        # Required: shell command run in cwd.
timeout = 60.0                      # Default: 60s for all hooks.
description = "Run ESLint"          # Optional.

[[hooks]]
name = "deny-rm-rf"
type = "pre_tool"
match = "bash"                      # Tool-name matcher (tool hooks only, default "*").
strict = true                       # Tool hooks only: escalate any failure to deny/clear.
command = "uv run python /path/to/guard-bash"
```

| Type | When it runs |
|---|---|
| `post_agent` | Once per turn, after the agent finishes responding (no pending tool calls). |
| `pre_tool` | Per tool call, before runtime policy is resolved. |
| `post_tool` | Per tool call, **iff the tool body actually ran**. `tool_status` is `success`, `failure`, or `cancelled`. Does not fire when the tool never executed (`pre_tool` denial, policy denial, `NEVER`, or cancellation before the body started). |

**Matcher syntax** (same as `enabled_tools`): fnmatch glob by default
(`"bash"`, `"read_*"`, case-insensitive), or a regex full-match when the
pattern starts with `re:` (`"re:(read_file|grep)"`). `match` is forbidden on
`post_agent`.

**Tool name conventions** for matchers:
- Built-in tools use their bare name (`bash`, `read_file`, …); see the Tools
  section above for the full list.
- MCP tools: `{server-name}_{raw-tool-name}` (e.g. `linear_create-issue`).
- Subagents all route through `task`. Match with `match = "task"` and read
  `tool_input.agent` to discriminate by subagent.

Subagent invocations inherit the parent's hook config. Their hook events are
logged to the subagent's session log and don't propagate to the parent's UI.

#### Wire protocol

Every hook is spawned in `cwd` and receives a JSON object on **stdin**
discriminated by `hook_event_name`:

```json
// post_agent
{"hook_event_name": "post_agent", "session_id": "...",
 "parent_session_id": null, "transcript_path": "...", "cwd": "..."}

// pre_tool
{"hook_event_name": "pre_tool", "session_id": "...", "parent_session_id": null,
 "transcript_path": "...", "cwd": "...",
 "tool_name": "bash", "tool_call_id": "call_42",
 "tool_input": {"command": "ls"}}

// post_tool
{"hook_event_name": "post_tool", "session_id": "...", "parent_session_id": null,
 "transcript_path": "...", "cwd": "...",
 "tool_name": "bash", "tool_call_id": "call_42",
 "tool_input": {"command": "ls"},
 "tool_status": "success",         // success | failure | cancelled
 "tool_output": {"stdout": "...", "stderr": "", "exit_code": 0, "returncode": 0},  // the tool's serialized result (success/cancelled); null otherwise
 "tool_output_text": "...",         // current text the LLM will see; mutable by prior hooks
 "tool_error": null,                // populated on failure/skipped
 "duration_ms": 42.5}
```

`parent_session_id` is set when running inside a subagent. Exceeding
`timeout` kills the whole process tree.

A hook signals back via its **exit code** and **stdout** (stderr is reserved
for diagnostics — Chartreux never parses it for control):

| Exit | Stdout | Behavior |
|---|---|---|
| `0` | empty | Pass through (no action). |
| `0` | valid structured-response JSON object (schema below) | Act per the JSON fields. |
| `0` | anything else (free-form text, broken JSON, scalar/array, schema mismatch) | Failure path (see below). The parse error is in the message. |
| non-zero / timeout / spawn failure | — | Failure path. Reason taken from stderr, then stdout, then the exit code. |

Structured-response schema:

```json
{
  "decision": "allow" | "deny",          // optional; default "allow"
  "reason": "string",                     // required when decision == "deny"
  "system_message": "string",             // optional UI note
  "hook_specific_output": {
    "tool_input": { ... },                // pre_tool only
    "additional_context": "string"        // post_tool only
  }
}
```

Unknown fields are tolerated at every level. Fields that aren't meaningful
for the current hook type are silently ignored.

**Don't self-name in `system_message` or `reason`** — the UI prefixes
hook-end-event content with `[hook-name]` automatically, and `pre_tool`
denials are wrapped as ``Tool 'X' was denied by hook 'Y': {reason}`` before
the LLM sees them. A hook that writes ``"reason": "guard: refused..."``
will produce ``hook 'guard': guard: refused...`` downstream.

`decision: "deny"` per hook type:

| Hook | Effect of `decision: "deny"` |
|---|---|
| `pre_tool` | Deny the tool call; `reason` is the tool error returned to the LLM. First deny short-circuits the remaining `pre_tool` hooks for this call. |
| `post_tool` | Replace `tool_output_text` with `reason`. Pipeline continues; subsequent hooks see the replacement. |
| `post_agent` | Inject `reason` as a retry user message. Capped at 3 retries per hook per user turn. |

Event-specific payloads:

- `hook_specific_output.tool_input` (`pre_tool`): full replacement of the
  model's arguments. Chartreux re-validates against the tool's schema **after each
  rewriting hook** — the first invalid rewrite aborts the chain and
  synthesizes a denial attributing the failure to that hook. Rewrites
  compose: hook N receives `tool_input` as rewritten by hooks 1..N-1.
- `hook_specific_output.additional_context` (`post_tool`): text appended
  (with `\n`) to the current `tool_output_text`. Composes with a same-hook
  `decision: "deny"`: deny replaces first, then `additional_context` is
  appended to the replacement.

**Failure path.** Any failure (non-zero exit, timeout, spawn failure,
non-conforming stdout) emits a UI warning and lets the gated action proceed
(fail open). With `strict = true` on a tool hook:

| Hook | Strict failure escalates to |
|---|---|
| `pre_tool` | Deny the tool call with the failure reason. |
| `post_tool` | Clear `tool_output_text` (replace with empty). |

`strict` is forbidden on `post_agent`.

#### Execution semantics

- Hooks of the same type fire sequentially in load order (project file first,
  then user file; declaration order within each file).
- Tool calls within a single LLM turn run **concurrently**; each call's hook
  chain runs serially but the chains run in parallel across calls. Hooks
  that touch shared state (filesystem, env) must coordinate themselves.
- `pre_tool` rewrites take effect everywhere downstream: runtime policy sees the
  rewritten arguments, the tool runs with them, and the assistant message is
  patched so subsequent LLM turns reflect what actually ran.

### Pattern Matching

Tool, skill, and agent names support three matching modes:
- **Exact**: `"bash"`, `"read_file"`
- **Glob**: `"bash*"`, `"mcp_*"`
- **Regex**: `"re:^serena_.*$"` (full match, case-insensitive)

## CLI Parameters

```
chartreux [PROMPT]                       # Start interactive session with optional prompt
chartreux -p TEXT / --prompt TEXT         # Programmatic one-shot mode, exit
chartreux --workdir DIR                  # Change working directory
chartreux --worktree NAME                # Create/reuse a git worktree under $CHARTREUX_HOME/worktrees on branch NAME and run inside it. Auto-cleanup only for worktrees Chartreux created this run and only after a session started; reused worktrees and attached (pre-existing) branches are kept unless confirmed. -p sessions keep worktrees. Ignored with --setup.
chartreux --worktree                     # Same, but Chartreux picks an unused name from the prompt (a random slug when there is no prompt) on a chartreux/<name> branch, and never reuses an existing worktree. The prompt must precede the flag or follow a `--`, since --worktree otherwise reads it as NAME.
chartreux --add-dir DIR                  # Extra working dir loaded for context (repeatable). Implicitly trusted.
chartreux --trust                        # Trust cwd for this invocation only (not persisted). Skips the trust prompt.
chartreux -c / --continue                # Continue most recent session in this terminal (TTY-scoped, falls back to latest in cwd)
chartreux --resume [SESSION_ID]          # Resume a specific session
chartreux -v / --version                 # Show version
chartreux --setup                        # Run onboarding/setup
chartreux --identity ID                  # Identity for a new conversation (e.g. coder, orchestrator, or user:name); frozen for that conversation
chartreux --workflow ID                  # Workflow for a new conversation (built-in or custom workflow name)
chartreux --orchestration-strategy ID    # Orchestration strategy for a new conversation (none, default, or a custom strategy name)
chartreux --max-turns N                  # Max assistant turns (programmatic mode)
chartreux --max-price DOLLARS            # Max cost limit (programmatic mode)
chartreux --max-tokens N                 # Max total session tokens (programmatic mode)
chartreux --enabled-tools TOOL           # Enable specific tools (repeatable)
chartreux --disabled-tools TOOL          # Disable specific tools (repeatable)
chartreux --output text|json|streaming   # Output format (programmatic mode)
```

## Built-in Subagents

Chartreux has one primary agent; `--agent` and `Shift+Tab` agent selection are
unavailable.
Routine tool execution is auto-approved without a dialog; runtime denials and
scope checks remain enforced.

### Subagents

Chartreux ships identity-based agent selection; use `identity` rather than the retired profile `agent_type` axis. The built-in identities are `coder`, `orchestrator`, `researcher`, `reviewer`, and `advisor`; their authority posture is enforced by the harness. The default child identity is `coder`; read-only analysis identities are available for research, advice, and review. The `@main` role is the default model role, and purpose routes in `[orchestration.routes.*]` bind model, identity, tier, and optional seat. Routes organize launches, not permissions. Multi-seat purposes require an explicit seat; review execution uses coder with the shipped review overlay, while analysis seats use reviewer. Model diversity is measured by canonical model, not provider deployment.

The global `[subagents]` settings control retention defaults and capacity: `retain_agents`, `idle_ttl_seconds`, `max_idle_agents`, `tombstone_ttl_seconds`, and `max_tombstones`. By default, background agents close when their run completes unless they are retained; completed results remain retrievable until result expiry. A per-launch `retain` overrides the global `[subagents].retain_agents` default. Its precedence is the launch argument, then the global `[subagents].retain_agents` setting; there is no identity-specific retention setting. On reuse, omit `retain` to inherit the instance's committed retention policy.

Use `task(identity="researcher", task=...)` to create an agent instance. `agent_id` identifies that instance, with a display-only name; `agent_id` remains the only handle. Launch in the background even when the result is a dependency, then wait for it with `wait_for_agent`. A wait returns `completed` or `still_running`; the default timeout is 1500 seconds and does not stop the run. `task_summary` identifies retained work in `check_agents`; `get_agent_result` retrieves completed results. The inventory reports `ttl_remaining_seconds` as an advisory estimate. Retained instances can be reused for a genuine continuation; `check_agents` shows their availability, effective model and thinking, task summaries, and TTL estimate. Reuse an `idle` agent only for a genuine continuation with the same identity and feature or subsystem; include current scope and intervening changes because retained conversation is not proof the working tree is unchanged. Persona is immutable on re-task. A `running` agent is busy; wait if its result is needed. Idle retained agents may be evicted by TTL or capacity. Closed and evicted agents cannot be reused, though results remain retrievable until expiry. `release_agent` discards retained results, so retrieve them first. Background agents cannot launch background agents.

The `/agents` browser shows identity and agent lifecycle state alongside task summaries, model, run status, and idle/TTL information. Use it to inspect or manage active, idle, and recent agents; release discards retained results.

### Cooldown override and the resumed-run deadline

When measured endpoint failures put a deployment's endpoint into a cooldown, a stuck launch can still be forced out: the explicit `override_cooldown: true` argument on `task` (fresh and replacement launches) and on `resume_agent` (suspended-run recovery) requests a forced un-stick attempt. The flag only REQUESTS the override — root launch admission mints the endpoint-bound permit from the root recovery ledger solely for the explicit request, and the ledger still denies the mint when the endpoint's episode is exhausted (two forced starts per episode), spacing-bound (30s between starts), or has an attempt in flight. A denied mint consumes nothing; a forced attempt is never automatically replayed, and automatic paths never acquire permits. An accepted override bypasses only the cooldown predicates — allowlists, compaction, thinking, and history checks still apply — and the launch result carries bounded un-stick diagnostics: the forced deployment, the exclusions bypassed, the allowance remaining in the episode, and the next permitted start.

Every resumed run is also bounded by the first-semantic-progress deadline: its first model call must observe a semantic delta (content, reasoning, or tool-call) within `[recovery.subagents].resume_progress_deadline_seconds` (default 180s) or the in-flight call is killed and the run fails with the measured `resume_progress_deadline` timeout. Heartbeat and usage-only chunks do not count as progress; ordinary (non-resumed) runs are structurally unaffected.

### Completion audit

A run's result can carry a typed, versioned completion-audit stamp (`completion_audit`, schema version 1) whenever the run had an audit surface: file tools that touched paths, shell commands with a write-classified footprint, git evidence, or a frozen contract. The stamp classifies the run's completion claims on separate axes — claim fidelity (`claim_matched`, `claim_unverified`, `claim_contradicted`), disclosure completeness, mandate satisfaction, declared outcome, and the verifier's primary runtime failure — plus an optional typed rejection reason. A stamp means the claims matched the harness-collected evidence; it never means the task was actually correct. Vacuous or irrelevant check passes, and artifacts the harness cannot observe, classify as unverified rather than credited.

Mandates are optional per launch: `task(completion_check=[...], required_artifacts=[...])` freezes a versioned envelope (marker `completion_envelope_v1`) into the child's prompt at admission, and `None` mandates disable only themselves (fail-open). The closed check templates are `pytest`, `ruff_check`, `pyright`, `git_head`, `git_status`, and `git_diff_stat`; the closed artifact kinds are `commit`, `touched_path`, `check_pass`, and `envelope`. Each requirement binds exactly one template or kind and carries a stable requirement id (`check:<template>` or `artifact:<kind>`); executor-minted occurrence ids pair the run's tool invocations with their evidence receipts. Substantive seats require the envelope; reviewer seats are exempt; read-only seats carry a reduced matrix (no commit mandate); shell-less launches are exempt from the check-based classes. The worker answers with a receipt carrying the same marker and its satisfied requirement ids — the last receipt-shaped JSON object in the final report is the operative claim. Grammar violations (a malformed, duplicate, or oversized envelope; unknown or duplicate requirement ids) are disclosure facts, never worker faults; only `claim_contradicted` is a worker-side disposition.

The `[completion]` config section sets the audit mode, frozen per run at admission. `stamp` (the default) attaches the classification to the result and never blocks publication. `enforce` additionally blocks the submission on the worker-side disposition or a rejected check claim: the result publishes as not completed with a bounded, actionable rejection message, and a background run finalizes FAILED with a rejection envelope. A rejection envelope is prohibited for recovery — it authorizes a cross-agent replacement, never a warm same-agent resume. Runtime failures and cancellations are never audit rejections: a cancelled run takes the cancellation disposition (its reason is recorded, no claim rides, and the bound checks are never re-run), and an honest abort or blocker outcome keeps its primary disposition.

The harness runs the mandated checks itself at finalization. Templates map to fixed argv (never shell-interpreted), spawn at most once, with a 120-second timeout and capped output capture. The check capability is run-scoped and template-only, derived from the parent's delegation authority; it never widens the child's identity posture and never prompts at finalization — a denied or unavailable authority stamps the check unavailable, an informational stamp, never a worker fault. Residuals to know: the acceptance check executes project code (conftest and test bodies run with the project's privileges), so authorized-root validation of the check target is path permission, not containment; a managed-job startup observation (a receipt without a completed result event) is not check evidence; and custom-tool and MCP-tool writes are outside the touched-path accounting — a claim about an artifact produced through them resolves unverified, never credited and never condemned.

The evidence is per-run and bounded: the chronology caps at 128 entries, and evidence that outlived the caps, a truncated snapshot, or a shell write surface that could not be established classify as unavailable — an availability fact, never a contradiction. The stamp rides on the completed result and stays retrievable until result expiry; no chronology content reaches the parent-facing result, and mixed-version consumption of a persisted result with a populated audit is unsupported. No TUI rendering of the stamp is provided.

## Built-in Slash Commands

- `/help` - Show help message
- `/model` - Select active model
- `/thinking` - Select the thinking level for this session (stored in the ephemeral
  session override, not `config.toml`)
- `/theme` - Select Textual UI theme; `auto` follows terminal/OS appearance (persisted in config)
- `/settings` - Browse and edit curated user settings in the bottom panel
- `/web-search` - Configure the web-search provider and credentials in Settings
- `/reload` - Refresh AGENTS.md, skills, and catalogs; frozen identity/workflow/strategy content stays bound to this conversation, and reports pending selections or resource edits for new conversations
- `/open-config-file` - Open the user `~/.chartreux/config.toml` in the user's editor
  (`$VISUAL`/`$EDITOR`, falling back to `nano`); creates the file with a
  commented template when absent and reloads configuration after editing if the
  file changed
- `/clear`, `/new` - Start a new conversation. Optionally pass a prompt to seed it
- `/log` - Show path to current interaction log file
- `/log-level` - Show the effective log-level chain (session, environment,
  configuration, effective).
- `/debug` - Toggle debug console
- `/agents` - Toggle the expanded Background Agents list above the input. The
  statusline counts active work only (running and finalizing), with a dim
  recent or idle note when no work is active. The list groups agents by
  lifecycle —
  Active, Retained Idle, and Recent (closed/evicted terminal metadata) —
  behind the Resident (default), Active, Idle, Recent, and All filter tabs,
  cycled with Left/Right. Rows show the agent's name with its handle dimmed,
  plus identity, task summaries, availability, run status, model, turns used,
  run ID, and idle/TTL information. Select an agent to open its bordered
  transcript pane, available within the terminal-metadata window when a saved
  transcript exists; running agents refresh live about once per second. Stop a
  running agent with `C` and release an idle or recent agent with `R` (both
  confirmed; release discards the agent's retained results). Use Up/Down and
  Enter or mouse selection, PageUp for older pages, `r` to refresh, and Escape
  (or the Main agent entry) to return to the conversation. `Ctrl+Shift+A` is
  the keyboard shortcut for the same toggle.
- `/compact` - Compact model context by summarizing. The session ID and visible
  conversation stay intact; the auto title is refreshed to reflect the
  compacted conversation (unless renamed manually).
- `/rename <title>` - Set a manual session title. Persists to `meta.json`
  (`title_source=manual`), updates the terminal tab title, and is never
  overwritten by automatic title generation.
- `/retry [additional instructions]` - Continue a model response interrupted by
  a backend error without repeating text already shown. Optional instructions
  are passed to the model for the continuation. Relevant error messages also
  hint at this command.
- `/status` - Display agent statistics
- `/usage` - Browse recorded usage and USD cost estimates. Read-only Day/Week/Month
  snapshots default to All projects; Current project filters the browser only.
  Available while a turn runs without interrupting work. Use `r` to Refresh,
  `d` for below-table Details, `F1` for help, and Escape to go back or close.
  The TOTAL row summarizes the window; clickable shortcuts offer the same actions.
  Recorded usage only: pre-ledger sessions are excluded; estimates are not provider invoices.
- `/copy` - Copy the last agent message to the clipboard
- `/paste-image` - Paste an image from the OS clipboard into the prompt.
  **macOS only** — the command is not registered on Linux.
- `/mcp` - Display MCP server status. The
  browser opens on the first item; press Up or Left to move into the fuzzy-search
  bar, and Up again to wrap to the last item. Pass a server name to
  list its tools or open its auth panel when authentication is required
- `/mcp add <url>` - Add a configured OAuth MCP server. Supports `--name <alias>`,
  repeatable `--scope <scope>`, `--transport <streamable-http>`, and
  `--no-login`. Starts OAuth login by default. OAuth-only; use
  `chartreux mcp add <name> --url <url> --api-key-env <var>` for API-key/static auth.
- `chartreux mcp remove <name>` - Remove an MCP server from the user configuration.
  Removal is not an identity migration; use `/mcp logout <alias>` to explicitly
  revoke stored OAuth credentials.
- `/mcp status` - Display MCP auth state (`ok`, `needs_auth`, `static`, `stdio`)
- `/mcp login <alias>` - Start OAuth login for an MCP server
- `/mcp logout <alias>` - Log out from an MCP server and delete stored OAuth
  secrets
- `/resume` (or `/continue`) - Browse and resume past sessions for the current
  folder. The picker header shows the folder being listed. Press `d` twice to
  delete a saved session; the active session cannot be deleted here.
- `/branch` - Fork the current conversation into a new resumable session,
  leaving this session unchanged. Resume the copy in another terminal with
  `chartreux --resume <id>` (the id is printed when the branch is created).
- `/rewind` - Rewind to a previous message. Also triggered by pressing `Esc`
  twice on an empty input; if the input has content, the first double-`Esc`
  clears it instead. In the rewind panel: `↑/↓` pick option, `Shift+↑/↓`
  scroll, `←`/`Esc` edit previous message, `→` edit next message, `Enter`
  accept, `q` quit.
- `/loop <interval> <prompt>` - Schedule a recurring prompt (e.g. `/loop 30s ping`).
  Intervals: `Ns/Nm/Nh/Nd`, minimum 30s, max 50 loops/session.
  - `/loop` (or `/loop list` / `/loop ls`) - List current scheduled loops.
  - `/loop cancel <id|all>` (aliases `rm`, `stop`, `delete`) - Cancel a loop.
  - Loops fire only when the agent is idle and the input bar is focused. At
    most one loop fires per poll. Overdue loops fire once on the next poll
    (no catch-up); `next_fire_at` advances to `now + interval`.
  - Loops are persisted in the session metadata (`loops` field of `meta.json`)
    and restored on `--resume`/`--continue`.
- `/proxy-setup` - Configure proxy and SSL certificate settings
- `/providers` - Open Provider Settings to add or manage model providers. Available only while no turn is active.
- `/exit` - Exit the application

## File Mentions (`@`)

Type `@` in the chat input to autocomplete files and folders from the
project tree. Pressing Tab/Enter inserts the chosen path. Your message text
is sent as-is (the `@path` stays in the prompt); behavior then depends on
the mention kind:

- **Text files** trigger a synthetic `read_file` tool call injected right
  after your message, so the file content arrives as a fresh tool result
  every turn (no caching/dedup). The same limits as the `read_file` tool
  apply (~2000 lines / 50 KB per call; larger files are truncated or
  reported as an error result). Re-mentioning a file always re-reads it.
- **Folders** are not read automatically — the path stays in your message
  text and the agent can `read_file`/`grep` it on demand.
- **Image files** (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) become image
  attachments — sent alongside the prompt as native multimodal content for
  vision-capable models.

Image attachments:

- Require `supports_images = true` on the active deployment in `models.toml`.
  Three shipped Codex deployments support images; non-vision deployments reject
  image messages with a clear error before adding them to the conversation.
- Snapshotted into `<session_dir>/attachments/<sha1>.<ext>` so that
  resumed sessions stay reproducible even if the source file is moved.
- Capped at 10 MiB per image and 8 images per message.
- Out-of-project paths work via `@/abs/path/to.png` (the picker only
  suggests project files, but the `@`-parser accepts absolute paths).
  Drag-and-drop from Finder into Terminal, iTerm2, or Ghostty is
  intercepted at paste time: if the pasted content is a single bare
  path to an image file (raw, `\\ `-escaped, or quoted), the input
  automatically prepends `@` (and quotes paths containing spaces).
  Non-image paths are pasted verbatim so non-image use cases are not
  affected.
- **Image copy/paste from the clipboard** (macOS and Linux):
  writes the image to the system temp dir as `clipboard-<ts>.png` or `.jpg`
  and inserts an `@<path>` token at the cursor. Two entry points:
  1. `Ctrl+V` keybinding inside the prompt.
  2. `/paste-image` slash command.

  Uses `osascript` with a TIFF→PNG fallback via `sips` on macOS.
  Linux reads PNG/JPEG via `wl-paste` on Wayland or `xclip` on X11;
  install the corresponding clipboard utility to enable image paste.
- Rendered in the chat bubble as one dim `attached image:` footer line
  per image, linking each attachment to its snapshot. Clicking opens the
  file with the OS default image viewer.

## Input Queue

Prompts submitted while the agent is running are accepted by the app server
and merged into a single follow-up turn, so everything queued during one
generation is delivered to the agent together (not one turn per prompt) once
it finishes. Each queued prompt keeps its own message so it stays individually
editable and removable. This includes plain prompts, `/skill ...`
prompts, and prompts with `@` mentions. `!bash` and non-side-channel slash
commands require an idle session and are rejected with a toast
while busy. **Ctrl+C** removes the newest queued prompt (LIFO); **Esc**
interrupts the active turn and pauses the remaining queue; pressing Enter
(empty or not) on a paused queue resumes it.

Allowlisted slash commands (`side_channel=True`) run immediately via a
side channel while the agent or bash is busy. Only one side-channel command
runs at a time. Commands that write config or session overrides (theme, model, thinking,
proxy) require idle, then write through the app server directly after
the user confirms the picker.

Commands not on the side-channel allowlist (e.g. `/clear`, `/compact`,
`/rewind`, `/resume`, `/reload`, `/retry`)
are rejected while busy and can be retried when
the session is idle.

While the queue is non-empty and the agent is busy, pressing **Up**
enters queue selection mode: the last queued item is highlighted and
the input is locked (no cursor, no typing). **Up/Down** navigate
between queued prompts, **Enter** loads the selected prompt into the input
for editing (press Enter again to update it in-place), **Backspace**
or **Delete** removes the selected item and moves selection to the
next, and **Esc** exits selection mode and restores the original
input text.

## Skills System

Skills are specialized instruction sets the model can load on demand.
Each skill is a directory containing a `SKILL.md` file with YAML frontmatter.

### Skill File Format

```markdown
---
name: my-skill
description: What this skill does and when to use it.
user-invocable: true
allowed-tools: bash read_file
---

# Skill Instructions

Detailed instructions for the model...
```

### Skill Search Order (first match wins)

1. Python built-in skills (reserved names; cannot be overridden)
2. `skill_paths` from config.toml
3. Skills shipped with Chartreux
4. `.chartreux/skills/` in trusted project directory
5. `.agents/skills/` in trusted project directory
6. `~/.chartreux/skills/` (user global)
7. `~/.agents/skills/` (user global, Agent Skills standard)

Configured, project, and user skills can shadow a shipped skill. Python built-in
skill names remain reserved.

### Built-in Skills

Two skills ship as Python built-ins (reserved names; they sit first in the
search order and cannot be overridden):

- `chartreux` — this self-awareness reference: the standalone knowledge of
  the running Chartreux version. It is model-only (`user-invocable: false`);
  load it with the `skill` tool when the user asks anything about Chartreux
  itself.
- `skill-creator` — the methodology for creating, updating, or deleting a
  Chartreux skill: SKILL.md frontmatter, discovery order, support files, and
  the permission flow. It is user-invocable (a user-invocable skill is
  invoked as `/skill-name`) and loadable with the `skill` tool.

When you create, update, or delete a skill, load `skill-creator` FIRST and
follow its methodology and permission flow before writing any SKILL.md. The
orchestrating executor follows the same rule: it loads `skill-creator` before
touching any skill file.

### Invoking Skills

Two entry points:
- The model loads a skill on demand via the `skill` tool.
- The user invokes a `user-invocable` skill by typing `/skill-name` (optionally
  followed by extra instructions). The user turn stays the literal `/skill-name`
  text; the skill is loaded programmatically and appears to the model as a
  synthetic `skill` tool call and result immediately after that turn — the model
  does not call the tool itself.

Skills with `user-invocable: false` are model-only: they are hidden from the
slash menu and `/skill-name` will not resolve them (it is treated as a plain
prompt). The model can still load them via the `skill` tool.

A `/` at the very start of the input opens the slash menu (commands and skills).
A `/word` typed mid-prompt (not the first word) instead shows an inline ghost-text
preview of the best-matching skill name; press `Tab` to accept it. Only skills are
offered inline, and no popup is shown.

## Environment Variables

- `CHARTREUX_HOME` - Override the Chartreux home directory (default: `~/.chartreux`)
- `MISTRAL_API_KEY` - API key for Mistral provider
- `CHARTREUX_ACTIVE_MODEL` - Override active model
- `CHARTREUX_*` - Override a config field using its schema name; nested fields use `__`, such as `CHARTREUX_SESSION_LOGGING__ENABLED`. Supplied unknown names or malformed values fail with an error identifying the `environment` source and field, without echoing the offending value. `CHARTREUX_ENABLED_AGENTS` is retired and rejected.
- Exact non-config controls remain honored by their consumers: `CHARTREUX_HOME`, `CHARTREUX_TYPING_GRACE_PERIOD_MS`, `CHARTREUX_ACP_LOGGING_ENABLED`, `CHARTREUX_TEST_DISABLE_KEYRING`, and `CHARTREUX_TEST_DISABLE_AUTO_TITLE`. Provider variables such as `MISTRAL_API_KEY` are separate. Legacy `VIBE_*` variables are not a fallback.
- `LOG_LEVEL` - Overrides `log_level` config for `$CHARTREUX_HOME/logs/chartreux.log`.
  One of `DEBUG`, `INFO`, `WARNING` (default), `ERROR`, `CRITICAL`. Invalid values
  fall back to `WARNING`. Use `/log-level` to change at runtime.
- `LOG_MAX_BYTES` - Max size in bytes of `chartreux.log` before rotation
  (default: `10485760`, i.e. 10 MiB).
- `DEBUG_MODE` - When `true`, forces `DEBUG`-level logging.
- `CHARTREUX_TYPING_GRACE_PERIOD_MS` - Milliseconds the agent waits for a typing
  pause before showing interactive user questions (default: `1000`). Set to `0` to disable. Negative or non-numeric
  values fall back to the default.

## API Keys (.env file)

The `.env` file in CHARTREUX_HOME stores API keys in dotenv format:

```
MISTRAL_API_KEY=your-key-here
```

This file is loaded on startup and its values are injected into the environment.

## Trusted Folders

Chartreux uses a trust system to prevent executing project-local config from untrusted
directories. The trust database is stored in `~/.chartreux/trusted_folders.toml`.
Project-local config (`.chartreux/` directory) is only loaded when the current
directory is explicitly trusted.

Interactive mode prompts to trust unknown folders. The prompt targets the
closest ancestor of the cwd (the cwd itself included) containing a `.git`
entry; the search excludes the user's home directory and the filesystem
root, and falls back to the cwd if no qualifying ancestor is found.
Programmatic mode (`-p`/`--prompt`) never prompts: the folder is untrusted.
Use `--trust` to trust cwd for the current invocation only (not persisted).
`--trust` and `--worktree` both skip the prompt: they grant the workspace trust
for the session, so there is no decision left to ask about. Without this a
`--worktree` run would prompt on every launch, since each worktree is a
directory the trust database has never seen.

## Sensitive Files — DO NOT READ OR EDIT

NEVER read, display, or edit any of these files:
- `~/.chartreux/.env` (or `$CHARTREUX_HOME/.env`) — contains API keys and secrets
- Any `.env`, `.env.*` file in the project or CHARTREUX_HOME

If the user asks to set or change an API key, instruct them to edit the `.env`
file themselves. Do not offer to read it, write it, or display its contents.
Do not use tools (`read_file`, `write_file`, bash cat/echo, etc.) to access these files.

## How to Modify Configuration

To help the user modify their Chartreux configuration:

1. **Read current config when authorized**: Read `~/.chartreux/config.toml` (or the path
   from `CHARTREUX_HOME` if set) only when filesystem authority permits it. A missing
   file means Chartreux is using built-in defaults.
2. **Create a backup when authorized**: Before editing an existing file, copy it to
   `config.toml.bak` in the same directory only when filesystem authority permits
   both operations. This applies to any existing config file you are about to modify
   (`config.toml`, `trusted_folders.toml`, `models.toml`, etc.)
3. **Edit the TOML file when authorized**: Make changes using the edit tool; do not
   claim that a permission prompt will be shown or that authorization will succeed.
4. **Reload**: The user can run `/reload` to apply changes without restarting

For API keys, tell the user to edit `~/.chartreux/.env` directly — never read or
write that file yourself.

For project-specific configuration, create/edit `.chartreux/config.toml` in the
project root (the folder must be trusted first)."""


SKILL = SkillInfo(
    name="chartreux",
    description="""Authoritative reference for Chartreux — the CLI agent you (the model) are running inside.

LOAD when the user:
- asks anything about Chartreux itself, even by indirect name ("this CLI", "this tool", "you");
- wants to change, inspect, or reset their setup;
- asks why the agent did or did not act;
- asks how to make the CLI do X, where X lives, or what a flag/command/setting does;
- asks any meta question about your own behavior;
- is unsure whether a command, flag, env var, or file is in scope — this skill is the source of truth.

SCOPE: config under `~/.chartreux/` and project-local `.chartreux/`; `CHARTREUX_*` and `LOG_*` env vars; models and providers; agents and subagents; skills; tools and their permission model; every slash command and CLI flag; hooks; MCP servers; trusted folders; `@`-file mentions; logs; themes.""",
    user_invocable=False,
    prompt=_PROMPT_TEMPLATE.replace("__CHARTREUX_VERSION__", __version__),
    source=SkillSource.BUILTIN,
)
