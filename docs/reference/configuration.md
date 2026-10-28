# Configuration reference

This page defines the accepted configuration surface. For locations, layering, and examples, see the [configuration guide](../guides/configuration.md); for models, see the [models guide](../guides/models.md); for the identity/workflow/strategy model, see the [identity architecture](identity-architecture.md).

## Files and precedence

`config.toml` selects runtime behavior. Open the user file in your editor with `/open-config-file`. The user file is `$CHARTREUX_HOME/config.toml` (`~/.chartreux/config.toml` by default). A trusted project may provide `.chartreux/config.toml`; discovery searches upward from the working directory. Effective precedence, low to high, is built-in defaults, user file, trusted project file, `CHARTREUX_*` environment overrides, and runtime selections (CLI > environment > config > shipped default for `[agent]` identity, workflow, and orchestration strategy).

`models.toml` is a separate user-only catalog overlay at `$CHARTREUX_HOME/models.toml`. Provider and deployment tables in `config.toml` are rejected. To move a legacy catalog, run `chartreux models migrate --apply`.

`graduation.toml` at `$CHARTREUX_HOME/graduation.toml` (`~/.chartreux/graduation.toml` by default) stores the graduation nudge's durable `second_model_saved`, `shown`, and `dismissed` flags. Need signals and attempt counters are session-local and are not stored there. This file does not select dispatch mode.

## `config.toml` keys

Unless noted, list defaults are `[]`, map defaults are `{}`, and booleans shown below are defaults. Pattern lists accept exact names, shell-style globs, or full-match regular expressions prefixed with `re:`.

### Model selection

| Key | Default | Accepted value |
| --- | --- | --- |
| `compaction_model` | `""` | Model expression; empty uses the current main model. The resolved model must share the main provider. |
| `allowed_models` | `[]` | Model-expression patterns. A non-empty list restricts selection. |
| `auto_compact_threshold` | `200000` | Integer fallback token threshold; `0` turns automatic compaction off when the deployment has no threshold of its own. A catalog deployment may override this fallback. |

### Agent selections

The `[agent]` table selects who performs the work and how it is orchestrated; the selections and their resource content freeze per conversation. See the [identity architecture](identity-architecture.md) for the model and the frozen lifecycle.

| Key | Default | Accepted value |
| --- | --- | --- |
| `[agent].identity` | `"coder"` | Built-in identity (`coder`, `orchestrator`, `researcher`, `reviewer`, `advisor`), a `builtin:`- or `user:`-qualified id, or a bare custom prompt id resolving to `<name>.md` in the user prompts directory. |
| `[agent].workflow` | `"minimal"` | Built-in workflow (`minimal`, `ceremony`), a `builtin:`- or `user:`-qualified id, or a bare custom body in `~/.chartreux/workflows/`. |
| `[agent].orchestration_strategy` | `"none"` | `none`, a built-in strategy (`builtin:default`), a `user:`-qualified id, or a bare custom body in `~/.chartreux/orchestration_strategies/`. |

New-conversation precedence is CLI flag (`--identity`, `--workflow`, `--orchestration-strategy`) > environment (`CHARTREUX_AGENT__IDENTITY`, `CHARTREUX_AGENT__WORKFLOW`, `CHARTREUX_AGENT__ORCHESTRATION_STRATEGY`) > `config.toml` > shipped default; an omitted flag is distinguished from an explicit selection at every level. Saved changes apply to new conversations only — `/reload` reports them as pending — and the Settings > Identity & Orchestration group edits the same keys with the same next-conversation timing. Unknown or malformed selections fail loudly at conversation start with actionable errors; reserved shipped names cannot be shadowed by custom files. `chartreux doctor` resolves the selections offline and reports retired settings with their successors.

### Tools and integrations

| Key | Default | Accepted value |
| --- | --- | --- |
| `tools` | tool defaults | Table keyed by tool name. See below. |
| `tool_paths` | `[]` | Paths to custom tool files or directories; directories are shallow-searched. |
| `enabled_tools` | `[]` | Tool-name patterns. A non-empty list is an allow-only filter. |
| `disabled_tools` | `[]` | Tool-name patterns removed after `enabled_tools` filtering. |
| `credential_env_passthrough` | `[]` | Environment-variable names exempt from credential scrubbing in child processes (shell commands, MCP stdio servers, hooks, client terminals). This setting is accepted only from the user configuration layer; project and other layers, including generic config patches, are rejected. |
| `mcp_servers` | `[]` | Array of [MCP server tables](#mcp-server-tables). |

Every ordinary `[tools.<name>]` table accepts `permission` (`always` or `never`), `allowlist`, `denylist`, and `sensitive_patterns`; defaults are `always`, `[]`, `[]`, and `[]`. The shell exceptions are `tools.bash`, which does not accept `allowlist`, and `bash_start`, which uses the canonical `tools.bash` configuration rather than a separate permission table. A tool implementation can accept additional fields. Shipped tool fields are:

| Tool table | Additional fields and defaults |
| --- | --- |
| `tools.read_file` | `max_read_bytes = 51200`; permission `always`. |
| `tools.write_file` | `max_write_bytes = 64000`, `create_parent_dirs = true`. |
| `tools.grep` | `max_output_bytes = 64000`, `default_max_matches = 100`, `context_lines = 2`, `default_timeout = 60`, `exclude_patterns` (the built-in exclusion list), `codeignore_file = ".chartreuxignore"`; permission `always`. |
| `tools.bash` | `max_output_bytes = 16000`, `default_timeout = 300`, `denylist`, `denylist_standalone`, and `sensitive_patterns`; it does not accept `allowlist`. |
| `tools.bash_read`, `tools.bash_stop`, `tools.bash_list` | Permission `always`; no additional documented fields. Job ownership checks also apply. |
| `tools.web_fetch` | `default_timeout = 30`, `max_timeout = 120`, `max_content_bytes = 120000`, `user_agent` (the built-in browser-like value). |
| `tools.web_search` | `provider = "auto"`, `api_key_env_var` and `base_url` unset, `timeout = 120` (> 0), `max_results = 5`, `model = "mistral-vibe-cli-with-tools"`. A blank or unset `base_url` uses the selected provider's default endpoint. Provider is `auto`, `mistral`, `exa`, `brave`, or `duckduckgo`; `auto` means Mistral only and never falls back. Configure it in Settings > Web search or with `/web-search`. `Configured; connection not verified` reflects configuration and credential availability, not a connectivity check. |
| `tools.task` | `allowlist = ["coder"]`, `denylist = []`; permission `always`. The allowlist auto-permits launches of matching identities; the denylist is an absolute veto applied after launch resolution, keyed on the resolved identity (see [Identity architecture](identity-architecture.md#the-identity-denylist)). |
| `tools.todo` | `max_todos = 100`; permission `always`. |
| `tools.read_image` | Permission `always`; no additional documented fields. |
| `tools.edit` | Permission `always`; no additional documented fields. |
| `tools.wait_for_agent`, `tools.ask_user_question`, `tools.get_agent_result`, `tools.skill`, `tools.check_agents`, `tools.cancel_agent`, `tools.release_agent` | Permission `always`; no additional documented fields. |

`bash_start` honors `[tools.bash]` launch permission and shell restrictions,
including parent restrictions; there is no separate `[tools.bash_start]`
permission key. `enabled_tools` and `disabled_tools` still filter the launch tool,
and canonical Bash denial cannot be bypassed by enabling it. Read, stop, and list
use their own ordinary permission tables; disabling launch alone does not disable
those recovery tools. Shell allowlists are not restored by the job API. Live jobs
or pending admissions can block execution-authority reductions, not cosmetic
status-line changes. See [managed shell jobs](../guides/tools-safety.md#managed-shell-jobs).

The shipped `[tools.bash]` `denylist` defaults no longer contain the recognized
dangerous git operations (`git push`, `git checkout`, `git restore`,
`git stash drop`, `git stash clear`, `git switch --discard-changes`,
`git switch -f`, `git reflog expire`, `git reflog delete`): those are governed by
the typed dangerous-command policy and its runtime approval flow instead. Explicit
`denylist` entries remain an absolute veto checked before any grant; setting
`denylist = []` no longer implicitly authorizes dangerous git commands — they need
a runtime grant. See [dangerous git commands](../guides/tools-safety.md#dangerous-git-commands).

`dangerous_command_grants_by_project` is a protected user-layer field (a map of
project key to granted operation IDs) that persists "Always for this project"
dangerous-command grants. Only the `git.push` operation is persistable, and only
the runtime approval flow or the Settings > Shell Permissions panel manages it:
programmatic or generic config writes to the field are rejected, and it is
accepted only from an installed user source. Revocation reports live and on-disk
outcomes separately through the app-server `policy/dangerous_commands/revoke`
method.

The built-in read/edit/write/image/grep configurations include sensitive patterns for `.env`-style files. Do not remove those protections casually.

### Agents and skills

| Key | Default | Accepted value |
| --- | --- | --- |
| `skill_paths` | `[]` | Directories containing skills. |
| `enabled_skills` | `[]` | Skill-name patterns; non-empty restricts active skills. |
| `disabled_skills` | `[]` | Skill-name patterns; ignored when `enabled_skills` is set. |
| `[subagents].idle_ttl_seconds` | `300` | Integer seconds, at least 0. Resident-runtime limit: seconds before a retained idle agent's runtime is retired; 0 disables TTL eviction. |
| `[subagents].max_idle_agents` | `4` | Integer, at least 0. Resident-runtime limit: maximum retained idle agents; 0 disables the count cap. |
| `[subagents].retain_agents` | `false` | Strict boolean. Global retention default: `true` keeps a completed background agent resident (idle) for reuse; `false` closes it when the run completes. A per-launch `retain` argument takes precedence over this default. |
| `[subagents].tombstone_ttl_seconds` | `900` | Integer seconds, at least 0. Recent-metadata bound: how long the terminal record of a closed or evicted agent stays browsable; 0 retains no terminal metadata. |
| `[subagents].max_tombstones` | `32` | Strict integer, at least 1. Recent-metadata bound: maximum Recent entries kept; the oldest expire first. |
| `[subagents].max_running_subagents` | `16` | Strict positive integer, at least 1. Active-work admission cap set in `config.toml` or via `CHARTREUX_SUBAGENTS__MAX_RUNNING_SUBAGENTS`; excluded from the settings-UI catalog. |
| `[subagents].run_max_price` | unset | Positive number (USD). Per-run budget default for subagent runs: run spend is measured relative to each run's admission baseline, and a run that exceeds the cap stops with the budget-exceeded outcome (the same run stop reason the session `max_price` cap produces); a response with unknown pricing during the run stops it with the budget-unverifiable outcome. Unset (the default) means no per-run budget — only the session limits apply. The per-session `run_max_price` option wins over this default for that session's subagents. |
| `[subagents].run_max_session_tokens` | unset | Strict integer, at least 1. Per-run budget default for subagent runs: total LLM tokens are measured relative to each run's admission baseline, and a run that exceeds the cap stops with the budget-exceeded outcome. Unset (the default) means no per-run budget. The per-session `run_max_session_tokens` option wins over this default for that session's subagents. |

Run-budget semantics: the per-session `run_max_price` and `run_max_session_tokens` options arm the root loop, not only subagent runs — a run-budget checkpoint joins the root pipeline after the absolute session backstops. Arming is not admission: an unanchored root writes its baseline at the checkpoint's first evaluation (the lazy self-anchor), so the root interval means "from the first evaluation onward", not from loop construction. A resumed root whose session state carries persisted anchors keeps them — the checkpoint's reset is a no-op — so the resumed interval's delta spans pre-resume consumption. Run budgets are checkpoints, not reservations: each evaluation compares anchor-relative consumption against the cap and stops the run only when consumption is already over it; nothing is reserved in advance.

The `[subagents]` lifecycle settings come in two classes. Resident-runtime limits — `retain_agents`, `idle_ttl_seconds`, and `max_idle_agents` — decide whether a completed background agent stays resident for reuse and for how long; by default agents are ephemeral and close on completion, keeping only the run result. Recent-metadata bounds — `tombstone_ttl_seconds` and `max_tombstones` — decide how long the terminal record of a closed or evicted agent stays in the agent browser's Recent group; they never bound stored results, which expire separately by count. The two zero meanings differ: `idle_ttl_seconds = 0` disables TTL eviction for retained agents, while `tombstone_ttl_seconds = 0` retains no terminal metadata at all, and `max_tombstones` must stay positive (there is no unlimited-metadata mode). Metadata expiry is not release: an expired record's stored result can remain retrievable, while release discards the record, its results, and browser access immediately. Retention is global by default plus the per-launch `retain` argument; there are no per-identity retention settings.

Agent selection is identity-based: the `[agent]` selections above choose the main-session identity, and the task tool launches child identities (see [Identity architecture](identity-architecture.md)). The retired agent-profile filters `agent_paths`, `enabled_agents`, and `disabled_agents` are no longer offered in the settings catalog, and `chartreux doctor` reports them with successor guidance — block launchable identities with `tools.task.denylist` (Settings > Identities) instead.

The active-work cap counts foreground and background runs, including reuse and pending child creation, but not idle retained agents. The root session's effective configuration controls admission; child configuration does not. Accepted configuration changes affect later admissions without stopping existing work. Completion, cancellation, release, or failed creation frees capacity. Busy `task(..., replace_run=True)` transfers the existing allocation rather than allocating another; the handoff holds capacity until replacement admission or unwind. `replace_run` is a call argument (default `false`), not configuration. Busy replacement requires `agent_id` and background mode and rejects launch `config` or identity changes before stopping; idle reuse still accepts its usual overrides. Configure `cancel_agent` through `[tools.cancel_agent]` and ordinary tool filters; it has no extra settings.

### Recovery and unattended questions

The `[recovery]` container holds one subsection per automatic-recovery
controller, and `[user_questions]` holds the unattended-expiry timer. All
three features are enabled by default since v0.5.6; see the
[configuration guide](../guides/configuration.md) for behavior and the
[subagents guide](../guides/subagents.md#failure-recovery-suspension-and-resume)
for the suspension/resume lifecycle. Unknown fields and invalid values are
rejected. Both subsections keep deep-merge semantics across sparse layers.

| Key | Default | Accepted value |
| --- | --- | --- |
| `[user_questions].unattended_timeout_enabled` | `true` | Strict boolean. Enables the server-owned unattended-expiry timer for eligible `ask_user_question` calls; only questions that declare an `unattended_default` can ever expire, and runtime approvals (dangerous-command, root access) are permanently ineligible. |
| `[user_questions].default_timeout_seconds` | `300.0` | Finite seconds, 60–3600. Default expiry timeout, snapshotted at callback open. |
| `[user_questions].max_timeout_seconds` | `3600.0` | Finite seconds, 60–3600; must be at least the default. Ceiling for per-request `timeout_seconds` overrides (rejected, not clamped). |
| `[recovery.subagents].enabled` | `true` | Strict boolean. The bounded automatic recovery controller for child subagents; when false it is inert (suspension and the explicit resume/reconcile ports are untouched). Requires `auto_retries_max >= 1` when enabled. |
| `[recovery.subagents].auto_retries_max` | `2` | Integer 0–10. Automatic resumes per recovery chain. |
| `[recovery.subagents].same_agent_resumes_max` | `2` | Integer, at least 0. Explicit plus automatic resumes per chain. |
| `[recovery.subagents].cooldowns_seconds` | `[30.0, 120.0]` | Finite, non-negative, non-decreasing durations; at least `auto_retries_max` entries. |
| `[recovery.subagents].jitter_fraction` | `0.2` | Number 0–1 of bounded jitter on the backoff. |
| `[recovery.subagents].episode_deadline_seconds` | `2400.0` | Positive finite seconds. Advisory aggregate bound anchored to the first failure: no new admission after it, never an in-flight kill. |
| `[recovery.subagents].endpoint_correlation_window_seconds` | `300.0` | Positive finite seconds. Window for endpoint-correlated rate-limit detection. |
| `[recovery.subagents].endpoint_correlation_min_events` | `2` | Integer, at least 2. Correlated events required within the window. |
| `[recovery.subagents].endpoint_correlation_min_models` | `2` | Integer, at least 2. Distinct base models required for endpoint correlation. |
| `[recovery.subagents].endpoint_cooldown_seconds` | `30.0` | Non-negative finite seconds. Cooldown armed on the endpoint when correlation fires. |
| `[recovery.subagents].eligible_classes` | all eight | Restricted to the transient allowlist `connection`, `timeout`, `server_error`, `overload`, `rate_limit`, `launch_failure`, `incomplete_stream`, `empty_response`. |
| `[recovery.subagents].suspension_ttl_seconds` | `1800.0` | Non-negative finite seconds. Warm-suspension TTL; 0 disables warm suspension. |
| `[recovery.subagents].max_suspended_agents` | `8` | Integer, at least 0. Suspended-record count cap; 0 disables suspension. |
| `[recovery.subagents].max_suspended_bytes` | `67108864` | Integer, at least 0. Total warm-conversation byte cap (oldest-first expiry); 0 disables byte accounting. |
| `[recovery.subagents].recovery_concurrency` | `2` | Integer, at least 1. Concurrent automatic recoveries. |
| `[recovery.subagents].fallback_require_endpoint_independence` | `true` | Strict boolean. Declared constraint for failure-time replacement candidates (enforced as a code invariant by the policy layer). |
| `[recovery.subagents].fallback_allowed_base_models` | `[]` | Canonical model names; empty adds no base-model restriction beyond the other constraints. |
| `[recovery.subagents].fallback_max_output_price_per_million` | unset | Positive finite price; bounds replacement candidates by output price. |
| `[recovery.subagents].fallback_replacements_max_per_package` | `2` | Integer, at least 0. Cross-model replacements per work package; the counter spans the package lineage. |
| `[recovery.root].enabled` | `true` | Strict boolean. The bounded automatic root-turn recovery controller; when false NO root automatic continuation runs at all (including the interrupted-stream path) and the manual retry is the only recovery. Requires `auto_retries_max >= 1` and a non-empty `eligible_classes` when enabled. |
| `[recovery.root].auto_retries_max` | `2` | Integer 0–10. ONE shared automatic-turn budget per user-turn episode, merging the interrupted-stream continuation. |
| `[recovery.root].deadline_seconds` | `2400.0` | Finite seconds, 1–86400. Advisory scheduling/admission bound anchored to the first failed logical call; never an in-flight kill. |
| `[recovery.root].cooldowns_seconds` | `[30.0, 120.0]` | Finite, non-negative, non-decreasing durations; at least `auto_retries_max` entries. |
| `[recovery.root].jitter_fraction` | `0.2` | Number 0–1 of bounded jitter on the backoff. |
| `[recovery.root].availability_recheck_seconds` | `1.0` | Positive finite seconds, at most `deadline_seconds`. Floor for read-only availability rechecks while waiting out a temporary exclusion. |
| `[recovery.root].eligible_classes` | all eight | Restricted to the transient allowlist (same eight values as `[recovery.subagents]`). |

### Interface, prompts, and project context

| Key | Default | Accepted value |
| --- | --- | --- |
| `theme` | `"auto"` | `auto`, `light`, or `dark`. |
| `disable_welcome_banner_animation` | `false` | Boolean. |
| `autocopy_to_clipboard` | `true` | Boolean. |
| `file_watcher_for_autocomplete` | `false` | Boolean. |
| `ask_confirmation_on_exit` | `true` | Boolean. Controls confirmation for idle Ctrl-C/Ctrl-D quits; `/exit` exits immediately while idle. Confirmation for consequential active work is always shown. |
| `displayed_workdir` | `""` | UI label for the working directory. |
| `context_warnings` | `false` | Boolean. |
| `show_thinking_nodes` | `false` | Boolean. |
| `show_message_timestamps` | `true` | Boolean. Show known message posting times, whole-turn totals, and settled tool durations (also in child transcripts). Capture and persistence continue when off; unstamped history stays unstamped. |
| `ascii_chrome` | `false` | Boolean. Use ASCII equivalents for application chrome glyphs. |
| `raise_on_compaction_failure` | `false` | Boolean. |
| `compaction_prompt_id` | `"compact"` | Compaction prompt ID; `compact` is the default built-in. |
| `include_commit_signature` | `true` | Boolean. |
| `include_model_info` | `true` | Boolean. |
| `include_project_context` | `true` | Boolean. |
| `include_prompt_detail` | `true` | Boolean. |
| `[project_context].default_commit_count` | `5` | Integer. |
| `[project_context].timeout_seconds` | `2.0` | Number of seconds. |

`system_prompt_id` is retired: it no longer selects the composed prompt — the `[agent]` identity/workflow/orchestration_strategy selections do — it is no longer offered in the settings catalog, and `chartreux doctor` reports it with successor guidance (a custom identity plus workflow/strategy composition replaces full-skeleton prompt replacement). The `include_prompt_detail` and related optional-detail settings still trim their composition sections as before.

### Status line

The `[status_line]` table controls the bottom session status row. Unknown fields and invalid values are rejected.

| Key | Default | Accepted value |
| --- | --- | --- |
| `segments` | `["directory", "pid", "context"]` | Ordered, unique list of `directory`, `pid`, `model`, `context`, `git-branch`, `spend-today`, `spend-week`, `spend-month`, or `background-jobs`. Both `directory` and `context` are required. |
| `directory_style` | `"name"` | `name` or `path`. |
| `context_style` | `"tokens-percent"` | `tokens` or `tokens-percent`. |
| `separator` | `"pipe"` | `space` or `pipe`. |

Context renders without a label, for example `135k/400k (34%)`; the denominator is the effective automatic-compaction threshold, not the model's maximum context window. The `tokens` variant omits the percentage. Git branch lookup is asynchronous and cached. Configure this row in Settings > Status line; see the [configuration guide](../guides/configuration.md#status-line-and-message-timing) for editor controls.

`background-jobs` is opt-in and renders `Jobs N`, including `Jobs 0` when no
managed jobs are active. It counts committed jobs whose owned cleanup has not
settled across the current root and all its children, including jobs surviving
child completion. It is not a count of background agent runs or retained finished
job records. The default segments remain `directory`, `pid`, and `context`.
The count comes from projected session state and refreshes between turns; it is
not an OS process scan. As an optional segment, it follows the same width
degradation as the other optional segments.

Spend segments show recorded USD cost estimates across all projects:
`spend-today` renders `Today $12.34`, `spend-week` renders `Week $12.34`,
and `spend-month` renders `Month $12.34`. Windows use the system-local
calendar day, Monday-start week, and calendar month. Calls are attributed at
completion time; stored timestamps are UTC. The `/usage` project filter does
not change this global scope.

- `—` means loading or unavailable (`-` in ASCII chrome), not zero spend.
- `$12.34+` is the known lower bound when some cost is unknown.
- `Unknown` means recorded calls have no priced usage.
- `$0.00` means a valid empty window or known zero-cost usage.

These are recorded usage only, priced from catalog rates captured at call time,
not provider invoices. Historical calls are not repriced after catalog changes.
See the [Usage browser](../guides/terminal.md#usage-browser) for model and token
details. Under width pressure, the row drops `pid` first, then optional segments
from the end, preserving directory and context.

### Sessions, logging, networking, and authority

| Key | Default | Accepted value |
| --- | --- | --- |
| `enable_notifications` | `true` | Boolean. |
| `enable_system_trust_store` | `false` | Boolean; use OS roots instead of bundled Certifi roots. See [networking](../integrations/networking.md). |
| `api_timeout` | `720.0` | Number of seconds. |
| `api_retry_max_elapsed_time` | `300.0` | Number of seconds. |
| `api_connect_timeout` | `10.0` | Number of seconds. |
| `api_write_timeout` | `30.0` | Number of seconds. |
| `api_pool_timeout` | `10.0` | Number of seconds. |
| `log_level` | unset | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`; unset leaves the logging default (`WARNING`). |
| `[session_logging].save_dir` | `$CHARTREUX_HOME/logs/session` | Directory path. |
| `[session_logging].session_prefix` | `"session"` | String. |
| `[session_logging].enabled` | `true` | Boolean. |
| `[session_logging].generate_titles` | `false` | Boolean. |
| `authorized_roots_by_project` | `{}` | User-layer-only map of absolute project paths to lists of absolute additional roots. It cannot be changed through general configuration writes. In the interactive TUI, choosing **Always for this project** saves an out-of-root grant here; saved grants apply at the next session start, and `/reload` applies them immediately. |

## MCP server tables

Each `[[mcp_servers]]` entry has `name`, optional `prompt`, `startup_timeout_sec = 10.0`, `tool_timeout_sec = 60.0`, `disabled = false`, and `disabled_tools = []`. `name` is normalized to letters, digits, `_`, and `-`.

For remote servers, set `transport = "streamable-http"`, `url`, and optional `[mcp_servers.auth]`:

| Auth key | Default | Accepted value |
| --- | --- | --- |
| `type` | `"static"` | `static` or `oauth`. |
| `headers` | `{}` | Static header map (`static` only). |
| `api_key_env` | `""` | Environment-variable name (`static` only). |
| `api_key_header` | `"Authorization"` | Valid header name (`static` only). |
| `api_key_format` | `"Bearer {token}"` | Format containing exactly `{token}` (`static` only). |
| `scopes` | required | List of OAuth scopes (`oauth` only; `[]` accepts the authorization-server default). |
| `client_id` | unset | Non-empty OAuth public client ID; mutually exclusive with `client_metadata_url`. |
| `client_metadata_url` | unset | OAuth client metadata URL; mutually exclusive with `client_id`. |
| `redirect_port` | `47823` | Integer from 1024 through 65535 (`oauth` only). |

For local servers, set `transport = "stdio"`, `command`, optional `args = []`, `env = {}`, and `cwd` (unset). `streamable-http` and `stdio` are the only transports.

## `models.toml` catalog overlay

The overlay patches the shipped catalog. Scalars replace shipped values, lists replace lists, deployments are matched by base model and provider, and roles are merged per role key. Provider IDs must not contain `/` or `@`; canonical model names, role names, role model values, and deployment identities must not contain `@`. `@main` is the sole saved main-assistant preset; the retired `@orchestrator` role fails loudly with rename guidance and no alias, and a `[roles.orchestrator]` overlay over the shipped base fails at load — rename the table to `[roles.main]`; the catalog file is never rewritten automatically. `/model` and `/thinking` affect the current session; `active_model` in `config.toml` is rejected with guidance to edit the main preset (`[roles.main]`).

Persisted `thinking_overrides` in user or project `config.toml` are also rejected.
Set the main assistant's or another role's `thinking` in `models.toml` (the main
assistant's preset is `[roles.main]`); use `/thinking` for a session choice or
`config.thinking` for one subagent launch.

| Table and field | Default / allowed value |
| --- | --- |
| `[providers."id"]` | Provider definition. `api_base` is a required HTTP(S) URL. |
| `api_key_env_var` | `""`; valid environment-variable name. |
| `api_style` | `"openai"`; `openai`, `openai-responses`, or `anthropic`. |
| `backend` | `"generic"`; the shipped Mistral provider uses `mistral`. |
| `reasoning_field_name` | `"reasoning_content"`. |
| `emits_finish_reason` | `true`. |
| `supports_tool_result_images` | `true`. |
| `extra_headers` | `{}` string-to-string map. |
| `disabled` | `false`; valid on providers, models, and deployments. |
| `[models."base"]` | Base-model definition; it must have at least one deployment. |
| `thinking` | `"medium"`; `off`, `low`, `medium`, `high`, or `max`. |
| `temperature` | Optional model-level key; when omitted, requests use the runtime default of `1.0`. |
| `[[models."base".deployments]]` | Deployment definition. `provider` and `name` are required. |
| `supports_images` | `false`. |
| `supported_thinking_levels` | Unset, or a list of known thinking levels. |
| `auto_compact_threshold` | Unset, or a positive whole integer token threshold. Fractional values and `0` are rejected; unset uses the global fallback, which allows `0` to disable automatic compaction. |
| `[models."base".deployments.prices]` | `input`, `output`, and `cached_input`: non-negative price per million tokens. Omit an unknown price; zero means explicitly free. |
| `[roles."name"]` | Default preset: `model` is one canonical base-model name; `thinking` is one of `off`, `low`, `medium`, `high`, or `max`; `description` is optional. Old `models` lists are rejected. |

The shipped role presets are `main` (the main assistant), `worker`, `scout`,
and `heavy`. Roles are model/thinking presets used as launch targets; they
carry no agent behavior — the identity does that (see
[Identity architecture](identity-architecture.md)).

<a id="dispatch-overlay"></a>

### Orchestration routes

Purpose routes — the executable launch bindings for `task` purposes and seats — live in the `[orchestration]` section of `config.toml` and deep-merge across sparse layers. Orchestration itself is selected through the `[agent]` identity/workflow/orchestration_strategy selections, not through the catalog; see [Identity architecture](identity-architecture.md#purpose-routes-seats-and-diversity) for resolution, seats, and diversity semantics.

| Table and field | Default / accepted value |
| --- | --- |
| `[orchestration.routes.<purpose>]` | Purpose-route table; sparse entries override the shipped route of the same purpose, and new keys introduce custom purposes. |
| `targets` | Required, non-empty list of route-target tables. |
| `targets[].model` | Required. A canonical model name, `$session` to bind the session model, or `$cheapest-capable` to default to the cheapest capable model in the catalog (resolved at launch from the catalog's price data; the shipped `implementation`, `mechanical-edit`, `search`, and `verification` routes use it, and a catalog with no priced model fails the launch with guidance to configure the route explicitly). |
| `targets[].identity` | Required. Built-in or custom identity id. |
| `targets[].tier` | Required. `mechanical` or `substantive` (the work-class label feeding the replacement policy). |
| `targets[].seat` | Optional seat name for multi-seat purposes. |
| `targets[].overlay` | Optional harness-owned instruction overlay id (shipped: `review-execution`); there is no user overlay directory. |
| `targets[].replacement_capability` | Optional, default `none`; `mechanical` or `substantive` — replacement eligibility, separate from the tier. |
| `requires_distinct_models` | Integer, at least 1, default 1; evaluated over canonical-model cardinality at the resolution of the seat composition. |

An unavailable model binding fails with an error telling you to configure it explicitly — no other canonical model is substituted. A legacy `[dispatch]` table in `models.toml` is superseded by this section: load reports it loudly with successor guidance naming the `[orchestration.routes.*]` tables above — it is never merged — the catalog write side no longer emits or manages dispatch, and a legacy table's values are preserved by catalog writes (comments and formatting are not preserved), never rewritten.

```toml
[orchestration.routes.research]
targets = [{ model = "$session", identity = "researcher", tier = "substantive" }]

[orchestration.routes."review.deep"]
requires_distinct_models = 3

[[orchestration.routes."review.deep".targets]]
model = "$session"
identity = "reviewer"
tier = "substantive"
seat = "analytical"
```

```toml
[providers."example-openai"]
api_base = "https://api.example.test/v1"
api_key_env_var = "EXAMPLE_API_KEY"
api_style = "openai"

[models."example-model"]
thinking = "medium"

[[models."example-model".deployments]]
provider = "example-openai"
name = "example-model-v1"
supports_images = false

[roles.custom-review]
description = "Default for independent review"
model = "example-model"
thinking = "high"
```

## Environment variables and `.env`

`$CHARTREUX_HOME/.env` is loaded at normal entrypoint startup and by `chartreux doctor --live` or `--smoke`, but never by bare `chartreux doctor`. A non-empty process environment value wins; an unset or empty process value may be filled by a non-empty `.env` value. Restart a long-running client after changing `.env`.

| Variable | Purpose |
| --- | --- |
| `CHARTREUX_HOME` | Changes the home directory; default `~/.chartreux`. Set it in the process environment before launch. |
| `CHARTREUX_<FIELD>` | Overrides a `config.toml` field. Use `__` for nesting, for example `CHARTREUX_PROJECT_CONTEXT__TIMEOUT_SECONDS`. Names are case-insensitive; empty values are still supplied and must validate. |
| `CHARTREUX_AGENT__IDENTITY`, `CHARTREUX_AGENT__WORKFLOW`, `CHARTREUX_AGENT__ORCHESTRATION_STRATEGY` | New-conversation `[agent]` selections, outranked only by the `--identity`, `--workflow`, and `--orchestration-strategy` CLI flags. An unset variable means no override; see [Agent selections](#agent-selections). |
| Provider key named by `api_key_env_var` | Credential for that catalog provider; the shipped Mistral provider uses `MISTRAL_API_KEY`. |
| `EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, or the variable selected by `tools.web_search.api_key_env_var` | Credentials for web-search providers. Exa and Brave use their corresponding variable by default. Mistral/auto uses `tools.web_search.api_key_env_var` when set; otherwise it selects the configured Mistral provider key, or `MISTRAL_API_KEY` if no Mistral provider is configured. Chartreux checks only the selected variable and reports a missing-key diagnostic if it is unavailable. |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`; outranks `log_level`. `DEBUG_MODE=true` forces debug logging. |
| `LOG_MAX_BYTES` | Maximum `chartreux.log` size before rotation; default `10485760`. |
| `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, `SSL_CERT_FILE`, `SSL_CERT_DIR` | Proxy and TLS settings; see [networking](../integrations/networking.md). |

Provider keys, search keys, proxy credentials, and private certificate paths are appropriate for `.env`; do not commit their values.
