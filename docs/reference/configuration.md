# Configuration reference

This page defines the accepted configuration surface. For locations, layering, and examples, see the [configuration guide](../guides/configuration.md); for models, see the [models guide](../guides/models.md).

## Files and precedence

`config.toml` selects runtime behavior. Open the user file in your editor with `/open-config-file`. The user file is `$CHARTREUX_HOME/config.toml` (`~/.chartreux/config.toml` by default). A trusted project may provide `.chartreux/config.toml`; discovery searches upward from the working directory. Effective precedence, low to high, is built-in defaults, user file, trusted project file, `CHARTREUX_*` environment overrides, agent profile, and runtime override.

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
| `tools.grep` | `max_output_bytes = 64000`, `default_max_matches = 100`, `default_timeout = 60`, `exclude_patterns` (the built-in exclusion list), `codeignore_file = ".chartreuxignore"`; permission `always`. |
| `tools.bash` | `max_output_bytes = 16000`, `default_timeout = 300`, `denylist`, `denylist_standalone`, and `sensitive_patterns`; it does not accept `allowlist`. |
| `tools.bash_read`, `tools.bash_stop`, `tools.bash_list` | Permission `always`; no additional documented fields. Job ownership checks also apply. |
| `tools.web_fetch` | `default_timeout = 30`, `max_timeout = 120`, `max_content_bytes = 120000`, `user_agent` (the built-in browser-like value). |
| `tools.web_search` | `provider = "auto"`, `api_key_env_var` and `base_url` unset, `timeout = 120` (> 0), `max_results = 5`, `model = "mistral-vibe-cli-with-tools"`. A blank or unset `base_url` uses the selected provider's default endpoint. Provider is `auto`, `mistral`, `exa`, `brave`, or `duckduckgo`; `auto` means Mistral only and never falls back. Configure it in Settings > Web search or with `/web-search`. `Configured; connection not verified` reflects configuration and credential availability, not a connectivity check. |
| `tools.task` | `allowlist = ["worker"]`; permission `always`. |
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

The built-in read/edit/write/image/grep configurations include sensitive patterns for `.env`-style files. Do not remove those protections casually.

### Agents and skills

| Key | Default | Accepted value |
| --- | --- | --- |
| `agent_paths` | `[]` | Directories containing agent profiles. |
| `enabled_agents` | `[]` | Agent-name patterns; non-empty restricts available profiles. |
| `disabled_agents` | `[]` | Agent-name patterns; ignored when `enabled_agents` is set. |
| `skill_paths` | `[]` | Directories containing skills. |
| `enabled_skills` | `[]` | Skill-name patterns; non-empty restricts active skills. |
| `disabled_skills` | `[]` | Skill-name patterns; ignored when `enabled_skills` is set. |
| `[subagents].idle_ttl_seconds` | `3600` | Integer seconds, at least 0. |
| `[subagents].max_idle_agents` | `16` | Integer, at least 0. |
| `[subagents].max_running_subagents` | `16` | Strict positive integer, at least 1. Active-work admission cap set in `config.toml` or via `CHARTREUX_SUBAGENTS__MAX_RUNNING_SUBAGENTS`; excluded from the settings-UI catalog. |

The active-work cap counts foreground and background runs, including reuse and pending child creation, but not idle retained agents. The root session's effective configuration controls admission; child configuration does not. Accepted configuration changes affect later admissions without stopping existing work. Completion, cancellation, release, or failed creation frees capacity. Busy `task(..., replace_run=True)` transfers the existing allocation rather than allocating another; the handoff holds capacity until replacement admission or unwind. `replace_run` is a call argument (default `false`), not configuration. Busy replacement requires `agent_id` and background mode and rejects launch `config` or profile changes before stopping; idle reuse still accepts its usual overrides. Configure `cancel_agent` through `[tools.cancel_agent]` and ordinary tool filters; it has no extra settings.

### Interface, prompts, and project context

| Key | Default | Accepted value |
| --- | --- | --- |
| `theme` | `"auto"` | `auto`, `light`, or `dark`. |
| `disable_welcome_banner_animation` | `false` | Boolean. |
| `show_greeting` | `true` | Boolean. |
| `autocopy_to_clipboard` | `true` | Boolean. |
| `file_watcher_for_autocomplete` | `false` | Boolean. |
| `ask_confirmation_on_exit` | `true` | Boolean. Controls confirmation for idle Ctrl-C/Ctrl-D quits; `/exit` exits immediately while idle. Confirmation for consequential active work is always shown. |
| `displayed_workdir` | `""` | UI label for the working directory. |
| `context_warnings` | `false` | Boolean. |
| `show_thinking_nodes` | `false` | Boolean. |
| `show_message_timestamps` | `true` | Boolean. Show known message posting times, whole-turn totals, and settled tool durations (also in child transcripts). Capture and persistence continue when off; unstamped history stays unstamped. |
| `ascii_chrome` | `false` | Boolean. Use ASCII equivalents for application chrome glyphs. |
| `raise_on_compaction_failure` | `false` | Boolean. |
| `system_prompt_id` | `"cli"` | Prompt ID. Built-ins are `cli`, `explore`, `tests`, `minimal`, `worker`, `advisor`, and `reviewer`; custom IDs resolve from prompt directories. |
| `compaction_prompt_id` | `"compact"` | Compaction prompt ID; `compact` is the default built-in. |
| `include_commit_signature` | `true` | Boolean. |
| `include_model_info` | `true` | Boolean. |
| `include_project_context` | `true` | Boolean. |
| `include_prompt_detail` | `true` | Boolean. |
| `[project_context].default_commit_count` | `5` | Integer. |
| `[project_context].timeout_seconds` | `2.0` | Number of seconds. |

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
| `authorized_roots_by_project` | `{}` | User-layer-only map of absolute project paths to lists of absolute additional roots. It cannot be changed through general configuration writes. |

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

The overlay patches the shipped catalog. Scalars replace shipped values, lists replace lists, deployments are matched by base model and provider, and roles are merged per role key. Provider IDs must not contain `/` or `@`; canonical model names, role names, role model values, and deployment identities must not contain `@`. `@orchestrator` is the sole saved main-assistant default. `/model` and `/thinking` affect the current session; `active_model` in `config.toml` is rejected with guidance to edit the orchestrator preset.

Persisted `thinking_overrides` in user or project `config.toml` are also rejected.
Set the orchestrator or other role's `thinking` in `models.toml`; use
`/thinking` for a session choice or `config.thinking` for one subagent launch.

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

The shipped role presets are `orchestrator` (the main assistant), `large`,
`medium`, and `small`. Worker and Reviewer profiles use `medium`; Advisor uses
`large`. The preset editor labels `orchestrator` as **Main**.

### Dispatch overlay

Dispatch policy is accepted only in the user-owned `models.toml` catalog
overlay. Project configuration and project profiles cannot select dispatch
policy. The `[dispatch]` table accepts `mode = "orchestrated"` or
`mode = "standalone"`; absent selection defaults to `standalone`. Explicit
selections are authoritative, regardless of legacy tier roles or roster size.
Standalone permits approved direct implementation but keeps delegated verification,
independent review, and all approval gates. Select `mode = "orchestrated"` to
delegate implementation too. A dismissible multi-model graduation nudge opens the
presets editor without changing mode automatically. Slots are keyed tables and may
sparsely patch a shipped slot. Omitted fields inherit; lists replace, rather
than append. New slots require their complete definition.

| Table and field | Accepted value |
| --- | --- |
| `[dispatch]` | Dispatch overlay table. |
| `mode` | `orchestrated` or `standalone`; no free-form values. |
| `[dispatch.slots.<name>]` | Slot binding table. |
| `profile` | Discovered agent profile name. |
| `role` | `@`-prefixed role from the merged model catalog. |
| `purposes` | List of purpose identifiers; replaces the inherited list. |
| `implements` | `never`, `routine`, or `escalation`. |
| `review_eligible` | Boolean review eligibility. |

Invalid dispatch overlays are rejected as a whole; startup uses the shipped
standalone policy and emits a visible diagnostic while retaining valid catalog
entries. The invalid source is not overwritten. Dispatch binds to a session;
saved changes apply only to the next session. See the [models guide](../guides/models.md)
for examples and [Subagents](../guides/subagents.md) for routing behavior.

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
| Provider key named by `api_key_env_var` | Credential for that catalog provider; the shipped Mistral provider uses `MISTRAL_API_KEY`. |
| `EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, or the variable selected by `tools.web_search.api_key_env_var` | Credentials for web-search providers. Exa and Brave use their corresponding variable by default. Mistral/auto uses `tools.web_search.api_key_env_var` when set; otherwise it selects the configured Mistral provider key, or `MISTRAL_API_KEY` if no Mistral provider is configured. Chartreux checks only the selected variable and reports a missing-key diagnostic if it is unavailable. |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`; outranks `log_level`. `DEBUG_MODE=true` forces debug logging. |
| `LOG_MAX_BYTES` | Maximum `chartreux.log` size before rotation; default `10485760`. |
| `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, `SSL_CERT_FILE`, `SSL_CERT_DIR` | Proxy and TLS settings; see [networking](../integrations/networking.md). |

Provider keys, search keys, proxy credentials, and private certificate paths are appropriate for `.env`; do not commit their values.
