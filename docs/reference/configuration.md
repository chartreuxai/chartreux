# Configuration reference

This page defines the accepted configuration surface. For locations, layering, and examples, see the [configuration guide](../guides/configuration.md); for models, see the [models guide](../guides/models.md).

## Files and precedence

`config.toml` selects runtime behavior. The user file is `$CHARTREUX_HOME/config.toml` (`~/.chartreux/config.toml` by default). A trusted project may provide `.chartreux/config.toml`; discovery searches upward from the working directory. Effective precedence, low to high, is built-in defaults, user file, trusted project file, `CHARTREUX_*` environment overrides, agent profile, and runtime override.

`models.toml` is a separate user-only catalog overlay at `$CHARTREUX_HOME/models.toml`. Provider and deployment tables in `config.toml` are rejected. To move a legacy catalog, run `chartreux models migrate --apply`.

## `config.toml` keys

Unless noted, list defaults are `[]`, map defaults are `{}`, and booleans shown below are defaults. Pattern lists accept exact names, shell-style globs, or full-match regular expressions prefixed with `re:`.

### Model selection

| Key | Default | Accepted value |
| --- | --- | --- |
| `active_model` | `""` | Canonical base-model name or `@role`; empty selects `@orchestrator`. |
| `compaction_model` | `""` | Model expression; empty uses the active model. The resolved model must share the active provider. |
| `allowed_models` | `[]` | Model-expression patterns. A non-empty list restricts selection. |
| `thinking_overrides` | `{}` | Table mapping a canonical base-model name to `off`, `low`, `medium`, `high`, or `max`. |
| `auto_compact_threshold` | `200000` | Positive fallback token threshold; a catalog deployment may set its own. |

### Tools and integrations

| Key | Default | Accepted value |
| --- | --- | --- |
| `tools` | tool defaults | Table keyed by tool name. See below. |
| `tool_paths` | `[]` | Paths to custom tool files or directories; directories are shallow-searched. |
| `enabled_tools` | `[]` | Tool-name patterns. A non-empty list is an allow-only filter. |
| `disabled_tools` | `[]` | Tool-name patterns removed after `enabled_tools` filtering. |
| `credential_env_passthrough` | `[]` | Environment-variable names exempt from credential scrubbing in child processes (shell commands, MCP stdio servers, hooks, client terminals). This setting is accepted only from the user configuration layer; project and other layers, including generic config patches, are rejected. |
| `mcp_servers` | `[]` | Array of [MCP server tables](#mcp-server-tables). |

Every `[tools.<name>]` table except `tools.bash` accepts `permission` (`always`, `ask`, or `never`), `allowlist`, `denylist`, and `sensitive_patterns`; defaults are `ask`, `[]`, `[]`, and `[]`. A tool implementation can accept additional fields. Shipped tool fields are:

| Tool table | Additional fields and defaults |
| --- | --- |
| `tools.read_file` | `max_read_bytes = 51200`; permission `always`. |
| `tools.write_file` | `max_write_bytes = 64000`, `create_parent_dirs = true`. |
| `tools.grep` | `max_output_bytes = 64000`, `default_max_matches = 100`, `default_timeout = 60`, `exclude_patterns` (the built-in exclusion list), `codeignore_file = ".chartreuxignore"`; permission `always`. |
| `tools.bash` | `max_output_bytes = 16000`, `default_timeout = 300`, `denylist`, `denylist_standalone`, and `sensitive_patterns`; it does not accept `allowlist`. |
| `tools.web_fetch` | `default_timeout = 30`, `max_timeout = 120`, `max_content_bytes = 120000`, `user_agent` (the built-in browser-like value). |
| `tools.web_search` | `provider = "auto"`, `api_key_env_var` and `base_url` unset, `timeout = 120` (> 0), `max_results = 5`, `model = "mistral-vibe-cli-with-tools"`. Provider is `auto`, `mistral`, `exa`, `brave`, or `duckduckgo`. |
| `tools.task` | `allowlist = ["worker"]`; permission `ask`. |
| `tools.todo` | `max_todos = 100`; permission `always`. |
| `tools.read_image` | Permission `always`; no additional documented fields. |
| `tools.edit` | Permission `ask`; no additional documented fields. |
| `tools.wait_for_agent`, `tools.ask_user_question`, `tools.get_agent_result`, `tools.skill`, `tools.check_agents`, `tools.release_agent` | Permission `always`; no additional documented fields. |

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

### Interface, prompts, and project context

| Key | Default | Accepted value |
| --- | --- | --- |
| `theme` | `"auto"` | `auto`, `light`, or `dark`. |
| `disable_welcome_banner_animation` | `false` | Boolean. |
| `show_greeting` | `true` | Boolean. |
| `autocopy_to_clipboard` | `true` | Boolean. |
| `file_watcher_for_autocomplete` | `false` | Boolean. |
| `ask_confirmation_on_exit` | `true` | Boolean. |
| `displayed_workdir` | `""` | UI label for the working directory. |
| `context_warnings` | `false` | Boolean. |
| `show_thinking_nodes` | `false` | Boolean. |
| `raise_on_compaction_failure` | `false` | Boolean. |
| `system_prompt_id` | `"cli"` | Prompt ID. Built-ins are `cli`, `explore`, `tests`, `minimal`, `worker`, `advisor`, and `reviewer`; custom IDs resolve from prompt directories. |
| `compaction_prompt_id` | `"compact"` | Compaction prompt ID; `compact` is the default built-in. |
| `include_commit_signature` | `true` | Boolean. |
| `include_model_info` | `true` | Boolean. |
| `include_project_context` | `true` | Boolean. |
| `include_prompt_detail` | `true` | Boolean. |
| `[project_context].default_commit_count` | `5` | Integer. |
| `[project_context].timeout_seconds` | `2.0` | Number of seconds. |

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

The overlay patches the shipped catalog. Scalars replace shipped values, lists replace lists, deployments are matched by base model and provider, and roles are merged per role key. Provider IDs contain `/`; canonical model names, role names, role members, and deployment identities must not contain `@`.

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
| `temperature` | Unset number. |
| `[[models."base".deployments]]` | Deployment definition. `provider` and `name` are required. |
| `supports_images` | `false`. |
| `supported_thinking_levels` | Unset, or a list of known thinking levels. |
| `auto_compact_threshold` | Unset positive number. |
| `[models."base".deployments.prices]` | `input`, `output`, and `cached_input`: non-negative price per million tokens. Omit an unknown price; zero means explicitly free. |
| `[roles."name"]` | Role definition: `description` is an optional string and `models` is a non-empty, unique ordered list of canonical base-model names. |

```toml
[providers."example/openai"]
api_base = "https://api.example.test/v1"
api_key_env_var = "EXAMPLE_API_KEY"
api_style = "openai"

[models."example-model"]
thinking = "medium"

[[models."example-model".deployments]]
provider = "example/openai"
name = "example-model-v1"
supports_images = false

[roles.reviewers]
description = "Models used for independent review"
models = ["glm-5-3", "example-model"]
```

## Environment variables and `.env`

`$CHARTREUX_HOME/.env` is loaded at entrypoint startup. A non-empty process environment value wins; an unset or empty process value may be filled by a non-empty `.env` value. Restart a long-running client after changing `.env`.

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
