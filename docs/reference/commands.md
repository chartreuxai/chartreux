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
| `--setup` | Run interactive setup—theme, provider, credentials, and model selection—then exit. Requires an interactive terminal; otherwise, it prints actionable guidance instead of launching the TUI. |
| `--workdir DIR` | Change to this directory before launch. |
| `--worktree [NAME]` | Run in a managed Git worktree. With a name, create or reuse it; without one, create a name from the prompt or a random slug. Ignored with `--setup`. |
| `--add-dir DIR` | Repeatable additional workspace root; trusted for this session. |
| `--trust` | Trust the working directory for this invocation only. |
| `-c`, `--continue` | Continue the most recent saved session. Mutually exclusive with `--resume`. |
| `--resume [SESSION_ID]` | Resume a session; without an ID, open the picker. |

### Subcommands

| Command | Options |
| --- | --- |
| `chartreux models migrate [--preview \| --apply]` | Preview (default) or move legacy catalog tables to `models.toml`. |
| `chartreux mcp remove NAME` | Remove a user-configured MCP server. |
| `chartreux mcp add NAME` | `--transport {streamable-http,stdio}` (default `streamable-http`), `--url`, `--command`, repeatable `--arg VALUE`, repeatable `--env NAME=VALUE`, repeatable `--header NAME=VALUE`, `--api-key-env VAR` (also `--bearer-token-env-var`), `--api-key-header HEADER`, `--api-key-format FORMAT`, `--no-login`, `--startup-timeout-sec SECONDS`, `--tool-timeout-sec SECONDS`. Remote servers require `--url`; stdio servers require `--command`. |

`chartreux-acp` accepts no argument for stdio operation, plus `-h`/`--help`,
`-v`/`--version`, and `--setup`. `chartreux-app-server` is a stdio JSON-RPC
server and currently accepts only `-h`/`--help`.

## Built-in slash commands

These are the current terminal command registry. Custom user-invocable skills
can add commands.

| Command | Action |
| --- | --- |
| `/help` | Show command and shortcut help. |
| `/model` | Select the active model. |
| `/thinking` | Select the session thinking level. |
| `/reload` | Reload configuration, instructions, and skills from disk. |
| `/config` | Open the user config file in your editor; creates the file with a commented template if absent, and reloads after editing if it changed. |
| `/clear`, `/new` | Start a new conversation; optionally provide a seed prompt. |
| `/copy` | Copy the last agent message. |
| `/paste-image` | Paste a clipboard image into the prompt (available on supported systems). |
| `/log` | Show the current interaction-log path. |
| `/log-level` | Change the session log level or persist it. |
| `/debug` | Toggle the debug console. |
| `/agents` | Toggle the expanded retained-background-agent list above the input. |
| `/compact [instructions]` | Summarize the conversation context. |
| `/exit`, `exit`, `quit`, `:q`, `:quit` | Exit. |
| `/status` | Display agent statistics. |
| `/proxy-setup` | Configure proxy and certificate settings. See [networking](../integrations/networking.md). |
| `/providers` | Add or manage model providers. |
| `/resume`, `/continue` | Browse, resume, or delete saved sessions. |
| `/rename` | Rename the current session. |
| `/mcp` | Show MCP servers; supports `add`, `status`, `login`, and `logout` operations. |
| `/rewind` | Rewind to a prior message; press Escape twice as a shortcut when input is empty. |
| `/branch` | Fork the current conversation into a resumable session. |
| `/retry [instructions]` | Continue an interrupted model response. |
| `/loop <interval> <prompt>` | Schedule a recurring prompt; `/loop list` and `/loop cancel <id\|all>` manage it. |
| `/theme` | Select `auto`, `light`, or `dark`. |

## Key shortcuts

| Shortcut | Action |
| --- | --- |
| `Enter` | Submit input. |
| `Ctrl+J` or `Shift+Enter` | Insert a newline. |
| `Escape` | Interrupt an agent or close a dialog; `Esc Esc` opens rewind when input is empty. |
| `Ctrl+C` | Interrupt or quit; clears non-empty input before quitting. |
| `Ctrl+D` | Delete right, or quit according to `ask_confirmation_on_exit`. |
| `Ctrl+Z` | Suspend with a message. |
| `Ctrl+G` | Open the current plan/input in an external editor. |
| `Ctrl+O` | Toggle tool output. |
| `Ctrl+Y` or `Ctrl+Shift+C` | Copy the current selection. |
| `Shift+Up` / `Shift+Down` | Scroll chat. |
| `Ctrl+\\` | Toggle the debug console. |
| `Ctrl+Shift+A` | Toggle the expanded background-agent list above the input. |
| `Alt+Left` / `Alt+Right` | Move by word in input. |
| `Ctrl+V` | Paste a clipboard image where that platform feature is available. |

Prefix a line with `!` to run a user-authored shell command directly, or type
`@` followed by a path for path completion.
