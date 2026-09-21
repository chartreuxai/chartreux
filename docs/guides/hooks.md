# Hooks

Hooks run commands at agent and tool lifecycle boundaries. They can gate a tool call, audit results, add context, or request a retry. Subagents inherit the parent hook configuration.

## Configuration

Put project hooks in `.chartreux/hooks.toml` in a trusted project, or user-wide hooks in `~/.chartreux/hooks.toml`. Project hooks load before user hooks; a project hook wins when names collide. A hook has a `name`, `type`, and `command`. Tool hooks may also have `match` and `strict`; those fields are invalid for `post_agent`. The default timeout is 60 seconds.

```toml
[[hooks]]
name = "guard-shell"
type = "pre_tool"
match = "bash"
command = "uv run python /opt/chartreux-hooks/guard_shell.py"
timeout = 15
strict = true
description = "Reject unsafe shell commands."
```

Commands are parsed with POSIX `shlex` and are executed without a shell. To use shell syntax such as a pipe or redirection, explicitly invoke `sh -c '...'`.

## Common contract

Chartreux writes one JSON invocation to hook stdin. Every event includes `session_id`, `parent_session_id`, `transcript_path`, `cwd`, and `hook_event_name`. Hooks return an exit status and stdout; use stderr for diagnostics.

- Exit 0 with empty stdout passes through.
- Exit 0 with a JSON object returns a structured response. It may contain `decision` (`allow` or `deny`), `reason`, `system_message`, and `hook_specific_output`.
- Malformed nonempty stdout, a nonzero exit, a timeout, or a launch failure is a hook failure. It warns and passes through by default; `strict = true` changes failure behavior for tool hooks.

Unknown response fields are ignored. Matching hooks run serially. A terminal `pre_tool` decision stops the remaining pre-tool chain.

## `post_agent`

`post_agent` runs after an assistant turn that ends without pending tool calls. It receives only the common invocation fields. A deny response with a reason injects that reason as a retry request; each hook is limited to three retries per user turn, after which further denials are warnings. `system_message` is UI-only.

## `pre_tool`

`pre_tool` runs for each tool call before runtime policy resolution. In addition to the common fields it receives `tool_name`, `tool_call_id`, and the raw `tool_input` object.

A deny response prevents the call and gives its reason to the model as the tool error. `hook_specific_output.tool_input` replaces the complete argument object, which is then schema-validated and used by subsequent hooks, runtime policy, the tool, and later model turns. Rewrites compose from left to right.

## `post_tool`

`post_tool` runs only when the tool body actually ran. It does not run after a pre-tool or policy denial, a `never` permission, or cancellation before tool execution. Its invocation adds `tool_name`, `tool_call_id`, post-rewrite `tool_input`, `tool_status` (`success`, `failure`, or `cancelled`), structured `tool_output`, mutable `tool_output_text`, `tool_error`, and `duration_ms`.

A deny response replaces the text shown to the model with its reason. `hook_specific_output.additional_context` appends text to that output; when both occur in one response, replacement happens before the append. Post-tool hooks run even when cancellation occurs during the tool body so they can audit the result.

## Example guard

This `pre_tool` hook can reject a shell command by writing a deny response:

```python
import json
import sys

call = json.load(sys.stdin)
command = call["tool_input"].get("command", "")
if "rm -rf" in command:
    print(json.dumps({"decision": "deny", "reason": "Destructive removal is blocked."}))
```

The script exits successfully and emits nothing for commands it allows. Keep hook scripts small, set a bounded timeout, and make strict hooks reliable enough that a failure should block or clear output.
