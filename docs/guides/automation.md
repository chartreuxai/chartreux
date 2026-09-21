# Automation

Programmatic mode sends one prompt, writes its result, and exits. It is selected
by a positional prompt, `-p`/`--prompt`, or non-empty piped standard input.
Tool calls in this mode are auto-approved, but normal tool and path-safety
policy still applies.

```bash
chartreux --prompt "Summarize the changes in this repository"
printf '%s\n' 'Review the failing tests' | chartreux
chartreux "List the public API changes"
```

Use `--trust` when unattended work must load configuration from a project that
has not already been trusted.

## Limits and tool filters

Apply budgets to bound a run:

- `--max-turns N` limits assistant turns.
- `--max-price DOLLARS` stops the session when its cost exceeds the limit.
- `--max-tokens N` limits combined prompt and completion tokens.

Restrict the available tool set with repeatable `--enabled-tools TOOL` and
`--disabled-tools TOOL`. Patterns may be exact tool names, globs such as
`bash*`, or regular expressions prefixed with `re:`. `--enabled-tools` first
narrows the set; `--disabled-tools` then removes matches.

```bash
chartreux -p "Run the focused tests" \
  --max-turns 5 \
  --max-price 1.00 \
  --max-tokens 50000 \
  --enabled-tools 'bash*' \
  --output json
```

## Output

Set `--output` to choose the output contract:

- `text` (the default) prints the final assistant text.
- `json` writes the complete public history as one JSON array after the run.
- `streaming` writes each completed public history entry as newline-delimited
  JSON (NDJSON).

The JSON formats are history-entry formats, not message-only formats; entries
other than messages can be present where applicable. Programmatic runs deny
callback requests, including interactive questions, rather than displaying a
prompt. See the [command reference](../reference/commands.md) for the full CLI
surface.
