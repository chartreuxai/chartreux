# Terminal

## Interactive mode

Run `chartreux` from a project directory to open the interactive interface. Type a request and press Enter. Use `@` to complete a file path and `!` to run a shell command directly.

- `Ctrl+J` or `Shift+Enter` inserts a newline.
- `Ctrl+G` opens the current input in an external editor.
- `Ctrl+O` toggles tool output.
- `Ctrl+\` toggles the debug console.

For commands, session controls, and programmatic invocation, see the [command reference](../reference/commands.md).

## Input and queueing

A prompt submitted while a turn is running joins the queued follow-up input. Queued prompts are combined into the next follow-up turn rather than executed as separate FIFO turns.

With an empty input, use `Ctrl+C` to remove the newest queued prompt. With nonempty input, `Ctrl+C` clears the input instead. `Escape` interrupts the active turn and pauses the remaining queue; press Enter on an empty input to resume it. To steer the active turn immediately, press `Ctrl+Enter` or `Super+Enter` with an empty input; this sends queued prompts into that turn unless queue selection is active.

When queued prompts exist, press Up from input history to enter queue selection. Up and Down move between prompts; Enter opens the selected prompt for editing; Backspace or Delete removes it; and Escape leaves selection. While editing, Enter saves the change and Escape discards it.

## Copying and selection

Select text with the mouse, then press `Ctrl+Y` or `Ctrl+Shift+C` to copy the selection. A successful copy, including any clipboard fallback, is reported in the interface. Double-click selects a word and triple-click selects a paragraph; drag to extend a selection at that granularity.

## Terminal requirements

The interactive interface runs on Linux. macOS has not been validated, and Windows is unsupported. A terminal emulator is required; Chartreux does not maintain a named-emulator compatibility list.

## Themes

The available themes are `auto` (the default), `light`, and `dark`. `auto` first attempts to infer the terminal background, then uses the operating-system preference, and falls back to dark. Choose a theme with `/theme` or configure it explicitly:

```toml
# config.toml
theme = "dark"
```

## Notifications

When `enable_notifications` is enabled (the default) and the interface does not have focus, Chartreux signals attention-required and completion events with a terminal bell and a temporary terminal-title change. Disable them in [configuration](configuration.md):

```toml
enable_notifications = false
```
