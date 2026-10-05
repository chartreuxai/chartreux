# Terminal

## Interactive mode

Run `chartreux` from a project directory to open the interactive interface. Type a request and press Enter. Use `@` to complete a file path and `!` to run a shell command directly.

- `Ctrl+J` or `Shift+Enter` inserts a newline.
- `Ctrl+G` opens the current input in an external editor.
- `Ctrl+O` toggles tool output.
- `Ctrl+\` toggles the debug console.

For commands, session controls, and programmatic invocation, see the [command reference](../reference/commands.md).

## Settings and provider setup

Use `/settings` to browse curated settings, `/providers` to open Provider
Settings, and `/web-search` to configure the search provider and credentials.
The first-run `chartreux --setup` flow connects a provider, configures its
models, offers **Add another provider** or **Continue to presets**, and checks
web-search readiness after presets. A ready automatic Mistral configuration or
ready explicit search provider is preserved without opening the Web search
step. If the step opens, it offers Exa, Brave, and DuckDuckGo; setup does not
offer `auto` or a second Mistral choice. **Back to presets** returns to preset
choices, **Skip for now** leaves current search settings unchanged, and
**Finish setup** completes when the configuration and required key are ready.
With edited settings, **Save and finish** saves and applies them before
completing setup. In standalone Settings, Mistral appears once; `auto` is still
accepted as a Mistral config alias. Follow each screen's labeled Save and
forward actions; you do not need to backtrack to advance.

On Provider Settings browser screens, Tab/Shift+Tab moves between visible focus
groups; arrows move within the current group and stop at its boundaries. Enter
edits or activates the selected control. Separate **Details** click targets open
inspection without activating the row's primary action. Text fields retain their
normal editing keys. Escape backs out one local level and restores its opener.
In the Web Search editor, arrows move within the current provider choice or text
field, Space selects a provider, Enter accepts the current draft, and Tab moves
focus between controls. Escape offers a Keep editing/Discard edits choice when
a draft needs one. Read each screen's help because some settings save immediately
while provider and web-search forms use explicit Save actions.

## Usage browser

Use `/usage` to open a read-only browser of recorded calls and USD cost estimates.
It defaults to **Day** and **All projects**. Choose **Week** or **Month**, or
**Current project**, to fetch another snapshot. Calendar windows use the system's
local timezone, with weeks starting Monday. Current project is unavailable when
there is **No attached project**; `F1` explains this restriction.

Model rows are grouped by model, provider, and wire name. The final **TOTAL** row
shows the selected window's authoritative requests, tokens, and known cost.
Large numbers use downward-truncated K/M/B suffixes; `+` marks a known lower
bound and `Unknown` means no cost could be priced. Costs are catalog-priced
estimates, not provider invoices; see [usage storage](sessions-workspaces.md#usage-storage)
and [cost notation](../reference/configuration.md#status-line).

Press `d` or Enter on a row to open **Details** in a wrapping, scrollable pane
below the table. Model Details shows the complete deployment identity and exact
values; TOTAL Details includes the component breakdown. Details contains only
accounting content, including completeness explanations. The table heading shows
USD; the window and scope selectors identify the displayed period and scope.
Details is available even while loading, after an error, or in an empty window.
There is no resting preview.

Tab/Shift+Tab cycles focus groups: period → scope → table → Details (when open).
Left/Right moves within selector groups; Up/Down or `j`/`k` moves through table
rows or scrolls Details. Arrows stop at group boundaries. Enter/Space activates
selectors; moving focus alone does not fetch a snapshot. In Details, Shift+Tab
returns to the table without hiding the pane; selecting another row updates its
content.

Live updates refresh automatically, preserving the selected row and table scroll
position. While Details is open, the displayed snapshot is frozen; closing the
pane silently catches up with pending updates. Use `r` or `Ctrl+R` to force a
refresh, including while Details is open. Use `d` to show or hide Details and
`F1` for local help. Escape hides Details or help, then closes the browser. The
final shortcut row also provides clickable Refresh, Details, Help, and Close
actions, with Back while Details is open. There is no Apply or Cancel action.

A bounded status line appears only for loading, failed reads, or unavailable
usage. A failed read retains the previous snapshot; `r` retries. An empty ready
window shows zero totals without a status message. The browser becomes full-screen
below 84 columns or 28 rows. Opening it while a turn runs does not make a model
call, interrupt work, or change queued input.

## Input and queueing

A prompt submitted while a turn is running normally joins the queued follow-up
input. Queued prompts are combined into the next follow-up turn rather than
executed as separate FIFO turns.

When the main agent is **only waiting on subagents**, submitting a message instead
steers the current turn immediately. Chartreux cancels the outstanding waits,
not the subagents, keeps the same turn, and sends the message to the model without
waiting for those runs to finish. The server checks that the turn is still
waiting-only; if it has moved on, confirmed rejection falls back to queueing.
Delivery receipts prevent duplicate injection on retries. If recovery cannot
prove whether a message was delivered, the UI reports an uncertain outcome
rather than silently queueing or resending a possible duplicate.

With an empty input, use `Ctrl+C` to remove the newest queued prompt. With nonempty input, `Ctrl+C` clears the input instead. `Escape` interrupts the active turn and pauses the remaining queue; press Enter on an empty input to resume it. To steer the active turn immediately, press `Ctrl+Enter` or `Super+Enter` with an empty input; this sends queued prompts into that turn unless queue selection is active.

When queued prompts exist, press Up from input history to enter queue selection. Up and Down move between prompts; Enter opens the selected prompt for editing; Backspace or Delete removes it; and Escape leaves selection. While editing, Enter saves the change and Escape discards it.

## Exit confirmation

When an exit confirmation appears, **Cancel** has default focus. Enter therefore
keeps Chartreux open until you explicitly select **Exit** with Tab or the arrow
keys, or click it. Escape cancels the confirmation without interrupting work.

## Debug console

Open the console with `/debug` or `Ctrl+\`. While it has focus, use Up/Down or
`j`/`k` to select a logical log row and `c` to copy it. The footer also has
clickable **Copy selected** and **Close** actions. Escape closes the focused
console; it does not also interrupt the turn.

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

Set `ascii_chrome = true` in `config.toml` to use ASCII equivalents for
application chrome glyphs such as arrows, status marks, and spinners. It is
`false` by default and does not change user-provided text.

## Notifications

When `enable_notifications` is enabled (the default) and the interface does not have focus, Chartreux signals attention-required and completion events with a terminal bell and a temporary terminal-title change. Disable them in [configuration](configuration.md):

```toml
enable_notifications = false
```
