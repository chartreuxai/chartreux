# Chartreux Design Language

Internal contributor document. This is the normative visual and interaction
contract for every chartreux TUI surface. It is not part of the published
documentation.

Audience: anyone styling or building chartreux UI. Follow this document
instead of making per-screen aesthetic decisions. When this document and an
existing screen disagree, this document wins; when two rules seem to conflict,
prefer the one that preserves keyboard visibility and stable geometry.

Applies to: `chartreux/cli/textual_ui/`, `chartreux/ui/`, `chartreux/setup/`.

## 1. Design values

1. **Content first.** Conversation, code, names, and values get the space.
   Decorative borders and headings must earn their place.
2. **Quiet by default.** Ordinary content is neutral. Accent means interaction;
   severity colors mean outcomes. Nothing else gets color.
3. **Keyboard position is always visible.** Users must be able to distinguish
   keyboard focus, the current row, checked membership, and the active
   configuration at a glance.
4. **One meaning per treatment.** A green checkbox never ambiguously means
   selected, enabled, healthy, or saved. Each visual treatment has exactly one
   meaning.
5. **Stable geometry.** Protect chrome, keyboard focus, and reading anchors from
   incidental shifts within a mode. Streaming may grow the transcript body;
   explicit mode changes may relayout.
6. **Local actions, local feedback.** Show available actions and their results
   beside the relevant task, not in competing status areas.
7. **Terminal-native resilience.** Design in cells, tolerate narrow screens,
   and retain meaning without color or special glyphs.

## 2. Color and tokens

There is no separate chartreux palette. Map design roles onto Textual theme
variables and use those directly in TCSS:

| Role | Textual source | Required use |
|---|---|---|
| `canvas` | `$background` | Application root, conversation |
| `surface` | `$surface` | Modal browsers, sheets, dialogs, docked panels, popups |
| `inset` | `$panel` | Code and tool-output blocks only |
| `text` | `$foreground` | Content, values, titles |
| `muted` | `$text-muted` | Descriptions, timestamps, secondary labels, help |
| `disabled` | `$text-disabled` | Unavailable controls; never essential explanation |
| `border` | `$foreground-muted` | Resting frames and separators |
| `interactive` | `$primary` | Focus borders, cursor, shortcut keys |
| `success` | `$success` | Confirmed successful outcomes |
| `warning` | `$warning` | Recoverable risk, attention required |
| `error` | `$error` | Failed operations, invalid values |
| `brand` | copper `#B87333` | Wordmark/startup identity only |

Full-screen flows and browsers use `canvas` (`$background`), not `surface`.

Rules:

- `$primary` is the only interaction accent. Do not consume `$accent` or
  `$secondary` in application chrome.
- No per-provider, per-agent, or per-role colors. No component-local hex
  values outside the `brand` wordmark.
- Copper is branding only. It must never color focused borders, agent
  activity, message roles, or operational headings. In mono or limited-color
  modes the wordmark renders in ordinary foreground.
- The focused row is bold reverse video across its full available width,
  using foreground over the containing surface background, including cursor
  and value markers; inline semantic colors within the focused row are
  suppressed. Textual's full-row block-cursor highlight satisfies this recipe.
  On blur, remove only the current row's reverse-video and focus-induced bold
  treatment; restore ordinary row styling and render its cursor muted.
  Membership markers, radio selection, saved `Default`, session choices, and semantic content remain
  visible and unchanged. Hover must not imitate keyboard focus. Never combine
  reverse video with arbitrary colored row backgrounds.
- Python/Rich rendering (diff rows, message assembly, status lines) resolves
  the same roles from the active theme. Do not maintain an independent
  named-color palette in Python.
- Role mappings do not excuse unreadable output. In bundled light and dark
  themes and in the screenshot/export palette, text-sized interactive,
  semantic, and muted content must have at least 4.5:1 contrast against its
  rendered background; focus borders and other non-text state indicators
  must have at least 3:1 contrast against adjacent colors. Shortcut keys must
  be at least as legible as their muted descriptions. If a theme's `$primary`
  fails this requirement, use a contrast-safe theme-derived primary variant
  for text, or `$foreground` when no such variant exists; retain bold weight
  and the explicit key label. Do not introduce component-local hex fallbacks.
  In ANSI or user-defined palettes, where exact contrast cannot be guaranteed,
  meaning must also survive through wording, weight, cursor/glyph shape, and
  borders.

### Agent activity

Role and execution state are separate dimensions. Never tint whole messages
by role.

| Role/state | Treatment |
|---|---|
| User | Bold `text` label: `You` |
| Assistant | Bold `text` label: `Assistant` or the agent name |
| Tool | `muted` label: `Tool: <name>`; output stays `text` |
| Running/streaming | `interactive` marker plus explicit activity wording (e.g. `Running`/`Streaming`) |
| Completed | `success` marker plus explicit outcome |
| Failed | `error` marker plus `Failed: <reason>` |
| Awaiting approval | `warning` marker plus `Approval required` |

Note: interactive tool/command approvals are not part of the current supported workflow (ADR 0004); these rules apply only if an approval surface is introduced as a separate feature.

`Running` and `Streaming` are examples, not required literal labels. Existing
semantic states such as `Generating`, `Thinking`, `Retrying`, `Interrupting`,
and `Initializing` satisfy the explicit-activity rule.

Code syntax and diff colors are content-specific exceptions; diffs always
retain `+`/`-` prefixes so meaning survives without color.

## 3. Surfaces

Every screen is one of six surface types. All borders are single-line `solid`:
no round, heavy, double, or mixed decorative frames. Padding is `0 1`
(vertical/horizontal) unless a rule below says otherwise.

| Surface | Frame/background | Title | Geometry |
|---|---|---|---|
| Full-screen flow/browser | No outer border; `canvas` | First row, left-aligned bold `text` | Fill viewport; content takes remaining height |
| Modal browser/dialog | `border` on `surface` | Left-aligned bold `text` border-title | Centered; browser width `min(92, viewport width - 2)`, height `viewport height - 2`; confirmation max-width 60 |
| Bottom sheet/picker | Top `border` only; `surface` | First interior row, bold `text` | Full available width; content-sized up to half viewport height, then scroll |
| Docked panel | One `border` on the edge adjoining main content; `surface` | First interior row, bold `text` | Right dock: 40 columns, only at viewport width >= 120; otherwise open as a full-screen browser |
| Inline conversation element | No outer frame; `canvas`; code/output may use `inset` | One left-aligned role/tool label | Full conversation width; one blank row between messages |
| Anchored popup | `border` on `surface` | None | Width fits longest item, capped at 48 columns and available viewport; at most 8 result rows |

Additional rules:

- Exactly one title per surface. Never duplicate a border-title with a
  standalone heading.
- Outer modal borders do not turn `interactive` merely because the modal is
  open. Focus belongs to the active control, not the container.
- Surface background differences are supplementary; borders and layout
  convey grouping.
- No nested decorative panels. Input frames and code blocks are allowed
  inside a surface.
- Below 84 columns or 28 rows, modal browsers become full-screen browsers:
  full viewport, no outer border, `canvas` background, title moved to the
  first content row. Live resizing preserves draft, filter, selected identity,
  and focused control. Small confirmations remain dialogs if they fit.
- A bottom sheet floating over live conversation content may keep a full
  four-side frame (a top-only border leaves the bottom edge ambiguous against
  scrolling text), but it must use `surface` background — never transparent.
- Reserve the title/context row and one shortcut row outside the scrolling
  body. At 80x24 the primary browser list shows at least five item rows when
  five items exist; section labels and state rows do not count toward the
  five. Selected-item help is bounded; overflow is exposed through
  keyboard-accessible details. Validation and confirmation text that cannot
  fit scrolls in the task body. Every bounded area containing validation or
  selected-item details must expose all overflow through keyboard focus and
  scrolling, or a dedicated keyboard-accessible detail view. Opening generic
  shortcut help does not satisfy access to hidden task-specific content.
- Content may scroll; titles, current validation feedback, and essential
  actions must remain reachable.

### Surface assignment

- Provider Settings, agent transcript pane replacement, and onboarding
  screens: full-screen flow/browser. Do not redesign Provider Settings as
  a dashboard of cards. See the [Provider Settings flow spec](PROVIDER_SETTINGS_SPEC.md)
  for its target interaction contract.
- After presets, first-run onboarding preserves ready automatic Mistral search
  and ready explicit search choices without opening the Web search editor. If
  the editor is needed, offer Exa, Brave, and DuckDuckGo only, with explicit
  **Back to presets** and **Skip for now** actions. Skipping preserves current
  search settings and does not disable the tool. Standalone Settings shows
  Mistral once; `auto` remains a Mistral configuration alias.
- Settings: modal browser.
- Trust folder dialog: modal dialog (confirmation, max-width 60).
- Model, thinking, log-level, theme, session, question, MCP, MCP-OAuth,
  proxy, and rewind pickers: bottom sheets.
- Debug console: docked panel.
- Completion popup: anchored popup.

## 4. Spacing and type

### Cell scale

Only `0`, `1`, and `2` are valid gaps and paddings.

- `0`: consecutive list rows; title followed by its metadata.
- `1`: horizontal content gutter; blank row between logical groups.
- `2`: horizontal separation between columns or shortcut pairs.
- No blank rows between individual fields unless validation or help requires
  one. Borders count toward the space budget.

### Type ramp

There is one font size. Hierarchy comes from weight, color, and case:

| Role | Treatment |
|---|---|
| Surface title | Bold `text`, sentence case |
| Section label | Regular `muted`, UPPERCASE, one row |
| Item/value/body | Regular `text`, sentence case |
| Secondary detail/help | Regular `muted`, sentence case |
| Shortcut key | Bold `interactive`; its action description `muted` |
| Disabled control | `disabled`, with an explicit availability explanation where needed |

In an unfiltered primary list with at least 100 selectable rows (excluding
headings and state rows), section landmarks may use the `── LABEL ──` form in
bold text. Once chosen for an open surface, filtering does not change the
section-label treatment. Shorter lists use regular muted UPPERCASE labels.

Application-authored copy names the affected object and describes observed state
in plain language. Distinguish confirmed causes from possible causes; offer only
supported recovery actions. Preserve literal identifiers, commands, paths, and
user or model content rather than rewriting them to match UI casing.

No italics, no underlines, no pervasive bold as an extra hierarchy level, and
never apply `dim` to already-muted theme colors. Exception: distance-based
carousel pickers may fade non-adjacent rows with a text-opacity ramp
(50/25/10%) to encode distance from the selection.

Scalar inputs and the default composer occupy three rows including their border.
The explicit multiline composer exception is in section 5. At 80x24, use no
side-by-side forms or decorative blank rows. Prefer fewer simultaneous regions
over smaller or cryptic labels.

## 5. Information density

The resting screen must answer: what am I looking at, what is happening, and is a decision required? Detailed inspection must be available without making every item detailed by default. "At rest" means not explicitly expanded or opened for inspection. "Current row" means keyboard position, not membership, persisted selection, or the active configuration. "Rows" in layout budgets means rendered terminal rows, including wrapping, borders, and padding.

### Row content

- Lists use one row per item.
- Every browser row must show its identity and the value or state needed to choose its next action.
- Settings rows must show the setting name and current value, membership, or collection count. Provider and model rows must show the destination identity and applicable saved default, session override, membership, or unavailable state. Agent rows must show agent identity, profile, and execution state.
- The current row must retain the same height and content columns as other rows. Moving the cursor updates a separate selected-item detail area; it must not expand the row.
- At-rest rows must not include explanatory prose, full connection addresses, timestamps, run identifiers, or diagnostic counters unless that information is necessary to distinguish otherwise identical items.
- Required state markers must survive truncation. Shorten descriptive text before removing `Default`, `This session`, unavailable, failed, or waiting-on-user state.
- Selected-item detail must present, in order: complete identity and value; unresolved error, risk, or availability explanation; action consequence or useful description; diagnostic metadata.
- Default selected-item detail is at most four rendered rows. A surface may use fewer rows. Its allocated height must remain stable as the current row changes.
- Detail overflow must remain available through keyboard-accessible scrolling or a detail mechanism. Opening shortcut help is not a substitute for inspecting item data.

### Tools and progressive disclosure

- Ordinary tool calls and results must rest as one summary row per call or result unit, with zero raw output lines shown by default. Implementations must not independently invent automatic multi-line previews.
- A summary must identify the operation, its principal target or command, and its current state or outcome. A completed result may replace the pending-call summary instead of repeating it.
- Arguments, raw output, stack traces, and verbose diagnostics belong behind explicit disclosure. Expanding content must reveal the retained result without silently imposing a second, undisclosed truncation.
- The summary must communicate failure or a warning requiring action even while details are folded. A long explanation may remain in details; the fact that attention is required must not.
- Edit diffs and written-file content are review content, not routine log output. Existing inline expanded presentations may remain. They scroll within the transcript and must not become fixed-height chrome or force a second preview system.
- Approval content is a decision surface and is exempt from ordinary tool folding. The user must be able to inspect the affected scope and proposed change before accepting.
- Disclosure choices must survive updates to the same entry and temporary navigation. New output must not reopen a deliberately closed item or close an item being inspected.
- Where output has been truncated at capture or storage time, the UI must say so. Disclosure must not imply that unavailable content can be recovered.

### Streaming conversation

- Assistant output must grow within its existing message. Stream fragments must not create additional role headings, timestamps, status rows, or blank separators.
- User-facing assistant prose remains expanded. Exposed reasoning and routine tool detail remain collapsed by default; a working-state label must communicate activity without requiring their expansion.
- The one-blank-row message separation rule applies to logical messages, not to stream fragments or individual members of a compact tool group.
- Streaming may grow the transcript body. It must not resize unrelated chrome, change keyboard focus, or move the reading anchor when the user is inspecting older content.
- Live following applies only while the user is following the newest content. Explicit scrolling away suspends following; an explicit return-to-latest action resumes it.
- Updates must not reset a selected history page or replace the identity of an inspected agent.

### Status and activity ownership

- The session status line is one row. It carries workspace identity and context usage when available. Context usage must distinguish used capacity from the limit; unavailable usage must not appear as zero.
- Process identity is optional diagnostic metadata. Under width pressure, remove it before shortening workspace identity or context usage.
- Activity feedback separately carries the current operation or phase, elapsed time when available, and currently applicable interrupt or steering actions.
- Background-agent overview occupies at most one row at rest and shows aggregate counts by execution state. Per-agent model, turns, run identity, lifetime, and full output belong in agent inspection.
- Cost is available through session statistics. A persistent cost field is optional, must identify its scope, and must not displace workspace, context, waiting state, or essential actions. Missing pricing must not be represented as a known zero cost.
- Do not repeat the same detailed outcome in activity feedback, session status, and a toast. A compact indication that attention is required may point to the one authoritative decision or error surface.

### Region budgets (rows, at 80x24)

- Ordinary chat, no secondary surface: at least 14 rows for the scrollable transcript; all non-transcript regions together at most 10 rows.
- Chat chrome: composer normally three rows; activity and its hints at most two rows including spacing; collapsed agent overview at most one; session status at most one. Remaining chrome must fit the total budget.
- Explicit multiline composing: composer may grow to six rows including its frame, then scroll internally; transcript must retain at least 11 rows.
- Browser: preserve title/context and the final shortcut row; selected-item detail at most four rows by default; at least five actual items visible when five exist. Allocate remaining height to the body, not decorative spacing.
- Bottom sheet or expanded agent overview: at most 12 rows for a bottom sheet; expanded agent overview at most 10. These are explicit inspection modes and may temporarily reduce the ordinary-chat transcript budget.
- Approval, field editor, or full-screen inspection: the decision or inspection body takes remaining height and scrolls; essential actions remain reachable. The ordinary-chat transcript minimum does not apply while chat is replaced or covered.
- Budgets apply when enough content exists to fill the region; they measure available viewport space, not a minimum number of populated messages.
- Ordinary browser navigation must not grow detail beyond its budget. Explicit opening of details is a mode change and may devote the body to that detail.
- At widths of 120 columns or more, the same vertical budgets apply at the same height. Extra width reveals more identifiers and values before additional metadata.
- The optional right dock remains 40 columns wide, its adjoining border counting within that allocation. It must not open automatically because the terminal became wider.
- Taller viewports give additional rows to transcript or browser content before enlarging help and status regions.
- Budgets do not require every optional region to be present or empty rows to be reserved for hidden regions.

### Adding and hiding metadata

Add metadata at rest only when it changes the user's immediate decision, distinguishes identities, or prevents a misleading interpretation of state. Otherwise put it in selected-item details or explicit inspection. Under space pressure, remove diagnostic metadata first, then optional descriptions; truncate identity only after preserving its distinguishing portion and required state. Never hide the only indication of pending approval, failure, unavailable configuration, or unapplied edits.

## 6. Components

### Lists: position is not value

| List kind | Marker | Key |
|---|---|---|
| Navigation/action list | `▸ ` current row | Enter activates |
| Independent membership | `[■]` / `[ ]` | Space toggles |
| Exclusive draft value | `(*)` / `( )` | Space selects; Enter accepts the field |
| Immediate destination picker | cursor only | Enter chooses |

- `▸` is the only current-row marker; `›` is banned everywhere, and rows
  never carry universal bullet prefixes.
- Use a two-column cursor gutter. Checkboxes and radios place their value
  marker after that gutter.
- When a list has focus, its current row follows the full-width focused-row
  recipe in section 2. When focus leaves, only the muted cursor remains.
- Checked/selected values never change because focus moved.
- In an exclusive draft-value list, Space changes the selected radio value.
  Moving the cursor does not change that value. Enter accepts the selected draft
  value, even when the cursor is on another row.
- A stored default, current-session override, draft value, and cursor position
  are separate states. Label a saved preset as `Default` and a temporary model
  choice as `This session`; never present either merely because the cursor
  moved. An unavailable saved choice must show its exact repair reason.
- An empty focused list renders one non-disabled state row that can receive
  the cursor (for example `No matching providers`); disabled placeholder
  options cannot hold focus and do not satisfy this rule.
- Any list that displays a stored default marks it independently of cursor and
  selection. A current-session override has its own label.
- Classify list rows by commit semantics: independent draft booleans use
  checkboxes and Space; an exclusive draft value uses radios and Space, with
  Enter accepting that field; navigation and immediate destinations use
  cursor-only rows and Enter. Browser rows may open editors but must not
  imply membership merely by highlighting.

### Inputs

| State | Treatment |
|---|---|
| Normal | `solid border` |
| Focused | `solid interactive` |
| Invalid | `solid error` plus `Error: <reason>` directly below |
| Invalid and focused | Error border wins; caret and focused label establish focus |

- Border width is constant across all states.
- Labels stay visible; placeholders are examples, not labels.
- Secret fields stay masked; validation and status text never echo secrets.
- Inline editing must not overlay neighboring rows. Scalar edits reserve an
  editor slot or replace the browser body with a field editor; the field label
  stays visible and validation space is reserved within the editor. Stable
  geometry forbids incidental shifts within a mode; explicit mode changes may
  relayout.

### Actions

- Browsers use action-list rows. Forms and confirmations use compact
  `[Apply] [Cancel]` controls. No filled "primary CTA" rectangles.
- Focused actions use the same bold reverse-video treatment as list rows.
- Destructive action labels use `error` color and explain the effect in text.
  Destructive operations are discarding a dirty multi-field draft, replacing
  a saved credential or customized connection, and deleting a stored entity.
  They require a confirmation with default focus on Cancel that names every
  affected scope and states what cancellation preserves. Inline `y/N`
  help-text confirmations are banned. Risk-acceptance decisions (for example
  trusting a folder) are a separate category: they may default to the
  affirmative action but must state the consequence of accepting.
- Use specific verbs: `Apply changes`, `Remove provider`, `Discard edits` —
  never `OK` or `Proceed`.

### Help and footer

One footer at the bottom of each surface:

1. Optional status or selected-item explanation above.
2. Shortcut row last, e.g. `↑↓ Move  Enter Edit  Space Toggle  Esc Back`.

- Keys use the shared helper's bold `$primary` treatment
  (`chartreux/ui/shortcut_hints.py`); descriptions stay `muted`.
- Measure available cells before rendering shortcuts. Keep the primary
  action and Escape on the final row; if secondary actions do not fit, show
  `F1 Help` and implement a keyboard-accessible help view. Never clip the
  only way out. Truncate list summaries to one row; expose complete
  identifiers in a reachable, wrapping detail view.
- Only advertise implemented, currently available actions.
- Chat must expose a keyboard-accessible Help entry without requiring a
  memorized command. Help must explain how to discover slash commands and reach
  settings, provider configuration, session browsing, and agent inspection
  where supported. Every secondary surface must expose how to reach its
  contextual shortcuts; unavailable actions must be identified as unavailable,
  not advertised as executable.

### Feedback and glyph inventory

| Meaning | Glyph | ASCII | Color | Wording |
|---|---|---|---|---|
| Information | `i` | `i` | `muted` | `Info: No configured providers` |
| Running | `…` | `...` | `interactive` | `Running: Discovering models` |
| Success | `✓` | `+` | `success` | `Saved: Provider updated` |
| Warning | `!` | `!` | `warning` | `Warning: Unsaved changes` |
| Error | `✗` | `x` | `error` | `Failed: Connection refused` |

Chrome glyphs:

| Purpose | Preferred | ASCII |
|---|---|---|
| Current row | `▸` | `>` |
| Checkbox | `[■]` / `[ ]` | `[x]` / `[ ]` |
| Radio | `(*)` / `( )` | same |
| Disclosure | `+` / `-` | same |
| Direction hints | `↑↓`, `←→` | `Up/Down`, `Left/Right` |
| Truncation | `…` | `...` |

- No emoji, no colored-circle status vocabulary, no play triangles, no extra
  warning symbols. Punctuation and framework border characters are not
  iconography.
- Errors and warnings persist until resolved or dismissed. Success persists
  until the next relevant action or navigation; no timer infrastructure.
- Dismissing feedback hides its explanation; it does not resolve the underlying
  state. While an item remains failed or unavailable, or a decision or unapplied
  edit remains pending, retain a compact indication and a route to its details.
  Resolved conditions need not remain in persistent chrome.
- Streaming may use a static marker; animation is optional and never required
  to understand progress.

## 7. Interaction contract

- **Up/Down** move among visible rows, fields, and actions without changing
  stored values. Within an open text editor, arrow keys retain normal caret
  behavior.
- **Left/Right** are text editing or explicitly documented hierarchical
  navigation; never hidden screen switching.
- **Tab/Shift-Tab** optionally move between controls in visual reading order.
  Every reachable control also has an Up/Down route; Tab is never required to
  cross from a list to actions or from a form field to the next control.
- **Enter** activates the selected action or destination, opens the selected
  field for editing, or accepts the current field and returns to screen
  navigation. In an exclusive draft-value list, it accepts the selected radio
  value, not the cursor row. Accepting a field in a draft editor does not
  persist the whole draft.
- **Space** toggles a checkbox or selects a radio; otherwise it is ordinary
  text input.
- **Escape** handles pending confirmation, open help view, field edit, nested
  picker/editor, nonempty filter of the current browser, browser back/close —
  dismissing only one level per press.
- Typing in a searchable list filters it, and the filter state is visible.
  Escape clears a nonempty filter before leaving the browser.
- Closing a surface restores focus and scroll position to its opener.
- Mouse activation follows the same state and commit rules as keyboard
  activation.
- Hover must not change keyboard position, draft values, or active
  configuration. Pointer controls must have an identifiable visible target.
  Wheel scrolling affects the visible scrollable region under the pointer and
  must not scroll obscured content; pointer scrolling away from latest suspends
  following under the same rule as keyboard scrolling.
- Every editor states whether acceptance changes a draft or persists
  immediately; a collection editor accepts item edits into its draft and
  persists only through an explicit `Apply changes` action. Saved credentials
  are never described as discardable draft changes.
- A guided setup screen identifies a forward save action and its destination.
  A completed provider or model step persists before moving forward; an
  incomplete preset never blocks structural provider/model saves. Finish
  validates runnable required presets and focuses a direct repair action.
  Backtracking is available to change earlier choices, never required to save.
- Distinguish loading, empty catalog, no filter matches, and failed loading
  with explicit state rows; state rows never activate. Busy feedback names the
  operation and updates advertised actions immediately; during an atomic save
  Escape does not dismiss the surface and the footer says so.
- An empty primary body must explain whether content does not yet exist, was
  filtered out, is loading, or could not be obtained. Where recovery or
  creation is supported, expose a specific next action in the same surface;
  otherwise state that no action is required. Informational state rows remain
  non-activating.
- Each operation has exactly one authoritative local feedback location; do
  not duplicate it in a toast. A compact pending-action indication pointing
  to the one authoritative decision surface is permitted and is not duplicate
  feedback. Navigation may hide item-specific feedback but must not erase
  unresolved errors or warnings; show them again on return. A later success
  for the same operation resolves its prior error.
- An operation failure must identify what failed, what remains usable, whether
  relevant edits or output were retained or saved, and an available recovery or
  exit action. Do not label an unsuccessful save as applied. Invalid or
  unreadable stored configuration must be identified before any proposed reset
  or replacement; do not silently replace it with defaults. Show retrying only
  while a retry is actually pending or running.

**Destructive confirmation:** discarding a dirty multi-field draft, replacing
a saved credential or customized connection, and deleting a stored entity
require confirmation with default focus on Cancel. Applying any change to a
previously customized provider connection requires confirmation, including
ordinary field edits and preset replacement; name all providers and connection
scopes changed by the batch. Creating a connection or leaving an existing
connection unchanged does not require this confirmation. Name every affected
scope, state the consequence, and state what cancellation preserves. Require an
explicit destructive action; inline `y/N` help-text confirmations are banned.
No type-the-name challenges for ordinary local configuration edits.
Risk-acceptance decisions (for example trusting a folder) are separate: they
may default to the affirmative action but must state the consequence of
accepting.

**Credentials:** saved keys and discardable catalog drafts are distinct
concepts and must stay visually and verbally distinct.

## 8. UX story

Chartreux supports a repeating rhythm: the user states intent, the agent works, the user occasionally intervenes, and the user reviews the outcome. Long periods without input are normal. The screen must distinguish useful background work from waiting on a human without demanding continuous attention.

### Attention hierarchy

- When a decision is required, the decision and its consequence take precedence.
- Otherwise, conversation is the primary content. Agent and tool activity explain the work; they do not replace the conversation with a dashboard.
- The composer is the primary control when the user is entering or steering intent.
- Session status is supporting context, not a second activity log.
- Background changes must not steal keyboard focus. An explicit user navigation or foreground decision mode may move focus; that transition must be visible and reversible.
- A stopped main turn and running background agents are independent facts. Neither must be presented as proof that all work is complete.

### Session phases — the screen must communicate

Note: interactive tool/command approvals are not part of the current supported workflow (ADR 0004); these rules apply only if an approval surface is introduced as a separate feature.

- User turn: the editable prompt, its target, and the available submit action. Draft content remains until submission is accepted or the user explicitly discards it.
- Before the first submission, if no usable provider and model are selected,
  show the missing prerequisite and an explicit route to configure or select it.
  Distinguish no configured destination from an unavailable selected
  destination. Returning after cancelled or failed setup must reflect
  successfully persisted steps and identify what remains; never report setup
  complete before a usable destination is selected.
- Agent working: the accepted user message, evolving assistant output when present, and an explicit working state. Thinking, generating, retrying, and interrupting may use distinct labels. Silence from the model is not presented as completion.
- Tool activity: operation and target, running or terminal state, and accessible arguments/result. Routine output remains compact; proposed changes and failures remain discoverable.
- Subagent activity: aggregate background state in chat, with explicit navigation to individual agents. Agent inspection identifies the agent and whether the displayed transcript is live or saved; inspection does not silently retarget the chat composer.
- Approval required: `Approval required`, the operation and affected scope, the consequence of approval, and explicit available decisions. Approval is not inferred from typing, focus, or returning from another surface.
- Question required: an explicit waiting-for-answer state and the question with available answer controls. This is not labelled ordinary generation.
- Turn complete: working feedback ends only on a terminal event. The final response or terminal outcome remains visible, including failure, cancellation, or interruption where applicable; the composer becomes ready for the next turn.
- Failure, interruption, or cancellation must not imply rollback of completed
  tool actions. Retain completed results and identify incomplete or unknown
  outcomes. A retry or resend action must state what it repeats; do not present
  it as resuming from the failure point unless that behavior is supported.
- Idle: no running treatment for completed work. The last outcome remains inspectable, workspace and context remain available, and the next prompt is the primary action.
- Activity labels describe observed state, not inferred progress. Do not invent a percentage complete, estimated completion time, or successful outcome.
- Elapsed time measures elapsed activity, not percent completion. A waiting-for-user interval must be distinguishable from active work.
- Context and cost are session resources, not evidence that a particular tool succeeded. Detailed statistics remain available without making the transcript a telemetry display.
- Completion must not require the model to emit a particular prose summary. If no final prose is available, show the observed terminal outcome.

### Steering during work

- The composer remains usable during work where the application supports queued input or steering.
- Before input is accepted, the applicable action must distinguish queuing another turn, steering current work, and interrupting work.
- Queued input must remain identifiable as queued until consumed. Its count and available edit or removal action must be discoverable.
- Interrupting must be shown as an intermediate state until acknowledged; the UI must not claim cancellation merely because the shortcut was pressed.
- Escape first dismisses the active local interaction level. It may interrupt a turn only when no higher-priority local dismissal applies and that behavior is advertised.
- Opening agent inspection does not send input to that agent. Any future change of input target must be an explicit action with a persistent target label.

### Approvals and questions

Note: interactive tool/command approvals are not part of the current supported workflow (ADR 0004); these rules apply only if an approval surface is introduced as a separate feature.

- A write approval must expose the affected paths and proposed change or a keyboard-accessible route to inspect it before acceptance. Command approval must expose the command and relevant scope.
- Controls must name their actual effect. Do not silently turn permission for one operation into permission for a broader scope.
- Incoming decisions must not overwrite a configuration draft or an answer being edited. While another surface is active, show a compact pending-action indication and a route to the authoritative decision surface.
- Closing or deferring a decision surface must not imply approval. The displayed state must follow the actual pending, rejected, cancelled, or answered outcome.
- After a decision, return to the prior interaction context when it still exists; do not jump to the newest transcript content unless live following was active.

### Surface journey

- Chat is the home context. Settings, provider configuration, pickers, session browsing, and agent inspection are temporary task contexts.
- Secondary surfaces use the existing surface assignments. This contract does not introduce a new navigation framework.
- Opening a secondary surface preserves the opener's composer draft, selected identity, filter, scroll anchor, disclosure choices, and focused control where applicable.
- Entering an editor must state whether acceptance updates a draft or persists immediately. Returning from a picker must not silently apply an enclosing configuration draft.
- Escape unwinds one level using the existing interaction precedence. Returning restores the opener's state; if the original item or control no longer exists, focus moves to the nearest valid control with an explanation when needed.
- For a route such as chat → settings → Provider Settings → picker, each return restores its immediate opener. Invoking Provider Settings directly from chat returns directly to chat.
- A configuration change must communicate its effective scope: current work, subsequent work, or a future session. Do not claim that already-running work adopted a new configuration.
- Opening a secondary surface does not imply that agent work paused. If the operation is unavailable during work, show that restriction before entering it.

### Resuming and inspecting history

- Session preview and session activation are distinct. Preview must identify the inspected session without presenting it as the active destination for new input.
- Moving among previews must not discard the active session's draft or alter its disclosure state.
- Acceptance activates the chosen session only after successful loading. Failure must leave an explicit error and a coherent active-session identity.
- Returning from a cancelled preview restores the previous session context.
- Before session activation or application exit affects running work, pending
  decisions, or queued input, state whether each continues, is cancelled, or
  blocks the action. Queued input and decisions must remain associated with
  their originating session and must not silently transfer to the newly active
  session. If background execution across sessions is unsupported, explain the
  restriction before switching.
- Historical running or approval records must not imply that an old process or decision is still live. Current activity comes from current runtime state.
- Saved or unavailable agent transcripts must be labelled accordingly; an empty transcript is not evidence that no work occurred.

## 9. Compatibility

### 80x24 is the acceptance viewport

- Every core workflow remains operable.
- Docks collapse; large modals promote to full-screen.
- Body content scrolls, not essential navigation.
- Validation messages and confirmations wrap.
- List summaries may truncate; complete identifiers appear in
  selected-item details.
- Below 80x24 further degradation is allowed, but Escape stays usable and an
  explicit size message appears if interaction becomes impossible.
- Live resize must preserve drafts, focused identity, disclosure choices, and
  follow-mode state. Transcript reflow must retain the inspected logical entry
  and nearest available reading position; only a view already following latest
  may remain pinned to latest. A size warning must preserve recoverable
  interaction state, and usable layout must return when sufficient space is
  restored.
- Every supported user action must have a keyboard-accessible route, including
  disclosure, detail inspection, help, and actions exposed through pointer
  controls. Focus must remain visible, and each temporary interaction must
  have a documented keyboard exit. Pointer or hover interaction must never be
  the sole way to obtain required task information.

### Light/dark

Inherit the selected Textual theme. No hardcoded black/white surfaces, no
manually inverted colors. Check primary text, muted help, errors, and focused
rows in both modes.

### Mono, limited color, color-blind use

Meaning survives with color removed:

- focus: cursor plus reverse/bold;
- membership: checkbox/radio shape;
- severity: marker plus words;
- diff: addition/removal prefixes.

Use framework color conversion; do not build separate 16/256/truecolor
palettes. Provide one application-wide ASCII override shared by Textual and
Rich chrome; it replaces cursor, membership, direction, disclosure, and
truncation glyphs and never rewrites user content. The application-wide ASCII
override substitutes every preferred chrome glyph with its ASCII equivalent
from both inventory tables, including information, running, success, warning,
and error markers. No custom capability-detection subsystem is required.

## 10. Codified conventions and migration

The migration order below is historical context, not evidence that every
surface currently conforms. Sections 1-9 are normative acceptance criteria
for all in-scope UI. A surface is conformant only after its current
implementation has been exercised at 80x24 and one larger viewport, with
representative focus, overflow, light, dark, no-color, and ASCII states
recorded. Known inherited or deferred gaps should be listed explicitly and
are not conformance exemptions. `prefer` and `may` express recommendations;
`must`, `never`, `do not`, and unqualified requirements are mandatory.

### Reuse targets

The following are reuse targets, not conformance exemptions; their focus,
color, glyph, and responsive behavior must still satisfy this document.

- The settings screen's information architecture: searchable flat list,
  UPPERCASE section labels, `▸` cursor, inline editing, muted help footer
  (`chartreux/cli/textual_ui/screens/settings.py`).
- The outlined checkbox widget (`chartreux/ui/widgets/checklist.py`).
- The shared shortcut-key formatting helper
  (`chartreux/ui/shortcut_hints.py`).

### Historical migration order and current audit targets

1. **Provider Settings.** Keep the full-screen architecture; unify
   header/filter/list/detail/footer placement; add explicit borderless-list
   focus; remove success-colored selected memberships. Target reading order:
   title → filter/context → primary list → selected-item help/status →
   shortcuts. Detail editing replaces the browser body rather than nesting
   another decorative panel.
2. **Settings and shared pickers.** Replace percentage-sized framing with the
   canonical responsive geometry; add the focused-row treatment.
3. **Bottom sheets and docked agent views.** Normalize titles, separators,
   help placement, and activity labels.
4. **Onboarding and trust/confirmation flows.** Apply full-screen/dialog
   rules and the same action semantics.
5. **Conversation, tool, and diff styling.** Remove operational copper and
   role-colored chrome; align Python-rendered statuses with the same roles.

For each surface, exercise applicable workflows at 80x24 and smoke-test one
larger viewport. Check representative focus, muted text, and severity
feedback in one bundled light theme, one bundled dark theme, and without
color; check ASCII mode once for shared chrome. Record the commands or manual
observations. Automated screenshot matrices and exhaustive terminal-palette
combinations are not required, but a passing historical snapshot is not
evidence for current conformance.

## 11. Scope limits

**Build:** this document, a small shared TCSS foundation, reuse of the
existing shared widgets, and a theme-role resolver for Rich rendering paths
only.

**Do not build:** a token compiler, a component framework, a theme editor,
an exhaustive component catalog, an animation system, or a screenshot matrix
for every terminal palette.

## 12. Inherited upstream elements

Chartreux is a fork of Mistral Vibe. Much of the TUI (chat widgets, pickers,
`app.tcss`, onboarding, the trust dialog, theme detection) derives from
upstream. Rules:

- This document governs chartreux's copies of inherited files. Divergence
  from upstream styling is expected; do not preserve an upstream convention
  that this document overrides.
- Most inherited violations (heavy or round borders, dim applied to muted
  colors, italics, off-scale padding, `$secondary`/copper in operational
  chrome) are chartreux-local TCSS changes. Glyph violations (`●`/`○`/`⚠` in
  the MCP, tool, and message widgets) live in widget code and must be
  changed there; TCSS cannot replace literal status characters.
- The chat input carries semantic states beyond normal/focus/invalid
  (warning, safe, recording). Map warning to `warning`, recording to
  `interactive`, and safe to neutral `text`. `brand` is never used in chrome,
  including these states.
- When porting future upstream changes, restyle ported elements to this
  document rather than importing upstream styling.
