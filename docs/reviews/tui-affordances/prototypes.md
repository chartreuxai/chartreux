# Hypothetical affordance prototypes

These are **text sketches for review, not captures of Chartreux**. They are deliberately ASCII so the action and selection cues can be judged without color. Each pair shows the same mixed surface in a compact 80×24 viewport and a taller 80×48 viewport; unused terminal rows are omitted from the sketch and are not evidence of final geometry. The blind baseline for supplied running-TUI fixtures was logged before these sketches were written. Initial Astra objections were addressed, and an independent visual reader checked the revised text frames. Bounded after-repair visual acceptance found no blocker, but these sketches are not an adopted product contract or tested production layouts. See [contract-proposal.md](contract-proposal.md) for proposed semantics and [findings.md](findings.md) for observed discrepancies.

`>` is keyboard focus only. `[x]` is independent selection. `(o)` is the chosen radio value. `Draft`, `Saved`, and `This session` describe scope. Entire actionable rows, including padding, are proposed mouse targets; headings and status text are inert. Explanations of this notation stay in review prose, outside the proposed product frames.

## 1. Settings: exclusive choice with an immediate field save

80×24 compact concept:

```text
+ User Settings / Appearance ---------------------------------------------+
| Theme                 Saved: Auto   Selected locally: Light           |
|   ( ) Auto                                                               |
|   (o) Light                                                              |
|   ( ) Dark                                                               |
|                                                                         |
| Notifications [x]        Saved to user settings                         |
|                                                                         |
| > Save selected theme to user settings and stay                         |
|                                                                         |
| Status: Theme selection not saved. Notifications already saved.         |
| Enter save selected Light  Esc browser (no save)                        |
+-------------------------------------------------------------------------+
```

80×48 tall concept, showing extra context rather than a different contract:

```text
+ User Settings / Appearance ---------------------------------------------+
| Theme                 Saved default: Auto   Selected locally: Light   |
|                                                                         |
|   ( ) Auto                                                               |
|   (o) Light                                                              |
|   ( ) Dark                                                               |
|                                                                         |
| Related settings                                                        |
|   Notifications [x]             Saved to user settings                   |
|   ASCII chrome [ ]               Saved                                    |
|                                                                         |
| > Save selected theme to user settings and stay                         |
|                                                                         |
| Status: Theme selection not saved. Notifications already saved.         |
| Enter save selected Light  Esc browser (no save)                        |
+-------------------------------------------------------------------------+
```

Both sketches test the chosen **per-field immediate-save policy**. Space selects a Theme radio locally; the focused `Save selected theme` action writes that field to user settings and remains in Settings. Escape returns to the Settings browser without saving the local Theme selection. A Notifications activation saves that toggle to user settings immediately, so its status says `Saved` rather than implying it shares the Theme selection. There is no global Settings draft or combined Save action. The footer describes the focused Save action; focus on a Theme radio would instead say `Space select theme locally`, and focus on Notifications would say `Click/Space/Enter save notification toggle to user settings`. Pending and failed saves need a visible outcome without closing the editor. The `>` cursor and `(o)` selected value stay separate; that distinction is explained here rather than as instructional text inside the UI. This prototype makes the scope explicit where the current enum and boolean cues are misleading (F-002/F-006).

## 2. Provider: inert identity, edit, navigation, and two save destinations

80×24 compact concept:

```text
+ Provider: Mistral                     Catalog draft -------------------+
| Name: mistral [read only]                                                |
| > Edit API base              Draft: https://gateway.example/v1         |
|   Edit API key environment   Saved: MISTRAL_API_KEY                     |
|   Open Models                3 enabled, 1 pending change               |
|                                                                         |
|   Save all catalog changes and stay                                      |
|   Save catalog & open Presets                                             |
|   Discard catalog drafts                                                 |
|                                                                         |
| Pending: Mistral API base, model a, Main preset. Key remains saved.     |
| Click/Enter edit API base  Esc Providers (asks about drafts)           |
+-------------------------------------------------------------------------+
```

80×48 tall concept:

```text
+ Provider: Mistral -----------------------------------------------------+
| Saved connection: mistral                 API key: Saved separately    |
| Catalog draft: 3 changes across provider, model, and preset            |
|                                                                         |
| Name: mistral [read only]                                                |
| > Edit API base          https://gateway.example/v1   Draft            |
|   Edit API style         OpenAI Responses             Saved            |
|   Edit API key environment MISTRAL_API_KEY             Saved            |
|   Open Models            3 enabled; 1 pending change                   |
|   Open Presets           4 configured pairs                            |
|                                                                         |
| Pending catalog changes                                                |
|   Provider mistral: API base                                            |
|   Model a: enabled state                                                |
|   Main preset: model and thinking                                       |
|                                                                         |
|   Save all catalog changes and stay                                     |
|   Save catalog & open Presets                                            |
|   Discard catalog drafts                                                |
|                                                                         |
| Status: 3 catalog edits pending. Saved API key is unaffected.          |
| Click/Enter edit API base  Esc Providers (asks about drafts)           |
+-------------------------------------------------------------------------+
```

Both sketches label the actual potential write scope: one catalog commit can include provider patches, models, and preset pairs. `Save all catalog changes and stay` persists those edits and remains here; `Save catalog & open Presets` persists them and navigates there. `Discard catalog drafts` opens a scoped confirmation, then returns to this provider view if confirmed. Escape with dirty catalog drafts asks whether to keep editing, save, or discard before returning to Providers; it never silently drops work. They intentionally omit the read-only Name focus treatment (F-005). `Edit API base` opens a scalar field editor that accepts into the catalog draft; `Open Models` and `Open Presets` navigate to full views with their own return routes. A model checkbox inside Models would advertise `Click/Space toggle; Enter details`; that instruction belongs on the Models screen, not this footer. During a save, replace both Save rows with inert `Saving catalog changes…` and prevent repeated submission. A failed save keeps the draft and focus; a successful write followed by runtime reload failure says what was saved and exposes Retry Runtime Reload. The saved API key is a separate credential scope and cannot be undone by Discard. No provider row should accept a second click intended for the browser after navigating to Actions (F-019).

## 3. Multi-answer question: select, submit, or send cancellation

80×24 compact concept:

```text
+ Question: Which setup topics should be covered? -----------------------+
| Choose answers locally, then Send to assistant.                        |
|   [x] Authentication                                                    |
| > [ ] Caching                                                           |
|   [ ] Logging                                                           |
|   [ ] Other...                                                          |
|                                                                         |
|   Send answers to assistant                                             |
|   Cancel question & notify assistant                                    |
|                                                                         |
| Status: 1 selected; no response sent.                                   |
| Click/Space/Enter toggle Caching  Esc cancel/notify                   |
+-------------------------------------------------------------------------+
```

80×48 tall concept of the same multi-answer flow:

```text
+ Question: Which setup topics should be covered? -----------------------+
| Choose answers locally, then Send to assistant.                        |
|                                                                         |
|   [x] Authentication       Login, tokens, and authorization            |
| > [ ] Caching              Local response and file caches              |
|   [ ] Logging              Saved session and diagnostic output         |
|   [ ] Other...             Type a separate answer                     |
|                                                                         |
|   Send answers to assistant                                             |
|   Cancel question & notify assistant                                    |
|                                                                         |
| Status: 1 selected; no response sent.                                   |
| No answers have been sent yet.                                         |
| Click/Space/Enter toggle Caching  Esc cancel/notify                   |
+-------------------------------------------------------------------------+
```

Both sizes test one **proposed change** to the existing multi-answer flow: clicking or pressing Space/Enter on an option updates local selection only; only the explicit `Send answers to assistant` row submits. Current `QuestionApp` can auto-submit when Enter accepts the last question (`question_app.py:457`), so implementation would have to change and then be tested. Cancel/Escape retains the existing callback meaning: it sends a cancellation result with no answers to the requester (`question_app.py:508`, `app.py:1891`), then returns to conversation; it does not keep the decision pending. The proposed visible Cancel row makes that route easier to predict; if the final UI keeps only Escape, its focused hint must still say cancellation is sent. Other opens a text-answer route, not a disabled row. With Other focused, the hint would say `Click/Enter type another answer`; with Send focused, `Click/Enter send answers`. This prototype does not invent file-approval controls or their persistence semantics. A separate future approval prototype requires source and runtime evidence. Destructive Discard/Delete elsewhere would default to a **visible** Cancel row, not the blank focus target in F-014; modal background controls would remain inert.

## Blind validation questions

Ask a reviewer who has not read the contract to predict, for every action row in each sketch: what a single click does, what Enter/Space do, whether disk/session/task state changes, where focus returns on Escape or failure, and whether a second click during a transition can act on the next view. Reject a prototype if the reviewer mistakes a cursor for a selected value, predicts a disabled row is informational, cannot identify Save scope, or expects a hidden clickable target. The final implementation should then be captured from the running TUI at 80×24 and 80×48 in dark/light plus named ASCII/no-color cases; these sketches cannot substitute for that evidence.
