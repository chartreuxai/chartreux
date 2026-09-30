# Whole-TUI surface ledger

**Status: partial inventory, blind predictions recorded before behavioral reconciliation.** Baseline commit `37abc9d`. A row marked `B` is a screenshot-only prediction, not an observed effect. `S` is a source inventory anchor, not a runtime test. `R` will denote runtime-confirmed behavior. Do not use a predicted action as a behavior oracle. The blind reader inspected the fresh 80×24 dark seed PNG files and selected 80×48 light PNG files before the live reviewer reported any outcomes; the original independent messages are retained in the review thread. Evidence IDs link to [evidence-index.md](evidence-index.md).

All controls below need mouse tests on the label, icon, and horizontal/vertical padding unless an actual hit area is recorded. `row?` records the blind prediction that the whole visible row is clickable; it is not a verified target. Focus/selection and scope are likewise predictions unless marked `R`. `draft?` means the screenshot did not establish durability. Source paths are inventory anchors supplied separately from the blind reading.

## Source route families and current coverage

| Family | Source inventory anchor | Seed visual coverage | Live coverage | Gaps |
| --- | --- | --- | --- | --- |
| L01 Composer, commands, attachments, queue | `chartreux/cli/textual_ui/app.py:681`; `widgets/chat_input/`; `cli/commands.py:35` | E-CHAT-SEED | Pending | Slash menu, path completion, attachments, queued input, command effects. |
| L02 Settings and providers | `screens/settings.py:279`; `chartreux/ui/providers/workbench.py:86` | E-SETTINGS-SEED, E-PROVIDERS-SEED, E-PROVIDER-ACTIONS-SEED, E-PROVIDER-MODELS-SEED, E-MODEL-DETAIL-SEED | Pending | Settings editors/save/dirty/error; provider choose, catalog, deployment, presets/pair, confirmation and transitions. |
| L03 Bottom pickers, MCP, proxy, question | `app.py:450`; matching `widgets/*_picker.py`, `widgets/*_app.py` | E-MODEL-SEED | Model picker partially R | Thinking/theme/log/MCP/OAuth/proxy/question states and effects. |
| L04 Sessions and rewind | `widgets/session_picker.py:132`; `widgets/rewind_app.py:41`; `widgets/agent_transcript.py:97` | E-SESSION-SEED | Session row partially R | Search, preview, delete, rewind/fork, transcript viewer. |
| L05 Transient decisions and trust | `widgets/question_app.py:33`; `quit_manager.py:19`; `setup/trusted_folders/trust_folder_dialog.py:29` | None | Pending | All approval/exit/trust/modal variants. |
| L06 Agent browser | `widgets/agent_bar.py:61`; `app.py:5739` | None | Pending | Status, nested transcript, task route. |
| L07 Transcript | `widgets/{messages,tools,tool_widgets,collapsible,links,virtual_output,entry_expansion,load_more}.py` | E-CHAT-SEED only | Pending | Disclosure, copy, links, virtualization, tool states. |
| L08 Help, updates, status, errors, debug | `widgets/{debug_console,loading,status_message,inline_notice,banner,context_progress}.py`; `app.py:3606,3805,5981` | E-CHAT-SEED hints only | Pending | Independent routes and loading/error states. |
| L09 First run, auth, trust | `setup/onboarding/`, `setup/auth/`, provider workbench onboarding mode | None | Pending | All first-run state variants; historical temporary images are not durable evidence. |
| L10 Shared focus, mouse, chrome | `ui/widgets/{navigable_option_list,checklist}.py`; `ui/{shortcut_hints,chrome_glyphs,theme}.py`; `app.tcss` | Shared across eight seed stems | Padding partially R | No-color, ASCII, resize, wide, focus and inert hit tests. |

## Source control templates and route closure

These `T` IDs are **source-only templates**, not verified individual affordances. A template lists every known dynamic row class to expand into `Lxx-NNN` controls as screenshots and live tests arrive. Its evidence cell distinguishes captured initial states from uncaptured views. Mouse target, focus transfer, persistence, and state transitions are `pending live` for every untested consumer. This table prevents an initial screenshot from being mistaken for route completion.

| Template | Consumer and source anchor | Dynamic visible control classes | Captured example / missing views | Review status |
| --- | --- | --- | --- | --- |
| T01 | Settings `screens/settings.py:279` | Browser rows/filter/provenance; boolean toggle; enum/radio; checklist; scalar editor/invalid; detail help; override/delete; dirty discard confirmation | E-SETTINGS-SEED, E-SETTINGS-INLINE, E-SETTINGS-CONFIRM; enum/list/invalid/dirty uncaptured | S; initial B partial; R enum/boolean partial |
| T02 | Provider workbench `workbench.py:86-106` | Browser, choose, connection, actions, models, catalog/filter, deployments, detail, editor, protocol, picker, presets, preset-editor, confirm; disabled headings/empty/status and conditional actions | Initial PNG files now exist for all named content views and confirmation; state/interaction variants remain largely uncaptured | S; initial B broad, live partial |
| T03 | Bottom pickers `app.py:450`; `widgets/{model,thinking,log_level}_picker.py`; `ui/widgets/theme_picker.py` | Model/thinking rows/default/current; theme preview/select/cancel; log level Session/config badges, draft/apply/discard | E-MODEL-SEED; thinking/theme/log uncaptured | S; model B/R partial; rest pending |
| T04 | Sessions `widgets/session_picker.py:132` | Rows/search/loading/empty/error/preview/delete confirmation | E-SESSION-SEED, E-SESSION-DELETE; other states uncaptured | S; row B/R partial |
| T05 | MCP `widgets/mcp_app.py:58` | List/search/detail/tools/empty/loading; enabled/disabled/status/tool informational rows | E-MCP-OVERVIEW, E-MCP-EMPTY; detail/loading uncaptured | S/B partial; live pending |
| T06 | OAuth `widgets/mcp_oauth_app.py:52` | Waiting/failed status, open browser, copy/show URL, retry, close | E-OAUTH-WAIT, E-OAUTH-OPEN-FAILED; retry/copy/close uncaptured | S; false-success R |
| T07 | Proxy `widgets/proxy_setup_app.py:19` | HTTP/HTTPS fields, per-field/form errors, Apply, Cancel, dirty Discard | E-PROXY-EMPTY, E-PROXY-ERROR; dirty/pending uncaptured | S/B partial; repeated Apply R at component level |
| T08 | Question `widgets/question_app.py:33` | Rendered single/multi option rows, Other/input, tabs, Submit/Cancel, request failure | E-QUESTION-SINGLE, E-QUESTION-MULTI; tabs/failure uncaptured | S/B partial; live pending |
| T09 | Rewind `widgets/rewind_app.py:41` | Turn/action choice, edit-and-restore/edit-only, in-place/fork, numeric navigation | E-REWIND; action/persistence variants uncaptured | S; live/vision pending |
| T10 | Agent bar/viewer `widgets/{agent_bar,agent_transcript}.py` | Compact bar/browser/details, nested transcript paging/latest/refresh/close | E-AGENT-TRANSCRIPT; bar/browser uncaptured | S/B partial; live pending |
| T11 | Composer `widgets/chat_input/`; `app.py:681` | Text/send, slash/path completion, attachments, queued input, external editor, `!shell`, `@path` | E-CHAT-SEED, E-COMPLETE; attachments/queue/path/editor uncaptured | S/B partial; live pending |
| T12 | Transcript `widgets/{messages,tools,collapsible,links,virtual_output,entry_expansion,load_more}.py` | Reasoning/tool group/tool result/diff/link/copy/virtual load/disclosure | E-TOOLS-PENDING, E-REASONING-EXPANDED, E-REASONING-COLLAPSED; diff/links/long uncaptured | S; reasoning body R; rest pending |
| T13 | Trust `setup/trusted_folders/trust_folder_dialog.py:29` | Trust folder/repo/decline, inspect files, storage text | E-TRUST-FOLDER, E-TRUST-REPO; file inspect uncaptured | S/B partial; live pending |
| T14 | Onboarding `setup/onboarding/screens/welcome.py:28`; provider onboarding mode | Animated Welcome first/second Enter, provider journey, preset finish/repair | Provider initial management stems only; first-run states uncaptured | S; live/vision pending |
| T15 | Quit `quit_manager.py:19` | Active consequential dialog Enter/Esc; idle PathDisplay confirmation | E-EXIT-CONSEQUENCES; idle variant uncaptured | S; live/vision pending |
| T16 | Help/status/debug/updates/loading `app.py:3606,3805,5981`; `widgets/{debug_console,status_message,inline_notice,loading}.py` | Help and status text, debug rows, notices, progress, errors | E-CHAT-SEED status only; dedicated views uncaptured | S; live/vision pending |
| T17 | Shared `ui/widgets/{navigable_option_list,checklist}.py`; `ui/{shortcut_hints,chrome_glyphs,theme}.py` | Cursor, checkbox/radio, disclosure, padding, disabled/inert, modal chrome, footer/banner | Shared in captures; no-color/ASCII/wide uncaptured | S; per-consumer exceptions pending |
| T18 | Command registry `chartreux/cli/commands.py:35-208` | 26 keys/aliases map to screen, inline effect, external action, or conditional status | E-COMPLETE is preview only; entry-to-effect coverage pending | S; route map below, live pending |

Provider T02 view closure uses the `WorkbenchView` enum at `workbench.py:86-106` as its denominator: **13 reachable content subroutes plus one confirmation overlay**. The enum names `CONFIRM`, but the overlay is controlled by `_confirm` in `_sync_view()` rather than assigning `_view = CONFIRM`. `providers` has heading/empty/provider rows/Add/Catalog/Presets/conditional Apply/Discard (E-PROVIDERS-SEED). `actions` has connection fields, credential status, Discover/Retry, Models, collision resolution, Apply/Presets/Discard/Retry Runtime Reload (E-PROVIDER-ACTIONS-SEED). `models` has checklist, Retry discovery, Edit connection, Add manually, Save and add another, Save and continue (E-PROVIDER-MODELS-SEED). `detail` has thinking, temperature, compaction, prices, images, Save model edits (E-MODEL-DETAIL-SEED). `choose` provider presets, `connection` fields and Save and configure, `catalog` provider filter/model rows/Apply/Discard/empty, `deployments` provider-name rows, `editor` scalar input, `protocol` choice rows, `picker` model/thinking choices and unavailable row, `presets` heading/four role rows/Finish/Add another, `preset-editor` Model/Thinking/Apply/Cancel, and `confirm` scoped Keep editing/Save/Discard/Continue all have **initial-state** captures linked in the evidence index. Conditional, disabled, empty, dirty, pending, error, keyboard, and mouse variants remain incomplete; each consequential row/state needs a separate control ID and fixture before closure.

T18 registry route map: `/model` → model picker; `/thinking` → thinking picker; `/theme` → theme picker; `/settings` → Settings; `/providers` → workbench; `/proxy-setup` → proxy; `/resume|/continue` → SessionPicker; `/mcp` → overview/add/status/login/logout; `/rewind` → rewind; `/agents` → agent bar/transcript; `/debug` → debug console. `/help`, `/status`, `/log`, `/copy`, `/rename`, `/reload`, `/clear|/new`, `/compact`, `/retry`, `/loop`, and `/branch` produce inline transcript, status, clipboard, loading, or fork effects to trace individually. `/paste-image` conditionally affects composer attachment, `/open-config-file` opens an external editor and reloads, and `/exit|exit|quit|:q|:quit` follows idle or consequential quit. Slash completion is a command preview, not proof of execution. `!shell`, `@path`, and Ctrl-G external editor are separate composer input modes. All effect mappings here are source inventory, **not** live-reviewed behavior.

## Blind control predictions: fresh seed captures

Source abbreviations: `SET` = `chartreux/cli/textual_ui/screens/settings.py`; `WB` = `chartreux/ui/providers/workbench.py`; `MP` = `chartreux/cli/textual_ui/widgets/model_picker.py`; `SP` = `chartreux/cli/textual_ui/widgets/session_picker.py`; `APP` = `chartreux/cli/textual_ui/app.py`. These are route anchors; exact handler lines remain to be reconciled. In the keyboard column, `Enter`/`Space` are predicted from visual labels or on-screen help. `—` means no interactive action predicted.

| Control ID | Route/source | Visible label or shape | Predicted class and consequence | Keyboard; mouse; focus/selection | Scope/state | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| L02-001 | Settings/SET | `show_greeting` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-002 | Settings/SET | `autocopy_to_clipboard` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-003 | Settings/SET | `ask_confirmation_on_exit` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-004 | Settings/SET | `file_watcher_for_autocomplete` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-005 | Settings/SET | `disable_welcome_banner_animation` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-006 | Settings/SET | `context_warnings` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-007 | Settings/SET | `show_thinking_nodes` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-008 | Settings/SET | `ascii_chrome` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-009 | Settings/SET | `enable_notifications` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-010 | Settings/SET | `include_commit_signature` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-011 | Settings/SET | `include_model_info` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-012 | Settings/SET | `include_project_context` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-013 | Settings/SET | `include_prompt_detail` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-014 | Settings/SET | `raise_on_compaction_failure` checkbox | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-015 | Settings/SET | `displayed_workdir` value | B open scalar editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-016 | Settings/SET | Filter hint/input | B type to narrow settings | typing; field?; input focus | local/transient | E-SETTINGS-SEED |
| L02-017 | Settings/SET | Saved/effective provenance labels | B inert information | —; inert; no activation | persisted/status | E-SETTINGS-SEED |
| L02-018 | Settings/SET | Section headings/help | B inert information | —; inert; no activation | information | E-SETTINGS-SEED |
| L02-019 | Settings/SET | `auto_compact_threshold` | B open scalar editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-020 | Settings/SET | `system_prompt_id` | B open scalar editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-021 | Settings/SET | `compaction_prompt_id` | B open scalar editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-022 | Settings/SET | `project_context.default_commit_count` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-023 | Settings/SET | `project_context.timeout_seconds` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-024 | Settings/SET | `subagents.idle_ttl_seconds` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-025 | Settings/SET | `subagents.max_idle_agents` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-026 | Settings/SET | `session_logging.save_dir` | B open scalar editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-027 | Settings/SET | `session_logging.session_prefix` | B open scalar editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-028 | Settings/SET | `api_timeout` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-029 | Settings/SET | `api_connect_timeout` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-030 | Settings/SET | `api_write_timeout` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-031 | Settings/SET | `api_pool_timeout` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-032 | Settings/SET | `api_retry_max_elapsed_time` | B open numeric editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-033 | Settings/SET | `session_logging.enabled` | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-034 | Settings/SET | `session_logging.generate_titles` | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-035 | Settings/SET | `enable_system_trust_store` | B toggle value | Space; box/row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-036 | Settings/SET | `agent_paths` `[0 items]` | B open list editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L02-037 | Settings/SET | `skill_paths` `[0 items]` | B open list editor | Enter; row?; highlighted row | draft? | E-SETTINGS-SEED |
| L04-001 | Session picker/SP | First session row | B resume and close | Enter/click row?; highlight is focus, not current | session navigation; pending? | E-SESSION-SEED |
| L04-002 | Session picker/SP | Second session row | B resume and close | Enter/click row?; highlight is focus, not current | session navigation; pending? | E-SESSION-SEED |
| L04-003 | Session picker/SP | `d Delete` hint | B scoped delete confirmation | d; footer click?; focus unclear | destructive, untested | E-SESSION-SEED |
| L04-004 | Session picker/SP | Workdir and metadata columns | B inert information | —; inert | status | E-SESSION-SEED |
| L03-001 | Model picker/MP | `Default` row | B apply configured default and close | Enter/click row?; highlight is cursor | session? | E-MODEL-SEED |
| L03-002 | Model picker/MP | `mistral-large` row | B apply model and close | Enter/click row?; highlight is cursor | session | E-MODEL-SEED |
| L03-003 | Model picker/MP | `devstral` row | B apply model and close | Enter/click row?; highlight is cursor | session | E-MODEL-SEED |
| L03-004 | Model picker/MP | `codestral` row | B apply model and close | Enter/click row?; highlight is cursor | session | E-MODEL-SEED |
| L01-001 | Chat/APP | Composer text area | B edit and send prompt | Enter; field click; text cursor | session action | E-CHAT-SEED |
| L01-002 | Chat/APP | `>` prompt marker | B inert prompt decoration | —; inert | information | E-CHAT-SEED |
| L08-001 | Chat/APP | `F1 Help` footer | B opens Help via key; mouse uncertain | F1; footer hit pending | navigation | E-CHAT-SEED |
| L08-002 | Chat/APP | `/help` hint | B command hint, not direct button | type command; footer hit pending | information/navigation | E-CHAT-SEED |
| L07-001 | Chat/APP | Banner/model/counts/workdir/PID/context/transcript text | B inert information | —; inert unless links/disclosures appear | status | E-CHAT-SEED |
| L02-038 | Provider browser/WB | Filter | B type to narrow providers | typing; input/row?; focus | local/transient | E-PROVIDERS-SEED |
| L02-039 | Provider browser/WB | `one` provider row | B open connection/deployments | Enter/click row?; highlight cursor | navigation | E-PROVIDERS-SEED |
| L02-040 | Provider browser/WB | `two` provider row | B open connection/deployments | Enter/click row?; highlight cursor | navigation | E-PROVIDERS-SEED |
| L02-041 | Provider browser/WB | `Key Set`, model count badges | B inert status | —; inert | saved/status | E-PROVIDERS-SEED |
| L02-042 | Provider browser/WB | Add provider | B open connection form | Enter/click row? | navigation | E-PROVIDERS-SEED |
| L02-043 | Provider browser/WB | Catalog | B open model catalog | Enter/click row? | navigation | E-PROVIDERS-SEED |
| L02-044 | Provider browser/WB | Presets | B open role preset list | Enter/click row? | navigation | E-PROVIDERS-SEED |
| L02-045 | Provider actions/WB | Name read-only row | B inert; focus/help conflict | Enter? / click?; row highlight | saved/status | E-PROVIDER-ACTIONS-SEED |
| L02-046 | Provider actions/WB | API base | B inline scalar editor | Enter/click row?; highlight | draft? | E-PROVIDER-ACTIONS-SEED |
| L02-047 | Provider actions/WB | API key environment | B inline scalar editor | Enter/click row?; highlight | draft? | E-PROVIDER-ACTIONS-SEED |
| L02-048 | Provider actions/WB | API style | B choice picker | Enter/click row?; highlight | draft? | E-PROVIDER-ACTIONS-SEED |
| L02-049 | Provider actions/WB | API key | B separate masked key editor | Enter/click row? | credential save separate? | E-PROVIDER-ACTIONS-SEED |
| L02-050 | Provider actions/WB | Discover | B run network discovery | Enter/click row? | external/read action | E-PROVIDER-ACTIONS-SEED |
| L02-051 | Provider actions/WB | Models | B navigate model selection | Enter/click row? | navigation | E-PROVIDER-ACTIONS-SEED |
| L02-052 | Provider actions/WB | Apply | B save catalog changes, stay | Enter/click row? | persisted? | E-PROVIDER-ACTIONS-SEED |
| L02-053 | Provider actions/WB | Presets | B navigate preset list | Enter/click row? | navigation | E-PROVIDER-ACTIONS-SEED |
| L02-054 | Provider actions/WB | Discard | B revert unsaved changes with confirmation | Enter/click row? | draft destructive? | E-PROVIDER-ACTIONS-SEED |
| L02-055 | Provider models/WB | Model checkbox | B toggle draft inclusion | Space/click square; selection vs highlight separate | draft | E-PROVIDER-MODELS-SEED |
| L02-056 | Provider models/WB | Model row label | B open model editor | Enter/click label?; highlight cursor | navigation | E-PROVIDER-MODELS-SEED |
| L02-057 | Model detail/WB | Thinking | B open choice/editor | Enter/click row? | local draft | E-MODEL-DETAIL-SEED |
| L02-058 | Model detail/WB | Temperature | B open scalar editor | Enter/click row? | local draft | E-MODEL-DETAIL-SEED |
| L02-059 | Model detail/WB | Auto compact threshold | B open scalar editor | Enter/click row? | local draft | E-MODEL-DETAIL-SEED |
| L02-060 | Model detail/WB | Input price | B open scalar editor | Enter/click row? | local draft | E-MODEL-DETAIL-SEED |
| L02-061 | Model detail/WB | Output price | B open scalar editor | Enter/click row? | local draft | E-MODEL-DETAIL-SEED |
| L02-062 | Model detail/WB | Cached input price | B open scalar editor | Enter/click row? | local draft | E-MODEL-DETAIL-SEED |
| L02-063 | Model detail/WB | Image support | B toggle | Space/click row? | local draft | E-MODEL-DETAIL-SEED |
| L02-064 | Model detail/WB | Save model edits | B persist and return to model list; persistence uncertain | Enter/click row? | draft or disk? | E-MODEL-DETAIL-SEED |

## Blind hypotheses from temporary historical screenshots

These were logged before the live behavior report, but the images are outside the repository and are **not durable evidence**. They require fresh offline captures before a finding can cite them: preset rows for orchestrator/large/medium/small appear to open a pair editor; Finish appears to save/complete; Add provider navigates. The pair editor's Model and Thinking rows appear to open pickers, Apply accepts both values, and Cancel/Esc discards local changes. The historical connection screen suggests scalar Name/Base/Env editors, API style picker, separate key editor, and Save & configure navigating to Models. The historical model list suggests separate Save & add provider and Save & continue actions, retry discovery, edit connection, manual add, and a scrollable model checklist. These predictions are not yet mapped to control IDs because no durable capture exists.

## Blind control predictions: fresh extended captures

The blind reader inspected these live-capture PNG files without reading their manifest or behavior oracle. A linked image is still visual evidence only. `pending` in a keyboard or mouse cell means the blind reader could not infer the target from the image. These controls are not claimed complete for their route.

| Control ID | Route/source | Visible label or shape | Predicted class and consequence | Keyboard; mouse; focus/selection | Scope/state | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| L01-003 | Slash completion, `widgets/chat_input/completion_popup.py` | `/main-test-generator` | B insert command into composer, not run | Enter/click row?; highlighted candidate | local input | [PNG](evidence/live/completion-slash-80x24-textual-dark.png) |
| L01-004 | Slash completion | `/main-debugging` | B insert command into composer, not run | Enter/click row? | local input | [PNG](evidence/live/completion-slash-80x24-textual-dark.png) |
| L01-005 | Slash completion | `/main-plan` | B insert command into composer, not run | Enter/click row? | local input | [PNG](evidence/live/completion-slash-80x24-textual-dark.png) |
| L01-006 | Slash completion | `/mcp` | B insert command into composer, not run | Enter/click row? | local input | [PNG](evidence/live/completion-slash-80x24-textual-dark.png) |
| L03-010 | MCP overview, `widgets/mcp_app.py` | Search | B filter server rows | type/click field?; text focus | local input | [PNG](evidence/live/mcp-overview-80x24-textual-dark.png) |
| L03-011 | MCP overview | `filesystem` server row | B open server tools/details | Enter/click row?; highlight | navigation | [PNG](evidence/live/mcp-overview-80x24-textual-dark.png) |
| L03-012 | MCP overview | `search` server row | B open server tools/details | Enter/click row?; highlight | navigation | [PNG](evidence/live/mcp-overview-80x24-textual-dark.png) |
| L03-013 | MCP overview | `d/e` hint | B disable/enable server | d/e; footer click pending | durability unknown | [PNG](evidence/live/mcp-overview-80x24-textual-dark.png) |
| L03-014 | MCP overview | Counts/transport/title | B inert information | —; inert | status | [PNG](evidence/live/mcp-overview-80x24-textual-dark.png) |
| L03-015 | MCP empty | `/mcp add <url>` suggestion | B inert command suggestion | type command; text itself inert | information | [PNG](evidence/live/mcp-empty-80x24-textual-dark.png) |
| L03-016 | Proxy, `widgets/proxy_setup_app.py` | HTTP proxy | B text field edit | Tab/Enter?; click field; text focus | draft | [PNG](evidence/live/proxy-empty-80x24-textual-dark.png) |
| L03-017 | Proxy | HTTPS proxy | B text field edit | Tab/Enter?; click field; text focus | draft | [PNG](evidence/live/proxy-empty-80x24-textual-dark.png) |
| L03-018 | Proxy | Apply changes | B persist config and close | Enter/click button; focused button | persistent? | [PNG](evidence/live/proxy-empty-80x24-textual-dark.png) |
| L03-019 | Proxy | Cancel | B discard and close | Enter/click button or Esc | local draft | [PNG](evidence/live/proxy-empty-80x24-textual-dark.png) |
| L03-020 | MCP OAuth, `widgets/mcp_oauth_app.py` | Open in browser | B open external authorization URL | Enter/click row? | external | [PNG](evidence/live/oauth-wait-80x24-textual-dark.png) |
| L03-021 | MCP OAuth | Copy/show URL | B copy or reveal manual URL | key/click pending | clipboard/local | [PNG](evidence/live/oauth-wait-80x24-textual-dark.png) |
| L03-022 | Proxy error | Error text | B inert validation feedback | —; inert | error | [PNG](evidence/live/proxy-error-80x24-textual-dark.png) |
| L05-001 | Single question, `widgets/question_app.py` | Other / Type your answer | B open text entry, despite dim treatment | Enter/click row?; highlight | local answer | [PNG](evidence/live/question-single-80x24-textual-dark.png) |
| L05-002 | Single question | Answer rows 1/2/3 | B submit one answer and close | Enter/click row?, numeric shortcut? | turn answer | [PNG](evidence/live/question-single-80x24-textual-dark.png) |
| L05-003 | Multi question | Auth checkbox | B toggle answer | Space/click box?; box is selection | draft answer | [PNG](evidence/live/question-multi-80x24-textual-dark.png) |
| L05-004 | Multi question | Caching checkbox | B toggle answer | Space/click box? | draft answer | [PNG](evidence/live/question-multi-80x24-textual-dark.png) |
| L05-005 | Multi question | Logging checkbox | B toggle answer | Space/click box? | draft answer | [PNG](evidence/live/question-multi-80x24-textual-dark.png) |
| L05-006 | Multi question | Other | B open free text and select | Enter/click row? | draft answer | [PNG](evidence/live/question-multi-80x24-textual-dark.png) |
| L05-007 | Multi question | Submit | B send selected answers and close | Enter/click row? | turn answer | [PNG](evidence/live/question-multi-80x24-textual-dark.png) |
| L05-008 | Trust folder, `setup/trusted_folders/trust_folder_dialog.py` | Trust folder | B persist trust for named folder and continue | Enter/click row?, number | durable trust | [PNG](evidence/live/trust-folder-80x24-textual-dark.png) |
| L05-009 | Trust repo | Trust full repo | B persist whole-repo trust and continue | Enter/click row?, number | durable trust | [PNG](evidence/live/trust-repo-80x24-textual-dark.png) |
| L05-010 | Trust dialog | Don't trust | B reject current context; duration unknown | Enter/click row?, number | scope unknown | [PNG](evidence/live/trust-folder-80x24-textual-dark.png) |
| L05-011 | Trust dialog | Tab inspect files | B open file detail/list | Tab; mouse target absent | navigation | [PNG](evidence/live/trust-folder-80x24-textual-dark.png) |
| L05-012 | Trust dialog | Path/storage explanation | B inert information | —; inert | status | [PNG](evidence/live/trust-folder-80x24-textual-dark.png) |
| L04-005 | Session delete confirm | Cancel | B keep session and return picker | Enter/click button or Esc | destructive cancellation | [PNG](evidence/live/session-delete-confirm-80x24-textual-dark.png) |
| L04-006 | Session delete confirm | Delete session | B permanently remove named history, return picker | Enter/click button | destructive | [PNG](evidence/live/session-delete-confirm-80x24-textual-dark.png) |
| L04-007 | Session delete confirm | Background session row | B inert while modal is open | —; click should not activate | blocked | [PNG](evidence/live/session-delete-confirm-80x24-textual-dark.png) |
| L02-065 | Settings inline editor | Text field / Enter accept | B edit then persist on Enter | Enter/click field; text focus | persistent? | [PNG](evidence/live/settings-inline-80x24-textual-dark.png) |
| L02-066 | Settings trust confirm | Trust authorities / footer Enter Cancel | B conflicting action/focus cue; cannot predict Enter reliably | arrows/Enter/Esc; row? | durable? | [PNG](evidence/live/settings-confirm-80x24-textual-dark.png) |
| L06-001 | Agent transcript, `widgets/agent_transcript.py` | Read file tool line | B expand tool result | Enter/click header? | local disclosure | [PNG](evidence/live/agent-transcript-80x24-textual-dark.png) |
| L07-002 | Agent transcript | Thought line | B collapse/expand reasoning | Enter/click header? | local disclosure | [PNG](evidence/live/agent-transcript-80x24-textual-dark.png) |
| L06-002 | Agent transcript | Called tools, thought summary | B possibly group disclosure | Enter/click pending | local disclosure? | [PNG](evidence/live/agent-transcript-80x24-textual-dark.png) |
| L06-003 | Agent transcript | Attached image filename | B uncertain link/action | Enter/click pending | unknown | [PNG](evidence/live/agent-transcript-80x24-textual-dark.png) |
| L06-004 | Agent transcript | PageUp / refresh / End / Esc hints | B older page, refresh, latest, close | named keys; footer click pending | navigation | [PNG](evidence/live/agent-transcript-80x24-textual-dark.png) |
| L07-003 | Pending tool row | Reading files | B status, perhaps disclosure | Enter/click pending | pending/status | [PNG](evidence/live/tools-pending-80x24-textual-dark.png) |
| L03-023 | OAuth waiting | `Copy URL` | B copy authorization URL | key/Enter/click pending | clipboard | [PNG](evidence/live/oauth-wait-80x24-textual-dark.png) |
| L03-024 | OAuth waiting | `Show URL` | B disclose URL inline | key/Enter/click pending | local disclosure | [PNG](evidence/live/oauth-wait-80x24-textual-dark.png) |
| L03-025 | OAuth waiting | Running/wait/help | B inert status | —; inert | pending | [PNG](evidence/live/oauth-wait-80x24-textual-dark.png) |
| L03-026 | OAuth waiting | Esc Close | B dismiss panel; whether login continues unclear | Esc; footer click uncertain | navigation/pending | [PNG](evidence/live/oauth-wait-80x24-textual-dark.png) |
| L03-027 | OAuth waiting | Retry | B restart auth or browser launch, ambiguous | R; footer click uncertain | external/pending | [PNG](evidence/live/oauth-wait-80x24-textual-dark.png) |
| L05-013 | Exit consequences | Enter Confirm shutdown | B end session with stated nonrollback effects | Enter; footer click uncertain | consequential quit | [PNG](evidence/live/exit-consequences-80x24-textual-dark.png) |
| L05-014 | Exit consequences | Esc keep working | B cancel quit | Esc; footer click uncertain | navigation | [PNG](evidence/live/exit-consequences-80x24-textual-dark.png) |
| L07-004 | Reasoning collapsed/expanded | Thought checkmark/heading | B completed status, perhaps disclosure | Enter/click pending | local disclosure? | [collapsed](evidence/live/reasoning-collapsed-80x24-textual-dark.png), [expanded](evidence/live/reasoning-expanded-80x24-textual-dark.png) |
| L07-005 | Resumed transcript | Read files: `test.txt` | B completed tool, perhaps disclose; file link uncertain | Enter/click pending | local/external unknown | [PNG](evidence/live/resumed-transcript-80x24-textual-dark.png) |
| L04-008 | Rewind panel | Cancel | B keep conversation/files intact | Enter/click row? | cancel | [PNG](evidence/live/rewind-panel-80x24-textual-dark.png) |
| L04-009 | Rewind panel | Edit & restore | B restore files at selected turn, open prompt edit | Enter/click row? | consequential draft/history | [PNG](evidence/live/rewind-panel-80x24-textual-dark.png) |
| L04-010 | Rewind panel | Edit without restoring | B edit prompt without restoring files | Enter/click row? | history only? | [PNG](evidence/live/rewind-panel-80x24-textual-dark.png) |
| L04-011 | Rewind panel | Highlighted transcript turn | B selected rewind target, not control focus | arrows/numbers?; click message uncertain | selection | [PNG](evidence/live/rewind-panel-80x24-textual-dark.png) |
| L04-012 | Rewind panel | Left/Right/q hints | B previous/next target; q closes rewind, not app | named keys; footer click uncertain | navigation | [PNG](evidence/live/rewind-panel-80x24-textual-dark.png) |
| L03-028 | Thinking picker | Off/Low/Medium/High/Max rows | B apply current-session thinking and close | Enter/click row?; colored ANSI export has no clear focus row | session | [PNG](evidence/live/thinking-picker-80x24-ansi-dark.png) |
| L03-029 | Thinking picker | Active badge | B inert applied-value indicator | —; inert, distinct from highlight | status | [PNG](evidence/live/thinking-picker-80x24-ansi-dark.png) |
| L03-030 | Theme picker | Auto/Light/Dark rows | B arrows preview, Enter persists and closes, Esc restores old | arrows/Enter/Esc; click preview vs commit unclear | preview then persisted? | [PNG](evidence/live/theme-picker-80x24-ansi-dark.png) |
| L03-031 | Theme picker | Active badge | B inert current-value indicator | —; inert | status | [PNG](evidence/live/theme-picker-80x24-ansi-dark.png) |
| L03-032 | Log-level picker | DEBUG/INFO/WARNING/ERROR/CRITICAL rows | B separate Session/config radio values; Enter applies | Tab scope, Space toggles, Enter applies; row hit pending | session plus config | [PNG](evidence/live/loglevel-picker-80x24-ansi-dark.png) |
| L03-033 | Log-level picker | Effective WARNING / Active | B inert value indicators | —; inert | status | [PNG](evidence/live/loglevel-picker-80x24-ansi-dark.png) |
| L08-010 | Debug console | Log rows | B inert log text, perhaps selectable for copy | click scope unknown; scroll expected | informational/clipboard? | [PNG](evidence/live/debug-console-80x24-ansi-dark.png) |
| L08-011 | Debug console | Ctrl+\\ Close footer | B close console | Ctrl+\\; footer click uncertain | navigation | [PNG](evidence/live/debug-console-80x24-ansi-dark.png) |
| L02-067 | Provider choose | Existing `one` / `two` | B open existing connection; cloning ambiguity | Enter/click row? | navigation | [PNG](evidence/live/provider-choose-80x24-ansi-dark.png) |
| L02-068 | Provider choose | Mistral / Ollama Cloud / OpenCode Go | B open prefilled new connection | Enter/click row? | local draft | [PNG](evidence/live/provider-choose-80x24-ansi-dark.png) |
| L02-069 | Provider choose | Generic OpenAI / Anthropic / Fully custom | B open new connection form | Enter/click row? | local draft | [PNG](evidence/live/provider-choose-80x24-ansi-dark.png) |
| L02-070 | Provider connection | Name | B open draft text editor | Enter/click row? | draft | [PNG](evidence/live/provider-connection-80x24-ansi-dark.png) |
| L02-071 | Provider connection | API base | B open draft text editor | Enter/click row? | draft | [PNG](evidence/live/provider-connection-80x24-ansi-dark.png) |
| L02-072 | Provider connection | API key environment | B open draft text editor | Enter/click row? | draft | [PNG](evidence/live/provider-connection-80x24-ansi-dark.png) |
| L02-073 | Provider connection | API style | B open protocol picker | Enter/click row? | draft | [PNG](evidence/live/provider-connection-80x24-ansi-dark.png) |
| L02-074 | Provider connection | API key | B open separate credential editor | Enter/click row? | separate save | [PNG](evidence/live/provider-connection-80x24-ansi-dark.png) |
| L02-075 | Provider connection | Save and configure models | B persist connection and move forward | Enter/click row? | durable | [PNG](evidence/live/provider-connection-80x24-ansi-dark.png) |
| L02-076 | Provider catalog | All Providers / `one` / `two` filter | B change visible model subset only | Enter/click row?; selected filter marker unclear | local view | [PNG](evidence/live/provider-catalog-filter-80x24-ansi-dark.png) |
| L02-077 | Provider catalog | Model `a` / `b` rows | B open model details or deployment picker | Enter/click row? | navigation | [PNG](evidence/live/provider-catalog-filter-80x24-ansi-dark.png) |
| L02-078 | Provider catalog | Counts/preset badges | B inert status | —; inert | status | [PNG](evidence/live/provider-catalog-filter-80x24-ansi-dark.png) |
| L02-079 | Deployment picker | `one/a` / `two/a` | B open selected deployment details, not session model | Enter/click row? | navigation | [PNG](evidence/live/provider-deployments-picker-80x24-ansi-dark.png) |
| L02-080 | Provider presets | Main / Large / Medium / Small rows | B open pair editor | Enter/click row? | draft navigation | [PNG](evidence/live/provider-presets-80x24-ansi-dark.png) |
| L02-081 | Provider presets | Save presets | B persist all changed pairs; destination unclear | Enter/click row? | durable? | [PNG](evidence/live/provider-presets-80x24-ansi-dark.png) |
| L02-082 | Provider presets | Add another provider | B navigate to provider setup | Enter/click row? | navigation | [PNG](evidence/live/provider-presets-80x24-ansi-dark.png) |
| L02-083 | Pair editor | Model | B open canonical model picker | Enter/click row? | pair draft | [PNG](evidence/live/provider-preset-editor-80x24-ansi-dark.png) |
| L02-084 | Pair editor | Thinking | B open level picker | Enter/click row? | pair draft | [PNG](evidence/live/provider-preset-editor-80x24-ansi-dark.png) |
| L02-085 | Pair editor | Apply model and thinking | B accept pair into parent preset draft, not disk | Enter/click row? | parent draft | [PNG](evidence/live/provider-preset-editor-80x24-ansi-dark.png) |
| L02-086 | Pair editor | Cancel | B discard local pair edits and return | Enter/click row? or Esc | local cancellation | [PNG](evidence/live/provider-preset-editor-80x24-ansi-dark.png) |
| L02-087 | Preset model picker | `a` / `b` | B select canonical model into pair editor; Current is pair value | Enter/click row? | pair draft | [PNG](evidence/live/provider-preset-model-picker-80x24-ansi-dark.png) |
| L02-088 | Preset thinking picker | Off/Low/Medium/High/Max | B select level into pair editor; Current distinct from focus | Enter/click row? | pair draft | [PNG](evidence/live/provider-preset-thinking-picker-80x24-ansi-dark.png) |
| L02-089 | Protocol picker | OpenAI/OpenAI Responses/Anthropic radio | B Space chooses local draft, Enter accepts, Esc restores old | Space/Enter/Esc; click choice? | connection draft | [PNG](evidence/live/provider-protocol-80x24-ansi-dark.png) |
| L02-090 | Inline editor | API base text field | B Enter accepts field into draft; Esc cancels field | Enter/Esc; click field | connection draft | [PNG](evidence/live/provider-inline-editor-80x24-ansi-dark.png) |
| L02-091 | Discard confirmation | Keep editing | B close modal, preserve drafts | Enter/click row? or Esc | cancellation | [PNG](evidence/live/provider-discard-confirm-80x24-ansi-dark.png) |
| L02-092 | Discard confirmation | Discard all pending changes | B discard provider/global preset drafts, not saved keys | Enter/click row? | destructive draft | [PNG](evidence/live/provider-discard-confirm-80x24-ansi-dark.png) |
| L02-093 | Discard confirmation | Background rows | B inert during modal | click should not activate | blocked | [PNG](evidence/live/provider-discard-confirm-80x24-ansi-dark.png) |
| L09-001 | Onboarding choose | Provider type rows | B same draft-opening behavior as management choose | Enter/click row? | draft | [PNG](evidence/live/provider-onboarding-choose-80x24-ansi-dark.png) |
| L09-002 | Onboarding connection | Name/Base/Env/Style/Key | B same local editors; Escape destination unclear | Enter/click row? | mixed draft/credential | [PNG](evidence/live/provider-onboarding-connection-80x24-ansi-dark.png) |

All non-control titles, section headings, help copy, counters, and footer key hints in these captures are predicted inert unless listed separately above. Their hit areas remain untested; the line is an inventory hypothesis rather than a blanket pass. The `oauth-open-failed` image visibly says `Opened in browser`; its filename is fixture context, not evidence that a blind reader could detect launch failure. The collapsed and expanded reasoning images look alike to the blind reader; they cannot alone prove a disclosure transition. The full-host ANSI question capture shows Other with normal contrast, so F-011's disabled-looking concern is palette/host dependent until a validated theme matrix proves it. Standalone widget hosts can place panels differently from the production shell; placement and clipping findings require `host_context` from the capture manifest.

## Later screenshot hypotheses, after initial oracle exposure

These are fresh reads of newly captured routes, but the same reviewer already knew behavior results for other controls. They are **not** part of the independent blind baseline. Preserve the distinction when scoring prediction accuracy; a new independent reviewer is needed for acceptance on these routes.

| Control ID | New route and visible target | Screenshot-only hypothesis | Evidence and next check |
| --- | --- | --- | --- |
| L06-005 | Agent bar Main row | Click/Enter returns conversation; status text inert | [Browser](evidence/live/agent-bar-browser-80x24-ansi-dark.png); mouse/keyboard return pending. |
| L06-006 | Agent bar review/build agent rows | Click/Enter opens saved transcript; D opens details | [Browser](evidence/live/agent-bar-browser-80x24-ansi-dark.png), [corrected details](evidence/live/agent-bar-details-80x24-ansi-dark.png); live agent-route effects are separately recorded below. |
| L01-007 | Queued composer `»` header | Possibly discloses queued items | [Queue](evidence/live/queued-composer-80x24-ansi-dark.png); click/keyboard pending. |
| L01-008 | Queued `You` entry | Likely informational pending message; no edit/remove target advertised | [Queue](evidence/live/queued-composer-80x24-ansi-dark.png); queue editing/order pending. |
| L01-009 | Queued composer | Enter may queue next prompt; timing/order unclear | [Queue](evidence/live/queued-composer-80x24-ansi-dark.png); live route pending. |
| L01-010 | File mention popup rows | Click/Enter should insert path without sending | [Mention](evidence/live/file-mention-popup-80x24-ansi-dark.png); label/padding hit test pending. |
| L01-011 | Full-shell slash popup rows | Click/Enter should insert command, not execute | [Popup](evidence/live/full-shell-slash-popup-80x24-ansi-dark.png), [corrected after-click frame](evidence/live/full-shell-slash-click-80x24-ansi-dark.png); shared popup mouse-inert effect is R in F-021. |
| L01-012 | Composer `@` image path | Enter may send path plus image, but no visible attachment marker confirms it | [Composer](evidence/live/image-path-composer-80x24-ansi-dark.png); source/live attachment status needed. |
| L08-013 | Loading/retrying banner | Spinner/status inert; Esc/Ctrl+C may interrupt main work; Enter may queue composer | [Loading](evidence/live/loading-retrying-80x24-ansi-dark.png); focused action context pending. |
| L08-014 | `/help` command list | Highlighted slash words look like references, not clickable controls | [Help](evidence/live/help-command-80x24-ansi-dark.png); hit test pending. |
| L07-006 | Edit result diff lines | Read-only information; no approval buttons visible | [Diff](evidence/live/edit-result-diff-80x24-ansi-dark.png); does not cover edit-approval route. |
| L03-034 | MCP Search hint/focus | Screenshot lacks search caret despite `Tab Search` hint | [After Tab](evidence/live/mcp-tab-search-80x24-ansi-dark.png); live F-022 shows Shift+Tab reaches field. |
| L02-094 | Settings detail F1 help | Bounded later hypothesis: F1 toggles help; setting rows persist | [Detail](evidence/live/settings-detail-80x24-ansi-dark.png); keyboard/help test pending. |
| L02-095 | Settings no-match row | Inert status but cursor/Enter Open hint implies action | [No match](evidence/live/settings-no-match-80x24-ansi-dark.png); F-024. |
| L02-096 | Settings enum radio | Selected star and focus separate; Enter Accept looks like draft acceptance | [Enum](evidence/live/settings-enum-80x24-ansi-dark.png); actual disk-write outcome F-002 was known to this reader. |
| L02-097 | Settings checklist tool boxes | Space/click likely toggles draft; pattern-controlled rows visually similar | [Checklist](evidence/live/settings-checklist-80x24-ansi-dark.png); exact box/control states pending. |
| L02-098 | Settings checklist Add Pattern / Apply | Add opens text editor; Apply persists collection | [Checklist](evidence/live/settings-checklist-80x24-ansi-dark.png); keyboard/mouse/durability pending. |
| L02-099 | Settings invalid timeout | Error inert; input fixable; Enter appears immediate save, Esc cancel | [Invalid](evidence/live/settings-invalid-80x24-ansi-dark.png); correction/return test pending. |
| L03-035 | OAuth Show URL / Copy | Shown URL appears informational; Copy has explicit target | [Show URL](evidence/live/oauth-show-url-80x24-ansi-dark.png); URL click/link test pending. |
| L03-036 | OAuth failed / retried | R likely retries authorization; Backspace/Esc closes | [Failed](evidence/live/oauth-failed-80x24-ansi-dark.png), [retried](evidence/live/oauth-retried-80x24-ansi-dark.png); images do not prove transition. |
| L03-037 | Proxy discard modal | Cancel/Esc expected preserve draft; Discard expected revert proxy fields only | [Modal](evidence/live/proxy-discard-80x24-ansi-dark.png); fake Pilot confirmed cancel path, persistence scope pending. |
| L03-038 | Full-host model picker Active/Default | Applied-state marker appears inconsistent with shell banner in fixture | [Full host](evidence/live/model-picker-full-host-80x24-ansi-dark.png); fixture/session state must be checked before claiming product defect. |

The [action-required question](evidence/live/action-required-question-80x24-ansi-dark.png) shows the generic loading hint and question-specific footer together. Its source-backed conflict is L08-012/F-028; it was discovered after the blind baseline and is not scored as a blind prediction.

## Runtime reconciliation queue

The live track reported the following effects after the blind predictions above were logged. These are `R` for the tested fixture/action only, not blanket route passes. A component host may lack the production screen's dismissal or persistence behavior.

Additional source/capture control **L08-012** is the generic loading hint behind a focused action-required QuestionApp. It appears in [this full-shell PNG](evidence/live/action-required-question-80x24-ansi-dark.png) and advertises `Enter queues next turn` while QuestionApp source binds Enter to accepting an answer. This was found after the blind baseline; it is **not** a blind prediction or runtime-confirmed combined key effect. See F-028.

| Control ID | Tested action and actual effect | Difference from blind prediction / remaining gap |
| --- | --- | --- |
| L03-001 | Model Default row click or Enter emitted `ModelSelected('')` immediately. | Consistent with selecting configured default; production dismissal and save-failure route pending. |
| L03-002 | Single click on row 1 emitted `ModelSelected('mistral-large')` and moved highlight; click at left padding x=0 did nothing. | Blind whole-row/padding target prediction failed. Host intentionally retained picker, so close behavior untested. |
| L04-001/002 | Click on session row 1 emitted `SessionSelected('local-session-0002')` and changed highlight. Repeated click/Enter emitted nothing while `selection_pending` stayed true. | Initial resume prediction partly supported; pending gate is a distinct state. Production resume/dismissal pending. |
| L04-005 | After `d` opened delete confirmation, Escape hid it but focused hidden Confirm Delete; Down/Enter no longer navigated or selected a session. Mouse Cancel does restore list focus. | Blind return-to-picker focus prediction failed, F-020. |
| L02-001–014 | Settings OptionList single body click moved cursor only; double click committed selected boolean; left x=0 padding and footer click inert; Enter committed boolean. | Blind box/row toggle prediction is not uniformly correct. Need per-row scope and state tests. |
| L02-039/040 | One click on provider row body or left padding opened Actions; a second click at the same coordinate opened Protocol from the new view. | Padding is active here; double-click can activate an unrelated next-screen control, F-019. |
| L02-055/056 | Provider model row body and padding click toggled inclusion; Enter opened detail. Escape restored model list and selection; resize to 80×48 retained them. | Blind split mouse-target prediction failed, F-007. Keyboard opener/resize behavior passed in tested case. |
| L02-066 | Clicking the visually blank first confirmation row dismissed it with no save. | The row is a hidden Cancel action despite visible Trust below, F-014. |
| L03-020 | Enter with fake browser opener returning `False` showed `✓ Opened in browser`. | False success, F-001; manual URL recovery not yet tested. |
| L03-018 | Five spaced mouse clicks and five Enter presses on retained Proxy Apply each emitted a saved close message. | Duplicate component activation, F-003; production write count pending. |
| L05-001/002/003–005 | Question single-answer click at left edge submitted; multi-answer click at left edge toggled. | Whole-row click works for these tested rows; Other, Submit, and modal host pending. |
| L05-008–010 | Mouse click on visible second trust option did nothing; Down selected it. | Blind click prediction failed, F-018. |
| L07-002 | Clicking expanded reasoning body collapsed it after header had expanded it. | Blind header-sized disclosure target prediction failed, F-016. |
| L01-003–006 | Clicking a visible slash completion row left the selected option and composer focus unchanged. Keyboard Down changed selection and Enter dismissed popup. | Blind clickable-row prediction failed, F-021. |
| L03-010 | On full-app `/mcp`, three Tab presses left focus on MCP options despite `Tab Search` hint; mouse click on search focused it, and typing updated `_query`. | Blind keyboard-search prediction failed, F-022. Disabled heading click was inert; server click opened detail; Escape returned to list. |
| L04-008–010 | Clicking `Edit without restoring files` body in an 80×24 RewindApp widget host left action step and selected row unchanged; keyboard Down/Enter advanced and eventually emitted `RewindConfirmed`. | Blind clickable-row prediction failed, F-023. Full-host mouse and Escape routes pending. |

Passing checks from the offline Pilot track are bounded: question single/multi row left-edge clicks acted; proxy dirty Escape opened scoped Discard and a second Escape backed out with draft intact; long question scrolling reached Submit; proxy Tab reached footer actions. Full-shell `/thinking` applied a selected level or canceled to composer; `/theme` previewed on cursor movement, retained preview/focus through 80×24→80×48 resize, reverted on Escape, and persisted on Enter; `/loglevel` draft/dirty-discard/return and Enter apply paths worked in fake state. AgentBar header and row clicks opened browser/selection; transcript PageUp/End changed follow mode; footer clicks were inert. Rewind q/Escape and keyboard submission, consequential Exit Enter/Escape, composer queued-input keyboard states, patched attachment link opening, F1 Help, inert banner/loading clicks, Settings inline resize/cancel, MCP filter resize/clear, and question selection resize were exercised. The existing `test_lazy_copy_and_click_does_not_fold` Pilot test passed (1/1): Copy full output clicked and focused Enter each copied once without folding the parent output. These are tested fixtures/actions, not whole-route closure. Remaining live limits: real browser/credential OAuth outcome, irreversible rewind commit, OS attachment opening, production proxy multi-write side effect, full queue mouse geometry, and native-terminal pointer behavior beyond Textual Pilot.

No family is complete. Per-control inventory for source-only templates, exact save scopes, mouse padding/disabled/inert targets, focus after transitions, and the viewport/theme/state matrix remain open. The [finding register](findings.md) records discrepancies without replacing the original blind predictions.
