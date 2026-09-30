# Chartreux TUI flow and consistency repair

**Status:** Completed
**Date:** 2026-09-29

## Purpose and design

Fix the ten reported TUI issues and systematically find analogous defects across the interface. Treat the reports as symptoms of shared navigation, focus, layout, and editing rules; repair the shared behavior where practical.

Use an explicit view enum and an opener-frame stack. Each frame preserves presentation context (view, focus, selected ID, scroll position, and filter); `ManagementState` owns long-lived mutable provider draft state. Escape dismisses a local filter, editor, help panel, or confirmation first, then returns to the actual opener. New provider setup is one continuous draft flow: Connection → Models → optional Model details → Models → Save, with no Escape/Tab detour. Pending models stay in the draft and never lead to a dead-end global route. Model filtering has its own control and applies only to the model list; Enter opens the model editor directly or the deployment picker. Scalar editors follow one consistent pattern.

All six roles remain visible at 120×36 or larger through automatic height and outer scrolling. “Active” is limited to usable orchestrator members: select `@orchestrator` automatically when it is the sole usable member, and provide an explicit repair path when there are zero usable members or an invalid pin. Restore focus after successful repair. The slash popup is wider and wraps selected-item help. Fix settings bleed at the render boundary. Audit analogous no-op state rows and clipped MCP descriptions.

## Work packages

- [x] **0. Baseline and reproduction matrix.** Recorded the ten reports and reproduced representative flows at 80×24, 120×36, and 120×72. Pilot and snapshot coverage includes no-color output and ANSI/ASCII borders.
- [x] **1. Navigation stack and focus.** Implemented explicit views and opener frames, local Escape precedence, and presentation-context restoration. `ManagementState` retains mutable provider drafts. Focus and immediate Up/Down behavior are covered by focused provider checks and snapshots.
- [x] **2. Provider draft, catalog, and active roles.** Provider setup and pending models remain in one draft through Save; model filtering is separate; Enter routes to the editor or deployment picker. Scalar editors and active-role repair paths are covered by the provider checks.
- [x] **3. Layout, editing, slash popup, and settings.** All six roles remain accessible at the required size; slash/help layout and settings render-boundary behavior are covered by the focused checks and snapshots.
- [x] **4. All-surface consistency audit.** Audited MCP, OAuth, session, trust, question, and proxy surfaces for state rows, navigation, and editing consistency. No-color and ASCII-border cases were included. The 329-character MCP description already wraps in its view, so no code change was needed for it.
- [x] **5. Verification, snapshots, and review.** Focused behavior checks passed; the full snapshot suite passed with 152 tests. Whole-repository `ruff check .` and `ruff format --check .` passed across 1,050 files; repository `typos` and `git diff --check` passed. Astra/code review found no remaining actionable defect.

Packages 0–5 are complete. The provider-focused suite reports 92 passing tests and 10 snapshots. The declared pre-commit command was attempted but blocked before hooks ran when GitHub DNS resolution failed while fetching action-validator. The broader non-snapshot suite was also attempted and environment-blocked by a sandbox asyncio wakeup issue in unrelated ACP/agent-loop tests; it is not a passing check.

Likely implementation areas include `chartreux/ui/providers/workbench.py`, `chartreux/cli/textual_ui/widgets/chat_input/completion_popup.py`, `chartreux/cli/textual_ui/screens/settings.py`, shared pickers and MCP views, relevant tests, and `scripts/capture_tui.py`. Refine this list during the baseline inventory; keep each work package independently checkable and serialize changes that share navigation or focus interfaces.

## Acceptance criteria

Baseline and acceptance matrix (observed pilot/snapshot evidence at 80×24, 120×36, and 120×72; no-color and ANSI/ASCII border coverage is included):

| # | Implemented surface | Observed evidence |
|---|---|---|
| 1 | Explicit view/opener-frame navigation and focus restoration | Provider focused suite; pilot and snapshots |
| 2 | Local Escape precedence for filters, editors, help, and confirmations | Provider focused suite; pilot and snapshots |
| 3 | Continuous provider setup draft through save | 92 provider tests; 10 snapshots |
| 4 | Pending models retained in provider draft | 92 provider tests; 10 snapshots |
| 5 | Separate model-filter control scoped to model list | 92 provider tests; 10 snapshots |
| 6 | Direct model-editor and deployment-picker entry routes | 92 provider tests; 10 snapshots |
| 7 | Consistent scalar editors and focus restoration | Provider focused suite; pilot and snapshots |
| 8 | Six-role layout and active-role repair paths | Provider focused suite; 80×24, 120×36, and 120×72 pilot/snapshots |
| 9 | Wrapped slash/help content and settings render boundary | Slash/help and settings focused checks; snapshots |
| 10 | MCP descriptions and actionable state rows across audited surfaces | MCP/OAuth/session/trust/question/proxy audit; snapshots; the 329-character MCP description already wraps |

The matrix maps each report to its implemented surface and observed pilot or snapshot coverage. The broad non-snapshot suite was attempted but environment-blocked as described above. The ten numbered outcomes are:

1. Navigation returns to the actual opener and restores its state.
2. Escape dismisses the innermost local filter/editor/help/confirmation before leaving its parent view.
3. Provider setup remains a continuous draft through connection, model selection, optional details, and save.
4. Pending models remain available in the draft and do not route to a dead end.
5. Model filtering is separate from list navigation and only filters models.
6. Enter opens the model editor directly or opens the deployment picker as appropriate.
7. Scalar editors use consistent controls, commit/cancel behavior, and focus restoration.
8. All six roles are reachable at 120×36 or larger; active roles are usable orchestrator members, with automatic sole-member selection and an explicit zero/invalid-pin repair path.
9. The slash popup wraps selected-item help, and settings content does not bleed beyond its render boundary.
10. Required MCP descriptions and state-row actions remain visible and actionable, with analogous defects found by the all-surface audit repaired.

Across all ten reports and audit findings: every action is keyboard reachable; transitions focus a primary enabled control; Up/Down work immediately; Escape restores the exact opener state; saving requires no backtracking; Enter never silently does nothing; required text is not clipped; and finite content that fits has no scrollbar. Preserve drafts on local cancellation and keep destructive actions behind clear confirmations.

## Risks and working notes

- The working tree has a broad dirty diff from prior work. Preserve it and limit each change to the owned behavior; do not reset unrelated edits.
- Refresh snapshots only after behavior tests pass so snapshots capture intended behavior.
- No commit is planned unless requested.

## Second-round follow-up - 2026-09-29

The provider and neighboring TUI review closed six provider issues: deployment-picker resize no longer changes the active model; synthetic Models actions retain their cursor and Enter routes; deployment picker title/help describes editing; provider-filter cursor state survives resize independently of the applied filter; draft providers appear in catalog filters; and a busy save shows truthful minimum-size guidance through shrink/restore without clearing busy state. The LogLevel discard confirmation now hides its option list and restores the draft, row, badge, focus, and help when backed out. The slash completion popup recalculates its width when its composer resizes without rebuilding rows or changing selection/focus.

Focused evidence: 117 provider workbench tests, 14 LogLevel/composer and adjacent focused checks, and all 152 snapshot tests passed. `ruff check`, `ruff format --check`, import-contract checks, and scoped `git diff --check` passed. Pilot geometry confirms the LogLevel confirmation fits at 80×24 and 120×36; the popup tracks 80×24 → 120×36 → 60×20 at widths 78 → 92 → 58 while retaining the selected row and input focus. Restricted-loop tests use the existing `install_snapshot_wake` fixture to let executor teardown complete in the sandbox.

## Provider setup and preset redesign - 2026-09-30

**Status:** The provider/preset redesign and the subsequent Web search onboarding extension are complete. Astra approved the original design, plan, source, and final onboarding captures. This section supersedes the earlier role-membership, Active picker, and onboarding journey targets above. Work remained on the existing feature branch; preserve all current user changes and do not commit.

### Product contract

The first-run journey is **Connect provider → Configure models → Add another provider or continue → Choose default presets → [Web search when needed] → Finish**. At the end of each provider's Models screen, show both forward actions explicitly. Saving a provider or its model selections must succeed when its definitions are structurally valid, even while presets are incomplete or credentials are unavailable. After presets, preserve ready automatic Mistral search or an already-ready explicit provider and skip the search editor. If search is not ready, the editor offers Exa, Brave, and DuckDuckGo only; standalone Settings shows Mistral once while retaining `auto` as a Mistral configuration alias. Readiness uses the configured credential variable through the credential resolver; catalog presence alone is insufficient. **Skip for now** leaves existing search settings and tool enablement unchanged. Finish checks that required presets reference configured, runnable deployments and valid thinking levels, and gives a direct repair action for each failure; a shown Web search editor checks configuration/key readiness without testing connectivity. A saved provider stays saved when the user continues, adds another provider, or reopens Settings. Backtracking is only for changing an earlier choice.

Provider Settings in management mode uses the same provider, model, and preset editors without forcing the first-run journey. Every screen identifies its primary next action, current saved/draft state, and what Esc returns to. Up/Down move among rows, fields, and actions immediately; Enter edits the selected field or activates the selected action; Space toggles a checkbox; Esc dismisses the local editor/confirmation before returning to its opener; Tab is optional. Within a text field, arrow keys keep normal text-editing behavior and Enter commits the field before screen navigation resumes. Focus and opener identity survive resizing, save, cancellation, and return. Dirty Escape presents a scoped save/discard choice only when needed; it is never the required way to proceed.

Each role is one named default preset: **one canonical model plus one thinking level**. Multiple roles may select the same model at different levels. There are no ordered role members and no alternative-model fallback during role resolution or subagent launch; an unavailable role produces a clear repair error. Explicit spawn thinking overrides its role preset. Resolve and validate the chosen pair through catalog resolution, launch materialization, committed subagent identity, and child configuration so it cannot be lost when a model is shared by roles. Preserve the existing deployment-selection rule *within the selected canonical model* as a separate concern; do not select another canonical model as a fallback. Never skip a configured default merely because credentials are missing: report the unavailable credential and guide repair.

The `orchestrator` preset is the single persisted default for the main assistant, including its thinking level. Remove the separate persisted `active_model` choice and Active picker. `/model` and `/thinking` may make session overrides, which do not silently rewrite the preset; the UI must label these as current-session choices. `compaction_model` remains a separate function and follows the chosen main model when left empty. Reject old role `models = [...]` definitions and old persisted Active configuration with actionable errors that identify the new preset fields; this pre-release fork does not require automatic migration. Existing `fan_out: true` depends on ordered role lists: retire that behavior and reject legacy fan-out requests with guidance to launch separate explicit tasks. No silent singleton expansion or implicit replacement fan-out.

### Implementation sequence and file ownership

1. **Catalog and configuration contract.** Update `chartreux/core/model_catalog/{schema,defaults,resolver,loader}.py` and the bundled catalog resource so each shipped and user role has `model` and `thinking`, with clear validation for missing/invalid pairs and legacy lists. Update `chartreux/core/config/chartreux_schema.py` to derive the main default from `@orchestrator`, apply session overrides only in their session scope, and remove persisted Active and role fallback logic. Update `chartreux/core/system_prompt.py` to describe selected presets. Convert catalog/config fixtures in `tests/core/config/{test_model_catalog,test_config_resolution}.py` and resolver tests; cover two roles using one model at different thinking levels, missing credentials, invalid levels, and old-schema errors. This is the dependency for runtime and UI work.
2. **Spawn and session runtime.** Update `chartreux/core/agents/{registry,launch}.py`, `chartreux/core/subagents.py`, and committed subagent configuration. Keep the role's model/thinking pair intact after materialization; an explicit `config.thinking` wins; validate the final pair against the selected deployment. Update `chartreux/app_server/{_sessions,_projector,_session_model,_runtime_resources,_config_write}.py` and relevant session logger/call sites so `/model` and `/thinking` remain session-scoped and never persist a second main default. Remove advertised fan-out and role-list expansion, while a legacy supplied `fan_out: true` produces the actionable error “Roles are single presets. Launch separate tasks with explicit presets/models for multiple agents.” Check result leases and projector effects before retiring dead paths. Update `tests/core/agents/test_launch.py`, `tests/app_server/test_subagents.py`, and relevant task, app-server, session, and projector tests, including committed/background launch and shared-model/different-thinking cases. Begin after the catalog contract settles.
3. **Provider state and persistence.** Update `chartreux/ui/providers/{contracts,management_state}.py` to hold preset edits directly, generate patches with model/thinking, and split provider/model structural save validation from Finish readiness. Keep credential storage independent and current catalog deployment selection intact. Cover partial setup, second provider, invalid required preset, and saved changes in `tests/ui/providers/{test_contracts,test_management_state}.py`. Begin after catalog contract settles; coordinate state interfaces before the screen edit.
4. **Screen flow and keyboard contract.** Give one owner `chartreux/ui/providers/workbench.py` and its focused UI tests; do not parallel-edit that file. Replace role checkboxes/completion and Active picker with a preset step and model-plus-thinking editor. Make Connection, Models, Add another/Continue, Presets, and Finish explicit forward actions; management entry stays direct. Normalize the current split list/form/filter/action focus so Up/Down are sufficient and Tab optional, while preserving text-field arrows and opener restoration. Remove help text that instructs Tab or Shift+Tab as the only route. Update `tests/ui/providers/test_workbench.py`, `tests/e2e/test_cli_tui_first_run_onboarding.py`, and `tests/snapshots/test_ui_snapshot_provider_workbench.py` after the behavior is stable. Begin after state interfaces are agreed.
5. **Bulk compatibility, documentation, and visuals.** Separately convert remaining old role-list fixtures and references across tests once the schema is stable, excluding tests owned by packages 1–4; the bundled catalog stays with package 1. Revise `PROVIDER_SETTINGS_SPEC.md`, `PROVIDER_SETTINGS_MOCKUPS.md`, `DESIGN.md`, `docs/getting-started.md`, `docs/guides/{models,subagents}.md`, `docs/reference/configuration.md`, relevant migration/help text, `chartreux/core/prompts/cli.md`, `chartreux/core/tools/builtins/prompts/task.md`, and `chartreux/skills/main-review/SKILL.md` plus other bundled skill instructions that rely on ordered roles or fan-out. Replace obsolete role-membership/Active mockup frames in `docs/assets/provider-settings/` with frames for the new journey at 80×24 and 80×48. Documentation must describe the implemented behavior, including explicit errors and session overrides. This package may proceed in parallel on disjoint files after the final schema/flow wording is settled.
6. **Integration, verification, and review.** Resolve fixture and API fallout across core, setup, UI, task/session, and snapshots. Run focused tests before snapshot refresh; then full repository pytest, `ruff check .`, `ruff format --check .`, `pyright`, `typos`, import-contract checks, and `git diff --check` (plus declared pre-commit hooks if available). Record executed results and environment blocks precisely. Conduct an independent final code review after verification and fix actionable findings. No commit without an explicit request.

### Acceptance matrix

| Journey or case | Behavior evidence | Visual evidence |
| --- | --- | --- |
| One provider, one model | Connection → Models → Presets → Finish, with saves forward and no return to prior screen | Dark and light, 80×24 and 80×48 |
| Add a second provider | Saved first provider stays saved; Add another opens Connection; second Models leads to Presets | Dark and light, 80×24 and 80×48 |
| Incomplete or unavailable preset | Provider/model save succeeds; Finish names the exact preset/credential/thinking problem and opens its repair route | Dark and light, 80×24 and 80×48 |
| Keyboard and nested editor | Arrows reach every row/action; Enter edits/activates; Space toggles; Esc restores opener; Tab never required; text arrows remain local | Dark and light, 80×24 and 80×48 |
| Preset and session semantics | Same model can serve distinct role thinking levels; explicit spawn/session override wins in its scope; persisted `@orchestrator` remains the sole main default | Preset editor and main-default status, dark/light |
| Legacy and fan-out input | Old lists, persisted Active, and `fan_out: true` fail with actionable guidance; no alternative-model routing | Error/repair frame where visible |

Capture and inspect live 80×24 and 80×48 screenshots in dark and light themes for every applicable journey, then update stable snapshots. Exercise no-color and ASCII-border modes where the existing pilot supports them. Verify that primary actions, focused row, selected value, unsaved state, validation text, and Esc destination remain legible without clipping or misleading scrollbars.

### Execution checkpoint - 2026-09-30

Packages 1–5 are implemented; package 6 awaits the final full-repository verification result. Astra reviewed the final source and reported no P1/P2 findings. Catalog roles now carry one `model` and `thinking` pair, with no alternative canonical-model routing. Old role lists and persisted `active_model` or `thinking_overrides` in user/project configuration receive actionable errors; session overrides remain runtime-only. Explicit spawn thinking still wins. Fan-out role expansion is retired; a legacy `fan_out: true` call directs users to launch separate tasks.

Observed focused checks so far: core 63, provider state 26, runtime 185 plus 15 CLI/backend and 3 session cases, bulk compatibility 248 plus 6 integration cases, public PTY onboarding 1, context/keyless 19, and provider snapshots 10. Final migrated workbench test count and full-repository runner are pending; do not treat this checkpoint as a full-suite pass. The planned seven-package offline wheelhouse check was unavailable, so record its exact limitation with the final verification result rather than inferring a pass.

Live 80×24 and 80×48 dark/light runs passed the forward two-provider journey, selecting both providers' models, arrow navigation through model detail (including Temperature and Image support), save, presets, and Finish. Missing-credential/failed-save repair passed at 80×24 in dark and light themes. Management supported-thinking filtering and a no-color happy path also passed. The current mockup index distinguishes model Default thinking from each role preset's thinking and retains prior untracked SVG explorations as historical artifacts.

The first full-suite run reported **11,672 passed, 16 failed, 2 skipped**, with seven offline fixture cases excluded because their wheelhouse was unavailable. At that checkpoint the failures were ten legacy persisted-Active fixtures, two committed-thinking expectations, three model-picker snapshots, and one worker end-to-end timeout. Whole-repository Ruff check and format check (1,051 files), Pyright (zero errors), typos, and diff whitespace checks passed. The final rerun and resolution are recorded below.

### Final verification - 2026-09-30

Packages 1–6 are complete. The final unified verifier run passed **11,688 tests**, with **2 skipped** and **6 warnings**, in 103 seconds (`/tmp/chartreux-preset-full-final3.log`). Seven offline wheelhouse cases were excluded because their required wheelhouse was unavailable; the result is a pass for the executed suite, not a claim that those seven cases ran. The earlier 16 failures were resolved, and the full rerun passed. Global Pyright reported zero errors; Ruff check and format passed across 1,051 files; typos and `git diff --check` passed. The frozen pilot passed 12 scenario/theme/size combinations at 80×24 and 80×48 in dark and light themes, plus a no-color path. Documentation verification parsed 16 SVGs and validated six current mockup links. Astra's final source review reported no P1/P2 findings. No commit was made.

## Exit and defaults refinements - 2026-09-30

**Status:** Complete. Refinement scope approved through user feedback, explicit exit-flow approval, and Astra advice. Keep the current branch and existing changes; do not commit.

Make typed `/exit` immediate when idle, keep its confirmation dialog for consequential active work, and let `ask_confirmation_on_exit` govern only idle Ctrl-C/Ctrl-D quits. Replace the six shipped role aliases with `orchestrator`, `large`, `medium`, and `small`; keep Worker and Reviewer profiles bound to `medium` and Advisor to `large`. Remove theme selection from first-run onboarding because `auto` is already the default. Update owned docs, prompts, skills, fixtures, and focused behavior checks to match.

Observed refinement evidence: 117 focused provider-workbench tests, 328 core tests, and 88 exit-focused tests passed. The live preset matrix passed for all four roles with 121 available models and an empty environment. The migrated public first-run PTY E2E passed (1 test), and Astra found no P1/P2 findings. The final suite passed 11,693 tests, with 2 skipped and 7 offline-fixture cases deselected because the wheelhouse was unavailable (103.34 seconds). Ruff check and format passed across 1,050 files; Pyright reported zero errors and warnings; typos and `git diff --check` passed.

## Whole-TUI visual affordance review - 2026-09-30

**Type:** research / refactor

**Status:** Bounded finding-scoped remediation and independent acceptance complete: 27 registered candidates have implemented, focused-verified repairs and F-011's original palette claim was withdrawn. The committed baseline is `37abc9d`; before and after offline evidence remains under `docs/reviews/tui-affordances/`. Full route/state and native-terminal coverage remain incomplete, so this status does not claim a completed whole-TUI review or adoption of the proposed contract into `DESIGN.md`.

**Goal:** A user should be able to predict what activating each visible TUI control will do, where it will take them, and when a change takes effect. Review the whole TUI, including informational elements that look interactive and interactive elements that look inert. Record the actual behavior before choosing a common visual language. Use `DESIGN.md` as the current design baseline; review findings may propose changes to it but do not silently amend its contract.

### Proposed contract to test

Use precise action verbs and a small set of shared shapes for these distinct outcomes: navigate, edit inline, open an editor or detail view, accept a draft, save while staying, save and continue, apply to the current session, toggle or choose one radio value, expand or collapse, open externally, copy, run, and confirm a scoped destructive action. Distinguish informational text from a mouse target. A disabled control must retain its action identity and show why that action is unavailable; informational text must not rely on disabled-control styling. A forward arrow must not resemble a cursor or focus glyph. Color must never be the only cue. Visible mouse targets, including padding, must match hit areas; focus, selection, and activation must remain distinguishable. Show whether a value is a local draft, a persisted setting, or a session change, and make cancellation, pending effects, and unsaved state explicit. Treat these as hypotheses to validate against real behavior and user prediction, not as preapproved implementation rules.

### Phase 0: Establish the complete baseline

- [ ] Enumerate every route, screen, modal, transient overlay, shared widget, command entry point, and meaningful state from the route registry and source inventory. Reconcile the inventory with existing screenshots and tests so omissions are visible. Start with `chartreux/cli/textual_ui/screens/`, `chartreux/cli/textual_ui/widgets/`, `scripts/capture_tui.py`, and `tests/snapshots/`; follow routes beyond those anchors.
- [ ] Build a surface ledger with one row per visible control or affordance, including informational elements that might be mistaken for controls. Record source path/route, shared widget, action class, label/glyph/shape, keyboard behavior, mouse hit area and padding, focus and selection behavior, persistence or draft scope, state variants, and reproducible fixture/evidence link. Give every control a stable ID. Explicitly record inert elements and any exceptional behavior.
- [x] Seed a repeatable, offline, fake-fixture capture harness from the existing capture script and snapshot setup. Existing temporary capture artifacts may seed investigation, but lasting evidence and reproduction instructions must live in the repository review artifacts. Do not use generated art as evidence of TUI behavior.

### Phase 1: Four independent review tracks

Run the tracks separately and record observations before reconciliation, so a behavior explanation cannot bias the visual prediction.

| Track | Question and evidence | Assignment |
| --- | --- | --- |
| Blind visual reading | From screenshots alone, predict each visible element's action, destination, persistence, and reversibility; record ambiguity before seeing the behavior oracle. | Vision-capable reviewer independent of implementation; GPT-6 Astra low for independent design review. |
| Live interaction | Use mouse and keyboard at the actual target and its padding; distinguish focus, selection, activation, return route, and pending transition. Record expected versus observed transitions. | GPT-6 Sol medium execution/review workers. |
| Source and state | Trace handlers for draft, save, session apply, cancel, disabled, error, pending, retry, and destructive scopes; distinguish save failure from a successful save followed by runtime reload failure, including truthful feedback and recovery. Reconcile visible promises with actual effects. | GPT-6 Sol medium execution/review workers. |
| Compact and accessible presentation | Inspect clipping, scrolling, hierarchy, text cues, and reachability across viewport and palette variants. | GPT-6 Sol medium execution/review workers; GPT-6 Luna high for mechanical inventory, captures, and indexing. |

Cover these surface families, including their shared components and routes:

1. Composer, slash commands, path completion, attachments, and queued input.
2. Settings, model and thinking pickers, theme, provider workbench, and role presets.
3. MCP setup, OAuth, external authorization, and proxy setup.
4. Questions, permissions, approvals, edit approval, and quit or exit dialogs.
5. Subagents, agent bar, task status, and nested transcripts.
6. Sessions, history, resume, preview, rewind, and fork.
7. Transcript tools, reasoning, collapse/disclosure, virtualization, copy, and links.
8. Help, updates, errors, loading, status notices, and debug views.
9. Onboarding, authentication, trust, and first-run repair routes.
10. Shared navigation, focus, mouse hit areas, modal chrome, footer, and banners.

For each distinct surface, capture actual TUI evidence at 80×24 and 80×48 in dark and light themes. Exercise normal, focused, selected, disabled, empty, dirty, pending, error, long or scrolled, resized, and return states wherever the surface supports them. Include no-color and ASCII chrome/glyph fallback paths for the canonical shared conventions and for exceptions, plus a representative wide layout. A full Cartesian matrix is unnecessary; document risk-based omissions and why they cannot hide a distinct behavior. Test every consequential exception individually. Compare blind prediction with the live outcome for every actionable element, and test both valid mouse targets and inert information. For disabled controls, verify that the action and reason unavailable remain clear without making information look disabled.

### Phase 2: Reconcile and prototype a contract

- [x] Merge independent findings by stable control ID, distinguish unique defects from shared-widget causes, and assign severity by wrong consequence, lost work, unreachable action, misleading state, or cosmetic ambiguity. Preserve disagreements and evidence rather than averaging them away.
- [x] Draft a small, named affordance vocabulary with exact verbs, shape/glyph roles, focus/selection treatment, persistence and draft cues, mouse rules, and allowed exceptions. Prototype it in three mixed surfaces: settings, provider details, and an approval or picker flow at both 80×24 and 80×48. Verify blind prediction again before accepting a shape or label.
- [x] Obtain an independent GPT-6 Astra design review of the proposed contract and prototypes. Resolve blocking objections and record any intentional exceptions with their reason and owner before scheduling implementation.

### Phase 3: Prioritize and implement bounded remediation

- [x] Group findings into reviewable batches by shared root cause. Fix shared widgets and terminology first; serialize changes to shared focus and mouse infrastructure; then handle individual surface families. Each batch must name affected controls, behavior contracts, owners, tests, and captures.
- [x] Obtain separate authorization for finding-scoped source and test changes. Astra advised bounded implementation after the independent proposal review; the user authorized this implementation turn. Continue to treat the proposal as unapproved `DESIGN.md` text until acceptance.
- [x] Implement independent batches with explicit file ownership: the provider workbench owner handles `chartreux/ui/providers/workbench.py` and its direct tests/captures; separate owners handle Settings, model/thinking picker retry, MCP/OAuth/proxy, and shared transcript/composer/navigation controls. Serialize shared click/focus changes and preserve each owner's edits.
- [x] Apply the agreed scope decisions: Settings enum choices state that Enter saves immediately; a provider double-click gesture cannot activate a newly exposed control, while a later deliberate click still works; failed model/thinking saves retain the picker and the selected draft for direct retry. Keep confirmation and credential scopes explicit.

### Phase 4: Independent acceptance after remediation

- [ ] Repeat blind screenshot prediction, then verify live mouse and keyboard behavior against the same ledger. Revisit every fixed finding, shared-widget consumer, and consequential exception, including disabled actions and informational text. Capture regression evidence across the viewport/theme matrix and named no-color/ASCII chrome and glyph fallback cases.
- [x] Run relevant focused source and interaction tests plus declared repository checks. The pre-review baseline is 11,693 passed, 2 skipped, with 7 offline wheelhouse cases excluded; record any continuing exclusions explicitly instead of inferring their result.
- [ ] Obtain independent GPT-6 Astra and vision acceptance reviews. Close only when every route and control is accounted for, every exception has a named reason, and no unresolved wrong-consequence prediction, dead mouse target, critical clipping, unreachable action, lost draft, duplicated pending effect, or focus/selection conflation remains.

**Artifacts:** Keep the durable review under `docs/reviews/tui-affordances/`: a surface ledger, finding register, proposed contract and exceptions, evidence index, and reproducible offline fixtures/capture instructions. Each finding needs an ID, severity, expected and observed behavior, source anchor, PNG evidence, owner, and verification result. Every capture records the source commit, route and fixture, viewport, theme or terminal mode, and actions taken. Screenshots should come from the running TUI; temporary files alone are insufficient evidence.

**Dependencies and risks:** Begin from the committed post-refinement baseline (`37abc9d`). Live capture must not rely on real credentials or external services. Shared widgets can conceal omissions, so reconcile route inventory, source controls, and screenshot states before declaring coverage. Focus and mouse infrastructure can affect unrelated screens, so implement those changes in serialized batches with broad regression capture. Keep behavior and visual reviewers independent until predictions are logged.

**Initial execution checkpoint:** The bounded review recorded blind predictions before live reconciliation for the supplied baseline fixtures, then source-only route templates and later post-oracle hypotheses separately. The current [surface ledger](docs/reviews/tui-affordances/surface-ledger.md) has 10 route families and 18 source control templates; the [finding register](docs/reviews/tui-affordances/findings.md) has 28 open candidates (20 P2, 8 P3, no P1), of which 14 have some live behavior evidence and the others remain source/visual hypotheses. The [evidence index](docs/reviews/tui-affordances/evidence-index.md) points to an offline manifest with 397 records and 82 fixture IDs at this checkpoint. Sixteen no-color PNG exports were visually invalid and do not support contrast conclusions. An independent visual review and Astra reviewed the initial [contract proposal](docs/reviews/tui-affordances/contract-proposal.md) and [text prototypes](docs/reviews/tui-affordances/prototypes.md); blocking proposal ambiguities were revised, but the proposal is not approved as `DESIGN.md` or implemented. Repeat blind acceptance, full route/state coverage, production/native-terminal checks, and remediation are pending. This checkpoint is review evidence only, not a product test pass.

**Bounded remediation and acceptance checkpoint:** The [finding register](docs/reviews/tui-affordances/findings.md) now has a per-ID disposition matrix: 27 implemented repairs passed focused verification and F-011's original dim-Other claim was withdrawn as an invalid early palette artifact. The frozen before manifest has 397 records; the [after manifest](docs/reviews/tui-affordances/evidence/after/captures.jsonl) has 210 valid SVG/PNG pairs across 51 fixture states and 29 route labels, with 80×24/80×48 ANSI dark/light for every state plus six provider ASCII variants. The after [capture README](docs/reviews/tui-affordances/evidence/after/README.md) states the synthetic/offline limits. Independent Astra and visual acceptance found no blocker within this bounded finding set. Focused Settings tests passed 95/95 with 15 snapshots; session tests passed 52/52 with two snapshots; rewind/completion tests passed 29/29; the independent snapshot suite passed 154 tests. The broad run passed 11,565 tests with one obsolete thinking fixture failure, two skipped, and seven unavailable wheelhouse cases excluded; a corrected thinking/session targeted rerun passed 78 tests. **The entire broad suite was not rerun after that fixture correction.** All nine declared pre-commit hooks passed using a writable temporary uv cache. Native terminal `NO_COLOR` contrast, real external OAuth/browser and credential/proxy outcomes, irreversible rewind and OS attachment effects, plus untested route/state combinations remain explicit coverage limits. This closes the bounded remediation and acceptance phase, not the whole-TUI review.

## Web search configuration - 2026-09-30

**Goal:** Let users configure web search in Settings while preserving the existing runtime default (`provider = "auto"`, which resolves to Mistral only). Keep credentials outside the search settings draft and report configuration readiness without claiming network connectivity.

### Ownership and scope

- **Backend (`websearch_backend`):** add a safe Settings read projection for supported `tools.web_search` leaves, effective/saved values and origins, readiness diagnostics, selected credential-variable name, and user revision. Validate edits at write time, apply changed search fields as one revision-checked user-config write, and preserve unrelated tools and permissions. Never expose secret values.
- **UI (`websearch_ui`):** add Settings > Web search and `/web-search`; support `auto`, Mistral, Exa, Brave, and DuckDuckGo; provide separate `Save API key` and `Save search settings` actions; show saved versus session-only credential outcomes; reset provider-specific environment-variable and endpoint overrides on provider switch and clear an unsaved key with an explicit cue.
- **Docs (`websearch_docs`):** update the command reference, tools/safety guide, configuration guide/reference, and this plan after behavior freezes. Preserve `auto` as the default and document that it means Mistral only, with no provider fallback.

### Acceptance checks

- Read projection includes only supported fields and safe readiness metadata; no resolved secret is serialized.
- Invalid search values and stale revisions cannot partially overwrite settings; unrelated tool values and permissions survive saves.
- Key persistence is a separate action. Saved credentials survive discarding an unsaved search-settings draft; session-only outcomes are stated plainly.
- Switching providers resets custom credential-variable and endpoint overrides to the new provider defaults and clears any unsaved key visibly.
- `/web-search` opens the dedicated Settings route; provider choices and relevant advanced fields match provider semantics.
- Readiness means configuration/key availability only; the UI performs no connectivity probe or implicit provider fallback.
- Focused backend/UI tests and compact route snapshots cover validation, revision conflict, partial save/application outcomes, key handling, provider switching, draft discard, and the default `auto` behavior.

### Acceptance results - 2026-09-30

Astra accepted the final source and visual review, including the light-palette and partial-retry states. The combined focused backend, UI, settings, and command run passed **177 tests**; the full snapshot suite passed **158 tests**. The final capture manifest contains **72 cases across 18 states**, each with paired SVG and PNG artifacts, covering 80×24 and 80×48 in ANSI dark and light. Captures use isolated fake services and do not verify real credentials or provider connectivity.

The full non-snapshot run reported **11,587 passed, 2 skipped, and 1 failed**: a stale `/web-search` title expectation. The corrected assertion passed in the later 177-test combined run, but the full suite was not rerun after that correction. Seven installed-contract cases were excluded because their offline wheelhouse was unavailable. Treat the focused and snapshot results as passing; do not describe the broad run as a clean full-suite pass.

### First-run Web search extension - 2026-09-30

**Status:** Complete; Astra accepted the source, documentation, and final
captures. This extends the completed preset journey without changing
`/web-search` behavior in standalone Settings.

After the user saves default presets, first-run onboarding opens the Web search
editor only when the current search configuration is not ready. Ready automatic
Mistral search and ready explicit provider choices are preserved without
opening it. A shown editor offers Exa, Brave, and DuckDuckGo, not `auto` or
Mistral fallback choices; standalone Settings shows Mistral once and continues
to accept `auto` as its configuration alias. **Save and finish** persists and
applies edited search settings before completion; **Finish setup** completes
without a settings write when readiness is already valid; **Skip for now**
leaves search settings and tool enablement unchanged; **Back to presets**
reopens saved preset choices. Unsaved search edits require an explicit discard.
A key saved separately remains saved after discarding the search draft.
Readiness uses configured settings and the credential resolver, not catalog
presence or network connectivity. The preset step suggests replacements only
from configured providers with a credential-resolver-ready credential, enabled
deployment, and supported thinking level; it does not suggest an unconfigured
catalog model as ready.

Acceptance checks passed for each forward/back/skip outcome, dirty-draft
discard, independently saved key retention, successful and blocked Finish, and
preservation of existing web-search configuration/tool enablement after Skip.
Preset suggestions exclude unconfigured or unready catalog models. Astra
accepted the final width-adjusted captures. The onboarding matrix contains **40
capture records across 10 states**; with the existing 72 web-search records,
the combined evidence contains **112 records**. The captures use fake services
and a local loopback fixture, not a public web-search provider.

Focused verification reported **193 scoped tests**, **19 additional focused
regression tests**, **14 onboarding snapshot tests**, and **1 first-run TUI
E2E** against an auto-approved local fake loopback. These results are listed
separately because their suites overlap; they are not an aggregate test count.
Strict MkDocs and the 47-test command-reference consistency suite passed. All
declared pre-commit hooks passed against changed and new paths, excluding the
local `.chartreuxhistory` artifact. Do not infer a full-suite pass from these
focused checks.

### Conditional Web search onboarding policy - 2026-09-30

**Status:** Complete; Astra accepted the source, docs, and versioned captures.
The independent policy run passed **173 tests**, including 16 snapshots, and
the first-run TUI E2E passed in 9.68 seconds. These results are separate from
the previously reported overlapping suites, not an aggregate. The current
versioned matrix contains **20 light/dark pairs across five states** and
asserts that the onboarding fallback contains only Exa, Brave, and
DuckDuckGo. The earlier Web Search and onboarding evidence remains historical:
72 plus 40 capture records, separately counted from the current 20 pairs.

After preset selection, preserve an already-ready search configuration and
skip the search editor. Readiness is determined using effective settings and
the credential resolver, including the configured credential-variable name;
catalog presence alone is not readiness. When automatic Mistral search is
ready, preserve it. Otherwise preserve a ready explicit choice; when neither
is ready, show Exa, Brave, and DuckDuckGo as onboarding choices. The standalone
provider list shows Mistral once and retains `auto` as its Mistral alias.
Preset suggestions likewise require credential-resolver readiness, an enabled
deployment, and a supported thinking level. Skipping the search editor leaves
the current search settings and tool enablement unchanged. No live connectivity
check or provider fallback is part of readiness.

### Exa and Brave blank-endpoint correction - 2026-09-30

**Cause:** The Web Search editor saves an empty `base_url` when the user chooses
the provider's default endpoint. Exa previously received that empty string as a
URL base, producing the relative path `/search`; HTTPX rejected it with
`UnsupportedProtocol` before a request could reach Exa.

**Fix:** Normalize an empty resolved `base_url` to no override. Exa and Brave
then use their built-in absolute endpoints; Mistral's empty override continues
to resolve through its configured provider endpoint or Mistral default. Custom
non-empty endpoint overrides are preserved. Search transport diagnostics now
classify common failures, including an invalid endpoint URL, without exposing
untrusted exception text.

Regression coverage uses a local mocked HTTP transport to assert URL formation
for unset, blank, and custom Exa/Brave endpoints and a blank Mistral override.
It does not test API keys against public providers or claim live connectivity.
The independent runtime/provider/diagnostic/settings run passed **103 tests**.
After strengthening the Mistral test-only custom-endpoint case, the focused
`tests/tools/test_web_search.py` file passed **32 tests**. These are separate
runs that may overlap, not an aggregate count. The checks use local mocks and
do not test public API keys or provider connectivity.
