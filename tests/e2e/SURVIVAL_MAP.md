# E2e survival map — stage 12d (W6a e2e lean)

Per-scenario mirror-then-retire record for the 12 pexpect files under
`tests/e2e/` (6 top-level `test_cli_tui_*.py` + 6 `agent_loop_characterization/`
files), per plan v5 §1.3 and design v3 §3 / §6.11.

Conventions:

- One section per e2e file, added in the file's retirement commit (one pexpect
  file per commit, in dependency order).
- *Retained* table: node IDs kept in the file because they observe PTY-level
  behavior with no protocol-level equivalent (design §6.11).
- *Retired* table: node IDs removed in the commit. The replacement column
  cites the surviving twin node ID(s) with a quote or assertion-equivalence
  note of the twin's key assertions; every cited twin was verified green
  before the retirement. `accepted-uncovered` rows carry an explicit
  rationale instead of a twin.
- *File-level check*: `retained == min(2–3, scenario_population)`, with
  §6.11 overriding the numeric target where the two conflict (e.g. a fully
  name-mirrored file retires entirely).
- A file is `git rm`ed when its retained set is empty, i.e. every scenario is
  name-mirrored by an acceptance-layer twin outside `tests/e2e/`.
- Carry-forward (stages 15/19/20): extract twin node IDs structurally from the
  Retired table's replacement column only (both tables contain backticked
  node IDs, so grepping every backticked node would validate retired,
  uncollectable ones), `sort -u`, intersect with the stage's touched set, and
  run the twin list as a gate; node-ID changes update this map in the same
  commit.

## tests/e2e/agent_loop_characterization/test_user_interaction.py — fully retired (stage 12d-i)

Population at retirement: 1 collected item (1 PTY scenario).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the file's single scenario is name-mirrored at the acceptance layer |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/agent_loop_characterization/test_user_interaction.py::test_ask_user_question_waits_for_answer_and_reports_it_to_the_model[ask-user-question]` | The `ask_user_question` widget renders ("Which mode should Vibe use?", option "Fast"), waits through the input grace period for the typed answer, and the model's follow-up request carries the tool result containing the selected answer. | `tests/tools/test_ask_user_question.py::test_requests_user_input_and_returns_result` — the tool requests user input and returns the selected answer (`result.answers[0].answer == "PostgreSQL"`); `tests/agent_loop/test_agent_tool_call.py::test_user_question_answered_by_request_event` — the question is answered via the interaction request, the ToolResultEvent is not skipped and carries the result (`result.skipped is False`, `result.error is None`, `result.result is not None`), and the turn completes; `tests/agent_loop/test_agent_tool_call.py::test_parallel_user_questions_both_receive_answers` — both parallel questions receive answers and both ToolResultEvents carry results (`tool_calls_agreed == 2`, `tool_calls_succeeded == 2`) |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the scenario's state transitions (question asked → user
  answer → tool result → model receives it) are all asserted by
  acceptance-layer twins, so the numeric retain target is overridden and the
  file is fully retired (`git rm`).

## tests/e2e/agent_loop_characterization/test_subagents.py — fully retired (stage 12d-ii)

Population at retirement: 1 collected item (1 PTY scenario).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the file's single scenario is name-mirrored at the acceptance layer |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/agent_loop_characterization/test_subagents.py::test_worker_subagent_returns_result_to_parent_model[worker-subagent]` | The parent model emits a `task` tool call; a worker subagent runs its own loop (its request is non-streaming with the task as user content); the worker's final result is returned to the parent model as the task tool result. | `tests/app_server/test_subagents_results.py::test_child_final_response_outcome_through_app_server[False-False]` — the foreground, non-empty-final parametrization of the app-server twin family (`[background-empty_final]`, also covering `[False-True]`, `[True-False]`, `[True-True]`): the parent's follow-up request carries the task tool result (a `Role.tool` message with `tool_call_id == "task-1"` in `parent_backend.requests_messages[1]`), the foreground child's final response is the tool result content (`effect.state.output["response"] == "Earlier child proseChild completed"`), the child's own request messages are asserted (`child_backend.requests_messages`; assistant messages "Earlier child prose"), and the parent completes ("Parent completed" in the session history). The child backend is non-streaming, matching the scenario's `stream is not True` observation. |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the scenario's state transitions (task tool call → worker
  subagent runs its own loop → final result returned to the parent model →
  parent continues) are all asserted by the acceptance-layer twin, so the
  numeric retain target is overridden and the file is fully retired
  (`git rm`).

## tests/e2e/agent_loop_characterization/test_tool_execution.py — fully retired (stage 12d-iii)

Population at retirement: 3 collected items (3 PTY scenarios).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: every scenario is name-mirrored at the acceptance layer |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/agent_loop_characterization/test_tool_execution.py::test_denylisted_bash_tool_does_not_run_and_is_reported_to_the_model[denied-bash-tool]` | A bash call matching the configured tool denylist never runs (the side-effect file is absent) and the denial is reported to the model as the tool result ("Command denied:"), after which the turn continues. | `tests/agent_loop/test_chartreux_policy_denials.py::test_direct_denial_no_approval_and_continued_work[path]` — a configured path/denylist denial raises no approval request, the denied tool's skip_reason is the model-facing tool message content (`results[0].skip_reason in (messages[0].content or "")`), the denied target is untouched (`target.read_text() == "harmless fixture"`), and the loop continues (second tool call executes, final reply "Continued safely"); `tests/agent_loop/e2e/test_e2e_bash.py::test_bash_denylisted_command_is_skipped` — a denylisted bash command is skipped with a denial reason (`result.skipped is True`, `"denied" in result.skip_reason.lower()`); `tests/agent_loop/test_agent_tool_call.py::test_tool_call_skipped_when_permission_is_never` — a policy-denied tool call is skipped and the skip_reason is the tool message content sent to the model (`events[3].skip_reason in (tool_msgs[-1].content or "")`) |
| `tests/e2e/agent_loop_characterization/test_tool_execution.py::test_failed_bash_tool_result_is_reported_to_the_model_and_turn_recovers[failing-bash-tool]` | A bash command exiting non-zero is reported to the model as the tool result containing the return code and the stderr text; the turn recovers (the model's second reply renders). | `tests/agent_loop/e2e/test_e2e_bash.py::test_bash_nonzero_exit_surfaces_as_error` — a non-zero exit surfaces as the tool result error (`"Return code: 3" in result.error`); `tests/agent_loop/e2e/test_e2e_bash.py::test_bash_captures_stderr` — the failing command's stderr is captured in the result (`"oops" in bash_result.stderr`); `tests/agent_loop/e2e/test_e2e_agent_loop.py::test_act_tool_call_round_trip` — the tool result is fed back for a final reply (the turn completes after the tool result). Assertion-equivalence: the model-facing tool message for a failed bash is the ToolError raised by `completed_shell_result` (`"Return code: {exit_code}"` + `"Stderr: {stderr}"`), which is the `result.error` the twins assert on |
| `tests/e2e/agent_loop_characterization/test_tool_execution.py::test_multiple_tool_calls_in_one_assistant_turn_return_distinct_results[multi-tool-turn]` | Two tool calls in one assistant turn both execute and both distinct results are returned to the model in the follow-up request. | `tests/agent_loop/test_agent_tool_call.py::test_parallel_conversation_history_has_all_tool_messages` — all parallel tool results appear in the conversation messages (`{m.tool_call_id for m in tool_msgs} == {"call_h0", ..., "call_h3"}`, `tool_calls_succeeded == 4`); `tests/agent_loop/test_agent_tool_call.py::test_parallel_tool_calls_produce_correct_events` — two tool calls in one response execute in parallel, both ToolResultEvents carry results, and the conversation history has both tool responses (`{m.tool_call_id for m in tool_msgs} == {"call_p1", "call_p2"}`, final assistant message "Both done."); `tests/agent_loop/e2e/test_e2e_agent_loop.py::test_act_tool_call_round_trip` — a tool call is parsed, executed, and its result fed back for a final reply |

### File-level check

- scenario_population: 3; min(2–3, 3) = 2–3; retained: 0.
- §6.11 override: every scenario's state transitions (denial without
  execution reported to the model with the turn continuing; tool failure
  reported with the turn recovering; parallel tool results all returned) are
  asserted by acceptance-layer twins, so the numeric retain target is
  overridden and the file is fully retired (`git rm`).

## tests/e2e/agent_loop_characterization/test_tool_permissions.py — fully retired (stage 12d-iv)

Population at retirement: 2 collected items (2 PTY scenarios).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: every scenario is name-mirrored at the acceptance layer |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/agent_loop_characterization/test_tool_permissions.py::test_write_file_auto_execution_creates_file_and_denylist_leaves_file_absent[write-file-auto-execution-and-denial]` | A write_file call auto-executes (creates the file; the tool result echoes the written content); a write_file call matching the configured path denylist leaves the file absent and the denial ("File access denied by a configured path rule") is reported to the model as the tool result. | `tests/agent_loop/e2e/test_e2e_tools.py::test_write_file_tool_creates_file` — write_file runs through the loop with no approval responder and creates the file (`target.read_text() == "hi\n"`); `tests/core/tools/test_denial_kinds.py::test_denial_kind_classification[path_rule-path_rule]` — a configured denylist path for write_file resolves to NEVER with denial_kind "path_rule" (`resolve_file_tool_permission(..., tool_name="write_file", denylist=[str(target)], ...)`; `result.permission is ToolPermission.NEVER`, `result.denial_kind == "path_rule"`) — the engine behind "File access denied by a configured path rule"; `tests/agent_loop/test_chartreux_policy_denials.py::test_direct_denial_no_approval_and_continued_work[path]` — the path-rule denial is reported to the model as the tool message content and the denied target is untouched; `tests/tools/test_scratchpad_permissions.py::TestFileToolScratchpadPermissions::test_non_scratchpad_outside_dir_denied` — WriteFile resolve_permission denies a configured-out path with NEVER + reason; `tests/agent_loop/e2e/test_e2e_agent_loop.py::test_act_tool_call_round_trip` — the tool result is fed back for a final reply. Assertion-equivalence: the twin pins the write side effect and the result-fed-back transition; the e2e additionally echoes the written content in the tool result, which is the same WriteFile result object whose side effect the twin asserts |
| `tests/e2e/agent_loop_characterization/test_tool_permissions.py::test_bash_without_allowlist_auto_executes_each_time[session-permission-memory]` | A bash call with no allowlist entry auto-executes, and a repeated identical call auto-executes again (no session-permission prompt); both tool results reach the model. | `tests/agent_loop/e2e/test_e2e_bash.py::test_bash_non_allowlisted_command_runs_without_approval` — a non-allowlisted bash command runs without approval (`result.skipped is False`; the side-effect file exists); `tests/agent_loop/e2e/test_e2e_agent_loop.py::test_act_tool_call_round_trip` — the tool result is fed back for a final reply. Assertion-equivalence: the scenario's second call re-exercises the same per-call permission resolution (a non-allowlisted bash command resolves to auto-execute with no approval request); the twin pins that resolution |

### File-level check

- scenario_population: 2; min(2–3, 2) = 2; retained: 0.
- §6.11 override: both scenarios' state transitions (write_file
  auto-execution with the side effect and the configured path-rule denial
  reported to the model; non-allowlisted bash auto-executing on each call with
  the results returned) are asserted by acceptance-layer twins, so the
  numeric retain target is overridden and the file is fully retired
  (`git rm`).

## tests/e2e/agent_loop_characterization/test_resume.py — fully retired (stage 12d-v)

Population at retirement: 1 collected item (1 PTY scenario).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the file's single scenario's state transitions are covered by composed acceptance-layer twins |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/agent_loop_characterization/test_resume.py::test_resumed_session_sends_prior_tool_call_and_result_history_to_the_model[resume-tool-history]` | A real TUI session runs a first turn (user prompt → `todo` tool call → tool result "Updated 1 todos" → assistant text) that is persisted to the session log; after quitting and respawning with `--resume <session_id>`, the resumed model request carries the full prior history — the prior user prompt, the assistant tool call (`call_todo_resume`, tool "todo"), the tool result for that call id, and the prior assistant text — plus the new user prompt. | `tests/app_server/test_session_resume_admission.py::test_compaction_keeps_session_and_restores_full_history` — after `attach_test_app_server_session(..., resume_session_id=session_id)`, the resumed backend's first request (`resumed_backend.requests_messages[0]`) carries the full restored persisted history (5 messages with asserted contents, including the compaction boundary) plus the new user turn — the same resume-into-model-request transition; `tests/session/persistence/test_session_loader.py::TestSessionLoaderEdgeCases::test_load_session_with_complex_messages` — a persisted session whose messages carry `tool_calls` loads back with the tool calls intact (`len(messages[0].tool_calls or []) == 1`, `len(messages[1].tool_calls or []) == 1`), and the loader loads every non-system role (session_loader.py:460), tool results included; `tests/cli/test_ui_session_resume.py::test_ui_displays_messages_when_resuming_session` — the resumed transcript renders the prior user message, the assistant message with its tool call (`ToolCallMessage._tool_name == "read"`), and the tool result (`ToolResultMessage` with `state.output_text == "File content here"`); `tests/core/hooks/test_hooks.py::TestAgentLoopIntegration::test_pre_tool_rewrite_is_persisted_to_messages_jsonl` — assistant tool calls are persisted to the on-disk `messages.jsonl` that resume loads (the persisted args are read back from the file). Assertion-equivalence: the chain persist (tool call and tool result written to `messages.jsonl`) → load (all non-system roles round-trip) → resume-into-request (loaded history sent to the model) → render (transcript widgets) is covered link by link; the residual — no single acceptance test asserts a `role=tool` message inside a resumed model request — rests on the loader's all-non-system-roles round-trip plus the compaction twin's request assertion |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the scenario's state transitions (first-turn persistence
  of the tool call and result; the resumed model request carrying the prior
  user prompt, tool call, tool result, and assistant text plus the new user
  prompt) are covered by the composed acceptance-layer twins cited above, so
  the numeric retain target is overridden and the file is fully retired
  (`git rm`).

## tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py — lean (stage 12d-vi)

Population at retirement: 14 collected items (12 PTY scenarios; two
2-parametrized functions). Retained: 2 (the reprompt-suppression pair).
Retired: 12.

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_deny_skips_execution_and_suppresses_the_reprompt[deny-and-refusal-memory]` | The dangerous-command question renders for the denied `git push`; the identical second call is refused from refusal memory with NO second question render — the rendered question-title count stays at the baseline. | Reprompt suppression after deny is the design §6.11 example of a PTY-level observation with no protocol-level equivalent: the acceptance twin tests/app_server/test_dangerous_command_acceptance.py::test_deny_refuses_execution_and_remembers_the_refusal[bash] asserts the refusal decision (no re-prompt request issued, DECLINED_FEEDBACK, no authority, config unchanged) but not the question widget's render count in the PTY output. |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_allow_once_executes_exactly_one_call_and_reprompts[allow-once-then-reprompt]` | The question renders for the first `git push`; the once-approval executes exactly one invocation (marker + remote main updated); the identical second call renders the question AGAIN — the rendered question-title count increases. | The design §3 example PTY-only observation (reprompt suppression after a consumed once-approval): the acceptance twin tests/app_server/test_dangerous_command_acceptance.py::test_allow_once_executes_exactly_one_call_and_reprompts[bash] asserts the permit consumption (exactly one invocation, no authority remains, the identical call prompts again) but not the question widget's re-render in the PTY output. |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_allow_this_session_skips_later_prompts[allow-this-session]` | A `git push` matching the dangerous-command policy opens the real question; answering "allow this session" executes the call (marker + remote main), and the identical second call runs with NO second question render; the session grant is never persisted to the user config. | `tests/app_server/test_dangerous_command_acceptance.py::test_allow_this_session_executes_subsequent_calls_without_prompting[bash]` — the second identical call resumes without prompting (`callbacks == []`), the authority is the (git.push, project-scope) session grant, and `_saved_grants(config_path) == {}` (never persisted); `tests/agent_loop/test_grant_lifecycle_harness.py::test_allow_session_grant_resumes_the_tool_call[dangerous-bash]` — `_authorities == [(frozenset({"git.push"}), scope_of(project))]` retained across the second call with `requests == []`; `tests/tools/test_dangerous_command_permissions.py::TestConsequentialSessionAuthority::test_session_authority_is_never_persisted` — the resolver is read-only: no saved grant appeared and nothing was written to disk |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_always_for_project_persists_the_grant_in_the_user_config[always-for-this-project]` | Answering "always for this project" executes the call and the user-layer config file gains `{project: ["git.push"]}`; the saved grant covers the identical second call with NO question render. | `tests/app_server/test_dangerous_command_acceptance.py::test_always_for_project_executes_and_persists_to_the_user_config[bash]` — the tool message reports "saved in your user config", `_saved_grants(config_path) == {str(project.resolve()): ["git.push"]}`, and the second call runs with `callbacks == []`; `tests/agent_loop/test_grant_lifecycle_harness.py::test_always_applies_the_grant_and_reports_the_saved_outcome[dangerous-bash]` — the save port is called with (session_id, project.resolve(), "git.push", user-rev-1) and the live session grant applies exactly like allow-session |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_saved_project_grant_does_not_cover_a_worktree_or_other_repo[worktree-and-other-repo-scopes]` | With a saved project grant, the session's own repository pushes with NO question; a sibling worktree (`git -C wt`) and a nested repository each prompt again (both denied); the denials persist nothing beyond the original project entry. | `tests/app_server/test_dangerous_command_acceptance.py::test_project_authority_does_not_cover_a_sibling_worktree_or_other_repo` — the saved grant covers the own repository (`callbacks == []`, main marker + remote), the sibling worktree and the nested repo each prompt (`len(callbacks) == 1`, markers absent), and `_saved_grants == {project: ["git.push"]}` is unchanged; `tests/agent_loop/test_dangerous_command_lifecycle_grants.py::test_sibling_worktree_scope_is_isolated` — the main worktree is a different repository identity: it prompts and denies; `tests/tools/test_dangerous_command_permissions.py::TestRepositoryScopeIsolation::test_sibling_worktree_cannot_consume_the_grant` — a linked worktree is a distinct scope that cannot consume the grant; `tests/tools/test_dangerous_command_permissions.py::TestSavedProjectGrants::test_saved_grant_scope_is_the_project_repository` — the grant never relocates to another repository (`git -C other push` → dangerous denial) |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_explicit_denylist_vetoes_the_question_even_with_a_saved_grant[denylist-veto-with-saved-grant]` | With a saved project grant AND an explicit `git push` denylist, the call is vetoed with NO question render; the tool result reports "matches denylist pattern 'git push'"; the marker and remote stay untouched. | `tests/app_server/test_dangerous_command_acceptance.py::test_explicit_denylist_vetoes_even_with_a_saved_grant` — no callbacks, marker/remote untouched, the tool text contains "matches denylist pattern 'git push'", the saved grant is accepted but the resolver denial carries no dangerous-command proposal (`denial_kind is None`, `dangerous_command_proposal is None`); `tests/tools/test_dangerous_command_permissions.py::TestSavedProjectGrants::test_explicit_denylist_is_absolute_even_with_a_saved_grant` — the veto is checked before grantability: "matches denylist pattern 'git push'" with no proposal; `tests/tools/test_dangerous_command_permissions.py::TestOrderingInvariant::test_denylist_veto_is_never_masked[first]` — the denylist veto is never masked by the dangerous-command gap (other denial, no proposal) |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_hard_guards_stay_denied_without_a_question[git-reset-hard]` | `git reset --hard` is denied with NO question render ("Command denied"); the guard marker is absent, the remote has no main, and the commit count is unchanged. | `tests/app_server/test_dangerous_command_acceptance.py::test_hard_guards_stay_denied_without_a_question[git reset --hard]` — no callbacks even under a session grant for publication, the tool text contains "Command denied" and the guard, the session grant cannot lift the guard, `_commit_count(project) == 1`; `tests/tools/test_dangerous_command_permissions.py::TestOrderingInvariant::test_policy_hard_guard_is_never_masked[first]` — the hard guard is never masked (other denial: `denial_kind is None`, no proposal); `tests/tools/test_dangerous_command_permissions.py::TestShippedDenylistMigration::test_hard_guards_survive_the_migration` — `git reset --hard HEAD` and `git clean -fd` stay hard-denied |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_hard_guards_stay_denied_without_a_question[git-clean-fd]` | `git clean -fd` is denied with NO question render ("Command denied"); the guard marker is absent, the remote has no main, and the commit count is unchanged. | `tests/app_server/test_dangerous_command_acceptance.py::test_hard_guards_stay_denied_without_a_question[git clean -fd]` — no callbacks even under a session grant for publication, the tool text contains "Command denied" and the guard, the session grant cannot lift the guard, `_commit_count(project) == 1`; `tests/tools/test_dangerous_command_permissions.py::TestShippedDenylistMigration::test_hard_guards_survive_the_migration` — `git reset --hard HEAD` and `git clean -fd` stay hard-denied; `tests/tools/test_dangerous_command_permissions.py::TestOrderingInvariant::test_policy_hard_guard_is_never_masked[first]` — the hard guard is never masked (other denial, no proposal) |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_other_denial_wins_over_the_dangerous_question[sensitive-segment]` | A compound `git push origin main; sudo true` is denied with NO question render; the tool result reports "Command denied: unsupported sudo form" (v0.5.9 P2-d: sudo left the shipped sensitive prefixes; the unmodeled form fails closed in the executor grammar); the remote has no main. | `tests/app_server/test_dangerous_command_acceptance.py::test_other_denial_wins_over_the_dangerous_question[git push origin main; sudo true-Command denied: unsupported sudo form]` — no callbacks, the tool text contains the exact expected denial, no dangerous-denial fragment, no authority; `tests/tools/test_dangerous_command_permissions.py::TestOrderingInvariant::test_sensitive_rule_is_never_masked[first]` — a user-configured sensitive rule is never masked ("sensitive command rule" in the reason, other denial, no proposal) |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_other_denial_wins_over_the_dangerous_question[path-segment]` | A compound `git push origin main; cat /etc/hostname` is denied with NO question render; the tool result reports "Shell path is outside the authorized workspace". | `tests/app_server/test_dangerous_command_acceptance.py::test_other_denial_wins_over_the_dangerous_question[git push origin main; cat /etc/hostname-Shell path is outside the authorized workspace]` — no callbacks, the tool text contains the exact expected denial, no dangerous-denial fragment; `tests/tools/test_dangerous_command_permissions.py::TestOrderingInvariant::test_path_restriction_is_never_masked[first]` — the path restriction is never masked ("outside the authorized workspace" in the reason, other denial); `tests/agent_loop/test_chartreux_policy_denials.py::test_outside_shell_denial_survives_stale_allowlist_and_bypass[cat {target}-*]` — an outside-workspace path denial survives a stale allowlist and bypass attempts |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_two_concurrent_identical_once_calls_need_two_approvals[two-concurrent-once-calls]` | Two identical concurrent `git push` calls each open the real question; one "allow once" approval executes exactly one invocation (marker + remote), and the second call's own question must be answered separately (denied); exactly one tool result carries the declined feedback. | `tests/app_server/test_dangerous_command_acceptance.py::test_two_concurrent_identical_once_calls_need_two_approvals[bash]` — two callbacks for the two identical concurrent calls, exactly one invocation executed (one declined tool result, the other without "Command denied"), no authority outlives either invocation; `tests/agent_loop/test_grant_lifecycle_harness.py::test_concurrent_calls_coalesce_or_need_separate_approvals[dangerous-bash]` — no coalescing: two prompts (`len(requests) == 2`), each permit consumed by its own invocation (`_authorities == []`) |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_routine_git_commands_execute_without_prompts[routine-git-commands]` | Four routine git commands (`git status`, `git log --oneline`, `git add seed.txt`, `git commit -m routine`) execute with NO question render; the commit count grows and the status result contains "On branch main". | `tests/app_server/test_dangerous_command_acceptance.py::test_routine_git_commands_execute_without_prompts` — no callbacks, all four tool results present with "On branch main" in the status result, `_commit_count(project) == 2`; `tests/tools/test_dangerous_command_permissions.py::TestFailClosedCheckpoint::test_routine_commands_are_unaffected` — `git status`, `git log --oneline`, `git commit -m x` and friends resolve ALWAYS with `denial_kind is None` and no dangerous-command proposal |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_subagent_dangerous_call_hits_the_parent_ceiling[subagent-parent-ceiling]` | A worker subagent's dangerous `git push` hits the parent ceiling: NO question renders, the call is skipped with "requires approval for a dangerous git operation", the marker and remote stay untouched. | `tests/app_server/test_dangerous_command_acceptance.py::test_child_approval_cannot_lift_a_parent_dangerous_ceiling` — no question reaches any client, the child result is skipped with the dangerous denial fragment, the resolver classifies the gap as `parent_ceiling` with no proposal, and even a child-side session grant cannot lift the ceiling; `tests/agent_loop/test_dangerous_command_lifecycle_session_boundaries.py::TestChildSessions::test_parent_dangerous_gap_is_a_ceiling_for_the_child` — the child's call is skipped with no prompt and no local grant (`requests == []`, `_authorities(child) == []`); `tests/tools/test_dangerous_command_permissions.py::TestParentCeilings::test_parent_gap_classifies_as_parent_ceiling[bash]` — `denial_kind == "parent_ceiling"`, `dangerous_command_proposal is None` |
| `tests/e2e/agent_loop_characterization/test_dangerous_git_permissions.py::test_resumed_session_does_not_restore_session_only_grants[session-grant-not-restored-on-resume]` | A session grant covers the second identical call with NO question; after quitting and resuming with `--resume <session_id>`, the same call prompts AGAIN (denied); the grant never reached the user config. | `tests/app_server/test_dangerous_command_acceptance.py::test_session_only_grants_do_not_survive_new_or_resumed_sessions` — the session grant covers the second call (`callbacks == []`), `_saved_grants == {}`, and both a brand-new session and a resumed session (persisted session id) start with no runtime authority (`read.live_grants == []`, the same call prompts again); `tests/agent_loop/test_dangerous_command_lifecycle_session_boundaries.py::TestSessionLifetime::test_resume_does_not_restore_runtime_grants` — a resumed conversation is a new loop: the runtime store starts empty (`_authorities(resumed) == []`), the same call prompts and denies |

### File-level check

- scenario_population: 14; min(2–3, 14) = 2–3; retained: 2.
- The numeric target is met exactly: the retained pair is the design §6.11
  PTY-only population (reprompt suppression after deny; the re-prompt after a
  consumed once-approval), so no §6.11 override is needed. The file is leaned,
  not `git rm`ed: the two PTY-only scenarios stay, the twelve mirrored
  scenarios are retired.

## tests/e2e/test_cli_tui_hooks.py — fully retired (stage 12d-vii)

Population at retirement: 1 collected item (1 PTY scenario).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the file's single scenario's state transitions are covered by acceptance-layer twins |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/test_cli_tui_hooks.py::test_spawn_cli_runs_configured_hook_after_turn` | A `post_agent` hook configured via `hooks.toml` in the user home runs after the assistant turn completes in a real TUI session; the hook subprocess receives the invocation JSON on stdin with `hook_event_name == "post_agent"`, a non-empty `cwd`, a non-empty `session_id`, and `parent_session_id is None` (a top-level session). | `tests/core/hooks/test_hooks.py::TestAgentLoopIntegration::test_post_agent_hook_runs_after_turn` — a configured POST_AGENT hook runs after the assistant turn through the real agent loop (HookStartEvent and HookEndEvent in the turn events); `tests/agent_loop/e2e/test_e2e_hooks.py::test_post_agent_hook_injects_retry_user_message` — the post_agent hook's deny decision injects a retry user message into the follow-up request (`"please continue" in mistral_api.model_facing_text(1)`) and the loop continues (`answers == ["first answer", "second answer"]`); `tests/core/hooks/test_hooks.py::TestHookExecutor::test_stdin_json_received` — the hook subprocess receives the invocation JSON on stdin (`d['session_id'] == "test-session"`); the payload is `PostAgentInvocation.model_dump_json()` (executor.py:45) whose fields are hook_event_name / session_id / transcript_path / cwd / parent_session_id (models.py:127-137, parent_session_id defaulting to None for a top-level session, matching the loop's `_hook_session_context` at agent_loop_hooks.py:125-130); `tests/core/hooks/test_hooks.py::TestPostToolHook::test_invocation_includes_status_and_output` — the hook stdin payload carries `hook_event_name` (assertion-equivalence for the payload field the scenario pins for post_agent); `tests/core/hooks/test_hooks.py::TestConfigLoading::test_load_from_global_file` — a `hooks.toml` in the global config dir loads the configured hook (the scenario's config-loading transition) |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the scenario's state transitions (a globally configured
  post_agent hook runs after the turn; the hook receives the invocation
  payload with the session context fields) are asserted by acceptance-layer
  twins, so the numeric retain target is overridden and the file is fully
  retired (`git rm`).

## tests/e2e/test_cli_tui_streaming.py — fully retired (stage 12d-viii)

Population at retirement: 1 collected item (1 PTY scenario).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the file's single scenario's state transitions are covered by acceptance-layer twins |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/test_cli_tui_streaming.py::test_spawn_cli_to_send_and_receive_message` | A real TUI session sends the typed user message as a streaming request (`stream is True`, `model == "mock-model"`, `stream_options.include_usage is True`, the user message in `messages`); the streamed response text renders in the TUI; the first-run welcome is absent (the home is configured). | `tests/backend/test_ssl_integration.py::test_generic_backend_streaming_uses_ssl_cert_file` — the real GenericBackend against the same `StreamingMockServer` with the same user message "Greet" and model "mock-model": `request_payload.get("stream") is True`, `request_payload.get("model") == "mock-model"`, and the streamed content `"Hello from mock server"` is received through `complete_streaming`; `tests/backend/test_backend.py::TestBackend::test_backend_streaming_payload_includes_stream_options[https://api.fireworks.ai-fireworks-expected_stream_options0]` — `payload["stream"] is True` and `payload["stream_options"] == {"include_usage": True}` (the fireworks parametrization); `tests/agent_loop/test_agent_observer_streaming.py::test_act_streams_chunks_in_order` — the loop with `enable_streaming=True` consumes the streamed chunks in order, one AssistantEvent per chunk with the exact contents (the streamed-response-consumption transition); `tests/cli/test_programmatic_setup.py::test_interactive_mode_still_runs_onboarding_on_missing_api_key` — the onboarding decision (`require_api_key_or_onboard` runs onboarding only when the key is missing), the complement of the scenario's welcome-absence assertion with a configured home; `tests/setup/onboarding/test_screens.py::test_enter_completes_welcome_animation_then_opens_providers` — the welcome screen itself renders and advances (the positive side of the welcome-absence assertion) |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the scenario's state transitions (the streaming request
  payload fields and the streamed response consumed by the loop; the
  no-onboarding decision with a configured home) are asserted by
  acceptance-layer twins, so the numeric retain target is overridden and the
  file is fully retired (`git rm`).

## tests/e2e/test_cli_tui_session_exit.py — fully retired (stage 12d-ix)

Population at retirement: 1 collected item (1 PTY scenario).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the file's single scenario's state transitions are covered by acceptance-layer twins |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/test_cli_tui_session_exit.py::test_resumed_session_prints_only_fresh_token_usage_on_exit[fresh-usage-after-resume]` | DEBUG_MODE exit print: the first run prints "Total tokens used this session: input=11 output=7 (total=18)" with the "Or: chartreux --resume \<id\>" hint and the saved session metadata carries 11/7; the resumed run (`--resume \<id\>`) prints only the fresh "input=2 output=1 (total=3)" and the saved metadata accumulates to 13/8. | `tests/cli/test_ui_session_exit.py::test_print_session_resume_message_prints_resume_commands_and_usage` — asserts the exact exit print "Total tokens used this session: input=14,867 output=6 (total=14,873)" plus the "Or: chartreux --resume \<id\>" hint (the first-run print leg); `tests/cli/test_ui_session_exit.py::test_print_session_resume_message_prints_zero_usage_for_resumed_run_without_llm_activity` — a resumed run's print carries its own (zero) usage, not the saved totals (the print leg for the resumed run); `tests/app_server/test_session_resume_admission.py::test_in_process_resume_rebases_exit_usage` — a session persisted with 11/7 resumes with `exit_summary().usage` 0/0 (the rebase the scenario asserts: the resumed run's print shows only the fresh usage); `tests/session/persistence/test_session_logger_save.py::TestSessionLoggerSaveInteraction::test_save_interaction_with_existing_messages` — a second save persists the accumulated stats into meta.json (`metadata["stats"]["steps"] == updated_stats.steps` with the accumulated AgentStats) (the accumulation leg, 11/7 → 13/8) |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the scenario's state transitions (the exact exit print with
  the resume hint; the usage rebase on resume; the stats accumulation into the
  saved metadata) are asserted by acceptance-layer twins, so the numeric
  retain target is overridden and the file is fully retired (`git rm`).

## tests/e2e/test_cli_tui_tool_approval.py — fully retired (stage 12d-x)

Population at retirement: 1 collected item (1 PTY scenario).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the file's single scenario's state transitions are covered by acceptance-layer twins |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/test_cli_tui_tool_approval.py::test_spawn_cli_auto_executes_bash_and_shows_tool_output[tool-call-stream]` | A real TUI session sends "Run a shell command"; the model emits a `bash` tool call (`echo __E2E_BASH_OK__`); the CLI auto-executes the call with no approval (no allowlist entry); the follow-up model request carries the tool result for `call_bash_1` containing the command output. | `tests/agent_loop/e2e/test_e2e_bash.py::test_bash_non_allowlisted_command_runs_without_approval` — a non-allowlisted bash command runs without approval (`result.skipped is False`; the side-effect file exists) (the auto-execution leg); `tests/agent_loop/e2e/test_e2e_bash.py::test_bash_captures_stdout` — the bash result carries the command stdout (`bash_result.exit_code == 0`, `"hello" in bash_result.stdout`) (the content the scenario asserts in the tool message); `tests/agent_loop/e2e/test_e2e_agent_loop.py::test_act_tool_call_round_trip` — a tool call is parsed, executed, and its result fed back for a final reply (ToolCallEvent + ToolResultEvent + final AssistantEvent "All done") (the tool-result-fed-back leg). Assertion-equivalence: the scenario's only assertions beyond the request count are on the follow-up request payload (the tool message content), which is the same Bash result object the twins assert on; the real spawn is the vehicle, not the observation |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the scenario's state transitions (non-allowlisted bash
  auto-executing without approval; the tool result carrying the command output
  fed back to the model) are asserted by acceptance-layer twins, so the
  numeric retain target is overridden and the file is fully retired (`git rm`).
  The file's `agent_loop_characterization/support.py` import retires with it;
  support.py itself stays (the lean dangerous-git file still imports it).

## tests/e2e/test_cli_tui_fresh_install.py — fully retired (stage 12d-xi)

Population at retirement: 1 collected item (1 PTY scenario with two legs:
the wheel build/install/spawn leg and the spawned-CLI happy-path leg).

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| — | — | none: the happy-path leg's state transitions are covered by acceptance-layer twins; the wheel leg is accepted-uncovered (below) |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/test_cli_tui_fresh_install.py::test_fresh_wheel_install_can_spawn_cli_and_complete_happy_path` | Happy-path leg: the spawned installed CLI shows the main screen (no "Welcome to Chartreux"), sends "Greet", receives the streamed "Hello from mock server", and the request carries `model == "mock-model"`. Wheel leg: `uv build --wheel` produces exactly one chartreux wheel, it installs into a fresh venv, and the installed `chartreux` executable spawns. | `tests/backend/test_ssl_integration.py::test_generic_backend_streaming_uses_ssl_cert_file` — the real GenericBackend against the same StreamingMockServer: `stream is True`, `model == "mock-model"`, the streamed content received (the happy-path request leg); `tests/backend/test_backend.py::TestBackend::test_backend_streaming_payload_includes_stream_options[https://api.fireworks.ai-fireworks-expected_stream_options0]` — `payload["stream"] is True` and `payload["stream_options"] == {"include_usage": True}` (the payload-shape leg); `tests/agent_loop/test_agent_observer_streaming.py::test_act_streams_chunks_in_order` — the loop with `enable_streaming=True` consumes the streamed chunks in order (the streamed-response-consumption leg); `tests/cli/test_programmatic_setup.py::test_interactive_mode_still_runs_onboarding_on_missing_api_key` — the onboarding decision (no welcome with a configured home); `tests/setup/onboarding/test_screens.py::test_enter_completes_welcome_animation_then_opens_providers` — the welcome screen itself (the complement of the welcome-absence assertion). Wheel leg: `accepted-uncovered` — the only candidate twin is tests/cli/test_installed_contracts.py (`test_exact_wheel_has_installed_identity_and_all_three_entrypoints` :909, `test_installed_programmatic_cli_turn_uses_only_fake_provider` :1043), which skips locally: the wheelhouse guard at test_installed_contracts.py:222–225 requires `UV_FIND_LINKS` pointing at a pre-provisioned wheelhouse (re-verified at retirement: 3 passed, 7 skipped locally; both candidate twins SKIPPED with that message). CI covers the installed contracts; the wheel leg's unique local value (real wheel build + fresh-venv install + installed-executable spawn) is recorded as accepted-uncovered per plan §1.3 rather than claimed as locally verified twin coverage |

### File-level check

- scenario_population: 1; min(2–3, 1) = 1; retained: 0.
- §6.11 override: the happy-path leg's state transitions (the streaming
  request payload fields, the streamed response consumed by the loop, the
  no-onboarding decision with a configured home) are asserted by
  acceptance-layer twins; the wheel leg has no locally verifiable twin
  (`accepted-uncovered`, CI-covered via the installed-contract suite), so the
  numeric retain target is overridden and the file is fully retired (`git rm`).

## tests/e2e/test_cli_tui_first_run_onboarding.py — lean (stage 12d-xii)

Population at retirement: 9 collected items (8 PTY journeys + 1 pure unit
test of a module-local helper). Retained: 2 PTY journeys (the keyboard-driven
onboarding-flow pair) + the unit test kept with its helper. Retired: 6 PTY
journeys.

### Retained

| node ID | PTY behavior observed | §6.11 rationale |
| --- | --- | --- |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_empty_home_onboarding_reaches_first_streaming_turn[auto-seeded]` | The public first-run onboarding flow driven by terminal key presses only, end to end in a real spawned CLI: welcome → provider settings → connection editor → credential env var + API key → model editor → presets → finish; the on-disk models.toml (provider, deployment, roles bound to the model, standalone dispatch), the .env write, and the first streaming turn (`model == "onboarding-mock-model"`) are asserted. | The composed end-to-end first-run flow through a real process and real terminal keystrokes: no acceptance-layer test drives the whole public flow in a real spawned CLI — the acceptance twins cover each transition separately (materialization, .env write, catalog writer, role merge, streaming payload). PTY-level observation with no protocol-level equivalent (design §6.11). |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_empty_home_onboarding_reaches_first_streaming_turn[customize]` | The same keyboard-driven flow with the preset-customization leg: the orchestrator preset is edited through the real UI (thinking set to off) before finish, and the saved catalog records `roles.orchestrator.thinking == "off"`. | The preset-thinking customization through the real TUI (choose thinking → Thinking off → apply) has no protocol-level equivalent in a real spawned CLI; the catalog-writer and dispatch-overlay twins cover the persisted shape, not the keystroke-driven UI transition. PTY-level observation with no protocol-level equivalent (design §6.11). |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_wait_for_moved_highlight_tolerates_marker_absence_before_movement` | Not a PTY scenario: a pure unit test of the module-local helper `_wait_for_moved_highlight` (a transient unparsable frame does not block later observed movement). | Kept with the helper: the retained journeys drive arrow keys through `_send_and_wait_for_text` → `_wait_for_moved_highlight`, so the helper and its direct unit coverage survive the lean. Not counted in the PTY retained target. |

### Retired

| node ID | PTY observation | surviving twin node ID(s) + key-assertion citation |
| --- | --- | --- |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_env_key_first_run_skips_onboarding_without_creating_a_catalog` | MISTRAL_API_KEY in the environment: no onboarding (no "Let's get you started", no "Provider Settings"), the startup banner counts both shipped default models ("2 models", "zai-glm-5-3"), provider settings shows "Key Set" + "2 runnable models", no models.toml is created, and no graduation nudge appears. | `tests/cli/test_programmatic_setup.py::test_interactive_mode_still_runs_onboarding_on_missing_api_key` — `require_api_key_or_onboard(orchestrator, interactive=True)` runs onboarding only when the active provider key is missing (the skip decision); `tests/setup/auth/test_auth_state.py::test_assess_process_env_when_key_is_only_in_process_env` — the provider key resolves from the process environment; `tests/banner/test_banner_initial_state.py::TestBannerInitialState::test_format_meta_counts` — the banner meta counts render "2 models" for two models; `tests/ui/providers/test_workbench_onboarding.py::test_ready_shipped_mistral_root_manages_and_finishes[size0]` — the shipped mistral row in the onboarding workbench at 80x24 shows "Key Set" and "2 runnable models" (in-process, real screen); `tests/core/config/test_model_catalog.py::test_14_missing_models_toml_loads_shipped_defaults` — a missing models.toml loads the shipped defaults (the shipped catalog serves both default models with no onboarding); `tests/ui/providers/test_graduation.py::test_fresh_two_model_roster_never_latches` — a fresh two-model roster never latches a save and shows no nudge (the graduation-nudge-absence leg) |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_interactive_mistral_key_onboarding_materializes_the_default_catalog` | No key in the environment; the Mistral key entered interactively completes onboarding; the materialized models.toml equals the shipped default template, the .env gains MISTRAL_API_KEY, and both default models are usable in the running TUI ("Key Set", "2 runnable models"). | `tests/setup/onboarding/test_materialization.py::test_completed_onboarding_materializes_the_default_template` — a completed onboarding materializes models.toml == render_default_template() and the loaded snapshot equals SHIPPED_CATALOG (the exact materialization the scenario asserts); `tests/setup/auth/test_api_key_persistence.py::test_persist_writes_api_key_to_env_file_and_process_env` — persisting the key writes it to the .env file and the process environment; `tests/ui/providers/test_workbench_onboarding.py::test_ready_shipped_mistral_root_manages_and_finishes[size0]` — the shipped mistral row shows "Key Set" and "2 runnable models" once the key is set; `tests/ui/providers/test_workbench_onboarding_summary.py::test_empty_home_keyless_setup_finishes_without_role_questions[standalone]` — finishing the onboarding flow binds every role to the selected model with standalone dispatch and no role questions (the role-binding leg) |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_no_key_welcome_decline_exits_without_materializing` | Escape on the no-key welcome prints "Setup cancelled", the process exits with status 0, and no models.toml is materialized. | `tests/setup/onboarding/test_materialization.py::test_cancelled_onboarding_does_not_materialize` — a cancelled onboarding raises SystemExit with code 0 and creates no models.toml (the exact exit-status + no-materialization transitions); `tests/acp/test_acp_entrypoint_smoke.py::test_vibe_acp_setup_shows_onboarding_and_exits_on_cancel` — a real spawn shows the onboarding, exits on cancel, and the captured output contains "Setup cancelled" (the rendered cancel message) |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_sparse_overlay_first_run_merges_without_rewriting_the_user_catalog` | A sparse user models.toml overriding only the Mistral base URL merges onto the shipped catalog at first run: onboarding is skipped (key present), inference is served through the merged catalog (`model == "zai-glm-5-3"`), and the user file is never rewritten. | `tests/core/config/test_model_catalog.py::test_10_sparse_provider_overlay_inherits_shipped_fields` — a sparse provider overlay inherits the shipped provider fields (the merge leg: the overlay overrides only the base URL, the credential wiring is inherited); `tests/setup/onboarding/test_materialization.py::test_completed_onboarding_never_overwrites_a_saved_catalog` — a saved user catalog is never overwritten (the never-rewrite leg); `tests/core/config/test_model_catalog.py::test_14_missing_models_toml_loads_shipped_defaults` — the shipped defaults serve the role models through the merged catalog (the inference leg resolves the shipped zai-glm-5-3 orchestrator model); `tests/cli/test_programmatic_setup.py::test_interactive_mode_still_runs_onboarding_on_missing_api_key` — the onboarding skip decision with a resolvable key |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_multi_key_env_first_run_sends_the_selected_provider_credential` | Several keys in the environment (MISTRAL_API_KEY, EXA_API_KEY, BRAVE_SEARCH_API_KEY): onboarding is skipped, the user catalog is never rewritten, and the request carries the Mistral credential — never an unrelated key. | `tests/backend/test_backend.py::test_generic_backend_auth_error_keeps_environment_api_key_origin` — the backend request's authorization header is the provider's env-var credential (`route.calls.last.request.headers["authorization"] == f"Bearer {token}"`) (the selected-provider credential leg); `tests/setup/auth/test_auth_state.py::test_assess_process_env_when_key_is_only_in_process_env` — the provider's key resolves from its own process env var (the per-provider selection the scenario pins); `tests/cli/test_programmatic_setup.py::test_interactive_mode_still_runs_onboarding_on_missing_api_key` — the onboarding skip decision; `tests/setup/onboarding/test_materialization.py::test_completed_onboarding_never_overwrites_a_saved_catalog` — the user catalog is never rewritten |
| `tests/e2e/test_cli_tui_first_run_onboarding.py::test_invalid_credential_first_run_reports_invalid_api_key` | A key the provider rejects (401): onboarding is skipped (the credential is present), the first turn surfaces "Invalid API key" in the TUI, the app stays up and exits cleanly, and the user catalog is never rewritten. | `tests/backend/test_api_key_provenance.py::test_rejected_key_flows_from_producer_through_server_to_tui_without_retry` — a rejected key flows from the backend producer through the server to the TUI as TurnErrorCode.INVALID_API_KEY with the env-var origin, no /retry offered and no retry presentation (the surfaced-error leg through the real TurnSupervisor mount path); `tests/backend/test_api_key_provenance.py::test_missing_key_and_rejected_key_follow_distinct_error_paths` — the rejected-key message contains "Invalid API key" (the exact text the scenario waits for); `tests/cli/test_programmatic_setup.py::test_interactive_mode_still_runs_onboarding_on_missing_api_key` — the onboarding skip decision (credential present); `tests/setup/onboarding/test_materialization.py::test_completed_onboarding_never_overwrites_a_saved_catalog` — the user catalog is never rewritten |

### File-level check

- PTY scenario_population: 8; min(2–3, 8) = 2–3; retained PTY: 2.
- The numeric target is met at its lower bound: the retained pair is the
  file's PTY-only population (the keyboard-driven public first-run flow, both
  parametrizations), so no §6.11 override is needed. The pure unit test of
  the module-local helper is kept with the helper and is not counted in the
  PTY retained target. The file is leaned, not `git rm`ed: the two PTY-only
  journeys and the helper's unit test stay, the six mirrored journeys are
  retired.
