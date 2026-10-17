"""Collected host for the shared grant-lifecycle twins harness (W6b, stage 15a).

The parametrized suite lives in ``_grant_lifecycle_harness.py``; this module
re-exports its cases so pytest collects them under stable node IDs of the form
``tests/agent_loop/test_grant_lifecycle_harness.py::<case>[<feature>-...]``.
Commit 15a-i introduces the harness with the smoke case only (zero deletions);
the migration commits add the cases they migrate and delete the twin tests
they replace — 15a-ii the ``test_dynamic_root_grants.py`` side, 15a-iii the
``test_dangerous_command_lifecycle.py`` side. The per-deletion survival
mapping is recorded in the stage-15 report.
"""

from __future__ import annotations

from tests.agent_loop._grant_lifecycle_harness import (
    test_allow_session_grant_resumes_the_tool_call as test_allow_session_grant_resumes_the_tool_call,
    test_always_applies_the_grant_and_reports_the_saved_outcome as test_always_applies_the_grant_and_reports_the_saved_outcome,
    test_always_fetches_the_user_revision_at_decision_time as test_always_fetches_the_user_revision_at_decision_time,
    test_always_reports_a_conflict_without_retrying as test_always_reports_a_conflict_without_retrying,
    test_always_reports_save_failures_honestly as test_always_reports_save_failures_honestly,
    test_always_without_a_readable_saved_surface_skips_the_save as test_always_without_a_readable_saved_surface_skips_the_save,
    test_always_without_a_save_transport_reports_it as test_always_without_a_save_transport_reports_it,
    test_bare_loop_without_a_port_keeps_the_plain_denial as test_bare_loop_without_a_port_keeps_the_plain_denial,
    test_cancelled_prompt_denies_and_records_refusal as test_cancelled_prompt_denies_and_records_refusal,
    test_child_session_is_bounded_and_never_prompts as test_child_session_is_bounded_and_never_prompts,
    test_concurrent_calls_coalesce_or_need_separate_approvals as test_concurrent_calls_coalesce_or_need_separate_approvals,
    test_deny_skips_the_tool_call_and_remembers_the_refusal as test_deny_skips_the_tool_call_and_remembers_the_refusal,
    test_harness_smoke_grant_lifecycle_round_trip as test_harness_smoke_grant_lifecycle_round_trip,
    test_noninteractive_session_denies_with_remediation as test_noninteractive_session_denies_with_remediation,
    test_prompt_shape_states_access_and_choices as test_prompt_shape_states_access_and_choices,
    test_session_grant_survives_compaction as test_session_grant_survives_compaction,
)
