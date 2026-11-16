"""The reviewed capability manifest for every registered app-server method.

Point 1 of v0.7.9 made silent method omissions structurally impossible to ship:
every ``SERVER_METHODS`` entry carries exactly one classification here, and the
manifest refuses to be a declaration of intent. ``test_capability_manifest.py``
fails CI on a missing, duplicate, unknown or unclassified entry, and every
``tests`` node id must resolve to a real test node in this tree.

The four classifications are the design's own (stdio-parity-design §6.1):

``carried``
    The operation crosses to the process that owns its state. On the stdio
    topology it is answered by the root agent through a typed wire carrier;
    in-process it is answered by the same owner behind the request handler.
``host``
    The coordinator's host services answer it identically on both transports
    (persisted sessions, workspace/git probes, usage, config catalogue).
``deliberate_refusal``
    This release keeps a typed refusal on the stdio topology -- the identity
    and history transactions that need a genuine cross-process design -- while
    the in-process route keeps its baseline. The refusal is deliberate and
    documented, never a silent gap.
``unsupported_on_both``
    No handler exists in either topology. The entry records that fact so the
    method is visible and never counted as a parity success.

``tests`` names the executed node(s) that keep each entry honest. It is a
node id (``tests/path.py::test_name``), optionally transport-parameterized; the
manifest test resolves each one against this tree. C5b adds the per-transport
executed-receipt binding on top of these ids.
``acp_only`` marks the one entry (``narration/summarize``) that belongs to the
ACP surface rather than either app-server transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from chartreux.app_server.transport import IN_PROCESS_TRANSPORT, STDIO_TRANSPORT

CARRIED: Final = "carried"
HOST: Final = "host"
DELIBERATE_REFUSAL: Final = "deliberate_refusal"
UNSUPPORTED_ON_BOTH: Final = "unsupported_on_both"

CLASSIFICATIONS: Final = (CARRIED, HOST, DELIBERATE_REFUSAL, UNSUPPORTED_ON_BOTH)

ACP_TRANSPORTS: Final = (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)


@dataclass(frozen=True, slots=True)
class MethodEntry:
    """One ``SERVER_METHODS`` entry and the evidence behind its classification."""

    method: str
    kind: str
    owner: str
    rationale: str
    tests: tuple[str, ...]
    acp_only: bool = False


@dataclass(frozen=True, slots=True)
class AcpExtension:
    """One ACP-only surface mapped onto the app-server method it reaches.

    The ACP layer is a mapping, so a drift there can hide behind the
    ``SERVER_METHODS`` key set unless the mapping itself is pinned. ``tests``
    must equal the C1j mapping-pin ids for the same ACP name.
    """

    name: str
    app_server_method: str | None
    owner: str
    rationale: str
    tests: tuple[str, ...]


def _mapping_pin_ids(module: str, function: str, acp_name: str) -> tuple[str, ...]:
    """The per-transport node ids the C1j mapping pins run under."""
    return tuple(
        f"{module}::{function}[{acp_name}-{transport}]" for transport in ACP_TRANSPORTS
    )


MANIFEST: Final[tuple[MethodEntry, ...]] = (
    MethodEntry(
        method="agent/transcript/get",
        kind=CARRIED,
        owner="chartreux/app_server/_handler.py (agent transcript page reader)",
        rationale=(
            "A child transcript lives where the child runs, so the owner answers "
            "the live page or its saved transcript instead of a coordinator copy."
        ),
        tests=(
            "tests/app_server/test_protocol.py::"
            "test_agent_transcript_uses_attached_backend_extension_route",
        ),
    ),
    MethodEntry(
        method="agents/cancel",
        kind=CARRIED,
        owner="subagent registry (chartreux/core/agents)",
        rationale=(
            "A run-pinned stop is the decision of the process holding the run "
            "handle, so the typed disposition is answered by the owner exactly once."
        ),
        tests=(
            "tests/app_server/test_canonical_methods.py::"
            "test_agents_cancel_lifecycle_allow_policy",
        ),
    ),
    MethodEntry(
        method="agents/inventory/read",
        kind=CARRIED,
        owner="subagent registry (chartreux/core/agents)",
        rationale=(
            "The live inventory is read where the children live: the owner "
            "answers one fresh typed request with the revision and the "
            "projection it sampled together, so no coordinator copy is ever "
            "served, and a dead stdio root answers the death receipt instead "
            "of a stale snapshot."
        ),
        tests=(
            "tests/app_server/backend_contract/test_agent_inventory.py::"
            "test_inventory_read_carries_the_incarnation_and_revision_on_the_"
            "selected_transport",
            "tests/app_server/backend_contract/test_agent_inventory.py::"
            "test_the_stdio_owner_answers_from_the_child_handles_it_spawned",
            "tests/app_server/test_root_process.py::"
            "test_agent_inventory_read_after_root_death_answers_the_death_receipt",
            "tests/app_server/test_root_process.py::"
            "test_a_child_without_the_inventory_capability_is_a_typed_skew_not_a_send",
        ),
    ),
    MethodEntry(
        method="agents/reconcile",
        kind=CARRIED,
        owner="subagent registry (chartreux/core/agents)",
        rationale=(
            "The reconciling barrier belongs to the failed run's owner, so the "
            "evidence is submitted there and its typed disposition carried back."
        ),
        tests=(
            "tests/app_server/test_subagent_management_surface.py::"
            "test_agents_reconcile_route_dispatches_to_registry_reconcile",
        ),
    ),
    MethodEntry(
        method="agents/release",
        kind=CARRIED,
        owner="subagent registry (chartreux/core/agents)",
        rationale=(
            "Release keeps the registry's rails: an active or run-mismatched "
            "child is a conflict that released nothing, judged where it lives."
        ),
        tests=(
            "tests/app_server/test_subagent_management_surface.py::"
            "test_agents_release_route_dispatches_to_guarded_registry_release",
        ),
    ),
    MethodEntry(
        method="agents/resume",
        kind=CARRIED,
        owner="subagent registry (chartreux/core/agents)",
        rationale=(
            "Warm resume is answered by the process that keeps the child's run "
            "records, so an unhonorable disposition stays typed."
        ),
        tests=(
            "tests/app_server/test_subagent_management_surface.py::"
            "test_agents_resume_route_dispatches_to_registry_resume",
        ),
    ),
    MethodEntry(
        method="callback/keepWaiting",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The deadline decision is taken under the request record's lock, so "
            "the answer crosses back from the owner rather than being guessed."
        ),
        tests=(
            "tests/app_server/test_root_wire_dispatch_tables.py::"
            "test_every_agent_request_table_entry_reaches_its_handler",
        ),
    ),
    MethodEntry(
        method="callback/result",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The request record lives in the loop's process, which settles it and "
            "publishes the terminal projection before the waiter unblocks."
        ),
        tests=(
            "tests/app_server/test_protocol.py::"
            "test_late_callback_delivery_error_is_ignored_after_semantic_answer",
        ),
    ),
    MethodEntry(
        method="config/fields/read",
        kind=CARRIED,
        owner="config orchestrator (chartreux/core/config)",
        rationale=(
            "Field wires pair the effective config with the owning process's "
            "layered sources and save revisions, which no coordinator copy holds."
        ),
        tests=(
            "tests/app_server/backend_contract/test_configuration.py::"
            "test_config_fields_read_is_answered_by_the_owning_agent",
        ),
    ),
    MethodEntry(
        method="config/settings/read",
        kind=CARRIED,
        owner="config orchestrator (chartreux/core/config)",
        rationale=(
            "Settings leaves are paired with the user revision carried by the "
            "agent's own layers, so the read crosses to that process."
        ),
        tests=(
            "tests/app_server/backend_contract/test_configuration.py::"
            "test_settings_read_pairs_leaves_with_the_user_revision",
        ),
    ),
    MethodEntry(
        method="config/proxy/read",
        kind=CARRIED,
        owner="config orchestrator (chartreux/core/config)",
        rationale=(
            "The proxy view is read from the env file the agent process resolves, "
            "so the value shown is the value that process uses."
        ),
        tests=(
            "tests/app_server/backend_contract/test_configuration.py::"
            "test_proxy_settings_round_trip_where_the_env_file_lives",
        ),
    ),
    MethodEntry(
        method="config/proxy/write",
        kind=CARRIED,
        owner="config orchestrator (chartreux/core/config)",
        rationale=(
            "The write lands in the agent process's env file; a refused value and "
            "a busy session keep the in-process refusal and conflict codes."
        ),
        tests=(
            "tests/app_server/backend_contract/test_configuration.py::"
            "test_rejected_proxy_write_keeps_the_in_process_param_refusal",
        ),
    ),
    MethodEntry(
        method="config/read",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "The shipped read is the host-scoped catalogue projection, answered "
            "identically on both transports; the session-scoped variant is not a "
            "shipped client path."
        ),
        tests=(
            "tests/app_server/test_local_spawn.py::"
            "test_config_read_serves_the_catalogue_with_and_without_a_session",
        ),
    ),
    MethodEntry(
        method="config/reload",
        kind=CARRIED,
        owner="config orchestrator (chartreux/core/config)",
        rationale=(
            "Convergence runs where the runtime lives: the agent prepares, "
            "reloads and finishes the MCP legs, and the coordinator relays the receipt."
        ),
        tests=(
            "tests/app_server/test_session_reload.py::"
            "test_reload_applies_route_edit_and_publishes_the_applied_notice",
        ),
    ),
    MethodEntry(
        method="config/policy/read",
        kind=CARRIED,
        owner="policy stores (chartreux/core/policy)",
        rationale=(
            "The serialized session policy is composed by the loop that enforces "
            "it, so the read is answered from that owner."
        ),
        tests=(
            "tests/app_server/test_tree_policy.py::test_public_serialized_session_only_policy",
        ),
    ),
    MethodEntry(
        method="config/policy/replace",
        kind=CARRIED,
        owner="policy stores (chartreux/core/policy)",
        rationale=(
            "A policy replacement is published from the loop that enforces it; "
            "the coordinator adopts the resulting runtime view."
        ),
        tests=(
            "tests/app_server/test_tree_policy.py::test_public_serialized_session_only_policy",
        ),
    ),
    MethodEntry(
        method="policy/dangerous_commands/read",
        kind=CARRIED,
        owner="dangerous-command authority (chartreux/core/dangerous_command_grants.py)",
        rationale=(
            "The authority generation lives with the loop's grant store, so the "
            "read crosses to it instead of a coordinator replica."
        ),
        tests=(
            "tests/app_server/backend_contract/test_dangerous_command_permissions.py::"
            "test_authority_param_and_session_refusals_on_the_selected_transport",
        ),
    ),
    MethodEntry(
        method="policy/dangerous_commands/revoke",
        kind=CARRIED,
        owner="dangerous-command authority (chartreux/core/dangerous_command_grants.py)",
        rationale=(
            "Revoke names the generation observed in the read; a superseded or "
            "mid-execution-consumed generation is the typed conflict outcome."
        ),
        tests=(
            "tests/app_server/backend_contract/test_dangerous_command_permissions.py::"
            "test_revocation_requires_the_literal_user_initiation",
        ),
    ),
    MethodEntry(
        method="policy/roots/read",
        kind=CARRIED,
        owner="agent roots (chartreux/core/workspace)",
        rationale=(
            "The authorized roots are composed by the agent's own source, so the "
            "projection is answered where that source lives."
        ),
        tests=(
            "tests/app_server/test_roots_read.py::"
            "test_roots_read_projects_saved_effective_and_dynamic_roots",
        ),
    ),
    MethodEntry(
        method="policy/roots/replace",
        kind=DELIBERATE_REFUSAL,
        owner="agent roots (chartreux/core/workspace)",
        rationale=(
            "A roots replacement rewrites the set every node enforces and has no "
            "cross-process publication path in this release; stdio refuses typed "
            "while the in-process route keeps its baseline."
        ),
        tests=(
            "tests/app_server/test_roots_read.py::"
            "test_roots_read_projects_saved_effective_and_dynamic_roots",
        ),
    ),
    MethodEntry(
        method="policy/roots/save",
        kind=CARRIED,
        owner="agent roots (chartreux/core/workspace)",
        rationale=(
            "A root grant is persisted where the agent's config lives; save "
            "remains supported and must not be confused with replace."
        ),
        tests=(
            "tests/app_server/test_root_save.py::"
            "test_save_requires_user_initiation_and_a_registered_session",
        ),
    ),
    MethodEntry(
        method="config/schema",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "The runtime JSON schema is a host catalogue served identically on "
            "both transports and needs no session state."
        ),
        tests=(
            "tests/acp/test_config_schema.py::test_config_schema_returns_runtime_schema",
        ),
    ),
    MethodEntry(
        method="config/write",
        kind=CARRIED,
        owner="config orchestrator (chartreux/core/config)",
        rationale=(
            "Writes apply in the owning agent and report its runtime, so both "
            "transports share one transaction outcome."
        ),
        tests=(
            "tests/app_server/test_config_write.py::"
            "test_real_wire_targets_and_catalog_write_enforcement",
        ),
    ),
    MethodEntry(
        method="diagnostics/list",
        kind=CARRIED,
        owner="diagnostics owners (chartreux/core/log_reader.py)",
        rationale=(
            "Session-scoped diagnostics are read where the loop and log live, "
            "with unchanged bounds and redaction."
        ),
        tests=(
            "tests/app_server/test_protocol.py::"
            "test_diagnostics_reads_cross_to_the_agent_that_owns_them",
        ),
    ),
    MethodEntry(
        method="diagnostics/logs/read",
        kind=CARRIED,
        owner="diagnostics owners (chartreux/core/log_reader.py)",
        rationale=(
            "The debug log page is produced by the process that writes it; no "
            "cross-process merge without provenance."
        ),
        tests=(
            "tests/app_server/test_protocol.py::"
            "test_diagnostics_reads_cross_to_the_agent_that_owns_them",
        ),
    ),
    MethodEntry(
        method="events/read",
        kind=HOST,
        owner="coordinator event stream (chartreux/app_server/server.py)",
        rationale=(
            "The v0.1 contract answers an empty replay batch on both transports; "
            "live updates travel as notifications."
        ),
        tests=(
            "tests/app_server/test_events_read.py::test_events_read_returns_empty_batch",
        ),
    ),
    MethodEntry(
        method="session/history/get",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "The persisted history page is deliberately always host-routed on "
            "both transports, so its routing is parity-safe by construction."
        ),
        tests=(
            "tests/app_server/test_session_spawn.py::"
            "test_session_history_get_returns_projected_entries_and_respects_limit",
        ),
    ),
    MethodEntry(
        method="identity/read",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The identity is session-scoped and resolved against the agent's own "
            "loop, answering a typed absent identity where no backend is bound."
        ),
        tests=(
            "tests/acp/test_identity.py::test_identity_read_answers_on_both_transports",
        ),
    ),
    MethodEntry(
        method="loops/clear",
        kind=CARRIED,
        owner="loop scheduler (chartreux/core/loop.py)",
        rationale=(
            "The schedule lives in the loop's process, so no second scheduler "
            "exists in the coordinator."
        ),
        tests=(
            "tests/app_server/backend_contract/test_loops.py::"
            "test_loop_carrier_answers_from_the_process_that_holds_the_schedule",
        ),
    ),
    MethodEntry(
        method="loops/create",
        kind=CARRIED,
        owner="loop scheduler (chartreux/core/loop.py)",
        rationale=(
            "Creation preserves idle admission and persistence semantics in the "
            "process that holds the schedule."
        ),
        tests=(
            "tests/app_server/backend_contract/test_loops.py::"
            "test_loop_carrier_answers_from_the_process_that_holds_the_schedule",
        ),
    ),
    MethodEntry(
        method="loops/delete",
        kind=CARRIED,
        owner="loop scheduler (chartreux/core/loop.py)",
        rationale=(
            "Deletion mutates the owner's schedule, so the typed loop state comes "
            "back from that owner."
        ),
        tests=(
            "tests/app_server/backend_contract/test_loops.py::"
            "test_loop_carrier_answers_from_the_process_that_holds_the_schedule",
        ),
    ),
    MethodEntry(
        method="loops/list",
        kind=CARRIED,
        owner="loop scheduler (chartreux/core/loop.py)",
        rationale=(
            "The list is the owner's schedule projection, identical on both transports."
        ),
        tests=(
            "tests/app_server/backend_contract/test_loops.py::"
            "test_loop_carrier_answers_from_the_process_that_holds_the_schedule",
        ),
    ),
    MethodEntry(
        method="mcp_catalog/add",
        kind=CARRIED,
        owner="MCP catalog owner (chartreux/app_server/mcp_catalog.py)",
        rationale=(
            "The agent owns the single orchestrator; a session-addressed mutation "
            "persists through it under the typed owner key."
        ),
        tests=(
            "tests/app_server/backend_contract/test_mcp_catalog.py::"
            "test_mcp_catalog_add_login_logout_and_notification_order",
        ),
    ),
    MethodEntry(
        method="mcp_catalog/login",
        kind=CARRIED,
        owner="MCP catalog owner (chartreux/app_server/mcp_catalog.py)",
        rationale=(
            "The catalog owner and the auth binding are the agent orchestrator's "
            "object identity, keyed across the wire instead of replicated."
        ),
        tests=(
            "tests/app_server/backend_contract/test_mcp_catalog.py::"
            "test_mcp_catalog_login_relays_auth_url_on_both_transports",
        ),
    ),
    MethodEntry(
        method="mcp_catalog/logout",
        kind=CARRIED,
        owner="MCP catalog owner (chartreux/app_server/mcp_catalog.py)",
        rationale=(
            "Logout removes the owner-side credentials under the same key that "
            "the login bound."
        ),
        tests=(
            "tests/app_server/backend_contract/test_mcp_catalog.py::"
            "test_mcp_catalog_add_login_logout_and_notification_order",
        ),
    ),
    MethodEntry(
        method="mcp_catalog/read",
        kind=CARRIED,
        owner="MCP catalog owner (chartreux/app_server/mcp_catalog.py)",
        rationale=(
            "The catalog projection enumerates the agent orchestrator's sources, "
            "and the coordinator holds only the projected view."
        ),
        tests=(
            "tests/app_server/backend_contract/test_mcp_catalog.py::"
            "test_mcp_catalog_owner_side_carriers_on_both_transports",
        ),
    ),
    MethodEntry(
        method="mcp_catalog/refresh",
        kind=CARRIED,
        owner="MCP catalog owner (chartreux/app_server/mcp_catalog.py)",
        rationale=(
            "Refresh converges the agent's own runtime and returns the typed "
            "partial-convergence receipt."
        ),
        tests=(
            "tests/app_server/backend_contract/test_mcp_catalog.py::"
            "test_mcp_catalog_read_refresh_toggle_remove",
        ),
    ),
    MethodEntry(
        method="mcp_catalog/remove",
        kind=CARRIED,
        owner="MCP catalog owner (chartreux/app_server/mcp_catalog.py)",
        rationale=(
            "Removal persists through the agent orchestrator as a typed, "
            "revision-stamped request."
        ),
        tests=(
            "tests/app_server/backend_contract/test_mcp_catalog.py::"
            "test_mcp_catalog_read_refresh_toggle_remove",
        ),
    ),
    MethodEntry(
        method="mcp_catalog/toggle",
        kind=CARRIED,
        owner="MCP catalog owner (chartreux/app_server/mcp_catalog.py)",
        rationale=(
            "Enablement is a source mutation owned by the agent orchestrator, "
            "with turn-active fencing preserved."
        ),
        tests=(
            "tests/app_server/backend_contract/test_mcp_catalog.py::"
            "test_mcp_catalog_read_refresh_toggle_remove",
        ),
    ),
    MethodEntry(
        method="narration/summarize",
        kind=UNSUPPORTED_ON_BOTH,
        owner="ACP narration surface",
        rationale=(
            "ACP-only: neither app-server topology registers a handler for it, so "
            "it is classified explicitly and never counted as a parity success."
        ),
        tests=(
            "tests/app_server/test_capability_manifest.py::"
            "test_narration_summarize_is_unsupported_on_both",
        ),
        acp_only=True,
    ),
    MethodEntry(
        method="recovery/root/admit",
        kind=CARRIED,
        owner="recovery gate (chartreux/core/recovery_models.py)",
        rationale=(
            "The gate stays coordinator-local but re-reads the loop-owned "
            "evidence over the wire at admission, never auto-replaying."
        ),
        tests=(
            "tests/app_server/test_protocol.py::"
            "test_recovery_root_carriers_answer_identically_in_process",
            "tests/app_server/test_protocol.py::"
            "test_recovery_root_carriers_reach_the_gate_on_the_stdio_transport",
        ),
    ),
    MethodEntry(
        method="recovery/root/cancel",
        kind=CARRIED,
        owner="recovery gate (chartreux/core/recovery_models.py)",
        rationale=(
            "Cancellation reads the coordinator-local session view and episode "
            "and never cancels an already-admitted successor."
        ),
        tests=(
            "tests/app_server/test_protocol.py::"
            "test_recovery_root_carriers_answer_identically_in_process",
            "tests/app_server/test_protocol.py::"
            "test_recovery_root_carriers_reach_the_gate_on_the_stdio_transport",
        ),
    ),
    MethodEntry(
        method="recovery/root/schedule",
        kind=CARRIED,
        owner="recovery gate (chartreux/core/recovery_models.py)",
        rationale=(
            "The typed terminal classification arrives on the agent's turn frame "
            "and the gate re-reads the full evidence set before scheduling."
        ),
        tests=(
            "tests/app_server/test_protocol.py::"
            "test_recovery_root_carriers_answer_identically_in_process",
            "tests/app_server/test_protocol.py::"
            "test_recovery_root_carriers_reach_the_gate_on_the_stdio_transport",
        ),
    ),
    MethodEntry(
        method="review/approve",
        kind=CARRIED,
        owner="review resource (chartreux/app_server/_review.py)",
        rationale=(
            "The review mutation runs where the checkout and its ownership state "
            "live, so no coordinator copy of the review applies it."
        ),
        tests=(
            "tests/acp/test_review.py::test_review_mutations_apply_on_both_transports",
        ),
    ),
    MethodEntry(
        method="review/baseline",
        kind=CARRIED,
        owner="review resource (chartreux/app_server/_review.py)",
        rationale=(
            "The baseline is computed against the owner's checkout, so the read "
            "crosses to that owner."
        ),
        tests=(
            "tests/acp/test_review.py::test_review_reads_answer_on_both_transports",
        ),
    ),
    MethodEntry(
        method="review/hunks",
        kind=CARRIED,
        owner="review resource (chartreux/app_server/_review.py)",
        rationale=(
            "Hunk pages are derived from the owner's working tree and cross as "
            "the same typed read on both transports."
        ),
        tests=(
            "tests/acp/test_review.py::test_review_reads_answer_on_both_transports",
        ),
    ),
    MethodEntry(
        method="review/revert",
        kind=CARRIED,
        owner="review resource (chartreux/app_server/_review.py)",
        rationale=(
            "Revert mutates the owner's checkout, so identity and ownership "
            "checks stay where the files are."
        ),
        tests=(
            "tests/acp/test_review.py::test_review_mutations_apply_on_both_transports",
        ),
    ),
    MethodEntry(
        method="review/state",
        kind=CARRIED,
        owner="review resource (chartreux/app_server/_review.py)",
        rationale=(
            "The review state crosses as its own method rather than riding a "
            "session read, so both transports answer the same view."
        ),
        tests=(
            "tests/acp/test_review.py::"
            "test_review_state_crosses_as_its_own_method_on_both_transports",
        ),
    ),
    MethodEntry(
        method="review/turnDiff",
        kind=CARRIED,
        owner="review resource (chartreux/app_server/_review.py)",
        rationale=(
            "The turn diff is computed by the owner from its own turn state and "
            "returned as the typed read."
        ),
        tests=(
            "tests/acp/test_review.py::test_review_reads_answer_on_both_transports",
        ),
    ),
    MethodEntry(
        method="runtime/read",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Every read asks the agent for a fresh snapshot and adopts it against "
            "the shared agent-sequence cursor, so a dead root answers the death "
            "receipt instead of a cached ready=true."
        ),
        tests=(
            "tests/app_server/test_root_process.py::"
            "test_runtime_read_serves_a_fresh_agent_snapshot_while_live",
        ),
    ),
    MethodEntry(
        method="session/compact",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Compaction runs where the transcript lives and publishes the "
            "compacted projection both transports adopt."
        ),
        tests=(
            "tests/app_server/test_compact_lifecycle_races.py::"
            "test_ordinary_input_and_a_second_compact_conflict_while_held",
        ),
    ),
    MethodEntry(
        method="session/continue",
        kind=HOST,
        owner="coordinator session lifecycle (chartreux/app_server/server.py)",
        rationale=(
            "The host opens the latest persisted session through the backend "
            "factory, so both transports share one lifecycle route."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_continue_attaches_the_latest_persisted_session_and_keeps_it_usable",
        ),
    ),
    MethodEntry(
        method="session/context/inject",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "An injection lands between turns in the loop that owns admission, "
            "and its after-response promotion is the owner's."
        ),
        tests=(
            "tests/app_server/backend_contract/test_turns.py::"
            "test_context_injection_adds_a_public_user_message",
        ),
    ),
    MethodEntry(
        method="session/delete",
        kind=HOST,
        owner="coordinator session lifecycle (chartreux/app_server/server.py)",
        rationale=(
            "Deletion is a host lifecycle operation on persisted state, answered "
            "identically on both transports."
        ),
        tests=(
            "tests/acp/test_agent_methods.py::"
            "test_delete_session_remains_retryable_after_cleanup_failure",
        ),
    ),
    MethodEntry(
        method="session/fork",
        kind=DELIBERATE_REFUSAL,
        owner="coordinator session lifecycle (chartreux/app_server/server.py)",
        rationale=(
            "Fork's identity transition needs a cross-process transaction this "
            "release does not design; stdio answers typed NOT_IMPLEMENTED while "
            "the in-process route keeps its baseline."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_fork_attaches_the_public_child_session",
            "tests/app_server/test_production_entrypoint.py::"
            "test_a_typed_agent_refusal_keeps_its_protocol_code",
        ),
    ),
    MethodEntry(
        method="session/history/clear",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The clear runs where the loop lives and returns the state it leaves "
            "behind; it has its own reconfigure route, not a refusal."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_history_clear_public_behavior_on_both_transports",
        ),
    ),
    MethodEntry(
        method="session/history/list",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "A live session's history is the agent's projection; the host route "
            "remains only for saved sessions and never silently serves an "
            "attached id."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_history_list_serves_the_live_projection_on_both_transports",
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_history_list_dead_but_attached_answers_the_death_receipt",
        ),
    ),
    MethodEntry(
        method="session/list",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "The catalog projection is a host read of persisted sessions and "
            "paginates identically on both transports."
        ),
        tests=(
            "tests/app_server/backend_contract/test_host_lifecycle.py::"
            "test_host_catalog_paginates_persisted_sessions_and_selects_the_latest",
        ),
    ),
    MethodEntry(
        method="session/log/read",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The session-log projection is produced from the agent's own snapshot, "
            "so the read is answered where the log lives."
        ),
        tests=(
            "tests/app_server/test_production_entrypoint.py::"
            "test_a_snapshot_read_reproduces_the_recovery_gate_decision",
        ),
    ),
    MethodEntry(
        method="session/read",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "The public snapshot read is answered by the session backend host on "
            "both transports, with the root's own state carried in it."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_session_read_returns_the_public_snapshot_and_event_watermark",
        ),
    ),
    MethodEntry(
        method="session/ready/read",
        kind=CARRIED,
        owner="process session backend (chartreux/app_server/process_session_backend.py)",
        rationale=(
            "Readiness is immediately ready only while the current generation is "
            "known live; after death the read answers the death receipt."
        ),
        tests=(
            "tests/app_server/test_root_process.py::"
            "test_ready_read_answers_ready_only_while_the_root_is_live",
        ),
    ),
    MethodEntry(
        method="session/ready/wait",
        kind=CARRIED,
        owner="process session backend (chartreux/app_server/process_session_backend.py)",
        rationale=(
            "The wait settles once the open handshake established the loop, and a "
            "dead generation answers the death receipt rather than a stale ready."
        ),
        tests=(
            "tests/app_server/test_session_spawn.py::"
            "test_ready_wait_response_carries_init_duration_ms",
        ),
    ),
    MethodEntry(
        method="session/relocate",
        kind=DELIBERATE_REFUSAL,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "A live session's workspace move needs a cross-process transaction "
            "this release does not design; stdio refuses typed before a frame is "
            "sent, while the unattached host path keeps its record-only behavior."
        ),
        tests=(
            "tests/app_server/test_relocate.py::"
            "test_a_process_root_refuses_a_relocate_before_a_frame_is_sent",
        ),
    ),
    MethodEntry(
        method="session/rename",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "A manual title is persisted by the session backend host on both "
            "transports; it is the wire method the dead title alias named."
        ),
        tests=(
            "tests/app_server/backend_contract/test_host_lifecycle.py::"
            "test_host_renames_the_attached_session",
        ),
    ),
    MethodEntry(
        method="session/resume",
        kind=HOST,
        owner="coordinator session lifecycle (chartreux/app_server/server.py)",
        rationale=(
            "Resume retires a dead backend, spawns one replacement and surfaces "
            "the death receipt and memorial state from the host lifecycle."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_resume_after_death_carries_receipt_and_memorial_on_both_transports",
        ),
    ),
    MethodEntry(
        method="session/rewind",
        kind=DELIBERATE_REFUSAL,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The rewind transaction is wire-deferred in this release and answers "
            "a typed refusal that names the workable route, never a silent gap."
        ),
        tests=(
            "tests/acp/test_agent_methods.py::test_rewind_refusal_is_worded_for_an_rpc_caller",
        ),
    ),
    MethodEntry(
        method="session/rewind/read",
        kind=DELIBERATE_REFUSAL,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The rewind preview shares the deferred transaction, so its refusal "
            "carries the same typed category as the rewind itself."
        ),
        tests=("tests/acp/test_agent_methods.py::test_rewind_mapping_pins",),
    ),
    MethodEntry(
        method="session/safety/read",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The SafetyLevel lives on the loop that enforces it, so the read "
            "crosses to the owning process on stdio and answers in-process "
            "elsewhere; a dead root answers the death receipt, never a stale level."
        ),
        tests=(
            "tests/cli/test_safety_command.py::TestSafetyWireModels::"
            "test_read_models_round_trip",
        ),
    ),
    MethodEntry(
        method="session/safety/transition",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The transition is gated where the loop lives (idle-only, writable "
            "audit sink, background jobs on revoke), so the decision crosses to "
            "the owning process rather than being made from a coordinator copy."
        ),
        tests=(
            "tests/cli/test_safety_command.py::TestSafetyWireModels::"
            "test_transition_params_round_trip",
        ),
    ),
    MethodEntry(
        method="session/settings/update",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Budget overrides apply to the loop that enforces them and return an "
            "empty typed response on both transports."
        ),
        tests=(
            "tests/app_server/test_session_backend_port.py::"
            "test_session_backend_contract_covers_the_complete_session_lifecycle",
        ),
    ),
    MethodEntry(
        method="session/shellCommand",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "A manual shell command queues as shell input and the agent process "
            "answers its record at idle, including redaction and caps."
        ),
        tests=(
            "tests/app_server/test_canonical_methods.py::"
            "test_shell_command_returns_direct_acknowledgement",
        ),
    ),
    MethodEntry(
        method="session/start",
        kind=HOST,
        owner="coordinator session lifecycle (chartreux/app_server/server.py)",
        rationale=(
            "The host opens the session through the backend factory, so the "
            "transport only chooses which backend kind is constructed."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_session_start_returns_the_attached_public_session",
        ),
    ),
    MethodEntry(
        method="session/stop",
        kind=CARRIED,
        owner="coordinator session lifecycle (chartreux/app_server/server.py)",
        rationale=(
            "The coordinator owns the turn queue and callback projection, so stop "
            "settles queued work before the ack and delivers every receipt before "
            "the connection closes."
        ),
        tests=(
            "tests/app_server/backend_contract/test_session_lifecycle.py::"
            "test_session_stop_settles_queued_work_then_acks_on_both_transports",
        ),
    ),
    MethodEntry(
        method="session/title/update",
        kind=UNSUPPORTED_ON_BOTH,
        owner="coordinator session lifecycle (chartreux/app_server/server.py)",
        rationale=(
            "A retained alias with no handler in either topology: the wire method "
            "is session/rename, and the alias answers typed NOT_IMPLEMENTED on "
            "both transports before any session routing."
        ),
        tests=(
            "tests/app_server/test_canonical_methods.py::"
            "test_session_title_update_alias_is_typed_unsupported_in_process",
            "tests/app_server/test_canonical_methods.py::"
            "test_session_title_update_alias_is_typed_unsupported_on_stdio",
        ),
    ),
    MethodEntry(
        method="session/turns/list",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The turn page is the live projection, answered through the shared "
            "helper so both topologies page identically."
        ),
        tests=(
            "tests/acp/test_agent_methods.py::"
            "test_session_turns_list_pages_the_live_projection_on_both_transports",
        ),
    ),
    MethodEntry(
        method="shell/interrupt",
        kind=UNSUPPORTED_ON_BOTH,
        owner="shell owner (chartreux/app_server/_shell.py)",
        rationale=(
            "No handler exists in either topology; the manual-command carrier is "
            "session/shellCommand, so this entry is recorded and never counted as "
            "a parity success."
        ),
        tests=(
            "tests/app_server/test_capability_manifest.py::"
            "test_shell_interrupt_is_unsupported_on_both",
        ),
    ),
    MethodEntry(
        method="shell/run",
        kind=UNSUPPORTED_ON_BOTH,
        owner="shell owner (chartreux/app_server/_shell.py)",
        rationale=(
            "No handler exists in either topology; shell execution reaches the "
            "user flow through session/shellCommand instead."
        ),
        tests=(
            "tests/app_server/test_capability_manifest.py::"
            "test_shell_run_is_unsupported_on_both",
        ),
    ),
    MethodEntry(
        method="skills/installed",
        kind=CARRIED,
        owner="skills manager (chartreux/core/skills)",
        rationale=(
            "Installed skills are enumerated by the owner's manager, so the same "
            "catalogue answers on both transports."
        ),
        tests=(
            "tests/acp/test_agent_methods.py::"
            "test_latent_catalog_reads_answer_on_both_transports",
        ),
    ),
    MethodEntry(
        method="skills/list",
        kind=CARRIED,
        owner="skills manager (chartreux/core/skills)",
        rationale=(
            "The discovered-skill catalogue is the owner's, carried as a typed "
            "read with the same validation on both routes."
        ),
        tests=(
            "tests/app_server/backend_contract/test_skills.py::"
            "test_skills_list_answers_with_the_same_catalogue",
        ),
    ),
    MethodEntry(
        method="stats/read",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Step and token statistics live on the loop that accumulates them, so "
            "the read crosses to it."
        ),
        tests=(
            "tests/acp/test_agent_methods.py::"
            "test_latent_catalog_reads_answer_on_both_transports",
        ),
    ),
    MethodEntry(
        method="tools/list",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The available tool inventory belongs to the loop's tool manager and "
            "is carried as the same typed read on both transports."
        ),
        tests=(
            "tests/acp/test_agent_methods.py::"
            "test_latent_catalog_reads_answer_on_both_transports",
        ),
    ),
    MethodEntry(
        method="app_server/session/turn/enqueue",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The queue is admission state of the loop, and both transports "
            "enqueue through the same backend operation."
        ),
        tests=(
            "tests/app_server/test_session_interrupt.py::"
            "test_enqueue_while_idle_promotes_without_leaving_stale_queue_state",
        ),
    ),
    MethodEntry(
        method="app_server/session/turn/queue/read",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "The queue projection is read from the backend that owns admission, "
            "never reconstructed coordinator-side."
        ),
        tests=(
            "tests/app_server/test_session_interrupt.py::"
            "test_queue_read_does_not_restore_an_item_removed_by_a_newer_event",
        ),
    ),
    MethodEntry(
        method="app_server/session/turn/queue/remove",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Removal reports concurrent promotion from the owner that performed "
            "the mutation."
        ),
        tests=(
            "tests/app_server/test_session_interrupt.py::"
            "test_remove_queued_turn_reports_concurrent_promotion",
        ),
    ),
    MethodEntry(
        method="app_server/session/turn/queue/replace",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Replacement preserves entry identity and FIFO order inside the "
            "queue owner."
        ),
        tests=(
            "tests/app_server/test_session_interrupt.py::"
            "test_replace_queued_turn_preserves_identity_and_fifo_order",
        ),
    ),
    MethodEntry(
        method="app_server/session/turn/queue/resume",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Resume promotes retained work from the queue owner, the only place "
            "that knows what was held."
        ),
        tests=(
            "tests/app_server/test_session_interrupt.py::"
            "test_interrupt_pauses_queue_until_remove_and_resume",
        ),
    ),
    MethodEntry(
        method="turn/interrupt",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Interrupt terminates the turn where it runs and is exempt from the "
            "lifecycle conflict so a stop stays reachable."
        ),
        tests=(
            "tests/app_server/backend_contract/test_turns.py::"
            "test_interrupt_terminates_an_active_public_turn",
        ),
    ),
    MethodEntry(
        method="turn/start",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Turn execution and streaming happen in the loop's process, so the "
            "public event stream is the owner's."
        ),
        tests=(
            "tests/app_server/backend_contract/test_turns.py::"
            "test_turn_streams_public_events_over_json_rpc",
        ),
    ),
    MethodEntry(
        method="turn/steer",
        kind=CARRIED,
        owner="agent loop (chartreux/core/agent_loop)",
        rationale=(
            "Steering arbitrates one delivery against a turn in flight inside the "
            "loop, with receipt replay and pre-commit rejection."
        ),
        tests=(
            "tests/app_server/backend_contract/test_turns.py::"
            "test_steering_adds_a_public_user_message_to_an_active_turn",
        ),
    ),
    MethodEntry(
        method="usage/read",
        kind=HOST,
        owner="coordinator usage service (chartreux/core/usage.py)",
        rationale=(
            "Usage is answered by the coordinator's usage service from persisted "
            "accounting on both transports, with a pre-session shape."
        ),
        tests=(
            "tests/app_server/test_usage_read.py::test_pre_session_read_and_response_shape",
        ),
    ),
    MethodEntry(
        method="workspace/git/branch",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "A local branch probe is best-effort host work with no session state "
            "and is answered identically on both transports."
        ),
        tests=(
            "tests/app_server/test_canonical_methods.py::"
            "test_workspace_branch_canonical_method",
        ),
    ),
    MethodEntry(
        method="workspace/git/checkouts",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "Checkout identity is a passive host probe of the repository, so it "
            "never opens a runtime."
        ),
        tests=(
            "tests/app_server/test_local_spawn.py::test_passive_host_reads_the_main_checkout",
        ),
    ),
    MethodEntry(
        method="workspace/git/worktrees/list",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "Linked worktrees are listed by the passive host probe, identically "
            "on both transports."
        ),
        tests=(
            "tests/app_server/test_local_spawn.py::"
            "test_passive_host_lists_linked_worktrees",
        ),
    ),
    MethodEntry(
        method="workspace/git/worktrees/remove",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "Removing a managed worktree is host lifecycle work that saves the "
            "work it removes and ignores unmanaged trees."
        ),
        tests=(
            "tests/app_server/test_local_shutdown.py::"
            "test_worktree_remove_deletes_a_clean_managed_worktree",
        ),
    ),
    MethodEntry(
        method="workspace/prompt/prepare",
        kind=CARRIED,
        owner="workspace preparation (chartreux/core/workspace)",
        rationale=(
            "Preparation depends on the live model, cwd, session directory and "
            "attachment snapshots, so it runs in the agent process and a timed-out "
            "preparation is never blindly retried."
        ),
        tests=(
            "tests/app_server/backend_contract/test_preparation.py::"
            "test_prompt_prepare_returns_the_prepared_prompt_on_the_selected_transport",
        ),
    ),
    MethodEntry(
        method="workspace/trust/decision",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "A trust decision is recorded by the host's trust store from the "
            "caller's cwd and is answered identically on both transports."
        ),
        tests=(
            "tests/app_server/test_workspace.py::"
            "test_decide_workspace_trust_preserves_validated_identity",
        ),
    ),
    MethodEntry(
        method="workspace/trust/untrustedConfig",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "The untrusted-config-folder listing is a passive host read of the "
            "workspace's trust state."
        ),
        tests=(
            "tests/app_server/test_workspace.py::"
            "test_read_untrusted_config_dirs_reports_dirs_and_settings_path",
        ),
    ),
    MethodEntry(
        method="workspace/trust/status",
        kind=HOST,
        owner="coordinator host handler (chartreux/app_server/_host.py)",
        rationale=(
            "The trust status is a passive host projection of the trust store, "
            "shared by both transports."
        ),
        tests=(
            "tests/app_server/test_workspace.py::"
            "test_group_b_consumer_never_reswallows_an_unusable_trust_store",
        ),
    ),
)


ACP_EXTENSIONS: Final[tuple[AcpExtension, ...]] = (
    AcpExtension(
        name="logLevel/read",
        app_server_method=None,
        owner="ACP agent (chartreux/acp/agent.py)",
        rationale=(
            "Answered from this process's own level chain; no app-server method "
            "is reached, and the pin asserts no request crosses."
        ),
        tests=_mapping_pin_ids(
            "tests/acp/test_log_level.py",
            "test_log_level_mapping_pins",
            "logLevel/read",
        ),
    ),
    AcpExtension(
        name="logLevel/write",
        app_server_method="config/write",
        owner="ACP agent (chartreux/acp/agent.py)",
        rationale=(
            "A config-level write crosses as config/write; the session override "
            "never crosses the wire."
        ),
        tests=_mapping_pin_ids(
            "tests/acp/test_log_level.py",
            "test_log_level_mapping_pins",
            "logLevel/write",
        ),
    ),
    AcpExtension(
        name="rewind/preview",
        app_server_method="session/rewind/read",
        owner="ACP agent (chartreux/acp/agent.py)",
        rationale=(
            "The ACP preview maps onto the rewind-read method, whose stdio "
            "refusal is the deliberate one recorded above."
        ),
        tests=_mapping_pin_ids(
            "tests/acp/test_agent_methods.py",
            "test_rewind_mapping_pins",
            "rewind/preview",
        ),
    ),
    AcpExtension(
        name="rewind/to",
        app_server_method="session/rewind",
        owner="ACP agent (chartreux/acp/agent.py)",
        rationale=(
            "The ACP rewind maps onto the deferred session/rewind method and "
            "inherits its typed refusal wording for RPC callers."
        ),
        tests=_mapping_pin_ids(
            "tests/acp/test_agent_methods.py", "test_rewind_mapping_pins", "rewind/to"
        ),
    ),
)
