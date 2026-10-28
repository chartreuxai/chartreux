"""Completion-coordination collaborator for AgentLoop (ALL cluster: attempts).

Owns the completion input/header/resource assembly, the failover candidate
and attempt helpers, the synchronous attempt publication/rollback helpers,
the entire attempt generator (with its cleanup paths), and the
nonstreaming completion facade, all extracted from _loop.py. The loop
keeps a compatibility delegate for the attempt generator that returns the
coordinator's iterator directly (no per-chunk forwarding layer), a thin
delegate for the nonstreaming facade, and the streaming facade returning
the delegate's iterator directly (no added generator forwarding layers).

Live loop values (config, committed model, stats, session identity, format
handler, accounting owner/sink/attribution, injected backend, max tokens)
are read through getters, so session reset, rebind and reconfiguration stay
visible. The config orchestrator, the backend lifetime and the LLM gateway
are borrowed: they are assigned once at loop construction and never
replaced, so backend borrowing, publication, drain and the gateway handoff
behave exactly as before. The patched module-level ``create_backend``
surface is resolved at invocation time through a facade callback, so test
patches of the _loop global keep intercepting. Launch-metadata installation,
backend-metadata assembly and replay admission are late-bound callbacks, so
instance patches keep intercepting, and the committed-model restoration
collapses the three-statement identity restore (committed model, config
attach, metadata install) into one late-bound helper call. The attempt
generator resolves every rider-referenced helper (candidates, inputs,
resources, model, budget, backend, rollback) and every loop-owned seam
(transcript append, provider/switch recording, identity publication)
through late-bound callbacks, so instance patches keep intercepting exactly
as when the generator lived on the loop. Usage accounting settlement stays
in the gateway's CallFinalizer: the coordinator only binds the resources.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
import contextlib
from dataclasses import replace

from chartreux.core.agent_loop._loop_types import RootAccountingOwner
from chartreux.core.agent_loop.backend_lifetime import BackendLifetime, BackendPublish
from chartreux.core.agent_loop.errors import AgentLoopLLMResponseError
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
    TranscriptAppend,
)
from chartreux.core.agent_loop.resume_progress import ResumeProgressWatch
from chartreux.core.compaction import reorder_for_tool_adjacency
from chartreux.core.compaction.context import select_model_context
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig, ProviderConfig
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm.failures import RequestRetryBudget, classify
from chartreux.core.llm.format import APIToolFormatHandler
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm_models import AvailableTool, LLMChunk, LLMMessage, StrToolChoice
from chartreux.core.model_catalog.availability import (
    EligibleDeployment,
    eligible_deployments,
)
from chartreux.core.recovery_models import (
    CallEvidence,
    attach_attempted_deployment,
    attach_call_evidence,
    canonical_endpoint_identity,
)
from chartreux.core.recovery_override import CooldownOverrideLedger, OverridePermit
from chartreux.core.session_types import AgentStats, CommittedModelIdentity
from chartreux.core.usage import AccountingSink, UsageAttribution, UsagePurpose
from chartreux.core.utils import RetryObserver
from chartreux.observability.logging import logger
from chartreux.utils.http import get_user_agent


class CompletionCoordinator:
    """Own completion input/resource assembly, the attempt helpers, and the
    attempt generator.

    ``build_completion_inputs`` assembles the frozen CompletionInputs, with
    the header assembly in ``_extra_headers``; ``build_call_resources``
    binds the live stats, accounting sink and usage attribution to a backend.
    ``failover_candidates`` enumerates eligible deployments;
    ``backend_for_attempt`` materializes the attempt backend through the
    patched ``create_backend`` facade; ``attempt_model`` resolves the
    candidate's model; ``attempt_retry_budget`` picks the per-attempt
    deadline. ``rollback_attempt``/``finalize_attempt`` are the synchronous
    publication restoration/publication helpers; ``release_unattempted_probes``
    releases the recovery probes of candidates that never ran.
    ``attempt`` is the whole attempt generator, cleanup paths included: the
    semantic-delta cutoff, the cancellation/GeneratorExit rollback, the
    probe release in the outer ``finally``, the backend borrowing, and the
    ``aclosing`` boundaries around the gateway streams. ``complete`` is the
    nonstreaming facade: it drains the attempt stream (resolved through the
    loop, so the loop's patched ``_attempt_completion`` keeps intercepting)
    and returns the single chunk, raising ``AgentLoopLLMResponseError`` when
    the generator produced none.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        config_getter: Callable[[], ChartreuxConfigSchema],
        config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        backend_lifetime: BackendLifetime,
        llm_gateway: LLMGateway,
        committed_model_getter: Callable[[], CommittedModelIdentity | None],
        stats_getter: Callable[[], AgentStats],
        format_handler_getter: Callable[[], APIToolFormatHandler],
        session_id_getter: Callable[[], str],
        parent_session_id_getter: Callable[[], str | None],
        is_subagent_getter: Callable[[], bool],
        launch_identity_getter: Callable[[], str | None],
        launch_purpose_getter: Callable[[], str | None],
        run_identity_getter: Callable[[], str | None],
        accounting_owner_getter: Callable[[], RootAccountingOwner | None],
        accounting_sink_getter: Callable[[], AccountingSink | None],
        usage_attribution_getter: Callable[[], UsageAttribution | None],
        injected_backend_getter: Callable[[], BackendLike | None],
        max_tokens_getter: Callable[[], int | None],
        notice_retry: RetryObserver,
        admit_replay_getter: Callable[[], Callable[[], bool]],
        create_backend: Callable[..., BackendLike],
        restore_committed_model: Callable[[CommittedModelIdentity | None], None],
        install_launch_metadata: Callable[[], None],
        backend_metadata: Callable[[str | None], dict[str, str]],
        enumerate_candidates: Callable[..., tuple[EligibleDeployment, ...]],
        prepare_inputs: Callable[..., CompletionInputs],
        bind_resources: Callable[
            [BackendLike, RequestRetryBudget | None], CallResources
        ],
        materialize_model: Callable[..., ModelConfig],
        select_budget: Callable[..., RequestRetryBudget],
        backend_for_candidate: Callable[[ModelConfig, RequestRetryBudget], BackendLike],
        rollback_publication: Callable[
            [object, CommittedModelIdentity | None, EligibleDeployment], None
        ],
        append_transcript: Callable[[TranscriptAppend], None],
        record_providers: Callable[[Sequence[str]], None],
        record_switch: Callable[..., None],
        commit_identity: Callable[[CommittedModelIdentity], None],
        attempt_stream: Callable[..., AsyncGenerator[LLMChunk]],
    ) -> None:
        # Borrowed shared ownership: the config orchestrator, the backend
        # lifetime and the LLM gateway are assigned once at loop construction
        # and never replaced.
        self._config_orchestrator = config_orchestrator
        self._backend_lifetime = backend_lifetime
        self._llm_gateway = llm_gateway
        # Live getters: config, committed model, stats, session identity,
        # format handler, accounting binding, injected backend and max tokens
        # are replaced by session reset/rebind or reconfiguration, so they
        # are read at operation time.
        self._config_getter = config_getter
        self._committed_model_getter = committed_model_getter
        self._stats_getter = stats_getter
        self._format_handler_getter = format_handler_getter
        self._session_id_getter = session_id_getter
        self._parent_session_id_getter = parent_session_id_getter
        self._is_subagent_getter = is_subagent_getter
        self._launch_identity_getter = launch_identity_getter
        self._launch_purpose_getter = launch_purpose_getter
        self._run_identity_getter = run_identity_getter
        self._accounting_owner_getter = accounting_owner_getter
        self._accounting_sink_getter = accounting_sink_getter
        self._usage_attribution_getter = usage_attribution_getter
        self._injected_backend_getter = injected_backend_getter
        self._max_tokens_getter = max_tokens_getter
        # Stable retry notifier: a loop method, never reassigned.
        self._notice_retry = notice_retry
        # Replay admission resolves the loop's bound method at bind time, so
        # instance patches of the loop's admission method keep intercepting.
        self._admit_replay_getter = admit_replay_getter
        # Facade callback: the patched module-level create_backend surface
        # resolves through the _loop global at invocation time.
        self._create_backend = create_backend
        # Late-bound identity restoration: the loop helper performs the
        # three-statement restore (committed model, config attach, metadata
        # install), so instance patches of the metadata install keep
        # intercepting.
        self._restore_committed_model = restore_committed_model
        # Late-bound metadata installation and backend metadata assembly:
        # instance patches of the loop's methods keep intercepting.
        self._install_launch_metadata = install_launch_metadata
        self._backend_metadata = backend_metadata
        # Late-bound attempt-generator seams: every rider-referenced helper
        # and every loop-owned operation resolves through the loop at call
        # time, so instance patches keep intercepting exactly as when the
        # generator lived on the loop (test_failover_coordinator.py patches
        # _failover_candidates :1023, _completion_inputs :526/:909/:1177,
        # _backend_for_attempt :926/:1113, and _append_transcript :642;
        # test_usage_accounting.py wraps _call_resources :189 and
        # _attempt_model :560/:664).
        self._enumerate_candidates = enumerate_candidates
        self._prepare_inputs = prepare_inputs
        self._bind_resources = bind_resources
        self._materialize_model = materialize_model
        self._select_budget = select_budget
        self._backend_for_candidate = backend_for_candidate
        self._rollback_publication = rollback_publication
        self._append_transcript = append_transcript
        self._record_providers = record_providers
        self._record_switch = record_switch
        # Late-bound identity publication: the loop owns the committed-model
        # attribute and the config attach (the mirror of the restoration
        # helper above).
        self._commit_identity = commit_identity
        # Late-bound attempt stream: the nonstreaming facade iterates the
        # loop-resolved _attempt_completion delegate, so the
        # test_failover_coordinator.py:1057 patch keeps intercepting.
        self._attempt_stream = attempt_stream
        # C3 forced-override binding channel: the (permit, ledger) pair for
        # the CURRENT logical call. attempt()/complete() set it from their
        # override kwargs for the duration of one attempt; failover_candidates
        # reads it so the candidate enumeration itself sees the permit. One
        # attempt runs at a time per loop, so the slot never interleaves.
        self._active_override: tuple[OverridePermit, CooldownOverrideLedger] | None = (
            None
        )
        # C5b: the resumed run's first-semantic-progress deadline watch. Set
        # only by the app-server's resume admission (the explicitly marked
        # resumed context); every other admission clears it, so ordinary
        # runs bind None into CallResources and are never watched.
        self._resume_progress_watch: ResumeProgressWatch | None = None

    def _extra_headers(self, provider: ProviderConfig | None = None) -> dict[str, str]:
        provider = (
            self._config_getter().get_active_provider()
            if provider is None
            else provider
        )
        headers: dict[str, str] = {}
        headers["user-agent"] = get_user_agent(provider.backend)
        headers["x-affinity"] = self._session_id_getter()
        return headers

    def build_completion_inputs(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None = None,
        account_conversation: bool = True,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ) -> CompletionInputs:
        provider = self._config_getter().get_provider_for_model(model)
        return CompletionInputs(
            model=model,
            provider_name=provider.name,
            emits_finish_reason=provider.emits_finish_reason,
            messages=tuple(messages),
            tools=tools,
            tool_choice=tool_choice,
            extra_headers=self._extra_headers(provider),
            metadata=self._backend_metadata(call_type),
            max_tokens=self._max_tokens_getter(),
            account_conversation=account_conversation,
            purpose=purpose,
        )

    def build_call_resources(
        self, backend: BackendLike, budget: RequestRetryBudget | None = None
    ) -> CallResources:
        if budget is None:
            budget = RequestRetryBudget(
                self._config_getter().api_retry_max_elapsed_time
            )
        owner = self._accounting_owner_getter()
        sink = self._accounting_sink_getter()
        attribution = self._usage_attribution_getter()
        if owner is not None:
            sink = owner.writer
            model = self._config_getter().get_active_model()
            attribution = UsageAttribution(
                root_session_id=owner.root_session_id,
                session_id=self._session_id_getter(),
                parent_session_id=self._parent_session_id_getter(),
                agent_role="subagent" if self._is_subagent_getter() else "root",
                agent_profile=self._launch_identity_getter(),
                content_class=self._launch_purpose_getter(),
                run_id=self._run_identity_getter(),
                project_key=owner.project_key,
                model=model.alias,
                provider=model.provider,
                wire_name=model.name,
            )
        # The gateway freezes the actual attempt's deployment, purpose and prices;
        # this binding freezes session identity even if a later rebind replaces it.
        return CallResources(
            backend=backend,
            stats=self._stats_getter(),
            process_message=self._format_handler_getter().process_api_response_message,
            accounting_sink=sink,
            usage_attribution=attribution,
            retry_budget=budget,
            on_retry=self._notice_retry,
            admit_replay=self._admit_replay_getter(),
            resume_progress=self._resume_progress_watch,
        )

    def failover_candidates(
        self,
        messages: Sequence[LLMMessage],
        attempted_model: ModelConfig,
        *,
        call_type: str | None = None,
        override_permit: OverridePermit | None = None,
        override_ledger: CooldownOverrideLedger | None = None,
    ) -> tuple[EligibleDeployment, ...]:
        config = self._config_getter()
        snapshot = config.catalog_snapshot
        committed = self._committed_model_getter()
        if snapshot is None or committed is None:
            return ()
        # C3: an attempt that carries the forced-override binding enumerates
        # its candidates THROUGH the permit, so the cooled-down forced
        # deployment is eligible; the explicit kwargs win over the binding.
        if override_permit is None:
            bound = self._active_override
            if bound is not None:
                override_permit, override_ledger = bound
        result = eligible_deployments(
            snapshot=snapshot,
            committed=committed,
            registry=self._config_orchestrator.availability_registry,
            config=config,
            history=reorder_for_tool_adjacency(select_model_context(messages)),
            thinking=attempted_model.thinking,
            compaction_base=config.compaction_model or None,
            compaction_only=call_type == "secondary_call",
            allowed_models=config.allowed_models,
            override_permit=override_permit,
            override_ledger=override_ledger,
        )
        return tuple(result.candidates)

    def backend_for_attempt(
        self, model: ModelConfig, budget: RequestRetryBudget
    ) -> BackendLike:
        injected = self._injected_backend_getter()
        if injected is not None:
            return injected
        provider = self._config_getter().get_provider_for_model(model)
        return self._create_backend(
            provider=provider,
            on_retry=self._notice_retry,
            timeout=self._config_getter().api_timeout,
            retry_max_elapsed_time=self._config_getter().api_retry_max_elapsed_time,
            retry_budget=budget,
            connect_timeout=self._config_getter().api_connect_timeout,
            write_timeout=self._config_getter().api_write_timeout,
            pool_timeout=self._config_getter().api_pool_timeout,
            enable_system_trust_store=self._config_getter().enable_system_trust_store,
        )

    def attempt_model(
        self, resolved: object, *, call_type: str | None, thinking: str | None = None
    ) -> ModelConfig:
        """Materialize the completion model for a candidate provider.

        A dedicated compaction base follows the selected conversation provider;
        eligibility has already established that such a deployment exists.
        """
        from chartreux.core.model_catalog.resolver import ResolvedModel, resolver_for

        assert isinstance(resolved, ResolvedModel)
        config = self._config_getter()
        target = resolved
        if call_type == "secondary_call" and config.compaction_model:
            base = resolver_for(config).canonicalize(config.compaction_model)
            definition = config.catalog_snapshot.catalog.models[base]  # type: ignore[union-attr]
            deployment = next(
                item
                for item in definition.deployments
                if item.provider == resolved.deployment.provider and not item.disabled
            )
            target = ResolvedModel(
                base,
                definition,
                deployment,
                config.catalog_snapshot.catalog.providers[deployment.provider],  # type: ignore[union-attr]
                config.catalog_snapshot.revision,  # type: ignore[union-attr]
            )
        return target.materialize(
            auto_compact_threshold=config.auto_compact_threshold,
            thinking=(
                thinking
                if thinking is not None
                else config.thinking_overrides.get(target.base_model)
            ),
        )

    def attempt_retry_budget(
        self, budget: RequestRetryBudget, *, has_alternative: bool, forced: bool = False
    ) -> RequestRetryBudget:
        """The per-attempt retry budget.

        A FORCED attempt (the override permit's endpoint) never retries:
        the zero budget keeps the backend's retry decorators from firing
        under a permit — the episode bound counts transport starts, so
        unfettered backend retries would multiply endpoint pressure past
        it (backlog C3: no retries for the forced attempt). The branch is
        distinct from the lone-deployment rule: a lone deployment keeps
        the shared budget, a forced one never does, even when it is also
        the lone candidate. Otherwise an alternative fails the attempt
        over immediately.
        """
        if forced:
            return RequestRetryBudget(0.0)
        if not has_alternative:
            return budget
        return RequestRetryBudget(0.0)

    def _failure_endpoint(self, model: ModelConfig) -> str | None:
        """Raw endpoint URL of the provider serving ``model``, if resolvable.

        Total by contract: endpoint evidence is additive — a provider that
        cannot be resolved reports ``None`` (unavailable), never a guess.
        The availability registry canonicalizes the URL before keying.
        """
        try:
            provider = self._config_getter().get_provider_for_model(model)
        except Exception:
            return None
        api_base = getattr(provider, "api_base", None)
        if isinstance(api_base, str) and api_base:
            return api_base
        return None

    def _release_unconsumed_override(
        self, permit: OverridePermit | None, ledger: CooldownOverrideLedger | None
    ) -> None:
        """Safety net: release a forced permit that never reached a transport.

        The gateway consumes the permit at the actual transport start and
        releases it when the attempt ends pretransport; this net covers the
        exits before the gateway ever saw the binding (enumeration or
        preparation failure, cancellation, an attempt that never reached the
        forced candidate). The release runs on the ledger's TOKEN check,
        not on ``permit_outstanding``: a permit stranded by an episode close
        is no longer outstanding by the episode check but is still the
        endpoint's live reservation, and only the token check frees it. A
        started (consumed), released, or foreign permit fails the token
        check, so this never frees another reservation and never refunds a
        consumed start.
        """
        if permit is None or ledger is None:
            return
        if ledger.reservation_token_matches(permit):
            ledger.release(permit)

    def release_unattempted_probes(
        self, candidates: Sequence[EligibleDeployment], attempted: set[str]
    ) -> None:
        registry = self._config_orchestrator.availability_registry
        for candidate in candidates:
            if (
                candidate.recovery_probe
                and candidate.resolved.deployment.provider not in attempted
            ):
                assert candidate.recovery_probe_token is not None
                registry.release_probe(
                    candidate.resolved.base_model,
                    candidate.resolved.deployment.provider,
                    candidate.recovery_probe_token,
                )

    def rollback_attempt(
        self,
        publication: object,
        previous_identity: CommittedModelIdentity | None,
        candidate: EligibleDeployment,
    ) -> None:
        restored = publication.rollback()  # type: ignore[attr-defined]
        try:
            if restored:
                self._restore_committed_model(previous_identity)
        finally:
            if candidate.recovery_probe:
                assert candidate.recovery_probe_token is not None
                self._config_orchestrator.availability_registry.release_probe(
                    candidate.resolved.base_model,
                    candidate.resolved.deployment.provider,
                    candidate.recovery_probe_token,
                )

    def finalize_attempt(
        self, publication: object, candidate: EligibleDeployment
    ) -> None:
        self._install_launch_metadata()
        # Review R13: the settlement carries the candidate's probe token, so
        # a successful recovery probe settles (and releases) its own claim —
        # and a late normal attempt can never delete a newer probe's claim.
        self._config_orchestrator.availability_registry.record_success(
            candidate.resolved.base_model,
            candidate.resolved.deployment.provider,
            probe_token=candidate.recovery_probe_token,
        )
        # C1 production wiring: a measured endpoint recovery RETIRES the
        # root-shared override ledger's episode for that endpoint (the
        # pinned boundary — only a new measured failure re-opens it).
        self._config_orchestrator.cooldown_override_ledger.note_endpoint_recovery(
            candidate.resolved.provider.api_base
        )
        publication.finalize(whole_turn_active=True)  # type: ignore[attr-defined]

    async def attempt(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
        self,
        *,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        transcript: bool,
        streaming: bool,
        fallback_model: ModelConfig,
        account_conversation: bool = True,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
        override_permit: OverridePermit | None = None,
        override_ledger: CooldownOverrideLedger | None = None,
    ) -> AsyncGenerator[LLMChunk]:
        """Run one shared preparation/publication/rollback attempt lifecycle.

        ``override_permit``/``override_ledger`` (C3): the forced-override
        binding for an explicit un-stick attempt. The candidate enumeration
        runs THROUGH the permit (the cooled-down forced deployment becomes
        eligible); the candidate serving the permit's canonical endpoint is
        the FORCED one — it carries the binding into the gateway, keeps the
        full attempt deadline (no failover acceleration), and is terminal:
        its failure is raised, never replayed on another provider. The permit
        itself is consumed at the ACTUAL transport start and released
        pretransport by the gateway; the coordinator only nets out bindings
        that never reached a transport.
        """
        budget = RequestRetryBudget(self._config_getter().api_retry_max_elapsed_time)
        if override_permit is None:
            inherited = self._active_override
            if inherited is not None:
                override_permit, override_ledger = inherited
        forced = (
            override_permit
            if override_permit is not None and override_ledger is not None
            else None
        )
        if forced is not None:
            assert override_ledger is not None
            self._active_override = (forced, override_ledger)
        try:
            try:
                candidates = self._enumerate_candidates(
                    messages, fallback_model, call_type=call_type
                )
            finally:
                self._active_override = None
        except BaseException as error:
            # Admission-phase failure (WP1): no backend was created or
            # borrowed for this logical call, so zero transport starts is a
            # MEASURED fact — launch_failure requires admission/init evidence.
            attach_call_evidence(error, CallEvidence.admission_phase())
            self._release_unconsumed_override(forced, override_ledger)
            raise
        if not candidates:  # noqa: PLR1702
            inputs = self._prepare_inputs(
                model=fallback_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                call_type=call_type,
                account_conversation=account_conversation,
                purpose=purpose,
            )
            provider_name = (
                self._config_getter().get_provider_for_model(fallback_model).name
            )
            try:
                with self._backend_lifetime.borrow() as backend:
                    resources = self._bind_resources(backend, budget)
                    if (
                        forced is not None
                        and canonical_endpoint_identity(
                            self._failure_endpoint(fallback_model)
                        )
                        == forced.endpoint
                    ):
                        # C3: the fallback call targets the permit's endpoint,
                        # so it IS the forced attempt's transport — the
                        # gateway consumes the permit at its actual start.
                        assert override_ledger is not None
                        resources = replace(
                            resources,
                            override_permit=forced,
                            override_ledger=override_ledger,
                        )
                    if streaming:
                        stream = self._llm_gateway.chat_streaming(
                            inputs, resources, transcript=self._append_transcript
                        )
                        async with contextlib.aclosing(stream):
                            async for chunk in stream:
                                yield chunk
                    elif transcript:
                        yield await self._llm_gateway.chat(
                            inputs, resources, transcript=self._append_transcript
                        )
                    else:
                        yield await self._llm_gateway.complete(inputs, resources)
            finally:
                self._record_providers((provider_name,))
                self._release_unconsumed_override(forced, override_ledger)
            return

        registry = self._config_orchestrator.availability_registry
        last_error: BaseException | None = None
        attempted: set[str] = set()
        providers_used: list[str] = []
        try:  # noqa: PLR1702
            for candidate_index, candidate in enumerate(candidates):
                resolved = candidate.resolved
                provider_name = resolved.deployment.provider
                if provider_name in attempted:
                    continue

                replacement: BackendLike | None = None
                try:
                    model = self._materialize_model(
                        resolved, call_type=call_type, thinking=fallback_model.thinking
                    )
                    # C3: the candidate serving the permit's canonical
                    # endpoint is the FORCED one — it gets the full attempt
                    # deadline (no failover acceleration) and is terminal.
                    is_forced = (
                        forced is not None
                        and canonical_endpoint_identity(self._failure_endpoint(model))
                        == forced.endpoint
                    )
                    if is_forced:
                        # C3 (retry-budget defect): the forced attempt's
                        # retry budget is zeroed through the forced-specific
                        # branch — the backend's retry decorators cannot
                        # fire under a permit, distinct from the
                        # lone-deployment rule that keeps the shared budget.
                        attempt_budget = self.attempt_retry_budget(
                            budget, has_alternative=False, forced=True
                        )
                    else:
                        attempt_budget = self._select_budget(
                            budget,
                            has_alternative=candidate_index < len(candidates) - 1,
                        )
                    replacement = self._backend_for_candidate(model, attempt_budget)
                    inputs = self._prepare_inputs(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        call_type=call_type,
                        account_conversation=account_conversation,
                        purpose=purpose,
                    )
                except BaseException:
                    if replacement is not None:
                        with contextlib.suppress(Exception):
                            await replacement.__aexit__(None, None, None)
                    raise

                if replacement is None:
                    raise AgentLoopLLMResponseError(
                        "Backend preparation produced no backend for publication"
                    )

                previous_identity = self._committed_model_getter()
                failover_identity = resolved.identity.model_copy(
                    update={
                        "thinking": self._config_getter().get_active_model().thinking
                    }
                )

                def transcript_sink(
                    outcome: TranscriptAppend,
                    identity: CommittedModelIdentity = failover_identity,
                ) -> None:
                    self._append_transcript(replace(outcome, committed_model=identity))

                publication: BackendPublish | None = None
                semantic_delta = False
                try:
                    publication = self._backend_lifetime.publish_reversible(replacement)
                    attempted.add(provider_name)
                    providers_used.append(provider_name)
                    self._commit_identity(failover_identity)
                    with self._backend_lifetime.borrow() as backend:
                        resources = self._bind_resources(backend, budget)
                        if is_forced:
                            # C3: only the forced candidate carries the
                            # binding — the gateway consumes the permit at
                            # this candidate's actual transport start.
                            assert override_ledger is not None
                            resources = replace(
                                resources,
                                override_permit=forced,
                                override_ledger=override_ledger,
                            )
                        if streaming:
                            stream = self._llm_gateway.chat_streaming(
                                inputs, resources, transcript=transcript_sink
                            )
                            async with contextlib.aclosing(stream):
                                async for chunk in stream:
                                    message = chunk.message
                                    semantic_delta = semantic_delta or bool(
                                        message.content
                                        or message.reasoning_content
                                        or message.tool_calls
                                    )
                                    yield chunk
                        elif transcript:
                            yield await self._llm_gateway.chat(
                                inputs, resources, transcript=transcript_sink
                            )
                        else:
                            yield await self._llm_gateway.complete(inputs, resources)
                    self.finalize_attempt(publication, candidate)
                except (asyncio.CancelledError, GeneratorExit):
                    if publication is not None:
                        self._rollback_publication(
                            publication, previous_identity, candidate
                        )
                    raise
                except Exception as error:
                    # Review R11: freeze the attempted deployment BEFORE the
                    # rollback restores the committed identity, so a
                    # failed-attempt envelope reports the deployment that
                    # actually served the failed attempt (the error's endpoint
                    # comes from this same attempt) instead of the restored
                    # one.
                    attach_attempted_deployment(
                        error,
                        (
                            failover_identity.base_model,
                            failover_identity.provider,
                            failover_identity.wire_name,
                        ),
                    )
                    if publication is not None:
                        self._rollback_publication(
                            publication, previous_identity, candidate
                        )
                    last_error = error
                    failure = classify(error)
                    if failure.failover_eligible:
                        # WP1 endpoint correlation: the raw endpoint URL (the
                        # registry canonicalizes it) plus a structured
                        # signature, so cross-model rate-limit evidence on one
                        # endpoint is representable WITHOUT log parsing.
                        endpoint = self._failure_endpoint(model)
                        registry.record_failure(
                            resolved.base_model,
                            provider_name,
                            failure,
                            endpoint=endpoint,
                            signature=(
                                f"{failure.category.value}:{failure.status}:"
                                f"{failure.error_code or ''}:"
                                f"{failure.error_type or ''}"
                            ),
                            # Review R13: token-owned probe settlement — the
                            # failed attempt settles its own probe claim, and
                            # a late normal attempt never replaces a newer
                            # in-flight probe's claim.
                            probe_token=candidate.recovery_probe_token,
                        )
                        # C1 production wiring: a measured endpoint failure
                        # OPENS the root-shared override ledger's episode for
                        # that endpoint (the pinned boundary — a failure
                        # inside an active episode changes nothing, and only
                        # a measured recovery retires it).
                        self._config_orchestrator.cooldown_override_ledger.note_endpoint_failure(
                            endpoint
                        )
                    self._backend_lifetime.drain(whole_turn_active=False)
                    if is_forced:
                        # C3: the forced attempt is terminal — no failover
                        # replay past the permit's transport, whatever the
                        # failure class. The reservation's start was consumed
                        # at the actual transport start (or released
                        # pretransport by the gateway); the measured failure
                        # was recorded above.
                        raise
                    if (
                        semantic_delta
                        or not failure.failover_eligible
                        or budget.exhausted
                    ):
                        raise
                    continue
                else:
                    if (
                        previous_identity is not None
                        and previous_identity.provider != provider_name
                    ):
                        reason = (
                            classify(last_error).category.value
                            if last_error
                            else "availability"
                        )
                        self._record_switch(
                            base_model=resolved.base_model,
                            old_provider=previous_identity.provider,
                            new_provider=provider_name,
                            reason=reason,
                        )
                        logger.warning(
                            "Model deployment switched base=%s old_provider=%s new_provider=%s reason=%s",
                            resolved.base_model,
                            previous_identity.provider,
                            provider_name,
                            reason,
                        )
                    return
            assert last_error is not None
            raise last_error
        finally:
            self._record_providers(providers_used)
            self.release_unattempted_probes(candidates, attempted)
            self._release_unconsumed_override(forced, override_ledger)

    async def complete(
        self,
        *,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        transcript: bool,
        fallback_model: ModelConfig,
        account_conversation: bool = True,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
        override_permit: OverridePermit | None = None,
        override_ledger: CooldownOverrideLedger | None = None,
    ) -> LLMChunk:
        # Drain the attempt generator and return its single nonstreaming
        # chunk (raises AgentLoopLLMResponseError when it produced none).
        result: LLMChunk | None = None
        if override_permit is not None and override_ledger is not None:
            # C3: the attempt stream resolves through the loop delegate,
            # whose signature carries no override kwargs — the binding rides
            # the coordinator slot into attempt().
            self._active_override = (override_permit, override_ledger)
        try:
            async for chunk in self._attempt_stream(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                call_type=call_type,
                transcript=transcript,
                streaming=False,
                fallback_model=fallback_model,
                account_conversation=account_conversation,
                purpose=purpose,
            ):
                result = chunk
        finally:
            self._active_override = None
        if result is None:
            raise AgentLoopLLMResponseError("Completion returned no result")
        return result
