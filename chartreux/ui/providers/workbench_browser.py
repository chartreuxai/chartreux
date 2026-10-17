"""Browser controller for the provider workbench with an injected host."""

from __future__ import annotations

from typing import Any

from textual.widgets.option_list import Option

from chartreux.core.model_catalog.contracts import DiscoveryError
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
)
from chartreux.ui.providers.management_state import credential_status
from chartreux.ui.providers.workbench_types import WorkbenchView
from chartreux.ui.providers.workbench_widgets import BrowserList, WorkbenchList
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class WorkbenchBrowser:
    """Provider browser and global catalog rendering; all state stays on the host.

    Plain object (no mixins): the screen constructs it in ``__init__`` with
    ``host=self`` and is its late-bound host — every collaboration is
    ``self._host._method(...)`` at call time, never a constructor-captured
    bound method. The browser filter, cursor, and catalog state stay on the
    screen; the controller mutates them through the host exactly where the
    screen's own code did.
    """

    def __init__(self, *, host: Any) -> None:
        self._host = host

    def refresh_browser(self) -> None:  # noqa: PLR0914
        browser = self._host.query_one("#wb-providers", BrowserList)
        position = self._host._position_before_rebuild(
            WorkbenchView.PROVIDERS, "wb-providers"
        )
        current = (
            str(browser.highlighted_option.id)
            if browser.highlighted_option
            else self._host._browser_cursor
        )
        query = self._host.filter_text.strip()
        sensitive = any(char.isupper() for char in query)
        catalog = (
            self._host.state.catalog
            if self._host.state is not None
            else self._host.snapshot.catalog
        )
        names = sorted(catalog.providers)
        configured = [
            name
            for name in names
            if name in self._host.snapshot.overlaid_providers
            or name not in SHIPPED_CATALOG.providers
            or bool(
                catalog.providers[name].api_key_env_var
                and self._host.credential_resolver(
                    catalog.providers[name].api_key_env_var
                )
            )
        ]
        available = [name for name in names if name not in configured]
        names = configured + available
        visible = [
            name
            for name in names
            if (query in name if sensitive else query.lower() in name.lower())
            or (
                query in self.friendly(name)
                if sensitive
                else query.lower() in self.friendly(name).lower()
            )
        ]
        browser.clear_options()
        browser.add_option(Option("PROVIDERS", id="\x00heading", disabled=True))
        if not visible:
            browser.add_option(
                Option(
                    "No matching providers" if query else "No configured providers",
                    id="\x00empty",
                )
            )
        for name in visible:
            count = self.runnable_provider_count(name)
            status = credential_status(
                self.provider_env(name), self._host.credential_resolver
            )
            provider = catalog.providers[name]
            if provider.disabled:
                status = "Disabled"
            elif not self.provider_has_enabled_model(name):
                status += "; No enabled models"
            if self._host.state is not None and isinstance(
                self._host.state.discoveries.get(name), DiscoveryError
            ):
                status += "; Discovery Failed"
            browser.add_option(
                Option(
                    self._host._summary(
                        f"{self.friendly(name)}"
                        f"{'  Pending' if self._host.state and name in self._host.state.staged_providers else ''}"
                        f"  {status}  {count} runnable model"
                        f"{'s' if count != 1 else ''}"
                    ),
                    id=name,
                )
            )
        root_actions = self._host.query_one("#wb-root-actions", WorkbenchList)
        action_position = self._host._position_before_rebuild(
            WorkbenchView.PROVIDERS, "wb-root-actions"
        )
        root_actions.clear_options()
        root_actions.add_option(Option("ACTIONS", id="\x00actions", disabled=True))
        actions = [("\x00add", "Add custom provider"), ("\x00models", "Model catalog")]
        actions.append(("\x00presets", "Choose default presets"))
        if self._host.state is not None and self._host.state.dirty:
            actions.extend([
                ("\x00apply", "Apply all catalog edits to disk"),
                ("\x00discard", "Discard all pending catalog edits"),
            ])
        for key, label in actions:
            root_actions.add_option(Option(label, id=key))
        root_actions.highlighted = 1
        self._host._restore_group_position(action_position)
        ids = [str(option.id) for option in browser.options]
        preferred: str | None = None
        if position is None:
            preferred = self._host._flow.preferred_root_cursor(visible)
        current_option = next(
            (option for option in browser.options if str(option.id) == current), None
        )
        current_actionable = bool(
            current_option
            and not current_option.disabled
            and current not in {"\x00heading", "\x00actions", "\x00empty"}
        )
        target = current if current_actionable else preferred
        browser.highlighted = ids.index(target) if target in ids else 1
        self._host.query_one("#wb-filter", NoMarkupStatic).update(
            f"Filter: {query or 'type to filter'}  ·  {self.provider_count_label(len(visible), len(names))}"
        )
        self._host._provider_visible_count = (len(visible), len(names))
        self._host.query_one("#wb-count", NoMarkupStatic).display = False
        self._host._restore_group_position(position)
        self._host._update_help()

    @staticmethod
    def friendly(provider_id: str) -> str:
        return provider_id

    def visible_provider_count(self) -> str:
        visible, total = self._host._provider_visible_count
        return self.provider_count_label(visible, total)

    @staticmethod
    def provider_count_label(visible: int, total: int) -> str:
        if visible == total:
            return f"{total} provider{'s' if total != 1 else ''}"
        return f"{visible} of {total} providers"

    def provider_env(self, name: str) -> str:
        provider = (
            self._host.state.catalog.providers.get(name)
            if self._host.state is not None
            else None
        ) or self._host.snapshot.catalog.providers.get(name)
        if self._host.state is not None:
            connection = self._host.state.connections.get(name)
            if connection is not None:
                return connection.api_key_env_var
        if self._host.state is not None and self._host.state.provider_id == name:
            return self._host.state.connection.api_key_env_var
        return provider.api_key_env_var if provider is not None else ""

    def credential_ready_for_provider(self, name: str) -> bool:
        env_var = self.provider_env(name)
        return (
            not env_var
            or env_var in self._host._saved_credential_envs
            or bool(self._host.credential_resolver(env_var))
        )

    def credential_value(self, env_var: str) -> str | None:
        return (
            "saved"
            if env_var in self._host._saved_credential_envs
            else self._host.credential_resolver(env_var)
        )

    def api_key_status(self, env_var: str) -> str:
        if env_var in self._host._saved_credential_envs:
            return "Saved"
        if env_var and self._host.credential_resolver(env_var):
            return "From environment"
        return "Not set"

    def runnable_provider_count(self, name: str) -> int:
        catalog = (
            self._host.state.catalog
            if self._host.state is not None
            else self._host.snapshot.catalog
        )
        provider = catalog.providers.get(name)
        if (
            provider is None
            or provider.disabled
            or not self.credential_ready_for_provider(name)
        ):
            return 0
        count = 0
        for base, model in catalog.models.items():
            if model.disabled:
                continue
            for dep in model.deployments:
                if dep.provider != name:
                    continue
                if self.deployment_enabled(base, dep):
                    count += 1
        if self._host.state is not None:
            count += sum(
                item.enabled
                for item in self._host.state.pending_by_provider.get(name, {}).values()
            )
        return count

    def mistral_needs_setup(self, name: str) -> bool:
        shipped = SHIPPED_CATALOG.providers.get(name)
        return (
            name == "mistral"
            and name not in self._host.snapshot.overlaid_providers
            and self._host.snapshot.catalog.providers.get(name) == shipped
            and not self.credential_ready_for_provider(name)
        )

    def deployment_enabled(self, base: str, deployment: DeploymentDefinition) -> bool:
        enabled = not deployment.disabled
        if self._host.state is not None:
            enabled = self._host.state.enabled_by_deployment.get(
                (base, deployment.provider), enabled
            )
        return enabled

    def provider_has_enabled_model(self, name: str) -> bool:
        catalog = (
            self._host.state.catalog
            if self._host.state is not None
            else self._host.snapshot.catalog
        )
        configured = any(
            not model.disabled
            and dep.provider == name
            and self.deployment_enabled(base, dep)
            for base, model in catalog.models.items()
            for dep in model.deployments
        )
        pending = (
            any(
                item.enabled
                for item in self._host.state.pending_by_provider.get(name, {}).values()
            )
            if self._host.state is not None
            else False
        )
        return configured or pending

    def open_catalog(
        self,
        provider: str | None = None,
        focus: str | None = None,
        *,
        push: bool = True,
    ) -> None:
        if push and self._host._view != WorkbenchView.CATALOG:
            self._host._push_view(WorkbenchView.CATALOG)
        self._host._models_view = True
        self._host._model_filter = provider
        self._host._detail_cursor = focus
        self.refresh_catalog(preserve_filter_cursor=False)
        self._host._activate_view(
            WorkbenchView.CATALOG,
            focus_id="wb-catalog",
            selected_id=f"model:{focus}" if focus else None,
        )

    def catalog_model_summary(
        self, name: str, definition: BaseModelDefinition | None
    ) -> str:
        state = self._host.state
        catalog = state.catalog if state else self._host.snapshot.catalog
        deployments = definition.deployments if definition is not None else ()
        entries: list[tuple[str, bool, bool]] = []
        for deployment in deployments:
            provider = catalog.providers.get(deployment.provider)
            enabled = bool(
                definition is not None
                and not definition.disabled
                and self.deployment_enabled(name, deployment)
            )
            runnable = bool(
                enabled
                and provider is not None
                and not provider.disabled
                and self.credential_ready_for_provider(deployment.provider)
            )
            entries.append((deployment.provider, enabled, runnable))
        if state is not None:
            entries.extend(
                (
                    provider_id,
                    True,
                    bool(
                        (provider := catalog.providers.get(provider_id)) is not None
                        and not provider.disabled
                        and self.credential_ready_for_provider(provider_id)
                    ),
                )
                for provider_id, pending_by_wire in state.pending_by_provider.items()
                for item in pending_by_wire.values()
                if item.canonical_name == name and item.enabled
            )
        providers = (
            ", ".join(sorted({provider for provider, _, _ in entries})) or "pending"
        )
        roles = sorted(
            role
            for role, entry in catalog.roles.items()
            if name == (state.preset(role)[0] if state else entry.model)
        )
        enabled_count = sum(enabled for _, enabled, _ in entries)
        runnable_count = sum(runnable for _, _, runnable in entries)
        role_summary = ", ".join(roles) or "no presets"
        return self._host._summary(
            f"{name} ({providers}) · {enabled_count} enabled, "
            f"{runnable_count} runnable · {role_summary}"
        )

    def refresh_catalog(self, *, preserve_filter_cursor: bool = True) -> None:  # noqa: PLR0914
        filters = self._host.query_one("#wb-catalog-filter", NavigableOptionList)
        filter_position = self._host._position_before_rebuild(
            WorkbenchView.CATALOG, "wb-catalog-filter"
        )
        position = self._host._position_before_rebuild(
            WorkbenchView.CATALOG, "wb-catalog"
        )
        current_filter = f"filter:{self._host._model_filter or ''}"
        state = self._host.state
        previous_filter = (
            str(filters.highlighted_option.id)
            if preserve_filter_cursor
            and filters.highlighted_option
            and filters.highlighted_option.id is not None
            else None
        )
        previous_filter_index = filters.highlighted or 1
        filters.clear_options()
        filters.add_option(
            Option("PROVIDER FILTER", id="\x00provider-filter", disabled=True)
        )
        filters.add_option(Option("All Providers", id="filter:"))
        provider_ids = set(
            state.catalog.providers if state else self._host.snapshot.catalog.providers
        )
        if state is not None:
            provider_ids.add(state.provider_id)
            provider_ids.update(state.pending_by_provider)
        for provider in sorted(provider_ids):
            filters.add_option(Option(f"{provider}", id=f"filter:{provider}"))
        filter_ids = [str(option.id) for option in filters.options]
        filter_target = previous_filter or current_filter
        filters.highlighted = (
            filter_ids.index(filter_target)
            if filter_target in filter_ids
            else min(previous_filter_index, len(filter_ids) - 1)
        )
        catalog = self._host.query_one("#wb-catalog", NavigableOptionList)
        previous = (
            str(catalog.highlighted_option.id)
            if catalog.highlighted_option and catalog.highlighted_option.id is not None
            else None
        )
        previous_index = catalog.highlighted or 0
        catalog.clear_options()
        actions = self._host.query_one("#wb-catalog-actions", WorkbenchList)
        action_position = self._host._position_before_rebuild(
            WorkbenchView.CATALOG, "wb-catalog-actions"
        )
        actions.clear_options()
        if state is not None and state.dirty:
            actions.add_option(
                Option("Apply all catalog edits to disk", id="\x00apply")
            )
            actions.add_option(
                Option("Discard all pending catalog edits", id="\x00discard")
            )
        actions.highlighted = 0 if actions.option_count else None
        self._host._restore_group_position(action_position)
        models = state.catalog.models if state else self._host.snapshot.catalog.models
        pending = (
            {
                item.canonical_name
                for items in state.pending_by_provider.values()
                for item in items.values()
                if item.enabled
            }
            if state
            else set()
        )
        shown = 0
        for name in sorted(models.keys() | pending):
            definition = models.get(name)
            deployments = definition.deployments if definition else ()
            if (
                self._host._model_filter
                and not any(
                    dep.provider == self._host._model_filter for dep in deployments
                )
                and not (
                    state
                    and any(
                        item.canonical_name == name
                        for item in state.pending_by_provider.get(
                            self._host._model_filter, {}
                        ).values()
                    )
                )
            ):
                continue
            shown += 1
            catalog.add_option(
                Option(self.catalog_model_summary(name, definition), id=f"model:{name}")
            )
        if not shown:
            catalog.add_option(Option("No models configured", id="\x00empty"))
        ids = [str(option.id) for option in catalog.options]
        target = previous or (
            f"model:{self._host._detail_cursor}" if self._host._detail_cursor else None
        )
        catalog.highlighted = (
            ids.index(target)
            if target in ids
            else (min(previous_index, len(ids) - 1) if ids else None)
        )
        self._host._restore_group_position(position)
        if preserve_filter_cursor:
            self._host._restore_group_position(filter_position)
        self._host._update_help()

    def catalog_model_owners(self, name: str) -> list[str]:
        """Return every live provider that owns a catalog or pending model."""
        catalog = (
            self._host.state.catalog
            if self._host.state
            else self._host.snapshot.catalog
        )
        definition = catalog.models.get(name)
        owners = (
            {dep.provider for dep in definition.deployments} if definition else set()
        )
        if self._host.state:
            owners.update(
                provider_id
                for provider_id, entries in self._host.state.pending_by_provider.items()
                if any(
                    item.canonical_name == name and item.enabled
                    for item in entries.values()
                )
            )
        if self._host._model_filter:
            owners = {owner for owner in owners if owner == self._host._model_filter}
        return sorted(owners)

    def select_catalog(self, key: str) -> None:
        if self._host._busy:
            return
        if key.startswith("filter:"):
            self._host._model_filter = key.removeprefix("filter:") or None
            self._host._detail_cursor = None
            self.refresh_catalog()
        elif key == "\x00provider-filter":
            return
        elif key == "\x00apply":
            self._host._request_commit()
        elif key == "\x00discard":
            self._host._confirm = "discard"
            self._host._update_help()
        elif key.startswith("model:"):
            name = key.removeprefix("model:")
            self._host._flow.remember_catalog_selection(name)
            self._host._detail_cursor = name
            providers = self.catalog_model_owners(name)
            provider = providers[0] if len(providers) == 1 else None
            if provider:
                if (
                    not self._host._navigation
                    or self._host._navigation[-1].view != WorkbenchView.CATALOG
                ):
                    self._host._push_view(WorkbenchView.CATALOG)
                self._host._expand(provider)
                self._host._open_detail(name, push=False)
            elif len(providers) > 1:
                self.open_deployment_picker(name, providers)
            else:
                self._host._message = "This model has no live provider owner to edit; open its provider draft first."
                self._host._update_help()

    def open_deployment_picker(self, name: str, providers: list[str]) -> None:
        """Choose the deployment that owns an ambiguous global model row."""
        self._host._push_view(WorkbenchView.DEPLOYMENTS)
        self._host._picker = True
        self._host._detail_cursor = name
        position = self._host._position_before_rebuild(
            WorkbenchView.DEPLOYMENTS, "wb-picker"
        )
        picker = self._host.query_one("#wb-picker", NavigableOptionList)
        picker.clear_options()
        for provider in providers:
            picker.add_option(
                Option(f"{provider}/{name}", id=f"deployment:{provider}:{name}")
            )
        picker.highlighted = 0
        self._host._restore_group_position(position)
        self._host._activate_view(WorkbenchView.DEPLOYMENTS, focus_id="wb-picker")
