"""Identity contracts for the workbench facade surface.

The workbench split (v0.5.5 stage 18) moves classes, constants, and
controller bodies out of ``chartreux.ui.providers.workbench`` into ten
satellite modules while the facade re-exports them, so importers must keep
resolving the same objects at the same import path. These contracts are the
18.4 move guards: they pin the pre-split public ``dir()`` snapshot, require
every preserved pre-split public name to stay identical to its defining
module, pin the dead-import sweep output, require the moved names to stay
identical to their satellite homes, and require the screen class to keep its
name and home module for the screen-ID literal consumers.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import chartreux.ui.providers.workbench as workbench

# Pre-split public dir() snapshot of chartreux.ui.providers.workbench
# (development @ 498f368, 5,793 lines), frozen at stage 18.4: every public
# name the unsplit module exposed.
PRE_SPLIT_PUBLIC_NAMES: tuple[str, ...] = (
    "BaseModelDefinition",
    "Binding",
    "BindingType",
    "BrowserList",
    "CHECKBOX_SEGMENTS",
    "CURSOR_GUTTER",
    "Callable",
    "CatalogChanges",
    "CatalogSnapshot",
    "CatalogValidationError",
    "CatalogWriteResult",
    "CatalogWriter",
    "Checklist",
    "ClassVar",
    "Color",
    "ComposeResult",
    "ConfigService",
    "ConnectionDraft",
    "Container",
    "Content",
    "Coroutine",
    "CredentialService",
    "CredentialStatusResolver",
    "DeploymentDefinition",
    "DetailHelp",
    "DiscoveryError",
    "DiscoveryItem",
    "DiscoveryResult",
    "DiscoveryService",
    "DispatchMode",
    "FULLY_CUSTOM",
    "Horizontal",
    "Input",
    "Label",
    "Literal",
    "MISTRAL",
    "ManagementState",
    "Mapping",
    "ModalScreen",
    "ModelChecklist",
    "ModelEdits",
    "NavigableOptionList",
    "NavigationFrame",
    "NoMarkupStatic",
    "Option",
    "OptionList",
    "OptionalEdit",
    "PRESETS",
    "PendingModel",
    "ProviderDefinition",
    "ProviderDraft",
    "ProviderPreset",
    "ProviderWorkbenchResult",
    "ProviderWorkbenchScreen",
    "SHIPPED_CATALOG",
    "Segment",
    "Selection",
    "SelectionList",
    "Sequence",
    "StrEnum",
    "Strip",
    "Style",
    "TLSConfig",
    "Text",
    "Vertical",
    "VerticalScroll",
    "Widget",
    "WorkbenchList",
    "WorkbenchView",
    "annotations",
    "asyncio",
    "bind_snapshot_policy",
    "cast",
    "cell_len",
    "chrome_glyph",
    "credential_status",
    "dataclass",
    "deepcopy",
    "events",
    "isfinite",
    "match_discovered_model",
    "partial",
    "project_catalog_changes",
    "re",
    "replace",
    "shortcut",
    "shortcut_hint",
    "usable_canonical_models",
    "valid_provider_name",
)

# Pre-split public names that were ambient imports whose only pre-split
# users moved into the satellite modules. The 18.4 dead-import sweep drops
# them from the facade deliberately: W9 3.11 re-exports the moved surface,
# not the moved bodies private imports.
SWEEP_DROPPED_IMPORTS: frozenset[str] = frozenset({
    "CHECKBOX_SEGMENTS",
    "Checklist",
    "Color",
    "Content",
    "DispatchMode",
    "ModelEdits",
    "OptionalEdit",
    "ProviderDefinition",
    "SHIPPED_CATALOG",
    "Segment",
    "StrEnum",
    "Strip",
    "Style",
    "bind_snapshot_policy",
    "cell_len",
    "credential_status",
    "dataclass",
    "deepcopy",
    "isfinite",
    "project_catalog_changes",
    "re",
    "shortcut",
    "shortcut_hint",
    "usable_canonical_models",
    "valid_provider_name",
})

PRESERVED_PUBLIC_NAMES: tuple[str, ...] = tuple(
    name for name in PRE_SPLIT_PUBLIC_NAMES if name not in SWEEP_DROPPED_IMPORTS
)

# The moved surface (W9 3.11): facade name -> defining satellite module.
MOVED_SURFACE: tuple[tuple[str, str], ...] = (
    ("BrowserList", "chartreux.ui.providers.workbench_widgets"),
    ("CURSOR_GUTTER", "chartreux.ui.providers.workbench_types"),
    ("DetailHelp", "chartreux.ui.providers.workbench_widgets"),
    ("ModelChecklist", "chartreux.ui.providers.workbench_widgets"),
    ("NavigationFrame", "chartreux.ui.providers.workbench_types"),
    ("WorkbenchList", "chartreux.ui.providers.workbench_widgets"),
    ("WorkbenchView", "chartreux.ui.providers.workbench_types"),
)

# The private re-exports (W9 3.11): facade name -> defining satellite module.
MOVED_PRIVATE_SURFACE: tuple[tuple[str, str], ...] = (
    ("_GroupPosition", "chartreux.ui.providers.workbench_types"),
    ("_bounded_cursor", "chartreux.ui.providers.workbench_widgets"),
)

# The controller wiring (W9 3.11): facade name -> defining satellite module.
CONTROLLER_SURFACE: tuple[tuple[str, str], ...] = (
    ("WorkbenchBrowser", "chartreux.ui.providers.workbench_browser"),
    ("WorkbenchConnection", "chartreux.ui.providers.workbench_connection"),
    ("WorkbenchFlowPolicy", "chartreux.ui.providers.workbench_flow_policy"),
    ("WorkbenchModelEditor", "chartreux.ui.providers.workbench_model_editor"),
    ("WorkbenchNavigation", "chartreux.ui.providers.workbench_navigation"),
    ("WorkbenchOperations", "chartreux.ui.providers.workbench_operations"),
    ("WorkbenchPresentation", "chartreux.ui.providers.workbench_presentation"),
    ("WorkbenchPresets", "chartreux.ui.providers.workbench_presets"),
)

SATELLITE_MODULES: tuple[str, ...] = (
    "workbench_browser",
    "workbench_connection",
    "workbench_flow_policy",
    "workbench_model_editor",
    "workbench_navigation",
    "workbench_operations",
    "workbench_presentation",
    "workbench_presets",
    "workbench_types",
    "workbench_widgets",
)


@pytest.mark.parametrize("name", PRESERVED_PUBLIC_NAMES)
def test_pre_split_public_name_stays_importable_and_identical(name: str) -> None:
    """Every preserved pre-split public name stays at the same import path."""
    value = getattr(workbench, name)
    module = getattr(value, "__module__", None)
    if not isinstance(module, str) or not module.startswith("chartreux."):
        return
    owner = importlib.import_module(module)
    assert getattr(owner, name) is value, name


def test_swept_imports_stay_out_of_the_facade() -> None:
    """The dead-import sweep output is pinned: no re-added dead import."""
    readded = sorted(name for name in SWEEP_DROPPED_IMPORTS if hasattr(workbench, name))
    assert not readded, readded


@pytest.mark.parametrize(("name", "home"), MOVED_SURFACE + MOVED_PRIVATE_SURFACE)
def test_moved_name_is_identical_to_its_satellite_home(name: str, home: str) -> None:
    """workbench.X is satellite.X for every name the split moved (W9 3.11)."""
    owner = importlib.import_module(home)
    assert getattr(workbench, name) is getattr(owner, name), name


@pytest.mark.parametrize(("name", "home"), CONTROLLER_SURFACE)
def test_controller_is_identical_to_its_satellite_home(name: str, home: str) -> None:
    """The facade wiring exposes each controller as its satellite class."""
    owner = importlib.import_module(home)
    assert getattr(workbench, name) is getattr(owner, name), name


def test_screen_keeps_its_name_and_home_module() -> None:
    """ProviderWorkbenchScreen never moves (screen-ID literal consumers)."""
    screen = workbench.ProviderWorkbenchScreen
    assert screen.__name__ == "ProviderWorkbenchScreen"
    assert screen.__module__ == "chartreux.ui.providers.workbench"


def test_no_stale_module_shadows_the_package_internal_imports() -> None:
    """The split added flat satellites: no package dir may shadow them."""
    facade_path = workbench.__file__
    assert facade_path is not None and facade_path.endswith("workbench.py")
    providers_dir = Path(facade_path).resolve().parent
    assert not (providers_dir / "workbench").exists()
    for satellite in SATELLITE_MODULES:
        module = importlib.import_module(f"chartreux.ui.providers.{satellite}")
        path = module.__file__
        assert path is not None and path.endswith(f"{satellite}.py")
        assert not (providers_dir / satellite).exists()
