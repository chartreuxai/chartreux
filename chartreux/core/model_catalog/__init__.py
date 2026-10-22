"""Model catalog schema, defaults, and loader."""

from __future__ import annotations

from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogLoadError,
    CatalogSnapshot,
    load_catalog,
    merge_catalog_overlay,
)
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
    ModelCatalog,
    Prices,
    ProviderDefinition,
)

__all__ = [
    "SHIPPED_CATALOG",
    "BaseModelDefinition",
    "CatalogLoadError",
    "CatalogSnapshot",
    "DeploymentDefinition",
    "ModelCatalog",
    "Prices",
    "ProviderDefinition",
    "load_catalog",
    "merge_catalog_overlay",
]
