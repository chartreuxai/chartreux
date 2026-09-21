from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.schema import ConfigSchema

CATALOG_DEFINITION_FIELDS = frozenset({"models", "providers"})


def layer_allows_catalog_definitions(layer: ConfigLayer[RawConfig]) -> bool:
    """Catalog definitions live exclusively in the external models.toml loader."""
    return False


def validate_catalog_scope(
    schema: type[ConfigSchema],
    fields: Iterable[str],
    *,
    layer: ConfigLayer[RawConfig],
    source: str,
) -> None:
    """Reject even empty or malformed definitions before inspecting their values."""
    if not issubclass(
        schema, ChartreuxConfigSchema
    ) or layer_allows_catalog_definitions(layer):
        return
    forbidden = CATALOG_DEFINITION_FIELDS.intersection(fields)
    if forbidden:
        raise ValidationError.from_exception_data(
            schema.__name__,
            [
                {
                    "type": PydanticCustomError(
                        "legacy_catalog",
                        "Catalog tables are no longer supported in config.toml. Run `chartreux models migrate`.",
                    ),
                    "loc": (source, field),
                    "input": None,
                }
                for field in sorted(forbidden)
            ],
            hide_input=True,
        ) from None
