from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.schema import ConfigSchema

CATALOG_DEFINITION_FIELDS = frozenset({"models", "providers"})
DISPATCH_AUTHORITY_FIELDS = frozenset({"dispatch"})
DISPATCH_AUTHORITY_MESSAGE = (
    "Dispatch policy tables are only supported in the user models.toml catalog "
    "overlay; remove [dispatch] from this source."
)


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
    # Materialize once: callers may pass single-use iterables of field names.
    present = set(fields)
    forbidden = CATALOG_DEFINITION_FIELDS.intersection(present)
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
    if dispatch_fields := DISPATCH_AUTHORITY_FIELDS.intersection(present):
        raise ValidationError.from_exception_data(
            schema.__name__,
            [
                {
                    "type": PydanticCustomError(
                        "dispatch_authority", DISPATCH_AUTHORITY_MESSAGE
                    ),
                    "loc": (source, field),
                    "input": None,
                }
                for field in sorted(dispatch_fields)
            ],
            hide_input=True,
        ) from None
