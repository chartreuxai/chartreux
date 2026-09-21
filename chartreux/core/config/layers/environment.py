from __future__ import annotations

import os
from typing import Any, cast

from pydantic import BaseModel, ValidationError, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource

from chartreux.core.config._catalog import validate_catalog_scope
from chartreux.core.config._root_authority import ROOTS_FIELD, validate_root_source
from chartreux.core.config._source_validation import (
    _environment_source_error,
    _sanitized_validation_error,
    validate_environment_names,
    validate_source,
)
from chartreux.core.config.fingerprint import create_dict_fingerprint
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.schema import ConfigSchema
from chartreux.core.config.types import LayerConfigSnapshot


class _EnvBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHARTREUX_",
        case_sensitive=False,
        env_nested_delimiter="__",
        env_ignore_empty=False,
        extra="ignore",
    )


class EnvironmentLayer(ConfigLayer[RawConfig]):
    """Reads CHARTREUX_* env vars via pydantic-settings, which handles type coercion
    and validation against the schema.
    """

    def __init__(self, *, name: str = "environment", schema: type[BaseModel]) -> None:
        super().__init__(name=name)

        self._schema = schema
        fields: dict[str, Any] = {
            field_name: (info.annotation, info)
            for field_name, info in schema.model_fields.items()
        }
        self._settings_class: type[BaseSettings] = create_model(
            "_EnvSchema", __base__=_EnvBase, **fields
        )

    async def _check_trust(self) -> bool:
        return True

    async def _build_config_snapshot(self) -> LayerConfigSnapshot:
        source = f"{self.name} ({self.source_locator})"
        schema = cast(type[ConfigSchema], self._schema)
        # Presence is forbidden, even an empty/malformed value or a nested name.
        # Do not decode values or include nested project paths in diagnostics.
        if any(
            name.upper().startswith("CHARTREUX_")
            and name[len("CHARTREUX_") :].split("__", 1)[0].lower() == ROOTS_FIELD
            for name in os.environ
        ):
            validate_root_source({ROOTS_FIELD: None}, layer=self)
        validate_catalog_scope(
            schema,
            (
                name[len("CHARTREUX_") :].split("__", 1)[0].lower()
                for name in os.environ
                if name.upper().startswith("CHARTREUX_")
            ),
            layer=self,
            source=source,
        )
        field_names = validate_environment_names(
            schema, os.environ, source=source, settings_class=self._settings_class
        )
        try:
            raw = EnvSettingsSource(self._settings_class)()
        except (TypeError, ValueError) as exc:
            matching_fields = {
                field for field in field_names.values() if field in str(exc)
            }
            raise _environment_source_error(
                schema, source, matching_fields or set(field_names.values())
            ) from None
        validate_source(schema, raw, source=source)
        try:
            data = self._settings_class.model_validate(raw).model_dump(
                exclude_unset=True
            )
        except ValidationError as exc:
            # Settings validation may include raw environment values in its
            # normal representation.  Publish only source/field/kind details;
            # the base layer will retain the sanitized error as its cause.
            raise _sanitized_validation_error(schema, exc, source=source) from None
        fingerprint = create_dict_fingerprint(data)
        return LayerConfigSnapshot(data=data, fingerprint=fingerprint)

    async def _save_to_store(self, _next_config: RawConfig) -> str:
        raise NotImplementedError(
            "EnvironmentLayer patch persistence is not implemented yet"
        )
