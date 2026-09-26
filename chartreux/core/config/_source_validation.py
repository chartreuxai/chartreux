from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
from functools import lru_cache
from typing import Annotated, Any, NoReturn, cast

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    create_model,
)
from pydantic_core import InitErrorDetails, PydanticCustomError
from pydantic_settings.sources import EnvSettingsSource

from chartreux.core.config._root_authority import ROOTS_FIELD
from chartreux.core.config.chartreux_schema import _expand_paths, _non_empty
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.config.schema import ConfigSchema, MergeFieldMetadata
from chartreux.core.utils.merge import MergeStrategy

_ENV_PREFIX = "CHARTREUX_"
# These names are consumed outside the configuration schema.  Keep this list
# exact: unknown CHARTREUX_* settings are configuration mistakes, while these
# established controls must continue to be available to their consumers.
_ENVIRONMENT_CONTROLS = frozenset({
    "HOME",
    "TYPING_GRACE_PERIOD_MS",
    "ACP_LOGGING_ENABLED",
    "TEST_DISABLE_KEYRING",
    "TEST_DISABLE_AUTO_TITLE",
})


@lru_cache(maxsize=256)
def _field_adapter(schema: type[ConfigSchema], key: str) -> TypeAdapter[Any]:
    field = copy.copy(schema.model_fields[key])
    # These validators belong to the effective snapshot, not a sparse source:
    # resolving shadowed paths performs filesystem I/O, and an empty model map
    # is a valid deep-merge patch even though the final catalog must be non-empty.
    field.metadata = [
        item
        for item in field.metadata
        if not (
            isinstance(item, BeforeValidator)
            and item.func is _expand_paths
            or isinstance(item, AfterValidator)
            and item.func is _non_empty
        )
    ]
    if field.annotation is SessionLoggingConfig:
        # Reuse declared field types/defaults without the settings constructor or
        # save_dir decorators, which resolve paths even for an empty patch.
        fields: dict[str, Any] = {
            name: (item.annotation, copy.copy(item))
            for name, item in SessionLoggingConfig.model_fields.items()
        }
        shape = create_model(
            "SessionLoggingSource", __config__=ConfigDict(extra="forbid"), **fields
        )
        return TypeAdapter(Annotated[shape, BeforeValidator(_logging_source_data)])
    return TypeAdapter(field.rebuild_annotation())


def _logging_source_data(value: Any) -> Any:
    if isinstance(value, SessionLoggingConfig):
        return value.model_dump()
    return value


def _source_error(
    source: str, field: str, *, kind: str = "source_validation"
) -> InitErrorDetails:
    return {
        "type": PydanticCustomError(
            "source_validation",
            "Invalid environment configuration field ({kind})",
            {"kind": "unknown" if kind == "extra_forbidden" else "invalid"},
        ),
        "loc": (source, field),
        "input": None,
    }


def _raise_sanitized_errors(
    schema: type[ConfigSchema], errors: list[InitErrorDetails]
) -> NoReturn:
    raise ValidationError.from_exception_data(
        schema.__name__, errors, hide_input=True
    ) from None


def _sanitized_validation_error(
    schema: type[ConfigSchema], exc: ValidationError, *, source: str
) -> ValidationError:
    """Copy only safe error kinds/locations; never retain Pydantic inputs or causes."""
    errors: list[InitErrorDetails] = []
    for error in exc.errors(include_input=False, include_context=False):
        kind = str(error["type"]).rsplit(".", 1)[-1]
        errors.append({
            "type": PydanticCustomError(
                "source_validation",
                "Invalid environment configuration field ({kind})",
                {"kind": kind},
            ),
            "loc": (source, *error["loc"]),
            "input": None,
        })
    if not errors:
        errors.append(_source_error(source, "<environment>"))
    return ValidationError.from_exception_data(schema.__name__, errors, hide_input=True)


def _environment_source_error(
    schema: type[ConfigSchema], source: str, fields: Iterable[str]
) -> ValidationError:
    errors = [_source_error(source, field) for field in dict.fromkeys(fields)]
    if not errors:
        errors.append(_source_error(source, "<environment>"))
    return ValidationError.from_exception_data(schema.__name__, errors, hide_input=True)


def validate_environment_names(
    schema: type[ConfigSchema],
    environ: Mapping[str, str],
    *,
    source: str,
    settings_class: type[Any] | None = None,
) -> dict[str, str]:
    """Reject unknown environment paths before settings filtering or decoding.

    Path traversal deliberately delegates nested-field semantics to
    ``pydantic-settings``.  In particular, dictionary keys remain open while
    model fields and scalar/list descendants must be known to its field walker.
    Values are still decoded only by ``EnvSettingsSource`` in the environment
    layer after this name-only check.
    """
    fields = {name.lower(): name for name in schema.model_fields}
    field_source = EnvSettingsSource(cast(Any, settings_class or schema))
    supplied: dict[str, str] = {}
    errors: list[InitErrorDetails] = []
    for environment_name in environ:
        if not environment_name.upper().startswith(_ENV_PREFIX):
            continue
        suffix = environment_name[len(_ENV_PREFIX) :]
        if suffix.upper() in _ENVIRONMENT_CONTROLS:
            continue
        path = suffix.split("__")
        field_label = suffix if len(path) == 1 else suffix.lower()
        top_level = path[0].lower()
        if top_level == ROOTS_FIELD:
            errors.append(_source_error("environment", ROOTS_FIELD))
            continue
        if top_level not in fields:
            errors.append(_source_error(source, field_label, kind="extra_forbidden"))
            continue

        current: Any = (settings_class or schema).model_fields[fields[top_level]]
        for segment in path[1:]:
            # EnvSettingsSource treats dict[str, Any] values as open-ended
            # configuration trees.  Preserve that behavior for tool
            # dictionaries rather than imposing a second value parser here.
            if current is Any:
                break
            current = field_source.next_field(current, segment, case_sensitive=False)
            if current is None:
                errors.append(
                    _source_error(source, field_label, kind="extra_forbidden")
                )
                break
        else:
            supplied[top_level] = fields[top_level]
    if errors:
        _raise_sanitized_errors(schema, errors)
    return supplied


def validate_source(
    schema: type[ConfigSchema], data: dict[str, Any], *, source: str
) -> None:
    """Validate supplied fields before shadowing, without completing sparse patches.

    Deep-merged model patches may omit required fields supplied by another layer.
    All other errors, including unknown nested fields, remain fatal. Validation
    output is deliberately discarded: defaults and normalization belong to the
    merged snapshot, not to the source or its policy ownership.
    """
    errors: list[InitErrorDetails] = []
    for key, value in data.items():
        field = schema.model_fields.get(key)
        if field is None:
            if (
                key in {"models", "providers"}
                and schema.__name__ == "ChartreuxConfigSchema"
            ):
                errors.append({
                    "type": PydanticCustomError(
                        "legacy_catalog",
                        "Catalog tables are no longer supported in config.toml. Run `chartreux models migrate`.",
                    ),
                    "loc": (source, key),
                    "input": None,
                })
            else:
                errors.append(_source_error(source, key, kind="extra_forbidden"))
            continue
        meta = MergeFieldMetadata.from_field(field)
        try:
            _field_adapter(schema, key).validate_python(
                copy.deepcopy(value), extra="forbid"
            )
        except ValidationError as exc:
            if key == ROOTS_FIELD:
                errors.append(_source_error("root-source", ROOTS_FIELD))
                continue
            for error in exc.errors(include_input=False, include_context=False):
                if (
                    error["type"] == "missing"
                    and meta is not None
                    and meta.merge_strategy == MergeStrategy.DEEP_MERGE
                ):
                    continue
                if (
                    key == "mcp_servers"
                    and isinstance(value, dict)
                    and error["type"] == "list_type"
                ):
                    # TOML [mcp_servers.<name>] yields a dict where the schema
                    # wants a list of tables; surface the actionable spelling
                    # before the merge, where real config.toml layers fail.
                    errors.append({
                        "type": PydanticCustomError(
                            "source_validation",
                            "Invalid configuration field ({kind}). "
                            "Use [[mcp_servers]] instead of [mcp_servers.<name>].",
                            {"kind": error["type"]},
                        ),
                        "loc": (source, key, *error["loc"]),
                        "input": None,
                    })
                    continue
                errors.append({
                    "type": PydanticCustomError(
                        "source_validation",
                        "Invalid configuration field ({kind})",
                        {"kind": error["type"]},
                    ),
                    "loc": (source, key, *error["loc"]),
                    "input": None,
                })
        except (ValueError, TypeError):
            errors.append(_source_error(source, key))
    if errors:
        _raise_sanitized_errors(schema, errors)
