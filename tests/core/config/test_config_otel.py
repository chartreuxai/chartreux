from __future__ import annotations

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema


def test_remote_telemetry_configuration_is_not_part_of_the_schema() -> None:
    fields = ChartreuxConfigSchema.model_fields
    assert "enable_telemetry" not in fields
    assert "enable_otel" not in fields
    assert "otel_endpoint" not in fields
    assert "otel_redaction" not in fields


def test_local_configuration_and_accounting_fields_remain() -> None:
    fields = ChartreuxConfigSchema.model_fields
    assert "session_logging" in fields
    assert "include_model_info" in fields
    assert "api_timeout" in fields
