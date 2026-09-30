from __future__ import annotations

from chartreux.ui.providers.contracts import (
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
)


def test_discovery_contract_types_construct_cleanly() -> None:
    result = DiscoveryResult((DiscoveryItem("wire", "Wire"),), ("one page",))
    error = DiscoveryError("auth_rejected", "Credentials were rejected")

    assert result.models[0].display_label == "Wire"
    assert error.code == "auth_rejected"
