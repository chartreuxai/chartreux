from __future__ import annotations

import subprocess
import sys

from chartreux.core.model_catalog.contracts import (
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
)


def test_discovery_contract_types_construct_cleanly() -> None:
    result = DiscoveryResult((DiscoveryItem("wire", "Wire"),), ("one page",))
    error = DiscoveryError("auth_rejected", "Credentials were rejected")

    assert result.models[0].display_label == "Wire"
    assert error.code == "auth_rejected"


def test_catalog_services_do_not_load_provider_ui() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import chartreux.core.model_catalog.discovery; "
            "import chartreux.core.model_catalog.loader; "
            "import chartreux.core.model_catalog.presets; "
            "assert not any(name.startswith('chartreux.ui.providers') "
            "for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
