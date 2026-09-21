from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys

from chartreux import __version__
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.paths import HISTORY_FILE, LITERAL_LOG_FILE, LOG_FILE
from chartreux.observability.logging import init_file_logging, logger

# Configure line buffering for subprocess communication
sys.stdout.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]
sys.stderr.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]
sys.stdin.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]


@dataclass
class Arguments:
    setup: bool


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="Run Chartreux in ACP mode")
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--setup", action="store_true", help="Setup API key and exit")
    args = parser.parse_args()
    return Arguments(setup=args.setup)


def bootstrap_config_files() -> None:
    history_file = HISTORY_FILE.path
    if not history_file.exists():
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history_file.write_text("Hello Chartreux!\n", "utf-8")
        except Exception as e:
            logger.error("Could not create history file: %s", e)
            raise


def main() -> None:
    init_harness_files_manager("user", "project")
    init_file_logging(LOG_FILE.path, repair_path=LITERAL_LOG_FILE.path)

    from chartreux.acp.agent import run_acp_server
    from chartreux.core.config import load_dotenv_values
    from chartreux.setup.onboarding import run_onboarding

    environ_before_dotenv_load = os.environ.copy()
    load_dotenv_values()
    bootstrap_config_files()
    args = parse_arguments()
    if args.setup:
        run_onboarding()
        sys.exit(0)

    run_acp_server(environ_before_dotenv_load=environ_before_dotenv_load)


if __name__ == "__main__":
    main()
