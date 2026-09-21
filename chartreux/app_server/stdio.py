from __future__ import annotations

import argparse
import asyncio

from chartreux.app_server._runtime import create_harness_server
from chartreux.app_server.transport import (
    BinaryLineReader,
    BinaryLineWriter,
    StdioJsonRpcTransport,
)
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.paths import LITERAL_LOG_FILE, LOG_FILE
from chartreux.observability.logging import init_file_logging


async def serve_stdio(
    *, reader: BinaryLineReader | None = None, writer: BinaryLineWriter | None = None
) -> None:
    transport = (
        StdioJsonRpcTransport.from_standard_streams()
        if reader is None or writer is None
        else StdioJsonRpcTransport(reader, writer)
    )
    harness = await create_harness_server(transport, transport_kind="stdio")
    await harness.serve()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Chartreux app server")
    return parser.parse_args()


def main() -> None:
    from chartreux.core.config import load_dotenv_values

    parse_arguments()
    init_harness_files_manager("user", "project")
    init_file_logging(LOG_FILE.path, repair_path=LITERAL_LOG_FILE.path)
    load_dotenv_values()
    asyncio.run(serve_stdio())
