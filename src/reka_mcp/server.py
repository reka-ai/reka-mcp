# ABOUTME: MCP server entry point for Reka Vision Agent.
# ABOUTME: Supports stdio and streamable HTTP transports.

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from reka_mcp.auth import StaticTokenVerifier
from reka_mcp.client import RekaClient
from reka_mcp.config import load_config
from reka_mcp.resources import register_resources
from reka_mcp.tools.groups import register_group_tools
from reka_mcp.tools.indexing import register_indexing_tools
from reka_mcp.tools.qa import register_qa_tools
from reka_mcp.tools.search import register_search_tools
from reka_mcp.tools.segment import register_segment_tools
from reka_mcp.tools.sub_resources import register_sub_resource_tools
from reka_mcp.tools.videos import register_video_tools

logger = logging.getLogger(__name__)


def create_server(
    api_url: str,
    api_key: str,
    index_timeout: int = 600,
    poll_interval: int = 5,
    auth_token: str | None = None,
) -> FastMCP:
    client = RekaClient(api_url=api_url, api_key=api_key)

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[None]:
        logger.info("server starting, api_url=%s", api_url)
        try:
            yield
        finally:
            logger.info("server shutting down")
            await client.close()

    kwargs: dict[str, Any] = {}
    if auth_token:
        from mcp.server.auth.settings import AuthSettings

        kwargs["token_verifier"] = StaticTokenVerifier(auth_token)
        kwargs["auth"] = AuthSettings(
            issuer_url="https://localhost",
            resource_server_url=None,
        )

    server = FastMCP("reka-vision", lifespan=lifespan, **kwargs)
    server._mcp_server.version = pkg_version("reka-mcp")
    register_video_tools(server, client)
    register_group_tools(server, client)
    register_indexing_tools(
        server,
        client,
        index_timeout=index_timeout,
        poll_interval=poll_interval,
    )
    register_search_tools(server, client)
    register_qa_tools(server, client)
    register_segment_tools(server, client)
    register_sub_resource_tools(server, client)
    register_resources(server, client)
    return server


def main() -> None:
    config = load_config()
    server = create_server(
        api_url=config.api_url,
        api_key=config.api_key,
        index_timeout=config.index_timeout,
        poll_interval=config.poll_interval,
        auth_token=config.auth_token,
    )

    transport: Literal["stdio", "streamable-http"] = (
        "streamable-http" if config.transport == "http" else "stdio"
    )
    logger.info(
        "transport=%s, port=%s, auth=%s",
        transport,
        config.http_port if config.transport == "http" else "n/a",
        "enabled" if config.auth_token else "disabled",
    )
    if config.transport == "http":
        server.settings.port = config.http_port
    server.run(transport=transport)
