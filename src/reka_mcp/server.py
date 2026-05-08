# ABOUTME: MCP server entry point for Reka Vision Agent.
# ABOUTME: Supports stdio and streamable HTTP transports.

from __future__ import annotations

import logging
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

if TYPE_CHECKING:
    from starlette.requests import Request

from reka_mcp.auth import StaticTokenVerifier
from reka_mcp.client import RekaClient
from reka_mcp.config import RuntimeMode, load_config
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
    api_key: str | None,
    index_timeout: int = 600,
    poll_interval: int = 5,
    auth_token: str | None = None,
    mode: RuntimeMode = "local",
    http_host: str = "127.0.0.1",
    http_port: int = 8000,
    http_path: str = "/mcp",
    allowed_hosts: tuple[str, ...] = (),
    allowed_origins: tuple[str, ...] = (),
) -> FastMCP:
    client = RekaClient(api_url=api_url, api_key=api_key)

    kwargs: dict[str, Any] = {}
    if auth_token:
        from mcp.server.auth.settings import AuthSettings

        kwargs["token_verifier"] = StaticTokenVerifier(auth_token)
        kwargs["auth"] = AuthSettings(
            issuer_url="https://localhost",
            resource_server_url=None,
        )

    transport_security: TransportSecuritySettings | None = None
    if mode == "hosted" or allowed_hosts or allowed_origins:
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(allowed_hosts),
            allowed_origins=list(allowed_origins),
        )

    # No lifespan: the MCP SDK runs the lifespan per session, not per server.
    # The RekaClient is shared across sessions; per-request auth is via contextvars.
    server = FastMCP(
        "reka-vision",
        host=http_host,
        port=http_port,
        streamable_http_path=http_path,
        transport_security=transport_security,
        **kwargs,
    )
    server._mcp_server.version = pkg_version("reka-mcp")

    if mode == "hosted":
        from starlette.responses import JSONResponse

        async def health_check(request: Request) -> JSONResponse:
            return JSONResponse({"status": "ok"})

        server.custom_route("/health", methods=["GET"], include_in_schema=False)(health_check)

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
    import sys

    if "--version" in sys.argv:
        print(pkg_version("reka-mcp"))
        return

    config = load_config()
    server = create_server(
        api_url=config.api_url,
        api_key=config.api_key,
        index_timeout=config.index_timeout,
        poll_interval=config.poll_interval,
        auth_token=config.auth_token,
        mode=config.mode,
        http_host=config.http_host,
        http_port=config.http_port,
        http_path=config.http_path,
        allowed_hosts=config.allowed_hosts,
        allowed_origins=config.allowed_origins,
    )

    transport: Literal["stdio", "streamable-http"] = (
        "streamable-http" if config.transport == "http" else "stdio"
    )
    logger.info(
        "mode=%s, transport=%s, host=%s, port=%s, path=%s, auth=%s",
        config.mode,
        transport,
        config.http_host if config.transport == "http" else "n/a",
        config.http_port if config.transport == "http" else "n/a",
        config.http_path if config.transport == "http" else "n/a",
        "enabled" if config.auth_token else "disabled",
    )
    server.run(transport=transport)
