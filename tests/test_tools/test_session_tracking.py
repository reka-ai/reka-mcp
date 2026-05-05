# ABOUTME: Tests for MCP session ID propagation through tool calls.
# ABOUTME: Verifies session ID flows from MCP request context to outgoing API headers.

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from reka_mcp.client import RekaClient, mcp_session_id_var
from reka_mcp.tools import _extract_mcp_session_id
from reka_mcp.tools.search import register_search_tools
from tests.conftest import mock_client


@dataclass
class FakeRequest:
    headers: dict[str, str]


class TestExtractMcpSessionId:
    def test_returns_session_id_from_request_headers(self) -> None:
        from mcp.server.lowlevel.server import request_ctx
        from mcp.shared.context import RequestContext

        ctx = RequestContext(
            request_id="req-1",
            meta=None,
            session=MagicMock(),
            lifespan_context=None,
            request=FakeRequest(headers={"mcp-session-id": "sess-xyz"}),
        )
        token = request_ctx.set(ctx)
        try:
            assert _extract_mcp_session_id() == "sess-xyz"
        finally:
            request_ctx.reset(token)

    def test_returns_none_when_no_request_context(self) -> None:
        assert _extract_mcp_session_id() is None

    def test_returns_none_when_request_is_none(self) -> None:
        from mcp.server.lowlevel.server import request_ctx
        from mcp.shared.context import RequestContext

        ctx = RequestContext(
            request_id="req-1",
            meta=None,
            session=MagicMock(),
            lifespan_context=None,
            request=None,
        )
        token = request_ctx.set(ctx)
        try:
            assert _extract_mcp_session_id() is None
        finally:
            request_ctx.reset(token)

    def test_returns_none_when_header_absent(self) -> None:
        from mcp.server.lowlevel.server import request_ctx
        from mcp.shared.context import RequestContext

        ctx = RequestContext(
            request_id="req-1",
            meta=None,
            session=MagicMock(),
            lifespan_context=None,
            request=FakeRequest(headers={}),
        )
        token = request_ctx.set(ctx)
        try:
            assert _extract_mcp_session_id() is None
        finally:
            request_ctx.reset(token)


class TestSessionIdPropagation:
    """End-to-end: MCP request context → @logged → contextvar → API header."""

    @pytest.fixture
    def mcp_server(self, client: RekaClient) -> FastMCP:
        server = FastMCP("test-reka-vision")
        register_search_tools(server, client)
        return server

    async def test_session_id_sent_to_api(self, client: RekaClient, mcp_server: FastMCP) -> None:
        from mcp.server.lowlevel.server import request_ctx
        from mcp.shared.context import RequestContext

        captured_headers: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_headers.update(req.headers)
            return httpx.Response(
                200,
                json={
                    "data": [],
                    "next_page_token": None,
                    "search_pool": {"video_count": 0, "total_duration": 0.0},
                },
            )

        mock_client(client, handler)

        ctx = RequestContext(
            request_id="req-1",
            meta=None,
            session=MagicMock(),
            lifespan_context=None,
            request=FakeRequest(headers={"mcp-session-id": "sess-end2end"}),
        )
        token = request_ctx.set(ctx)
        try:
            await mcp_server.call_tool("search_videos", {"query": "test"})
        finally:
            request_ctx.reset(token)

        assert captured_headers["x-mcp-session-id"] == "sess-end2end"

    async def test_no_session_header_without_mcp_context(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        captured_headers: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_headers.update(req.headers)
            return httpx.Response(
                200,
                json={
                    "data": [],
                    "next_page_token": None,
                    "search_pool": {"video_count": 0, "total_duration": 0.0},
                },
            )

        mock_client(client, handler)
        await mcp_server.call_tool("search_videos", {"query": "test"})
        assert "x-mcp-session-id" not in captured_headers

    async def test_contextvar_reset_after_tool_call(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        from mcp.server.lowlevel.server import request_ctx
        from mcp.shared.context import RequestContext

        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [],
                    "next_page_token": None,
                    "search_pool": {"video_count": 0, "total_duration": 0.0},
                },
            ),
        )

        ctx = RequestContext(
            request_id="req-1",
            meta=None,
            session=MagicMock(),
            lifespan_context=None,
            request=FakeRequest(headers={"mcp-session-id": "sess-temp"}),
        )
        token = request_ctx.set(ctx)
        try:
            await mcp_server.call_tool("search_videos", {"query": "test"})
        finally:
            request_ctx.reset(token)

        assert mcp_session_id_var.get() is None
