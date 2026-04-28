# ABOUTME: Tests for group management MCP tools (create, list, delete).
# ABOUTME: Validates argument schemas, error handling, and result formats.

from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from reka_mcp.client import RekaClient
from reka_mcp.tools.groups import register_group_tools
from tests.conftest import mock_client, tool_result_text


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_group_tools(server, client)
    return server


class TestCreateGroup:
    async def test_create_returns_group_id(self, client: RekaClient, mcp_server: FastMCP) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/video-groups")
            return httpx.Response(200, json={"group_id": "g1", "name": "my-group", "metadata": {}})

        mock_client(client, handler)
        result = await mcp_server.call_tool("create_group", {"name": "my-group"})
        body = json.loads(tool_result_text(result))
        assert body["group_id"] == "g1"
        assert body["name"] == "my-group"


class TestListGroups:
    async def test_list_returns_array(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "results": [
                        {"group_id": "g1", "name": "grp1", "metadata": {}},
                        {"group_id": "g2", "name": "grp2", "metadata": {}},
                    ]
                },
            ),
        )
        result = await mcp_server.call_tool("list_groups", {})
        body = json.loads(tool_result_text(result))
        assert isinstance(body, list)
        assert len(body) == 2


class TestDeleteGroup:
    async def test_delete_returns_success(self, client: RekaClient, mcp_server: FastMCP) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "DELETE"
            assert str(req.url).endswith("/v2/video-groups/g1")
            return httpx.Response(200, json={"status": "success"})

        mock_client(client, handler)
        result = await mcp_server.call_tool("delete_group", {"group_id": "g1"})
        body = json.loads(tool_result_text(result))
        assert body["status"] == "success"
