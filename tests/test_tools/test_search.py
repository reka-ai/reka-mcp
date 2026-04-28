# ABOUTME: Tests for the search_videos MCP tool.
# ABOUTME: Covers ranked results, empty results, video_ids filter, and error handling.

from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from reka_mcp.client import RekaClient
from reka_mcp.tools.search import register_search_tools
from tests.conftest import mock_client, tool_result_text


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_search_tools(server, client)
    return server


class TestSearchVideos:
    async def test_returns_ranked_results(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "video_id": "v1",
                            "video_name": "Lecture",
                            "start": 10.0,
                            "end": 15.0,
                            "score": 0.95,
                            "rank": 1,
                            "caption": "A person presenting",
                            "transcript": "Today we discuss...",
                        },
                        {
                            "video_id": "v2",
                            "video_name": "Demo",
                            "start": 30.0,
                            "end": 35.0,
                            "score": 0.80,
                            "rank": 2,
                            "caption": None,
                            "transcript": "Let me show you...",
                        },
                    ],
                    "next_page_token": None,
                    "search_pool": {"video_count": 5, "total_duration": 3600.0},
                },
            ),
        )
        result = await mcp_server.call_tool("search_videos", {"query": "person speaking"})
        body = json.loads(tool_result_text(result))
        results = body["results"]
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["video_id"] == "v1"
        assert results[0]["score"] == 0.95
        assert results[0]["start"] == 10.0
        assert results[0]["end"] == 15.0

    async def test_empty_results(self, client: RekaClient, mcp_server: FastMCP) -> None:
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
        result = await mcp_server.call_tool(
            "search_videos", {"query": "xyzzy_gibberish_no_match_12345"}
        )
        body = json.loads(tool_result_text(result))
        assert isinstance(body["results"], list)
        assert len(body["results"]) == 0

    async def test_with_video_ids_filter(self, client: RekaClient, mcp_server: FastMCP) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body_data = json.loads(req.content)
            assert body_data["video_ids"] == ["v1"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "video_id": "v1",
                            "start": 5.0,
                            "end": 10.0,
                            "score": 0.9,
                            "rank": 1,
                        }
                    ],
                    "next_page_token": None,
                    "search_pool": {"video_count": 1, "total_duration": 120.0},
                },
            )

        mock_client(client, handler)
        result = await mcp_server.call_tool(
            "search_videos",
            {"query": "chart", "video_ids": ["v1"]},
        )
        body = json.loads(tool_result_text(result))
        assert len(body["results"]) == 1
        assert body["results"][0]["video_id"] == "v1"

    async def test_response_includes_hint(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [
                        {"video_id": "v1", "start": 10.0, "end": 15.0, "score": 0.9, "rank": 1},
                    ],
                    "next_page_token": None,
                    "search_pool": {"video_count": 1, "total_duration": 120.0},
                },
            ),
        )
        result = await mcp_server.call_tool("search_videos", {"query": "person speaking"})
        body = json.loads(tool_result_text(result))
        assert "hint" in body
        assert "ask_video" in body["hint"]
        assert "results" in body
        assert isinstance(body["results"], list)

    async def test_error_raises_tool_error(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(401, json={"error": {"message": "bad key"}}),
        )
        with pytest.raises(ToolError, match=r"(?i)authentication failed"):
            await mcp_server.call_tool("search_videos", {"query": "test"})
