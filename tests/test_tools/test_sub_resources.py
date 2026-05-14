# ABOUTME: Tests for sub-resource MCP tools (transcript, captions, scenes).
# ABOUTME: Covers response shaping, truncation metadata, max_results capping, and errors.

from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from reka_mcp.client import RekaClient
from reka_mcp.tools.sub_resources import register_sub_resource_tools
from tests.conftest import mock_client, tool_result_text


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_sub_resource_tools(server, client)
    return server


class TestGetTranscript:
    async def test_text_format_with_truncation(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        long_text = "a" * 15000
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"text": long_text}),
        )
        result = await mcp_server.call_tool("get_transcript", {"video_id": "v1", "format": "text"})
        body = json.loads(tool_result_text(result))
        assert body["truncated"] is True
        assert body["total_chars"] == 15000
        assert len(body["text"]) == 10000

    async def test_text_format_short_not_truncated(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"text": "short text"}),
        )
        result = await mcp_server.call_tool("get_transcript", {"video_id": "v1", "format": "text"})
        body = json.loads(tool_result_text(result))
        assert body["truncated"] is False
        assert body["text"] == "short text"
        assert body["total_chars"] == 10

    async def test_text_format_custom_max_chars(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"text": "a" * 1000}),
        )
        result = await mcp_server.call_tool(
            "get_transcript",
            {"video_id": "v1", "format": "text", "max_chars": 500},
        )
        body = json.loads(tool_result_text(result))
        assert body["truncated"] is True
        assert len(body["text"]) == 500

    async def test_segments_format_with_cap(self, client: RekaClient, mcp_server: FastMCP) -> None:
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"seg{i}"} for i in range(150)
        ]
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"data": segments, "next_page_token": None}),
        )
        result = await mcp_server.call_tool(
            "get_transcript", {"video_id": "v1", "format": "segments"}
        )
        body = json.loads(tool_result_text(result))
        assert len(body["data"]) == 100
        assert body["returned_count"] == 100
        assert body["truncated"] is True

    async def test_segments_custom_max_results(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        segments = [{"start": float(i), "end": float(i + 1), "text": f"seg{i}"} for i in range(10)]
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"data": segments, "next_page_token": None}),
        )
        result = await mcp_server.call_tool(
            "get_transcript",
            {"video_id": "v1", "format": "segments", "max_results": 5},
        )
        body = json.loads(tool_result_text(result))
        assert len(body["data"]) == 5
        assert body["returned_count"] == 5
        assert body["truncated"] is True

    async def test_segments_stops_fetching_at_max_results(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            offset = (call_count - 1) * 50
            segments = [
                {"start": float(i), "end": float(i + 1), "text": f"seg{i}"}
                for i in range(offset, offset + 50)
            ]
            token = f"page{call_count + 1}" if call_count < 5 else None
            return httpx.Response(
                200,
                json={"data": segments, "next_page_token": token},
            )

        mock_client(client, handler)
        result = await mcp_server.call_tool(
            "get_transcript",
            {"video_id": "v1", "format": "segments", "max_results": 10},
        )
        body = json.loads(tool_result_text(result))
        assert len(body["data"]) == 10
        assert body["truncated"] is True
        assert call_count == 1

    async def test_invalid_format_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        with pytest.raises(ToolError, match="Input should be"):
            await mcp_server.call_tool("get_transcript", {"video_id": "v1", "format": "csv"})

    async def test_unindexed_feature_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                409,
                json={
                    "error": {"message": "Feature 'transcript' is not ready. Current status: none"}
                },
            ),
        )
        with pytest.raises(ToolError, match="not ready"):
            await mcp_server.call_tool("get_transcript", {"video_id": "v1"})

    async def test_not_found_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(404, json={"error": {"message": "Video not found"}}),
        )
        with pytest.raises(ToolError, match=r"(?i)not found"):
            await mcp_server.call_tool(
                "get_transcript",
                {"video_id": "00000000-0000-0000-0000-000000000000"},
            )


class TestGetCaptions:
    async def test_returns_capped_data(self, client: RekaClient, mcp_server: FastMCP) -> None:
        captions = [
            {"start": float(i), "end": float(i + 1), "caption": f"cap{i}"} for i in range(80)
        ]
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"data": captions, "next_page_token": None}),
        )
        result = await mcp_server.call_tool("get_captions", {"video_id": "v1"})
        body = json.loads(tool_result_text(result))
        assert len(body["data"]) == 50
        assert body["returned_count"] == 50
        assert body["truncated"] is True

    async def test_not_truncated_when_under_limit(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        captions = [{"start": 0.0, "end": 10.0, "caption": "A person speaking"}]
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"data": captions, "next_page_token": None}),
        )
        result = await mcp_server.call_tool("get_captions", {"video_id": "v1"})
        body = json.loads(tool_result_text(result))
        assert body["truncated"] is False
        assert body["returned_count"] == 1


class TestGetScenes:
    async def test_returns_scenes(self, client: RekaClient, mcp_server: FastMCP) -> None:
        scenes = [
            {"index": 0, "start": 0.0, "end": 15.0},
            {"index": 1, "start": 15.0, "end": 30.0},
        ]
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"data": scenes, "next_page_token": None}),
        )
        result = await mcp_server.call_tool("get_scenes", {"video_id": "v1"})
        body = json.loads(tool_result_text(result))
        assert len(body["data"]) == 2
        assert body["returned_count"] == 2
        assert body["truncated"] is False

    async def test_caps_at_max_results(self, client: RekaClient, mcp_server: FastMCP) -> None:
        scenes = [{"index": i, "start": float(i), "end": float(i + 1)} for i in range(600)]
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"data": scenes, "next_page_token": None}),
        )
        result = await mcp_server.call_tool("get_scenes", {"video_id": "v1"})
        body = json.loads(tool_result_text(result))
        assert body["returned_count"] <= 200
        assert body["truncated"] is True

    async def test_custom_max_results(self, client: RekaClient, mcp_server: FastMCP) -> None:
        scenes = [{"index": i, "start": float(i), "end": float(i + 1)} for i in range(20)]
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"data": scenes, "next_page_token": None}),
        )
        result = await mcp_server.call_tool("get_scenes", {"video_id": "v1", "max_results": 5})
        body = json.loads(tool_result_text(result))
        assert len(body["data"]) == 5
        assert body["returned_count"] == 5
        assert body["truncated"] is True


class TestGetFeatureCatalog:
    async def test_returns_catalog(self, client: RekaClient, mcp_server: FastMCP) -> None:
        catalog = {
            "features": [
                {
                    "name": "transcript",
                    "description": "Speech-to-text",
                    "depends_on": [],
                    "produces": ["transcript", "scenes"],
                },
                {
                    "name": "captions",
                    "description": "Visual descriptions",
                    "depends_on": ["transcript"],
                },
                {
                    "name": "embeddings",
                    "description": "Vector embeddings",
                    "depends_on": ["transcript", "captions"],
                },
            ]
        }
        mock_client(
            client,
            lambda req: httpx.Response(200, json=catalog),
        )
        result = await mcp_server.call_tool("get_feature_catalog", {})
        body = json.loads(tool_result_text(result))
        assert "transcript" in body
        assert "captions" in body
        assert "embeddings" in body
        assert body["transcript"]["depends_on"] == []
        assert body["transcript"]["produces"] == ["transcript", "scenes"]


class TestSummarizeVideo:
    async def test_returns_compact_overview(self, client: RekaClient, mcp_server: FastMCP) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            url = str(req.url)

            if url.endswith("/v2/videos/v1"):
                return httpx.Response(
                    200,
                    json={
                        "video_id": "v1",
                        "status": "uploaded",
                        "metadata": {
                            "video_name": "Lecture 3",
                            "duration": 3600.0,
                        },
                        "features": {
                            "transcript": "ready",
                            "captions": "ready",
                            "embeddings": "ready",
                        },
                    },
                )
            if "/transcript" in url:
                return httpx.Response(
                    200,
                    json={"text": "Welcome to today's lecture on distributed systems."},
                )
            if "/scenes" in url:
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"index": 0, "start": 0.0, "end": 15.0},
                            {"index": 1, "start": 15.0, "end": 30.0},
                        ],
                        "next_page_token": None,
                    },
                )
            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)
        result = await mcp_server.call_tool("summarize_video", {"video_id": "v1"})
        body = json.loads(tool_result_text(result))
        assert body["video_id"] == "v1"
        assert body["name"] == "Lecture 3"
        assert body["duration_seconds"] == 3600.0
        assert body["status"] == "uploaded"
        assert body["features"]["transcript"] == "ready"
        assert body["scene_count"] == 2
        assert "lecture" in body["transcript_preview"].lower()

    async def test_unindexed_video_omits_optional_fields(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "video_id": "v2",
                    "status": "uploaded",
                    "metadata": {
                        "video_name": "Raw Video",
                        "duration": 60.0,
                    },
                    "features": {
                        "transcript": "none",
                        "captions": "none",
                        "embeddings": "none",
                    },
                },
            ),
        )
        result = await mcp_server.call_tool("summarize_video", {"video_id": "v2"})
        body = json.loads(tool_result_text(result))
        assert body["video_id"] == "v2"
        assert body["name"] == "Raw Video"
        assert body["features"]["transcript"] == "none"
        assert "scene_count" not in body
        assert "transcript_preview" not in body

    async def test_failed_sub_resources_produce_warnings(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            if url.endswith("/v2/videos/v1"):
                return httpx.Response(
                    200,
                    json={
                        "video_id": "v1",
                        "status": "uploaded",
                        "metadata": {"video_name": "Test", "duration": 60.0},
                        "features": {
                            "transcript": "ready",
                        },
                    },
                )
            if "/transcript" in url:
                return httpx.Response(
                    200,
                    json={"text": "Hello world."},
                )
            if "/scenes" in url:
                return httpx.Response(500, json={"error": {"message": "internal error"}})
            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)
        result = await mcp_server.call_tool("summarize_video", {"video_id": "v1"})
        body = json.loads(tool_result_text(result))
        assert body["transcript_preview"] == "Hello world."
        assert "scene_count" not in body
        assert len(body["warnings"]) == 1
        assert any("scenes" in w for w in body["warnings"])
