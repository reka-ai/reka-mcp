# ABOUTME: Tests for video management MCP tools.
# ABOUTME: Covers upload, list, get, delete tools and error propagation.

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from reka_mcp.client import RekaClient
from reka_mcp.tools.videos import register_video_tools
from tests.conftest import mock_client, tool_result_text


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_video_tools(server, client)
    return server


class TestUploadVideo:
    async def test_upload_returns_video_id_and_status(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(202, json={"video_id": "vid-abc", "status": "uploading"}),
        )
        result = await mcp_server.call_tool(
            "upload_video",
            {"video_url": "https://example.com/video.mp4", "name": "test"},
        )
        body = json.loads(tool_result_text(result))
        assert body["video_id"] == "vid-abc"
        assert body["status"] == "uploading"

    async def test_upload_without_optional_params(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(202, json={"video_id": "vid-1", "status": "uploading"}),
        )
        result = await mcp_server.call_tool(
            "upload_video",
            {"video_url": "https://example.com/video.mp4"},
        )
        body = json.loads(tool_result_text(result))
        assert body["video_id"] == "vid-1"

    async def test_upload_response_includes_hint(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(202, json={"video_id": "vid-abc", "status": "uploading"}),
        )
        result = await mcp_server.call_tool(
            "upload_video",
            {"video_url": "https://example.com/video.mp4"},
        )
        body = json.loads(tool_result_text(result))
        assert "hint" in body
        assert "index_video" in body["hint"]


class TestUploadVideoFromFile:
    async def test_upload_from_file_path(
        self, client: RekaClient, mcp_server: FastMCP, tmp_path: Any
    ) -> None:
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"fake video content")

        mock_client(
            client,
            lambda req: httpx.Response(202, json={"video_id": "vid-file", "status": "uploading"}),
        )
        result = await mcp_server.call_tool(
            "upload_video",
            {"file_path": str(video_file), "name": "my clip"},
        )
        body = json.loads(tool_result_text(result))
        assert body["video_id"] == "vid-file"
        assert "hint" in body

    async def test_upload_file_requires_name(
        self, client: RekaClient, mcp_server: FastMCP, tmp_path: Any
    ) -> None:
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"data")

        with pytest.raises(ToolError, match=r"(?i)name.*required"):
            await mcp_server.call_tool(
                "upload_video",
                {"file_path": str(video_file)},
            )

    async def test_upload_rejects_both_url_and_file(
        self, client: RekaClient, mcp_server: FastMCP, tmp_path: Any
    ) -> None:
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"data")

        with pytest.raises(ToolError, match=r"(?i)exactly one"):
            await mcp_server.call_tool(
                "upload_video",
                {
                    "video_url": "https://example.com/v.mp4",
                    "file_path": str(video_file),
                },
            )

    async def test_upload_rejects_neither_url_nor_file(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        with pytest.raises(ToolError, match=r"(?i)exactly one"):
            await mcp_server.call_tool("upload_video", {})

    async def test_upload_file_not_found(self, client: RekaClient, mcp_server: FastMCP) -> None:
        with pytest.raises(ToolError, match=r"(?i)not found|no such file"):
            await mcp_server.call_tool(
                "upload_video",
                {"file_path": "/nonexistent/video.mp4", "name": "test"},
            )


class TestListVideos:
    async def test_list_returns_array(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={"results": [{"video_id": "v1", "status": "uploaded", "features": {}}]},
            ),
        )
        result = await mcp_server.call_tool("list_videos", {})
        body = json.loads(tool_result_text(result))
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["video_id"] == "v1"

    async def test_list_with_group_id_uses_group_endpoint(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert "/v2/video-groups/g1/videos" in str(req.url)
            return httpx.Response(
                200,
                json={"results": [{"video_id": "v1", "status": "uploaded", "features": {}}]},
            )

        mock_client(client, handler)
        result = await mcp_server.call_tool("list_videos", {"group_id": "g1"})
        body = json.loads(tool_result_text(result))
        assert len(body) == 1


class TestGetVideo:
    async def test_get_returns_full_details(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "video_id": "vid-123",
                    "status": "uploaded",
                    "metadata": {"duration": 120.5, "width": 1920, "height": 1080},
                    "features": {"transcript": "ready", "embeddings": "processing"},
                },
            ),
        )
        result = await mcp_server.call_tool("get_video", {"video_id": "vid-123"})
        body = json.loads(tool_result_text(result))
        assert body["video_id"] == "vid-123"
        assert body["metadata"]["duration"] == 120.5
        assert body["features"]["transcript"] == "ready"

    async def test_get_not_found_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(404, json={"error": {"message": "Video not found"}}),
        )
        with pytest.raises(ToolError, match=r"(?i)not found"):
            await mcp_server.call_tool(
                "get_video", {"video_id": "00000000-0000-0000-0000-000000000000"}
            )


class TestDeleteVideo:
    async def test_delete_returns_success(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200, json={"status": "success", "message": "Video deleted successfully"}
            ),
        )
        result = await mcp_server.call_tool("delete_video", {"video_id": "vid-123"})
        body = json.loads(tool_result_text(result))
        assert body["status"] == "success"


class TestAuthError:
    async def test_auth_error_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(401, json={"error": {"message": "bad key"}}),
        )
        with pytest.raises(ToolError, match=r"(?i)authentication failed"):
            await mcp_server.call_tool("get_video", {"video_id": "vid-1"})
