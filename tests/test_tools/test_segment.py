# ABOUTME: Tests for the segment_video MCP tool.
# ABOUTME: Covers detection results, empty results, hints, parameter validation, and errors.

from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from reka_mcp.client import RekaClient
from reka_mcp.tools.segment import register_segment_tools
from tests.conftest import mock_client, tool_result_text

SEGMENT_RESPONSE = {
    "frames": [
        {
            "timestamp": 10.0,
            "detections": [
                {
                    "label": "person",
                    "prompt_index": 0,
                    "score": 0.95,
                    "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8},
                },
                {
                    "label": "laptop",
                    "prompt_index": 1,
                    "score": 0.82,
                    "bbox": {"x_min": 0.6, "y_min": 0.5, "x_max": 0.9, "y_max": 0.7},
                },
            ],
        },
        {
            "timestamp": 11.0,
            "detections": [
                {
                    "label": "person",
                    "prompt_index": 0,
                    "score": 0.91,
                    "bbox": {"x_min": 0.12, "y_min": 0.22, "x_max": 0.52, "y_max": 0.82},
                },
            ],
        },
    ],
    "frame_size": {"width": 1920, "height": 1080},
    "frame_count": 2,
}


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_segment_tools(server, client)
    return server


class TestSegmentVideo:
    async def test_returns_detections_with_summary(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(200, json=SEGMENT_RESPONSE),
        )
        result = await mcp_server.call_tool(
            "segment_video",
            {"video_id": "v1", "prompts": ["person", "laptop"], "start": 10.0, "end": 12.0},
        )
        body = json.loads(tool_result_text(result))
        assert body["frame_count"] == 2
        assert len(body["frames"]) == 2
        assert body["frames"][0]["detections"][0]["label"] == "person"
        assert body["summary"]["total_detections"] == 3
        assert sorted(body["summary"]["detected_labels"]) == ["laptop", "person"]
        assert body["summary"]["frames_with_detections"] == 2

    async def test_empty_detections_hint(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "frames": [{"timestamp": 5.0, "detections": []}],
                    "frame_size": {"width": 1920, "height": 1080},
                    "frame_count": 1,
                },
            ),
        )
        result = await mcp_server.call_tool(
            "segment_video",
            {"video_id": "v1", "prompts": ["helicopter"], "start": 5.0},
        )
        body = json.loads(tool_result_text(result))
        assert body["summary"]["total_detections"] == 0
        assert "hint" in body
        assert "threshold" in body["hint"].lower() or "prompt" in body["hint"].lower()

    async def test_detections_found_hint(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(200, json=SEGMENT_RESPONSE),
        )
        result = await mcp_server.call_tool(
            "segment_video",
            {"video_id": "v1", "prompts": ["person"], "start": 10.0},
        )
        body = json.loads(tool_result_text(result))
        assert "hint" in body
        assert "ask_video" in body["hint"]

    async def test_passes_threshold_to_api(self, client: RekaClient, mcp_server: FastMCP) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["threshold"] == 0.7
            return httpx.Response(
                200,
                json={
                    "frames": [],
                    "frame_size": {"width": 1920, "height": 1080},
                    "frame_count": 0,
                },
            )

        mock_client(client, handler)
        await mcp_server.call_tool(
            "segment_video",
            {"video_id": "v1", "prompts": ["car"], "start": 0.0, "threshold": 0.7},
        )

    async def test_passes_end_to_api(self, client: RekaClient, mcp_server: FastMCP) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["start"] == 5.0
            assert body["end"] == 18.0
            return httpx.Response(
                200,
                json={
                    "frames": [],
                    "frame_size": {"width": 1920, "height": 1080},
                    "frame_count": 0,
                },
            )

        mock_client(client, handler)
        await mcp_server.call_tool(
            "segment_video",
            {"video_id": "v1", "prompts": ["dog"], "start": 5.0, "end": 18.0},
        )

    async def test_not_found_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(404, json={"error": {"message": "Video not found"}}),
        )
        with pytest.raises(ToolError, match=r"(?i)not found"):
            await mcp_server.call_tool(
                "segment_video",
                {"video_id": "missing", "prompts": ["person"], "start": 0.0},
            )

    async def test_validation_error_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                422, json={"error": {"message": "start exceeds video duration"}}
            ),
        )
        with pytest.raises(ToolError, match="start exceeds"):
            await mcp_server.call_tool(
                "segment_video",
                {"video_id": "v1", "prompts": ["person"], "start": 9999.0},
            )

    async def test_multiple_prompts_in_request(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["prompts"] == [
                {"type": "text", "text": "person"},
                {"type": "text", "text": "car"},
                {"type": "text", "text": "bicycle"},
            ]
            return httpx.Response(
                200,
                json={
                    "frames": [],
                    "frame_size": {"width": 1920, "height": 1080},
                    "frame_count": 0,
                },
            )

        mock_client(client, handler)
        await mcp_server.call_tool(
            "segment_video",
            {"video_id": "v1", "prompts": ["person", "car", "bicycle"], "start": 0.0},
        )

    async def test_summary_counts_frames_with_detections(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        response = {
            "frames": [
                {
                    "timestamp": 0.0,
                    "detections": [
                        {
                            "label": "cat",
                            "prompt_index": 0,
                            "score": 0.9,
                            "bbox": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
                        },
                    ],
                },
                {"timestamp": 1.0, "detections": []},
                {"timestamp": 2.0, "detections": []},
                {
                    "timestamp": 3.0,
                    "detections": [
                        {
                            "label": "cat",
                            "prompt_index": 0,
                            "score": 0.85,
                            "bbox": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
                        },
                    ],
                },
            ],
            "frame_size": {"width": 1280, "height": 720},
            "frame_count": 4,
        }
        mock_client(
            client,
            lambda req: httpx.Response(200, json=response),
        )
        result = await mcp_server.call_tool(
            "segment_video",
            {"video_id": "v1", "prompts": ["cat"], "start": 0.0, "end": 4.0},
        )
        body = json.loads(tool_result_text(result))
        assert body["summary"]["total_detections"] == 2
        assert body["summary"]["frames_with_detections"] == 2
        assert body["summary"]["detected_labels"] == ["cat"]
