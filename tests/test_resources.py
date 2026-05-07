# ABOUTME: Tests for MCP resources (feature catalog, video metadata, workflow guide).
# ABOUTME: Verifies resource registration, content, and URI routing.

from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from reka_mcp.client import RekaClient
from reka_mcp.resources import register_resources
from tests.conftest import mock_client


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_resources(server, client)
    return server


class TestFeatureCatalogResource:
    async def test_listed_in_resources(self, mcp_server: FastMCP) -> None:
        resources = await mcp_server.list_resources()
        uris = [str(r.uri) for r in resources]
        assert "reka://features" in uris

    async def test_returns_catalog(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "name": "transcript",
                            "description": "Speech-to-text",
                            "depends_on": [],
                        },
                    ]
                },
            ),
        )
        contents = await mcp_server.read_resource("reka://features")
        body = json.loads(contents[0].content)
        assert "transcript" in body


class TestWorkflowGuideResource:
    async def test_listed_in_resources(self, mcp_server: FastMCP) -> None:
        resources = await mcp_server.list_resources()
        uris = [str(r.uri) for r in resources]
        assert "reka://docs/guide" in uris

    async def test_returns_text_content(self, mcp_server: FastMCP) -> None:
        contents = await mcp_server.read_resource("reka://docs/guide")
        text = contents[0].content
        assert isinstance(text, str)
        assert len(text) > 100

    async def test_covers_key_workflows(self, mcp_server: FastMCP) -> None:
        contents = await mcp_server.read_resource("reka://docs/guide")
        text = contents[0].content
        assert "search_videos" in text
        assert "segment_video" in text
        assert "ask_video" in text
        assert "get_transcript" in text
        assert "summarize_video" in text

    async def test_covers_refinement_strategies(self, mcp_server: FastMCP) -> None:
        contents = await mcp_server.read_resource("reka://docs/guide")
        text = contents[0].content
        assert "get_scenes" in text
        assert "scene" in text.lower()
        assert "second" in text.lower()
        assert "more API calls" in text or "more expensive" in text or "cost" in text.lower()

    async def test_covers_upload_index_polling_workflow(self, mcp_server: FastMCP) -> None:
        contents = await mcp_server.read_resource("reka://docs/guide")
        text = contents[0].content
        assert "upload_video" in text
        assert "index_video" in text
        assert "poll get_video" in text
        assert "features" in text


class TestVideoResource:
    async def test_listed_in_templates(self, mcp_server: FastMCP) -> None:
        templates = await mcp_server.list_resource_templates()
        uri_templates = [t.uriTemplate for t in templates]
        assert "reka://videos/{video_id}" in uri_templates

    async def test_returns_video_metadata(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "video_id": "v1",
                    "status": "uploaded",
                    "metadata": {"video_name": "Test", "duration": 120.0},
                    "features": {"transcript": "ready"},
                },
            ),
        )
        contents = await mcp_server.read_resource("reka://videos/v1")
        body = json.loads(contents[0].content)
        assert body["video_id"] == "v1"
        assert body["status"] == "uploaded"
