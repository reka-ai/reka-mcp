# ABOUTME: Hosted tool-policy tests for mode-specific upload and indexing behavior.
# ABOUTME: Locks hosted MCP to URL-only uploads.

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from reka_mcp.server import create_server
from tests.conftest import tool_result_text


def _hosted_server(**kwargs):
    return create_server(
        api_url="http://test-api.local",
        api_key=None,
        mode="hosted",
        http_host="0.0.0.0",
        http_port=8080,
        http_path="/mcp",
        allowed_hosts=("localhost:*",),
        allowed_origins=("http://localhost:*",),
        **kwargs,
    )


class TestHostedUploadPolicy:
    async def test_hosted_upload_video_schema_is_url_only(self) -> None:
        server = _hosted_server()

        tools = {tool.name: tool for tool in await server.list_tools()}
        upload_schema = tools["upload_video"].inputSchema["properties"]

        assert "video_url" in upload_schema
        assert "file_path" not in upload_schema

    async def test_hosted_upload_video_rejects_file_path_before_api_auth(self, tmp_path) -> None:
        server = _hosted_server()
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"local file must not be accepted by hosted MCP")

        with pytest.raises(ToolError, match=r"(?i)video_url|file_path"):
            await server.call_tool(
                "upload_video",
                {"file_path": str(video_file), "name": "Local file"},
            )


class TestHostedIndexVideoFilePathRejection:
    async def test_hosted_index_video_rejects_file_path(self, tmp_path) -> None:
        server = _hosted_server()
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"should be rejected in hosted mode")

        with pytest.raises(ToolError):
            await server.call_tool(
                "index_video",
                {"file_path": str(video_file), "pipeline": "search_only"},
            )

    async def test_hosted_index_video_schema_excludes_file_path(self) -> None:
        server = _hosted_server()
        tools = {tool.name: tool for tool in await server.list_tools()}
        index_schema = tools["index_video"].inputSchema["properties"]

        assert "video_id" in index_schema
        assert "file_path" not in index_schema

    async def test_hosted_index_video_description_omits_file_path(self) -> None:
        server = _hosted_server()
        tools = {tool.name: tool for tool in await server.list_tools()}
        description = tools["index_video"].description or ""

        assert "file_path" not in description


class TestHostedIndexingPolicy:
    async def test_hosted_index_video_description_mentions_pipelines(self) -> None:
        server = _hosted_server()

        tools = {tool.name: tool for tool in await server.list_tools()}
        description = tools["index_video"].description or ""

        assert "pipeline" in description.lower()
        assert "features" in description

    async def test_hosted_index_video_polls_until_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.plan_calls = 0
                self.triggered: list[str] = []

            async def close(self) -> None:
                pass

            async def get_video(self, video_id: str):
                return SimpleNamespace(video_id=video_id, status="uploaded", features={})

            async def plan_features(self, video_id: str, desired: list[str]):
                self.plan_calls += 1
                if self.plan_calls == 1:
                    return SimpleNamespace(
                        actionable=["transcript"],
                        statuses={
                            "transcript": "none",
                            "captions": "none",
                            "embeddings": "none",
                        },
                    )
                return SimpleNamespace(
                    actionable=[],
                    statuses={
                        "transcript": "ready",
                        "captions": "ready",
                        "embeddings": "ready",
                    },
                )

            async def trigger_feature(
                self,
                video_id: str,
                feature: str,
                force: bool = False,
                body: dict | None = None,
            ):
                self.triggered.append(str(feature))
                return SimpleNamespace(
                    video_id=video_id, feature=str(feature), status="processing"
                )

        fake_client = FakeClient()

        from reka_mcp import server as server_module

        monkeypatch.setattr(
            server_module,
            "RekaClient",
            lambda api_url, api_key: fake_client,
        )
        server = _hosted_server()

        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            result = await server.call_tool(
                "index_video",
                {"video_id": "vid-hosted", "pipeline": "search_only"},
            )

        body = json.loads(tool_result_text(result))
        assert body["video_id"] == "vid-hosted"
        assert body["status"] == "ready"
        assert fake_client.plan_calls == 2
        assert "transcript" in fake_client.triggered
