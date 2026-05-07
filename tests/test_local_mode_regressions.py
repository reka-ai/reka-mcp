# ABOUTME: Local-mode regression tests that must stay stable while hosted MCP is added.
# ABOUTME: Covers stdio entrypoint/config, static API keys, file upload, blocking indexing, and tools.

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from mcp.server.fastmcp import FastMCP

from reka_mcp.client import RekaClient
from reka_mcp.config import load_config
from reka_mcp.server import create_server
from reka_mcp.tools.indexing import register_indexing_tools
from reka_mcp.tools.videos import register_video_tools
from tests.conftest import BASE_URL, mock_client, tool_result_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_uvx_reka_mcp_console_script_points_to_stdio_entrypoint() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    server_json = json.loads((PROJECT_ROOT / "server.json").read_text())

    assert pyproject["project"]["scripts"]["reka-mcp"] == "reka_mcp.server:main"
    assert server_json["packages"][0]["registryType"] == "pypi"
    assert server_json["packages"][0]["identifier"] == "reka-mcp"
    assert server_json["packages"][0]["transport"]["type"] == "stdio"


def test_local_config_defaults_to_stdio_with_static_env_api_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("REKA_VISION_API_KEY", "local-static-key")
    monkeypatch.delenv("REKA_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("REKA_MCP_AUTH_TOKEN", raising=False)

    config = load_config()

    assert config.transport == "stdio"
    assert config.api_key == "local-static-key"
    assert config.auth_token is None


def test_main_runs_stdio_with_static_reka_api_key(monkeypatch) -> None:
    from reka_mcp import server as server_module

    created_with: dict[str, object] = {}

    class FakeServer:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(port=9999)
            self.run_transports: list[str] = []

        def run(self, *, transport: str) -> None:
            self.run_transports.append(transport)

    fake_server = FakeServer()

    def fake_create_server(**kwargs: object) -> FakeServer:
        created_with.update(kwargs)
        return fake_server

    monkeypatch.setenv("REKA_VISION_API_KEY", "local-static-key")
    monkeypatch.setenv("REKA_VISION_API_URL", "http://local-api.test")
    monkeypatch.setenv("REKA_MCP_INDEX_TIMEOUT", "321")
    monkeypatch.setenv("REKA_MCP_POLL_INTERVAL", "7")
    monkeypatch.delenv("REKA_MCP_MODE", raising=False)
    monkeypatch.delenv("REKA_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("REKA_MCP_HTTP_HOST", raising=False)
    monkeypatch.delenv("REKA_MCP_HTTP_PORT", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("REKA_MCP_HTTP_PATH", raising=False)
    monkeypatch.delenv("REKA_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("REKA_MCP_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("REKA_MCP_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(server_module, "create_server", fake_create_server)

    server_module.main()

    assert created_with == {
        "api_url": "http://local-api.test",
        "api_key": "local-static-key",
        "index_timeout": 321,
        "poll_interval": 7,
        "auth_token": None,
        "mode": "local",
        "http_host": "127.0.0.1",
        "http_port": 8080,
        "http_path": "/mcp",
        "allowed_hosts": (),
        "allowed_origins": (),
    }
    assert fake_server.run_transports == ["stdio"]
    assert fake_server.settings.port == 9999


async def test_local_server_registers_current_tools_and_file_upload_schema() -> None:
    server = create_server(api_url="http://localhost:8000", api_key="test-key")
    tools = {tool.name: tool for tool in await server.list_tools()}

    assert sorted(tools) == sorted(
        [
            "upload_video",
            "list_videos",
            "get_video",
            "update_video",
            "delete_video",
            "create_group",
            "list_groups",
            "delete_group",
            "index_video",
            "search_videos",
            "ask_video",
            "get_transcript",
            "get_captions",
            "get_scenes",
            "segment_video",
            "get_feature_catalog",
            "summarize_video",
        ]
    )

    upload_schema = tools["upload_video"].inputSchema["properties"]
    assert "video_url" in upload_schema
    assert "file_path" in upload_schema


async def test_local_upload_video_accepts_file_path_and_posts_file_multipart(tmp_path) -> None:
    client = RekaClient(api_url=BASE_URL, api_key="local-static-key")
    server = FastMCP("test-reka-vision")
    register_video_tools(server, client)

    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"local video bytes")
    captured: dict[str, bytes | str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["method"] = req.method
        captured["content_type"] = req.headers.get("content-type", "")
        captured["body"] = req.content
        captured["api_key"] = req.headers["x-api-key"]
        return httpx.Response(202, json={"video_id": "vid-file", "status": "uploading"})

    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=BASE_URL,
    )

    result = await server.call_tool(
        "upload_video",
        {"file_path": str(video_file), "name": "Local Clip"},
    )

    body = json.loads(tool_result_text(result))
    multipart_body = captured["body"]
    assert isinstance(multipart_body, bytes)
    assert body["video_id"] == "vid-file"
    assert captured["method"] == "POST"
    assert captured["api_key"] == "local-static-key"
    assert "multipart/form-data" in str(captured["content_type"])
    assert b"local video bytes" in multipart_body
    assert b'filename="clip.mp4"' in multipart_body
    assert b"video_name" in multipart_body
    assert b"Local Clip" in multipart_body


async def test_local_index_video_blocks_and_polls_until_ready(client: RekaClient) -> None:
    server = FastMCP("test-reka-vision")
    register_indexing_tools(server, client, index_timeout=600, poll_interval=5)

    plan_calls = 0
    triggered: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal plan_calls
        url = str(req.url)

        if url.endswith("/v2/videos/vid-local") and req.method == "GET":
            return httpx.Response(200, json={"video_id": "vid-local", "status": "uploaded"})

        if url.endswith("/v2/videos/vid-local/features/plan") and req.method == "POST":
            plan_calls += 1
            if plan_calls == 1:
                return httpx.Response(
                    200,
                    json={
                        "done": False,
                        "actionable": ["transcript", "captions", "embeddings"],
                        "blocked": [],
                        "statuses": {
                            "transcript": "none",
                            "captions": "none",
                            "embeddings": "none",
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "done": True,
                    "actionable": [],
                    "blocked": [],
                    "statuses": {
                        "transcript": "ready",
                        "captions": "ready",
                        "embeddings": "ready",
                    },
                },
            )

        for feature in ("transcript", "captions", "embeddings"):
            if url.endswith(f"/v2/videos/vid-local/features/{feature}") and req.method == "POST":
                triggered.append(feature)
                return httpx.Response(
                    202,
                    json={
                        "video_id": "vid-local",
                        "feature": feature,
                        "status": "processing",
                    },
                )

        return httpx.Response(404, json={"error": {"message": "not found"}})

    mock_client(client, handler)

    with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        result = await server.call_tool(
            "index_video",
            {"video_id": "vid-local", "pipeline": "search_only"},
        )

    body = json.loads(tool_result_text(result))
    assert body["status"] == "ready"
    assert plan_calls == 2
    assert set(triggered) == {"transcript", "captions", "embeddings"}
    sleep_mock.assert_awaited_once_with(5)
