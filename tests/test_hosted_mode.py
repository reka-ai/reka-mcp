# ABOUTME: Hosted MCP regression tests for Streamable HTTP and request-scoped auth.
# ABOUTME: Defines hosted behavior before implementing ECS-facing HTTP mode.

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from starlette.testclient import TestClient

from reka_mcp.client import RekaClient
from reka_mcp.config import load_config
from reka_mcp.server import create_server
from reka_mcp.tools.videos import register_video_tools
from tests.conftest import BASE_URL


@dataclass
class FakeRequest:
    headers: dict[str, str]


def _set_request_context(headers: dict[str, str]):
    from mcp.server.lowlevel.server import request_ctx
    from mcp.shared.context import RequestContext

    ctx = RequestContext(
        request_id="req-1",
        meta=None,
        session=MagicMock(),
        lifespan_context=None,
        request=FakeRequest(headers=headers),
    )
    return request_ctx, request_ctx.set(ctx)


def _hosted_video_server(
    handler,
) -> tuple[RekaClient, FastMCP]:
    client = RekaClient(api_url=BASE_URL, api_key=None)
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=BASE_URL,
    )
    server = FastMCP("test-hosted-reka-vision")
    register_video_tools(server, client)
    return client, server


class TestHostedConfig:
    def test_hosted_defaults_to_http_without_process_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REKA_MCP_MODE", "hosted")
        monkeypatch.delenv("REKA_VISION_API_KEY", raising=False)
        monkeypatch.delenv("REKA_MCP_TRANSPORT", raising=False)
        monkeypatch.setenv("PORT", "8181")
        monkeypatch.setenv("REKA_MCP_HTTP_PORT", "9090")
        monkeypatch.delenv("REKA_MCP_HTTP_HOST", raising=False)
        monkeypatch.delenv("REKA_MCP_HTTP_PATH", raising=False)
        monkeypatch.delenv("REKA_MCP_ALLOWED_HOSTS", raising=False)
        monkeypatch.delenv("REKA_MCP_ALLOWED_ORIGINS", raising=False)

        config = load_config()

        assert config.mode == "hosted"
        assert config.transport == "http"
        assert config.api_key is None
        assert config.http_host == "0.0.0.0"
        assert config.http_port == 8181
        assert config.http_path == "/mcp"
        assert "mcp.reka.ai" in config.allowed_hosts
        assert "staging.mcp.reka.ai" in config.allowed_hosts
        assert "https://mcp.reka.ai" in config.allowed_origins
        assert "https://staging.mcp.reka.ai" in config.allowed_origins

    def test_hosted_main_runs_streamable_http_with_hosted_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reka_mcp import server as server_module

        created_with: dict[str, object] = {}

        class FakeServer:
            def __init__(self) -> None:
                self.settings = SimpleNamespace(host="", port=0, streamable_http_path="")
                self.run_transports: list[str] = []

            def run(self, *, transport: str) -> None:
                self.run_transports.append(transport)

        fake_server = FakeServer()

        def fake_create_server(**kwargs: object) -> FakeServer:
            created_with.update(kwargs)
            return fake_server

        monkeypatch.setenv("REKA_MCP_MODE", "hosted")
        monkeypatch.delenv("REKA_VISION_API_KEY", raising=False)
        monkeypatch.delenv("REKA_MCP_TRANSPORT", raising=False)
        monkeypatch.setenv("REKA_VISION_API_URL", "http://hosted-api.test")
        monkeypatch.setenv("REKA_MCP_HTTP_HOST", "0.0.0.0")
        monkeypatch.setenv("PORT", "8088")
        monkeypatch.setenv("REKA_MCP_HTTP_PATH", "/mcp")
        monkeypatch.setenv("REKA_MCP_ALLOWED_HOSTS", "mcp.reka.ai,staging.mcp.reka.ai")
        monkeypatch.setenv(
            "REKA_MCP_ALLOWED_ORIGINS",
            "https://mcp.reka.ai,https://staging.mcp.reka.ai",
        )
        monkeypatch.setattr(server_module, "create_server", fake_create_server)

        server_module.main()

        assert created_with["mode"] == "hosted"
        assert created_with["api_url"] == "http://hosted-api.test"
        assert created_with["api_key"] is None
        assert created_with["http_host"] == "0.0.0.0"
        assert created_with["http_port"] == 8088
        assert created_with["http_path"] == "/mcp"
        assert created_with["allowed_hosts"] == ("mcp.reka.ai", "staging.mcp.reka.ai")
        assert created_with["allowed_origins"] == (
            "https://mcp.reka.ai",
            "https://staging.mcp.reka.ai",
        )
        assert fake_server.run_transports == ["streamable-http"]


class TestHostedHttpServer:
    def test_hosted_server_configures_streamable_http_and_health_route(self) -> None:
        server = create_server(
            api_url="http://localhost:8000",
            api_key=None,
            mode="hosted",
            http_host="0.0.0.0",
            http_port=8080,
            http_path="/mcp",
            allowed_hosts=("mcp.reka.ai", "staging.mcp.reka.ai"),
            allowed_origins=("https://mcp.reka.ai", "https://staging.mcp.reka.ai"),
        )

        assert server.settings.host == "0.0.0.0"
        assert server.settings.port == 8080
        assert server.settings.streamable_http_path == "/mcp"
        assert server.settings.transport_security is not None
        assert server.settings.transport_security.allowed_hosts == [
            "mcp.reka.ai",
            "staging.mcp.reka.ai",
        ]
        assert server.settings.transport_security.allowed_origins == [
            "https://mcp.reka.ai",
            "https://staging.mcp.reka.ai",
        ]

        app = server.streamable_http_app()
        route_paths = {getattr(route, "path", None) for route in app.routes}
        assert "/mcp" in route_paths
        assert "/health" in route_paths

        with TestClient(app) as http:
            response = http.get("/health", headers={"host": "mcp.reka.ai"})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestHostedRequestScopedAuth:
    async def test_x_reka_api_key_is_forwarded_as_reka_api_key(self) -> None:
        captured_keys: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured_keys.append(req.headers["x-api-key"])
            return httpx.Response(200, json={"results": []})

        _, server = _hosted_video_server(handler)
        request_ctx, token = _set_request_context({"x-reka-api-key": "rk-user-1"})
        try:
            await server.call_tool("list_videos", {})
        finally:
            request_ctx.reset(token)

        assert captured_keys == ["rk-user-1"]

    async def test_missing_x_reka_api_key_fails_before_api_call(self) -> None:
        called = False

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"results": []})

        _, server = _hosted_video_server(handler)
        request_ctx, token = _set_request_context({})
        try:
            with pytest.raises(ToolError, match="X-Reka-API-Key"):
                await server.call_tool("list_videos", {})
        finally:
            request_ctx.reset(token)

        assert called is False

    async def test_two_hosted_requests_keep_api_keys_isolated(self) -> None:
        captured_keys: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured_keys.append(req.headers["x-api-key"])
            return httpx.Response(200, json={"results": []})

        _, server = _hosted_video_server(handler)

        request_ctx, token = _set_request_context({"x-reka-api-key": "rk-user-a"})
        try:
            await server.call_tool("list_videos", {})
        finally:
            request_ctx.reset(token)

        request_ctx, token = _set_request_context({"x-reka-api-key": "rk-user-b"})
        try:
            await server.call_tool("list_videos", {})
        finally:
            request_ctx.reset(token)

        assert captured_keys == ["rk-user-a", "rk-user-b"]

    async def test_hosted_api_key_is_redacted_from_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "rk-secret-value"

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": []})

        _, server = _hosted_video_server(handler)
        caplog.set_level(logging.DEBUG, logger="reka_mcp")

        request_ctx, token = _set_request_context({"x-reka-api-key": secret})
        try:
            await server.call_tool("list_videos", {})
        finally:
            request_ctx.reset(token)

        assert secret not in caplog.text
        assert "X-Reka-API-Key=[REDACTED]" in caplog.text
