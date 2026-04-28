# ABOUTME: Tests for HTTP transport authentication.
# ABOUTME: Covers token verifier, config loading, and server wiring.

from __future__ import annotations

import pytest

from reka_mcp.auth import StaticTokenVerifier
from reka_mcp.config import load_config
from reka_mcp.server import create_server


class TestStaticTokenVerifier:
    async def test_valid_token_returns_access_token(self) -> None:
        verifier = StaticTokenVerifier("secret-123")
        result = await verifier.verify_token("secret-123")
        assert result is not None
        assert result.token == "secret-123"

    async def test_invalid_token_returns_none(self) -> None:
        verifier = StaticTokenVerifier("secret-123")
        result = await verifier.verify_token("wrong-token")
        assert result is None

    async def test_empty_token_returns_none(self) -> None:
        verifier = StaticTokenVerifier("secret-123")
        result = await verifier.verify_token("")
        assert result is None


class TestAuthConfig:
    def test_auth_token_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REKA_VISION_API_KEY", "test-key")
        monkeypatch.delenv("REKA_MCP_AUTH_TOKEN", raising=False)
        config = load_config()
        assert config.auth_token is None

    def test_auth_token_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REKA_VISION_API_KEY", "test-key")
        monkeypatch.setenv("REKA_MCP_AUTH_TOKEN", "my-secret")
        config = load_config()
        assert config.auth_token == "my-secret"


class TestServerAuthWiring:
    async def test_server_without_auth_token_has_no_verifier(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        assert server._token_verifier is None

    async def test_server_with_auth_token_has_verifier(self) -> None:
        server = create_server(
            api_url="http://localhost:8000",
            api_key="test-key",
            auth_token="my-secret",
        )
        assert server._token_verifier is not None

    async def test_server_with_auth_still_registers_tools(self) -> None:
        server = create_server(
            api_url="http://localhost:8000",
            api_key="test-key",
            auth_token="my-secret",
        )
        tools = await server.list_tools()
        assert len(tools) == 16
