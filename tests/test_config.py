# ABOUTME: Tests for config module — env var loading and validation.
# ABOUTME: Covers defaults, required fields, and custom values.


import pytest

from reka_mcp.config import load_config


class TestLoadConfig:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REKA_VISION_API_KEY", raising=False)
        with pytest.raises(ValueError, match="REKA_VISION_API_KEY"):
            load_config()

    def test_defaults_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REKA_VISION_API_KEY", "test-key-123")
        monkeypatch.delenv("REKA_VISION_API_URL", raising=False)
        monkeypatch.delenv("REKA_MCP_INDEX_TIMEOUT", raising=False)
        monkeypatch.delenv("REKA_MCP_POLL_INTERVAL", raising=False)
        monkeypatch.delenv("REKA_MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("REKA_MCP_HTTP_PORT", raising=False)
        monkeypatch.delenv("REKA_MCP_AUTH_TOKEN", raising=False)

        config = load_config()

        assert config.api_key == "test-key-123"
        assert config.api_url == "https://vision-agent.api.reka.ai"
        assert config.index_timeout == 600
        assert config.poll_interval == 5
        assert config.transport == "stdio"
        assert config.http_port == 8080
        assert config.auth_token is None

    def test_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REKA_VISION_API_KEY", "custom-key")
        monkeypatch.setenv("REKA_VISION_API_URL", "http://localhost:9000")
        monkeypatch.setenv("REKA_MCP_INDEX_TIMEOUT", "300")
        monkeypatch.setenv("REKA_MCP_POLL_INTERVAL", "10")
        monkeypatch.setenv("REKA_MCP_TRANSPORT", "http")
        monkeypatch.setenv("REKA_MCP_HTTP_PORT", "9090")

        config = load_config()

        assert config.api_key == "custom-key"
        assert config.api_url == "http://localhost:9000"
        assert config.index_timeout == 300
        assert config.poll_interval == 10
        assert config.transport == "http"
        assert config.http_port == 9090

    def test_error_message_includes_help_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REKA_VISION_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"platform\.reka\.ai"):
            load_config()
