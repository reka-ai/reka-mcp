# ABOUTME: Docker and container boot tests for hosted MCP deployment.
# ABOUTME: Validates Dockerfile structure and hosted-mode container entry point.

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = PROJECT_ROOT / "Dockerfile"


class TestDockerfileExists:
    def test_dockerfile_present_at_project_root(self) -> None:
        assert DOCKERFILE.is_file(), "Dockerfile must exist at project root"


class TestDockerfileStructure:
    @pytest.fixture(autouse=True)
    def _load_dockerfile(self) -> None:
        self.lines = DOCKERFILE.read_text().splitlines()

    def test_base_image_is_python_312(self) -> None:
        from_lines = [line for line in self.lines if line.strip().startswith("FROM")]
        assert from_lines, "Dockerfile must have a FROM instruction"
        assert "python" in from_lines[0].lower()
        assert "3.12" in from_lines[0]

    def test_exposes_port_80(self) -> None:
        expose_lines = [line for line in self.lines if line.strip().startswith("EXPOSE")]
        assert expose_lines, "Dockerfile must EXPOSE a port"
        assert any("80" in line for line in expose_lines)

    def test_sets_hosted_mode_env(self) -> None:
        env_lines = [line for line in self.lines if line.strip().startswith("ENV")]
        env_text = "\n".join(env_lines)
        assert "REKA_MCP_MODE" in env_text, "Dockerfile must set REKA_MCP_MODE"
        assert "hosted" in env_text

    def test_sets_port_env(self) -> None:
        env_lines = [line for line in self.lines if line.strip().startswith("ENV")]
        env_text = "\n".join(env_lines)
        assert "PORT" in env_text, "Dockerfile must set PORT"

    def test_entrypoint_runs_reka_mcp(self) -> None:
        cmd_lines = [line for line in self.lines if line.strip().startswith(("CMD", "ENTRYPOINT"))]
        assert cmd_lines, "Dockerfile must have a CMD or ENTRYPOINT"
        cmd_text = "\n".join(cmd_lines)
        assert "reka-mcp" in cmd_text or "reka_mcp" in cmd_text


class TestHostedBootProcess:
    """Verify the server boots correctly under container-like hosted env."""

    def test_hosted_boot_creates_asgi_app_with_health_and_mcp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate container boot: REKA_MCP_MODE=hosted, PORT=80, no API key."""
        monkeypatch.setenv("REKA_MCP_MODE", "hosted")
        monkeypatch.setenv("PORT", "80")
        monkeypatch.setenv("REKA_MCP_HTTP_PATH", "/mcp")
        monkeypatch.delenv("REKA_VISION_API_KEY", raising=False)
        monkeypatch.delenv("REKA_MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("REKA_MCP_HTTP_HOST", raising=False)
        monkeypatch.delenv("REKA_MCP_ALLOWED_HOSTS", raising=False)
        monkeypatch.delenv("REKA_MCP_ALLOWED_ORIGINS", raising=False)

        from reka_mcp.config import load_config
        from reka_mcp.server import create_server

        config = load_config()
        server = create_server(
            api_url=config.api_url,
            api_key=config.api_key,
            mode=config.mode,
            http_host=config.http_host,
            http_port=config.http_port,
            http_path=config.http_path,
            allowed_hosts=config.allowed_hosts,
            allowed_origins=config.allowed_origins,
        )

        assert server.settings.host == "0.0.0.0"
        assert server.settings.port == 80
        assert server.settings.streamable_http_path == "/mcp"

        app = server.streamable_http_app()
        route_paths = {getattr(r, "path", None) for r in app.routes}
        assert "/health" in route_paths
        assert "/mcp" in route_paths

    def test_hosted_boot_health_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Container health check must succeed immediately after boot."""
        from starlette.testclient import TestClient

        from reka_mcp.server import create_server

        server = create_server(
            api_url="http://localhost:8000",
            api_key=None,
            mode="hosted",
            http_host="0.0.0.0",
            http_port=80,
            http_path="/mcp",
            allowed_hosts=("mcp.reka.ai",),
            allowed_origins=("https://mcp.reka.ai",),
        )
        app = server.streamable_http_app()

        with TestClient(app) as http:
            resp = http.get("/health", headers={"host": "mcp.reka.ai"})

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_hosted_boot_mcp_path_accepts_post(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Container /mcp endpoint must accept POST (Streamable HTTP transport)."""
        from starlette.testclient import TestClient

        from reka_mcp.server import create_server

        server = create_server(
            api_url="http://localhost:8000",
            api_key=None,
            mode="hosted",
            http_host="0.0.0.0",
            http_port=80,
            http_path="/mcp",
            allowed_hosts=("mcp.reka.ai",),
            allowed_origins=("https://mcp.reka.ai",),
        )
        app = server.streamable_http_app()

        with TestClient(app) as http:
            resp = http.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0.1"},
                    },
                },
                headers={
                    "host": "mcp.reka.ai",
                    "content-type": "application/json",
                    "accept": "application/json, text/event-stream",
                },
            )

        assert resp.status_code == 200
