# ABOUTME: Configuration loaded from environment variables.
# ABOUTME: Validates required settings and provides defaults for optional ones.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

RuntimeMode = Literal["local", "hosted"]


@dataclass(frozen=True)
class Config:
    mode: RuntimeMode
    api_url: str
    api_key: str | None
    index_timeout: int
    poll_interval: int
    transport: Literal["stdio", "http"]
    http_host: str
    http_port: int
    http_path: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    auth_token: str | None


def _csv_env(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _path(value: str) -> str:
    return value if value.startswith("/") else f"/{value}"


def load_config() -> Config:
    mode = os.environ.get("REKA_MCP_MODE", "local")
    if mode not in ("local", "hosted"):
        raise ValueError(f"REKA_MCP_MODE must be 'local' or 'hosted', got '{mode}'")

    api_key = os.environ.get("REKA_VISION_API_KEY") or None
    if mode == "local" and not api_key:
        raise ValueError(
            "REKA_VISION_API_KEY environment variable is required. "
            "Get your API key from https://platform.reka.ai"
        )

    transport = os.environ.get("REKA_MCP_TRANSPORT", "http" if mode == "hosted" else "stdio")
    if transport not in ("stdio", "http"):
        raise ValueError(f"REKA_MCP_TRANSPORT must be 'stdio' or 'http', got '{transport}'")

    hosted_allowed_hosts = ("mcp.reka.ai", "staging.mcp.reka.ai")
    hosted_allowed_origins = ("https://mcp.reka.ai", "https://staging.mcp.reka.ai")

    return Config(
        mode=mode,  # type: ignore[arg-type]
        api_url=os.environ.get("REKA_VISION_API_URL", "https://vision-agent.api.reka.ai"),
        api_key=api_key,
        index_timeout=int(os.environ.get("REKA_MCP_INDEX_TIMEOUT", "600")),
        poll_interval=int(os.environ.get("REKA_MCP_POLL_INTERVAL", "5")),
        transport=transport,  # type: ignore[arg-type]
        http_host=os.environ.get(
            "REKA_MCP_HTTP_HOST", "0.0.0.0" if mode == "hosted" else "127.0.0.1"
        ),
        http_port=int(
            (os.environ.get("PORT") if mode == "hosted" else None)
            or os.environ.get("REKA_MCP_HTTP_PORT", "8080")
        ),
        http_path=_path(os.environ.get("REKA_MCP_HTTP_PATH", "/mcp")),
        allowed_hosts=_csv_env(
            "REKA_MCP_ALLOWED_HOSTS", hosted_allowed_hosts if mode == "hosted" else ()
        ),
        allowed_origins=_csv_env(
            "REKA_MCP_ALLOWED_ORIGINS", hosted_allowed_origins if mode == "hosted" else ()
        ),
        auth_token=os.environ.get("REKA_MCP_AUTH_TOKEN") or None,
    )
