# ABOUTME: Configuration loaded from environment variables.
# ABOUTME: Validates required settings and provides defaults for optional ones.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Config:
    api_url: str
    api_key: str
    index_timeout: int
    poll_interval: int
    transport: Literal["stdio", "http"]
    http_port: int
    auth_token: str | None


def load_config() -> Config:
    api_key = os.environ.get("REKA_VISION_API_KEY", "")
    if not api_key:
        raise ValueError(
            "REKA_VISION_API_KEY environment variable is required. "
            "Get your API key from https://platform.reka.ai"
        )

    transport = os.environ.get("REKA_MCP_TRANSPORT", "stdio")
    if transport not in ("stdio", "http"):
        raise ValueError(f"REKA_MCP_TRANSPORT must be 'stdio' or 'http', got '{transport}'")

    return Config(
        api_url=os.environ.get("REKA_VISION_API_URL", "https://vision-api.reka.ai"),
        api_key=api_key,
        index_timeout=int(os.environ.get("REKA_MCP_INDEX_TIMEOUT", "600")),
        poll_interval=int(os.environ.get("REKA_MCP_POLL_INTERVAL", "5")),
        transport=transport,  # type: ignore[arg-type]
        http_port=int(os.environ.get("REKA_MCP_HTTP_PORT", "8080")),
        auth_token=os.environ.get("REKA_MCP_AUTH_TOKEN") or None,
    )
