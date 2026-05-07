# ABOUTME: Shared test fixtures for mcp-server tests.
# ABOUTME: Provides helper to swap RekaClient transport for testing.

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from reka_mcp.client import RekaClient

BASE_URL = "http://test-api.local"


@pytest.fixture
def client() -> RekaClient:
    return RekaClient(api_url=BASE_URL, api_key="test-key")


def mock_client(
    client: RekaClient,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Replace the client's HTTP transport with a mock handler."""
    transport = httpx.MockTransport(handler)
    client._http = httpx.AsyncClient(
        transport=transport,
        base_url=BASE_URL,
    )


def tool_result_text(result: tuple) -> str:
    """Extract the text content from a FastMCP call_tool result."""
    return result[0][0].text
