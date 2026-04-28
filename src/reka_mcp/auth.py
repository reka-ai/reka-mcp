# ABOUTME: Bearer token authentication for the HTTP transport.
# ABOUTME: Validates tokens against a pre-shared secret via the MCP TokenVerifier protocol.

from __future__ import annotations

import hmac

from mcp.server.auth.provider import AccessToken


class StaticTokenVerifier:
    """Verifies bearer tokens against a pre-shared secret."""

    def __init__(self, expected_token: str) -> None:
        self._expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or not hmac.compare_digest(token, self._expected_token):
            return None
        return AccessToken(
            token=token,
            client_id="static",
            scopes=[],
        )
