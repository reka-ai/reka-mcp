# ABOUTME: MCP tool implementations for Reka Vision Agent.
# ABOUTME: Each module registers tools on the MCP server instance.

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.lowlevel.server import request_ctx
from mcp.types import ToolAnnotations

from reka_mcp.client import RekaAPIError, mcp_session_id_var, reka_api_key_var

logger = logging.getLogger("reka_mcp.tools")

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True, idempotentHint=True)


def _get_header(headers: object, name: str) -> str | None:
    if hasattr(headers, "get"):
        value = headers.get(name)
        if value is not None:
            return str(value)
    if isinstance(headers, dict):
        target = name.lower()
        for key, value in headers.items():
            if key.lower() == target:
                return str(value)
    return None


def _extract_mcp_session_id() -> str | None:
    try:
        ctx = request_ctx.get()
    except LookupError:
        return None
    req = ctx.request
    if req is not None and hasattr(req, "headers"):
        session_id: str | None = req.headers.get("mcp-session-id")
        return session_id
    return None


def _extract_reka_api_key() -> str | None:
    try:
        ctx = request_ctx.get()
    except LookupError:
        return None
    req = ctx.request
    if req is not None and hasattr(req, "headers"):
        return _get_header(req.headers, "x-reka-api-key")
    return None


def _redact_log_arg(name: str, value: object) -> str:
    if any(sensitive in name.lower() for sensitive in ("api_key", "token", "password", "secret")):
        return "[REDACTED]"
    return repr(value)


def with_request_context[**P](fn: Callable[P, Awaitable[str]]) -> Callable[P, Awaitable[str]]:
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        session_token = mcp_session_id_var.set(_extract_mcp_session_id())
        api_key = _extract_reka_api_key()
        api_key_token = reka_api_key_var.set(api_key)
        if api_key:
            logger.debug("X-Reka-API-Key=[REDACTED]")
        try:
            return await fn(*args, **kwargs)
        finally:
            mcp_session_id_var.reset(session_token)
            reka_api_key_var.reset(api_key_token)

    return wrapper


def logged[**P](fn: Callable[P, Awaitable[str]]) -> Callable[P, Awaitable[str]]:
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        args_str = ", ".join(
            f"{k}={_redact_log_arg(k, v)}" for k, v in kwargs.items() if v is not None
        )
        logger.info("%s(%s)", fn.__name__, args_str)
        t0 = time.monotonic()
        try:
            result = await fn(*args, **kwargs)
        except RekaAPIError as e:
            logger.warning(
                "%s failed (%.0fms): %s", fn.__name__, (time.monotonic() - t0) * 1000, e
            )
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.warning(
                "%s failed (%.0fms): %s", fn.__name__, (time.monotonic() - t0) * 1000, e
            )
            raise
        logger.debug("%s ok (%.0fms)", fn.__name__, (time.monotonic() - t0) * 1000)
        return result

    return wrapper
