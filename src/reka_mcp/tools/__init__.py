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

from reka_mcp.client import RekaAPIError, mcp_session_id_var

logger = logging.getLogger("reka_mcp.tools")

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True, idempotentHint=True)


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


def logged[**P](fn: Callable[P, Awaitable[str]]) -> Callable[P, Awaitable[str]]:
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        token = mcp_session_id_var.set(_extract_mcp_session_id())
        args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)
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
        finally:
            mcp_session_id_var.reset(token)
        logger.debug("%s ok (%.0fms)", fn.__name__, (time.monotonic() - t0) * 1000)
        return result

    return wrapper
