# ABOUTME: MCP tools for video group management (create, list, delete).
# ABOUTME: Each operation is a separate tool with a clear schema.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations

from reka_mcp.tools import DESTRUCTIVE, READ_ONLY, logged, with_request_context

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient


def register_group_tools(server: FastMCP, client: RekaClient) -> None:
    @server.tool(
        name="create_group",
        description=(
            "Create a new video group. Groups organize videos into "
            "collections. Returns the new group's ID and name."
        ),
        annotations=ToolAnnotations(),
    )
    @with_request_context
    @logged
    async def create_group(name: str, rationale: str | None = None) -> str:
        result = await client.create_group(name)
        return json.dumps(result.model_dump())

    @server.tool(
        name="list_groups",
        description=(
            "List all video groups. Use list_videos with a group_id to see "
            "videos in a specific group."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def list_groups(rationale: str | None = None) -> str:
        groups = await client.list_groups()
        return json.dumps([g.model_dump() for g in groups])

    @server.tool(
        name="delete_group",
        description=(
            "Delete a video group. Videos in the group are not deleted — "
            "they are simply removed from the group."
        ),
        annotations=DESTRUCTIVE,
    )
    @with_request_context
    @logged
    async def delete_group(group_id: str, rationale: str | None = None) -> str:
        result = await client.delete_group(group_id)
        return json.dumps(result.model_dump())
