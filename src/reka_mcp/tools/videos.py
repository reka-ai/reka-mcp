# ABOUTME: MCP tools for video CRUD operations (upload, list, get, delete).
# ABOUTME: Each tool wraps a RekaClient call and returns JSON for the LLM.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from reka_mcp.tools import DESTRUCTIVE, READ_ONLY, logged, with_request_context

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient


def register_video_tools(server: FastMCP, client: RekaClient) -> None:
    @server.tool(
        name="upload_video",
        description=(
            "Upload a video from a URL. Returns a video_id. Local file paths "
            "are not accepted; upload files outside the MCP server and pass a "
            "reachable video_url. The upload runs asynchronously — poll "
            "get_video until status is 'uploaded', then call index_video to "
            "enable search and analysis."
        ),
        annotations=ToolAnnotations(),
    )
    @with_request_context
    @logged
    async def upload_video(
        video_url: str,
        name: str | None = None,
        description: str | None = None,
        group_id: str | None = None,
        rationale: str | None = None,
    ) -> str:
        if not video_url:
            raise ToolError(
                "'video_url' is required. Local file paths are not accepted; "
                "upload files outside the MCP server and pass a reachable URL."
            )

        result = await client.upload_video(
            video_url=video_url,
            name=name,
            description=description,
            group_id=group_id,
        )

        body = result.model_dump()
        body["hint"] = (
            "Poll get_video until status is 'uploaded', then call "
            "index_video to enable search and analysis."
        )
        return json.dumps(body)

    @server.tool(
        name="list_videos",
        description=(
            "List all videos in your account, or filter to a specific group "
            "by passing group_id. Shows upload status and which features have "
            "been indexed for each video."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def list_videos(group_id: str | None = None, rationale: str | None = None) -> str:
        if group_id:
            videos = await client.list_group_videos(group_id)
        else:
            videos = await client.list_videos()
        return json.dumps([v.model_dump() for v in videos])

    @server.tool(
        name="get_video",
        description=(
            "Get detailed information about a video including upload status, "
            "metadata (duration, resolution, fps), and per-feature indexing "
            "status. Use this to check if upload or indexing is complete."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def get_video(video_id: str, rationale: str | None = None) -> str:
        result = await client.get_video(video_id)
        return json.dumps(result.model_dump())

    @server.tool(
        name="update_video",
        description=(
            "Update a video's display name, title, description, or move it to a "
            "different group. At least one field must be provided. To remove a "
            "video from its group, pass group_id as null."
        ),
        annotations=ToolAnnotations(idempotentHint=True),
    )
    @with_request_context
    @logged
    async def update_video(
        video_id: str,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        group_id: str | None = None,
        move_group: bool = False,
        rationale: str | None = None,
    ) -> str:
        result = await client.update_video(
            video_id,
            name=name,
            title=title,
            description=description,
            group_id=group_id,
            group_id_provided=move_group or group_id is not None,
        )
        return json.dumps(result.model_dump())

    @server.tool(
        name="delete_video",
        description=(
            "Permanently delete a video and all its indexed data (transcript, "
            "captions, embeddings, etc.). This cannot be undone."
        ),
        annotations=DESTRUCTIVE,
    )
    @with_request_context
    @logged
    async def delete_video(video_id: str, rationale: str | None = None) -> str:
        result = await client.delete_video(video_id)
        return json.dumps(result.model_dump())
