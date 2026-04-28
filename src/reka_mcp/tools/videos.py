# ABOUTME: MCP tools for video CRUD operations (upload, list, get, delete).
# ABOUTME: Each tool wraps a RekaClient call and returns JSON for the LLM.

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from reka_mcp.tools import DESTRUCTIVE, READ_ONLY, logged

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient


def register_video_tools(server: FastMCP, client: RekaClient) -> None:
    @server.tool(
        name="upload_video",
        description=(
            "Upload a video from a URL or local file path. Returns a video_id. "
            "Provide exactly one of video_url or file_path. For file uploads, "
            "name is required. The upload runs asynchronously — poll get_video "
            "until status is 'uploaded', then call index_video to enable search "
            "and analysis."
        ),
        annotations=ToolAnnotations(),
    )
    @logged
    async def upload_video(
        video_url: str | None = None,
        file_path: str | None = None,
        name: str | None = None,
        description: str | None = None,
        group_id: str | None = None,
        rationale: str | None = None,
    ) -> str:
        if bool(video_url) == bool(file_path):
            raise ToolError("Exactly one of 'video_url' or 'file_path' must be provided.")

        if file_path:
            path = Path(file_path)
            if not path.is_file():
                raise ToolError(f"File not found: {file_path}")
            if not name:
                raise ToolError("'name' is required for file uploads.")
            content = await asyncio.to_thread(path.read_bytes)
            result = await client.upload_file(
                file_content=content,
                filename=path.name,
                name=name,
                description=description,
                group_id=group_id,
            )
        else:
            result = await client.upload_video(
                video_url=video_url,  # type: ignore[arg-type]
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
    @logged
    async def get_video(video_id: str, rationale: str | None = None) -> str:
        result = await client.get_video(video_id)
        return json.dumps(result.model_dump())

    @server.tool(
        name="delete_video",
        description=(
            "Permanently delete a video and all its indexed data (transcript, "
            "captions, embeddings, etc.). This cannot be undone."
        ),
        annotations=DESTRUCTIVE,
    )
    @logged
    async def delete_video(video_id: str, rationale: str | None = None) -> str:
        result = await client.delete_video(video_id)
        return json.dumps(result.model_dump())
