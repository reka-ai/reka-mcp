# ABOUTME: MCP tool for semantic search across indexed videos.
# ABOUTME: Wraps RekaClient.search_videos and returns ranked results.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from reka_mcp.tools import READ_ONLY, logged, with_request_context

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient


def register_search_tools(server: FastMCP, client: RekaClient) -> None:
    @server.tool(
        name="search_videos",
        description=(
            "Find WHEN and WHERE something happens across your videos. "
            "Returns timestamped results ranked by relevance — use these "
            "timestamps as start/end in ask_video for focused analysis.\n\n"
            "This is the recommended first step for most questions. Instead "
            "of asking ask_video about the entire video, search first to "
            "narrow down the relevant moments.\n\n"
            "Requires search_only or full pipeline."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def search_videos(
        query: str,
        video_ids: list[str] | None = None,
        group_id: str | None = None,
        max_results: int = 10,
        rationale: str | None = None,
    ) -> str:
        results = await client.search_videos(
            query,
            video_ids=video_ids,
            group_id=group_id,
            max_results=max_results,
        )
        return json.dumps(
            {
                "results": results,
                "hint": (
                    "Pass these timestamps as start/end to ask_video for "
                    "deeper visual analysis, or to segment_video to detect "
                    "and locate specific objects in those moments."
                ),
            }
        )
