# ABOUTME: MCP resources for read-only reference data (feature catalog, video metadata).
# ABOUTME: Resources are presented differently from tools in client UIs and can be subscribed to.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient


WORKFLOW_GUIDE = """\
# Reka Vision — Tool Usage Guide

Choose the right tools for your question instead of sending everything \
to ask_video. ask_video works best on short, focused segments.

## Counting or locating objects/people
1. get_objects (filter with object_type, e.g. "person") → count distinct \
tracking_ids
2. Optionally, ask_video with start/end on ambiguous segments to verify

## Finding when something happens
1. search_videos with a natural-language query → get timestamped results
2. ask_video with start/end from those timestamps → deeper visual analysis

## Understanding what was said
1. get_transcript (format="text") → read the actual spoken words
2. search_videos → find when a specific topic is discussed

## General video overview
1. summarize_video → metadata, feature status, transcript preview, \
scene count, detected object types
2. get_scenes → scene boundaries with timestamps
3. ask_video on individual scenes for detailed analysis

## Cross-video comparison
1. search_videos across videos → find comparable moments
2. ask_video with videos list, each with start/end → compare \
specific segments side by side

## Pipeline selection
- search_only: enables search_videos (transcription + captions + embeddings)
- qa_only: enables ask_video (transcription + captions)
- full: enables everything including get_objects (adds object detection)

## When to use ask_video directly
Only when you already have a specific time range, or the question is about \
a very short video (under ~2 minutes). For longer videos, always narrow \
first with search_videos, get_objects, or get_scenes.
"""


def register_resources(server: FastMCP, client: RekaClient) -> None:
    @server.resource(
        "reka://docs/guide",
        name="workflow_guide",
        description=(
            "Recommended workflows for common question types. Read this "
            "before calling ask_video to choose the best tool sequence."
        ),
        mime_type="text/plain",
    )
    async def workflow_guide() -> str:
        return WORKFLOW_GUIDE

    @server.resource(
        "reka://features",
        name="feature_catalog",
        description=(
            "Available video analysis features with their dependencies and descriptions."
        ),
        mime_type="application/json",
    )
    async def feature_catalog() -> str:
        catalog = await client.get_feature_catalog()
        return json.dumps(catalog)

    @server.resource(
        "reka://videos/{video_id}",
        name="video",
        description="Video metadata, upload status, and per-feature indexing status.",
        mime_type="application/json",
    )
    async def video(video_id: str) -> str:
        result = await client.get_video(video_id)
        return json.dumps(result.model_dump())
