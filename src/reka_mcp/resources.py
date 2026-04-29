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

## Detecting or locating objects
Pick the path that matches what you know:
- If you don't know WHEN: search_videos → take top timestamps → \
segment_video with text prompts on those windows
- If you want to scan the whole video systematically: get_scenes → \
call segment_video once per scene (scenes are typically <15s, which fits \
segment_video's max range cleanly)
- If you already have a time range: segment_video directly with prompts \
(e.g. "person", "red car", "safety helmet")

segment_video returns per-frame detections with bounding boxes and \
confidence scores. Optionally follow up with ask_video on the same range \
for contextual understanding.

## Finding when something happens
1. search_videos with a natural-language query → get timestamped results
2. ask_video with start/end from those timestamps → deeper visual analysis

## Understanding what was said
1. get_transcript (format="text") → read the actual spoken words
2. search_videos → find when a specific topic is discussed

## General video overview
1. summarize_video → metadata, feature status, transcript preview, scene count
2. get_scenes → scene boundaries with timestamps
3. Per-scene drill-down:
   - ask_video on individual scenes for detailed analysis
   - segment_video on individual scenes to detect specific objects \
(scenes typically fit in segment_video's 15s max range)

## Cross-video comparison
1. search_videos across videos → find comparable moments
2. ask_video with videos list, each with start/end → compare \
specific segments side by side

## Pipeline selection
- search_only: enables search_videos (transcription + captions + embeddings)
- qa_only: enables ask_video (transcription + captions)
- full: enables everything (adds object indexing on top of search + QA)

## When to use ask_video directly
Only when you already have a specific time range, or the question is about \
a very short video (under ~2 minutes). For longer videos, always narrow \
first with search_videos, segment_video, or get_scenes.

## Improving detail on long segments
ask_video accuracy improves with narrower time ranges. If results on a wide \
segment seem vague or miss details, break it down:
1. Scene-by-scene: get_scenes → call ask_video or segment_video once per \
scene
2. Second-by-second: call ask_video with 1-second windows across the range \
of interest

These produce significantly better results but cost more API calls — use \
them when the default answer lacks the detail you need, not as a first pass.
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
