# ABOUTME: MCP tool for indexing videos through a pipeline of AI features.
# ABOUTME: Orchestrates feature DAG client-side using plan + trigger endpoints.

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from reka_mcp.pipelines import PIPELINE_FEATURES, Feature, FeatureStatus, Pipeline
from reka_mcp.tools import logged

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient


def register_indexing_tools(
    server: FastMCP,
    client: RekaClient,
    index_timeout: int = 600,
    poll_interval: int = 5,
) -> None:
    @server.tool(
        name="index_video",
        description=(
            "Index a video for search, QA, or full analysis. Processes the video "
            "through a pipeline of AI features. This may take 2-10 minutes "
            "depending on video length.\n\n"
            "Pipelines:\n"
            "- search_only: transcription + captions + embeddings (enables search_videos)\n"
            "- qa_only: transcription + captions (enables ask_video)\n"
            "- full: all features including object detection (enables all tools)\n\n"
            "Prerequisites: video must be in 'uploaded' status (upload complete). "
            "Use get_video to check status before calling this tool."
        ),
        annotations=ToolAnnotations(idempotentHint=True),
    )
    @logged
    async def index_video(
        video_id: str,
        pipeline: Pipeline = "search_only",
        rationale: str | None = None,
    ) -> str:
        result = await _run_indexing(client, video_id, pipeline, index_timeout, poll_interval)
        return json.dumps(result)


async def _run_indexing(
    client: RekaClient,
    video_id: str,
    pipeline: Pipeline,
    index_timeout: int,
    poll_interval: int,
) -> dict[str, str | dict[str, str]]:
    features = PIPELINE_FEATURES[pipeline]

    video = await client.get_video(video_id)
    if video.status != "uploaded":
        raise ToolError(
            f"Video is in '{video.status}' status. Wait for 'uploaded' status before indexing."
        )

    deadline = time.monotonic() + index_timeout
    desired = sorted(features)

    try:
        while time.monotonic() < deadline:
            plan = await client.plan_features(video_id, desired)

            feature_statuses: dict[str, str] = {
                str(f): plan.statuses.get(f, FeatureStatus.NONE) for f in desired
            }

            # Check our desired features directly — the server's `done` flag may
            # include co-produced features (e.g. scenes) we didn't request.
            if all(s == FeatureStatus.READY for s in feature_statuses.values()):
                return {
                    "video_id": video_id,
                    "status": "ready",
                    "features": feature_statuses,
                    "hint": (
                        "Video ready. Use search_videos to find moments, "
                        "segment_video to detect specific objects, or "
                        "ask_video with start/end for focused visual analysis."
                    ),
                }

            failed = [f for f, s in feature_statuses.items() if s == FeatureStatus.FAILED]
            if failed:
                return {
                    "video_id": video_id,
                    "status": "failed",
                    "features": feature_statuses,
                    "error": f"Features failed: {', '.join(failed)}",
                }

            if plan.actionable:
                await asyncio.gather(
                    *(
                        client.trigger_feature(
                            video_id,
                            Feature(feat),
                            body=_transcript_body(pipeline)
                            if feat == Feature.TRANSCRIPT
                            else None,
                        )
                        for feat in plan.actionable
                    )
                )

            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        logger.info("indexing cancelled for video %s", video_id)
        raise

    video = await client.get_video(video_id)
    return {
        "video_id": video_id,
        "status": "timeout",
        "features": video.features or {},
        "message": f"Indexing timed out after {index_timeout}s. Use get_video to check progress.",
    }


def _transcript_body(pipeline: Pipeline) -> dict[str, dict[str, bool]] | None:
    if pipeline == "full":
        return {"chunking_config": {"use_scene_detection": True}}
    return None
