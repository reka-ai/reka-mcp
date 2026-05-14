# ABOUTME: MCP tool for indexing videos through a pipeline of AI features.
# ABOUTME: Orchestrates feature DAG client-side using plan + trigger endpoints.

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from reka_mcp.pipelines import PIPELINE_FEATURES, Feature, FeatureStatus, Pipeline
from reka_mcp.tools import logged, with_request_context

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient
    from reka_mcp.config import RuntimeMode


def register_indexing_tools(
    server: FastMCP,
    client: RekaClient,
    index_timeout: int = 600,
    poll_interval: int = 5,
    mode: RuntimeMode = "local",
) -> None:
    base_description = (
        "Index a video for search, QA, or full analysis. Processes the video "
        "through a pipeline of AI features. Typically takes 3-7 minutes; "
        "longer for long videos or the 'full' pipeline. Times out after "
        "10 minutes by default.\n\n"
        "Pipelines:\n"
        "- search_only: transcription + captions + embeddings (enables search_videos)\n"
        "- qa_only: transcription + captions (enables ask_video)\n"
        "- full: transcription + captions + embeddings (enables all tools)\n\n"
        "Scene detection is enabled by default and produces scene boundaries "
        "for get_scenes. Pass scene_detection=False to skip it.\n\n"
        "Prerequisites: if using video_id, the video must be in 'uploaded' status. "
        "Use get_video to check status before calling this tool."
    )

    if mode == "local":
        description = base_description + (
            "\n\nAccepts either video_id (for an already-uploaded video) or "
            "file_path (a local file to upload and index in one step). "
            "Provide exactly one."
        )

        @server.tool(
            name="index_video",
            description=description,
            annotations=ToolAnnotations(idempotentHint=True),
        )
        @with_request_context
        @logged
        async def index_video(
            video_id: str | None = None,
            file_path: str | None = None,
            pipeline: Pipeline = "search_only",
            scene_detection: bool = True,
            rationale: str | None = None,
        ) -> str:
            if video_id and file_path:
                raise ToolError("Provide video_id or file_path, not both.")
            if not video_id and not file_path:
                raise ToolError("Provide either video_id or file_path.")

            if file_path:
                video_id = await _upload_and_wait(client, file_path, poll_interval, index_timeout)

            assert video_id is not None

            result = await _run_indexing(
                client, video_id, pipeline, index_timeout, poll_interval, scene_detection
            )
            return json.dumps(result)
    else:

        @server.tool(
            name="index_video",
            description=base_description,
            annotations=ToolAnnotations(idempotentHint=True),
        )
        @with_request_context
        @logged
        async def index_video_hosted(
            video_id: str,
            pipeline: Pipeline = "search_only",
            scene_detection: bool = True,
            rationale: str | None = None,
        ) -> str:
            result = await _run_indexing(
                client, video_id, pipeline, index_timeout, poll_interval, scene_detection
            )
            return json.dumps(result)


async def _upload_and_wait(
    client: RekaClient,
    file_path: str,
    poll_interval: int,
    timeout: int,
) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise ToolError(f"File does not exist: {file_path}")

    upload_resp = await client.upload_video_file(file_path=file_path)
    video_id = upload_resp.video_id

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        video = await client.get_video(video_id)
        if video.status == "uploaded":
            return video_id
        if video.status not in ("uploading", "pending"):
            raise ToolError(f"Upload failed for video {video_id}: status is '{video.status}'.")
        await asyncio.sleep(poll_interval)

    raise ToolError(f"Upload timed out for video {video_id} after {timeout}s.")


async def _run_indexing(
    client: RekaClient,
    video_id: str,
    pipeline: Pipeline,
    index_timeout: int,
    poll_interval: int,
    scene_detection: bool = True,
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
                            body=_transcript_body(scene_detection)
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


def _transcript_body(scene_detection: bool) -> dict[str, dict[str, bool]]:
    return {"chunking_config": {"use_scene_detection": scene_detection}}
