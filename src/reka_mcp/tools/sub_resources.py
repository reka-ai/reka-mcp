# ABOUTME: MCP tools for reading video sub-resources (transcript, captions, scenes).
# ABOUTME: Handles pagination transparently and caps response size for token efficiency.

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from reka_mcp.pipelines import FeatureStatus
from reka_mcp.tools import READ_ONLY, logged, with_request_context

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import JsonDict, RekaClient


class TextTranscriptResponse(TypedDict):
    text: str
    total_chars: int
    truncated: bool


class CappedListResponse(TypedDict):
    data: list[JsonDict]
    returned_count: int
    truncated: bool


class VideoSummary(TypedDict):
    video_id: str
    name: str
    duration_seconds: float | None
    status: str | None
    features: dict[str, str]
    transcript_preview: NotRequired[str]
    scene_count: NotRequired[int]
    warnings: NotRequired[list[str]]


MAX_TRANSCRIPT_RESULTS = 500
MAX_CAPTIONS_RESULTS = 200


def register_sub_resource_tools(server: FastMCP, client: RekaClient) -> None:
    @server.tool(
        name="get_transcript",
        description=(
            "Get the spoken words in a video. Use this instead of ask_video "
            "when you need to read what was said — it returns the actual text, "
            "not a summary.\n\n"
            "Use start/end to narrow results for long videos.\n\n"
            "Requires the transcript feature to be indexed."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def get_transcript(
        video_id: str,
        format: Literal["text", "segments", "words"] = "text",
        start: float | None = None,
        end: float | None = None,
        max_results: int = 100,
        max_chars: int = 10000,
        rationale: str | None = None,
    ) -> str:
        if format == "text":
            return json.dumps(await _get_text_transcript(client, video_id, start, end, max_chars))
        max_results = min(max_results, MAX_TRANSCRIPT_RESULTS)
        segments = await client.get_transcript(
            video_id,
            format=format,
            start=start,
            end=end,
            max_items=max_results + 1,
        )
        assert isinstance(segments, list)
        return json.dumps(_cap_list(segments, max_results))

    @server.tool(
        name="get_captions",
        description=(
            "Get AI-generated visual descriptions of what happens on screen. "
            "Use this to understand the visual content without watching — "
            "each caption describes a short segment with timestamps.\n\n"
            "Use start/end to narrow results.\n\n"
            "Requires the captions feature (qa_only or full pipeline)."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def get_captions(
        video_id: str,
        start: float | None = None,
        end: float | None = None,
        max_results: int = 50,
        rationale: str | None = None,
    ) -> str:
        max_results = min(max_results, MAX_CAPTIONS_RESULTS)
        data = await client.get_captions(
            video_id,
            start=start,
            end=end,
            max_items=max_results + 1,
        )
        return json.dumps(_cap_list(data, max_results))

    @server.tool(
        name="get_scenes",
        description=(
            "Get detected scene boundaries with start/end timestamps. "
            "Use this to understand the video's structure, then pass scene "
            "timestamps as start/end to:\n"
            "- ask_video for per-scene contextual analysis\n"
            "- segment_video to detect specific objects per scene "
            "(scenes typically fit in segment_video's 15s max range)\n\n"
            "Requires transcript indexed with scene detection (full pipeline)."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def get_scenes(video_id: str, rationale: str | None = None) -> str:
        data = await client.get_scenes(video_id)
        return json.dumps(
            {
                "data": data,
                "returned_count": len(data),
                "truncated": False,
            }
        )

    @server.tool(
        name="get_feature_catalog",
        description=(
            "List available video analysis features with their dependencies "
            "and descriptions. Use this to understand what features exist and "
            "what pipelines to use with index_video."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def get_feature_catalog(rationale: str | None = None) -> str:
        catalog = await client.get_feature_catalog()
        return json.dumps(catalog)

    @server.tool(
        name="summarize_video",
        description=(
            "Start here. Get a compact overview of a video: metadata, which "
            "features are indexed, a transcript preview, and scene count. "
            "Use this to decide which tools to call next — then use "
            "segment_video to detect specific objects in time ranges of "
            "interest."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def summarize_video(video_id: str, rationale: str | None = None) -> str:
        return json.dumps(await _build_summary(client, video_id))


async def _get_text_transcript(
    client: RekaClient,
    video_id: str,
    start: float | None,
    end: float | None,
    max_chars: int,
) -> TextTranscriptResponse:
    resp = await client.get_transcript(video_id, format="text", start=start, end=end)
    assert isinstance(resp, dict)
    text: str = resp.get("text", "")
    total_chars = len(text)
    truncated = total_chars > max_chars
    if truncated:
        text = text[:max_chars]
    return TextTranscriptResponse(text=text, total_chars=total_chars, truncated=truncated)


def _cap_list(data: list[JsonDict], max_results: int) -> CappedListResponse:
    truncated = len(data) > max_results
    capped = data[:max_results]
    return CappedListResponse(data=capped, returned_count=len(capped), truncated=truncated)


async def _build_summary(client: RekaClient, video_id: str) -> VideoSummary:
    video = await client.get_video(video_id)

    metadata = video.metadata
    features = video.features or {}

    name = ""
    if metadata:
        name = metadata.video_name or metadata.title or ""

    summary = VideoSummary(
        video_id=video_id,
        name=name,
        duration_seconds=metadata.duration if metadata else None,
        status=video.status,
        features=features,
    )

    transcript_ready = features.get("transcript") == FeatureStatus.READY

    tasks: dict[str, Awaitable[JsonDict | list[JsonDict]]] = {}
    if transcript_ready:
        tasks["transcript"] = client.get_transcript(video_id, format="text")
        tasks["scenes"] = client.get_scenes(video_id)

    if tasks:
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        resolved = dict(zip(tasks.keys(), results, strict=True))
        warnings: list[str] = []

        if "transcript" in resolved:
            transcript_result = resolved["transcript"]
            if isinstance(transcript_result, BaseException):
                warnings.append(f"transcript: {transcript_result}")
            elif isinstance(transcript_result, dict):
                text: str = transcript_result.get("text", "")
                summary["transcript_preview"] = text[:800]

        if "scenes" in resolved:
            scenes_result = resolved["scenes"]
            if isinstance(scenes_result, BaseException):
                warnings.append(f"scenes: {scenes_result}")
            elif isinstance(scenes_result, list):
                summary["scene_count"] = len(scenes_result)

        if warnings:
            summary["warnings"] = warnings

    return summary
