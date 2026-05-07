# ABOUTME: MCP tool for on-demand object detection in video segments.
# ABOUTME: Detects arbitrary objects via text prompts and returns per-frame bounding boxes.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from reka_mcp.tools import READ_ONLY, logged, with_request_context

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import JsonDict, RekaClient

SEGMENT_MAX_RANGE_SECONDS = 15


def register_segment_tools(server: FastMCP, client: RekaClient) -> None:
    @server.tool(
        name="segment_video",
        description=(
            "Detect objects in a video segment using text prompts. Describe "
            "what to look for and get per-frame detections with bounding "
            "boxes and confidence scores.\n\n"
            "Prompt tips:\n"
            "- Use broad, visual categories: 'animal', 'vehicle', 'person', "
            "'text on screen'\n"
            "- Specific labels ('rabbit', 'Toyota') are less reliable — the "
            "detector matches visual patterns, not semantic concepts\n"
            "- Best for confirming whether a category of object appears in a "
            "time window, not for precise identification\n\n"
            "How to pick a time range:\n"
            "- Use search_videos to find WHEN something appears, then pass "
            "those timestamps here\n"
            "- Use get_scenes to scan systematically — call segment_video "
            "once per scene (scenes typically fit in the 15s window)\n"
            "- Or pass any range you already know\n\n"
            "Maximum range is 15 seconds per call; for longer spans, make "
            "multiple calls with consecutive windows.\n\n"
            "Does NOT require any feature indexing — works on any uploaded "
            "video."
        ),
        annotations=READ_ONLY,
    )
    @with_request_context
    @logged
    async def segment_video(
        video_id: str,
        prompts: list[str],
        start: float,
        end: float | None = None,
        threshold: float = 0.3,
        rationale: str | None = None,
    ) -> str:
        result = await client.segment_video(
            video_id,
            prompts=prompts,
            start=start,
            end=end,
            threshold=threshold,
        )

        summary = _build_summary(result)
        result["summary"] = summary

        if summary["total_detections"] == 0:
            result["hint"] = (
                "No objects matched the prompts in this segment. Try "
                "different prompt wording, lower the threshold (current: "
                f"{threshold}), or check a different time range."
            )
        else:
            result["hint"] = (
                "Use ask_video with the same start/end for contextual "
                "understanding of what's happening in this segment. To "
                "extend coverage, call segment_video on adjacent windows "
                "or iterate over scene boundaries from get_scenes."
            )

        return json.dumps(result)


def _build_summary(result: JsonDict) -> JsonDict:
    frames = result.get("frames", [])
    total = 0
    labels: set[str] = set()
    frames_with = 0
    for frame in frames:
        detections = frame.get("detections", [])
        if detections:
            frames_with += 1
        for det in detections:
            total += 1
            if det.get("label"):
                labels.add(det["label"])
    return {
        "total_detections": total,
        "detected_labels": sorted(labels),
        "frames_with_detections": frames_with,
    }
