# ABOUTME: MCP tool for asking questions about video content.
# ABOUTME: Manages conversation state for multi-turn Q&A with context preservation.

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NotRequired, TypedDict

from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from reka_mcp.tools import logged

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reka_mcp.client import RekaClient


class VideoContext(TypedDict, total=False):
    video_id: str
    start: float
    end: float


@dataclass
class ConversationState:
    messages: list[dict[str, str]] = field(default_factory=list)
    context: list[VideoContext] = field(default_factory=list)


class QAResponse(TypedDict):
    answer: str
    conversation_id: str
    hint: NotRequired[str]


MAX_CONVERSATIONS = 100
MAX_MESSAGES = 50


def register_qa_tools(
    server: FastMCP,
    client: RekaClient,
    max_conversations: int = MAX_CONVERSATIONS,
    max_messages: int = MAX_MESSAGES,
) -> None:
    conversations: OrderedDict[str, ConversationState] = OrderedDict()

    def _store(key: str, state: ConversationState) -> None:
        conversations[key] = state
        while len(conversations) > max_conversations:
            conversations.popitem(last=False)

    def _refresh(key: str) -> None:
        conversations.move_to_end(key)

    @server.tool(
        name="ask_video",
        description=(
            "Ask a question about one or more videos with visual analysis. "
            "Most effective on focused time ranges — use start/end to specify "
            "the segment to analyze.\n\n"
            "BEFORE calling this tool, read the reka://docs/guide resource "
            "for recommended workflows. In most cases, you should first:\n"
            "- search_videos to find WHEN something happens, then pass "
            "those timestamps here as start/end\n"
            "- segment_video to detect and locate specific objects\n"
            "- get_transcript to read what was said\n\n"
            "For single-video questions, pass video_id with start/end. "
            "For cross-video questions, pass videos — a list of "
            "video references with start/end each.\n\n"
            "For follow-up questions, pass conversation_id from the previous "
            "response. You can add start/end to drill into a specific moment "
            "while keeping the conversation context.\n\n"
            "Requires qa_only or full pipeline."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    @logged
    async def ask_video(
        question: str,
        video_id: str | None = None,
        start: float | None = None,
        end: float | None = None,
        videos: list[VideoContext] | None = None,
        conversation_id: str | None = None,
        rationale: str | None = None,
    ) -> str:
        error = _validate(video_id, start, videos, conversation_id)
        if error:
            raise ToolError(error)

        if conversation_id and conversation_id in conversations:
            _refresh(conversation_id)
            state = conversations[conversation_id]
            if len(state.messages) >= max_messages:
                raise ToolError(
                    f"Conversation has reached the {max_messages} message limit. "
                    "Start a new conversation."
                )
            if start is not None:
                state.context = _override_time_range(state.context, start, end)
        elif conversation_id:
            raise ToolError(f"Conversation '{conversation_id}' not found.")
        else:
            context = _build_context(video_id, start, end, videos)
            conversation_id = str(uuid.uuid4())
            state = ConversationState(context=context)
            _store(conversation_id, state)

        state.messages.append({"role": "user", "content": question})

        try:
            resp = await client.chat(
                messages=list(state.messages),
                context=list(state.context),
            )
        except Exception:
            state.messages.pop()
            raise

        answer = resp.get("response", "")
        state.messages.append({"role": "assistant", "content": answer})

        result = QAResponse(
            answer=answer,
            conversation_id=conversation_id,
        )

        if video_id is not None and start is None:
            result["hint"] = (
                "For more accurate results, narrow the time range first: "
                "use search_videos to find relevant moments, or "
                "segment_video to detect specific objects, then call "
                "ask_video with start/end."
            )
        elif video_id is not None and start is not None and end is not None and end - start > 60:
            result["hint"] = (
                "This covers a wide time range. For more detailed results, "
                "try scene-by-scene analysis: get_scenes, then call "
                "ask_video or segment_video once per scene."
            )

        return json.dumps(result)


def _validate(
    video_id: str | None,
    start: float | None,
    videos: list[VideoContext] | None,
    conversation_id: str | None,
) -> str | None:
    if conversation_id:
        if video_id or videos:
            return (
                "Pass conversation_id alone (or with start/end to narrow the "
                "time range) for follow-ups, or video_id/videos for new "
                "questions — not both."
            )
        return None
    if video_id and videos:
        return "Pass video_id or videos, not both."
    if not video_id and not videos:
        return (
            "Provide video_id (single video), videos (multiple), or conversation_id (follow-up)."
        )
    if start is not None and not video_id:
        return "start/end are only valid with video_id, not videos."
    return None


def _override_time_range(
    context: list[VideoContext],
    start: float,
    end: float | None,
) -> list[VideoContext]:
    overridden: list[VideoContext] = []
    for ctx in context:
        updated = VideoContext(video_id=ctx["video_id"], start=start)
        if end is not None:
            updated["end"] = end
        overridden.append(updated)
    return overridden


def _build_context(
    video_id: str | None,
    start: float | None,
    end: float | None,
    videos: list[VideoContext] | None,
) -> list[VideoContext]:
    if videos:
        return list(videos)
    ctx = VideoContext(video_id=video_id or "")
    if start is not None:
        ctx["start"] = start
    if end is not None:
        ctx["end"] = end
    return [ctx]
