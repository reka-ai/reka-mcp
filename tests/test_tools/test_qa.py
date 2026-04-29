# ABOUTME: Tests for the ask_video MCP tool with conversation state management.
# ABOUTME: Covers single/multi-video QA, follow-ups, visual context, and validation.

from __future__ import annotations

import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from reka_mcp.client import RekaClient
from reka_mcp.tools.qa import register_qa_tools
from tests.conftest import mock_client, tool_result_text


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_qa_tools(server, client)
    return server


def _chat_handler(expected_context=None):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if expected_context is not None:
            assert body["context"] == expected_context
        return httpx.Response(
            200,
            json={"response": "The video shows a presentation.", "model": "gemini-2.0"},
        )

    return handler


class TestAskVideoSingleVideo:
    async def test_text_only_question(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(client, _chat_handler(expected_context=[{"video_id": "v1"}]))
        result = await mcp_server.call_tool(
            "ask_video",
            {"question": "What is this video about?", "video_id": "v1"},
        )
        body = json.loads(tool_result_text(result))
        assert body["answer"] == "The video shows a presentation."
        assert "conversation_id" in body

    async def test_visual_context_with_start_end(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            _chat_handler(expected_context=[{"video_id": "v1", "start": 30.0, "end": 35.0}]),
        )
        result = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "What does the slide say?",
                "video_id": "v1",
                "start": 30.0,
                "end": 35.0,
            },
        )
        body = json.loads(tool_result_text(result))
        assert body["answer"]

    async def test_start_only_no_end(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(
            client,
            _chat_handler(expected_context=[{"video_id": "v1", "start": 120.0}]),
        )
        result = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "What's happening here?",
                "video_id": "v1",
                "start": 120.0,
            },
        )
        body = json.loads(tool_result_text(result))
        assert body["answer"]


class TestResponseHints:
    async def test_hint_when_no_time_bounds(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(client, _chat_handler(expected_context=[{"video_id": "v1"}]))
        result = await mcp_server.call_tool(
            "ask_video",
            {"question": "How many people are there?", "video_id": "v1"},
        )
        body = json.loads(tool_result_text(result))
        assert "hint" in body
        assert "search_videos" in body["hint"]

    async def test_no_hint_when_narrow_start_end_provided(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            _chat_handler(expected_context=[{"video_id": "v1", "start": 30.0, "end": 35.0}]),
        )
        result = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "What's happening?",
                "video_id": "v1",
                "start": 30.0,
                "end": 35.0,
            },
        )
        body = json.loads(tool_result_text(result))
        assert "hint" not in body

    async def test_hint_when_wide_time_range(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            _chat_handler(expected_context=[{"video_id": "v1", "start": 0.0, "end": 120.0}]),
        )
        result = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "What's happening?",
                "video_id": "v1",
                "start": 0.0,
                "end": 120.0,
            },
        )
        body = json.loads(tool_result_text(result))
        assert "hint" in body
        assert "get_scenes" in body["hint"]

    async def test_no_hint_on_follow_up(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(client, _chat_handler())
        r1 = await mcp_server.call_tool(
            "ask_video",
            {"question": "Q1", "video_id": "v1"},
        )
        conv_id = json.loads(tool_result_text(r1))["conversation_id"]

        result = await mcp_server.call_tool(
            "ask_video",
            {"question": "Tell me more", "conversation_id": conv_id},
        )
        body = json.loads(tool_result_text(result))
        assert "hint" not in body

    async def test_no_hint_on_multi_video(self, client: RekaClient, mcp_server: FastMCP) -> None:
        mock_client(client, _chat_handler())
        result = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "Compare these",
                "videos": [{"video_id": "v1"}, {"video_id": "v2"}],
            },
        )
        body = json.loads(tool_result_text(result))
        assert "hint" not in body


class TestAskVideoMultiVideo:
    async def test_cross_video_question(self, client: RekaClient, mcp_server: FastMCP) -> None:
        expected = [
            {"video_id": "v1", "start": 30.0, "end": 35.0},
            {"video_id": "v2", "start": 120.0},
        ]
        mock_client(client, _chat_handler(expected_context=expected))
        result = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "Compare these charts",
                "videos": [
                    {"video_id": "v1", "start": 30.0, "end": 35.0},
                    {"video_id": "v2", "start": 120.0},
                ],
            },
        )
        body = json.loads(tool_result_text(result))
        assert body["answer"]
        assert "conversation_id" in body


class TestConversationState:
    async def test_follow_up_preserves_context(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(req.content)
            if call_count == 1:
                assert len(body["messages"]) == 1
                assert body["context"] == [{"video_id": "v1", "start": 30.0, "end": 35.0}]
            else:
                assert len(body["messages"]) == 3
                assert body["messages"][0]["content"] == "What's on screen?"
                assert body["messages"][1]["role"] == "assistant"
                assert body["messages"][2]["content"] == "What color is it?"
                assert body["context"] == [{"video_id": "v1", "start": 30.0, "end": 35.0}]
            return httpx.Response(
                200,
                json={"response": f"Answer {call_count}", "model": "gemini-2.0"},
            )

        mock_client(client, handler)

        result1 = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "What's on screen?",
                "video_id": "v1",
                "start": 30.0,
                "end": 35.0,
            },
        )
        body1 = json.loads(tool_result_text(result1))
        conv_id = body1["conversation_id"]

        result2 = await mcp_server.call_tool(
            "ask_video",
            {"question": "What color is it?", "conversation_id": conv_id},
        )
        body2 = json.loads(tool_result_text(result2))
        assert body2["answer"] == "Answer 2"
        assert body2["conversation_id"] == conv_id

    async def test_follow_up_with_time_range_overrides_context(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(req.content)
            if call_count == 1:
                assert body["context"] == [{"video_id": "v1"}]
            else:
                assert len(body["messages"]) == 3
                assert body["context"] == [{"video_id": "v1", "start": 10.0, "end": 20.0}]
            return httpx.Response(
                200,
                json={"response": f"Answer {call_count}", "model": "gemini-2.0"},
            )

        mock_client(client, handler)

        result1 = await mcp_server.call_tool(
            "ask_video",
            {"question": "What is this video about?", "video_id": "v1"},
        )
        body1 = json.loads(tool_result_text(result1))
        conv_id = body1["conversation_id"]

        result2 = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "What's happening at this moment?",
                "conversation_id": conv_id,
                "start": 10.0,
                "end": 20.0,
            },
        )
        body2 = json.loads(tool_result_text(result2))
        assert body2["answer"] == "Answer 2"
        assert body2["conversation_id"] == conv_id

    async def test_follow_up_without_time_range_keeps_original_context(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(req.content)
            if call_count == 2:
                assert body["context"] == [{"video_id": "v1", "start": 30.0, "end": 35.0}]
            return httpx.Response(
                200,
                json={"response": f"Answer {call_count}", "model": "gemini-2.0"},
            )

        mock_client(client, handler)

        result1 = await mcp_server.call_tool(
            "ask_video",
            {
                "question": "What's on screen?",
                "video_id": "v1",
                "start": 30.0,
                "end": 35.0,
            },
        )
        body1 = json.loads(tool_result_text(result1))
        conv_id = body1["conversation_id"]

        result2 = await mcp_server.call_tool(
            "ask_video",
            {"question": "Tell me more", "conversation_id": conv_id},
        )
        body2 = json.loads(tool_result_text(result2))
        assert body2["answer"] == "Answer 2"

    async def test_invalid_conversation_id_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        with pytest.raises(ToolError, match="nonexistent"):
            await mcp_server.call_tool(
                "ask_video",
                {"question": "Follow up", "conversation_id": "nonexistent"},
            )


class TestValidation:
    async def test_neither_video_id_nor_videos_nor_conversation(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        with pytest.raises(ToolError, match=r"(?i)provide video_id"):
            await mcp_server.call_tool("ask_video", {"question": "What?"})

    async def test_both_video_id_and_videos(self, client: RekaClient, mcp_server: FastMCP) -> None:
        with pytest.raises(ToolError, match="not both"):
            await mcp_server.call_tool(
                "ask_video",
                {
                    "question": "What?",
                    "video_id": "v1",
                    "videos": [{"video_id": "v2"}],
                },
            )

    async def test_start_without_video_id(self, client: RekaClient, mcp_server: FastMCP) -> None:
        with pytest.raises(ToolError, match="only valid with video_id"):
            await mcp_server.call_tool(
                "ask_video",
                {
                    "question": "What?",
                    "videos": [{"video_id": "v1"}],
                    "start": 30.0,
                },
            )

    async def test_api_error_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                409,
                json={"error": {"message": "Video not ready for QA"}},
            ),
        )
        with pytest.raises(ToolError, match="not ready"):
            await mcp_server.call_tool(
                "ask_video",
                {"question": "What?", "video_id": "v1"},
            )


class TestConversationEviction:
    async def test_oldest_conversation_evicted_when_full(
        self,
        client: RekaClient,
    ) -> None:
        server = FastMCP("test-reka-vision")
        register_qa_tools(server, client, max_conversations=2)

        mock_client(client, _chat_handler())

        r1 = await server.call_tool("ask_video", {"question": "Q1", "video_id": "v1"})
        conv1 = json.loads(tool_result_text(r1))["conversation_id"]

        r2 = await server.call_tool("ask_video", {"question": "Q2", "video_id": "v2"})
        conv2 = json.loads(tool_result_text(r2))["conversation_id"]

        r3 = await server.call_tool("ask_video", {"question": "Q3", "video_id": "v3"})
        json.loads(tool_result_text(r3))["conversation_id"]

        # conv1 should be evicted, conv2 and conv3 should work
        with pytest.raises(ToolError, match="not found"):
            await server.call_tool(
                "ask_video",
                {"question": "Follow up", "conversation_id": conv1},
            )

        r4 = await server.call_tool(
            "ask_video",
            {"question": "Follow up", "conversation_id": conv2},
        )
        assert json.loads(tool_result_text(r4))["answer"]

    async def test_message_limit_raises_tool_error(
        self,
        client: RekaClient,
    ) -> None:
        server = FastMCP("test-reka-vision")
        register_qa_tools(server, client, max_messages=4)

        mock_client(client, _chat_handler())

        r1 = await server.call_tool("ask_video", {"question": "Q1", "video_id": "v1"})
        conv_id = json.loads(tool_result_text(r1))["conversation_id"]

        # 2 messages after first call (user + assistant). Second call adds 2 more = 4.
        await server.call_tool(
            "ask_video",
            {"question": "Q2", "conversation_id": conv_id},
        )

        # Third call would push to 5 messages, exceeding the limit of 4.
        with pytest.raises(ToolError, match="message limit"):
            await server.call_tool(
                "ask_video",
                {"question": "Q3", "conversation_id": conv_id},
            )
