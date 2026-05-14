# ABOUTME: Tests for RekaClient search, chat, segment, and sub-resource methods.
# ABOUTME: Covers search_videos, chat, segment_video, transcript, captions, scenes, objects, feature catalog.

from __future__ import annotations

import json

import httpx
import pytest

from reka_mcp.client import ConflictError, RekaClient
from tests.conftest import mock_client


class TestSearchVideos:
    async def test_posts_search_request(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/search")
            body = json.loads(req.content)
            assert body["query"] == "person speaking"
            assert body["page_limit"] == 10
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "video_id": "v1",
                            "start": 10.0,
                            "end": 15.0,
                            "score": 0.95,
                            "rank": 1,
                        }
                    ],
                    "next_page_token": None,
                    "search_pool": {"video_count": 5, "total_duration": 3600.0},
                },
            )

        mock_client(client, handler)
        results = await client.search_videos("person speaking")
        assert len(results) == 1
        assert results[0]["video_id"] == "v1"
        assert results[0]["score"] == 0.95

    async def test_search_with_filters(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["video_ids"] == ["v1", "v2"]
            assert body["group_ids"] == ["g1"]
            assert body["page_limit"] == 5
            return httpx.Response(
                200,
                json={
                    "data": [],
                    "next_page_token": None,
                    "search_pool": {"video_count": 0, "total_duration": 0.0},
                },
            )

        mock_client(client, handler)
        results = await client.search_videos(
            "query",
            video_ids=["v1", "v2"],
            group_id="g1",
            max_results=5,
        )
        assert results == []

    async def test_search_empty_results(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [],
                    "next_page_token": None,
                    "search_pool": {"video_count": 3, "total_duration": 100.0},
                },
            ),
        )
        results = await client.search_videos("xyzzy_gibberish")
        assert results == []


class TestChat:
    async def test_posts_chat_request(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/chat")
            body = json.loads(req.content)
            assert body["messages"] == [{"role": "user", "content": "What is this?"}]
            assert body["context"] == [{"video_id": "v1", "start": 10.0, "end": 15.0}]
            return httpx.Response(
                200,
                json={"response": "This is a presentation.", "model": "gemini-2.0"},
            )

        mock_client(client, handler)
        result = await client.chat(
            messages=[{"role": "user", "content": "What is this?"}],
            context=[{"video_id": "v1", "start": 10.0, "end": 15.0}],
        )
        assert result["response"] == "This is a presentation."
        assert result["model"] == "gemini-2.0"

    async def test_chat_with_history(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert len(body["messages"]) == 3
            assert body["messages"][0]["role"] == "user"
            assert body["messages"][1]["role"] == "assistant"
            assert body["messages"][2]["role"] == "user"
            return httpx.Response(200, json={"response": "More details.", "model": "gemini-2.0"})

        mock_client(client, handler)
        result = await client.chat(
            messages=[
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "A video."},
                {"role": "user", "content": "Tell me more."},
            ],
            context=[{"video_id": "v1"}],
        )
        assert result["response"] == "More details."


class TestSegmentVideo:
    async def test_posts_segment_request(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/videos/v1/segment")
            body = json.loads(req.content)
            assert body["prompts"] == [
                {"type": "text", "text": "person"},
                {"type": "text", "text": "laptop"},
            ]
            assert body["start"] == 10.0
            assert body["end"] == 25.0
            assert body["threshold"] == 0.5
            return httpx.Response(
                200,
                json={
                    "frames": [
                        {
                            "timestamp": 10.0,
                            "detections": [
                                {
                                    "label": "person",
                                    "prompt_index": 0,
                                    "score": 0.95,
                                    "bbox": {
                                        "x_min": 0.1,
                                        "y_min": 0.2,
                                        "x_max": 0.5,
                                        "y_max": 0.8,
                                    },
                                }
                            ],
                        }
                    ],
                    "frame_size": {"width": 1920, "height": 1080},
                    "frame_count": 1,
                },
            )

        mock_client(client, handler)
        result = await client.segment_video(
            "v1",
            prompts=["person", "laptop"],
            start=10.0,
            end=25.0,
            threshold=0.5,
        )
        assert len(result["frames"]) == 1
        assert result["frames"][0]["detections"][0]["label"] == "person"
        assert result["frame_count"] == 1

    async def test_omits_optional_fields(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert "end" not in body
            assert "threshold" not in body
            return httpx.Response(
                200,
                json={
                    "frames": [],
                    "frame_size": {"width": 1920, "height": 1080},
                    "frame_count": 0,
                },
            )

        mock_client(client, handler)
        result = await client.segment_video("v1", prompts=["car"], start=0.0)
        assert result["frames"] == []


class TestGetTranscript:
    async def test_text_format_no_pagination(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "GET"
            assert "/v2/videos/v1/transcript" in str(req.url)
            assert "format=text" in str(req.url)
            return httpx.Response(
                200,
                json={"text": "Hello world this is a transcript."},
            )

        mock_client(client, handler)
        result = await client.get_transcript("v1", format="text")
        assert result["text"] == "Hello world this is a transcript."

    async def test_segments_format_uses_paginate(self, client: RekaClient) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    200,
                    json={
                        "data": [{"start": 0.0, "end": 5.0, "text": "Hello"}],
                        "next_page_token": "page2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "data": [{"start": 5.0, "end": 10.0, "text": "World"}],
                    "next_page_token": None,
                },
            )

        mock_client(client, handler)
        result = await client.get_transcript("v1", format="segments")
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "World"
        assert call_count == 2

    async def test_words_format_uses_paginate(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [
                        {"start": 0.0, "end": 0.3, "text": "Hello"},
                        {"start": 0.3, "end": 0.6, "text": "world"},
                    ],
                    "next_page_token": None,
                },
            ),
        )
        result = await client.get_transcript("v1", format="words")
        assert len(result) == 2

    async def test_passes_time_range(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            assert "start=10.0" in url
            assert "end=30.0" in url
            return httpx.Response(200, json={"text": "partial"})

        mock_client(client, handler)
        await client.get_transcript("v1", format="text", start=10.0, end=30.0)

    async def test_409_raises_conflict(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                409,
                json={
                    "error": {"message": "Feature 'transcript' is not ready. Current status: none"}
                },
            ),
        )
        with pytest.raises(ConflictError):
            await client.get_transcript("v1")


class TestGetCaptions:
    async def test_fetches_captions(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [{"start": 0.0, "end": 10.0, "caption": "A person speaking"}],
                    "next_page_token": None,
                },
            ),
        )
        result = await client.get_captions("v1")
        assert len(result) == 1
        assert result[0]["caption"] == "A person speaking"

    async def test_passes_time_range(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            assert "start=5.0" in url
            assert "end=15.0" in url
            return httpx.Response(200, json={"data": [], "next_page_token": None})

        mock_client(client, handler)
        await client.get_captions("v1", start=5.0, end=15.0)


class TestGetScenes:
    async def test_fetches_scenes(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "start": 0.0, "end": 15.0},
                        {"index": 1, "start": 15.0, "end": 30.0},
                    ],
                    "next_page_token": None,
                },
            ),
        )
        result = await client.get_scenes("v1")
        assert len(result) == 2
        assert result[0]["index"] == 0


class TestGetFeatureCatalog:
    async def test_fetches_catalog(self, client: RekaClient) -> None:
        catalog = {
            "features": [
                {
                    "name": "transcript",
                    "description": "Speech-to-text",
                    "depends_on": [],
                    "produces": ["transcript", "scenes"],
                },
                {
                    "name": "captions",
                    "description": "Visual descriptions",
                    "depends_on": ["transcript"],
                },
            ]
        }

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "GET"
            assert str(req.url).endswith("/v2/features")
            return httpx.Response(200, json=catalog)

        mock_client(client, handler)
        result = await client.get_feature_catalog()
        assert "transcript" in result
        assert result["transcript"]["depends_on"] == []
        assert result["transcript"]["produces"] == ["transcript", "scenes"]
        assert "captions" in result
        assert result["captions"]["depends_on"] == ["transcript"]
