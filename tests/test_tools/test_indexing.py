# ABOUTME: Tests for the index_video MCP tool with DAG orchestration.
# ABOUTME: Covers all acceptance scenarios: pipelines, DAG order, failure, timeout, errors.

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from reka_mcp.client import RekaClient
from reka_mcp.tools.indexing import _run_indexing, register_indexing_tools
from tests.conftest import mock_client, tool_result_text


@pytest.fixture
def mcp_server(client: RekaClient) -> FastMCP:
    server = FastMCP("test-reka-vision")
    register_indexing_tools(server, client, index_timeout=600, poll_interval=5)
    return server


def _plan_response(
    done: bool = False,
    actionable: list[str] | None = None,
    blocked: list[str] | None = None,
    statuses: dict[str, str] | None = None,
) -> dict[str, str | bool | list[str] | dict[str, str]]:
    return {
        "done": done,
        "actionable": actionable or [],
        "blocked": blocked or [],
        "statuses": statuses or {},
    }


class TestSearchOnlyPipeline:
    """Scenario 1 + 8: search_only indexes in DAG order."""

    async def test_indexes_through_dag_order(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        plan_call = 0
        triggered: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal plan_call
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                plan_call += 1
                if plan_call == 1:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["transcript"],
                            statuses={
                                "transcript": "none",
                                "captions": "none",
                                "embeddings": "none",
                            },
                        ),
                    )
                elif plan_call == 2:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["captions"],
                            statuses={
                                "transcript": "ready",
                                "captions": "none",
                                "embeddings": "none",
                            },
                        ),
                    )
                elif plan_call == 3:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["embeddings"],
                            statuses={
                                "transcript": "ready",
                                "captions": "ready",
                                "embeddings": "none",
                            },
                        ),
                    )
                else:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            done=True,
                            statuses={
                                "transcript": "ready",
                                "captions": "ready",
                                "embeddings": "ready",
                            },
                        ),
                    )

            for feat in ("transcript", "captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    triggered.append(feat)
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)

        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            result = await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "search_only"},
            )

        body = json.loads(tool_result_text(result))
        assert body["status"] == "ready"
        assert body["features"]["transcript"] == "ready"
        assert body["features"]["captions"] == "ready"
        assert body["features"]["embeddings"] == "ready"
        assert triggered == ["transcript", "captions", "embeddings"]

    async def test_ready_response_includes_hint(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})
            if "features/plan" in url:
                return httpx.Response(
                    200,
                    json=_plan_response(
                        done=True,
                        statuses={
                            "transcript": "ready",
                            "captions": "ready",
                            "embeddings": "ready",
                        },
                    ),
                )
            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)
        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            result = await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "search_only"},
            )

        body = json.loads(tool_result_text(result))
        assert body["status"] == "ready"
        assert "hint" in body
        assert "search_videos" in body["hint"]


class TestQaOnlyPipeline:
    """Scenario 2: qa_only skips embeddings."""

    async def test_only_triggers_transcript_and_captions(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        plan_call = 0
        triggered: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal plan_call
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                plan_call += 1
                if plan_call == 1:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["transcript"],
                            statuses={
                                "transcript": "none",
                                "captions": "none",
                            },
                        ),
                    )
                elif plan_call == 2:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["captions"],
                            statuses={
                                "transcript": "ready",
                                "captions": "none",
                            },
                        ),
                    )
                else:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            done=True,
                            statuses={
                                "transcript": "ready",
                                "captions": "ready",
                            },
                        ),
                    )

            for feat in ("transcript", "captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    triggered.append(feat)
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)

        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            result = await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "qa_only"},
            )

        body = json.loads(tool_result_text(result))
        assert body["status"] == "ready"
        assert "embeddings" not in triggered


class TestFullPipeline:
    """Scenario 3: full pipeline triggers captions + objects in parallel."""

    async def test_parallel_triggers_after_transcript(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        plan_call = 0
        triggered_per_round: list[list[str]] = []
        current_round: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal plan_call, current_round
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                if current_round:
                    triggered_per_round.append(current_round)
                    current_round = []
                plan_call += 1
                if plan_call == 1:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["transcript"],
                            statuses={
                                "transcript": "none",
                                "captions": "none",
                                "embeddings": "none",
                                "objects": "none",
                            },
                        ),
                    )
                elif plan_call == 2:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["captions", "objects"],
                            statuses={
                                "transcript": "ready",
                                "captions": "none",
                                "embeddings": "none",
                                "objects": "none",
                            },
                        ),
                    )
                elif plan_call == 3:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["embeddings"],
                            statuses={
                                "transcript": "ready",
                                "captions": "ready",
                                "embeddings": "none",
                                "objects": "ready",
                            },
                        ),
                    )
                else:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            done=True,
                            statuses={
                                "transcript": "ready",
                                "captions": "ready",
                                "embeddings": "ready",
                                "objects": "ready",
                            },
                        ),
                    )

            for feat in ("transcript", "captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    current_round.append(feat)
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)

        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            result = await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "full"},
            )

        if current_round:
            triggered_per_round.append(current_round)

        body = json.loads(tool_result_text(result))
        assert body["status"] == "ready"
        # Round 1: transcript only; Round 2: captions + objects in parallel
        assert triggered_per_round[0] == ["transcript"]
        assert set(triggered_per_round[1]) == {"captions", "objects"}


class TestFullPipelineSceneDetection:
    """Full pipeline passes use_scene_detection=true when triggering transcript."""

    async def test_transcript_trigger_includes_scene_detection(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        plan_call = 0
        transcript_body: dict | None = None

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal plan_call, transcript_body
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                plan_call += 1
                if plan_call == 1:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["transcript"],
                            statuses={
                                "transcript": "none",
                                "captions": "none",
                                "embeddings": "none",
                                "objects": "none",
                            },
                        ),
                    )
                else:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            done=True,
                            statuses={
                                "transcript": "ready",
                                "captions": "ready",
                                "embeddings": "ready",
                                "objects": "ready",
                            },
                        ),
                    )

            if url.endswith("/features/transcript") and req.method == "POST":
                transcript_body = json.loads(req.content)
                return httpx.Response(
                    202,
                    json={
                        "video_id": "vid-1",
                        "feature": "transcript",
                        "status": "processing",
                    },
                )

            for feat in ("captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)

        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "full"},
            )

        assert transcript_body is not None
        assert transcript_body["chunking_config"] == {
            "use_scene_detection": True,
        }

    async def test_search_only_does_not_include_scene_detection(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        plan_call = 0
        transcript_body: dict | None = None

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal plan_call, transcript_body
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                plan_call += 1
                if plan_call == 1:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["transcript"],
                            statuses={
                                "transcript": "none",
                                "captions": "none",
                                "embeddings": "none",
                            },
                        ),
                    )
                else:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            done=True,
                            statuses={
                                "transcript": "ready",
                                "captions": "ready",
                                "embeddings": "ready",
                            },
                        ),
                    )

            if url.endswith("/features/transcript") and req.method == "POST":
                transcript_body = json.loads(req.content)
                return httpx.Response(
                    202,
                    json={
                        "video_id": "vid-1",
                        "feature": "transcript",
                        "status": "processing",
                    },
                )

            for feat in ("captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)

        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "search_only"},
            )

        assert transcript_body is not None
        assert "chunking_config" not in transcript_body


class TestAlreadyIndexed:
    """Scenario 4: already-indexed video returns immediately."""

    async def test_returns_ready_without_triggering(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        triggered: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                return httpx.Response(
                    200,
                    json=_plan_response(
                        done=True,
                        statuses={
                            "transcript": "ready",
                            "captions": "ready",
                            "embeddings": "ready",
                        },
                    ),
                )

            for feat in ("transcript", "captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    triggered.append(feat)
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)
        result = await mcp_server.call_tool(
            "index_video",
            {"video_id": "vid-1", "pipeline": "search_only"},
        )

        body = json.loads(tool_result_text(result))
        assert body["status"] == "ready"
        assert triggered == []


class TestTimeout:
    """Scenario 5: timeout returns partial status."""

    async def test_returns_timeout_with_partial_features(
        self,
        client: RekaClient,
    ) -> None:
        server = FastMCP("test-reka-vision")
        register_indexing_tools(server, client, index_timeout=0, poll_interval=5)

        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "video_id": "vid-1",
                        "status": "uploaded",
                        "features": {
                            "transcript": "processing",
                            "captions": "none",
                            "embeddings": "none",
                        },
                    },
                )

            if "features/plan" in url:
                return httpx.Response(
                    200,
                    json=_plan_response(
                        actionable=["transcript"],
                        statuses={
                            "transcript": "none",
                            "captions": "none",
                            "embeddings": "none",
                        },
                    ),
                )

            for feat in ("transcript", "captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)

        result = await server.call_tool(
            "index_video",
            {"video_id": "vid-1", "pipeline": "search_only"},
        )

        body = json.loads(tool_result_text(result))
        assert body["status"] == "timeout"
        assert body["video_id"] == "vid-1"
        assert "timed out" in body["message"].lower()


class TestVideoNotUploaded:
    """Scenario 6: video not in 'uploaded' status."""

    async def test_raises_tool_error_for_uploading_video(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"video_id": "vid-1", "status": "uploading"}),
        )
        with pytest.raises(ToolError, match="uploading"):
            await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "search_only"},
            )


class TestUnknownPipeline:
    """Scenario 7: unknown pipeline."""

    async def test_returns_error_with_valid_options(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        with pytest.raises(ToolError, match="'search_only', 'qa_only' or 'full'"):
            await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "nonexistent"},
            )


class TestFeatureFailure:
    """Scenario 9: feature failure stops with error."""

    async def test_returns_failed_status(self, client: RekaClient, mcp_server: FastMCP) -> None:
        plan_call = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal plan_call
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                plan_call += 1
                if plan_call == 1:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            actionable=["transcript"],
                            statuses={
                                "transcript": "none",
                                "captions": "none",
                                "embeddings": "none",
                            },
                        ),
                    )
                else:
                    return httpx.Response(
                        200,
                        json=_plan_response(
                            statuses={
                                "transcript": "ready",
                                "captions": "failed",
                                "embeddings": "none",
                            },
                        ),
                    )

            for feat in ("transcript", "captions", "embeddings", "objects"):
                if url.endswith(f"/features/{feat}") and req.method == "POST":
                    return httpx.Response(
                        202,
                        json={
                            "video_id": "vid-1",
                            "feature": feat,
                            "status": "processing",
                        },
                    )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)

        with patch("reka_mcp.tools.indexing.asyncio.sleep", new_callable=AsyncMock):
            result = await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "search_only"},
            )

        body = json.loads(tool_result_text(result))
        assert body["status"] == "failed"
        assert "captions" in body["error"]


class TestScenesCoproduction:
    """Server adds SCENES to required when TRANSCRIPT is desired, but SCENES
    is co-produced (not independently triggerable). done=False even when all
    our desired features are ready. Tool must check desired features directly."""

    async def test_returns_ready_when_desired_features_ready_despite_scenes_none(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)

            if url.endswith("/v2/videos/vid-1") and req.method == "GET":
                return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})

            if "features/plan" in url:
                # Server includes scenes in required; scenes stays none.
                # done=False even though transcript+captions+embeddings are ready.
                return httpx.Response(
                    200,
                    json={
                        "done": False,
                        "actionable": [],
                        "blocked": [],
                        "statuses": {
                            "transcript": "ready",
                            "captions": "ready",
                            "embeddings": "ready",
                            "scenes": "none",
                        },
                    },
                )

            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)
        result = await mcp_server.call_tool(
            "index_video",
            {"video_id": "vid-1", "pipeline": "search_only"},
        )

        body = json.loads(tool_result_text(result))
        assert body["status"] == "ready"
        assert body["features"]["transcript"] == "ready"
        assert body["features"]["captions"] == "ready"
        assert body["features"]["embeddings"] == "ready"


class TestCancellation:
    """Cancellation (e.g. client disconnect) stops the polling loop."""

    @staticmethod
    def _processing_handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url.endswith("/v2/videos/vid-1") and req.method == "GET":
            return httpx.Response(200, json={"video_id": "vid-1", "status": "uploaded"})
        if "features/plan" in url:
            return httpx.Response(
                200,
                json=_plan_response(
                    statuses={
                        "transcript": "processing",
                        "captions": "none",
                        "embeddings": "none",
                    },
                ),
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    async def test_cancelled_during_poll_propagates(
        self,
        client: RekaClient,
    ) -> None:
        mock_client(client, self._processing_handler)

        sleep_count = 0

        async def cancel_on_second_sleep(seconds: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch("reka_mcp.tools.indexing.asyncio.sleep", cancel_on_second_sleep),
            pytest.raises(asyncio.CancelledError),
        ):
            await _run_indexing(client, "vid-1", "search_only", 600, 5)
        assert sleep_count == 2

    async def test_cancellation_is_logged(
        self,
        client: RekaClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_client(client, self._processing_handler)

        async def cancel_immediately(seconds: float) -> None:
            raise asyncio.CancelledError()

        with (
            caplog.at_level(logging.INFO, logger="reka_mcp.tools.indexing"),
            patch("reka_mcp.tools.indexing.asyncio.sleep", cancel_immediately),
            pytest.raises(asyncio.CancelledError),
        ):
            await _run_indexing(client, "vid-1", "search_only", 600, 5)

        assert any("vid-1" in r.message and "cancel" in r.message.lower() for r in caplog.records)


class TestApiError:
    """API errors raise ToolError with isError=true."""

    async def test_api_error_raises_tool_error(
        self, client: RekaClient, mcp_server: FastMCP
    ) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(404, json={"error": {"message": "Video not found"}}),
        )
        with pytest.raises(ToolError, match=r"(?i)not found"):
            await mcp_server.call_tool(
                "index_video",
                {"video_id": "vid-1", "pipeline": "search_only"},
            )
