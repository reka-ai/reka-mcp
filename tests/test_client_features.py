# ABOUTME: Tests for RekaClient feature planning and triggering methods.
# ABOUTME: Covers plan_features and the trigger_feature dispatcher.

from __future__ import annotations

import json

import httpx
import pytest

from reka_mcp.client import RekaClient
from tests.conftest import mock_client


class TestPlanFeatures:
    async def test_posts_to_plan_endpoint(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/videos/vid-1/features/plan")
            body = json.loads(req.content)
            assert body == {"desired": ["captions", "transcript"]}
            return httpx.Response(
                200,
                json={
                    "done": False,
                    "actionable": ["transcript"],
                    "blocked": [],
                    "statuses": {"transcript": "none", "captions": "none"},
                },
            )

        mock_client(client, handler)
        result = await client.plan_features("vid-1", ["captions", "transcript"])
        assert result.done is False
        assert result.actionable == ["transcript"]

    async def test_plan_returns_done_when_all_ready(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "done": True,
                    "actionable": [],
                    "blocked": [],
                    "statuses": {"transcript": "ready", "captions": "ready"},
                },
            ),
        )
        result = await client.plan_features("vid-1", ["transcript", "captions"])
        assert result.done is True


class TestTriggerFeature:
    async def test_posts_to_correct_endpoint(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/videos/vid-1/features/embeddings")
            body = json.loads(req.content)
            assert body == {"force": False}
            return httpx.Response(
                202,
                json={
                    "video_id": "vid-1",
                    "feature": "embeddings",
                    "status": "processing",
                },
            )

        mock_client(client, handler)
        result = await client.trigger_feature("vid-1", "embeddings")
        assert result.feature == "embeddings"
        assert result.status == "processing"

    async def test_passes_force_flag(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["force"] is True
            return httpx.Response(
                202,
                json={
                    "video_id": "vid-1",
                    "feature": "transcript",
                    "status": "processing",
                },
            )

        mock_client(client, handler)
        await client.trigger_feature("vid-1", "transcript", force=True)

    async def test_all_triggerable_features(self, client: RekaClient) -> None:
        for feature in ("transcript", "captions", "embeddings"):

            def handler(req: httpx.Request, _f: str = feature) -> httpx.Response:
                assert str(req.url).endswith(f"/features/{_f}")
                return httpx.Response(
                    202,
                    json={
                        "video_id": "vid-1",
                        "feature": _f,
                        "status": "processing",
                    },
                )

            mock_client(client, handler)
            result = await client.trigger_feature("vid-1", feature)
            assert result.feature == feature

    async def test_passes_extra_body_fields(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body == {
                "force": False,
                "chunking_config": {"use_scene_detection": True},
            }
            return httpx.Response(
                202,
                json={
                    "video_id": "vid-1",
                    "feature": "transcript",
                    "status": "processing",
                },
            )

        mock_client(client, handler)
        await client.trigger_feature(
            "vid-1",
            "transcript",
            body={"chunking_config": {"use_scene_detection": True}},
        )

    async def test_raises_for_unknown_feature(self, client: RekaClient) -> None:
        with pytest.raises(ValueError, match="is not a valid Feature"):
            await client.trigger_feature("vid-1", "bogus")
