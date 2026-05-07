# ABOUTME: Tests for RekaClient — async HTTP wrapper for the V2 API.
# ABOUTME: Covers auth, retry, error handling, CRUD methods, and pagination.

from __future__ import annotations

import httpx
import pytest

from reka_mcp.client import (
    AuthError,
    ConflictError,
    NotFoundError,
    RekaClient,
    ServerError,
    ValidationError,
)
from tests.conftest import mock_client


class TestAuthHeader:
    async def test_requests_include_api_key_header(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(200, json={"results": []}),
        )
        await client.list_videos()

    async def test_client_injects_static_auth_header_per_request(self) -> None:
        c = RekaClient(api_url="http://test.local", api_key="my-secret")
        captured_headers: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_headers.update(req.headers)
            return httpx.Response(200, json={"results": []})

        mock_client(c, handler)
        await c.list_videos()

        assert captured_headers["x-api-key"] == "my-secret"
        assert "x-api-key" not in c._http.headers


class TestMcpSessionIdHeader:
    async def test_sends_session_id_when_contextvar_set(self, client: RekaClient) -> None:
        from reka_mcp.client import mcp_session_id_var

        captured_headers: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_headers.update(req.headers)
            return httpx.Response(200, json={"results": []})

        mock_client(client, handler)
        token = mcp_session_id_var.set("sess-abc")
        try:
            await client.list_videos()
        finally:
            mcp_session_id_var.reset(token)
        assert captured_headers["x-mcp-session-id"] == "sess-abc"

    async def test_omits_session_id_when_contextvar_unset(self, client: RekaClient) -> None:
        captured_headers: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_headers.update(req.headers)
            return httpx.Response(200, json={"results": []})

        mock_client(client, handler)
        await client.list_videos()
        assert "x-mcp-session-id" not in captured_headers


class TestErrorHandling:
    async def test_401_raises_auth_error(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(401, json={"error": {"message": "bad key"}}),
        )
        with pytest.raises(AuthError, match="Authentication failed"):
            await client.get_video("vid-1")

    async def test_403_raises_auth_error(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(403, json={"error": {"message": "forbidden"}}),
        )
        with pytest.raises(AuthError, match="Authentication failed"):
            await client.get_video("vid-1")

    async def test_404_raises_not_found_error(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(404, json={"error": {"message": "Video not found"}}),
        )
        with pytest.raises(NotFoundError, match="not found"):
            await client.get_video("missing-id")

    async def test_409_raises_conflict_error_with_api_message(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(409, json={"error": {"message": "Already exists"}}),
        )
        with pytest.raises(ConflictError, match="Already exists"):
            await client.create_group("dup-group")

    async def test_422_raises_validation_error_with_detail(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                422, json={"error": {"message": "Invalid video_url format"}}
            ),
        )
        with pytest.raises(ValidationError, match="Invalid video_url"):
            await client.upload_video(video_url="bad-url")


class TestRetry:
    async def test_retries_on_500(self, client: RekaClient) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(500, json={"error": {"message": "oops"}})
            return httpx.Response(200, json={"video_id": "v1", "status": "uploaded"})

        mock_client(client, handler)
        client._retry_base_delay = 0
        result = await client.get_video("v1")
        assert result.video_id == "v1"
        assert call_count == 3

    async def test_gives_up_after_max_retries(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(502, json={"error": {"message": "bad gateway"}}),
        )
        client._retry_base_delay = 0
        with pytest.raises(ServerError, match="temporarily unavailable"):
            await client.get_video("v1")

    async def test_no_retry_on_4xx(self, client: RekaClient) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(404, json={"error": {"message": "not found"}})

        mock_client(client, handler)
        client._retry_base_delay = 0
        with pytest.raises(NotFoundError):
            await client.get_video("v1")
        assert call_count == 1


class TestVideoMethods:
    async def test_upload_video(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/videos")
            return httpx.Response(202, json={"video_id": "new-vid", "status": "uploading"})

        mock_client(client, handler)
        result = await client.upload_video(
            video_url="https://example.com/video.mp4", name="test-vid"
        )
        assert result.video_id == "new-vid"
        assert result.status == "uploading"

    async def test_upload_video_sends_form_data(self, client: RekaClient) -> None:
        captured_body: bytes = b""

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal captured_body
            captured_body = req.content
            return httpx.Response(202, json={"video_id": "vid-1", "status": "uploading"})

        mock_client(client, handler)
        await client.upload_video(
            video_url="https://example.com/v.mp4",
            name="my-vid",
            description="desc",
            group_id="g1",
        )
        body_str = captured_body.decode()
        assert "video_url" in body_str
        assert "video_name" in body_str
        assert "description" in body_str
        assert "group_id" in body_str

    async def test_list_videos(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "GET"
            assert "/v2/videos" in str(req.url)
            return httpx.Response(
                200,
                json={"results": [{"video_id": "v1", "status": "uploaded", "features": {}}]},
            )

        mock_client(client, handler)
        result = await client.list_videos()
        assert len(result) == 1
        assert result[0].video_id == "v1"

    async def test_list_videos_with_ids(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert "ids=v1" in str(req.url)
            assert "ids=v2" in str(req.url)
            return httpx.Response(200, json={"results": []})

        mock_client(client, handler)
        await client.list_videos(video_ids=["v1", "v2"])

    async def test_get_video(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "GET"
            assert str(req.url).endswith("/v2/videos/vid-123")
            return httpx.Response(
                200,
                json={
                    "video_id": "vid-123",
                    "status": "uploaded",
                    "metadata": {"duration": 120.5},
                    "features": {"transcript": "ready"},
                },
            )

        mock_client(client, handler)
        result = await client.get_video("vid-123")
        assert result.video_id == "vid-123"
        assert result.metadata is not None
        assert result.metadata.duration == 120.5

    async def test_delete_video(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "DELETE"
            assert str(req.url).endswith("/v2/videos/vid-123")
            return httpx.Response(
                200, json={"status": "success", "message": "Video deleted successfully"}
            )

        mock_client(client, handler)
        result = await client.delete_video("vid-123")
        assert result.status == "success"


class TestUploadFile:
    async def test_upload_file_posts_multipart(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/videos")
            content_type = req.headers.get("content-type", "")
            assert "multipart/form-data" in content_type
            body = req.content.decode("utf-8", errors="replace")
            assert b"fake video bytes" in req.content
            assert "video_name" in body
            return httpx.Response(202, json={"video_id": "vid-file", "status": "uploading"})

        mock_client(client, handler)
        result = await client.upload_file(
            file_content=b"fake video bytes",
            filename="clip.mp4",
            name="my clip",
        )
        assert result.video_id == "vid-file"

    async def test_upload_file_sends_optional_fields(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            body = req.content.decode("utf-8", errors="replace")
            assert "description" in body
            assert "group_id" in body
            return httpx.Response(202, json={"video_id": "vid-2", "status": "uploading"})

        mock_client(client, handler)
        await client.upload_file(
            file_content=b"data",
            filename="v.mp4",
            name="n",
            description="d",
            group_id="g1",
        )


class TestGroupMethods:
    async def test_create_group(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert str(req.url).endswith("/v2/video-groups")
            body = req.content.decode()
            assert "test-group" in body
            return httpx.Response(
                200, json={"group_id": "g1", "name": "test-group", "metadata": {}}
            )

        mock_client(client, handler)
        result = await client.create_group("test-group")
        assert result.group_id == "g1"

    async def test_list_groups(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={"results": [{"group_id": "g1", "name": "grp", "metadata": {}}]},
            ),
        )
        result = await client.list_groups()
        assert len(result) == 1
        assert result[0].group_id == "g1"

    async def test_delete_group(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "DELETE"
            assert str(req.url).endswith("/v2/video-groups/g1")
            return httpx.Response(200, json={"status": "success"})

        mock_client(client, handler)
        result = await client.delete_group("g1")
        assert result.status == "success"

    async def test_list_group_videos(self, client: RekaClient) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "GET"
            assert "/v2/video-groups/g1/videos" in str(req.url)
            return httpx.Response(
                200,
                json={"results": [{"video_id": "v1", "status": "uploaded", "features": {}}]},
            )

        mock_client(client, handler)
        result = await client.list_group_videos("g1")
        assert len(result) == 1


class TestPaginate:
    async def test_single_page(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200, json={"data": [{"id": 1}, {"id": 2}], "next_page_token": None}
            ),
        )
        data = await client.paginate("/v2/some-endpoint")
        assert len(data) == 2

    async def test_multiple_pages(self, client: RekaClient) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    200,
                    json={"data": [{"id": 1}, {"id": 2}], "next_page_token": "page2"},
                )
            return httpx.Response(200, json={"data": [{"id": 3}], "next_page_token": None})

        mock_client(client, handler)
        data = await client.paginate("/v2/some-endpoint")
        assert len(data) == 3
        assert call_count == 2

    async def test_max_items_truncates(self, client: RekaClient) -> None:
        mock_client(
            client,
            lambda req: httpx.Response(
                200,
                json={
                    "data": [{"id": 1}, {"id": 2}, {"id": 3}],
                    "next_page_token": "more",
                },
            ),
        )
        data = await client.paginate("/v2/endpoint", max_items=2)
        assert len(data) == 2

    async def test_passes_page_token(self, client: RekaClient) -> None:
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                assert "page_token" not in str(req.url)
                return httpx.Response(200, json={"data": [{"id": 1}], "next_page_token": "tok2"})
            assert "page_token=tok2" in str(req.url)
            return httpx.Response(200, json={"data": [{"id": 2}], "next_page_token": None})

        mock_client(client, handler)
        await client.paginate("/v2/endpoint")
        assert call_count == 2
