# ABOUTME: Async HTTP client wrapping the Reka Vision V2 API.
# ABOUTME: Handles auth, retry on 5xx, typed exceptions, and pagination.

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from reka_mcp.tools.qa import VideoContext

import httpx

from reka_mcp.models import (
    DeleteResponse,
    FeaturePlanResponse,
    FeatureTriggerResponse,
    VideoGroupResponse,
    VideoResponse,
    VideoUploadResponse,
)
from reka_mcp.pipelines import Feature

logger = logging.getLogger(__name__)

mcp_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_session_id", default=None
)

JsonDict = dict[str, Any]


class ChatResponse(TypedDict):
    response: str
    model: str


class RekaAPIError(Exception):
    pass


class AuthError(RekaAPIError):
    pass


class NotFoundError(RekaAPIError):
    pass


class ConflictError(RekaAPIError):
    pass


class ValidationError(RekaAPIError):
    pass


class ServerError(RekaAPIError):
    pass


class RekaClient:
    _MAX_RETRIES = 3
    _retry_base_delay: float = 1.0

    def __init__(self, api_url: str, api_key: str) -> None:
        self._http = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            headers={"x-api-key": api_key},
            timeout=60.0,
        )

    async def close(self) -> None:
        await self._http.aclose()

    # -- Video operations --

    async def upload_video(
        self,
        *,
        video_url: str,
        name: str | None = None,
        description: str | None = None,
        group_id: str | None = None,
    ) -> VideoUploadResponse:
        data: dict[str, str] = {"video_url": video_url}
        if name is not None:
            data["video_name"] = name
        if description is not None:
            data["description"] = description
        if group_id is not None:
            data["group_id"] = group_id
        return VideoUploadResponse.model_validate(await self._post_form("/v2/videos", data))

    async def upload_file(
        self,
        *,
        file_content: bytes,
        filename: str,
        name: str,
        description: str | None = None,
        group_id: str | None = None,
    ) -> VideoUploadResponse:
        data: dict[str, str] = {"video_name": name}
        if description is not None:
            data["description"] = description
        if group_id is not None:
            data["group_id"] = group_id
        files = {"file": (filename, file_content, "application/octet-stream")}
        return VideoUploadResponse.model_validate(
            await self._request("POST", "/v2/videos", data=data, files=files)
        )

    async def list_videos(self, *, video_ids: list[str] | None = None) -> list[VideoResponse]:
        params: JsonDict = {}
        if video_ids:
            params["ids"] = video_ids
        resp = await self._get("/v2/videos", params=params)
        return [VideoResponse.model_validate(v) for v in resp["results"]]

    async def get_video(self, video_id: str) -> VideoResponse:
        return VideoResponse.model_validate(await self._get(f"/v2/videos/{video_id}"))

    async def update_video(
        self,
        video_id: str,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        group_id: str | None = None,
        group_id_provided: bool = False,
    ) -> VideoResponse:
        body: JsonDict = {}
        if name is not None:
            body["name"] = name
        if title is not None:
            body["title"] = title
        if description is not None:
            body["description"] = description
        if group_id_provided:
            body["group_id"] = group_id
        return VideoResponse.model_validate(
            await self._request("PATCH", f"/v2/videos/{video_id}", json=body)
        )

    async def delete_video(self, video_id: str) -> DeleteResponse:
        return DeleteResponse.model_validate(
            await self._request("DELETE", f"/v2/videos/{video_id}")
        )

    # -- Feature operations --

    async def plan_features(self, video_id: str, desired: list[Feature]) -> FeaturePlanResponse:
        return FeaturePlanResponse.model_validate(
            await self._post_json(
                f"/v2/videos/{video_id}/features/plan",
                {"desired": desired},
            )
        )

    async def trigger_feature(
        self,
        video_id: str,
        feature: Feature,
        force: bool = False,
        body: JsonDict | None = None,
    ) -> FeatureTriggerResponse:
        feature = Feature(feature)
        request_body: JsonDict = {"force": force}
        if body:
            request_body.update(body)
        return FeatureTriggerResponse.model_validate(
            await self._post_json(
                f"/v2/videos/{video_id}/features/{feature}",
                request_body,
            )
        )

    # -- Group operations --

    async def create_group(self, name: str) -> VideoGroupResponse:
        return VideoGroupResponse.model_validate(
            await self._post_json("/v2/video-groups", {"name": name})
        )

    async def list_groups(self) -> list[VideoGroupResponse]:
        resp = await self._get("/v2/video-groups")
        return [VideoGroupResponse.model_validate(g) for g in resp["results"]]

    async def delete_group(self, group_id: str) -> DeleteResponse:
        return DeleteResponse.model_validate(
            await self._request("DELETE", f"/v2/video-groups/{group_id}")
        )

    async def list_group_videos(self, group_id: str) -> list[VideoResponse]:
        resp = await self._get(f"/v2/video-groups/{group_id}/videos")
        return [VideoResponse.model_validate(v) for v in resp["results"]]

    # -- Search --

    async def search_videos(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        group_id: str | None = None,
        max_results: int = 10,
    ) -> list[JsonDict]:
        body: JsonDict = {"query": query, "page_limit": max_results}
        if video_ids is not None:
            body["video_ids"] = video_ids
        if group_id is not None:
            body["group_ids"] = [group_id]
        resp = await self._post_json("/v2/search", body)
        return list(resp.get("data", []))

    # -- QA --

    async def chat(
        self,
        messages: list[dict[str, str]],
        context: list[VideoContext],
    ) -> ChatResponse:
        resp = await self._post_json("/v2/chat", {"messages": messages, "context": context})
        return ChatResponse(response=resp.get("response", ""), model=resp.get("model", ""))

    # -- Segment --

    async def segment_video(
        self,
        video_id: str,
        *,
        prompts: list[str],
        start: float,
        end: float | None = None,
        threshold: float | None = None,
    ) -> JsonDict:
        body: JsonDict = {
            "prompts": [{"type": "text", "text": p} for p in prompts],
            "start": start,
        }
        if end is not None:
            body["end"] = end
        if threshold is not None:
            body["threshold"] = threshold
        return await self._post_json(f"/v2/videos/{video_id}/segment", body)

    # -- Sub-resources --

    _SUB_RESOURCE_PAGE_LIMIT = 50

    async def get_transcript(
        self,
        video_id: str,
        *,
        format: str = "segments",
        start: float | None = None,
        end: float | None = None,
        max_items: int | None = None,
    ) -> JsonDict | list[JsonDict]:
        params = self._time_range_params(start, end, format=format)
        path = f"/v2/videos/{video_id}/transcript"
        if format == "text":
            return await self._get(path, params=params)
        data = await self.paginate(path, params=params, max_items=max_items)
        return data

    async def get_captions(
        self,
        video_id: str,
        *,
        start: float | None = None,
        end: float | None = None,
        max_items: int | None = None,
    ) -> list[JsonDict]:
        data = await self.paginate(
            f"/v2/videos/{video_id}/captions",
            params=self._time_range_params(start, end),
            max_items=max_items,
        )
        return data

    async def get_scenes(self, video_id: str) -> list[JsonDict]:
        data = await self.paginate(
            f"/v2/videos/{video_id}/scenes",
            params=self._time_range_params(),
        )
        return data

    async def get_objects(
        self,
        video_id: str,
        *,
        object_type: str | None = None,
        start: float | None = None,
        end: float | None = None,
        max_items: int | None = None,
    ) -> list[JsonDict]:
        params = self._time_range_params(start, end)
        if object_type is not None:
            params["type"] = object_type
        data = await self.paginate(
            f"/v2/videos/{video_id}/objects", params=params, max_items=max_items
        )
        return data

    def _time_range_params(
        self,
        start: float | None = None,
        end: float | None = None,
        **extra: str,
    ) -> JsonDict:
        params: JsonDict = {"page_limit": self._SUB_RESOURCE_PAGE_LIMIT}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        params.update(extra)
        return params

    async def get_feature_catalog(self) -> JsonDict:
        resp = await self._get("/v2/features")
        return {feature["name"]: feature for feature in resp.get("features", [])}

    # -- Pagination helper --

    async def paginate(
        self,
        path: str,
        params: JsonDict | None = None,
        max_items: int | None = None,
    ) -> list[JsonDict]:
        all_data: list[JsonDict] = []
        current_params = dict(params or {})
        while True:
            resp = await self._get(path, params=current_params)
            all_data.extend(resp.get("data", []))
            if max_items and len(all_data) >= max_items:
                all_data = all_data[:max_items]
                break
            next_token = resp.get("next_page_token")
            if not next_token:
                break
            current_params["page_token"] = next_token
        return all_data

    # -- Internal HTTP helpers --

    async def _get(self, path: str, params: JsonDict | None = None) -> JsonDict:
        return await self._request("GET", path, params=params)

    async def _post_json(self, path: str, body: JsonDict) -> JsonDict:
        return await self._request("POST", path, json=body)

    async def _post_form(self, path: str, data: dict[str, str]) -> JsonDict:
        return await self._request("POST", path, data=data)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: JsonDict | None = None,
        json: JsonDict | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> JsonDict:
        headers: dict[str, str] = {}
        session_id = mcp_session_id_var.get()
        if session_id:
            headers["x-mcp-session-id"] = session_id

        last_exc: Exception | None = None
        t0 = time.monotonic()
        for attempt in range(self._MAX_RETRIES):
            resp = await self._http.request(
                method,
                path,
                params=params,
                json=json,
                data=data,
                files=files,
                headers=headers,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            if resp.status_code < 500:
                logger.debug("%s %s → %d (%.0fms)", method, path, resp.status_code, elapsed_ms)
                self._raise_for_client_error(resp)
                result: JsonDict = resp.json()
                return result
            logger.warning(
                "%s %s → %d (attempt %d/%d, %.0fms)",
                method,
                path,
                resp.status_code,
                attempt + 1,
                self._MAX_RETRIES,
                elapsed_ms,
            )
            last_exc = ServerError("Reka API temporarily unavailable. Try again.")
            if attempt < self._MAX_RETRIES - 1:
                await asyncio.sleep(self._retry_base_delay * (2**attempt))
        raise last_exc  # type: ignore[misc]

    def _raise_for_client_error(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
            message = body.get("error", {}).get("message", resp.text)
        except Exception:
            message = resp.text

        status = resp.status_code
        logger.warning("API error %d: %s", status, message)
        if status in (401, 403):
            raise AuthError("Authentication failed. Check your REKA_VISION_API_KEY.")
        if status == 404:
            raise NotFoundError(message)
        if status == 409:
            raise ConflictError(message)
        if status == 422:
            raise ValidationError(message)
        raise RekaAPIError(f"HTTP {status}: {message}")
