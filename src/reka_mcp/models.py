# ABOUTME: Pydantic response models for the Reka Vision V2 API.
# ABOUTME: Validates and structures JSON responses from the API.

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    width: int | None = None
    height: int | None = None
    avg_fps: float | None = None
    video_name: str | None = None
    title: str | None = None
    duration: float | None = None
    thumbnail: str | None = None
    description: str | None = None
    source: str | None = None


class VideoResponse(BaseModel):
    video_id: str
    url: str | None = None
    status: str | None = None
    metadata: VideoMetadata | None = None
    features: dict[str, str] | None = None
    group_id: str | None = None
    error: str | None = None


class VideoUploadResponse(BaseModel):
    video_id: str
    status: str
    metadata: VideoMetadata | None = None


class DeleteResponse(BaseModel):
    status: str
    message: str | None = None


class VideoGroupResponse(BaseModel):
    group_id: str
    name: str
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class FeatureTriggerResponse(BaseModel):
    video_id: str
    feature: str
    status: str


class FeaturePlanResponse(BaseModel):
    done: bool
    actionable: list[str]
    blocked: list[str] = Field(default_factory=list)
    statuses: dict[str, str]
