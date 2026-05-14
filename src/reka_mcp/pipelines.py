# ABOUTME: Domain types for video features, pipelines, and statuses.
# ABOUTME: Maps pipeline names to the set of features each pipeline requires.

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class Feature(StrEnum):
    TRANSCRIPT = "transcript"
    CAPTIONS = "captions"
    EMBEDDINGS = "embeddings"


class FeatureStatus(StrEnum):
    NONE = "none"
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    BLOCKED = "blocked"


Pipeline = Literal["search_only", "qa_only", "full"]

PIPELINE_FEATURES: dict[Pipeline, set[Feature]] = {
    "search_only": {Feature.TRANSCRIPT, Feature.CAPTIONS, Feature.EMBEDDINGS},
    "qa_only": {Feature.TRANSCRIPT, Feature.CAPTIONS},
    "full": {Feature.TRANSCRIPT, Feature.CAPTIONS, Feature.EMBEDDINGS},
}
