# ABOUTME: Tests for pipeline definitions and validation.
# ABOUTME: Verifies pipeline names and feature sets.

from __future__ import annotations

from reka_mcp.pipelines import PIPELINE_FEATURES


class TestPipelineFeatures:
    def test_search_only_has_expected_features(self) -> None:
        assert PIPELINE_FEATURES["search_only"] == {
            "transcript",
            "captions",
            "embeddings",
        }

    def test_qa_only_has_expected_features(self) -> None:
        assert PIPELINE_FEATURES["qa_only"] == {"transcript", "captions"}

    def test_full_has_expected_features(self) -> None:
        assert PIPELINE_FEATURES["full"] == {
            "transcript",
            "captions",
            "embeddings",
            "objects",
        }

    def test_all_pipeline_features_are_triggerable(self) -> None:
        triggerable = {"transcript", "captions", "embeddings", "objects"}
        for pipeline, features in PIPELINE_FEATURES.items():
            for feat in features:
                assert feat in triggerable, (
                    f"Pipeline '{pipeline}' has non-triggerable feature '{feat}'"
                )
