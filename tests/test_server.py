# ABOUTME: Tests for MCP server setup and tool registration.
# ABOUTME: Verifies server creation, tool listing, and config validation.

from __future__ import annotations

import pytest

from reka_mcp.server import create_server


class TestCreateServer:
    async def test_server_registers_all_tools(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = await server.list_tools()
        tool_names = sorted(t.name for t in tools)
        expected = sorted(
            [
                "upload_video",
                "list_videos",
                "get_video",
                "delete_video",
                "create_group",
                "list_groups",
                "delete_group",
                "index_video",
                "search_videos",
                "ask_video",
                "get_transcript",
                "get_captions",
                "get_scenes",
                "get_objects",
                "get_feature_catalog",
                "summarize_video",
            ]
        )
        assert tool_names == expected

    async def test_each_tool_has_description(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = await server.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"

    async def test_each_tool_has_input_schema(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = await server.list_tools()
        for tool in tools:
            assert tool.inputSchema, f"Tool {tool.name} has no input schema"

    async def test_each_tool_has_annotations(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = await server.list_tools()
        for tool in tools:
            assert tool.annotations is not None, f"Tool {tool.name} has no annotations"

    async def test_destructive_tools_marked_correctly(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = {t.name: t for t in await server.list_tools()}
        destructive = {"delete_video", "delete_group"}
        read_only = {
            "list_videos",
            "get_video",
            "list_groups",
            "search_videos",
            "get_transcript",
            "get_captions",
            "get_scenes",
            "get_objects",
            "get_feature_catalog",
            "summarize_video",
        }
        for name in destructive:
            assert tools[name].annotations.destructiveHint is True, name
            assert tools[name].annotations.readOnlyHint is not True, name
        for name in read_only:
            assert tools[name].annotations.readOnlyHint is True, name
            assert tools[name].annotations.destructiveHint is not True, name

    async def test_server_version_matches_package(self) -> None:
        from importlib.metadata import version as pkg_version

        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        opts = server._mcp_server.create_initialization_options()
        assert opts.server_version == pkg_version("reka-mcp")


class TestRationale:
    async def test_every_tool_accepts_rationale(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = await server.list_tools()
        for tool in tools:
            props = tool.inputSchema.get("properties", {})
            assert "rationale" in props, f"Tool {tool.name} missing rationale parameter"
            schema = props["rationale"]
            has_string = schema.get("type") == "string" or any(
                alt.get("type") == "string" for alt in schema.get("anyOf", [])
            )
            assert has_string, f"Tool {tool.name} rationale not typed as string"

    async def test_rationale_is_not_required(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = await server.list_tools()
        for tool in tools:
            required = tool.inputSchema.get("required", [])
            assert "rationale" not in required, f"Tool {tool.name} should not require rationale"


class TestAgentGuidance:
    async def test_ask_video_references_workflow_guide(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = {t.name: t for t in await server.list_tools()}
        desc = tools["ask_video"].description
        assert "reka://docs/guide" in desc

    async def test_ask_video_directs_to_other_tools(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = {t.name: t for t in await server.list_tools()}
        desc = tools["ask_video"].description
        assert "search_videos" in desc
        assert "get_objects" in desc
        assert "get_transcript" in desc

    async def test_search_videos_suggests_ask_video_with_timestamps(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = {t.name: t for t in await server.list_tools()}
        desc = tools["search_videos"].description
        assert "ask_video" in desc
        assert "start/end" in desc

    async def test_get_objects_positions_itself_for_counting(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        tools = {t.name: t for t in await server.list_tools()}
        desc = tools["get_objects"].description
        assert "count" in desc.lower()
        assert "tracking" in desc.lower()

    async def test_workflow_guide_resource_exists(self) -> None:
        server = create_server(api_url="http://localhost:8000", api_key="test-key")
        resources = await server.list_resources()
        uris = [str(r.uri) for r in resources]
        assert "reka://docs/guide" in uris


class TestMissingApiKey:
    def test_missing_api_key_raises_on_config_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REKA_VISION_API_KEY", raising=False)
        from reka_mcp.config import load_config

        with pytest.raises(ValueError, match="REKA_VISION_API_KEY"):
            load_config()
