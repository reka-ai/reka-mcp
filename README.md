# Reka Vision MCP Server

mcp-name: ai.reka/mcp

MCP server that lets AI agents upload, index, search, and analyze videos through the Reka Vision API. Agents can search across videos using natural language, ask questions about video content with visual analysis, and read processed data like transcripts, captions, scenes, and detected objects.

## Quick Start

```bash
# Run the published stdio server
uvx reka-mcp
```

```bash
# Run with an API key
REKA_VISION_API_KEY="your-api-key" uvx reka-mcp
```

For MCP clients that need explicit command configuration:

```json
{
  "command": "uvx",
  "args": ["reka-mcp"],
  "env": {
    "REKA_VISION_API_KEY": "your-api-key-here"
  }
}
```

For local development:

```bash
uv sync
uv run pre-commit install
REKA_VISION_API_KEY="test-key" uv run reka-mcp
```

The stdio transport is the default. Use HTTP only when hosting the MCP server:

```bash
REKA_MCP_TRANSPORT=http REKA_MCP_AUTH_TOKEN="your-secret" uv run python -m reka_mcp
```

## Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reka-mcp": {
      "command": "uvx",
      "args": ["reka-mcp"],
      "env": {
        "REKA_VISION_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Cursor Setup

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "reka-mcp": {
      "command": "uvx",
      "args": ["reka-mcp"],
      "env": {
        "REKA_VISION_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Claude Code Setup

```bash
claude mcp add reka-mcp -e REKA_VISION_API_KEY=your-api-key-here -- uvx reka-mcp
```

## Updating

To update to the latest version, clear the cached package and restart your client:

```bash
uv cache clean reka-mcp
```

To check which version you're running:

```bash
uvx reka-mcp --version
```

## Available Tools

| Tool | Description |
|------|-------------|
| `upload_video` | Upload a video from a URL or local file path |
| `list_videos` | List videos in your account or a group |
| `get_video` | Get video details, metadata, and feature status |
| `update_video` | Update a video's name, title, description, or group |
| `delete_video` | Permanently delete a video and all indexed data |
| `create_group` | Create a new video group |
| `list_groups` | List all video groups |
| `delete_group` | Delete a video group |
| `index_video` | Index a video for search/QA/analysis (2-10 min) |
| `search_videos` | Semantic search across indexed videos |
| `ask_video` | Ask questions about video content with visual analysis |
| `get_transcript` | Get transcript as text, segments, or words |
| `get_captions` | Get AI-generated visual descriptions |
| `get_scenes` | Get detected scene boundaries |
| `get_objects` | Get detected objects with bounding boxes |
| `get_feature_catalog` | List available features and dependencies |
| `summarize_video` | Compact overview of video content and status |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REKA_VISION_API_KEY` | *(required)* | API key from https://platform.reka.ai |
| `REKA_VISION_API_URL` | `https://vision-api.reka.ai` | API base URL |
| `REKA_MCP_INDEX_TIMEOUT` | `600` | Max seconds to wait for indexing |
| `REKA_MCP_POLL_INTERVAL` | `5` | Seconds between index status polls |
| `REKA_MCP_TRANSPORT` | `stdio` | Transport: `stdio` or `http` |
| `REKA_MCP_HTTP_PORT` | `8080` | Port for HTTP transport |
| `REKA_MCP_AUTH_TOKEN` | *(none)* | Bearer token for HTTP transport auth. Clients must send `Authorization: Bearer <token>`. Recommended when using HTTP transport. |

## Release Checks

```bash
cd /path/to/reka-mcp
uv build
uv run twine check dist/*
# Publish only after explicit approval:
uv run twine upload dist/*
```

## Example Workflows

### Search and Visual Q&A

```
Agent: search_videos(query="revenue chart")
→ [{video_id: "v1", start: 30.0, end: 35.0, score: 0.95}]

Agent: ask_video(question="What numbers are on the chart?",
                 video_id="v1", start=30.0, end=35.0)
→ {answer: "Q3 revenue of $4.2M, up 32%...", conversation_id: "c1"}

Agent: ask_video(question="What's the percentage change?",
                 conversation_id="c1")
→ {answer: "Revenue increased by 32% quarter-over-quarter..."}
```

### Cross-Video Comparison

```
Agent: search_videos(query="quarterly revenue")
→ [{video_id: "v1", start: 30.0, ...}, {video_id: "v2", start: 120.0, ...}]

Agent: ask_video(question="How do the revenue figures compare?",
                 videos=[
                   {video_id: "v1", start: 30.0, end: 35.0},
                   {video_id: "v2", start: 120.0, end: 125.0}
                 ])
→ {answer: "Video 1 shows Q3 at $4.2M while Video 2 shows Q4 at $5.1M..."}
```

### Video Summary and Transcript Extraction

```
Agent: summarize_video(video_id="v1")
→ {name: "Lecture 3", duration_seconds: 3600, features: {...},
   scene_count: 42, transcript_preview: "Welcome to today's lecture..."}

Agent: get_transcript(video_id="v1", format="segments", start=0, end=60)
→ {data: [{start: 0.0, end: 5.2, text: "Welcome..."}, ...],
   total_count: 12, truncated: false}
```
