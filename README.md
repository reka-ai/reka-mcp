# Reka Vision MCP Server

mcp-name: ai.reka/mcp

MCP server that lets AI agents upload, index, search, and analyze videos through the Reka Vision API. Agents can search across videos using natural language, ask questions about video content with visual analysis, and read processed data like transcripts, captions, scenes, and detected objects.

## Quick Start

```bash
# Run the published local stdio server with your Reka API key
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

The default mode is local stdio. In local mode, `REKA_VISION_API_KEY` is read once
from the process environment and used for all requests.

## Hosted Mode

Hosted mode runs the same MCP tools over Streamable HTTP. It does not use a
process-wide Reka API key. Instead, each MCP HTTP request must include the user's
key in `X-Reka-API-Key`; the server forwards that value to the Reka Vision API as
`x-api-key` for that request only.

Production-style hosted startup:

```bash
REKA_MCP_MODE=hosted \
REKA_MCP_HTTP_HOST=0.0.0.0 \
PORT=8080 \
uv run reka-mcp
```

Endpoints:

- MCP Streamable HTTP: `http://<host>:<port>/mcp`
- Health check: `http://<host>:<port>/health`

Hosted clients must send:

```http
X-Reka-API-Key: your-api-key
```

`REKA_MCP_AUTH_TOKEN` is optional MCP transport auth. When set, HTTP clients must
also send `Authorization: Bearer <token>`.

`index_video` works the same in both modes: it polls the feature DAG until all
requested features are ready (or times out). After `upload_video`, poll
`get_video` until the video status is `uploaded`, then call `index_video`.

### Run Hosted Mode Locally

Hosted mode enables DNS rebinding protection. Its default allowed hosts and
origins are production/staging domains, so override them for localhost testing:

```bash
REKA_MCP_MODE=hosted \
REKA_MCP_TRANSPORT=http \
REKA_MCP_HTTP_HOST=0.0.0.0 \
REKA_MCP_HTTP_PORT=8080 \
PORT=8080 \
REKA_MCP_HTTP_PATH=/mcp \
REKA_MCP_ALLOWED_HOSTS="localhost:*,127.0.0.1:*" \
REKA_MCP_ALLOWED_ORIGINS="http://localhost:*,http://127.0.0.1:*" \
uv run reka-mcp
```

Then connect to `http://localhost:8080/mcp` and configure your MCP client or
inspector to send `X-Reka-API-Key`. Check the server with:

```bash
curl -H "Host: localhost:8080" http://localhost:8080/health
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
| `upload_video` | Upload a video from a URL |
| `list_videos` | List videos in your account or a group |
| `get_video` | Get video details, metadata, and feature status |
| `update_video` | Update a video's name, title, description, or group |
| `delete_video` | Permanently delete a video and all indexed data |
| `create_group` | Create a new video group |
| `list_groups` | List all video groups |
| `delete_group` | Delete a video group |
| `index_video` | Index a video for search/QA/analysis. Waits until all requested features are ready (2-10 min). |
| `search_videos` | Semantic search across indexed videos |
| `ask_video` | Ask questions about video content with visual analysis |
| `get_transcript` | Get transcript as text, segments, or words |
| `get_captions` | Get AI-generated visual descriptions |
| `get_scenes` | Get detected scene boundaries |
| `get_feature_catalog` | List available features and dependencies |
| `summarize_video` | Compact overview of video content and status |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REKA_MCP_MODE` | `local` | Runtime mode: `local` or `hosted` |
| `REKA_VISION_API_KEY` | *(required in local mode)* | API key from https://platform.reka.ai. Not used as the primary auth source in hosted mode. |
| `REKA_VISION_API_URL` | `https://vision-agent.api.reka.ai` | API base URL |
| `REKA_MCP_INDEX_TIMEOUT` | `600` | Max seconds to wait for indexing |
| `REKA_MCP_POLL_INTERVAL` | `5` | Seconds between index status polls |
| `REKA_MCP_TRANSPORT` | `stdio` in local, `http` in hosted | Transport: `stdio` or `http` |
| `REKA_MCP_HTTP_HOST` | `127.0.0.1` in local, `0.0.0.0` in hosted | Host for HTTP transport |
| `REKA_MCP_HTTP_PORT` | `8080` | Port for HTTP transport. In hosted mode, `PORT` takes precedence when set. |
| `REKA_MCP_HTTP_PATH` | `/mcp` | Streamable HTTP endpoint path |
| `REKA_MCP_ALLOWED_HOSTS` | `mcp.reka.ai,mcp-staging.reka.ai` in hosted | Comma-separated allowed HTTP Host values for DNS rebinding protection |
| `REKA_MCP_ALLOWED_ORIGINS` | `https://mcp.reka.ai,https://mcp-staging.reka.ai` in hosted | Comma-separated allowed Origin values |
| `REKA_MCP_AUTH_TOKEN` | *(none)* | Optional bearer token for HTTP transport auth. Clients must send `Authorization: Bearer <token>` when set. |

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
