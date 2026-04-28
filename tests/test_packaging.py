# ABOUTME: Tests for PyPI and MCP registry packaging metadata.
# ABOUTME: Keeps package metadata, README install docs, and server.json in sync.

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def test_pyproject_has_publishable_pypi_metadata() -> None:
    project = read_pyproject()["project"]

    assert project["name"] == "reka-mcp"
    assert re.fullmatch(r"\d+\.\d+\.\d+", project["version"])
    assert project["description"]
    assert project["readme"] == "README.md"
    assert project["requires-python"].startswith(">=3.12")
    assert project["license"]
    assert project["authors"]
    assert project["maintainers"]
    assert "mcp" in project["keywords"]
    assert "Framework :: FastAPI" not in project["classifiers"]
    assert any(c.startswith("Programming Language :: Python :: 3") for c in project["classifiers"])
    assert any(c.startswith("License ::") for c in project["classifiers"])
    assert project["scripts"]["reka-mcp"] == "reka_mcp.server:main"

    urls = project["urls"]
    assert urls["Homepage"]
    assert urls["Repository"]
    assert urls["Issues"]
    assert urls["Documentation"]

    dev_dependencies = read_pyproject()["dependency-groups"]["dev"]
    assert any(dependency.startswith("twine") for dependency in dev_dependencies)


def test_readme_has_registry_marker_and_local_install_docs() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()

    assert "mcp-name: ai.reka/mcp" in readme
    assert "uvx reka-mcp" in readme
    assert 'REKA_VISION_API_KEY="your-api-key" uvx reka-mcp' in readme
    assert '"command": "uvx"' in readme
    assert '"args": ["reka-mcp"]' in readme
    assert 'REKA_VISION_API_KEY": "your-api-key-here"' in readme
    assert "uv sync" in readme
    assert 'REKA_VISION_API_KEY="test-key" uv run reka-mcp' in readme


def test_server_json_matches_project_and_readme_registry_metadata() -> None:
    pyproject = read_pyproject()
    project = pyproject["project"]
    readme = (PROJECT_ROOT / "README.md").read_text()
    server_json = json.loads((PROJECT_ROOT / "server.json").read_text())

    registry_name_match = re.search(r"^mcp-name:\s*(\S+)$", readme, re.MULTILINE)
    assert registry_name_match is not None

    assert server_json["name"] == registry_name_match.group(1)
    assert server_json["version"] == project["version"]

    package = server_json["packages"][0]
    assert package["registryType"] == "pypi"
    assert package["identifier"] == project["name"]
    assert package["version"] == project["version"]
    assert package["transport"]["type"] == "stdio"
