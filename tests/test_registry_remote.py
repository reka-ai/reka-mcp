# ABOUTME: Tests for hosted remote entry in server.json registry metadata.
# ABOUTME: Verifies server.json has both local packages and hosted remotes.

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_server_json() -> dict:
    return json.loads((PROJECT_ROOT / "server.json").read_text())


class TestRemoteEntry:
    def test_server_json_has_remotes_key(self) -> None:
        data = _load_server_json()
        assert "remotes" in data, "server.json must have a 'remotes' key"

    def test_remote_uses_streamable_http(self) -> None:
        remote = _load_server_json()["remotes"][0]
        assert remote["type"] == "streamable-http"

    def test_remote_url_points_to_mcp_reka_ai(self) -> None:
        remote = _load_server_json()["remotes"][0]
        assert remote["url"] == "https://mcp.reka.ai/mcp"

    def test_remote_requires_x_reka_api_key_header(self) -> None:
        remote = _load_server_json()["remotes"][0]
        headers = remote["headers"]
        api_key_header = next(h for h in headers if h["name"] == "X-Reka-API-Key")
        assert api_key_header["isRequired"] is True
        assert api_key_header["isSecret"] is True

    def test_remote_header_has_description(self) -> None:
        remote = _load_server_json()["remotes"][0]
        api_key_header = next(h for h in remote["headers"] if h["name"] == "X-Reka-API-Key")
        assert api_key_header["description"]


class TestPackagesAndRemotesCoexist:
    def test_server_json_has_both_packages_and_remotes(self) -> None:
        data = _load_server_json()
        assert "packages" in data
        assert "remotes" in data

    def test_local_package_entry_unchanged(self) -> None:
        data = _load_server_json()
        package = data["packages"][0]
        assert package["registryType"] == "pypi"
        assert package["identifier"] == "reka-mcp"
        assert package["transport"]["type"] == "stdio"
