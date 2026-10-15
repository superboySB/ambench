# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Check source identity and immutable published media before remote writes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools/research/publish_trained_release.py"
SPEC = importlib.util.spec_from_file_location("publish_trained_release", MODULE_PATH)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
COMMIT = "a" * 40


def test_reject_existing_tag_on_another_commit(monkeypatch) -> None:
    monkeypatch.setattr(publisher, "request_json", lambda *args: {"object": {"type": "commit", "sha": "b" * 40}})
    with pytest.raises(ValueError, match="different commit"):
        publisher.verify_tag("https://api.github.com/repos/example/repo", "media-v1", COMMIT, "test-token")


def test_resolve_annotated_tag_to_commit(monkeypatch) -> None:
    calls = []

    def request(url: str, token: str) -> dict:
        calls.append(url)
        return (
            {"object": {"type": "tag", "sha": "c" * 40}}
            if len(calls) == 1
            else {"object": {"type": "commit", "sha": COMMIT}}
        )

    monkeypatch.setattr(publisher, "request_json", request)
    publisher.verify_tag("https://api.github.com/repos/example/repo", "media-v1", COMMIT, "test-token")
    assert len(calls) == 2
    assert calls[-1].endswith("/git/tags/" + "c" * 40)


def test_published_release_missing_asset_stops_without_upload(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "evidence.json"
    source.write_text("{}\n")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({
            "repository": "example/repo",
            "tag": "media-v1",
            "assets": [{
                "name": source.name,
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }],
        })
    )
    calls = []

    def request(url: str, token: str, method: str = "GET", **kwargs) -> dict | list:
        calls.append(method)
        if "/git/ref/tags/" in url:
            return {"object": {"type": "commit", "sha": COMMIT}}
        if "/assets?" in url:
            return []
        return {"id": 1, "target_commitish": COMMIT, "draft": False}

    monkeypatch.setattr(publisher, "request_json", request)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(sys, "argv", [str(MODULE_PATH), "--plan", str(plan), "--target-commit", COMMIT, "--upload"])
    with pytest.raises(SystemExit) as error:
        publisher.main()
    assert error.value.code == 2
    assert calls and set(calls) == {"GET"}
