# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Upload verified experiment media as Release assets without adding Git blobs.

The default is a local dry run. ``--upload`` creates or resumes a draft release;
``--publish`` also publishes it after every asset matches its expected digest.
Authentication uses GITHUB_TOKEN. Existing assets with different bytes are
rejected rather than replaced. The upload plan stays in an ignored output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = "https://api.github.com"


def sha256(path: Path) -> str:
    """Hash an asset without holding a complete original recording in memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def request_json(
    url: str, token: str, method: str = "GET", data: bytes | None = None, content_type: str = "application/json"
) -> dict | list:
    """Call GitHub without printing authentication headers or remote diagnostics."""
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": content_type,
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def verify_tag(endpoint: str, tag: str, commit: str, token: str, required: bool = False) -> None:
    """Resolve lightweight or annotated tags to their actual commit identity."""
    try:
        reference = request_json(f"{endpoint}/git/ref/tags/{tag}", token)
    except urllib.error.HTTPError as error:
        if error.code == 404 and not required:
            return
        raise
    target = reference["object"]
    for _ in range(8):
        if target["type"] == "commit":
            if target["sha"] != commit:
                raise ValueError("Existing release tag points to a different commit")
            return
        if target["type"] != "tag":
            raise ValueError("Release tag does not resolve to a commit")
        target = request_json(f"{endpoint}/git/tags/{target['sha']}", token)["object"]
    raise ValueError("Release tag nesting exceeds the verification limit")


def release_assets(endpoint: str, release_id: int, token: str) -> dict[str, dict]:
    """Read the complete paginated inventory before and after publication."""
    assets = {}
    page = 1
    while True:
        rows = request_json(f"{endpoint}/releases/{release_id}/assets?per_page=100&page={page}", token)
        assets.update({row["name"]: row for row in rows})
        if len(rows) < 100:
            return assets
        page += 1


def main() -> int:
    """Validate a plan and optionally upload and publish its exact assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--target-commit", required=True)
    parser.add_argument("--notes-file", type=Path)
    parser.add_argument("--receipt", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--upload", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    repository = plan.get("repository", "superboySB/ambench")
    tag = plan.get("tag", plan.get("release_tag"))
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not tag or not re.fullmatch(r"[\w.-]+", tag):
        parser.error("Invalid repository or release tag")
    if not re.fullmatch(r"[0-9a-f]{40}", args.target_commit):
        parser.error("--target-commit must be an exact commit SHA")
    assets = plan["assets"]
    if not 0 < len(assets) <= 1000:
        parser.error("A release needs 1..1000 assets")
    names = set()
    for asset in assets:
        name = asset.get("name", asset.get("asset_name"))
        if not name or Path(name).name != name or name in names:
            parser.error("Asset filenames must be unique basenames")
        names.add(name)
        path = Path(asset["path"])
        path = path if path.is_absolute() else REPO_ROOT / path
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size < 2 * 1024**3:
            parser.error(f"Missing, empty, symlink or oversized asset: {name}")
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            parser.error(f"Asset changed after validation: {name}")
        if path.suffix.lower() in {".mp4", ".gif"}:
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", "--", str(path)], cwd=REPO_ROOT, capture_output=True, check=False
            )
            if ignored.returncode:
                parser.error(f"Generated media must stay ignored: {name}")
        asset["name"], asset["path"] = name, str(path)
    print(f"Verified {len(assets)} assets, {sum(a['bytes'] for a in assets)} bytes", flush=True)
    if not args.upload and not args.publish:
        print("Dry run complete; no remote changes")
        return 0
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("Set GITHUB_TOKEN for Release uploads")
    endpoint = f"{API_ROOT}/repos/{repository}"
    verify_tag(endpoint, tag, args.target_commit, token)
    try:
        release = request_json(f"{endpoint}/releases/tags/{tag}", token)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        releases = request_json(f"{endpoint}/releases?per_page=100", token)
        release = next((row for row in releases if row["tag_name"] == tag), None)
    if release is None:
        payload = {
            "tag_name": tag,
            "target_commitish": args.target_commit,
            "name": plan.get("title", "UAQuad PressButton: trained ACT / DP recordings"),
            "body": (
                args.notes_file.read_text()
                if args.notes_file
                else "Independent first-attempt evaluation: ACT 0/30, DP 30/30. See manifest for source and protocol."
            ),
            "draft": True,
            "prerelease": True,
        }
        release = request_json(f"{endpoint}/releases", token, "POST", json.dumps(payload).encode())
    if release["target_commitish"] != args.target_commit:
        parser.error("Existing release targets a different source commit")
    existing = release_assets(endpoint, release["id"], token)
    if set(existing) - names:
        parser.error("Release contains assets absent from this plan")
    if not release["draft"] and set(existing) != names:
        parser.error("Published releases are immutable; missing assets will not be appended")

    def upload(asset: dict) -> dict:
        name = asset["name"]
        row = existing.get(name)
        if row is None:
            url = release["upload_url"].split("{", 1)[0] + "?" + urllib.parse.urlencode({"name": name})
            row = request_json(
                url,
                token,
                "POST",
                Path(asset["path"]).read_bytes(),
                asset.get("content_type") or mimetypes.guess_type(name)[0] or "application/octet-stream",
            )
        if row["size"] != asset["bytes"] or row.get("digest") != "sha256:" + asset["sha256"]:
            raise ValueError(f"Release asset digest mismatch: {name}")
        print(f"Verified uploaded asset {name}", flush=True)
        return {"name": name, "bytes": row["size"], "sha256": asset["sha256"], "url": row["browser_download_url"]}

    with ThreadPoolExecutor(max_workers=3) as executor:
        uploaded = list(executor.map(upload, assets))
    if args.publish and release["draft"]:
        verify_tag(endpoint, tag, args.target_commit, token)
        release = request_json(
            f"{endpoint}/releases/{release['id']}",
            token,
            "PATCH",
            json.dumps({"draft": False, "make_latest": "false"}).encode(),
        )
    if not release["draft"]:
        verify_tag(endpoint, tag, args.target_commit, token, required=True)
        # Upload responses from a draft contain temporary untagged URLs.
        # Read the published assets again before recording public links.
        published_assets = release_assets(endpoint, release["id"], token)
        if set(published_assets) != names:
            raise ValueError("Published asset inventory differs from the upload plan")
        for asset in uploaded:
            row = published_assets[asset["name"]]
            if row["size"] != asset["bytes"] or row.get("digest") != "sha256:" + asset["sha256"]:
                raise ValueError(f"Published asset digest mismatch: {asset['name']}")
            url = row["browser_download_url"]
            if not url.startswith(f"https://github.com/{repository}/releases/download/{tag}/"):
                raise ValueError("Published asset still has a temporary draft URL")
            asset["url"] = url
    receipt = {
        "repository": repository,
        "tag": tag,
        "target_commit": args.target_commit,
        "published": not release["draft"],
        "release_url": release["html_url"],
        "assets": uploaded,
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Release {'published' if receipt['published'] else 'draft'}: {release['html_url']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"GitHub API failed (HTTP {error.code}); response withheld") from None
