# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage and publish the verified video gallery with standard-library tools.

Use ``--dry-run`` to create a reviewable site under ignored
``outputs/research/pages-<timestamp>``. Use ``--publish`` only after committing
the research branch with a clean worktree. Publishing creates a normal commit
on ``gh-pages`` and pushes it to the configured ``origin`` without checking out
another branch or changing the research worktree. Configure GitHub Pages to
publish from the ``gh-pages`` branch, root directory, before viewing the site.

Only the gallery HTML, documentation previews and their manifest are copied.
Each preview must pass the media release checker. Individual files must remain
smaller than 100,000,000 bytes, and the growing documentation site must remain
smaller than 200,000,000 bytes. These are separate limits: a site's total size
is not the Git single-file limit. The release manifest hashes the payload
excluding itself; the printed artifact digest also includes release.json.
Git output that might contain remote credentials is captured and withheld.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_ARTIFACT_BYTES = 200_000_000
MAX_FILE_BYTES = 100_000_000
MAX_MEDIA_BYTES = 450 * 1024
PAGES_REF = "refs/heads/gh-pages"
ZERO_OID = "0" * 40


def run_git(repo_root: Path, args: list[str], data: bytes | None = None, index: Path | None = None) -> str:
    """Run Git while withholding stderr that may contain configured credentials."""
    environment = os.environ.copy()
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        input=data,
        capture_output=True,
        env=environment,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Git {args[0]} failed (exit {result.returncode}); remote diagnostics withheld")
    return result.stdout.decode("utf-8", errors="replace")


def file_records(directory: Path) -> list[dict[str, str | int]]:
    """Hash every regular file in a staged artifact in deterministic path order."""
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("The Pages artifact must not contain symbolic links")
        if not path.is_file():
            continue
        data = path.read_bytes()
        if len(data) >= MAX_FILE_BYTES:
            raise ValueError("A staged file reaches the 100,000,000-byte limit")
        records.append({
            "path": path.relative_to(directory).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    return records


def stage_site(repo_root: Path, research_commit: str, worktree_clean: bool) -> tuple[Path, dict[str, object]]:
    """Copy only approved documentation assets and record their release provenance."""
    generated_at = datetime.now(timezone.utc)
    stage = repo_root / "outputs" / "research" / f"pages-{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    run_git(repo_root, ["check-ignore", "--", str(stage.relative_to(repo_root))])
    stage.mkdir(parents=True)
    media_dir = repo_root / "usage_assets" / "animations"
    manifest_data = (media_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("The animation manifest must be a nonempty list")
    gallery = repo_root / "usage_assets" / "playback.html"
    if gallery.is_symlink() or not gallery.is_file():
        raise ValueError("The video gallery must be a regular HTML file")
    (stage / "index.html").write_bytes(gallery.read_bytes())
    (stage / ".nojekyll").write_bytes(b"")
    staged_media = stage / "animations"
    staged_media.mkdir()
    (staged_media / "manifest.json").write_bytes(manifest_data)
    attributes = media_dir / ".gitattributes"
    if attributes.exists():
        if attributes.is_symlink() or not attributes.is_file():
            raise ValueError("Documentation attributes must be a regular file")
        (staged_media / ".gitattributes").write_bytes(attributes.read_bytes())
    source_files = set()
    experiments = set()
    for row in manifest:
        source_files.add((row["source"], row["source_sha256"]))
        experiments.add(row.get("experiment_id", row["name"]))
        for suffix in ("gif", "mp4"):
            filename = row[suffix]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Preview filenames must not contain directories")
            path = media_dir / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("Each preview must be a regular file")
            data = path.read_bytes()
            if not 0 < len(data) <= MAX_MEDIA_BYTES:
                raise ValueError("Each preview must be at most 450 KiB")
            if len(data) != row[f"{suffix}_bytes"] or hashlib.sha256(data).hexdigest() != row[f"{suffix}_sha256"]:
                raise ValueError("Preview bytes changed after the media release check")
            (staged_media / filename).write_bytes(data)
    for topic in ("uaquad", "tilting"):
        topic_dir = repo_root / "usage_assets" / topic
        if not (topic_dir / "index.html").exists():
            continue
        staged_topic = stage / topic
        staged_topic.mkdir()
        for filename in (
            "index.html",
            "trials.json",
            "trajectories.json",
            "validation.json",
            "recording_specs.json",
            "animation_specs.json",
            "policy_specs.json",
            "policy_trials.json",
            "training_provenance.json",
        ):
            source = topic_dir / filename
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Missing regular {topic} documentation file: {filename}")
            (staged_topic / filename).write_bytes(source.read_bytes())
    payload = file_records(stage)
    payload_bytes = sum(int(record["bytes"]) for record in payload)
    release = {
        "generated_at_utc": generated_at.isoformat(),
        "research_commit": research_commit,
        "source_worktree_clean": worktree_clean,
        "source_file_count": len(source_files),
        "experiment_count": len(experiments),
        "preview_count": len(manifest),
        "gif_count": len(manifest),
        "mp4_count": len(manifest),
        "source_manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "payload_bytes": payload_bytes,
        "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "payload_files": payload,
        "artifact_file_count": len(payload) + 1,
        "artifact_bytes": payload_bytes,
        "maximum_file_bytes": max(int(record["bytes"]) for record in payload),
    }
    release_path = stage / "release.json"
    for _ in range(10):
        release_data = (json.dumps(release, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        artifact_bytes = payload_bytes + len(release_data)
        maximum_bytes = max(len(release_data), max(int(record["bytes"]) for record in payload))
        if (release["artifact_bytes"], release["maximum_file_bytes"]) == (artifact_bytes, maximum_bytes):
            release_path.write_bytes(release_data)
            break
        release["artifact_bytes"] = artifact_bytes
        release["maximum_file_bytes"] = maximum_bytes
    else:
        raise ValueError("Unable to determine the release manifest byte count")
    artifact = file_records(stage)
    total_bytes = sum(int(record["bytes"]) for record in artifact)
    if total_bytes >= MAX_ARTIFACT_BYTES:
        raise ValueError("The complete Pages artifact must be smaller than 200,000,000 bytes")
    summary = {
        "research_commit": research_commit,
        "stage": stage.relative_to(repo_root).as_posix(),
        "source_files": len(source_files),
        "previews": len(manifest),
        "artifact_files": len(artifact),
        "artifact_bytes": total_bytes,
        "maximum_file_bytes": max(int(record["bytes"]) for record in artifact),
        "artifact_sha256": hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest(),
    }
    return stage, summary


def publish_site(repo_root: Path, stage: Path, research_commit: str) -> str:
    """Push a checked artifact as a regular fast-forward commit on gh-pages."""
    if run_git(repo_root, ["symbolic-ref", "--short", "HEAD"]).strip() != "research":
        raise ValueError("Publishing requires the research branch to be checked out")
    if run_git(repo_root, ["status", "--porcelain"]).strip():
        raise ValueError("Commit all research changes before publishing; the worktree must be clean")
    if run_git(repo_root, ["rev-parse", "HEAD"]).strip() != research_commit:
        raise ValueError("The research commit changed after staging; rerun the publisher")
    worktrees = run_git(repo_root, ["worktree", "list", "--porcelain"])
    if f"branch {PAGES_REF}" in worktrees.splitlines():
        raise ValueError("The gh-pages branch is checked out in another worktree")
    artifact = file_records(stage)
    release = json.loads((stage / "release.json").read_bytes())
    payload = [record for record in artifact if record["path"] != "release.json"]
    total_bytes = sum(int(record["bytes"]) for record in artifact)
    if release["research_commit"] != research_commit or payload != release["payload_files"]:
        raise ValueError("The staged artifact differs from its research release manifest")
    if total_bytes != release["artifact_bytes"] or total_bytes >= MAX_ARTIFACT_BYTES:
        raise ValueError("The Pages artifact byte count is invalid or reaches the 200,000,000-byte site limit")
    remote_lines = run_git(repo_root, ["ls-remote", "--heads", "origin", PAGES_REF]).splitlines()
    remote_commit = remote_lines[0].split()[0] if remote_lines else None
    local_lines = run_git(repo_root, ["for-each-ref", "--format=%(refname) %(objectname)", PAGES_REF]).splitlines()
    local_refs = dict(line.split() for line in local_lines)
    local_commit = local_refs.get(PAGES_REF)
    if local_commit is not None and local_commit != remote_commit:
        raise ValueError("Local gh-pages differs from origin; reconcile that branch before publishing")
    if remote_commit is not None:
        run_git(repo_root, ["fetch", "--no-tags", "origin", PAGES_REF])
    with tempfile.TemporaryDirectory(prefix="ambench-pages-index-") as temporary:
        index = Path(temporary) / "index"
        run_git(repo_root, ["read-tree", "--empty"], index=index)
        entries = []
        for record in artifact:
            data = (stage / str(record["path"])).read_bytes()
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("A staged artifact changed during publication")
            object_id = run_git(repo_root, ["hash-object", "-w", "--stdin"], data=data).strip()
            if int(run_git(repo_root, ["cat-file", "-s", object_id])) != len(data):
                raise ValueError("A Pages Git blob has the wrong byte count")
            entries.append(f"100644 {object_id}\t{record['path']}\n")
        run_git(repo_root, ["update-index", "--index-info"], data="".join(entries).encode(), index=index)
        tree = run_git(repo_root, ["write-tree"], index=index).strip()
        args = ["commit-tree", tree]
        if remote_commit is not None:
            args.extend(["-p", remote_commit])
        message = f"Publish research video gallery from {research_commit}\n"
        commit = run_git(repo_root, args, data=message.encode()).strip()
    if run_git(repo_root, ["status", "--porcelain"]).strip():
        raise ValueError("The research worktree changed during preparation; nothing was pushed")
    if run_git(repo_root, ["rev-parse", "HEAD"]).strip() != research_commit:
        raise ValueError("The research commit changed during preparation; nothing was pushed")
    run_git(repo_root, ["-c", "core.hooksPath=/dev/null", "push", "origin", f"{commit}:{PAGES_REF}"])
    run_git(repo_root, ["update-ref", PAGES_REF, commit, local_commit or ZERO_OID])
    return commit


def main() -> int:
    """Check media, stage a static site and optionally push the reviewed artifact."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Check and stage only; never contact or modify GitHub")
    mode.add_argument("--publish", action="store_true", help="Stage and push gh-pages from committed, clean research")
    args = parser.parse_args()
    try:
        check = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/research/check_media_release.py"), "--history-ref", "all"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        print(check.stdout, end="", flush=True)
        if check.returncode:
            print(check.stderr, end="", file=sys.stderr)
            raise ValueError("Media release checks failed; the Pages artifact was not staged")
        research_commit = run_git(REPO_ROOT, ["rev-parse", "refs/heads/research"]).strip()
        worktree_clean = not run_git(REPO_ROOT, ["status", "--porcelain"]).strip()
        if args.publish and (
            not worktree_clean or run_git(REPO_ROOT, ["symbolic-ref", "--short", "HEAD"]).strip() != "research"
        ):
            raise ValueError("Publishing requires committed research changes and a clean worktree")
        stage, summary = stage_site(REPO_ROOT, research_commit, worktree_clean)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if args.publish:
            commit = publish_site(REPO_ROOT, stage, research_commit)
            print(f"Published gh-pages commit {commit}; verify the configured GitHub Pages build and live videos")
        else:
            print("Dry run complete; the staged site is ready for local review")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
