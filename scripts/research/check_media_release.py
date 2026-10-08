# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Check Git history and documentation previews using only the standard library.

Run ``python3 scripts/research/check_media_release.py`` from the repository root,
or use the script's absolute path from another directory. ``--history-ref all``
checks all local refs; the default checks history reachable from ``research``. Media
checks read the current worktree, so new previews can be checked before committing.
Original experiment outputs are not needed for this release check. Frame coverage
is checked against the export manifest; media decoding remains the exporter's job.
Additional condition metadata, including experiment_id, env_seed and camera, is
accepted without requiring it in previews exported before those fields existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BLOB_BYTES = 100_000_000
MAX_MEDIA_BYTES = 450 * 1024
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def run_git(repo_root: Path, args: list[str], input_text: str | None = None) -> str:
    """Run a local Git query without exposing stderr that could contain credentials."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Git {args[0]} failed; check the local repository and revision")
    return result.stdout


def check_history(repo_root: Path, history_ref: str) -> tuple[list[str], int, int]:
    """Find oversized blobs anywhere in reachable history, including deleted files."""
    if run_git(repo_root, ["rev-parse", "--is-shallow-repository"]).strip() != "false":
        raise ValueError("History check requires a full clone; fetch with depth 0 before publishing")
    if history_ref == "all":
        revisions = ["--all"]
    else:
        revision = run_git(repo_root, ["rev-parse", "--verify", "--end-of-options", f"{history_ref}^{{commit}}"])
        revisions = [revision.strip()]
    objects = run_git(repo_root, ["rev-list", "--objects", *revisions])
    sizes = run_git(
        repo_root,
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)"],
        input_text=objects,
    )
    errors = []
    blob_count = 0
    largest = 0
    for line in sizes.splitlines():
        fields = line.split(" ", 3)
        if len(fields) < 3 or fields[1] != "blob":
            continue
        size = int(fields[2])
        blob_count += 1
        largest = max(largest, size)
        if size >= MAX_BLOB_BYTES:
            path = fields[3] if len(fields) == 4 else ""
            errors.append(f"oversized Git blob: oid={fields[0]} bytes={size} path={json.dumps(path)}")
    return errors, blob_count, largest


def check_media(repo_root: Path, media_dir: Path) -> tuple[list[str], int, int]:
    """Validate worktree preview files, hashes, Git filters and source frame coverage."""
    media_dir = media_dir.resolve()
    if not media_dir.is_relative_to(repo_root.resolve()):
        raise ValueError("The media directory must be inside the repository")
    manifest = json.loads((media_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("The animation manifest must be a nonempty list")
    errors = []
    names = set()
    expected_files = set()
    file_count = 0
    largest = 0
    for row_index, row in enumerate(manifest):
        if not isinstance(row, dict):
            errors.append(f"manifest row {row_index}: expected an object")
            continue
        name = row.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            errors.append(f"manifest row {row_index}: invalid preview name")
            continue
        if name in names:
            errors.append(f"{name}: duplicate preview name")
        names.add(name)
        for media_type in ("gif", "mp4"):
            filename = row.get(media_type)
            if filename != f"{name}.{media_type}":
                errors.append(f"{name}: {media_type} filename must be {name}.{media_type}")
                continue
            if filename in expected_files:
                errors.append(f"{name}: duplicate media filename {filename}")
            expected_files.add(filename)
            path = media_dir / filename
            if path.is_symlink() or not path.is_file():
                errors.append(f"{filename}: missing regular worktree file")
                continue
            size = path.stat().st_size
            file_count += 1
            largest = max(largest, size)
            if not 0 < size <= MAX_MEDIA_BYTES:
                errors.append(f"{filename}: bytes={size}, expected 1..{MAX_MEDIA_BYTES}")
                continue
            content = path.read_bytes()
            if content.startswith(LFS_PREFIX):
                errors.append(f"{filename}: Git LFS pointer instead of ordinary media bytes")
                continue
            if media_type == "gif" and content[:6] not in (b"GIF87a", b"GIF89a"):
                errors.append(f"{filename}: invalid GIF signature")
            if media_type == "mp4" and content[4:8] != b"ftyp":
                errors.append(f"{filename}: missing MP4 ftyp header")
            declared_size = row.get(f"{media_type}_bytes")
            if type(declared_size) is not int or declared_size != size:
                errors.append(f"{filename}: manifest byte count does not match the file")
            declared_hash = row.get(f"{media_type}_sha256")
            if (
                not isinstance(declared_hash, str)
                or not SHA256_PATTERN.fullmatch(declared_hash)
                or hashlib.sha256(content).hexdigest() != declared_hash
            ):
                errors.append(f"{filename}: manifest SHA256 does not match the file")
        source_count = row.get("source_frame_count")
        if type(source_count) is not int or source_count < 2:
            errors.append(f"{name}: source_frame_count must be an integer of at least 2")
            continue
        for prefix in ("", "mp4_"):
            field = f"{prefix}source_frame_indices"
            indices = row.get(field)
            if not isinstance(indices, list) or len(indices) < 2:
                errors.append(f"{name}: {field} must contain the first and last source frame")
                continue
            if any(type(index) is not int or not 0 <= index < source_count for index in indices):
                errors.append(f"{name}: {field} contains invalid source frame indices")
                continue
            if indices[0] != 0 or indices[-1] != source_count - 1:
                errors.append(f"{name}: {field} does not cover the first and last source frame")
            if any(left > right for left, right in zip(indices, indices[1:])):
                errors.append(f"{name}: {field} is not in source order")
            timestamps = row.get(f"{prefix}source_timestamps_s")
            if (
                not isinstance(timestamps, list)
                or len(timestamps) != len(indices)
                or any(type(value) not in (int, float) or not -1e300 < value < 1e300 for value in timestamps)
            ):
                errors.append(f"{name}: {prefix}source_timestamps_s must match the sampled source frames")
            elif any(left > right for left, right in zip(timestamps, timestamps[1:])):
                errors.append(f"{name}: {prefix}source_timestamps_s is not in source order")
    actual_files = {
        path.relative_to(media_dir).as_posix()
        for path in media_dir.rglob("*")
        if path.suffix.lower() in (".gif", ".mp4")
    }
    for filename in sorted(actual_files - expected_files):
        errors.append(f"unlisted preview file: {filename}")
    for filename in sorted(expected_files - actual_files):
        errors.append(f"manifest preview is absent: {filename}")
    if expected_files:
        paths = [(media_dir / filename).relative_to(repo_root).as_posix() for filename in sorted(expected_files)]
        attributes = run_git(repo_root, ["check-attr", "-z", "filter", "--", *paths]).split("\0")
        for offset in range(0, len(attributes) - 1, 3):
            path, _, value = attributes[offset : offset + 3]
            if value not in ("unset", "unspecified"):
                errors.append(f"{path}: Git filter {value!r} would prevent publishing ordinary media blobs")
    return errors, file_count, largest


def main() -> int:
    """Check the selected Git history and worktree previews, returning a release status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--history-ref", default="research", help="Commit/revision to check, or all for every local ref"
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=Path("usage_assets/animations"),
        help="Preview directory containing manifest.json; relative paths use the repository root",
    )
    args = parser.parse_args()
    media_dir = args.media_dir if args.media_dir.is_absolute() else REPO_ROOT / args.media_dir
    try:
        history_errors, blob_count, largest_blob = check_history(REPO_ROOT, args.history_ref)
        media_errors, file_count, largest_media = check_media(REPO_ROOT, media_dir)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for error in history_errors + media_errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"History: {blob_count} blobs checked; largest={largest_blob} bytes; limit=<{MAX_BLOB_BYTES}")
    print(f"Media: {file_count} files checked; largest={largest_media} bytes; limit<={MAX_MEDIA_BYTES}")
    if history_errors or media_errors:
        return 1
    print("Media release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
