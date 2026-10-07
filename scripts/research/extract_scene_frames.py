# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Extract one traceable camera frame per task family from a matrix video run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    """Hash a video or frame file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Extract midpoint frames only from completed, passing EE PID video probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-families", type=int, default=12)
    args = parser.parse_args()
    if args.expect_families < 1:
        parser.error("--expect-families must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    report = json.loads(args.results_json.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for result in report["results"]:
        task_id = result["task_id"]
        if result["status"] != "passed" or "-Am-EE-Abs-PID-Direct-v0" not in task_id:
            continue
        if not result.get("video_paths"):
            continue
        video = Path(result["video_paths"].split(";", maxsplit=1)[0])
        if not video.is_file():
            raise FileNotFoundError(f"Matrix video is missing: {video}")
        capture = cv2.VideoCapture(str(video))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < 1:
                raise ValueError(f"Matrix video has no frames: {video}")
            frame_number = count // 2
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            okay, frame = capture.read()
            if not okay:
                raise ValueError(f"Could not read midpoint frame {frame_number} from {video}")
        finally:
            capture.release()
        output = args.output_dir / f"{result['family'].lower()}_ee_pid.png"
        if not cv2.imwrite(str(output), frame):
            raise OSError(f"Could not write {output}")
        manifest.append({
            "family": result["family"],
            "task_id": task_id,
            "source_video": str(video),
            "source_video_sha256": sha256(video),
            "frame_number": frame_number,
            "frame_count": count,
            "image": output.name,
            "image_sha256": sha256(output),
        })

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Extracted {len(manifest)} real camera frames to {args.output_dir}")
    if len(manifest) != args.expect_families:
        print(f"Expected {args.expect_families} passing family videos, got {len(manifest)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
