# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Merge scripted expert reports and verify their canonical LeRobot sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse source reports and the merged output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="Scripted results.json files in run order.")
    parser.add_argument("--output", type=Path, required=True, help="New merged JSON file.")
    parser.add_argument("--expect-families", type=int, default=None)
    args = parser.parse_args()
    return args


def resolve_artifact(value: str, report_path: Path) -> Path | None:
    """Map container report paths onto the current repository checkout."""
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        container_root = Path("/workspace/ambench")
        if path.is_relative_to(container_root):
            return REPO_ROOT / path.relative_to(container_root)
        return path
    return (report_path.parent / path).resolve()


def merge_reports(args: argparse.Namespace) -> dict[str, Any]:
    """Validate saved sessions and retain every expert attempt."""
    reports = [path.resolve() for path in args.reports]
    if len(set(reports)) != len(reports):
        raise ValueError("The same source report was supplied more than once.")
    if args.output.resolve() in reports:
        raise ValueError("Merged output cannot overwrite a source report.")

    attempts_by_family: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    registry_counts: set[int] = set()
    for report_path in reports:
        contents = report_path.read_bytes()
        report = json.loads(contents)
        if not isinstance(report, dict) or not isinstance(report.get("results"), list):
            raise ValueError(f"Invalid scripted report: {report_path}")
        registry_counts.add(report["registry_count"])
        sources.append({
            "path": str(report_path),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "generated_at_utc": report.get("generated_at_utc"),
            "passed_count": report.get("passed_count"),
            "failed_count": report.get("failed_count"),
        })
        task_ids = [row["task_id"] for row in report["results"]]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"Duplicate task ID within one report: {report_path}")

        for row in report["results"]:
            family = row["family"]
            if not row["task_id"].startswith(f"{family}-Am-"):
                raise ValueError(f"Task ID and family differ: {row['task_id']}")
            record_log = resolve_artifact(row.get("record_log", ""), report_path)
            validation_log = resolve_artifact(row.get("validation_log", ""), report_path)
            session_root = resolve_artifact(row.get("session_root", ""), report_path)
            if record_log is None or not record_log.is_file():
                raise ValueError(f"Missing recorder log for {row['task_id']}: {record_log}")
            attempt: dict[str, Any] = {
                "task_id": row["task_id"],
                "status": row["status"],
                "stage": row.get("stage"),
                "duration_s": row.get("duration_s"),
                "seed": row.get("seed", report.get("seed")),
                "episode_length_s": row.get("episode_length_s"),
                "error": row.get("error", ""),
                "record_exit_code": row.get("record_exit_code"),
                "validation_exit_code": row.get("validation_exit_code"),
                "episodes": row.get("episodes"),
                "source_report": str(report_path),
                "source_report_generated_at_utc": report.get("generated_at_utc"),
                "record_log": str(record_log),
                "validation_log": str(validation_log) if validation_log else None,
                "session_root": str(session_root) if session_root else None,
            }
            if row["status"] == "passed":
                if not report.get("validate") or row.get("record_exit_code") != 0:
                    raise ValueError(f"Passed expert lacks a successful recording and validator: {row['task_id']}")
                if row.get("validation_exit_code") != 0 or validation_log is None or not validation_log.is_file():
                    raise ValueError(f"Missing successful validation log for {row['task_id']}")
                if session_root is None or not session_root.is_dir():
                    raise ValueError(f"Missing canonical session for {row['task_id']}: {session_root}")
                info_path = session_root / "lerobot" / "meta" / "info.json"
                if not info_path.is_file():
                    raise ValueError(f"Missing LeRobot metadata for {row['task_id']}: {info_path}")
                info = json.loads(info_path.read_text())
                if info.get("total_episodes", 0) < 1 or info.get("total_episodes") != row.get("episodes"):
                    raise ValueError(f"Episode count differs from LeRobot metadata: {row['task_id']}")
                if info.get("total_frames", 0) < 1 or info.get("fps", 0) < 1:
                    raise ValueError(f"Invalid LeRobot frame count or FPS: {row['task_id']}")
                ambench = info.get("ambench", {})
                if ambench.get("action_semantics") not in ("ee_absolute", "base_joint_absolute"):
                    raise ValueError(f"Missing canonical action semantics: {row['task_id']}")
                if not ambench.get("state_keys"):
                    raise ValueError(f"Missing canonical state keys: {row['task_id']}")
                cameras = sorted(key for key in info.get("features", {}) if key.startswith("observation.images."))
                if not cameras:
                    raise ValueError(f"Missing canonical camera feature: {row['task_id']}")
                if report.get("video") and not list((session_root / "videos").glob("*.mp4")):
                    raise ValueError(f"Requested expert MP4 is missing: {row['task_id']}")
                attempt.update({
                    "fps": info["fps"],
                    "total_frames": info["total_frames"],
                    "action_semantics": ambench["action_semantics"],
                    "state_keys": ambench["state_keys"],
                    "camera_keys": cameras,
                })
            attempts_by_family.setdefault(family, []).append(attempt)

    if len(registry_counts) != 1:
        raise ValueError(f"Registry size differs across reports: {sorted(registry_counts)}")
    results = []
    for family, attempts in sorted(attempts_by_family.items()):
        selected = next((attempt for attempt in reversed(attempts) if attempt["status"] == "passed"), attempts[-1])
        results.append({"family": family, "status": selected["status"], "selected": selected, "attempts": attempts})
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_count": registry_counts.pop(),
        "family_count": len(results),
        "passed_count": sum(row["status"] == "passed" for row in results),
        "failed_count": sum(row["status"] != "passed" for row in results),
        "source_reports": sources,
        "results": results,
    }


def main() -> int:
    """Write a traceable merged report and report incomplete coverage."""
    args = parse_args()
    try:
        merged = merge_reports(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot merge scripted reports: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(f"Report: {args.output.resolve()}")
    print(f"Passed: {merged['passed_count']}/{merged['family_count']} families")
    matches_expected = args.expect_families is None or merged["family_count"] == args.expect_families
    return 0 if merged["failed_count"] == 0 and matches_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
