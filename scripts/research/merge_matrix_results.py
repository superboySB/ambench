# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Merge serial environment matrix reports while retaining every attempt."""

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
    """Parse source reports and the output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="Matrix results.json files in run order.")
    parser.add_argument("--output", type=Path, required=True, help="New merged JSON file.")
    parser.add_argument("--min-steps", type=int, default=8, help="Required steps for a passed probe.")
    parser.add_argument("--expect-count", type=int, default=None)
    parser.add_argument("--expect-families", type=int, default=None)
    args = parser.parse_args()
    if args.min_steps < 1:
        parser.error("--min-steps must be positive")
    reports = [path.resolve() for path in args.reports]
    if len(set(reports)) != len(reports):
        parser.error("the same source report was supplied more than once")
    if args.output.resolve() in reports:
        parser.error("merged output cannot overwrite a source report")
    return args


def resolve_artifact(value: str, report_path: Path) -> Path | None:
    """Resolve a Docker artifact path when merging from either host or container."""
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


def load_report(path: Path) -> tuple[dict[str, Any], str]:
    """Read one report and its SHA-256 digest."""
    contents = path.read_bytes()
    report = json.loads(contents)
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError(f"Invalid matrix report: {path}")
    if report.get("completed_probe_count") != len(report["results"]):
        raise ValueError(f"Completed probe count does not match results: {path}")
    task_ids = [row.get("task_id") for row in report["results"]]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Duplicate task ID within one report: {path}")
    return report, hashlib.sha256(contents).hexdigest()


def merge_reports(args: argparse.Namespace) -> dict[str, Any]:
    """Validate child records and select the last successful attempt per ID."""
    attempts_by_task: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    registry_counts: set[int] = set()

    for report_arg in args.reports:
        report_path = report_arg.resolve()
        report, digest = load_report(report_path)
        registry_counts.add(report["registry_count"])
        sources.append({
            "path": str(report_path),
            "sha256": digest,
            "generated_at_utc": report.get("generated_at_utc"),
            "completed_probe_count": report["completed_probe_count"],
            "passed_count": report.get("passed_count"),
            "failed_count": report.get("failed_count"),
        })
        for row in report["results"]:
            task_id = row["task_id"]
            record_path = resolve_artifact(row.get("result_path", ""), report_path)
            log_path = resolve_artifact(row.get("log_path", ""), report_path)
            if log_path is None or not log_path.is_file():
                raise ValueError(f"Missing log for {task_id}: {log_path}")
            if row["status"] == "passed" and (record_path is None or not record_path.is_file()):
                raise ValueError(f"Missing passed child record for {task_id}: {record_path}")
            if record_path is not None and record_path.is_file():
                child = json.loads(record_path.read_text())
                for key in ("task_id", "status", "steps_completed"):
                    if key == "status" and row["status"] == "timeout" and child.get(key) == "failed":
                        # The parent observed a teardown timeout after the child saved its failure.
                        continue
                    if child.get(key) != row.get(key):
                        raise ValueError(f"{key} differs between report and child record: {task_id}")
            if row["status"] == "passed":
                if row.get("exit_code") != 0 or row.get("steps_completed", 0) < args.min_steps:
                    raise ValueError(f"Passed probe lacks a successful exit or {args.min_steps} steps: {task_id}")
            attempt = {
                "status": row["status"],
                "stage": row.get("stage"),
                "steps_completed": row.get("steps_completed"),
                "duration_s": row.get("duration_s"),
                "seed": row.get("seed", report.get("seed")),
                "exit_code": row.get("exit_code"),
                "error": row.get("error", ""),
                "error_type": row.get("error_type", ""),
                "source_report": str(report_path),
                "source_report_generated_at_utc": report.get("generated_at_utc"),
                "source_record": str(record_path) if record_path and record_path.is_file() else None,
                "source_log": str(log_path),
                "video_paths": row.get("video_paths", ""),
            }
            task_attempts = attempts_by_task.setdefault(task_id, [])
            if task_attempts and (
                task_attempts[0]["family"] != row["family"] or task_attempts[0]["robot"] != row["robot"]
            ):
                raise ValueError(f"Family or robot differs across attempts: {task_id}")
            attempt["family"] = row["family"]
            attempt["robot"] = row["robot"]
            task_attempts.append(attempt)

    if len(registry_counts) != 1:
        raise ValueError(f"Registry size differs across reports: {sorted(registry_counts)}")
    results = []
    for task_id, attempts in sorted(attempts_by_task.items()):
        selected = next((attempt for attempt in reversed(attempts) if attempt["status"] == "passed"), attempts[-1])
        results.append({
            "task_id": task_id,
            "family": selected["family"],
            "robot": selected["robot"],
            "status": selected["status"],
            "steps_completed": selected["steps_completed"],
            "seed": selected["seed"],
            "selected_source_report": selected["source_report"],
            "selected_source_record": selected["source_record"],
            "selected_source_log": selected["source_log"],
            "attempts": attempts,
        })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "min_steps": args.min_steps,
        "registry_count": registry_counts.pop(),
        "task_count": len(results),
        "family_count": len({row["family"] for row in results}),
        "passed_count": sum(row["status"] == "passed" for row in results),
        "failed_count": sum(row["status"] != "passed" for row in results),
        "source_reports": sources,
        "results": results,
    }


def main() -> int:
    """Write the merged report, preserving failures for review."""
    args = parse_args()
    try:
        output = merge_reports(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Cannot merge matrix reports: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Report: {args.output.resolve()}")
    print(f"Passed: {output['passed_count']}/{output['task_count']}; families: {output['family_count']}")
    matches_expected = (args.expect_count is None or output["task_count"] == args.expect_count) and (
        args.expect_families is None or output["family_count"] == args.expect_families
    )
    return 0 if output["failed_count"] == 0 and matches_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
