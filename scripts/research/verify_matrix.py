# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Probe the live AM-Bench environment registry one Isaac Sim process at a time."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = Path(__file__).with_name("verify_environment.py")
CSV_FIELDS = (
    "task_id",
    "family",
    "robot",
    "status",
    "stage",
    "steps_completed",
    "duration_s",
    "seed",
    "exit_code",
    "error_type",
    "error",
    "log_path",
    "result_path",
    "video_paths",
)


def parse_args() -> argparse.Namespace:
    """Parse filtering, runtime, and artifact options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Run directory; defaults to outputs/research/<UTC timestamp>."
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Save and check the live registry without probing scenes."
    )
    parser.add_argument("--all", action="store_true", help="Probe all live registry IDs (the default selection).")
    parser.add_argument("--family", action="append", default=[], help="Restrict to a task family; repeatable.")
    parser.add_argument("--robot", action="append", default=[], help="Restrict to a robot tag; repeatable.")
    parser.add_argument("--task-id", action="append", default=[], help="Restrict to an exact Gym ID; repeatable.")
    parser.add_argument("--steps", type=int, default=8, help="Number of steps per ordinary probe.")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel scene instances per probe.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to each environment probe.")
    parser.add_argument("--timeout-s", type=float, default=180, help="Wall-clock limit for each probe.")
    parser.add_argument(
        "--registry-timeout-s", type=float, default=300, help="Wall-clock limit for registry enumeration."
    )
    parser.add_argument("--device", default="cuda:0", help="Isaac Sim device passed to AppLauncher.")
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--disturbance", action="store_true", help="Enable saturation, aerodynamics, and wind.")
    parser.add_argument("--action-noise", action="store_true", help="Enable default action noise.")
    parser.add_argument("--observation-noise", action="store_true", help="Enable default observation noise.")
    parser.add_argument(
        "--wind-force", type=float, nargs=3, metavar=("X", "Y", "Z"), help="World-frame wind force in newtons."
    )
    parser.add_argument(
        "--video-family-representatives",
        action="store_true",
        help="Save an EE-camera MP4 for one ID per selected family.",
    )
    parser.add_argument(
        "--video-task-id", action="append", default=[], help="Save an MP4 for an exact selected ID; repeatable."
    )
    parser.add_argument("--video-steps", type=int, default=60, help="Steps for probes selected for video.")
    parser.add_argument("--video-camera-name", default="ee_camera", help="Camera to record for selected video probes.")
    parser.add_argument("--expect-count", type=int, default=106, help="Expected full registry size.")
    parser.add_argument("--expect-families", type=int, default=12, help="Expected family count.")
    args = parser.parse_args()
    if args.steps < 1 or args.video_steps < 1 or args.num_envs < 1:
        parser.error("--steps, --video-steps, and --num-envs must be positive")
    if args.timeout_s <= 0 or args.registry_timeout_s <= 0:
        parser.error("--timeout-s and --registry-timeout-s must be positive")
    if args.expect_count < 1 or args.expect_families < 1:
        parser.error("expected registry counts must be positive")
    if args.all and (args.family or args.robot or args.task_id):
        parser.error("--all cannot be combined with --family, --robot, or --task-id")
    if args.wind_force is not None and not args.disturbance:
        parser.error("--wind-force requires --disturbance")
    return args


def safe_name(task_id: str) -> str:
    """Turn a Gym ID into a safe artifact directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)


def run_child(command: list[str], *, log_path: Path, timeout_s: float) -> tuple[int | None, bool, float]:
    """Run one isolated process and kill its whole process group on completion."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("w") as log_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            print(f"Failed to start child process: {error}", file=log_file)
            return None, False, round(time.monotonic() - started, 3)
        try:
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGINT)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            exit_code = process.poll()
        finally:
            # New session ensures this cannot signal the matrix runner or other jobs.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return exit_code, timed_out, round(time.monotonic() - started, 3)


def save_results(
    output_dir: Path,
    *,
    registry: list[dict[str, str]],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    selected: list[dict[str, str]] | None = None,
) -> None:
    """Persist the current report after each probe so interrupted runs remain useful."""
    full_families = sorted({item["family"] for item in registry})
    selected_ids = [item["task_id"] for item in results]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_count": len(registry),
        "registry_families": full_families,
        "expected_count": args.expect_count,
        "expected_families": args.expect_families,
        "inventory_matches_expected": len(registry) == args.expect_count and len(full_families) == args.expect_families,
        "completed_probe_count": len(results),
        "selected_probe_count": len(selected or []),
        "selected_task_ids": [item["task_id"] for item in selected or []],
        "passed_count": sum(item["status"] == "passed" for item in results),
        "failed_count": sum(item["status"] != "passed" for item in results),
        "completed_task_ids": selected_ids,
        "probe_config": {
            "steps": args.steps,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "timeout_s": args.timeout_s,
            "device": args.device,
            "disable_fabric": args.disable_fabric,
            "disturbance": args.disturbance,
            "action_noise": args.action_noise,
            "observation_noise": args.observation_noise,
            "wind_force": args.wind_force,
            "video_family_representatives": args.video_family_representatives,
            "video_task_ids": args.video_task_id,
            "video_steps": args.video_steps,
            "video_camera_name": args.video_camera_name,
        },
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "results.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in CSV_FIELDS})


def main() -> int:
    """Enumerate the live registry and run selected bounded probes sequentially."""
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or REPO_ROOT / "outputs" / "research" / timestamp).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Output directory is not empty: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "registry.json"
    registry_log = output_dir / "logs" / "registry.log"
    command = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--list-json",
        str(registry_path),
        "--headless",
        "--device",
        args.device,
    ]
    exit_code, timed_out, _ = run_child(command, log_path=registry_log, timeout_s=args.registry_timeout_s)
    if timed_out or exit_code != 0 or not registry_path.is_file():
        print(
            f"Registry enumeration failed (exit={exit_code}, timeout={timed_out}). Log: {registry_log}",
            file=sys.stderr,
        )
        return 2

    try:
        registry: list[dict[str, str]] = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"Invalid registry JSON ({error}); log: {registry_log}", file=sys.stderr)
        return 2
    families = sorted({item["family"] for item in registry})
    inventory_ok = len(registry) == args.expect_count and len(families) == args.expect_families
    print(
        f"Live registry: {len(registry)} IDs across {len(families)} families; inventory expected: {inventory_ok}",
        flush=True,
    )
    if args.list_only:
        save_results(output_dir, registry=registry, results=[], args=args)
        print(f"Registry: {registry_path}")
        return 0 if inventory_ok else 1

    known_ids = {item["task_id"] for item in registry}
    unknown_ids = sorted((set(args.task_id) | set(args.video_task_id)) - known_ids)
    if unknown_ids:
        print(f"Unknown task IDs: {unknown_ids}", file=sys.stderr)
        return 2
    selected = [
        item
        for item in registry
        if (not args.family or item["family"] in args.family)
        and (not args.robot or item["robot"] in args.robot)
        and (not args.task_id or item["task_id"] in args.task_id)
    ]
    if not selected:
        print("No registered IDs matched the requested filters.", file=sys.stderr)
        return 2
    selected_ids = {item["task_id"] for item in selected}
    if not set(args.video_task_id).issubset(selected_ids):
        print("Every --video-task-id must also match the selected probe filters.", file=sys.stderr)
        return 2

    video_ids = set(args.video_task_id)
    if args.video_family_representatives:
        for family in {item["family"] for item in selected}:
            candidates = [item["task_id"] for item in selected if item["family"] == family]
            preferred = f"{family}-Am-EE-Abs-PID-Direct-v0"
            video_ids.add(preferred if preferred in candidates else candidates[0])

    results: list[dict[str, Any]] = []
    save_results(output_dir, registry=registry, results=results, args=args, selected=selected)
    for index, item in enumerate(selected, start=1):
        task_id = item["task_id"]
        name = safe_name(task_id)
        log_path = output_dir / "logs" / f"{name}.log"
        result_path = output_dir / "records" / f"{name}.json"
        video_dir = output_dir / "videos" / name if task_id in video_ids else None
        command = [
            sys.executable,
            str(PROBE_SCRIPT),
            "--task",
            task_id,
            "--result-json",
            str(result_path),
            "--steps",
            str(args.video_steps if video_dir is not None else args.steps),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--headless",
            "--device",
            args.device,
        ]
        if args.disable_fabric:
            command.append("--disable-fabric")
        for enabled, flag in (
            (args.disturbance, "--disturbance"),
            (args.action_noise, "--action-noise"),
            (args.observation_noise, "--observation-noise"),
        ):
            if enabled:
                command.append(flag)
        if args.wind_force is not None:
            command.extend(["--wind-force", *(str(value) for value in args.wind_force)])
        if video_dir is not None:
            command.extend(["--video-dir", str(video_dir), "--video-camera-name", args.video_camera_name])
        print(f"[{index}/{len(selected)}] {task_id}", flush=True)
        exit_code, timed_out, duration_s = run_child(command, log_path=log_path, timeout_s=args.timeout_s)
        try:
            child_result: dict[str, Any] = json.loads(result_path.read_text()) if result_path.is_file() else {}
        except (OSError, json.JSONDecodeError) as error:
            child_result = {"status": "failed", "stage": "result", "error": f"Invalid probe JSON: {error}"}
        if timed_out:
            status = "timeout"
        elif exit_code == 0 and child_result.get("status") == "passed":
            status = "passed"
        else:
            status = "failed"
        record: dict[str, Any] = {
            "task_id": task_id,
            "family": item["family"],
            "robot": item["robot"],
            "status": status,
            "stage": child_result.get("stage", "unknown"),
            "steps_completed": child_result.get("steps_completed", 0),
            "duration_s": duration_s,
            "seed": child_result.get("seed", args.seed),
            "exit_code": exit_code,
            "error_type": child_result.get("error_type", ""),
            "error": (
                "" if status == "passed" else child_result.get("error") or ("Timed out." if timed_out else "See log.")
            ),
            "log_path": str(log_path),
            "result_path": str(result_path) if result_path.is_file() else "",
            "video_paths": ";".join(child_result.get("videos", [])),
        }
        results.append(record)
        save_results(output_dir, registry=registry, results=results, args=args, selected=selected)
        print(f"  {status}: {record['steps_completed']} steps; log: {log_path}", flush=True)

    print(f"Report: {output_dir / 'results.json'}", flush=True)
    return 0 if inventory_ok and all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
