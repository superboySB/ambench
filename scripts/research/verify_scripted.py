# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect and validate one scripted EE-PID demonstration per task family."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_matrix import PROBE_SCRIPT, REPO_ROOT, run_child, safe_name

RECORDER_SCRIPT = REPO_ROOT / "scripts" / "data" / "record_demos_scripted.py"
VALIDATOR_SCRIPT = REPO_ROOT / "scripts" / "data" / "validate_lerobotdataset.py"
PROMPT_MAP = REPO_ROOT / "scripts" / "data" / "am_bench_language_instructions.json"
DEFAULT_EPISODE_LENGTH_S = {
    "CabinetPickPlace": 60,
    "FrameAssembly": 30,
    "LemonHarvesting": 30,
    "NDT": 20,
    "OpenDoor": 15,
    "PegInHole": 20,
    "PressButton": 20,
    "PullLever": 20,
    "PushSlider": 26,
    "RotateValve": 20,
    "TossBall": 10,
    "WipeWindow": 20,
}
CSV_FIELDS = (
    "family",
    "task_id",
    "status",
    "stage",
    "duration_s",
    "seed",
    "episode_length_s",
    "record_exit_code",
    "validation_exit_code",
    "session_root",
    "episodes",
    "record_log",
    "validation_log",
    "error",
)


def parse_args() -> argparse.Namespace:
    """Parse bounded scripted collection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--family", action="append", default=[], help="Only test this family; repeatable.")
    parser.add_argument("--task-id", action="append", default=[], help="Test this exact registered ID; repeatable.")
    parser.add_argument("--timeout-s", type=float, default=600, help="Wall-clock limit per family.")
    parser.add_argument("--registry-timeout-s", type=float, default=300)
    parser.add_argument("--step-hz", type=int, default=120)
    parser.add_argument("--target-hz", type=int, default=20, help="LeRobot validation sampling rate.")
    parser.add_argument("--env-length-s", type=int, default=None, help="Override every task episode length.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to each scripted recorder.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action="store_true", help="Save additional recorder MP4s.")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.timeout_s <= 0 or args.registry_timeout_s <= 0:
        parser.error("timeouts must be positive")
    if args.step_hz < 1 or args.target_hz < 1 or args.step_hz % args.target_hz:
        parser.error("--step-hz must be a positive integer multiple of --target-hz")
    if args.env_length_s is not None and args.env_length_s < 1:
        parser.error("--env-length-s must be positive")
    if args.family and args.task_id:
        parser.error("--family and --task-id cannot be combined")
    return args


def save_report(
    output_dir: Path,
    registry: list[dict[str, str]],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write a durable report after each completed family."""
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_count": len(registry),
        "family_count": len({item["family"] for item in registry}),
        "selected_families": (
            sorted({task_id.split("-Am-")[0] for task_id in args.task_id})
            if args.task_id
            else args.family or sorted({item["family"] for item in registry})
        ),
        "selected_task_ids": args.task_id,
        "step_hz": args.step_hz,
        "target_hz": args.target_hz,
        "timeout_s": args.timeout_s,
        "device": args.device,
        "seed": args.seed,
        "episode_length_s_override": args.env_length_s,
        "video": args.video,
        "validate": args.validate,
        "results": results,
        "passed_count": sum(item["status"] == "passed" for item in results),
        "failed_count": sum(item["status"] != "passed" for item in results),
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "results.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in CSV_FIELDS})


def session_root_from_log(log_path: Path) -> Path | None:
    """Find the finalized session marker printed by the public recorder."""
    matches = re.findall(r"^SESSION_ROOT=(.+)$", log_path.read_text(errors="replace"), flags=re.MULTILINE)
    return Path(matches[-1]).resolve() if matches else None


def main() -> int:
    """Run the public scripted recorder and validator for each selected family."""
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or REPO_ROOT / "outputs" / "research" / f"scripted-{timestamp}").resolve()
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
        print(f"Registry enumeration failed; log: {registry_log}", file=sys.stderr)
        return 2
    try:
        registry: list[dict[str, str]] = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"Invalid registry JSON ({error}); log: {registry_log}", file=sys.stderr)
        return 2
    families = sorted({item["family"] for item in registry})
    unknown_families = sorted(set(args.family) - set(families))
    if unknown_families:
        print(f"Unknown task families: {unknown_families}", file=sys.stderr)
        return 2
    selected_families = [family for family in families if not args.family or family in args.family]
    registered_ids = {item["task_id"] for item in registry}
    unknown_task_ids = sorted(set(args.task_id) - registered_ids)
    if unknown_task_ids:
        print(f"Unknown task IDs: {unknown_task_ids}", file=sys.stderr)
        return 2
    selected_tasks = (
        [(task_id.split("-Am-")[0], task_id) for task_id in args.task_id]
        if args.task_id
        else [(family, f"{family}-Am-EE-Abs-PID-Direct-v0") for family in selected_families]
    )
    prompt_map: dict[str, str] = json.loads(PROMPT_MAP.read_text())
    results: list[dict[str, Any]] = []
    save_report(output_dir, registry, results, args)

    for index, (family, task_id) in enumerate(selected_tasks, start=1):
        record_log = output_dir / "logs" / f"{safe_name(task_id)}.log"
        validation_log = output_dir / "logs" / f"{safe_name(task_id)}-validation.log"
        episode_length_s = args.env_length_s or DEFAULT_EPISODE_LENGTH_S.get(family)
        record: dict[str, Any] = {
            "family": family,
            "task_id": task_id,
            "status": "failed",
            "stage": "registration",
            "duration_s": 0,
            "seed": args.seed,
            "episode_length_s": episode_length_s,
            "record_exit_code": "",
            "validation_exit_code": "",
            "session_root": "",
            "episodes": 0,
            "record_log": str(record_log),
            "validation_log": "",
            "error": "",
        }
        print(f"[{index}/{len(selected_tasks)}] {task_id}", flush=True)
        if task_id not in registered_ids:
            record["error"] = "EE PID task variant is absent from the live registry."
            results.append(record)
            save_report(output_dir, registry, results, args)
            continue
        if family not in DEFAULT_EPISODE_LENGTH_S and args.env_length_s is None:
            record["error"] = "No episode length is known for this family; pass --env-length-s."
            results.append(record)
            save_report(output_dir, registry, results, args)
            continue

        if "-BaseJoint-" in task_id:
            snake_case_family = re.sub(r"(?<!^)(?=[A-Z])", "_", family).lower()
            repo_id = f"am_bench/{snake_case_family}_base_joint_absolute"
            state_keys = ["base_pos", "base_quat", "arm_joint_pos", "gripper_width"]
        else:
            repo_id = f"am_bench/{family.lower()}_ee_absolute"
            state_keys = ["ee_pos", "ee_quat", "gripper_width"]
        prompt = prompt_map.get(f"{family}EEAbsPID", "inspect the marked surface")
        command = [
            sys.executable,
            str(RECORDER_SCRIPT),
            "--task",
            task_id,
            "--dataset_root",
            str(output_dir / "datasets" / family),
            "--repo_id",
            repo_id,
            "--state_keys",
            *state_keys,
            "--task_prompt",
            prompt,
            "--step_hz",
            str(args.step_hz),
            "--num_envs",
            "1",
            "--num_demos",
            "1",
            "--env_length_s",
            str(episode_length_s),
            "--seed",
            str(args.seed),
            "--camera_names",
            "ee_camera",
            "--headless",
            "--device",
            args.device,
        ]
        if args.video:
            command.append("--video")
        record["stage"] = "record"
        exit_code, timed_out, duration_s = run_child(command, log_path=record_log, timeout_s=args.timeout_s)
        record["record_exit_code"] = exit_code
        record["duration_s"] = duration_s
        if timed_out:
            record["status"] = "timeout"
            record["error"] = "Scripted recorder exceeded its wall-clock limit."
        elif exit_code != 0:
            record["error"] = "Scripted recorder failed; inspect the log."
        else:
            session_root = session_root_from_log(record_log)
            if session_root is None:
                record["error"] = "Recorder did not print a finalized SESSION_ROOT marker."
            else:
                record["session_root"] = str(session_root)
                info_path = session_root / "lerobot" / "meta" / "info.json"
                if not info_path.is_file():
                    record["error"] = "Finalized LeRobot meta/info.json is missing."
                else:
                    try:
                        info = json.loads(info_path.read_text())
                        record["episodes"] = int(info.get("total_episodes", 0))
                    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                        record["error"] = f"Invalid finalized LeRobot metadata: {error}"
                    if not record["error"]:
                        if record["episodes"] < 1:
                            record["error"] = "Finalized LeRobot dataset contains no episodes."
                        elif args.video and not list((session_root / "videos").glob("*.mp4")):
                            record["error"] = "Recorder video was requested, but no MP4 was saved."
                        elif args.validate:
                            record["stage"] = "validate"
                            record["validation_log"] = str(validation_log)
                            validator = [
                                sys.executable,
                                str(VALIDATOR_SCRIPT),
                                "--dataset_root",
                                str(session_root),
                                "--repo_id",
                                repo_id,
                                "--target_hz",
                                str(args.target_hz),
                            ]
                            validation_exit, validation_timeout, validation_duration = run_child(
                                validator, log_path=validation_log, timeout_s=args.timeout_s
                            )
                            record["duration_s"] += validation_duration
                            record["validation_exit_code"] = validation_exit
                            if validation_timeout:
                                record["status"] = "timeout"
                                record["error"] = "LeRobot validator exceeded its wall-clock limit."
                            elif validation_exit != 0:
                                record["error"] = "LeRobot validator failed; inspect the log."
                            else:
                                record["status"] = "passed"
                                record["stage"] = "complete"
                        else:
                            record["status"] = "passed"
                            record["stage"] = "complete"
        results.append(record)
        save_report(output_dir, registry, results, args)
        print(f"  {record['status']}; log: {record_log}", flush=True)

    print(f"Report: {output_dir / 'results.json'}", flush=True)
    inventory_ok = len(registry) == 106 and len(families) == 12
    return 0 if inventory_ok and all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
