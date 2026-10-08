# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Record one honest scripted attempt per media specification, including failures.

Run inside the simulation Docker. Each child uses the public recorder, one
environment and a one-episode limit. A task timeout is a completed experiment;
missing outcomes, video or invalid canonical data are execution failures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from verify_matrix import REPO_ROOT, run_child
from verify_scripted import (
    PROMPT_MAP,
    RECORDER_SCRIPT,
    VALIDATOR_SCRIPT,
    session_root_from_log,
)


def main() -> int:
    """Run the configured recordings sequentially on one simulation GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs-json", type=Path, default=REPO_ROOT / "usage_assets/variant_recording_specs.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", action="append", default=[], help="Only record these names; repeatable.")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    specs = json.loads(args.specs_json.read_text())
    if not isinstance(specs, list) or not specs:
        parser.error("Recording specifications must be a nonempty list")
    names = [spec["name"] for spec in specs]
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_-]+", name) for name in names):
        parser.error("Recording names may contain only lowercase letters, digits, underscores and hyphens")
    if len(set(names)) != len(names) or set(args.name) - set(names):
        parser.error("Recording names must be unique and --name must match the specification")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Use a new empty --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(PROMPT_MAP.read_text())
    report = {"specs_json": str(args.specs_json), "max_episodes": 1, "results": []}
    selected = [spec for spec in specs if not args.name or spec["name"] in args.name]
    for index, spec in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {spec['name']}", flush=True)
        family = spec["task_id"].split("-Am-")[0]
        base_joint = "-BaseJoint-" in spec["task_id"]
        state_keys = (
            ["base_pos", "base_quat", "arm_joint_pos", "gripper_width"]
            if base_joint
            else [
                "ee_pos",
                "ee_quat",
                "gripper_width",
            ]
        )
        repo_id = f"am_bench/{family.lower()}_{'base_joint' if base_joint else 'ee'}_absolute"
        record_log = output_dir / "logs" / f"{spec['name']}.log"
        command = [
            sys.executable,
            str(RECORDER_SCRIPT),
            "--task",
            spec["task_id"],
            "--dataset_root",
            str(output_dir / "datasets" / spec["name"]),
            "--repo_id",
            repo_id,
            "--state_keys",
            *state_keys,
            "--task_prompt",
            prompts.get(f"{family}EEAbsPID", "inspect the marked surface"),
            "--step_hz",
            "120",
            "--num_envs",
            "1",
            "--num_demos",
            "1",
            "--max_episodes",
            "1",
            "--save_failed_episodes",
            "--env_length_s",
            str(spec["episode_length_s"]),
            "--seed",
            str(spec["seed"]),
            "--camera_names",
            *spec["camera_names"],
            "--scene_camera_width",
            str(spec["scene_camera_width"]),
            "--scene_camera_height",
            str(spec["scene_camera_height"]),
            "--scene_camera_position",
            *map(str, spec["scene_camera_position"]),
            "--scene_camera_look_at",
            *map(str, spec["scene_camera_look_at"]),
            "--video",
            "--headless",
            "--device",
            args.device,
        ]
        code, timed_out, duration = run_child(command, log_path=record_log, timeout_s=args.timeout_s)
        row = {
            "name": spec["name"],
            "spec": spec,
            "status": "failed",
            "record_exit_code": code,
            "wall_timeout": timed_out,
            "wall_duration_s": duration,
            "record_log": str(record_log),
        }
        session_root = session_root_from_log(record_log)
        try:
            if timed_out or session_root is None:
                raise ValueError("Recorder did not finish; inspect the log")
            outcomes_path = session_root / "episode_outcomes.json"
            outcomes = json.loads(outcomes_path.read_text())
            if outcomes["status"] not in {"completed", "episode_limit"}:
                raise ValueError("Recorder did not reach a completed episode or episode limit")
            if outcomes["seed"] != spec["seed"] or outcomes["task"] != spec["task_id"]:
                raise ValueError("Recorded seed or task disagrees with the specification")
            if sorted(outcomes["camera_names"]) != sorted(spec["camera_names"]):
                raise ValueError("Actual recorded cameras disagree with the specification")
            if len(outcomes["episodes"]) != 1:
                raise ValueError("Expected exactly one completed attempt")
            episode = outcomes["episodes"][0]
            if episode["termination_reason"] not in {"success", "timeout"}:
                raise ValueError("Unexpected recorded task outcome")
            if code != (0 if episode["termination_reason"] == "success" else 1):
                raise ValueError("Exit code disagrees with the recorded task outcome")
            videos = [session_root / "videos" / f"scripted-{camera}-env0-eps0.mp4" for camera in spec["camera_names"]]
            if not episode["exported"] or not videos or not all(path.is_file() for path in videos):
                raise ValueError("Completed attempt or videos were not saved")
            validation_log = output_dir / "logs" / f"{spec['name']}-validation.log"
            validation_code, validation_timeout, _ = run_child(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--dataset_root",
                    str(session_root),
                    "--repo_id",
                    repo_id,
                    "--target_hz",
                    "20",
                ],
                log_path=validation_log,
                timeout_s=args.timeout_s,
            )
            if validation_timeout or validation_code:
                raise ValueError("Canonical validation failed; inspect the validation log")
            row.update({
                "status": "completed",
                "session_root": str(session_root),
                "outcomes_json": str(outcomes_path),
                "termination_reason": episode["termination_reason"],
                "steps": episode["step_count"],
                "validation_log": str(validation_log),
            })
        except (OSError, ValueError, KeyError) as error:
            row["error"] = str(error)
        report["results"].append(row)
        (output_dir / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"  {row['status']}: {row.get('termination_reason', row.get('error'))}", flush=True)
    return 0 if all(row["status"] == "completed" for row in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
