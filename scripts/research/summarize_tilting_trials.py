# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate recorded tilt allocation, servo targets and measured joint response.

The shared scripted summarizer checks task outcomes, all telemetry rows, action
semantics and source hashes first. This entrypoint adds tilt-specific checks;
it never treats a terminal automatic-reset state as the flight response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from summarize_uaquad_trials import (
    _provenance_path,
    _resolve_existing_path,
    _rpy_degrees,
    _vector,
)

JOINT_COUNT = 6


def main() -> int:
    """Build an honest report, publishing it only after the tilt checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, action="append", required=True)
    parser.add_argument("--expected-specs-json", type=Path)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-trajectories", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ambench-tilt-summary-") as temporary:
        temporary_root = Path(temporary)
        baseline = temporary_root / "trials.json"
        trajectories = temporary_root / "trajectories.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("summarize_uaquad_trials.py")),
            "--output-summary",
            str(baseline),
            "--output-trajectories",
            str(trajectories),
        ]
        for report in args.report_json:
            command.extend(["--report-json", str(report)])
        if args.expected_specs_json is not None:
            command.extend(["--expected-specs-json", str(args.expected_specs_json)])
        if args.require_complete:
            command.append("--require-complete")
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        if run.returncode:
            print(run.stdout, end="")
            print(run.stderr, end="", file=sys.stderr)
            return run.returncode
        summary = json.loads(baseline.read_text())
        sampled = json.loads(trajectories.read_text())
    by_name = {row["name"]: row for row in sampled["trials"]}
    verified = 0
    try:
        for trial in summary["trials"]:
            if trial.get("status") != "verified" or trial["profile"]["robot_id"] != "omni_hexa":
                continue
            tilt_summary, tilt_points = summarize_motor_tilt(trial, by_name[trial["name"]])
            trial["motor_tilt"] = tilt_summary
            by_name[trial["name"]]["points"] = tilt_points
            by_name[trial["name"]]["sampled_point_count"] = len(tilt_points)
            verified += 1
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"Tilt telemetry validation failed: {error}", file=sys.stderr)
        return 1
    summary["motor_tilt_verification"] = {
        "status": "passed",
        "verified_omni_trials": verified,
        "terminal_post_step_excluded": True,
        "scope": "Recorded allocation and joint response; not a reconstruction of hardware rotor forces.",
    }
    sampled["source_summary"] = _provenance_path(args.output_summary.resolve())
    sampled["max_points_per_trial"] = 168
    sampled["sampling"] = "At most 120 uniform points plus measured per-joint tilt/rate/error extrema."
    write_json_atomic(args.output_summary, summary)
    write_json_atomic(args.output_trajectories, sampled)
    print(f"{summary['verification_status']}: {verified} OmniHexa tilt traces verified")
    return 0


def summarize_motor_tilt(trial: dict, sampled: dict) -> tuple[dict, list[dict]]:
    """Check every tilt row and accumulate phase-specific joint statistics."""
    outcome_path = _resolve_existing_path(trial["source"]["episode_outcomes"]["path"])
    metadata = json.loads(outcome_path.read_text())["motor_tilt"]
    joint_names = metadata["joint_names"]
    if joint_names != [f"base_motor_arm{index}" for index in range(1, JOINT_COUNT + 1)]:
        raise ValueError(f"{trial['name']}: unexpected motor joint order")
    limits = [_vector(pair, 2, "soft_joint_limits_rad") for pair in metadata["soft_joint_limits_rad"]]
    if len(limits) != JOINT_COUNT or any(lower >= upper for lower, upper in limits):
        raise ValueError(f"{trial['name']}: invalid tilt joint limits")
    groups = ("allocated", "target", "actual_before", "actual_after")
    bounds = {key: [[math.inf, -math.inf] for _ in range(JOINT_COUNT)] for key in groups}
    square_sum = {key: [0.0] * JOINT_COUNT for key in ("pre", "post")}
    max_error = {key: [0.0] * JOINT_COUNT for key in ("pre", "post")}
    command_rate = [0.0] * JOINT_COUNT
    velocity_max = [0.0] * JOINT_COUNT
    clamp_steps = nonterminal_steps = row_count = 0
    previous_target = previous_after = None
    selected = {point["step_index"] for point in sampled["points"]}
    all_points = []
    digest = hashlib.sha256()
    telemetry = _resolve_existing_path(trial["source"]["telemetry"]["path"])
    with telemetry.open("rb") as stream:
        for index, raw in enumerate(stream):
            digest.update(raw)
            row_count += 1
            row = json.loads(raw)
            output = row["controller_output"]
            before = row["observation_before_step"]
            after = row["observation_after_step"]
            if index == 0:
                initial = _vector(metadata["initial_joint_pos_rad"], JOINT_COUNT, "initial motor angles")
                first = _vector(before["motor_arm_joint_pos"], JOINT_COUNT, "first motor angles")
                if any(abs(a - b) > 1e-5 for a, b in zip(initial, first)):
                    raise ValueError(f"{trial['name']}: initial motor angles disagree with metadata")
            allocated = _vector(output["motor_arm_angles"], JOINT_COUNT, "allocated angle")
            target = _vector(output["motor_arm_position_targets"], JOINT_COUNT, "clamped target")
            actual_before = _vector(before["motor_arm_joint_pos"], JOINT_COUNT, "actual pre-step angle")
            actual_after = _vector(after["motor_arm_joint_pos"], JOINT_COUNT, "actual post-step angle")
            velocity_before = _vector(before["motor_arm_joint_vel"], JOINT_COUNT, "pre-step velocity")
            velocity_after = _vector(after["motor_arm_joint_vel"], JOINT_COUNT, "post-step velocity")
            reset_state = bool(row["terminated"] or row["truncated"])
            if after["is_reset_state"] is not reset_state or reset_state != (index == trial["steps"] - 1):
                raise ValueError(f"{trial['name']}: reset phase mismatch at step {index}")
            if previous_after is not None and any(abs(a - b) > 1e-5 for a, b in zip(previous_after, actual_before)):
                raise ValueError(f"{trial['name']}: pre/post continuity mismatch at step {index}")
            clamped = [max(pair[0], min(pair[1], value)) for pair, value in zip(limits, allocated)]
            if not reset_state and any(abs(a - b) > 1e-5 for a, b in zip(clamped, target)):
                raise ValueError(f"{trial['name']}: target is not allocation clipped to soft limits at step {index}")
            clamp_steps += not reset_state and any(abs(a - b) > 1e-5 for a, b in zip(allocated, target))
            for key, values in (("allocated", allocated), ("target", target), ("actual_before", actual_before)):
                if reset_state and key != "actual_before":
                    continue
                for joint, value in enumerate(values):
                    bounds[key][joint][0] = min(bounds[key][joint][0], value)
                    bounds[key][joint][1] = max(bounds[key][joint][1], value)
            for joint in range(JOINT_COUNT):
                if not reset_state:
                    error = abs(actual_before[joint] - target[joint])
                    square_sum["pre"][joint] += error * error
                    max_error["pre"][joint] = max(max_error["pre"][joint], error)
                velocity_max[joint] = max(velocity_max[joint], abs(velocity_before[joint]))
                if previous_target is not None and not reset_state:
                    command_rate[joint] = max(
                        command_rate[joint], abs(target[joint] - previous_target[joint]) / trial["step_dt_s"]
                    )
                if not reset_state:
                    value = actual_after[joint]
                    bounds["actual_after"][joint][0] = min(bounds["actual_after"][joint][0], value)
                    bounds["actual_after"][joint][1] = max(bounds["actual_after"][joint][1], value)
                    error = abs(value - target[joint])
                    square_sum["post"][joint] += error * error
                    max_error["post"][joint] = max(max_error["post"][joint], error)
                    velocity_max[joint] = max(velocity_max[joint], abs(velocity_after[joint]))
            nonterminal_steps += not reset_state
            previous_target = target
            previous_after = None if reset_state else actual_after
            roll, pitch, yaw = _rpy_degrees(before["base_quat"])
            all_points.append({
                "step_index": index,
                "timestamp_s": row["timestamp_s"],
                "base_pos_m": before["base_pos"],
                "ee_pos_m": before["ee_pos"],
                "action": row["action"],
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "motor_thrusts_n": output["motor_thrusts"],
                "motor_arm_allocated_rad": None if reset_state else allocated,
                "motor_arm_target_rad": None if reset_state else target,
                "motor_arm_actual_before_rad": actual_before,
                "motor_arm_actual_after_rad": None if reset_state else actual_after,
                "motor_arm_velocity_before_rad_s": velocity_before,
                "motor_arm_velocity_after_rad_s": None if reset_state else velocity_after,
                "post_step_is_reset_state": reset_state,
            })
    if row_count != trial["steps"] or digest.hexdigest() != trial["source"]["telemetry"]["sha256"]:
        raise ValueError(f"{trial['name']}: tilt trace changed after baseline validation")
    if nonterminal_steps == 0:
        raise ValueError(f"{trial['name']}: no nonterminal tilt response")
    result = {
        "joint_names": joint_names,
        "soft_limits_rad": limits,
        "sample_count": row_count,
        "nonterminal_post_step_count": nonterminal_steps,
        "command_sample_count": nonterminal_steps,
        "timing": (
            "Actual before step is t=i*dt; nonterminal command targets belong to that step, actual after to (i+1)*dt."
            " Terminal post-step and command buffers may have been reset, so both are excluded from tilt"
            " command/response statistics."
        ),
        **{
            key: {"joint_min_rad": [b[0] for b in value], "joint_max_rad": [b[1] for b in value]}
            for key, value in bounds.items()
        },
        "lag": {
            phase: {
                "rms_per_joint_rad": [math.sqrt(value / nonterminal_steps) for value in square_sum[phase]],
                "max_abs_error_per_joint_rad": max_error[phase],
            }
            for phase in ("pre", "post")
        },
        "clamped_step_count": clamp_steps,
        "command_rate_max_rad_s": command_rate,
        "actual_velocity_max_rad_s": velocity_max,
    }
    return result, sample_tilt_extrema(all_points, selected)


def sample_tilt_extrema(points: list[dict], selected: set[int]) -> list[dict]:
    """Retain short real transients that uniform plot sampling could miss."""
    valid = range(len(points) - 1)
    for joint in range(JOINT_COUNT):
        for field, reverse in (("motor_arm_allocated_rad", False), ("motor_arm_allocated_rad", True)):
            order = max if reverse else min
            selected.add(order(valid, key=lambda index: points[index][field][joint]))
        for actual in ("motor_arm_actual_before_rad", "motor_arm_actual_after_rad"):
            selected.add(
                max(
                    valid,
                    key=lambda index: abs(points[index][actual][joint] - points[index]["motor_arm_target_rad"][joint]),
                )
            )
        selected.add(
            max(
                valid,
                key=lambda index: max(
                    abs(points[index]["motor_arm_velocity_before_rad_s"][joint]),
                    abs(points[index]["motor_arm_velocity_after_rad_s"][joint]),
                ),
            )
        )
        if len(points) > 2:
            selected.add(
                max(
                    range(1, len(points) - 1),
                    key=lambda index: abs(
                        points[index]["motor_arm_target_rad"][joint] - points[index - 1]["motor_arm_target_rad"][joint]
                    ),
                )
            )
    clipped = [
        index
        for index in valid
        if any(
            abs(a - b) > 1e-5
            for a, b in zip(points[index]["motor_arm_allocated_rad"], points[index]["motor_arm_target_rad"])
        )
    ]
    if clipped:
        selected.add(clipped[0])
    if len(selected) > 168:
        raise ValueError("Tilt extrema sampling exceeded its documented point budget")
    return [points[index] for index in sorted(selected)]


def write_json_atomic(path: Path, value: dict) -> None:
    """Write finite JSON through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".tilting-", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.write("\n")
    temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
