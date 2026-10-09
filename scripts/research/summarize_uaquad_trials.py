# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate scripted trial telemetry and summarize real observed trajectories.

This tool reads existing recorder outputs only. An incomplete experiment remains
``incomplete`` until every name in the recording specifications is verified.
Use ``--require-complete`` for the final gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_SOURCES = {
    "ua_quad": ("UAQuad", "source/ambench/ambench/robots/ua_quad.py", 4),
    "ua_hexa": ("UAHexa", "source/ambench/ambench/robots/ua_hexa.py", 6),
    "fa_hexa": ("FAHexa", "source/ambench/ambench/robots/fa_hexa.py", 6),
    "omni_hexa": ("OmniHexa", "source/ambench/ambench/robots/omni_hexa.py", 6),
}
MAX_TRAJECTORY_POINTS = 120


def main() -> int:
    """Summarize completed trials and report experiment coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json", type=Path, action="append", required=True, help="Repeat for each results.json."
    )
    parser.add_argument("--expected-specs-json", type=Path, help="Override the specs_json path embedded in reports.")
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument(
        "--output-trajectories", type=Path, help="Optional sampled real telemetry, at most 120 points/trial."
    )
    parser.add_argument(
        "--require-complete", action="store_true", help="Exit nonzero while any expected trial is missing."
    )
    args = parser.parse_args()

    reports = []
    expected_sources = []
    expected_by_name = {}
    for report_arg in args.report_json:
        report_path = _resolve_existing_path(str(report_arg))
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        if not isinstance(report, dict):
            raise ValueError(f"Expected report JSON object: {report_path}")
        if not isinstance(report.get("results"), list):
            raise ValueError(f"Report has no results list: {report_path}")
        reports.append((report_path, report, hashlib.sha256(report_bytes).hexdigest()))
        if args.expected_specs_json is None:
            specs_path = _resolve_existing_path(report["specs_json"], report_path)
            if specs_path not in expected_sources:
                expected_sources.append(specs_path)

    if args.expected_specs_json is not None:
        expected_sources = [_resolve_existing_path(str(args.expected_specs_json))]
    for specs_path in expected_sources:
        specs = json.loads(specs_path.read_text())
        if not isinstance(specs, list) or not specs:
            raise ValueError(f"Expected nonempty specification list: {specs_path}")
        for spec in specs:
            name = spec["name"]
            if name in expected_by_name and expected_by_name[name] != spec:
                raise ValueError(f"Conflicting expected specifications for {name}")
            expected_by_name[name] = spec

    seen_names = set()
    duplicate_names = set()
    trials = []
    trajectory_trials = []
    for report_path, report, report_sha in reports:
        for result in report["results"]:
            name = result.get("name", "<missing-name>")
            if name in seen_names:
                duplicate_names.add(name)
            seen_names.add(name)
            if result.get("status") != "completed":
                trials.append({
                    "name": name,
                    "status": "invalid",
                    "error": f"Result status is {result.get('status')!r}: {result.get('error', '')}",
                    "report_path": str(report_path),
                    "report_sha256": report_sha,
                })
                continue
            try:
                trial, sampled = _summarize_trial(result, report_path, report_sha)
            except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
                trials.append({
                    "name": name,
                    "status": "invalid",
                    "error": str(error),
                    "report_path": str(report_path),
                    "report_sha256": report_sha,
                })
            else:
                trials.append(trial)
                trajectory_trials.append(sampled)

    verified_names = {trial["name"] for trial in trials if trial["status"] == "verified"}
    expected_names = set(expected_by_name)
    missing_names = sorted(expected_names - verified_names)
    extra_names = sorted(seen_names - expected_names)
    invalid_names = sorted({trial["name"] for trial in trials if trial["status"] == "invalid"})
    if invalid_names or duplicate_names or extra_names:
        verification_status = "failed"
    elif missing_names:
        verification_status = "incomplete"
    else:
        verification_status = "passed"

    summary = {
        "schema_version": 1,
        "verification_status": verification_status,
        "coverage": {
            "expected_count": len(expected_names),
            "verified_count": len(verified_names & expected_names),
            "missing_names": missing_names,
            "extra_names": extra_names,
            "duplicate_names": sorted(duplicate_names),
            "invalid_names": invalid_names,
        },
        "expected_specs": [{"path": _provenance_path(path), "sha256": _sha256(path)} for path in expected_sources],
        "reports": [{"path": _provenance_path(path), "sha256": sha} for path, _, sha in reports],
        "metric_basis": {
            "position_frame": "environment_origin_relative_m",
            "ee_target_error": "observation_before_step.ee_pos minus same-step 8D absolute EE action xyz",
            "attitude": "WXYZ base quaternion, standard intrinsic XYZ roll/pitch/yaw in degrees",
            "motor_limits": "per-robot episode metadata, or AST-parsed checked-in robot specification for old sessions",
        },
        "trials": trials,
    }
    _write_json(args.output_summary, summary)
    if args.output_trajectories is not None:
        _write_json(
            args.output_trajectories,
            {
                "schema_version": 1,
                "source_summary": _provenance_path(args.output_summary.resolve()),
                "max_points_per_trial": MAX_TRAJECTORY_POINTS,
                "trials": trajectory_trials,
            },
        )
    print(
        f"{verification_status}: {len(verified_names & expected_names)}/{len(expected_names)} "
        "expected trials verified; "
        f"summary={args.output_summary}"
    )
    if verification_status == "failed" or (args.require_complete and verification_status != "passed"):
        return 1
    return 0


def _summarize_trial(result: dict[str, Any], report_path: Path, report_sha: str) -> tuple[dict, dict]:
    name = result["name"]
    session_root = _resolve_existing_path(result["session_root"], report_path)
    outcomes_path = _resolve_existing_path(result["outcomes_json"], report_path)
    if outcomes_path.parent != session_root:
        raise ValueError(f"{name}: outcomes_json is outside session_root")
    outcomes = _read_object(outcomes_path)
    episodes = outcomes["episodes"]
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError(f"{name}: expected exactly one recorded episode")
    episode = episodes[0]
    if outcomes.get("status") not in ("completed", "episode_limit"):
        raise ValueError(f"{name}: episode outcomes were not completed")
    if not episode.get("exported"):
        raise ValueError(f"{name}: episode was not exported")
    spec = result["spec"]
    if spec["name"] != name or outcomes["task"] != spec["task_id"] or outcomes["seed"] != spec["seed"]:
        raise ValueError(f"{name}: report specification disagrees with episode outcomes")
    if result.get("termination_reason") != episode["termination_reason"]:
        raise ValueError(f"{name}: report termination disagrees with episode outcomes")
    if result.get("wall_timeout"):
        raise ValueError(f"{name}: external wall-clock timeout occurred")
    exit_code = result.get("record_exit_code")
    if episode["termination_reason"] == "success" and exit_code != 0:
        raise ValueError(f"{name}: successful recorder exited with code {exit_code}")
    if episode["termination_reason"] == "timeout" and exit_code not in (0, 1):
        raise ValueError(f"{name}: completed timeout recorder exited with code {exit_code}")
    steps = episode["step_count"]
    if not isinstance(steps, int) or steps <= 0 or result.get("steps") != steps:
        raise ValueError(f"{name}: report and outcome step counts disagree")
    if outcomes.get("telemetry_rows") != steps:
        raise ValueError(f"{name}: outcome telemetry_rows does not equal episode step_count")
    dt = _number(outcomes["step_dt_s"], "step_dt_s")
    if dt <= 0 or not math.isclose(episode["simulation_elapsed_s"], steps * dt, abs_tol=1e-4):
        raise ValueError(f"{name}: episode simulation time disagrees with step count and dt")
    telemetry_name = outcomes.get("telemetry_file")
    if not isinstance(telemetry_name, str) or Path(telemetry_name).name != telemetry_name:
        raise ValueError(f"{name}: missing or unsafe telemetry_file")
    telemetry_path = session_root / telemetry_name
    if not telemetry_path.is_file():
        raise FileNotFoundError(f"{name}: telemetry is missing: {telemetry_path}")

    robot_id, robot_id_source = _robot_id(outcomes, outcomes["task"])
    robot_token, robot_source, rotor_count = ROBOT_SOURCES[robot_id]
    limit_min, limit_max, limits_source = _thrust_limits(outcomes, robot_source)
    saturation_enabled, saturation_source = _saturation(outcomes, spec)
    if (spec.get("saturation") or spec.get("disturbance")) and not saturation_enabled:
        raise ValueError(f"{name}: requested saturation is disabled in recorded conditions")
    action_mode, action_mode_source = _action_mode(outcomes, outcomes["task"])

    sampled_count = min(steps, MAX_TRAJECTORY_POINTS)
    selected_indices = (
        {index * (steps - 1) // (sampled_count - 1) for index in range(sampled_count)} if steps > 1 else {0}
    )
    sampled_points = []
    base_start = base_end = ee_start = ee_end = None
    base_bounds = [[math.inf, -math.inf] for _ in range(3)]
    ee_bounds = [[math.inf, -math.inf] for _ in range(3)]
    base_path_length = ee_path_length = error_square_sum = error_max = 0.0
    attitude_bounds = {axis: [math.inf, -math.inf] for axis in ("roll", "pitch", "yaw")}
    motor_min = math.inf
    motor_max = -math.inf
    negative_steps = below_min_steps = above_max_steps = out_of_limit_steps = 0
    max_abs_fx = max_abs_fy = max_force_xy = 0.0
    digest = hashlib.sha256()
    row_count = 0
    final_truncated = False
    with telemetry_path.open("rb") as telemetry_file:
        for row_index, raw_line in enumerate(telemetry_file):
            row_count += 1
            digest.update(raw_line)
            if row_index >= steps:
                raise ValueError(f"{name}: telemetry has more than {steps} rows")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{name}: invalid JSONL row {row_index}: {error}") from error
            timestamp, base_pos, ee_pos, base_quat, action = _validate_telemetry_row(
                row, name, episode, row_index, dt, action_mode
            )
            is_last = row_index == steps - 1
            if is_last:
                final_truncated = bool(row["truncated"])
            output = row["controller_output"]
            force = _vector(output["force_b"], 3, "force_b")
            thrusts = _vector(output["motor_thrusts"], rotor_count, "motor_thrusts")
            roll, pitch, yaw = _rpy_degrees(base_quat)

            if row_index == 0:
                base_start, ee_start = base_pos, ee_pos
            else:
                base_path_length += math.dist(base_end, base_pos)
                ee_path_length += math.dist(ee_end, ee_pos)
            base_end, ee_end = base_pos, ee_pos
            for axis in range(3):
                base_bounds[axis][0] = min(base_bounds[axis][0], base_pos[axis])
                base_bounds[axis][1] = max(base_bounds[axis][1], base_pos[axis])
                ee_bounds[axis][0] = min(ee_bounds[axis][0], ee_pos[axis])
                ee_bounds[axis][1] = max(ee_bounds[axis][1], ee_pos[axis])
            for axis, angle in (("roll", roll), ("pitch", pitch), ("yaw", yaw)):
                attitude_bounds[axis][0] = min(attitude_bounds[axis][0], angle)
                attitude_bounds[axis][1] = max(attitude_bounds[axis][1], angle)
            if action_mode == "absolute_ee_pose":
                error = math.dist(ee_pos, action[:3])
                error_square_sum += error * error
                error_max = max(error_max, error)
            motor_min = min(motor_min, *thrusts)
            motor_max = max(motor_max, *thrusts)
            negative_steps += any(thrust < 0 for thrust in thrusts)
            below = any(thrust < limit_min for thrust in thrusts)
            above = any(thrust > limit_max for thrust in thrusts)
            below_min_steps += below
            above_max_steps += above
            out_of_limit_steps += below or above
            max_abs_fx = max(max_abs_fx, abs(force[0]))
            max_abs_fy = max(max_abs_fy, abs(force[1]))
            max_force_xy = max(max_force_xy, math.hypot(force[0], force[1]))
            if row_index in selected_indices:
                sampled_points.append({
                    "step_index": row_index,
                    "timestamp_s": timestamp,
                    "base_pos_m": base_pos,
                    "ee_pos_m": ee_pos,
                    "action": action,
                    "roll_deg": roll,
                    "pitch_deg": pitch,
                    "yaw_deg": yaw,
                    "motor_thrusts_n": thrusts,
                })
    if row_count != steps:
        raise ValueError(f"{name}: telemetry has {row_count} rows, expected {steps}")

    trial = {
        "name": name,
        "status": "verified",
        "task_id": outcomes["task"],
        "seed": outcomes["seed"],
        "termination_reason": episode["termination_reason"],
        "success": episode["termination_reason"] == "success",
        "timeout": episode["termination_reason"] == "timeout",
        "final_truncated": final_truncated,
        "record_exit_code": exit_code,
        "wall_duration_s": result.get("wall_duration_s"),
        "steps": steps,
        "telemetry_rows": steps,
        "step_dt_s": dt,
        "simulation_elapsed_s": episode["simulation_elapsed_s"],
        "profile": {
            "robot_id": robot_id,
            "robot_id_source": robot_id_source,
            "action_mode": action_mode,
            "action_mode_source": action_mode_source,
            "rotor_count": rotor_count,
            "robot_token": robot_token,
        },
        "conditions": {"enable_saturation": saturation_enabled, "saturation_source": saturation_source},
        "source": {
            "report": {"path": _provenance_path(report_path), "sha256": report_sha},
            "episode_outcomes": {"path": _provenance_path(outcomes_path), "sha256": _sha256(outcomes_path)},
            "telemetry": {"path": _provenance_path(telemetry_path), "sha256": digest.hexdigest()},
        },
        "base": {
            "start_pos_m": base_start,
            "end_pos_m": base_end,
            "position_bounds_m": base_bounds,
            "path_length_m": base_path_length,
            "roll_deg": _angle_stats(attitude_bounds["roll"]),
            "pitch_deg": _angle_stats(attitude_bounds["pitch"]),
            "yaw_deg": _angle_stats(attitude_bounds["yaw"]),
        },
        "ee": {
            "start_pos_m": ee_start,
            "end_pos_m": ee_end,
            "position_bounds_m": ee_bounds,
            "path_length_m": ee_path_length,
            "target_error_m": (
                {"rms": math.sqrt(error_square_sum / steps), "max": error_max}
                if action_mode == "absolute_ee_pose"
                else None
            ),
        },
        "motors": {
            "thrust_limits_n": [limit_min, limit_max],
            "limits_source": limits_source,
            "min_thrust_n": motor_min,
            "max_thrust_n": motor_max,
            "negative_step_count": negative_steps,
            "negative_step_fraction": negative_steps / steps,
            "below_min_step_count": below_min_steps,
            "above_max_step_count": above_max_steps,
            "out_of_limits_step_count": out_of_limit_steps,
            "out_of_limits_step_fraction": out_of_limit_steps / steps,
        },
        "body_force": {
            "max_abs_fx_n": max_abs_fx,
            "max_abs_fy_n": max_abs_fy,
            "max_xy_norm_n": max_force_xy,
        },
    }
    sampled = {
        "name": name,
        "task_id": outcomes["task"],
        "termination_reason": episode["termination_reason"],
        "source_telemetry_sha256": digest.hexdigest(),
        "original_row_count": steps,
        "sampled_point_count": len(sampled_points),
        "points": sampled_points,
    }
    return trial, sampled


def _validate_telemetry_row(
    row: dict, name: str, episode: dict, row_index: int, dt: float, action_mode: str
) -> tuple[float, list[float], list[float], list[float], list[float]]:
    """Verify one row's time, termination, action and initial-state contract."""
    if row.get("env_id") != episode["env_id"] or row.get("step_index") != row_index:
        raise ValueError(f"{name}: telemetry index/env mismatch at row {row_index}")
    timestamp = _number(row["timestamp_s"], "timestamp_s")
    if not math.isclose(timestamp, row_index * dt, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{name}: telemetry timestamp mismatch at row {row_index}")
    is_last = row_index == episode["step_count"] - 1
    expected_reason = episode["termination_reason"]
    if bool(row["terminated"]) != (is_last and expected_reason == "success"):
        raise ValueError(f"{name}: terminated flag mismatch at row {row_index}")
    if not is_last and bool(row["truncated"]):
        raise ValueError(f"{name}: truncated flag mismatch at row {row_index}")
    if is_last and expected_reason == "timeout" and not bool(row["truncated"]):
        raise ValueError(f"{name}: timeout has no final truncated flag")
    observation = row["observation_before_step"]
    base_pos = _vector(observation["base_pos"], 3, "base_pos")
    ee_pos = _vector(observation["ee_pos"], 3, "ee_pos")
    base_quat = _vector(observation["base_quat"], 4, "base_quat")
    action = _vector(row["action"], None, "action")
    if action_mode == "absolute_ee_pose" and len(action) != 8:
        raise ValueError(f"{name}: absolute EE action is not 8D at row {row_index}")
    if action_mode == "absolute_base_joints" and len(action) < 8:
        raise ValueError(f"{name}: invalid base+joint action at row {row_index}")
    if row_index == 0:
        initial = episode["initial_observation"]
        for key, vector in (("base_pos", base_pos), ("ee_pos", ee_pos), ("base_quat", base_quat)):
            initial_vector = _vector(initial[key], len(vector), f"initial_{key}")
            if any(not math.isclose(a, b, abs_tol=1e-5) for a, b in zip(vector, initial_vector)):
                raise ValueError(f"{name}: first telemetry observation does not match initial {key}")
    return timestamp, base_pos, ee_pos, base_quat, action


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve_existing_path(raw: str, relative_to: Path | None = None) -> Path:
    path = Path(raw)
    candidates = [path]
    if relative_to is not None and not path.is_absolute():
        candidates.append(relative_to.parent / path)
    for marker in ("outputs", "usage_assets"):
        if marker in path.parts:
            index = path.parts.index(marker)
            candidates.append(REPO_ROOT.joinpath(*path.parts[index:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve recorded path {raw!r}; tried {candidates}")


def _provenance_path(path: Path) -> str:
    """Keep public source paths portable when artifacts belong to this checkout."""
    resolved = path.resolve()
    return resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT) else str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric {label}, got {value!r}")
    return float(value)


def _vector(value: Any, length: int | None, label: str) -> list[float]:
    if not isinstance(value, list) or (length is not None and len(value) != length) or not value:
        raise ValueError(f"Expected {length or 'nonempty'}-element {label} vector, got {value!r}")
    return [_number(item, label) for item in value]


def _rpy_degrees(quaternion: list[float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"Base quaternion is not normalized: norm={norm}")
    w, x, y, z = (value / norm for value in quaternion)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(map(math.degrees, (roll, pitch, yaw)))


def _angle_stats(bounds: list[float]) -> dict[str, float]:
    return {"min": bounds[0], "max": bounds[1], "max_abs": max(abs(bounds[0]), abs(bounds[1]))}


def _robot_id(outcomes: dict, task_id: str) -> tuple[str, str]:
    profile = outcomes.get("robot_profile")
    if isinstance(profile, dict) and profile.get("robot_id") in ROBOT_SOURCES:
        robot_id = profile["robot_id"]
        token = ROBOT_SOURCES[robot_id][0]
        if f"-{token}-" not in task_id:
            raise ValueError(f"Robot profile {robot_id} disagrees with task ID {task_id}")
        return robot_id, "episode_outcomes.robot_profile.robot_id"
    for robot_id, (token, _, _) in ROBOT_SOURCES.items():
        if f"-{token}-" in task_id:
            return robot_id, "task_id_token_fallback"
    raise ValueError(f"Unknown robot in task ID {task_id}")


def _action_mode(outcomes: dict, task_id: str) -> tuple[str, str]:
    profile = outcomes.get("robot_profile")
    if isinstance(profile, dict) and profile.get("action_mode") in ("absolute_ee_pose", "absolute_base_joints"):
        return profile["action_mode"], "episode_outcomes.robot_profile.action_mode"
    if "-BaseJoint-" in task_id:
        return "absolute_base_joints", "task_id_token_fallback"
    return "absolute_ee_pose", "task_id_token_fallback"


def _thrust_limits(outcomes: dict, robot_source: str) -> tuple[float, float, dict]:
    source_path = REPO_ROOT / robot_source
    source_sha = _sha256(source_path)
    runtime_limits = []
    for key in ("thrust_limits", "thrust_limits_n", "rotor_thrust_limits_n"):
        runtime_limits.extend(_find_named_values(outcomes, key))
    if runtime_limits:
        limits = {_limit_pair(value) for _, value in runtime_limits}
        if len(limits) != 1:
            raise ValueError(f"Conflicting episode thrust limits: {runtime_limits}")
        lower, upper = limits.pop()
        provenance = {
            "kind": "episode_outcomes",
            "json_paths": sorted(path for path, _ in runtime_limits),
            "robot_source_path": _provenance_path(source_path),
            "robot_source_sha256": source_sha,
        }
    else:
        tree = ast.parse(source_path.read_text())
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RotorActuatorCfg":
                for keyword in node.keywords:
                    if keyword.arg == "thrust_limits":
                        found.append(_limit_pair(ast.literal_eval(keyword.value)))
        if len(found) != 1:
            raise ValueError(f"Cannot uniquely find RotorActuatorCfg thrust limits in {source_path}")
        lower, upper = found[0]
        provenance = {
            "kind": "robot_source_ast_fallback",
            "path": _provenance_path(source_path),
            "sha256": source_sha,
            "field": "RotorActuatorCfg.thrust_limits",
        }
    if lower < 0 or upper <= lower:
        raise ValueError(f"Invalid thrust limits {[lower, upper]}")
    return lower, upper, provenance


def _limit_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid thrust_limits {value!r}")
    return _number(value[0], "min_thrust"), _number(value[1], "max_thrust")


def _find_named_values(value: Any, name: str, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        path = f"{prefix}.{name}" if prefix else name
        found = [(path, value[name])] if name in value else []
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_named_values(child, name, child_path))
        return found
    if isinstance(value, list):
        found = []
        for index, child in enumerate(value):
            found.extend(_find_named_values(child, name, f"{prefix}[{index}]"))
        return found
    return []


def _saturation(outcomes: dict, spec: dict) -> tuple[bool, dict]:
    conditions = outcomes.get("experiment_conditions")
    if isinstance(conditions, dict) and isinstance(conditions.get("enable_saturation"), bool):
        return conditions["enable_saturation"], {"kind": "episode_outcomes.experiment_conditions"}
    if spec.get("saturation") or spec.get("disturbance"):
        return True, {"kind": "report_spec_flags_fallback"}
    source_path = REPO_ROOT / "source/ambench/ambench/tasks/base/base_env_cfg.py"
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BaseEnvCfg":
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    if statement.target.id == "enable_saturation":
                        default = ast.literal_eval(statement.value)
                        if not isinstance(default, bool):
                            raise ValueError("BaseEnvCfg.enable_saturation is not a literal bool")
                        return default, {
                            "kind": "base_env_source_ast_fallback",
                            "path": _provenance_path(source_path),
                            "sha256": _sha256(source_path),
                            "field": "BaseEnvCfg.enable_saturation",
                        }
    raise ValueError("Cannot find BaseEnvCfg.enable_saturation default")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
