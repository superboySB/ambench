# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Audit one recorded Omni preflight on CPU without starting Isaac Sim.

Checks canonical samples, every telemetry/physics row and every MP4 frame.
Historical MoviePy duration rounding must be selected explicitly; the original
recording and any failed audit remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

CAMERAS = ["scene_camera", "base_camera", "ee_camera"]
TASKS = {
    "PressButton-Am-OmniHexa-Abs-PID-Direct-v0",
    "WipeWindow-Am-OmniHexa-Abs-PID-Direct-v0",
}


def require(condition: bool, message: str) -> None:
    """Reject inconsistent evidence independently of Python optimization flags."""
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    """Hash a source artifact without loading it completely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """Read a JSON object from an experiment artifact."""
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def vector(value: list, size: int, label: str) -> np.ndarray:
    """Validate a finite numeric vector."""
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (size,) and bool(np.isfinite(array).all()), f"Invalid {label}")
    return array


def expected_video_frames(steps: int, stride: int, fps: int, legacy: bool) -> tuple[int, int]:
    """Return sampled/encoded counts for the explicitly selected writer behavior.

    The old ImageSequenceClip sums one duration per image; iter_frames floors
    that sum times FPS. Reproduce that exact arithmetic, without a tolerance
    that could conceal an unrelated truncated recording.
    """
    require(steps > 0 and stride > 0 and fps > 0, "Video counts and rates must be positive")
    sampled = len(range(0, steps, stride))
    encoded = int(sum([1.0 / fps] * sampled) * fps) if legacy else sampled
    require(encoded > 0, "The selected writer behavior yields no video frames")
    return sampled, encoded


def audit_preflight(session: Path, physics: Path, task_id: str, seed: int, legacy: bool) -> dict:
    """Validate a single native preflight and retain its actual task outcome."""
    cv2.setNumThreads(1)
    outcome = read_json(session / "episode_outcomes.json")
    info = read_json(session / "lerobot/meta/info.json")
    require(
        outcome["task"] == task_id and type(outcome["seed"]) is int and outcome["seed"] == seed, "Task/seed mismatch"
    )
    require(outcome["status"] in {"completed", "episode_limit"}, "Recording did not finalize")
    require(outcome["max_episodes"] == outcome["requested_successes"] == 1, "Expected one native attempt")
    require(outcome["episode_length_s"] == 20 and outcome["step_dt_s"] == 1 / 120, "Unexpected native timing")
    require(outcome["save_failed_episodes"] is True, "Failed attempts must also be retained")
    require(outcome["camera_names"] == outcome["requested_camera_names"] == CAMERAS, "Camera mismatch")
    require(
        outcome["robot_profile"]
        == {
            "robot_id": "omni_hexa",
            "pipeline": "ManipulatorIKPipeline",
            "controller": "PID6DOFController",
            "action_mode": "absolute_ee_pose",
        },
        "Unexpected robot/control profile",
    )
    require(
        all(
            outcome["experiment_conditions"][key] is False
            for key in (
                "enable_saturation",
                "enable_aerodynamic_effects",
                "enable_wind_effect",
            )
        ),
        "Expected the declared undisturbed preflight",
    )
    require(info["robot_type"] == "omni_hexa" and info["fps"] == 120, "Canonical robot/rate mismatch")
    require(
        info["ambench"]
        == {
            "action_semantics": "ee_absolute",
            "state_keys": ["ee_pos", "ee_quat", "gripper_width"],
        },
        "Canonical state/action semantics mismatch",
    )
    require(
        info["features"]["action"]["shape"] == info["features"]["observation.state"]["shape"] == [8],
        "Expected eight-dimensional canonical samples",
    )
    require(
        {key.removeprefix("observation.images.") for key in info["features"] if key.startswith("observation.images.")}
        == set(CAMERAS),
        "Canonical cameras mismatch",
    )
    require("\nnum_rerenders_on_reset: 1\n" in (session / "env_cfg.yaml").read_text(), "Reset rerender mismatch")
    require(len(outcome["episodes"]) == info["total_episodes"] == 1, "Expected exactly one exported episode")
    episode = outcome["episodes"][0]
    require(
        episode["env_id"] == episode["episode_index"] == episode["exported_episode_index"] == 0
        and episode["exported"] is True,
        "Episode export mapping mismatch",
    )
    require(episode["termination_reason"] in {"success", "timeout"}, "Unknown native outcome")
    success = episode["termination_reason"] == "success"
    require(outcome["successful_episodes"] == int(success), "Success count disagrees with episode")
    steps = episode["step_count"]
    require(
        type(steps) is int and 0 < steps <= 2400 and info["total_frames"] == steps, "Native/canonical step mismatch"
    )
    require(episode["simulation_elapsed_s"] == steps * outcome["step_dt_s"], "Simulation duration mismatch")
    fps, stride = outcome["video_fps"], outcome["video_frame_skip"]
    require(fps == 30 and stride == 4, "Unexpected recording sample rate")
    sampled, encoded = expected_video_frames(steps, stride, fps, legacy)

    telemetry_path = session / "telemetry.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    require(len(telemetry) == outcome["telemetry_rows"] == steps, "Incomplete telemetry")
    limits = np.asarray(outcome["motor_tilt"]["soft_joint_limits_rad"], dtype=float)
    require(
        limits.shape == (6, 2) and bool(np.isfinite(limits).all()) and bool((limits[:, 0] < limits[:, 1]).all()),
        "Invalid motor joint limits",
    )
    require(outcome["motor_tilt"]["joint_names"] == [f"base_motor_arm{i}" for i in range(1, 7)], "Motor order mismatch")
    peak_actual = peak_target = peak_lag = 0.0
    for index, row in enumerate(telemetry):
        terminal = index == steps - 1
        require(row["env_id"] == 0 and row["step_index"] == index, "Telemetry step order mismatch")
        require(math.isclose(row["timestamp_s"], index / 120, abs_tol=1e-9), "Telemetry timestamp mismatch")
        require(type(row["terminated"]) is bool and type(row["truncated"]) is bool, "Untyped terminal state")
        require(
            (row["terminated"] or row["truncated"]) == terminal and (not terminal or row["terminated"] == success),
            "Telemetry native outcome mismatch",
        )
        before, after, command = row["observation_before_step"], row["observation_after_step"], row["controller_output"]
        require(after["is_reset_state"] is terminal, "Automatic-reset phase mismatch")
        actual = vector(before["motor_arm_joint_pos"], 6, "pre-step tilt")
        vector(before["motor_arm_joint_vel"], 6, "pre-step tilt velocity")
        post = vector(after["motor_arm_joint_pos"], 6, "post-step tilt")
        vector(after["motor_arm_joint_vel"], 6, "post-step tilt velocity")
        allocated = vector(command["motor_arm_angles"], 6, "allocated tilt")
        target = vector(command["motor_arm_position_targets"], 6, "target tilt")
        if index:
            previous = telemetry[index - 1]["observation_after_step"]["motor_arm_joint_pos"]
            require(bool(np.allclose(actual, previous, rtol=0, atol=1e-5)), "Tilt state continuity mismatch")
        peak_actual = max(peak_actual, float(np.abs(actual).max()))
        if not terminal:
            require(
                bool(np.allclose(target, np.clip(allocated, limits[:, 0], limits[:, 1]), rtol=0, atol=1e-5)),
                "Tilt target is not the clipped allocation",
            )
            peak_target = max(peak_target, float(np.abs(target).max()))
            peak_lag = max(peak_lag, float(np.abs(post - target).max()))

    parquet_files = sorted((session / "lerobot/data").rglob("*.parquet"))
    require(bool(parquet_files), "Canonical parquet is missing")
    canonical_rows = 0
    for path in parquet_files:
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["frame_index", "episode_index", "action", "observation.state"]
        ):
            for source in batch.to_pylist():
                require(
                    source["episode_index"] == 0 and source["frame_index"] == canonical_rows,
                    "Canonical episode/frame order mismatch",
                )
                require(canonical_rows < steps, "Extra canonical rows")
                row = telemetry[canonical_rows]
                state = [
                    value
                    for key in ("ee_pos", "ee_quat", "gripper_width")
                    for value in row["observation_before_step"][key]
                ]
                for name, expected in (("action", row["action"]), ("observation.state", state)):
                    require(
                        bool(
                            np.array_equal(
                                np.asarray(source[name], dtype=np.float32), np.asarray(expected, dtype=np.float32)
                            )
                        ),
                        f"Canonical {name} differs from telemetry at frame {canonical_rows}",
                    )
                canonical_rows += 1
    require(canonical_rows == steps, "Canonical rows are incomplete")

    metadata_path = physics / "physics-metadata.json"
    metadata = read_json(metadata_path)
    require(
        metadata["measurement_stage"] == "after_original_get_dones_return_before_automatic_reset"
        and metadata["calls_get_success_again"] is False
        and metadata["changes_policy_physics_rewards_native_thresholds_or_timeout"] is False,
        "Physics hook changes native task semantics",
    )
    native_path = physics / "native-before-reset.jsonl"
    native = [json.loads(line) for line in native_path.read_text().splitlines()]
    require(len(native) == steps, "Incomplete pre-reset physics trace")
    for index, row in enumerate(native):
        require(row["native_step"] == index + 1, "Physics step order mismatch")
        require(
            row["terminated"] == telemetry[index]["terminated"] and row["truncated"] == telemetry[index]["truncated"],
            "Physics/telemetry terminal mismatch",
        )
        for key in ("motor_arm_joint_pos_rad", "motor_arm_position_targets_rad", "allocated_motor_arm_angles_rad"):
            vector(row[key], 6, key)
        if task_id.startswith("PressButton"):
            require(math.isfinite(row["button_joint_pos_m"]), "Non-finite button displacement")
        else:
            require(metadata["sponge_attachment_extra_local"] == [0.0, 0.0, 0.0], "Sponge attachment changed")
            require(len(row["stain_visible"]) == len(row["stain_contact_counts"]) == 4, "Stain count mismatch")
            require(
                math.isfinite(row["native_contact_force_magnitude_N"])
                and math.isfinite(row["sponge_attachment_position_error_m"]),
                "Non-finite contact evidence",
            )
    if task_id.startswith("PressButton"):
        require(
            metadata["native_joint_threshold_m"] == 0.004 and metadata["native_joint_comparison"] == "gte",
            "Button success criterion changed",
        )
        require(
            (native[-1]["button_joint_pos_m"] >= 0.004) == success, "Native success contradicts button displacement"
        )
    elif success:
        require(
            native[-1]["stain_visible"] == [False] * 4
            and all(value >= 1 for value in native[-1]["stain_contact_counts"]),
            "Native success contradicts stain state",
        )
        require(
            all(native[-1]["native_success_criteria_cached"][f"stain_{i}_removed"] == [True] for i in range(4)),
            "Native stain success cache mismatch",
        )

    videos = []
    expected_paths = {session / f"videos/scripted-{camera}-env0-eps0.mp4" for camera in CAMERAS}
    require(set((session / "videos").glob("*.mp4")) == expected_paths, "Unexpected or missing video files")
    endpoint_indices = {0, (encoded - 1) * stride}
    for camera in CAMERAS:
        endpoints = {}
        for path in parquet_files:
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=64, columns=["frame_index", f"observation.images.{camera}"]
            ):
                for source in batch.to_pylist():
                    index = source["frame_index"]
                    if index in endpoint_indices:
                        payload = source[f"observation.images.{camera}"]["bytes"]
                        endpoints[index] = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(
            set(endpoints) == endpoint_indices and all(value is not None for value in endpoints.values()),
            f"Missing canonical video endpoints: {camera}",
        )
        path = session / f"videos/scripted-{camera}-env0-eps0.mp4"
        capture = cv2.VideoCapture(str(path))
        require(capture.isOpened() and abs(capture.get(cv2.CAP_PROP_FPS) - fps) < 1e-6, f"Invalid video: {camera}")
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count, comparisons = 0, []
        pose = outcome["camera_poses"][camera]
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            require(frame.shape == (pose["height"], pose["width"], 3), f"Video dimensions mismatch: {camera}")
            if count in {0, encoded - 1}:
                source_index = count * stride
                error = float(np.abs(frame.astype(np.float32) - endpoints[source_index]).mean())
                require(error <= 8.0, f"Video/canonical endpoint mismatch: {camera}, source {source_index}")
                comparisons.append({"video_frame": count, "canonical_frame": source_index, "RGB_mean_abs_error": error})
            count += 1
        capture.release()
        require(declared == count == encoded, f"Video frame count mismatch: {camera}, expected {encoded}, got {count}")
        videos.append({
            "camera": camera,
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "decoded_frames": count,
            "fps": fps,
            "shape_hw": [pose["height"], pose["width"]],
            "canonical_endpoint_comparisons": comparisons,
        })

    sources = [
        session / "episode_outcomes.json",
        session / "env_cfg.yaml",
        session / "lerobot/meta/info.json",
        telemetry_path,
        metadata_path,
        native_path,
        *parquet_files,
    ]
    return {
        "schema_version": 1,
        "status": "passed_fresh_Omni_preflight" if success else "failed_native_preflight",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": sha256(Path(__file__)),
        "task_id": task_id,
        "seed": seed,
        "native_successes": int(success),
        "native_attempts": 1,
        "native_outcome": episode["termination_reason"],
        "native_steps": steps,
        "simulation_duration_s": episode["simulation_elapsed_s"],
        "all_telemetry_rows_verified": steps,
        "all_canonical_numeric_rows_verified": canonical_rows,
        "all_pre_reset_physics_rows_verified": len(native),
        "native_terminal_before_reset": native[-1],
        "motor_tilt": {
            "peak_actual_rad": peak_actual,
            "peak_nonterminal_target_rad": peak_target,
            "maximum_nonterminal_post_step_error_rad": peak_lag,
            "terminal_reset_excluded": True,
        },
        "video_timing": {
            "mode": "legacy_summed_duration_floor" if legacy else "complete_sample_sequence",
            "sampled_frames": sampled,
            "encoded_frames": encoded,
            "omitted_tail_samples": sampled - encoded,
            "frame_skip": stride,
            "first_canonical_frame": 0,
            "last_canonical_frame": (encoded - 1) * stride,
            "encoded_duration_s": encoded / fps,
            "endpoint_RGB_mean_abs_error_limit": 8.0,
            "sampling": "Pre-step image at raw indices 0,stride,...; terminal state is not an image.",
        },
        "videos": videos,
        "sources_sha256": {str(path): sha256(path) for path in sources},
        "historical_artifacts_modified": False,
        "Isaac_started": False,
        "GPU_used": False,
        "trained": False,
        "claim_limit": "One scripted native simulation attempt; no learned-policy or hardware success claim.",
    }


def main() -> int:
    """Write a new audit receipt after all consistency checks succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--task-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--legacy-moviepy-timing", action="store_true", help="Audit original clips encoded before the duration fix."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Set CUDA_VISIBLE_DEVICES=-1 for this CPU-only audit")
    require(not args.output.exists(), "Keep existing receipts; use a new --output path")
    receipt = audit_preflight(args.session, args.physics, args.task_id, args.seed, args.legacy_moviepy_timing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "native_successes", "native_attempts", "native_steps")}))
    return 0 if receipt["native_successes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
