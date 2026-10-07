# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""List registered AM-Bench environments or run one bounded reset/step probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
operation = parser.add_mutually_exclusive_group(required=True)
operation.add_argument("--list-json", type=Path, help="Write the live Gym registry to this file.")
operation.add_argument("--task", help="Exact registered environment ID to probe.")
parser.add_argument("--result-json", type=Path, help="Write a machine-readable probe result.")
parser.add_argument("--steps", type=int, default=8, help="Number of environment steps after reset.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel scene instances.")
parser.add_argument("--seed", type=int, default=42, help="Seed for environment randomization.")
parser.add_argument("--disable-fabric", action="store_true", help="Use USD I/O instead of Fabric.")
parser.add_argument("--video-dir", type=Path, help="Record an EE camera MP4 in this directory.")
parser.add_argument("--video-camera-name", default="ee_camera", help="Camera used by --video-dir.")
parser.add_argument("--disturbance", action="store_true", help="Enable saturation, aerodynamics, and wind.")
parser.add_argument("--action-noise", action="store_true", help="Enable shared default action noise.")
parser.add_argument("--observation-noise", action="store_true", help="Enable shared default observation noise.")
parser.add_argument(
    "--wind-force", type=float, nargs=3, metavar=("X", "Y", "Z"), help="World-frame wind force in newtons."
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim and Isaac Lab imports must follow AppLauncher construction.
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab.utils.noise import (  # noqa: E402
    GaussianNoiseCfg,
    NoiseModelCfg,
    NoiseModelWithAdditiveBiasCfg,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import ambench.tasks  # noqa: E402,F401
from ambench.recording import RecordVideo  # noqa: E402


def list_registry() -> list[dict[str, str]]:
    """Return every public AM-Bench registration from the live Gym registry."""
    registry = []
    for task_id, spec in gym.registry.items():
        if "-Am-" not in task_id:
            continue
        suffix = task_id.split("-Am-", maxsplit=1)[1]
        registry.append({
            "task_id": task_id,
            "family": task_id.split("-Am-", maxsplit=1)[0],
            "robot": suffix.split("-", maxsplit=1)[0],
            "entry_point": str(spec.entry_point),
            "config_entry_point": str(spec.kwargs.get("env_cfg_entry_point", "")),
            "scripted_policy_entry_point": str(spec.kwargs.get("scripted_policy_entry_point", "")),
        })
    return sorted(registry, key=lambda item: item["task_id"])


def run_probe() -> dict[str, object]:
    """Create, reset, and step the requested environment with neutral actions."""
    if args.steps < 1 or args.num_envs < 1:
        raise ValueError("--steps and --num-envs must both be positive.")
    if args.wind_force is not None and not args.disturbance:
        raise ValueError("--wind-force requires --disturbance.")
    if args.task not in gym.registry or "-Am-" not in args.task:
        raise ValueError(f"Not a registered AM-Bench environment: {args.task}")

    env = None
    stage = "configure"
    started = time.monotonic()
    result: dict[str, object] = {
        "task_id": args.task,
        "status": "failed",
        "stage": stage,
        "steps_completed": 0,
        "seed": args.seed,
    }
    try:
        print(f"AMBENCH_RESEARCH_STAGE={stage}", flush=True)
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        env_cfg.seed = args.seed
        env_cfg.sim.logging_level = "INFO"
        if args.disturbance:
            env_cfg.enable_saturation = True
            env_cfg.enable_aerodynamic_effects = True
            env_cfg.enable_wind_effect = True
        if args.wind_force is not None:
            multirotor = env_cfg.robot_profile.robot.multirotor
            if multirotor is None or multirotor.aerodynamics is None:
                raise ValueError("--wind-force requires a physical robot with an aerodynamic configuration.")
            multirotor.aerodynamics.wind_force_w = tuple(args.wind_force)
        if args.action_noise:
            env_cfg.enable_action_noise = True
            env_cfg.action_noise_model = NoiseModelCfg(
                noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="add")
            )
        if args.observation_noise:
            env_cfg.enable_observation_noise = True
            env_cfg.observation_noise_model = NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
                bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
            )
        if "-MPC-" in args.task:
            # acados writes c_generated_code relative to the process working directory.
            # The simulator mounts source read-only; build in the writable report tree.
            work_dir = (
                args.result_json.parent.parent / "mpc_build" / args.task
                if args.result_json is not None
                else Path(tempfile.mkdtemp(prefix="ambench-mpc-"))
            )
            work_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(work_dir)
            result["work_dir"] = str(work_dir.resolve())

        stage = "construct"
        print(f"AMBENCH_RESEARCH_STAGE={stage}", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        if args.video_dir is not None:
            env = RecordVideo(
                env,
                video_folder=str(args.video_dir),
                camera_names=[args.video_camera_name],
                name_prefix="research",
                fps=30,
                frame_skip=1,
                output_format="mp4",
            )

        stage = "reset"
        print(f"AMBENCH_RESEARCH_STAGE={stage}", flush=True)
        env.reset()
        action_shape = env.action_space.shape
        if action_shape is None:
            raise RuntimeError("Environment action space has no shape.")
        init_state = unwrapped.robot.cfg.init_state
        action = torch.zeros(action_shape, device=unwrapped.device)
        action[..., 0:3] = torch.tensor(init_state.pos, device=unwrapped.device)
        action[..., 3:7] = torch.tensor(init_state.rot, device=unwrapped.device)
        action[..., -1] = 1.0

        stage = "step"
        print(f"AMBENCH_RESEARCH_STAGE={stage}", flush=True)
        with torch.inference_mode():
            for step in range(args.steps):
                if not simulation_app.is_running():
                    raise RuntimeError(f"Isaac Sim stopped after {step} of {args.steps} steps.")
                env.step(action)
                result["steps_completed"] = step + 1

        result.update({
            "status": "passed",
            "stage": "complete",
            "action_shape": list(action_shape),
            "num_envs": args.num_envs,
            "sensor_names": sorted(unwrapped.scene.sensors.keys()),
        })
    except Exception as error:
        result.update({
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                if result["status"] == "passed":
                    result.update(
                        {"status": "failed", "stage": "close", "error_type": type(error).__name__, "error": str(error)}
                    )
        if args.video_dir is not None and result["status"] == "passed":
            videos = sorted(args.video_dir.glob("*.mp4"))
            if videos:
                result["videos"] = [str(path.resolve()) for path in videos]
            else:
                result.update({"status": "failed", "stage": "video", "error": "No MP4 was saved."})
        result["duration_s"] = round(time.monotonic() - started, 3)
        if args.result_json is not None:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            with args.result_json.open("w") as result_file:
                result_file.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
                result_file.flush()
                os.fsync(result_file.fileno())
        print("AMBENCH_RESEARCH_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    """Run exactly one requested registry or environment operation."""
    try:
        if args.list_json is not None:
            registry = list_registry()
            args.list_json.parent.mkdir(parents=True, exist_ok=True)
            args.list_json.write_text(json.dumps(registry, indent=2) + "\n")
            print(f"AMBENCH_RESEARCH_REGISTRY_COUNT={len(registry)}", flush=True)
            return 0
        result = run_probe()
        if result["status"] != "passed" and result["stage"] == "construct":
            # A partially constructed acados controller can hang during Kit shutdown.
            # This probe owns its process, and the matrix runner reaps the process group.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 0 if result["status"] == "passed" else 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
