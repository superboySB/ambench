# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace why the WipeWindow scripted expert does or does not remove each stain."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result-json", type=Path, required=True)
parser.add_argument("--steps", type=int, default=4400)
parser.add_argument("--sample-every", type=int, default=60)
parser.add_argument("--episode-length-s", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.sample_every < 1 or args.episode_length_s < 1:
    parser.error("--steps, --sample-every, and --episode-length-s must be positive")
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac imports must follow AppLauncher construction.
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import ambench.tasks  # noqa: E402,F401
from ambench.policies.scripted.wipe_window import WipeWindowPolicy  # noqa: E402


def run_trace() -> dict[str, object]:
    """Run one bounded expert episode and sample contact and geometry terms."""
    task_id = "WipeWindow-Am-EE-Abs-PID-Direct-v0"
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
    env_cfg.episode_length_s = args.episode_length_s
    env = gym.make(task_id, cfg=env_cfg).unwrapped
    try:
        obs, _ = env.reset()
        policy = WipeWindowPolicy(env=env)
        threshold = env.cfg.stain_removal_force_threshold
        radius = env.cfg.stain_removal_contact_radius
        trace: list[dict[str, object]] = []
        max_force = 0.0
        min_distances = [float("inf")] * env.cfg.num_dots
        force_and_near_steps = [0] * env.cfg.num_dots
        previous_visible = env._stain_visible_buf[0].clone()
        final_status = "step_limit"

        with torch.inference_mode():
            for step in range(args.steps):
                action = policy.advance(obs, 0)
                obs, _, terminated, truncated, _ = env.step(action)
                force = torch.norm(env.contact_sensor.data.net_forces_w[0, 0]).item()
                dot_pos = torch.stack([dot.data.root_pos_w[0] for dot in env.dot_objects])
                sponge_pos = env.sponge_object.data.root_pos_w[0]
                distances = torch.norm(dot_pos - sponge_pos.unsqueeze(0), dim=1).tolist()
                visible = env._stain_visible_buf[0].clone()
                max_force = max(max_force, force)
                for dot_idx, distance in enumerate(distances):
                    min_distances[dot_idx] = min(min_distances[dot_idx], distance)
                    if force > threshold and distance <= radius:
                        force_and_near_steps[dot_idx] += 1
                changed = not torch.equal(visible, previous_visible)
                if step == 0 or (step + 1) % args.sample_every == 0 or changed or terminated[0] or truncated[0]:
                    trace.append({
                        "step": step + 1,
                        "target_pos_local": action[0, :3].tolist(),
                        "ee_pos_local": obs["policy"][0]["ee_pos"].tolist(),
                        "sponge_pos_w": sponge_pos.tolist(),
                        "dot_pos_w": dot_pos.tolist(),
                        "distances_m": distances,
                        "force_n": force,
                        "visible": visible.tolist(),
                        "contact_counts": env._dot_contact_count[0].tolist(),
                        "terminated": bool(terminated[0]),
                        "truncated": bool(truncated[0]),
                    })
                previous_visible = visible
                if (step + 1) % 600 == 0:
                    print(f"step={step + 1} visible={visible.tolist()} force={force:.3f}", flush=True)
                if terminated[0] or truncated[0]:
                    final_status = "success" if terminated[0] else "truncated"
                    break

        return {
            "task_id": task_id,
            "status": final_status,
            "steps_completed": step + 1,
            "episode_length_s": args.episode_length_s,
            "force_threshold_n": threshold,
            "contact_radius_m": radius,
            "max_force_n": max_force,
            "min_distances_m": min_distances,
            "force_and_near_steps": force_and_near_steps,
            "waypoint_times": [waypoint.t for waypoint in policy.waypoints],
            "samples": trace,
        }
    finally:
        env.close()


def main() -> int:
    """Persist the diagnostic trace even when scene stepping raises."""
    try:
        result = run_trace()
    except Exception as error:
        result = {"status": "error", "error": str(error), "traceback": traceback.format_exc()}
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with args.result_json.open("w") as result_file:
        result_file.write(json.dumps(result, indent=2) + "\n")
        result_file.flush()
        os.fsync(result_file.fileno())
    print(f"Trace: {args.result_json}", flush=True)
    if result["status"] != "success":
        # Kit's close() exits with zero in this standalone diagnostic process.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
