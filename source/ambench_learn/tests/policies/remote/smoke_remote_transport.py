# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Send synthetic observations through a real ACT/DP HTTP model server."""

from __future__ import annotations

import argparse
import json

import numpy as np

from ambench_learn.policies.remote.protocol import RemotePolicyClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("act", "dp"))
    parser.add_argument("--remote-url", required=True)
    parser.add_argument("--action-semantics", choices=("ee_absolute", "base_joint_absolute"), default="ee_absolute")
    args = parser.parse_args()
    client = RemotePolicyClient(args.remote_url)
    info = client.info()
    if info.get("policy") != args.policy:
        raise RuntimeError(f"Expected {args.policy} server, got {info.get('policy')!r}.")

    action_dim = 12 if args.action_semantics == "base_joint_absolute" else 8
    if args.policy == "act":
        reset = client.reset(action_semantics=args.action_semantics)
        state = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.5] if action_dim == 12 else [0, 0, 0, 1, 0, 0, 0, 0.5]
        observation = {
            "observation.state": np.array([state], dtype=np.float32),
        }
        for key, shape in info["image_features"].items():
            observation[key] = np.zeros((1, *shape), dtype=np.float32)
        actions = [client.infer(observation=observation)["action"] for _ in range(2)]
        if any(action.shape != (1, action_dim) or not np.isfinite(action).all() for action in actions):
            raise RuntimeError(f"ACT HTTP server did not return two finite {action_dim}D actions.")
        result = {"policy": "act", "reset": reset, "action_shapes": [list(a.shape) for a in actions]}
    else:
        observation = {
            "ee_pos": np.zeros(3, dtype=np.float32),
            "ee_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "gripper_width": np.array([0.5], dtype=np.float32),
            "arm_joint_pos": np.zeros(4, dtype=np.float32),
            "base_pos": np.zeros(3, dtype=np.float32),
            "base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        }
        reset = client.reset(observation=observation)
        images = {
            camera: np.zeros((shape[1], shape[2], 3), dtype=np.uint8) for camera, shape in info["camera_shapes"].items()
        }
        actions = client.infer(timestep=0, observation=observation, images=images)["actions"]
        if actions.ndim != 2 or actions.shape[1] != action_dim or not np.isfinite(actions).all():
            raise RuntimeError(f"DP HTTP server did not return a finite {action_dim}D action plan.")
        result = {"policy": "dp", "reset": reset, "action_shape": list(actions.shape)}

    print(
        json.dumps(
            {
                **result,
                "action_semantics": args.action_semantics,
                "remote_url": args.remote_url,
                "model_forward": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
