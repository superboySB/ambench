# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Check a real ACT/DP checkpoint forward on CPU using synthetic observations.

Run in the corresponding policy Docker Python environment with CUDA devices
hidden. This checks loading, processors and finite action shapes; it provides
no evidence of learned task success.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from ambench_learn.policies.remote.server import ACTBackend, DPBackend


def main() -> int:
    """Perform bounded caller-managed, deterministic CPU inference checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("act", "dp"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        parser.error("Hide all CUDA devices with CUDA_VISIBLE_DEVICES=-1 before this CPU check")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    state = np.asarray([0, 0, 1, 1, 0, 0, 0, 0.05], dtype=np.float32)
    reset_observation = {"ee_pos": state[:3], "ee_quat": state[3:7], "gripper_width": state[7:]}
    if args.policy == "act":
        backend = ACTBackend(args.checkpoint, "cpu", None, None)
        reset = backend.reset(action_semantics="ee_absolute")
        observation = {"observation.state": state[None, :]}
        for key, shape in backend.info()["image_features"].items():
            observation[key] = np.zeros((1, *shape), dtype=np.float32)
        actions = [backend.infer(observation=observation)["action"] for _ in range(2)]
    else:
        backend = DPBackend(args.checkpoint, "cpu")
        reset = backend.reset(observation=reset_observation)
        images = {name: np.zeros((384, 384, 3), dtype=np.uint8) for name in backend.camera_key_mapping}
        actions = [backend.infer(timestep=0, observation=reset_observation, images=images)["actions"]]
    if any(
        action is None or action.ndim != 2 or action.shape[-1] != 8 or not np.isfinite(action).all()
        for action in actions
    ):
        raise ValueError("Checkpoint produced invalid EE absolute action shapes or nonfinite values")
    print(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "synthetic_observation_cpu_interface_smoke",
                "policy": args.policy,
                "checkpoint": str(args.checkpoint),
                "device": "cpu",
                "cuda_available": False,
                "inference_seed": 42,
                "reset": reset,
                "backend": backend.info(),
                "inference_calls": len(actions),
                "action_shapes": [list(action.shape) for action in actions],
                "all_finite": True,
                "claim_limit": "Synthetic observations test the checkpoint interface, not learned task performance.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
