# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Request one real OpenPI action chunk without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-semantics", choices=("ee_absolute", "base_joint_absolute"), default="ee_absolute")
    args = parser.parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    image = np.zeros((384, 384, 3), dtype=np.uint8)
    image[..., 1] = 127
    action_shapes = []
    for offset in (0.0, 0.01):
        output = client.infer({
            "am_bench/ee_pos": np.array([offset, 0, 0], dtype=np.float32),
            "am_bench/ee_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "am_bench/base_pos": np.array([offset, 0, 0], dtype=np.float32),
            "am_bench/base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "am_bench/arm_joint_pos": np.zeros(4, dtype=np.float32),
            "am_bench/gripper_width": np.array([0.05], dtype=np.float32),
            "am_bench/ee_image": image,
            "prompt": "press the button",
        })
        actions = np.asarray(output["actions"])
        action_dim = 12 if args.action_semantics == "base_joint_absolute" else 8
        if actions.ndim != 2 or actions.shape[1] != action_dim or not np.isfinite(actions).all():
            raise RuntimeError(f"Expected a finite (H, {action_dim}) action chunk, got {actions.shape}.")
        action_shapes.append(list(actions.shape))
    print(
        json.dumps(
            {
                "metadata": client.get_server_metadata(),
                "action_semantics": args.action_semantics,
                "action_shapes": action_shapes,
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
