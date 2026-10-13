# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the real scripted valve trajectory at the arc endpoint."""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _rotate(quat, vector):
    cross = 2.0 * torch.linalg.cross(quat[:, 1:], vector)
    return vector + quat[:, :1] * cross + torch.linalg.cross(quat[:, 1:], cross)


def _load_policy(monkeypatch, class_name, wall_yaw):
    """Load BasePolicy and RotateValvePolicy with only the simulator imports stubbed."""
    isaaclab = ModuleType("isaaclab")
    isaaclab.utils = ModuleType("isaaclab.utils")
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.sample_uniform = lambda lower, upper, size, device: torch.full(
        (size,) if isinstance(size, int) else size, (lower + upper) / 2
    )
    math_utils.quat_apply = _rotate
    math_utils.quat_slerp = lambda first, second, fraction: first
    isaaclab.utils.math = math_utils
    env_module = ModuleType("ambench.tasks.base.base_env")
    env_module.BaseEnv = object
    utils = ModuleType("ambench.policies.utils")
    utils.compute_wall_axes = lambda q: tuple(_rotate(q.unsqueeze(0), axis.unsqueeze(0))[0] for axis in torch.eye(3))
    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.utils": isaaclab.utils,
        "isaaclab.utils.math": math_utils,
        "ambench.tasks.base.base_env": env_module,
        "ambench.policies.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    package = Path(__file__).parents[2] / "ambench/policies/scripted"
    for name in ("base", "rotate_valve"):
        spec = importlib.util.spec_from_file_location(f"ambench.policies.scripted.{name}", package / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)

    state = torch.zeros((1, 1, 13))
    state[0, 0, :3] = torch.tensor([0.3, 0.0, 0.7])
    state[0, 0, 3] = 1.0
    wall_quat = torch.tensor([[math.cos(wall_yaw / 2), 0.0, 0.0, math.sin(wall_yaw / 2)]])

    class Scene(dict):
        env_origins = torch.zeros((1, 3))

    env = SimpleNamespace(
        device="cpu",
        ee_link_idx=0,
        robot=SimpleNamespace(data=SimpleNamespace(body_link_state_w=state)),
        scene=Scene(wall=SimpleNamespace(data=SimpleNamespace(root_quat_w=wall_quat))),
    )
    env.unwrapped = env
    policy = getattr(module, class_name)(env)
    policy.generate_trajectory({"policy": [{"goal_pos": torch.tensor([1.9, 0.1, 1.2])}]})
    return policy


@pytest.mark.parametrize("class_name", ["RotateValvePolicy", "RotateValvePolicyFast"])
@pytest.mark.parametrize("wall_yaw", [0.0, 0.3])
def test_arc_end_and_release_have_no_position_jump(monkeypatch, class_name, wall_yaw):
    policy = _load_policy(monkeypatch, class_name, wall_yaw)
    last_arc, _, closed = policy._get_target_pose(policy.t_end_arc - 1)
    first_hold, _, still_closed = policy._get_target_pose(policy.t_end_arc)
    released, _, opened = policy._get_target_pose(policy.t_release_valve)
    final_hold, _, final_opened = policy._get_target_pose(policy.t_release_valve + 100)
    assert torch.allclose(last_arc, first_hold, atol=1e-7)
    assert torch.equal(first_hold, policy.circle_points[-1])
    assert torch.equal(released, first_hold)
    assert torch.equal(final_hold, first_hold)
    assert (closed, still_closed, opened, final_opened) == (-1.0, -1.0, 1.0, 1.0)
    assert len(policy.circle_points) == policy.t_end_arc - policy.t_start_arc
    assert policy.arc_sweep_angle == pytest.approx(1.8 * math.pi)
    assert policy.t_release_valve == (1400 if class_name == "RotateValvePolicy" else 650)
    assert (
        math.dist(policy.circle_points[-1].tolist(), policy.circle_points[len(policy.circle_points) // 2].tolist())
        > 0.2
    )
