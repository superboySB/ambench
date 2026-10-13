# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU geometry checks for the scripted sponge trajectory; no simulator needed."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@dataclass
class _Waypoint:
    t: int
    xyz: torch.Tensor
    quat: torch.Tensor
    gripper: float


class _BasePolicy:
    def __init__(self, env):
        self.env = env
        self.device = "cpu"

    def identity_quat(self):
        return torch.tensor([1.0, 0.0, 0.0, 0.0])

    def _log_trajectory_summary(self, *args):
        pass


class _RobotIO:
    """Reference attachment kinematics for the identity command used by the policy."""

    def __init__(self, offset_link, command_to_link):
        self.offset_link = torch.tensor(offset_link)
        self.command_to_link = torch.tensor(command_to_link or (1.0, 0.0, 0.0, 0.0))

    def command_to_link_quat(self, quat):
        assert torch.equal(quat, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        return self.command_to_link.unsqueeze(0)

    def compute_tool_tip_pos_w(self, state):
        q = state[:, 3:7]
        vector = self.offset_link.unsqueeze(0).expand(state.shape[0], -1)
        cross = 2.0 * torch.linalg.cross(q[:, 1:], vector)
        rotated = vector + q[:, :1] * cross + torch.linalg.cross(q[:, 1:], cross)
        return state[:, :3] + rotated


def _policy(monkeypatch, offset_link, command_to_link, sponge_size=(0.12, 0.04, 0.08)):
    """Load the real trajectory method with simulator-only imports stubbed."""
    isaaclab = ModuleType("isaaclab")
    isaaclab.sim = ModuleType("isaaclab.sim")
    omni = ModuleType("omni")
    omni.usd = ModuleType("omni.usd")
    omni.usd.get_context = lambda: None
    pxr = ModuleType("pxr")
    pxr.UsdGeom = SimpleNamespace()
    base = ModuleType("ambench.policies.scripted.base")
    base.BasePolicy = _BasePolicy
    base.Waypoint = _Waypoint
    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.sim": isaaclab.sim,
        "omni": omni,
        "omni.usd": omni.usd,
        "pxr": pxr,
        "ambench.policies.scripted.base": base,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "ambench/policies/scripted/wipe_window.py"
    spec = importlib.util.spec_from_file_location("ambench.policies.scripted.wipe_window", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot_io = _RobotIO(offset_link, command_to_link)
    env = SimpleNamespace(cfg=SimpleNamespace(episode_length_s=20, sponge_size=sponge_size), robot_io=robot_io)
    policy = module.WipeWindowPolicy(env)
    dots = [torch.tensor([2.1, y, z]) for y, z in ((0.1, 0.9), (-0.1, 1.0), (0.15, 1.1), (0.0, 1.2))]
    policy._get_visible_dots = lambda env_id: list(enumerate(dots))
    policy.generate_trajectory({"policy": [{"ee_pos": torch.tensor([0.3, 0.0, 0.7])}]})
    return policy, robot_io, dots


@pytest.mark.parametrize(
    ("offset_link", "command_to_link", "expected_offset"),
    [
        ((0.18, 0.0, 0.0), None, (0.18, 0.0, 0.0)),
        ((0.011, -0.032, -0.075), (0.5, -0.5, -0.5, 0.5), (0.075, 0.011, 0.032)),
        ((0.0, 0.0, 0.0445), (2**-0.5, 0.0, 2**-0.5, 0.0), (0.0445, 0.0, 0.0)),
    ],
)
def test_all_stains_have_a_reachable_contact_target(monkeypatch, offset_link, command_to_link, expected_offset):
    policy, robot_io, dots = _policy(monkeypatch, offset_link, command_to_link)
    for contact_step, dot in zip((800, 1290, 1780, 2270), dots):
        contact = next(w for w in policy.waypoints if w.t == contact_step)
        expected_center = dot + torch.tensor([-0.055, 0.01, 0.0])
        assert torch.allclose(contact.xyz, expected_center - torch.tensor(expected_offset), atol=1e-6)
        state = torch.cat((contact.xyz, robot_io.command_to_link_quat(contact.quat.unsqueeze(0))[0])).unsqueeze(0)
        center = robot_io.compute_tool_tip_pos_w(state)[0]
        assert torch.allclose(center, expected_center, atol=1e-6)
        assert (center - dot).norm().item() < 0.08
        assert center[0].item() + 0.06 == pytest.approx(dot[0].item() + 0.005, abs=1e-6)
        assert contact_step < 20 * 120 - 1


def test_default_ee_geometry_preserves_existing_approach_and_contact(monkeypatch):
    policy, _, dots = _policy(monkeypatch, (0.18, 0.0, 0.0), None)
    first = dots[0]
    expected = {
        400: first + torch.tensor([-0.40, 0.0, -0.1]),
        700: first + torch.tensor([-0.28, 0.01, 0.0]),
        800: first + torch.tensor([-0.235, 0.01, 0.0]),
        920: first + torch.tensor([-0.235, -0.03, 0.0]),
        1000: first + torch.tensor([-0.40, 0.0, -0.1]),
    }
    for step, position in expected.items():
        waypoint = next(w for w in policy.waypoints if w.t == step)
        assert torch.allclose(waypoint.xyz, position, atol=1e-6)
        assert waypoint.gripper == -1.0


def test_sponge_thickness_sets_the_contact_surface(monkeypatch):
    policy, robot_io, dots = _policy(monkeypatch, (0.18, 0.0, 0.0), None, sponge_size=(0.10, 0.04, 0.08))
    contact = next(w for w in policy.waypoints if w.t == 800)
    state = torch.cat((contact.xyz, robot_io.command_to_link_quat(contact.quat.unsqueeze(0))[0])).unsqueeze(0)
    center = robot_io.compute_tool_tip_pos_w(state)[0]
    assert center[0].item() + 0.05 == pytest.approx(dots[0][0].item() + 0.005, abs=1e-6)
