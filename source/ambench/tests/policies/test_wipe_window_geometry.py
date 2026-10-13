# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU geometry checks for the scripted sponge trajectory; no simulator needed."""

import ast
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
        return _quat_mul(quat, self.command_to_link.unsqueeze(0).expand(quat.shape[0], -1))

    def link_to_command_quat(self, quat):
        inverse = self.command_to_link * torch.tensor([1.0, -1.0, -1.0, -1.0])
        return _quat_mul(quat, inverse.unsqueeze(0).expand(quat.shape[0], -1))

    def compute_tool_tip_pos_w(self, state):
        q = state[:, 3:7]
        vector = self.offset_link.unsqueeze(0).expand(state.shape[0], -1)
        cross = 2.0 * torch.linalg.cross(q[:, 1:], vector)
        rotated = vector + q[:, :1] * cross + torch.linalg.cross(q[:, 1:], cross)
        return state[:, :3] + rotated


class _RigidObject:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = SimpleNamespace(root_state_w=torch.zeros((1, 13)))

    def write_root_state_to_sim(self, state, env_ids):
        self.data.root_state_w[env_ids] = state


class _VisibilityView:
    def set_visibility(self, visible, indices):
        self.visible = visible


class _BaseEnvironment:
    def _setup_scene(self):
        pass

    def _reset_idx(self, env_ids):
        pass

    def _grasp_rigid_object_at_ee(self, obj, env_ids, grasp_width):
        ee_state = self.robot.data.body_link_state_w[env_ids, self.ee_link_idx]
        state = torch.zeros((len(env_ids), 13))
        state[:, :3] = self.robot_io.compute_tool_tip_pos_w(ee_state)
        state[:, 3:7] = self.robot_io.link_to_command_quat(ee_state[:, 3:7])
        obj.write_root_state_to_sim(state, env_ids)


class _Joint:
    def __init__(self):
        self.attributes = {}

    def __getattr__(self, name):
        def setter(value):
            self.attributes[name] = value

        return lambda: SimpleNamespace(Set=setter, SetTargets=setter)


def _quat_mul(left, right):
    real = left[:, :1] * right[:, :1] - (left[:, 1:] * right[:, 1:]).sum(-1, keepdim=True)
    imag = left[:, :1] * right[:, 1:] + right[:, :1] * left[:, 1:] + torch.linalg.cross(left[:, 1:], right[:, 1:])
    return torch.cat((real, imag), dim=-1)


def _quat_apply(quat, vector):
    cross = 2.0 * torch.linalg.cross(quat[:, 1:], vector)
    return vector + quat[:, :1] * cross + torch.linalg.cross(quat[:, 1:], cross)


def _environment(offset_link, command_to_link, sponge_size, mount_offset):
    """Execute real attachment, reset and geometry methods with simulator I/O stubbed."""
    path = Path(__file__).parents[2] / "ambench/tasks/wipe_window/wipe_window_env.py"
    tree = ast.parse(path.read_text())
    environment = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WipeWindow")
    methods = {"sponge_attachment_point_local", "compute_sponge_attachment_pos_w", "_setup_scene", "_reset_idx"}
    environment.body = [node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    joint = _Joint()
    prim = SimpleNamespace(
        IsValid=lambda: True, GetName=lambda: "ee_link", HasAPI=lambda api: True, GetPath=lambda: "/EE"
    )
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prim)
    namespace = {
        "BaseEnv": _BaseEnvironment,
        "torch": torch,
        "RigidObject": _RigidObject,
        "ContactSensor": lambda cfg: SimpleNamespace(),
        "get_context": lambda: SimpleNamespace(get_stage=lambda: stage),
        "Usd": SimpleNamespace(PrimRange=lambda prim: [prim]),
        "UsdPhysics": SimpleNamespace(
            RigidBodyAPI=object, FixedJoint=SimpleNamespace(Define=lambda stage, path: joint)
        ),
        "Gf": SimpleNamespace(Vec3f=lambda *values: values, Quatf=lambda real, imag: (real, *imag)),
        "sim_utils": SimpleNamespace(XformPrimView=lambda path, device: _VisibilityView()),
        "math_utils": SimpleNamespace(quat_apply=_quat_apply),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), environment],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    env = namespace["WipeWindow"].__new__(namespace["WipeWindow"])
    frame = SimpleNamespace(tool_tip_offset_local=offset_link, command_to_link_quat_wxyz=command_to_link)
    spec = SimpleNamespace(ee_body_name="ee_link", end_effector=frame)
    object_cfg = SimpleNamespace(prim_path="/Object")
    env.cfg = SimpleNamespace(
        robot_profile=SimpleNamespace(robot=spec),
        sponge_attachment_offset_local=mount_offset,
        episode_length_s=20,
        sponge_size=sponge_size,
        num_dots=4,
        window_object_cfg=object_cfg,
        sponge_object_cfg=object_cfg,
        contact_sensor_cfg=None,
        **{f"dot_{index}_object_cfg": object_cfg for index in range(4)},
    )
    env.device = "cpu"
    env.num_envs = 1
    env.scene = SimpleNamespace(env_prim_paths=["/Env"], rigid_objects={}, sensors={})
    env.ee_link_idx = 0
    env.robot_io = _RobotIO(offset_link, command_to_link)
    ee_state = torch.zeros((1, 1, 13))
    ee_state[:, 0, :3] = torch.tensor([0.3, 0.0, 0.7])
    ee_state[:, 0, 3:7] = env.robot_io.command_to_link_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    env.robot = SimpleNamespace(_ALL_INDICES=torch.tensor([0]), data=SimpleNamespace(body_link_state_w=ee_state))
    env._setup_scene()
    return env, joint


def _configured_mount_offsets():
    path = Path(__file__).parents[2] / "ambench/tasks/wipe_window/wipe_window_env_cfg.py"
    result = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.ClassDef):
            continue
        for assignment in node.body:
            names = (
                [assignment.target.id]
                if isinstance(assignment, ast.AnnAssign) and isinstance(assignment.target, ast.Name)
                else (
                    [target.id for target in assignment.targets if isinstance(target, ast.Name)]
                    if isinstance(assignment, ast.Assign)
                    else []
                )
            )
            if "sponge_attachment_offset_local" in names:
                result[node.name] = ast.literal_eval(assignment.value)
    return result


def _policy(monkeypatch, offset_link, command_to_link, sponge_size=(0.12, 0.04, 0.08), mount_offset=(0.0, 0.0, 0.0)):
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
    env, _ = _environment(offset_link, command_to_link, sponge_size, mount_offset)
    robot_io = env.robot_io
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


def test_ua_mount_joint_reset_and_trajectory_use_one_physical_attachment(monkeypatch):
    offsets = _configured_mount_offsets()
    assert offsets == {
        "WipeWindowEnvDefaultCfg": (0.0, 0.0, 0.0),
        "WipeWindowEnvUAQuadAbsPIDCfg": (0.0, 0.0, -0.06),
    }
    mount = offsets["WipeWindowEnvUAQuadAbsPIDCfg"]
    policy, _, dots = _policy(monkeypatch, (0.011, -0.032, -0.075), (0.5, -0.5, -0.5, 0.5), mount_offset=mount)
    env = policy.env
    assert env.sponge_attachment_point_local == pytest.approx((0.011, -0.032, -0.135))
    contact = next(waypoint for waypoint in policy.waypoints if waypoint.t == 800)
    expected_center = dots[0] + torch.tensor([-0.055, 0.01, 0.0])
    assert torch.allclose(contact.xyz, expected_center - torch.tensor([0.135, 0.011, 0.032]), atol=1e-6)
    env, joint = _environment((0.011, -0.032, -0.075), (0.5, -0.5, -0.5, 0.5), (0.12, 0.04, 0.08), mount)
    assert joint.attributes["CreateLocalPos0Attr"] == env.sponge_attachment_point_local
    env._reset_idx([0])
    assert torch.allclose(env.sponge_object.data.root_state_w[0, :3], torch.tensor([0.435, 0.011, 0.732]), atol=1e-6)
    assert torch.allclose(env.sponge_object.data.root_state_w[0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.count_nonzero(env.sponge_object.data.root_state_w[:, 7:]) == 0
    # Exact authored collider support from 43,476 vertices per finger; closure
    # moves along task Y and leaves this forward support unchanged.
    collider_front = 0.18552525981227302
    assert 0.135 + 0.06 - collider_front == pytest.approx(0.009475, abs=1e-6)


def test_shifted_mount_reset_tracks_a_nonidentity_measured_link():
    env, _ = _environment((0.011, -0.032, -0.075), (0.5, -0.5, -0.5, 0.5), (0.12, 0.04, 0.08), (0.0, 0.0, -0.06))
    pitch = torch.tensor([[3**0.5 / 2.0, 0.0, 0.5, 0.0]])
    link_quat = env.robot_io.command_to_link_quat(pitch)
    env.robot.data.body_link_state_w[:, 0, 3:7] = link_quat
    env._reset_idx([0])
    expected_position = torch.tensor([[0.3, 0.0, 0.7]]) + _quat_apply(pitch, torch.tensor([[0.135, 0.011, 0.032]]))
    torch.testing.assert_close(env.sponge_object.data.root_state_w[:, :3], expected_position)
    torch.testing.assert_close(env.sponge_object.data.root_state_w[:, 3:7], pitch)


@pytest.mark.parametrize(
    ("offset_link", "command_to_link"),
    [((0.18, 0.0, 0.0), None), ((0.011, -0.032, -0.075), (0.5, -0.5, -0.5, 0.5))],
)
def test_zero_mount_offset_preserves_default_attachment_and_reset(offset_link, command_to_link):
    env, joint = _environment(offset_link, command_to_link, (0.12, 0.04, 0.08), (0.0, 0.0, 0.0))
    assert joint.attributes["CreateLocalPos0Attr"] == offset_link
    state = env.robot.data.body_link_state_w[:, 0]
    torch.testing.assert_close(env.compute_sponge_attachment_pos_w(state), env.robot_io.compute_tool_tip_pos_w(state))
    env._reset_idx([0])
    torch.testing.assert_close(env.sponge_object.data.root_state_w[:, :3], env.robot_io.compute_tool_tip_pos_w(state))
