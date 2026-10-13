# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU frame-contract checks using production methods and independent SO(3) geometry."""

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

FRAME_CASES = [
    ((0.5, -0.5, -0.5, 0.5), (0.011, -0.032, -0.075), (0.075, 0.011, 0.032)),
    ((2**-0.5, 0.0, 2**-0.5, 0.0), (0.0, 0.0, 0.0445), (0.0445, 0.0, 0.0)),
]


def _rotation(quat: torch.Tensor) -> Rotation:
    return Rotation.from_quat(quat.detach().numpy()[..., [1, 2, 3, 0]])


def _tensor(rotation: Rotation, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(rotation.as_quat()[..., [3, 0, 1, 2]], dtype=dtype)


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _tensor(_rotation(left) * _rotation(right), left.dtype)


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_rotation(quat).apply(vector.numpy()), dtype=vector.dtype)


@pytest.fixture(scope="module")
def robot_io_methods() -> dict:
    """Execute unchanged methods without importing the simulator or starting Kit.

    SciPy supplies active quaternion multiplication/application primitives. This
    tests production frame composition against world-frame geometric invariants.
    """
    path = Path(__file__).parents[2] / "ambench/robots/robot_io.py"
    tree = ast.parse(path.read_text())
    robot_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RobotIO")
    names = {"command_to_link_quat", "link_to_command_quat", "compute_tool_tip_pos_w"}
    methods = [node for node in robot_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
        type_ignores=[],
    )
    namespace = {"torch": torch, "math_utils": SimpleNamespace(quat_mul=_quat_mul, quat_apply=_quat_apply)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def _robot_io(methods: dict, offset: tuple | None, tip: tuple = (0.0, 0.0, 0.0)) -> SimpleNamespace:
    quat = None if offset is None else torch.tensor(offset, dtype=torch.float32)
    inverse = None if quat is None else quat * torch.tensor([1.0, -1.0, -1.0, -1.0])
    robot_io = SimpleNamespace(
        _command_to_link_quat_wxyz=quat,
        _link_to_command_quat_wxyz=inverse,
        spec=SimpleNamespace(end_effector=SimpleNamespace(tool_tip_offset_local=tip)),
    )
    for name in ("command_to_link_quat", "link_to_command_quat", "compute_tool_tip_pos_w"):
        setattr(robot_io, name, MethodType(methods[name], robot_io))
    return robot_io


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("offset", "tip", "identity_tip_world"), FRAME_CASES)
def test_world_rotation_commutes_with_local_frame_conversion(
    robot_io_methods: dict, offset: tuple, tip: tuple, identity_tip_world: tuple, dtype: torch.dtype
) -> None:
    robot_io = _robot_io(robot_io_methods, offset, tip)
    world_rotations = Rotation.random(128, random_state=23001)
    command_rotations = Rotation.random(128, random_state=33001)
    commands = _tensor(command_rotations, dtype)
    rotated_commands = _tensor(world_rotations * command_rotations, dtype)
    link = robot_io.command_to_link_quat(commands)
    rotated_link = robot_io.command_to_link_quat(rotated_commands)
    np.testing.assert_allclose(
        _rotation(rotated_link).as_matrix(), (world_rotations * _rotation(link)).as_matrix(), atol=3e-7
    )
    raw_links = _tensor(command_rotations, dtype)
    rotated_raw_links = _tensor(world_rotations * command_rotations, dtype)
    command = robot_io.link_to_command_quat(raw_links)
    rotated_command = robot_io.link_to_command_quat(rotated_raw_links)
    np.testing.assert_allclose(
        _rotation(rotated_command).as_matrix(), (world_rotations * _rotation(command)).as_matrix(), atol=3e-7
    )
    assert link.dtype == dtype
    assert link.device.type == "cpu"


@pytest.mark.parametrize(("offset", "tip", "identity_tip_world"), FRAME_CASES)
def test_nonidentity_commands_produce_the_physical_world_tool_pose(
    robot_io_methods: dict, offset: tuple, tip: tuple, identity_tip_world: tuple
) -> None:
    robot_io = _robot_io(robot_io_methods, offset, tip)
    commands = Rotation.from_euler("xyz", [[0, 30, 0], [30, 0, 0], [0, 0, 30], [23, -31, 17]], degrees=True)
    link_quat = robot_io.command_to_link_quat(_tensor(commands))
    link_rotations = _rotation(link_quat)
    local_tool_rotation = _rotation(robot_io._link_to_command_quat_wxyz)
    np.testing.assert_allclose((link_rotations * local_tool_rotation).as_matrix(), commands.as_matrix(), atol=1e-7)
    state = torch.zeros((4, 13), dtype=torch.float64)
    state[:, :3] = torch.tensor([1.2, -0.4, 0.7])
    state[:, 3:7] = link_quat
    before = state.clone()
    tool_position = robot_io.compute_tool_tip_pos_w(state)
    expected_position = state[:, :3].numpy() + commands.apply(np.asarray(identity_tip_world))
    np.testing.assert_allclose(tool_position.numpy(), expected_position, atol=1e-8)
    assert torch.equal(state, before)


@pytest.mark.parametrize(("offset", "tip", "identity_tip_world"), FRAME_CASES)
def test_inverse_returns_physical_tool_world_orientation_for_held_objects(
    robot_io_methods: dict, offset: tuple, tip: tuple, identity_tip_world: tuple
) -> None:
    robot_io = _robot_io(robot_io_methods, offset, tip)
    raw_link_world = Rotation.random(128, random_state=13001)
    tool_world = robot_io.link_to_command_quat(_tensor(raw_link_world))
    local_tool_axes = Rotation.from_quat(np.asarray(offset)[[1, 2, 3, 0]]).inv().as_matrix()
    expected_world_axes = raw_link_world.as_matrix() @ local_tool_axes
    np.testing.assert_allclose(_rotation(tool_world).as_matrix(), expected_world_axes, atol=1e-7)


@pytest.mark.parametrize(("offset", "tip", "identity_tip_world"), FRAME_CASES)
def test_legacy_action_migration_preserves_raw_link_pose_and_observation(
    robot_io_methods: dict, offset: tuple, tip: tuple, identity_tip_world: tuple
) -> None:
    robot_io = _robot_io(robot_io_methods, offset, tip)
    offset_rotation = Rotation.from_quat(np.asarray(offset)[[1, 2, 3, 0]])
    old_commands = Rotation.random(128, random_state=13005)
    old_links = offset_rotation * old_commands
    migrated_commands = offset_rotation * old_commands * offset_rotation.inv()
    new_links = robot_io.command_to_link_quat(_tensor(migrated_commands))
    np.testing.assert_allclose(_rotation(new_links).as_matrix(), old_links.as_matrix(), atol=1e-7)
    canonical_state = torch.zeros((128, 13), dtype=torch.float64)
    canonical_state[:, :3] = torch.arange(384).reshape(128, 3) / 100.0
    canonical_state[:, 3:7] = _tensor(old_links)
    old_tool_points = robot_io.compute_tool_tip_pos_w(canonical_state)
    preserved_state = canonical_state.clone()
    state_with_new_target = canonical_state.clone()
    state_with_new_target[:, 3:7] = new_links
    torch.testing.assert_close(robot_io.compute_tool_tip_pos_w(state_with_new_target), old_tool_points)
    assert torch.equal(canonical_state, preserved_state)
    np.testing.assert_allclose(
        (offset_rotation.inv() * migrated_commands * offset_rotation).as_matrix(), old_commands.as_matrix(), atol=1e-12
    )


@pytest.mark.parametrize("offset", [None, (1.0, 0.0, 0.0, 0.0), FRAME_CASES[0][0], FRAME_CASES[1][0]])
def test_identity_commands_and_profiles_without_offsets_keep_existing_behavior(robot_io_methods: dict, offset) -> None:
    robot_io = _robot_io(robot_io_methods, offset)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    expected_link = torch.tensor([offset or (1.0, 0.0, 0.0, 0.0)], dtype=torch.float64)
    link = robot_io.command_to_link_quat(identity)
    np.testing.assert_allclose(_rotation(link).as_matrix(), _rotation(expected_link).as_matrix(), atol=1e-7)
    np.testing.assert_allclose(_rotation(robot_io.link_to_command_quat(link)).as_matrix(), np.eye(3)[None], atol=1e-7)
    if offset is None:
        arbitrary = _tensor(Rotation.random(8, random_state=5))
        assert robot_io.command_to_link_quat(arbitrary) is arbitrary
        assert robot_io.link_to_command_quat(arbitrary) is arbitrary
