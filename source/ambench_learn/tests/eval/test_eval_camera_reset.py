# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera reset configuration contracts, independent of Isaac rendering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ambench_learn.eval.common import configure_env_cfg


@pytest.mark.parametrize("camera_kind", ["ee", "base", "scene_existing", "scene_override"])
@pytest.mark.parametrize(("requested_renders", "expected_renders"), [(0, 1), (1, 1), (3, 3)])
def test_camera_reset_renders_require_one_and_preserve_larger_budget(
    camera_kind: str, requested_renders: int, expected_renders: int
) -> None:
    camera = object()
    config = SimpleNamespace(
        num_rerenders_on_reset=requested_renders,
        robot_profile=SimpleNamespace(
            ee_camera=camera if camera_kind == "ee" else None,
            base_camera=camera if camera_kind == "base" else None,
        ),
        scene_camera_cfg=camera if camera_kind == "scene_existing" else None,
    )
    override = camera if camera_kind == "scene_override" else None

    result = configure_env_cfg(
        parse_env_cfg=lambda *args, **kwargs: config,
        task="Task-v0",
        device="cpu",
        num_envs=1,
        scene_camera_cfg=override,
    )

    assert result is config
    assert result.num_rerenders_on_reset == expected_renders


@pytest.mark.parametrize("requested_renders", [0, 1, 3])
def test_configuration_without_cameras_preserves_render_budget(requested_renders: int) -> None:
    config = SimpleNamespace(
        num_rerenders_on_reset=requested_renders,
        robot_profile=SimpleNamespace(ee_camera=None, base_camera=None),
        scene_camera_cfg=None,
    )

    result = configure_env_cfg(parse_env_cfg=lambda *args, **kwargs: config, task="Task-v0", device="cpu", num_envs=1)

    assert result.num_rerenders_on_reset == requested_renders


def test_legacy_camera_configuration_does_not_gain_unsupported_render_field() -> None:
    config = SimpleNamespace(robot_profile=SimpleNamespace(ee_camera=object(), base_camera=None))

    result = configure_env_cfg(parse_env_cfg=lambda *args, **kwargs: config, task="Task-v0", device="cpu", num_envs=1)

    assert not hasattr(result, "num_rerenders_on_reset")


def test_scene_camera_override_refreshes_without_robot_profile() -> None:
    config = SimpleNamespace(num_rerenders_on_reset=0)
    camera = object()

    result = configure_env_cfg(
        parse_env_cfg=lambda *args, **kwargs: config,
        task="Task-v0",
        device="cpu",
        num_envs=1,
        scene_camera_cfg=camera,
    )

    assert result.scene_camera_cfg is camera
    assert result.num_rerenders_on_reset == 1
