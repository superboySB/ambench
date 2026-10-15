# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of MP4 sample coverage and automatic-reset image boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import gymnasium as gym
import numpy as np
import pytest
import torch

from ambench.recording.video import RecordVideo, _EpisodeVideoWriter


@pytest.mark.parametrize("frame_count", [30, 256])
def test_mp4_keeps_every_sampled_frame(tmp_path, frame_count: int) -> None:
    writer = _EpisodeVideoWriter(tmp_path, fps=30, frame_skip=4, output_format="mp4", gc_trigger=None)
    for step in range(frame_count * 4):
        value = 240 if step == (frame_count - 1) * 4 else 16
        writer.add_step({"ee_camera": np.full((16, 16, 3), value, dtype=np.uint8)})
    writer.finalize_episode(0)

    capture = cv2.VideoCapture(str(tmp_path / "video-ee_camera-env0-eps0.mp4"))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
    decoded = []
    while True:
        okay, frame = capture.read()
        if not okay:
            break
        decoded.append(frame)
    capture.release()

    assert len(decoded) == frame_count
    assert float(decoded[0].mean()) == pytest.approx(16, abs=3)
    assert float(decoded[-1].mean()) == pytest.approx(240, abs=3)
    assert writer.has_data is False


@pytest.mark.parametrize("termination", ["terminated", "truncated"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_video_excludes_automatic_reset_images(tmp_path, termination: str, num_envs: int) -> None:
    class AutoResetEnv(gym.Env):
        """Expose a conspicuous reset image on the first environment's done step."""

        def __init__(self) -> None:
            self.num_envs = num_envs
            self.action_space = gym.spaces.Box(-1, 1, shape=(num_envs, 1))
            self.observation_space = gym.spaces.Box(-1, 1, shape=(num_envs, 1))
            self.sensor = SimpleNamespace(data=SimpleNamespace(output={}))
            self.scene = SimpleNamespace(sensors={"ee_camera": self.sensor})
            self.steps = 0

        def reset(self, **kwargs):
            self.steps = 0
            self.sensor.data.output["rgb"] = torch.stack(
                [torch.full((16, 16, 3), value, dtype=torch.uint8) for value in [16, 32][:num_envs]]
            )
            return np.zeros((num_envs, 1)), {}

        def step(self, action):
            self.steps += 1
            values = [64, 96] if self.steps == 1 else [240, 128]
            self.sensor.data.output["rgb"] = torch.stack(
                [torch.full((16, 16, 3), value, dtype=torch.uint8) for value in values[:num_envs]]
            )
            terminated, truncated = torch.zeros(num_envs, dtype=torch.bool), torch.zeros(num_envs, dtype=torch.bool)
            if self.steps == 2:
                (terminated if termination == "terminated" else truncated)[0] = True
            return np.zeros((num_envs, 1)), np.zeros(num_envs), terminated, truncated, {}

    env = RecordVideo(AutoResetEnv(), str(tmp_path), camera_names="ee_camera", output_format="mp4", frame_skip=1)
    env.reset()
    env.step(np.zeros((num_envs, 1)))
    env.step(np.zeros((num_envs, 1)))
    env.close()

    assert env.episode_ids == ([1] if num_envs == 1 else [1, 0])
    assert sorted(path.name for path in tmp_path.glob("*.mp4")) == [
        f"video-ee_camera-env{env_id}-eps0.mp4" for env_id in range(num_envs)
    ]
    for env_id, expected in enumerate([[16, 64], [32, 96, 128]][:num_envs]):
        capture = cv2.VideoCapture(str(tmp_path / f"video-ee_camera-env{env_id}-eps0.mp4"))
        assert capture.isOpened()
        decoded = []
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            decoded.append(float(frame.mean()))
        capture.release()
        assert decoded == pytest.approx(expected, abs=3)
    assert env._writers[0].has_data is False
