# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise episode splits against real LeRobot shards without Isaac or a GPU."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from datasets import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info

from ambench_learn.data.resampled_lerobot_dataset import ResampledLeRobotDataset
from ambench_learn.data.selected_lerobot_dataset import SelectedLeRobotDataset


class ResampledEpisodeSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name) / "lerobot"
        cls.lengths = [5, 12, 18, 24]
        cls.colors = [0, 20, 250, 80]
        cls.rows = []
        writer = LeRobotDataset.create(
            repo_id="ambench/test_episode_splits",
            root=cls.root,
            robot_type="ua_quad",
            fps=120,
            use_videos=False,
            features={
                "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
                "action": {"dtype": "float32", "shape": (8,), "names": None},
                "observation.images.ee_camera": {
                    "dtype": "image",
                    "shape": (8, 8, 3),
                    "names": ["height", "width", "channel"],
                },
            },
        )
        writer.meta.info["ambench"] = {
            "action_semantics": "ee_absolute",
            "state_keys": ["ee_pos", "ee_quat", "gripper_width"],
        }
        write_info(writer.meta.info, cls.root)
        for episode_index, length in enumerate(cls.lengths):
            episode_rows = []
            for frame_index in range(length):
                state = np.array(
                    [episode_index * 10 + frame_index * 0.001, 0, 1, 1, 0, 0, 0, episode_index * 0.1],
                    dtype=np.float32,
                )
                action = state.copy()
                action[0] += 500 if episode_index == 2 else episode_index * 0.01 + frame_index * 0.0001
                action[7] = -1 if episode_index % 2 else 1
                writer.add_frame({
                    "observation.state": state,
                    "action": action,
                    "observation.images.ee_camera": np.full((8, 8, 3), cls.colors[episode_index], dtype=np.uint8),
                    "task": "press the button",
                })
                episode_rows.append((torch.from_numpy(state), torch.from_numpy(action)))
            writer.save_episode()
            cls.rows.append(episode_rows)
        writer.finalize()
        cls.hashes = cls.file_hashes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    @classmethod
    def file_hashes(cls) -> dict[str, str]:
        return {
            str(path.relative_to(cls.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in cls.root.rglob("*")
            if path.is_file()
        }

    def make_raw(self, episodes: list[int] | None = None) -> LeRobotDataset:
        return SelectedLeRobotDataset("ambench/test_episode_splits", root=self.root, episodes=episodes)

    def test_full_contiguous_and_noncontiguous_splits_keep_frame_identity(self) -> None:
        for selection in (None, [0, 1], [2, 3], [1, 3], [3, 1], [3]):
            with self.subTest(selection=selection):
                raw = self.make_raw(selection)
                dataset = ResampledLeRobotDataset(raw, target_fps=20, action_window_length=4)
                expected_episodes = list(range(4)) if selection is None else sorted(selection)
                expected_lengths = [self.lengths[index] // 6 for index in expected_episodes]
                self.assertEqual(len(dataset), sum(expected_lengths))
                self.assertEqual(dataset.build_report()["source_episode_indices"], tuple(expected_episodes))
                cursor = 0
                for source_index in expected_episodes:
                    for frame_index in range(0, (self.lengths[source_index] // 6) * 6, 6):
                        item = dataset[cursor]
                        state, action = self.rows[source_index][frame_index]
                        self.assertEqual(int(item["episode_index"]), source_index)
                        self.assertEqual(int(item["raw_frame_index"]), frame_index)
                        self.assertEqual(int(item["raw_index"]), sum(self.lengths[:source_index]) + frame_index)
                        self.assertEqual(int(item["index"]), cursor)
                        torch.testing.assert_close(item["observation.state"][:3], torch.zeros(3))
                        torch.testing.assert_close(item["observation.state"][3:7], torch.tensor([1.0, 0, 0, 0]))
                        torch.testing.assert_close(item["observation.state"][7], state[7])
                        torch.testing.assert_close(item["action"][0, :3], action[:3] - state[:3])
                        torch.testing.assert_close(item["action"][0, 7], action[7])
                        logical_left = self.lengths[source_index] // 6 - frame_index // 6
                        self.assertEqual(
                            item["action_is_pad"].tolist(), [offset >= logical_left for offset in range(4)]
                        )
                        self.assertAlmostEqual(
                            float(item["observation.images.ee_camera"].mean()), self.colors[source_index] / 255
                        )
                        cursor += 1
                with self.assertRaises(IndexError):
                    dataset[len(dataset)]
        self.assertEqual(self.file_hashes(), self.hashes)

    def test_training_normalization_excludes_held_out_numeric_and_rgb_values(self) -> None:
        raw = self.make_raw([1, 3])
        dataset = ResampledLeRobotDataset(raw, target_fps=20, action_window_length=4)
        self.assertGreater(float(raw.meta.stats["observation.images.ee_camera"]["max"].max()), 0.9)
        self.assertAlmostEqual(float(dataset.meta.stats["observation.images.ee_camera"]["max"].max()), 80 / 255)
        self.assertAlmostEqual(float(dataset.meta.stats["observation.images.ee_camera"]["min"].min()), 20 / 255)
        self.assertLess(float(dataset.meta.stats["action"]["max"][0]), 0.1)
        self.assertLess(float(dataset.meta.stats["observation.state"]["max"][7]), 0.31)
        self.assertEqual(int(dataset.meta.stats["observation.state"]["count"][0]), 6)
        self.assertEqual(int(dataset.meta.stats["action"]["count"][0]), 24)
        self.assertEqual(self.file_hashes(), self.hashes)

    def test_numeric_initialization_does_not_decode_rgb(self) -> None:
        with patch.object(Image, "decode_example", side_effect=AssertionError("Unexpected RGB decoding")):
            raw = self.make_raw([1, 3])
            dataset = ResampledLeRobotDataset(raw, target_fps=20, action_window_length=4)
        self.assertEqual(len(dataset), 6)
        self.assertAlmostEqual(float(dataset[0]["observation.images.ee_camera"].mean()), 20 / 255)
        self.assertEqual(self.file_hashes(), self.hashes)

    def test_empty_and_too_short_selections_fail_before_reading_full_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one selected episode"):
            self.make_raw([])
        raw = copy.copy(self.make_raw())
        raw.episodes = []
        with self.assertRaisesRegex(ValueError, "at least one selected episode"):
            ResampledLeRobotDataset(raw, target_fps=20, action_window_length=4)
        with self.assertRaisesRegex(ValueError, "no complete policy-rate"):
            ResampledLeRobotDataset(self.make_raw([0]), target_fps=20, action_window_length=4)
        self.assertEqual(self.file_hashes(), self.hashes)


if __name__ == "__main__":
    unittest.main()
