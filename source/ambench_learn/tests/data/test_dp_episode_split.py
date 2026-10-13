# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate explicit DP episode splits and training-only normalization on CPU."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.umi_dataset import UmiDataset


class DiffusionEpisodeSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "dataset.zarr.zip"
        cls.shape_meta = {
            "obs": {
                "camera0_rgb": {
                    "shape": [3, 4, 4],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 1,
                    "type": "rgb",
                },
                "robot0_eef_pos": {"shape": [3], "horizon": 2, "latency_steps": 0, "down_sample_steps": 1},
                "robot0_eef_rot_axis_angle": {"shape": [6], "horizon": 2, "latency_steps": 0, "down_sample_steps": 1},
                "robot0_gripper_width": {"shape": [1], "horizon": 2, "latency_steps": 0, "down_sample_steps": 1},
            },
            "action": {"shape": [10], "horizon": 2, "latency_steps": 0, "down_sample_steps": 1},
        }
        replay = ReplayBuffer.create_empty_numpy()
        for episode_index in range(100):
            position = np.zeros((8, 3), dtype=np.float32)
            position[:, 0] = episode_index
            action = np.zeros((8, 7), dtype=np.float32)
            action[:, :3] = position
            action[:, 0] += 500 if episode_index >= 80 else 0.01
            action[:, 6] = -1
            replay.add_episode({
                "robot0_eef_pos": position,
                "robot0_eef_rot_axis_angle": np.zeros((8, 3), dtype=np.float32),
                "robot0_gripper_width": np.full((8, 1), episode_index / 1000, dtype=np.float32),
                "camera0_rgb": np.full((8, 4, 4, 3), episode_index, dtype=np.uint8),
                "action": action,
            })
        with zarr.ZipStore(str(cls.path), mode="w") as store:
            replay.save_to_store(store)
        cls.source_sha256 = hashlib.sha256(cls.path.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def make_dataset(self, **kwargs) -> UmiDataset:
        return UmiDataset(
            shape_meta=self.shape_meta,
            dataset_path=str(self.path),
            pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
            **kwargs,
        )

    def test_explicit_first_80_last_20_preserves_episode_identity(self) -> None:
        dataset = self.make_dataset(val_episode_indices=list(range(80, 100)), val_ratio=0.05)
        validation = dataset.get_validation_dataset()
        self.assertEqual(dataset.split_selection, "explicit")
        self.assertEqual(dataset.train_episode_indices, tuple(range(80)))
        self.assertEqual(dataset.val_episode_indices, tuple(range(80, 100)))
        self.assertEqual(len(dataset), 80 * 7)
        self.assertEqual(len(validation), 20 * 7)
        for view, expected_indices in [(dataset, set(range(80))), (validation, set(range(80, 100)))]:
            actual_indices = set()
            for index in range(len(view)):
                item = view[index]
                source_index = round(float(item["obs"]["camera0_rgb"][-1].mean()) * 255)
                actual_indices.add(source_index)
                self.assertIn(source_index, expected_indices)
                expected_x = 500 if source_index >= 80 else 0.01
                self.assertAlmostEqual(float(item["action"][0, 0]), expected_x, places=4)
            self.assertEqual(actual_indices, expected_indices)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.source_sha256)

    def test_normalizer_uses_only_training_sampler(self) -> None:
        dataset = self.make_dataset(val_episode_indices=list(range(80, 100)), normalizer_num_workers=0)
        real_loader = torch.utils.data.DataLoader
        requested_workers = []

        def cpu_loader(*args, **kwargs):
            requested_workers.append(kwargs["num_workers"])
            return real_loader(*args, **kwargs)

        with patch("torch.utils.data.DataLoader", side_effect=cpu_loader):
            normalizer = dataset.get_normalizer()
        self.assertEqual(requested_workers, [0])
        stats = normalizer["action"].get_input_stats()
        self.assertLess(float(stats["max"][0]), 0.1)
        self.assertLess(float(normalizer["robot0_gripper_width"].get_input_stats()["max"][0]), 0.08)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.source_sha256)

    def test_default_random_split_and_explicit_empty_split(self) -> None:
        dataset = self.make_dataset(val_ratio=0.2, seed=42)
        self.assertEqual(dataset.split_selection, "random")
        np.testing.assert_array_equal(dataset.val_mask, get_val_mask(100, 0.2, 42))
        empty = self.make_dataset(val_ratio=0.5, val_episode_indices=[])
        self.assertEqual(empty.train_episode_indices, tuple(range(100)))
        self.assertEqual(empty.val_episode_indices, ())
        self.assertEqual(len(empty.get_validation_dataset()), 0)

    def test_invalid_or_all_validation_indices_are_rejected(self) -> None:
        for indices in ([80, 80], [-1], [100], [1.5], [True], list(range(100))):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.make_dataset(val_episode_indices=indices)
        for workers in (-1, 1.5, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.make_dataset(normalizer_num_workers=workers)


if __name__ == "__main__":
    unittest.main()
