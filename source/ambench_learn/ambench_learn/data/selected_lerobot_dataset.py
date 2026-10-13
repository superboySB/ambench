# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""LeRobot episode selection without decoding RGB while building row indices."""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hf_transform_to_torch


class SelectedLeRobotDataset(LeRobotDataset):
    """Keep LeRobot's selection semantics and restrict its constructor index scan.

    LeRobot 0.4.4 scans ``hf_dataset["index"]`` to build the mapping for a
    selected episode subset. Its whole-row transform otherwise decodes RGB for
    that numeric scan. Restore the official transform before exposing samples.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        episodes = kwargs.get("episodes", args[2] if len(args) > 2 else None)
        if episodes is not None and len(episodes) == 0:
            raise ValueError("The dataset requires at least one selected episode.")
        self._constructing_episode_index = True
        super().__init__(*args, **kwargs)
        self._constructing_episode_index = False
        self.hf_dataset.set_transform(hf_transform_to_torch)

    def load_hf_dataset(self) -> Dataset:
        """Temporarily expose only indices during the initial subset scan."""
        dataset = super().load_hf_dataset()
        if getattr(self, "_constructing_episode_index", False) and self.episodes is not None:
            return dataset.with_format("torch", columns=["index"])
        return dataset
