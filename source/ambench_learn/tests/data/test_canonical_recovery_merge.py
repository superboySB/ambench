# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Recover damaged metadata and merge only native successful canonical episodes."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import test_resampled_lerobot_dataset as fixtures
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_script(name: str):
    root = Path(os.environ.get("AMBENCH_DATA_TEST_SCRIPTS", str(Path(__file__).resolve().parents[4] / "scripts/data")))
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RECOVER = load_script("recover_lerobot_session")
MERGE = load_script("merge_lerobot_sessions")


class CanonicalRecoveryMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = fixtures.ResampledEpisodeSplitTest
        cls.fixture.setUpClass()
        cls.directory = cls.fixture.root.parent
        cls.broken = cls.directory / "broken/lerobot"
        shutil.copytree(cls.fixture.root, cls.broken)
        raw = LeRobotDataset("ambench/recovery_source", root=cls.broken)
        cls.paths = [str(raw.meta.get_data_file_path(index)) for index in range(4)]
        cls.outcomes = {
            "task": "PressButton-Am-UAQuad-Abs-PID-Direct-v0",
            "seed": 13001,
            "status": "recording",
            "episodes": [
                {
                    "episode_index": index,
                    "exported_episode_index": index,
                    "exported": True,
                    "termination_reason": "timeout" if index == 2 else "success",
                    "step_count": length,
                }
                for index, length in enumerate(cls.fixture.lengths)
            ],
        }
        (cls.broken.parent / "episode_outcomes.json").write_text(json.dumps(cls.outcomes))
        metadata = next((cls.broken / "meta/episodes").rglob("*.parquet"))
        metadata.write_bytes(metadata.read_bytes()[:-4])
        cls.paths[3] = "data/chunk-000/file-040.parquet"
        last_shard = cls.broken / cls.paths[3]
        last_shard.write_bytes(b"PAR1unfinished-unselected-shard")
        cls.before = RECOVER.file_hashes(cls.broken)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.tearDownClass()

    def test_recovery_preserves_original_and_recomputes_selected_statistics(self) -> None:
        selected = self.directory / "recover-selection.json"
        selected.write_text(
            json.dumps({
                "source_root": str(self.broken),
                "episodes": [{"episode_index": index, "data_file": self.paths[index]} for index in [0, 1]],
            })
        )
        recovered = self.directory / "recovered/lerobot"
        with patch.object(
            sys,
            "argv",
            [
                "recover",
                "--selection-json",
                str(selected),
                "--output-root",
                str(recovered),
                "--repo-id",
                "ambench/recovered",
            ],
        ):
            RECOVER.main()
        raw = LeRobotDataset("ambench/recovered", root=recovered)
        self.assertEqual(raw.meta.total_episodes, 2)
        self.assertEqual(raw.meta.total_frames, 17)
        self.assertLess(float(raw.meta.stats["action"]["max"][0]), 100)
        cursor = 0
        for index in [0, 1]:
            for state, action in self.fixture.rows[index]:
                row = raw[cursor]
                torch.testing.assert_close(row["observation.state"], state)
                torch.testing.assert_close(row["action"], action)
                cursor += 1
        self.assertEqual(RECOVER.file_hashes(self.broken), self.before)
        report = json.loads((recovered.parent / "recovery_report.json").read_text())
        self.assertEqual(report["selected_source_episode_indices"], [0, 1])
        self.assertTrue(report["official_writer_recomputed_episode_statistics"])

        second = self.directory / "second/lerobot"
        shutil.copytree(self.fixture.root, second)
        second_outcomes = {**self.outcomes, "seed": 13002, "status": "completed"}
        (second.parent / "episode_outcomes.json").write_text(json.dumps(second_outcomes))
        manifest = self.directory / "merge-sources.json"
        manifest.write_text(
            json.dumps({
                "sources": [
                    {"dataset_root": str(recovered), "episode_indices": [0, 1]},
                    {"dataset_root": str(second), "episode_indices": [1, 3]},
                ]
            })
        )
        merged = self.directory / "merged/lerobot"
        with patch.object(
            sys,
            "argv",
            ["merge", "--sources-json", str(manifest), "--output-root", str(merged), "--repo-id", "ambench/merged"],
        ):
            MERGE.main()
        combined = LeRobotDataset("ambench/merged", root=merged)
        self.assertEqual(combined.meta.total_episodes, 4)
        self.assertEqual(combined.meta.total_frames, 53)
        self.assertEqual(combined.meta.info["ambench"]["action_semantics"], "ee_absolute")
        merge_report = json.loads((merged.parent / "merge_report.json").read_text())
        self.assertEqual([row["source_seed"] for row in merge_report["episode_mapping"]], [13001, 13001, 13002, 13002])
        self.assertEqual([row["source_episode_index"] for row in merge_report["episode_mapping"]], [0, 1, 1, 3])
        self.assertLess(float(combined.meta.stats["action"]["max"][0]), 100)
        self.assertEqual(RECOVER.file_hashes(self.broken), self.before)

    def test_failed_or_damaged_selected_shard_is_rejected(self) -> None:
        for index in [2, 3]:
            selection = self.directory / f"invalid-{index}.json"
            selection.write_text(
                json.dumps({
                    "source_root": str(self.broken),
                    "episodes": [{"episode_index": index, "data_file": self.paths[index]}],
                })
            )
            output = self.directory / f"rejected-{index}/lerobot"
            with self.subTest(index=index), patch.object(
                sys,
                "argv",
                [
                    "recover",
                    "--selection-json",
                    str(selection),
                    "--output-root",
                    str(output),
                    "--repo-id",
                    "ambench/rejected",
                ],
            ), self.assertRaises(ValueError):
                RECOVER.main()
            self.assertFalse(output.exists())
        self.assertEqual(RECOVER.file_hashes(self.broken), self.before)


if __name__ == "__main__":
    unittest.main()
