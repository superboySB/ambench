# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Check training-only OpenPI export using actual multi-episode recordings."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import test_resampled_lerobot_dataset as fixtures

SCRIPT_PATH = Path(
    os.environ.get(
        "AMBENCH_EXPORT_TEST_SCRIPT",
        str(Path(__file__).resolve().parents[4] / "scripts/data/export_lerobot_to_openpi.py"),
    )
)
SPEC = importlib.util.spec_from_file_location("ambench_openpi_subset_export", SCRIPT_PATH)
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


class OpenPIEpisodeSubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = fixtures.ResampledEpisodeSplitTest
        cls.fixture.setUpClass()
        cls.outcome_path = cls.fixture.root.parent / "episode_outcomes.json"
        cls.outcome_path.write_text(
            json.dumps({
                "task": "PressButton-Am-UAQuad-Abs-PID-Direct-v0",
                "episodes": [
                    {
                        "exported": True,
                        "exported_episode_index": index,
                        "termination_reason": "timeout" if index == 2 else "success",
                    }
                    for index in range(4)
                ],
            })
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.tearDownClass()

    def export(self, name: str, *arguments: str) -> Path:
        output = self.fixture.root.parent / name
        argv = [
            str(SCRIPT_PATH),
            "--dataset_roots",
            str(self.fixture.root),
            "--output_root",
            str(output),
            "--repo_id",
            "ambench/subset_fixture",
            "--target_hz",
            "20",
            "--omit_base_image",
            *arguments,
        ]
        with patch.object(sys, "argv", argv):
            EXPORT.main()
        return output

    def test_noncontiguous_successful_subset_excludes_held_out_actions_and_stats(self) -> None:
        output = self.export("train", "--episodes", "1", "3", "--require_success")
        report = json.loads((output / "ambench_openpi_export_report.json").read_text())
        self.assertTrue(report["success_verified"])
        self.assertEqual(report["total_episodes"], 2)
        self.assertEqual(report["total_frames"], 6)
        source = report["sources"][0]
        self.assertEqual(source["selected_episode_indices"], [1, 3])
        self.assertEqual(source["selected_successful_episodes"], 2)
        self.assertEqual(source["robot_type"], "ua_quad")
        self.assertEqual([entry["source_episode_index"] for entry in source["episodes"]], [1, 3])
        self.assertEqual([entry["output_episode_index"] for entry in source["episodes"]], [0, 1])
        self.assertIn("meta/info.json", source["source_files_sha256"])
        stats = [json.loads(line) for line in (output / "meta/episodes_stats.jsonl").read_text().splitlines()]
        for episode_stats in stats:
            self.assertLess(episode_stats["stats"]["actions"]["max"][0], 100)
        for path in output.glob("data/**/*.parquet"):
            for row in pq.read_table(path).to_pylist():
                self.assertLess(row["actions"][0], 100)
        self.assertEqual(self.fixture.file_hashes(), self.fixture.hashes)

    def test_default_keeps_full_export_compatibility(self) -> None:
        output = self.export("default_all")
        report = json.loads((output / "ambench_openpi_export_report.json").read_text())
        self.assertEqual(report["sources"][0]["selected_episode_indices"], [0, 1, 2, 3])
        self.assertEqual(report["total_episodes"], 3)
        self.assertEqual(report["total_frames"], 9)
        self.assertFalse(report["success_verified"])

    def test_invalid_failed_and_all_success_required_selections_fail_before_writing(self) -> None:
        cases = [
            ("duplicate", ["--episodes", "1", "1"]),
            ("negative", ["--episodes", "-1"]),
            ("out_of_range", ["--episodes", "4"]),
            ("known_timeout", ["--episodes", "2"]),
            ("all_success_required", ["--require_success"]),
        ]
        for name, arguments in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.export(name, *arguments)
            self.assertFalse((self.fixture.root.parent / name).exists())


if __name__ == "__main__":
    unittest.main()
