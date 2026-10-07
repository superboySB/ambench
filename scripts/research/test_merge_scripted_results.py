# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical session checks for merged scripted expert reports."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from merge_scripted_results import merge_reports


class MergeScriptedResultsTests(unittest.TestCase):
    """A validated retry must retain an earlier failed attempt."""

    def test_retry_validates_lerobot_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reports = []
            for index, status in enumerate(("failed", "passed")):
                run_dir = root / f"run-{index}"
                run_dir.mkdir()
                record_log = run_dir / "record.log"
                record_log.write_text("recorder log\n")
                validation_log = run_dir / "validation.log"
                session_root = run_dir / "session"
                if status == "passed":
                    validation_log.write_text("validator passed\n")
                    info_dir = session_root / "lerobot" / "meta"
                    info_dir.mkdir(parents=True)
                    (info_dir / "info.json").write_text(
                        json.dumps({
                            "fps": 120,
                            "total_episodes": 1,
                            "total_frames": 100,
                            "ambench": {
                                "action_semantics": "ee_absolute",
                                "state_keys": ["ee_pos", "ee_quat", "gripper_width"],
                            },
                            "features": {"observation.images.ee_camera": {"shape": [384, 384, 3]}},
                        })
                    )
                report_path = run_dir / "results.json"
                report_path.write_text(
                    json.dumps({
                        "registry_count": 106,
                        "generated_at_utc": f"2026-09-30T00:00:0{index}+00:00",
                        "validate": True,
                        "video": False,
                        "results": [{
                            "family": "PressButton",
                            "task_id": "PressButton-Am-EE-Abs-PID-Direct-v0",
                            "status": status,
                            "stage": "complete" if status == "passed" else "record",
                            "error": "initial failure" if status == "failed" else "",
                            "record_exit_code": 0 if status == "passed" else 1,
                            "validation_exit_code": 0 if status == "passed" else "",
                            "episodes": 1 if status == "passed" else 0,
                            "record_log": str(record_log),
                            "validation_log": str(validation_log) if status == "passed" else "",
                            "session_root": str(session_root) if status == "passed" else "",
                        }],
                    })
                )
                reports.append(report_path)

            args = argparse.Namespace(reports=reports, output=root / "merged.json")
            merged = merge_reports(args)
            self.assertEqual(merged["passed_count"], 1)
            self.assertEqual(merged["results"][0]["selected"]["total_frames"], 100)
            self.assertEqual([a["status"] for a in merged["results"][0]["attempts"]], ["failed", "passed"])

            info_path = root / "run-1" / "session" / "lerobot" / "meta" / "info.json"
            info = json.loads(info_path.read_text())
            info["total_episodes"] = 2
            info_path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "Episode count differs"):
                merge_reports(args)


if __name__ == "__main__":
    unittest.main()
